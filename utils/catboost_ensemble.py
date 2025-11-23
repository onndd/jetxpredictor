"""
CatBoost Ensemble Manager - Ultra Aggressive Training için

10 model ensemble yönetimi, weighted averaging, variance-based confidence tracking.

GÜNCELLEME:
- Predict Proba desteği eklendi (Classifier için)
- 2 Modlu yapıya uygun confidence hesaplamaları
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import json
from catboost import CatBoostRegressor, CatBoostClassifier
import joblib

logger = logging.getLogger(__name__)


class CatBoostEnsemble:
    """
    CatBoost model ensemble'ı yönetir.
    
    Features:
    - Multiple model training with different seeds
    - Weighted averaging based on performance
    - Variance-based confidence scores
    - Performance tracking
    """
    
    def __init__(
        self, 
        model_type: str = 'regressor',
        n_models: int = 10,
        base_params: Optional[Dict] = None
    ):
        """
        Args:
            model_type: 'regressor' veya 'classifier'
            n_models: Ensemble'daki model sayısı
            base_params: Temel CatBoost parametreleri
        """
        self.model_type = model_type
        self.n_models = n_models
        self.base_params = base_params or {}
        
        self.models: List = []
        self.weights: np.ndarray = np.ones(n_models) / n_models  # Başlangıçta eşit ağırlık
        self.performance_history: List[Dict] = []
        
    def create_model(self, seed: int, subsample: float, bagging_temp: float) -> object:
        """
        Tekil model oluştur
        """
        params = self.base_params.copy()
        params.update({
            'random_seed': seed,
            'subsample': subsample,
            'bagging_temperature': bagging_temp
        })
        
        if self.model_type == 'regressor':
            return CatBoostRegressor(**params)
        else:
            return CatBoostClassifier(**params)
    
    def train_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool = True
    ) -> Dict:
        """
        Ensemble'daki tüm modelleri eğit
        """
        self.models = []
        individual_scores = []
        
        # Farklı parametrelerle modeller oluştur
        seeds = [42 + i * 100 for i in range(self.n_models)]
        subsamples = np.linspace(0.7, 0.9, self.n_models)
        bagging_temps = np.linspace(0.5, 1.5, self.n_models)
        
        for i in range(self.n_models):
            if verbose:
                print(f"\n{'='*80}")
                print(f"🤖 MODEL {i+1}/{self.n_models} EĞİTİMİ")
                print(f"{'='*80}")
                print(f"Seed: {seeds[i]}, Subsample: {subsamples[i]:.3f}, Bagging Temp: {bagging_temps[i]:.3f}")
            
            # Model oluştur ve eğit
            model = self.create_model(seeds[i], subsamples[i], bagging_temps[i])
            
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                verbose=100 if verbose else False
            )
            
            self.models.append(model)
            
            # Validation performansı
            if self.model_type == 'classifier':
                # Classifier için predict_proba kullanmak daha iyi olabilir ama metric accuracy ise predict yeterli
                val_pred = model.predict(X_val)
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y_val, val_pred)
                individual_scores.append(score)
                if verbose: print(f"\n✅ Model {i+1} VAL ACC: {score*100:.2f}%")
            else:
                val_pred = model.predict(X_val)
                from sklearn.metrics import mean_absolute_error
                score = mean_absolute_error(y_val, val_pred)
                individual_scores.append(score)
                if verbose: print(f"\n✅ Model {i+1} VAL MAE: {score:.4f}")
        
        # Performansa göre ağırlıkları hesapla
        if self.model_type == 'regressor':
            # MAE için düşük = iyi, tersini al
            inverse_scores = 1.0 / (np.array(individual_scores) + 1e-8)
            self.weights = inverse_scores / inverse_scores.sum()
        else:
            # Accuracy için yüksek = iyi
            scores_array = np.array(individual_scores)
            self.weights = scores_array / scores_array.sum()
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"📊 ENSEMBLE AĞIRLIKLARI")
            print(f"{'='*80}")
            for i, (score, weight) in enumerate(zip(individual_scores, self.weights)):
                print(f"Model {i+1}: Score={score:.4f}, Weight={weight:.4f}")
        
        return {
            'individual_scores': individual_scores,
            'weights': self.weights.tolist(),
            'mean_score': np.mean(individual_scores),
            'std_score': np.std(individual_scores)
        }
    
    def predict(self, X, return_variance: bool = False) -> np.ndarray:
        """
        Ensemble tahmin (Regressor: Değer, Classifier: Sınıf)
        """
        if not self.models:
            raise ValueError("Ensemble henüz eğitilmedi!")
        
        # Her modelden tahmin al
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)  # Shape: (n_models, n_samples)
        
        # Ağırlıklı ortalama (Regressor için değer, Classifier için sınıf ortalaması - dikkat!)
        # Classifier için hard voting daha doğru olabilir, ama weighted average soft voting'e benzer.
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)
        
        if self.model_type == 'classifier':
            # Sınıf tahminlerini tekrar 0/1'e yuvarla (Soft voting sonrası hard decision)
            weighted_pred = (weighted_pred >= 0.5).astype(int)
        
        if return_variance:
            # Variance hesapla (model agreement için)
            variance = np.var(predictions, axis=0)
            return weighted_pred, variance
        
        return weighted_pred

    def predict_proba(self, X, return_variance: bool = False) -> np.ndarray:
        """
        Classifier için probability tahmini (Soft Voting)
        """
        if self.model_type != 'classifier':
            raise ValueError("predict_proba sadece classifier için kullanılabilir!")
        
        if not self.models:
            raise ValueError("Ensemble henüz eğitilmedi!")
        
        # Her modelden probability al
        probabilities = []
        for model in self.models:
            # [:, 1] -> Class 1 (1.5 üstü) olasılığı
            # predict_proba (n_samples, 2) döner
            proba = model.predict_proba(X)
            probabilities.append(proba)
        
        probabilities = np.array(probabilities) # Shape: (n_models, n_samples, 2)
        
        # Ağırlıklı ortalama
        weighted_proba = np.average(probabilities, axis=0, weights=self.weights)
        
        if return_variance:
            # Class 1 için variance (1.5 üstü olma olasılığı üzerinden)
            variance = np.var(probabilities[:, :, 1], axis=0)
            return weighted_proba, variance
        
        return weighted_proba
    
    def get_confidence(self, X: np.ndarray) -> np.ndarray:
        """
        Tahmin güven skorları (variance-based)
        Düşük variance = yüksek güven
        """
        if self.model_type == 'regressor':
            _, variance = self.predict(X, return_variance=True)
        else:
            _, variance = self.predict_proba(X, return_variance=True)
        
        # Variance'ı confidence'a çevir (0-1 arası)
        # Düşük variance = yüksek confidence
        max_variance = np.max(variance) if np.max(variance) > 0 else 1.0
        confidence = 1.0 - (variance / max_variance)
        
        return confidence
    
    def save_ensemble(self, save_dir: str):
        """Ensemble'ı kaydet"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Her modeli kaydet
        for i, model in enumerate(self.models):
            model_path = save_path / f"model_{i}.cbm"
            model.save_model(str(model_path))
        
        # Metadata kaydet
        metadata = {
            'model_type': self.model_type,
            'n_models': self.n_models,
            'weights': self.weights.tolist(),
            'base_params': self.base_params,
            'performance_history': self.performance_history
        }
        
        with open(save_path / 'ensemble_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Ensemble kaydedildi: {save_path}")
    
    def load_ensemble(self, load_dir: str):
        """Ensemble'ı yükle"""
        load_path = Path(load_dir)
        
        # Metadata yükle
        with open(load_path / 'ensemble_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        self.model_type = metadata['model_type']
        self.n_models = metadata['n_models']
        self.weights = np.array(metadata['weights'])
        self.base_params = metadata['base_params']
        self.performance_history = metadata.get('performance_history', [])
        
        # Modelleri yükle
        self.models = []
        for i in range(self.n_models):
            model_path = load_path / f"model_{i}.cbm"
            
            if self.model_type == 'regressor':
                model = CatBoostRegressor()
            else:
                model = CatBoostClassifier()
            
            model.load_model(str(model_path))
            self.models.append(model)
        
        logger.info(f"Ensemble yüklendi: {load_path}")


class CrossValidatedEnsemble:
    """
    Cross-validation ile eğitilmiş ensemble
    """
    
    def __init__(
        self,
        model_type: str = 'regressor',
        n_folds: int = 5,
        base_params: Optional[Dict] = None
    ):
        self.model_type = model_type
        self.n_folds = n_folds
        self.base_params = base_params or {}
        
        self.fold_models: List = []
        self.fold_scores: List[float] = []
        
    def time_series_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Kronolojik cross-validation split (time-series için)
        """
        n_samples = len(X)
        fold_size = n_samples // (n_folds + 1)
        
        splits = []
        for i in range(n_folds):
            train_end = fold_size * (i + 1)
            val_end = train_end + fold_size
            
            X_train = X[:train_end]
            y_train = y[:train_end]
            X_val = X[train_end:val_end]
            y_val = y[train_end:val_end]
            
            splits.append((X_train, X_val, y_train, y_val))
        
        return splits
    
    def train_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True
    ) -> Dict:
        """Cross-validation ile eğit"""
        self.fold_models = []
        self.fold_scores = []
        
        # Time-series split
        splits = self.time_series_split(X, y, self.n_folds)
        
        for fold_idx, (X_train, X_val, y_train, y_val) in enumerate(splits):
            if verbose:
                print(f"\n{'='*80}")
                print(f"📊 FOLD {fold_idx + 1}/{self.n_folds}")
                print(f"{'='*80}")
            
            # Model oluştur ve eğit
            if self.model_type == 'regressor':
                model = CatBoostRegressor(**self.base_params)
            else:
                model = CatBoostClassifier(**self.base_params)
            
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                verbose=100 if verbose else False
            )
            
            self.fold_models.append(model)
            
            # Validation performansı
            val_pred = model.predict(X_val)
            
            if self.model_type == 'regressor':
                from sklearn.metrics import mean_absolute_error
                score = mean_absolute_error(y_val, val_pred)
                self.fold_scores.append(score)
                if verbose: print(f"\n✅ Fold {fold_idx + 1} VAL MAE: {score:.4f}")
            else:
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y_val, val_pred)
                self.fold_scores.append(score)
                if verbose: print(f"\n✅ Fold {fold_idx + 1} VAL ACC: {score*100:.2f}%")
        
        return {
            'fold_scores': self.fold_scores,
            'mean_score': np.mean(self.fold_scores),
            'std_score': np.std(self.fold_scores)
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """CV ensemble tahmini (tüm fold modellerin ortalaması)"""
        if not self.fold_models:
            raise ValueError("CV ensemble henüz eğitilmedi!")
        
        predictions = []
        for model in self.fold_models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Ortalama al
        return np.mean(predictions, axis=0)
