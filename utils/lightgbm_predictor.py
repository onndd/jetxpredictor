"""
LightGBM Predictor - CPU Optimized Lightweight Model

LightGBM ile CPU üzerinde hızlı eğitim ve tahmin yapan hafif model.
Binary classification, multi-class classification ve regression destekler.
"""

import numpy as np
import pandas as pd
import joblib
import os
import logging
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# LightGBM için lazy import
try:
    import lightgbm as lgb
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM yüklü değil. 'pip install lightgbm' ile yükleyin.")


class LightGBMPredictor:
    """
    LightGBM tabanlı tahmin sınıfı
    
    Özellikler:
    - CPU optimized hyperparameters
    - Binary classification (1.5x threshold)
    - Multi-class classification (kategoriler)
    - Regression mode
    - Feature importance extraction
    - Cross-validation support
    - Early stopping
    """
    
    # Kategoriler
    CATEGORIES = {
        0: 'Düşük (< 1.5x)',
        1: 'Orta (1.5x - 10x)',
        2: 'Yüksek (10x - 50x)',
        3: 'Mega (50x+)'
    }
    
    def __init__(
        self,
        model_path: str = "models/cpu/lightgbm_model.txt",
        scaler_path: str = "models/cpu/lightgbm_scaler.pkl",
        threshold: float = 1.5,
        mode: str = 'classification'
    ):
        """
        Args:
            model_path: LightGBM model dosyası
            scaler_path: Scaler dosyası
            threshold: Eşik değeri (varsayılan 1.5)
            mode: Model modu ('classification', 'regression', 'multiclass')
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM yüklü değil. 'pip install lightgbm' ile yükleyin.")
            
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.threshold = threshold
        self.mode = mode
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        
        # CPU optimized default parameters
        self.default_params = {
            'classification': {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'max_depth': 8,
                'learning_rate': 0.03,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_data_in_leaf': 20,
                'min_sum_hessian_in_leaf': 1e-3,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'device': 'cpu',
                'verbosity': -1,
                'random_state': 42
            },
            'multiclass': {
                'objective': 'multiclass',
                'num_class': 4,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'max_depth': 8,
                'learning_rate': 0.03,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_data_in_leaf': 20,
                'min_sum_hessian_in_leaf': 1e-3,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'device': 'cpu',
                'verbosity': -1,
                'random_state': 42
            },
            'regression': {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'max_depth': 8,
                'learning_rate': 0.03,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_data_in_leaf': 20,
                'min_sum_hessian_in_leaf': 1e-3,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'device': 'cpu',
                'verbosity': -1,
                'random_state': 42
            }
        }
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Optional[Dict] = None,
        cv_folds: int = 5,
        early_stopping_rounds: int = 50,
        num_boost_round: int = 1000,
        validation_split: float = 0.2
    ) -> Dict:
        """
        Model eğitimi
        
        Args:
            X: Feature matrix
            y: Target values
            params: LightGBM parametreleri
            cv_folds: Cross-validation fold sayısı
            early_stopping_rounds: Early stopping rounds
            num_boost_round: Maximum boosting rounds
            validation_split: Validation split ratio
            
        Returns:
            Training metrics dictionary
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM yüklü değil.")
            
        logger.info(f"LightGBM {self.mode} modeli eğitiliyor...")
        
        # Parametreleri ayarla
        if params is None:
            params = self.default_params[self.mode].copy()
        
        # Veri split
        n_samples = len(X)
        val_size = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)
        
        train_idx = indices[val_size:]
        val_idx = indices[:val_size]
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # LightGBM Dataset oluştur
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Model eğitimi
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=num_boost_round,
            callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(0)]
        )
        
        # Feature names kaydet
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Validation metrics hesapla
        val_pred = self.model.predict(X_val, num_iteration=self.model.best_iteration)
        
        metrics = self._calculate_metrics(y_val, val_pred)
        
        # Cross-validation
        if cv_folds > 1:
            cv_scores = self._cross_validate(X, y, params, cv_folds)
            metrics['cv_scores'] = cv_scores
            metrics['cv_mean'] = np.mean(cv_scores)
            metrics['cv_std'] = np.std(cv_scores)
        
        self.is_trained = True
        logger.info(f"Model eğitimi tamamlandı. Best iteration: {self.model.best_iteration}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Tahmin yapma
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmemiş. Önce train() metodunu çağırın.")
        
        predictions = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        if self.mode == 'classification':
            # Binary classification için threshold uygula
            return (predictions >= 0.5).astype(int)
        elif self.mode == 'multiclass':
            # Multi-class için argmax
            return np.argmax(predictions, axis=1)
        else:
            # Regression için direkt döndür
            return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Probability tahminleri
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability predictions
        """
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmemiş.")
        
        if self.mode == 'regression':
            raise ValueError("Probability prediction sadece classification modunda kullanılabilir.")
        
        return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Feature importance çıkarma
        
        Args:
            importance_type: 'gain', 'split', 'split_count'
            
        Returns:
            Feature importance DataFrame
        """
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmemiş.")
        
        importance = self.model.feature_importance(importance_type=importance_type)
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Metrics hesaplama"""
        metrics = {}
        
        if self.mode == 'classification':
            metrics['accuracy'] = accuracy_score(y_true, (y_pred >= 0.5).astype(int))
            metrics['threshold_prob'] = np.mean(y_pred)
            
        elif self.mode == 'multiclass':
            y_pred_class = np.argmax(y_pred, axis=1)
            metrics['accuracy'] = accuracy_score(y_true, y_pred_class)
            
        else:  # regression
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_true, y_pred)
        
        return metrics
    
    def _cross_validate(self, X: np.ndarray, y: np.ndarray, params: Dict, cv_folds: int) -> List[float]:
        """Cross-validation"""
        if self.mode == 'classification':
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scoring = 'accuracy'
        else:
            cv = cv_folds
            scoring = 'neg_mean_absolute_error'
        
        # LightGBM için cross-validation
        cv_results = lgb.cv(
            params,
            lgb.Dataset(X, label=y),
            num_boost_round=1000,
            nfold=cv_folds,
            stratified=(self.mode == 'classification'),
            shuffle=True,
            return_cvbooster=False
        )
        
        if self.mode == 'classification':
            return cv_results['binary_logloss-mean']
        else:
            return cv_results['mae-mean']
    
    def save_model(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None):
        """
        Model kaydetme
        
        Args:
            model_path: Model dosya yolu
            scaler_path: Scaler dosya yolu
        """
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmemiş.")
        
        model_path = model_path or self.model_path
        scaler_path = scaler_path or self.scaler_path
        
        # Klasörleri oluştur
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        
        # Model kaydet
        self.model.save_model(model_path)
        
        # Scaler kaydet (varsa)
        if self.scaler is not None:
            joblib.dump(self.scaler, scaler_path)
        
        # Model info kaydet
        info_path = model_path.replace('.txt', '_info.json')
        import json
        model_info = {
            'mode': self.mode,
            'threshold': self.threshold,
            'feature_names': self.feature_names,
            'best_iteration': self.model.best_iteration,
            'num_features': len(self.feature_names) if self.feature_names else 0
        }
        
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Model kaydedildi: {model_path}")
    
    def load_model(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None):
        """
        Model yükleme
        
        Args:
            model_path: Model dosya yolu
            scaler_path: Scaler dosya yolu
        """
        model_path = model_path or self.model_path
        scaler_path = scaler_path or self.scaler_path
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
        
        # Model yükle
        self.model = lgb.Booster(model_file=model_path)
        
        # Scaler yükle (varsa)
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        # Model info yükle
        info_path = model_path.replace('.txt', '_info.json')
        if os.path.exists(info_path):
            import json
            with open(info_path, 'r') as f:
                model_info = json.load(f)
                self.mode = model_info.get('mode', self.mode)
                self.threshold = model_info.get('threshold', self.threshold)
                self.feature_names = model_info.get('feature_names', self.feature_names)
        
        self.is_trained = True
        logger.info(f"Model yüklendi: {model_path}")
    
    def get_model_info(self) -> Dict:
        """Model bilgilerini döndür"""
        if not self.is_trained:
            return {'status': 'not_trained'}
        
        return {
            'mode': self.mode,
            'threshold': self.threshold,
            'best_iteration': self.model.best_iteration,
            'num_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'model_path': self.model_path,
            'scaler_path': self.scaler_path
        }
