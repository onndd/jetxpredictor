"""
LightGBM Predictor - CPU Optimized Lightweight Model

LightGBM ile CPU üzerinde hızlı eğitim ve tahmin yapan hafif model.
Binary classification, multi-class classification ve regression destekler.

GÜNCELLEME:
- 2 Modlu Yapı (Normal/Rolling) entegrasyonu.
- Threshold Manager entegrasyonu.
"""

import numpy as np
import pandas as pd
import joblib
import os
import logging
from typing import Dict, List, Optional, Tuple, Union
import warnings
from utils.threshold_manager import get_threshold_manager

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
        mode: str = 'classification'
    ):
        """
        Args:
            model_path: LightGBM model dosyası
            scaler_path: Scaler dosyası
            mode: Model modu ('classification', 'regression', 'multiclass')
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM yüklü değil. 'pip install lightgbm' ile yükleyin.")
            
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.mode = mode
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        
        # Threshold Manager
        tm = get_threshold_manager()
        self.THRESHOLD_NORMAL = tm.get_normal_threshold()   # 0.85
        self.THRESHOLD_ROLLING = tm.get_rolling_threshold() # 0.95
        
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
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'device': 'cpu',
                'verbosity': -1,
                'random_state': 42
            }
        }
        
        # Model varsa yükle
        if os.path.exists(model_path):
            self.load_model()

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
        """Model eğitimi"""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM yüklü değil.")
            
        logger.info(f"LightGBM {self.mode} modeli eğitiliyor...")
        
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
        
        self.is_trained = True
        logger.info(f"Model eğitimi tamamlandı. Best iteration: {self.model.best_iteration}")
        
        return metrics
    
    def predict(self, X) -> Dict:
        """
        Tahmin yapma (2 Modlu)
        """
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmemiş. Önce train() metodunu çağırın.")
        
        # Input validation ve DataFrame dönüşümü
        try:
            if isinstance(X, list): X = np.array(X)
            elif isinstance(X, pd.Series): X = X.values.reshape(1, -1)
            if isinstance(X, pd.DataFrame): X = X.values
        except Exception as e:
            logger.error(f"LightGBM input validation hatası: {e}")
            return {'error': str(e)}
        
        raw_pred = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        result = {}
        confidence = 0.0
        
        if self.mode == 'classification':
            # Binary classification: raw_pred olasılıktır (1. sınıf olma ihtimali)
            confidence = float(raw_pred[0])
            result['threshold_probability'] = confidence
            result['prediction'] = 1 if confidence >= 0.5 else 0
            
        elif self.mode == 'multiclass':
            # Multi-class: En yüksek olasılık
            class_idx = np.argmax(raw_pred[0])
            confidence = float(raw_pred[0][class_idx])
            result['prediction'] = int(class_idx)
            # Multiclass'ta "1.5 üstü" olma ihtimali Class 1,2,3 toplamı olabilir
            # Basitlik için confidence'ı direkt alıyoruz
            
        else: # Regression
            # Regression'da confidence hesaplamak zordur, dummy değer
            prediction = float(raw_pred[0])
            confidence = 0.5 # Nötr
            result['prediction'] = prediction
            
        # Mod Kararları
        should_bet_normal = confidence >= self.THRESHOLD_NORMAL
        should_bet_rolling = confidence >= self.THRESHOLD_ROLLING
        
        result.update({
            'confidence': confidence,
            'should_bet_normal': should_bet_normal,
            'should_bet_rolling': should_bet_rolling,
            'recommendation': self._get_recommendation(confidence)
        })
        
        return result

    def _get_recommendation(self, confidence: float) -> str:
        if confidence >= self.THRESHOLD_ROLLING: return 'ROLLING MOD (Çok Güçlü)'
        elif confidence >= self.THRESHOLD_NORMAL: return 'NORMAL MOD (Güçlü)'
        else: return 'BEKLE'
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Metrics hesaplama"""
        metrics = {}
        if self.mode == 'classification':
            metrics['accuracy'] = accuracy_score(y_true, (y_pred >= 0.5).astype(int))
        elif self.mode == 'multiclass':
            y_pred_class = np.argmax(y_pred, axis=1)
            metrics['accuracy'] = accuracy_score(y_true, y_pred_class)
        else: # regression
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
        return metrics

    def save_model(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None):
        """Model kaydetme"""
        if not self.is_trained: raise ValueError("Model henüz eğitilmemiş.")
        model_path = model_path or self.model_path
        scaler_path = scaler_path or self.scaler_path
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        
        self.model.save_model(model_path)
        if self.scaler is not None: joblib.dump(self.scaler, scaler_path)
        logger.info(f"Model kaydedildi: {model_path}")

    def load_model(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None):
        """Model yükleme"""
        model_path = model_path or self.model_path
        scaler_path = scaler_path or self.scaler_path
        
        if not os.path.exists(model_path): raise FileNotFoundError(f"Model bulunamadı: {model_path}")
        
        self.model = lgb.Booster(model_file=model_path)
        if os.path.exists(scaler_path): self.scaler = joblib.load(scaler_path)
        
        self.is_trained = True
        logger.info(f"Model yüklendi: {model_path}")
