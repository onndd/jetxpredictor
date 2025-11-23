"""
AutoGluon Predictor - Otomatik ML Champion

AutoGluon birden fazla modeli otomatik olarak dener ve en iyisini seçer.
Genel amaçlı 1.5x eşik tahmini için kullanılır.

GÜNCELLEME:
- 2 Modlu Yapı (Normal/Rolling) entegrasyonu.
- Normal Mod Eşik: 0.85
- Rolling Mod Eşik: 0.95
"""

import numpy as np
import pandas as pd
import joblib
import os
import logging
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# AutoGluon için lazy import
try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False
    logger.warning("AutoGluon yüklü değil. 'pip install autogluon' ile yükleyin.")


class AutoGluonPredictor:
    """
    AutoGluon tabanlı tahmin sınıfı
    """
    
    # Eşikler
    THRESHOLD_NORMAL = 0.85
    THRESHOLD_ROLLING = 0.95
    
    def __init__(
        self,
        model_path: str = "models/autogluon_model",
        scaler_path: str = "models/autogluon_scaler.pkl",
        threshold: float = 1.5
    ):
        """
        Args:
            model_path: AutoGluon model klasörü
            scaler_path: Scaler dosyası
            threshold: Eşik değeri (varsayılan 1.5)
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.threshold = threshold
        self.predictor = None
        self.scaler = None
        
        if not AUTOGLUON_AVAILABLE:
            raise RuntimeError("AutoGluon yüklü değil! 'pip install autogluon' ile yükleyin.")
        
        # Model varsa yükle
        if os.path.exists(model_path):
            self.load_model()
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        time_limit: int = 3600,
        presets: str = 'best_quality',
        eval_metric: str = 'roc_auc',
        **kwargs
    ) -> Dict:
        """AutoGluon modelini eğit"""
        logger.info("=" * 70)
        logger.info("AutoGluon Eğitimi Başlıyor...")
        logger.info(f"Time limit: {time_limit}s (~{time_limit/60:.0f} dakika)")
        logger.info("=" * 70)
        
        # Train dataframe oluştur
        train_data = X_train.copy()
        train_data['above_threshold'] = y_train
        
        # AutoGluon predictor oluştur
        self.predictor = TabularPredictor(
            label='above_threshold',
            problem_type='binary',
            eval_metric=eval_metric,
            path=self.model_path
        )
        
        # Eğit
        self.predictor.fit(
            train_data,
            presets=presets,
            time_limit=time_limit,
            num_bag_folds=5,  # 5-fold bagging
            num_stack_levels=2,  # 2-level stacking
            **kwargs
        )
        
        # Leaderboard
        leaderboard = self.predictor.leaderboard(silent=True)
        
        logger.info("\nAutoGluon Eğitimi Tamamlandı!")
        
        # Feature importance
        try:
            feature_importance = self.predictor.feature_importance(train_data)
        except:
            feature_importance = None
        
        return {
            'leaderboard': leaderboard,
            'feature_importance': feature_importance,
            'best_model': leaderboard.iloc[0]['model'],
            'best_score': leaderboard.iloc[0]['score_val']
        }
    
    def predict(
        self,
        X: np.ndarray,
        return_proba: bool = True
    ) -> Dict:
        """
        Tahmin yap (2 Modlu)
        """
        if self.predictor is None:
            raise RuntimeError("Model henüz yüklenmedi veya eğitilmedi!")
        
        # Input validation
        try:
            # Type conversion ve validation
            if isinstance(X, list): X = np.array(X)
            elif isinstance(X, pd.Series): X = X.values.reshape(1, -1)
            
            # DataFrame'e çevir
            if isinstance(X, np.ndarray):
                X_df = pd.DataFrame(X)
            else:
                X_df = X
                
        except Exception as e:
            logger.error(f"AutoGluon input validation hatası: {e}")
            return {'error': str(e)}
        
        # Tahmin
        if return_proba:
            proba = self.predictor.predict_proba(X_df)
            # Binary classification için 1.5 üstü olma olasılığı
            # AutoGluon predict_proba 0 ve 1 sınıfları için sütunlar döner
            # Genelde 1. sınıf (True/1) ikinci sütundadır (index 1)
            if 1 in proba.columns:
                threshold_prob = float(proba.iloc[0, 1]) if len(proba.shape) > 1 else float(proba.iloc[0])
            else:
                 # Eğer sütun isimleri farklıysa, son sütunu al (genelde positive class)
                 threshold_prob = float(proba.iloc[0, -1])
        else:
            prediction = self.predictor.predict(X_df)
            threshold_prob = float(prediction.iloc[0])
        
        # Confidence
        confidence = threshold_prob # Direkt 1.5 üstü olma olasılığı
        
        # Mod Kararları
        should_bet_normal = confidence >= self.THRESHOLD_NORMAL
        should_bet_rolling = confidence >= self.THRESHOLD_ROLLING
        
        return {
            'threshold_probability': threshold_prob,
            'confidence': confidence,
            'above_threshold': threshold_prob >= 0.5,
            'should_bet_normal': should_bet_normal,
            'should_bet_rolling': should_bet_rolling,
            'recommendation': self._get_recommendation(confidence)
        }
    
    def _get_recommendation(self, confidence: float) -> str:
        """Öneri oluştur (2 Modlu)"""
        if confidence >= self.THRESHOLD_ROLLING:
            return 'ROLLING MOD (Çok Güçlü)'
        elif confidence >= self.THRESHOLD_NORMAL:
            return 'NORMAL MOD (Güçlü)'
        else:
            return 'BEKLE'

    def load_model(self):
        """Modeli yükle"""
        try:
            self.predictor = TabularPredictor.load(self.model_path)
            logger.info(f"✅ AutoGluon modeli yüklendi: {self.model_path}")
            
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"✅ Scaler yüklendi: {self.scaler_path}")
        except Exception as e:
            logger.error(f"❌ Model yükleme hatası: {e}")
            self.predictor = None
    
    def save_scaler(self, scaler):
        """Scaler'ı kaydet"""
        joblib.dump(scaler, self.scaler_path)
        logger.info(f"✅ Scaler kaydedildi: {self.scaler_path}")
    
    def get_leaderboard(self) -> pd.DataFrame:
        """Model leaderboard'unu döndür"""
        if self.predictor is None:
            raise RuntimeError("Model henüz yüklenmedi!")
        return self.predictor.leaderboard(silent=True)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Feature importance döndür"""
        if self.predictor is None:
            raise RuntimeError("Model henüz yüklenmedi!")
        # Feature importance hesabı zaman alabilir, önceden hesaplanmışsa onu kullanmak daha iyi
        # Burada boş bir veri seti veremeyeceğimiz için uyarı verip geçiyoruz
        logger.warning("Feature importance için eğitim verisi gerekli. Eğitim sırasında hesaplanan değerleri kullanın.")
        return pd.DataFrame()
