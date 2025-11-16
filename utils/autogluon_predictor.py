"""
AutoGluon Predictor - Otomatik ML Champion

AutoGluon birden fazla modeli otomatik olarak dener ve en iyisini seçer.
Genel amaçlı 1.5x eşik tahmini için kullanılır.
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
    
    Özellikler:
    - 50+ farklı modeli otomatik dener
    - Ensemble ve stacking otomatik yapar
    - Hyperparameter tuning otomatik
    - 1.5x eşik tahmini için optimize edilmiş
    """
    
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
        """
        AutoGluon modelini eğit
        
        Args:
            X_train: Training features
            y_train: Training labels (binary: 0=below threshold, 1=above threshold)
            time_limit: Maksimum eğitim süresi (saniye)
            presets: AutoGluon preset ('best_quality', 'high_quality', 'medium_quality')
            eval_metric: Değerlendirme metriği
            **kwargs: Ek AutoGluon parametreleri
            
        Returns:
            Training sonuçları dict
        """
        logger.info("=" * 70)
        logger.info("AutoGluon Eğitimi Başlıyor...")
        logger.info(f"Time limit: {time_limit}s (~{time_limit/60:.0f} dakika)")
        logger.info(f"Preset: {presets}")
        logger.info(f"Eval metric: {eval_metric}")
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
        
        logger.info("\n" + "=" * 70)
        logger.info("AutoGluon Eğitimi Tamamlandı!")
        logger.info("=" * 70)
        logger.info(f"\nEn İyi Model: {leaderboard.iloc[0]['model']}")
        logger.info(f"Score: {leaderboard.iloc[0]['score_val']:.4f}")
        logger.info(f"\nToplam {len(leaderboard)} model eğitildi")
        
        # Feature importance
        feature_importance = self.predictor.feature_importance(train_data)
        
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
        Tahmin yap
        
        Args:
            X: Input features
            return_proba: Olasılık döndür mü?
            
        Returns:
            Tahmin sonuçları dict
        """
        if self.predictor is None:
            raise RuntimeError("Model henüz yüklenmedi veya eğitilmedi!")
        
        # Input validation
        try:
            # Type conversion ve validation
            if isinstance(X, list):
                X = np.array(X)
            elif isinstance(X, pd.Series):
                X = X.values.reshape(1, -1)
            elif not isinstance(X, (np.ndarray, pd.DataFrame)):
                raise ValueError(f"Geçersiz input tipi: {type(X)}")
            
            # NaN kontrolü
            if hasattr(X, 'isna'):
                if X.isna().any().any():
                    raise ValueError("Input verisinde NaN değerler var")
            elif pd.isna(X).any():
                raise ValueError("Input verisinde NaN değerler var")
            
            # DataFrame'e çevir
            if isinstance(X, np.ndarray):
                X_df = pd.DataFrame(X)
            else:
                X_df = X
                
        except Exception as e:
            logger.error(f"AutoGluon input validation hatası: {e}")
            # Hata durumunda varsayılan sonuç dön
            return {
                'threshold_probability': 0.5,
                'confidence': 0.5,
                'above_threshold': False,
                'recommendation': 'BEKLE',
                'error': str(e)
            }
        
        # Tahmin
        if return_proba:
            proba = self.predictor.predict_proba(X_df)
            # Binary classification için 1.5 üstü olma olasılığı
            threshold_prob = float(proba.iloc[0, 1]) if len(proba.shape) > 1 else float(proba.iloc[0])
        else:
            prediction = self.predictor.predict(X_df)
            threshold_prob = 0.5  # Default
        
        # Confidence ve recommendation
        confidence = max(threshold_prob, 1 - threshold_prob)
        above_threshold = threshold_prob >= 0.5
        
        return {
            'threshold_probability': threshold_prob,
            'confidence': confidence,
            'above_threshold': above_threshold,
            'recommendation': 'OYNA' if (above_threshold and confidence >= 0.65) else 'BEKLE'
        }
    
    def load_model(self):
        """Modeli yükle"""
        try:
            self.predictor = TabularPredictor.load(self.model_path)
            logger.info(f"✅ AutoGluon modeli yüklendi: {self.model_path}")
            
            # Scaler varsa yükle
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
        # Boş dataframe ile çağır (eğitimden bilgi kullanır)
        return self.predictor.feature_importance()
