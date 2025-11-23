"""
TabNet Predictor - Yüksek X Specialist

TabNet attention-based deep learning modeli ile yüksek çarpanları tespit eder.

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
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# TabNet için lazy import
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    logger.warning("TabNet yüklü değil. 'pip install pytorch-tabnet' ile yükleyin.")


class TabNetHighXPredictor:
    """
    TabNet tabanlı yüksek X tahmin sınıfı
    """
    
    # Kategoriler
    CATEGORIES = {
        0: 'Düşük (< 1.5x)',
        1: 'Orta (1.5x - 10x)',
        2: 'Yüksek (10x - 50x)',
        3: 'Mega (50x+)'
    }
    
    # Eşikler
    THRESHOLD_NORMAL = 0.85
    THRESHOLD_ROLLING = 0.95
    
    def __init__(
        self,
        model_path: str = "models/tabnet_high_x.pkl",
        scaler_path: str = "models/tabnet_scaler.pkl"
    ):
        """
        Args:
            model_path: TabNet model dosyası
            scaler_path: Scaler dosyası
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        
        if not TABNET_AVAILABLE:
            raise RuntimeError("TabNet yüklü değil! 'pip install pytorch-tabnet' ile yükleyin.")
        
        # Model varsa yükle
        if os.path.exists(model_path):
            self.load_model()
    
    @staticmethod
    def categorize_value(value) -> int:
        """Değeri kategoriye çevir"""
        try:
            if isinstance(value, str): value = float(value)
            elif not isinstance(value, (int, float)): raise ValueError(f"Geçersiz değer tipi: {type(value)}")
            
            if pd.isna(value) or value is None: raise ValueError("Değer None veya NaN")
            if value <= 0: raise ValueError(f"Değer pozitif olmalı: {value}")
            
            if value < 1.5: return 0
            elif value < 10: return 1
            elif value < 50: return 2
            else: return 3
            
        except (ValueError, TypeError) as e:
            logger.error(f"categorize_value hatası: {e}, value: {value}")
            return 0
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        max_epochs: int = 200,
        patience: int = 20,
        batch_size: int = 256,
        **kwargs
    ) -> Dict:
        """TabNet modelini eğit"""
        logger.info("=" * 70)
        logger.info("TabNet Eğitimi Başlıyor (Yüksek X Specialist)...")
        logger.info(f"Max epochs: {max_epochs}")
        logger.info(f"Patience: {patience}")
        logger.info("=" * 70)
        
        tabnet_params = {
            'n_d': kwargs.get('n_d', 64),
            'n_a': kwargs.get('n_a', 64),
            'n_steps': kwargs.get('n_steps', 5),
            'gamma': kwargs.get('gamma', 1.5),
            'lambda_sparse': kwargs.get('lambda_sparse', 1e-3),
            'optimizer_fn': torch.optim.Adam,
            'optimizer_params': dict(lr=kwargs.get('lr', 2e-2)),
            'mask_type': kwargs.get('mask_type', 'entmax'),
            'scheduler_params': {"step_size": 10, "gamma": 0.9},
            'scheduler_fn': torch.optim.lr_scheduler.StepLR,
            'seed': 42,
            'verbose': 1
        }
        
        self.model = TabNetClassifier(**tabnet_params)
        
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        self.model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=eval_set,
            eval_name=['val'] if eval_set else None,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )
        
        feature_importances = self.model.feature_importances_
        
        logger.info("\nTabNet Eğitimi Tamamlandı!")
        
        return {
            'feature_importances': feature_importances,
            'best_epoch': self.model.best_epoch,
            'best_cost': self.model.best_cost
        }
    
    def predict(
        self,
        X: np.ndarray,
        return_attention: bool = False
    ) -> Dict:
        """
        Tahmin yap (2 Modlu)
        """
        if self.model is None:
            raise RuntimeError("Model henüz yüklenmedi veya eğitilmedi!")
        
        predictions = self.model.predict(X)
        proba = self.model.predict_proba(X)
        
        # İlk tahmin için detaylı sonuç
        predicted_category = int(predictions[0])
        category_name = self.CATEGORIES[predicted_category]
        category_probs = {
            self.CATEGORIES[i]: float(proba[0][i])
            for i in range(len(self.CATEGORIES))
        }
        
        # Yüksek X olasılığı (kategori 2 ve 3'ün toplamı)
        high_x_prob = float(proba[0][2] + proba[0][3])
        mega_x_prob = float(proba[0][3])
        
        # Confidence (1.5 üstü olma olasılığı = 1 - Class 0)
        # TabNet High X Specialist olduğu için yüksek çarpanlara odaklanır
        # Ancak "1.5 Üstü" kararı için Class 0 dışındakilerin toplamı mantıklıdır
        confidence = 1.0 - float(proba[0][0])
        
        # Mod Kararları
        should_bet_normal = confidence >= self.THRESHOLD_NORMAL
        should_bet_rolling = confidence >= self.THRESHOLD_ROLLING
        
        result = {
            'predicted_category': predicted_category,
            'category_name': category_name,
            'confidence': confidence,
            'high_x_probability': high_x_prob,
            'mega_x_probability': mega_x_prob,
            'category_probabilities': category_probs,
            'should_bet_normal': should_bet_normal,
            'should_bet_rolling': should_bet_rolling,
            'recommendation': self._get_recommendation(confidence)
        }
        
        if return_attention:
            try:
                explain_matrix, masks = self.model.explain(X)
                result['attention_masks'] = masks
                result['explain_matrix'] = explain_matrix
            except Exception as e:
                logger.warning(f"Attention mask alınamadı: {e}")
        
        return result
    
    def _get_recommendation(self, confidence: float) -> str:
        """Öneri oluştur (2 Modlu)"""
        if confidence >= self.THRESHOLD_ROLLING:
            return 'ROLLING MOD (Çok Güçlü)'
        elif confidence >= self.THRESHOLD_NORMAL:
            return 'NORMAL MOD (Güçlü)'
        elif confidence >= 0.70:
            return 'ORTA GÜVEN (Riskli)'
        else:
            return 'DÜŞÜK GÜVEN (Bekle)'
    
    def save_model(self):
        """Modeli kaydet"""
        if self.model is None:
            raise RuntimeError("Kaydedilecek model yok!")
        
        self.model.save_model(self.model_path)
        logger.info(f"✅ TabNet modeli kaydedildi: {self.model_path}")
    
    def load_model(self):
        """Modeli yükle"""
        try:
            self.model = TabNetClassifier()
            self.model.load_model(self.model_path)
            logger.info(f"✅ TabNet modeli yüklendi: {self.model_path}")
            
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"✅ Scaler yüklendi: {self.scaler_path}")
        except Exception as e:
            logger.error(f"❌ Model yükleme hatası: {e}")
            self.model = None

def create_tabnet_high_x_predictor(
    model_path: str = "models/tabnet_high_x.pkl",
    scaler_path: str = "models/tabnet_scaler.pkl"
) -> TabNetHighXPredictor:
    return TabNetHighXPredictor(model_path=model_path, scaler_path=scaler_path)
