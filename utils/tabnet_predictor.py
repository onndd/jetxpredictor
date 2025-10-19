"""
TabNet Predictor - Yüksek X Specialist

TabNet attention-based deep learning modeli ile yüksek çarpanları (10x+, 20x+, 50x+, 100x+) tespit eder.
Feature importance ve attention visualization özellikleri ile interpretable tahminler yapar.
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
    from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
    import torch
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    logger.warning("TabNet yüklü değil. 'pip install pytorch-tabnet' ile yükleyin.")


class TabNetHighXPredictor:
    """
    TabNet tabanlı yüksek X tahmin sınıfı
    
    Özellikler:
    - Yüksek çarpanları tespit etmeye özelleşmiş
    - Multi-class classification (Düşük/Orta/Yüksek/Mega)
    - Attention mechanism ile feature importance
    - Step-wise attention visualization
    - Interpretable predictions
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
    def categorize_value(value: float) -> int:
        """Değeri kategoriye çevir"""
        if value < 1.5:
            return 0  # Düşük
        elif value < 10:
            return 1  # Orta
        elif value < 50:
            return 2  # Yüksek
        else:
            return 3  # Mega
    
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
        """
        TabNet modelini eğit
        
        Args:
            X_train: Training features
            y_train: Training labels (kategorik: 0, 1, 2, 3)
            X_val: Validation features (opsiyonel)
            y_val: Validation labels (opsiyonel)
            max_epochs: Maksimum epoch sayısı
            patience: Early stopping patience
            batch_size: Batch size
            **kwargs: Ek TabNet parametreleri
            
        Returns:
            Training sonuçları dict
        """
        logger.info("=" * 70)
        logger.info("TabNet Eğitimi Başlıyor (Yüksek X Specialist)...")
        logger.info(f"Max epochs: {max_epochs}")
        logger.info(f"Patience: {patience}")
        logger.info(f"Batch size: {batch_size}")
        logger.info("=" * 70)
        
        # TabNet parametreleri
        tabnet_params = {
            'n_d': kwargs.get('n_d', 64),  # Width of decision prediction layer
            'n_a': kwargs.get('n_a', 64),  # Width of attention embedding
            'n_steps': kwargs.get('n_steps', 5),  # Number of steps in the architecture
            'gamma': kwargs.get('gamma', 1.5),  # Coefficient for feature reusage
            'lambda_sparse': kwargs.get('lambda_sparse', 1e-3),  # Sparsity loss weight
            'optimizer_fn': torch.optim.Adam,
            'optimizer_params': dict(lr=kwargs.get('lr', 2e-2)),
            'mask_type': kwargs.get('mask_type', 'entmax'),  # 'sparsemax' or 'entmax'
            'scheduler_params': {"step_size": 10, "gamma": 0.9},
            'scheduler_fn': torch.optim.lr_scheduler.StepLR,
            'seed': 42,
            'verbose': 1
        }
        
        # Model oluştur
        self.model = TabNetClassifier(**tabnet_params)
        
        # Validation set hazırla
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        # Eğit
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
        
        # Feature importance
        feature_importances = self.model.feature_importances_
        
        logger.info("\n" + "=" * 70)
        logger.info("TabNet Eğitimi Tamamlandı!")
        logger.info("=" * 70)
        logger.info(f"\nEn önemli 5 özellik:")
        top_features = np.argsort(feature_importances)[-5:][::-1]
        for idx in top_features:
            logger.info(f"  Feature {idx}: {feature_importances[idx]:.4f}")
        
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
        Tahmin yap
        
        Args:
            X: Input features
            return_attention: Attention mask'lerini döndür mü?
            
        Returns:
            Tahmin sonuçları dict
        """
        if self.model is None:
            raise RuntimeError("Model henüz yüklenmedi veya eğitilmedi!")
        
        # Tahmin
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
        
        # Mega X olasılığı (kategori 3)
        mega_x_prob = float(proba[0][3])
        
        # Confidence (en yüksek olasılık)
        confidence = float(np.max(proba[0]))
        
        result = {
            'predicted_category': predicted_category,
            'category_name': category_name,
            'confidence': confidence,
            'high_x_probability': high_x_prob,  # 10x+ olasılığı
            'mega_x_probability': mega_x_prob,  # 50x+ olasılığı
            'category_probabilities': category_probs,
            'recommendation': self._get_recommendation(high_x_prob, confidence)
        }
        
        # Attention masks (interpretability için)
        if return_attention:
            try:
                explain_matrix, masks = self.model.explain(X)
                result['attention_masks'] = masks
                result['explain_matrix'] = explain_matrix
            except Exception as e:
                logger.warning(f"Attention mask alınamadı: {e}")
        
        return result
    
    def _get_recommendation(self, high_x_prob: float, confidence: float) -> str:
        """Öneri oluştur"""
        if high_x_prob >= 0.6 and confidence >= 0.7:
            return 'YÜKSEK X BEKLENİYOR'
        elif high_x_prob >= 0.4 and confidence >= 0.6:
            return 'ORTA RİSK - YÜKSEK X OLABİLİR'
        else:
            return 'DÜŞÜK X BEKLENİYOR'
    
    def get_feature_importance(self) -> np.ndarray:
        """Feature importance döndür"""
        if self.model is None:
            raise RuntimeError("Model henüz yüklenmedi!")
        return self.model.feature_importances_
    
    def explain_prediction(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Tahmin açıklaması oluştur (interpretability)
        
        Args:
            X: Input features (tek örnek)
            feature_names: Özellik isimleri (opsiyonel)
            
        Returns:
            Açıklama dict
        """
        if self.model is None:
            raise RuntimeError("Model henüz yüklenmedi!")
        
        # Explain
        explain_matrix, masks = self.model.explain(X)
        
        # Feature importance for this specific prediction
        prediction_importance = explain_matrix[0]
        
        # Top 10 önemli özellik
        top_indices = np.argsort(prediction_importance)[-10:][::-1]
        
        top_features = []
        for idx in top_indices:
            feature_dict = {
                'index': int(idx),
                'importance': float(prediction_importance[idx]),
                'value': float(X[0][idx])
            }
            if feature_names and idx < len(feature_names):
                feature_dict['name'] = feature_names[idx]
            top_features.append(feature_dict)
        
        return {
            'top_features': top_features,
            'explain_matrix': explain_matrix,
            'attention_masks': masks
        }
    
    def save_model(self):
        """Modeli kaydet"""
        if self.model is None:
            raise RuntimeError("Kaydedilecek model yok!")
        
        # TabNet modelini kaydet
        self.model.save_model(self.model_path)
        logger.info(f"✅ TabNet modeli kaydedildi: {self.model_path}")
    
    def load_model(self):
        """Modeli yükle"""
        try:
            self.model = TabNetClassifier()
            self.model.load_model(self.model_path)
            logger.info(f"✅ TabNet modeli yüklendi: {self.model_path}")
            
            # Scaler varsa yükle
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"✅ Scaler yüklendi: {self.scaler_path}")
        except Exception as e:
            logger.error(f"❌ Model yükleme hatası: {e}")
            self.model = None
    
    def save_scaler(self, scaler):
        """Scaler'ı kaydet"""
        joblib.dump(scaler, self.scaler_path)
        logger.info(f"✅ Scaler kaydedildi: {self.scaler_path}")
    
    def visualize_attention(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Attention mekanizmasını görselleştir
        
        Args:
            X: Input features
            feature_names: Özellik isimleri
            save_path: Grafik kayıt yolu (opsiyonel)
        """
        if self.model is None:
            raise RuntimeError("Model henüz yüklenmedi!")
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Explain
            explain_matrix, masks = self.model.explain(X)
            
            # Attention masks visualization
            fig, axes = plt.subplots(1, len(masks), figsize=(20, 4))
            
            for i, mask in enumerate(masks):
                if len(masks) > 1:
                    ax = axes[i]
                else:
                    ax = axes
                
                # Heatmap
                sns.heatmap(
                    mask[0].reshape(1, -1),
                    cmap='YlOrRd',
                    ax=ax,
                    cbar=True,
                    xticklabels=feature_names if feature_names else False
                )
                ax.set_title(f'Step {i+1} Attention')
                ax.set_ylabel('Sample')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"✅ Attention visualization kaydedildi: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib/Seaborn yüklü değil, görselleştirme yapılamıyor")
        except Exception as e:
            logger.error(f"Görselleştirme hatası: {e}")


def create_tabnet_high_x_predictor(
    model_path: str = "models/tabnet_high_x.pkl",
    scaler_path: str = "models/tabnet_scaler.pkl"
) -> TabNetHighXPredictor:
    """
    TabNet High X predictor factory function
    
    Args:
        model_path: Model dosyası yolu
        scaler_path: Scaler dosyası yolu
        
    Returns:
        TabNetHighXPredictor instance
    """
    return TabNetHighXPredictor(
        model_path=model_path,
        scaler_path=scaler_path
    )
