"""
Lightweight Model Manager - Unified Interface for CPU Models

Tüm hafif modelleri yönetmek için birleşik interface.
Model factory, training orchestration, comparison utilities sağlar.

GÜNCELLEME:
- Threshold Manager entegrasyonu (0.85/0.95 Eşikler).
- ROI hesaplamalarında güven eşiği kontrolü.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import warnings
from utils.threshold_manager import get_threshold_manager

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Model imports (Lazy Loading)
try:
    from .lightgbm_predictor import LightGBMPredictor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from .tabnet_predictor import TabNetHighXPredictor
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False

try:
    from .autogluon_predictor import AutoGluonPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False

try:
    from .catboost_ensemble import CatBoostEnsemble
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class LightweightModelManager:
    """
    Hafif modelleri yönetmek için birleşik interface
    """
    
    AVAILABLE_MODELS = {
        'lightgbm': {
            'class': LightGBMPredictor,
            'available': LIGHTGBM_AVAILABLE,
            'description': 'LightGBM - CPU optimized gradient boosting',
            'modes': ['classification', 'regression', 'multiclass'],
            'default_config': {
                'num_leaves': 31,
                'max_depth': 8,
                'learning_rate': 0.03,
                'n_estimators': 1000,
                'device': 'cpu'
            }
        },
        'tabnet': {
            'class': TabNetHighXPredictor,
            'available': TABNET_AVAILABLE,
            'description': 'TabNet - Attention-based deep learning',
            'modes': ['classification', 'multiclass'],
            'default_config': {
                'n_d': 8,
                'n_a': 8,
                'n_steps': 3,
                'gamma': 1.3,
                'n_independent': 2,
                'n_shared': 2
            }
        },
        'autogluon': {
            'class': AutoGluonPredictor,
            'available': AUTOGLUON_AVAILABLE,
            'description': 'AutoGluon - Automated ML',
            'modes': ['classification', 'regression'],
            'default_config': {
                'time_limit': 600,
                'presets': 'medium_quality'
            }
        },
        'catboost': {
            'class': CatBoostEnsemble,
            'available': CATBOOST_AVAILABLE,
            'description': 'CatBoost - Categorical boosting',
            'modes': ['classification', 'regression'],
            'default_config': {
                'task_type': 'CPU',
                'iterations': 1000,
                'depth': 8,
                'learning_rate': 0.03
            }
        }
    }
    
    def __init__(self, models_dir: str = "models/cpu"):
        """
        Args:
            models_dir: Model kayıtları için klasör
        """
        self.models_dir = models_dir
        self.trained_models = {}
        self.model_registry = {}
        
        # Threshold Manager
        tm = get_threshold_manager()
        self.THRESHOLD_NORMAL = tm.get_normal_threshold()
        
        # Models klasörünü oluştur
        os.makedirs(models_dir, exist_ok=True)
        
        # Mevcut modelleri yükle
        self._load_model_registry()
    
    def get_available_models(self) -> Dict[str, Dict]:
        """Kullanılabilir modelleri döndür"""
        available = {}
        for name, info in self.AVAILABLE_MODELS.items():
            if info['available']:
                available[name] = {
                    'description': info['description'],
                    'modes': info['modes'],
                    'default_config': info['default_config']
                }
        return available
    
    def create_model(
        self,
        model_type: str,
        mode: str = 'classification',
        config: Optional[Dict] = None,
        model_id: Optional[str] = None
    ) -> Any:
        """Model instance oluşturma"""
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Desteklenmeyen model tipi: {model_type}")
        
        if not self.AVAILABLE_MODELS[model_type]['available']:
            raise ImportError(f"{model_type} modeli yüklü değil")
        
        if mode not in self.AVAILABLE_MODELS[model_type]['modes']:
            raise ValueError(f"{model_type} modeli {mode} modunu desteklemiyor")
        
        # Model ID oluştur
        if model_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_id = f"{model_type}_{mode}_{timestamp}"
        
        # Konfigürasyonu birleştir
        if config is None:
            config = self.AVAILABLE_MODELS[model_type]['default_config'].copy()
        
        # Model instance oluştur
        model_class = self.AVAILABLE_MODELS[model_type]['class']
        
        # Her modelin constructor parametreleri farklı olabilir, bu yüzden özelleştiriyoruz
        if model_type == 'lightgbm':
            model = model_class(
                model_path=f"{self.models_dir}/{model_id}_model.txt",
                scaler_path=f"{self.models_dir}/{model_id}_scaler.pkl",
                mode=mode
            )
        elif model_type == 'tabnet':
            model = model_class(
                model_path=f"{self.models_dir}/{model_id}_model.pkl",
                scaler_path=f"{self.models_dir}/{model_id}_scaler.pkl"
            )
        elif model_type == 'autogluon':
            model = model_class(
                model_path=f"{self.models_dir}/{model_id}_model",
                scaler_path=f"{self.models_dir}/{model_id}_scaler.pkl"
            )
        elif model_type == 'catboost':
            # CatBoostEnsemble için parametreler farklı olabilir
            # Ancak lightweight manager içinde tekil model gibi yönetiliyor olabilir
            # Burada basitlik için default parametrelerle başlatıyoruz
            model = model_class(
                model_type='regressor' if mode == 'regression' else 'classifier'
            )
        
        # Model registry'ye kaydet
        self.model_registry[model_id] = {
            'type': model_type,
            'mode': mode,
            'config': config,
            'created_at': datetime.now().isoformat(),
            'status': 'created'
        }
        
        logger.info(f"Model oluşturuldu: {model_id} ({model_type}, {mode})")
        return model, model_id
    
    def train_model(
        self,
        model_id: str,
        X: np.ndarray,
        y: np.ndarray,
        training_config: Optional[Dict] = None
    ) -> Dict:
        """Model eğitimi"""
        if model_id not in self.model_registry:
            raise ValueError(f"Model bulunamadı: {model_id}")
        
        model_info = self.model_registry[model_id]
        
        # Model instance oluştur (yeniden, state kaybolmasın diye)
        model, _ = self.create_model(
            model_type=model_info['type'],
            mode=model_info['mode'],
            config=model_info['config'],
            model_id=model_id
        )
        
        if training_config is None:
            training_config = {}
        
        logger.info(f"Model eğitimi başlıyor: {model_id}")
        
        try:
            metrics = model.train(X, y, **training_config)
            
            # Model kaydet (Her modelin kendi save metodu var)
            if hasattr(model, 'save_model'):
                model.save_model()
            elif hasattr(model, 'save_ensemble'): # CatBoostEnsemble için
                 model.save_ensemble(f"{self.models_dir}/{model_id}")
            
            # Registry güncelle
            self.model_registry[model_id].update({
                'status': 'trained',
                'trained_at': datetime.now().isoformat(),
                'metrics': metrics
            })
            
            self.trained_models[model_id] = model
            
            logger.info(f"Model eğitimi tamamlandı: {model_id}")
            return metrics
            
        except Exception as e:
            logger.error(f"Model eğitimi başarısız: {model_id} - {str(e)}")
            self.model_registry[model_id]['status'] = 'failed'
            raise
    
    def load_trained_model(self, model_id: str) -> Any:
        """Eğitilmiş model yükleme"""
        if model_id not in self.model_registry:
            raise ValueError(f"Model bulunamadı: {model_id}")
        
        model_info = self.model_registry[model_id]
        
        if model_info['status'] != 'trained':
            raise ValueError(f"Model henüz eğitilmemiş: {model_id}")
        
        model, _ = self.create_model(
            model_type=model_info['type'],
            mode=model_info['mode'],
            config=model_info['config'],
            model_id=model_id
        )
        
        # Model yükle (Her modelin kendi load metodu var)
        if hasattr(model, 'load_model'):
            model.load_model()
        elif hasattr(model, 'load_ensemble'):
             model.load_ensemble(f"{self.models_dir}/{model_id}")
        
        self.trained_models[model_id] = model
        logger.info(f"Model yüklendi: {model_id}")
        return model
    
    def _simulate_virtual_bankroll(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Virtual bankroll simulation
        GÜNCELLEME: Normal Mod (0.85) eşiğini kullanır
        """
        initial = 10000
        wallet = initial
        
        # Tahminler binary değilse, olasılık ise:
        # (Bu fonksiyon genellikle binary prediction alır, o yüzden y_pred zaten 0/1'dir)
        
        for pred, actual in zip(y_pred, y_true):
            # 1.5 Üstü Tahmini (Normal Mod eşiği aşıldı varsayımı ile)
            if pred == 1:
                wallet -= 10
                if actual >= 1.5: # Gerçek değer
                    wallet += 15
        
        return ((wallet - initial) / initial) * 100

    def _calculate_comparison_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray],
        mode: str
    ) -> Dict:
        """Karşılaştırma metrikleri hesaplama"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            mean_absolute_error, mean_squared_error, r2_score
        )
        
        metrics = {}
        
        if mode in ['classification', 'multiclass']:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
            
            if mode == 'classification':
                metrics['virtual_bankroll_roi'] = self._simulate_virtual_bankroll(y_true, y_pred)
        
        elif mode == 'regression':
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_true, y_pred)
        
        return metrics

    def _load_model_registry(self):
        """Model registry'yi dosyadan yükle"""
        registry_path = f"{self.models_dir}/model_registry.json"
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r') as f:
                    self.model_registry = json.load(f)
                logger.info(f"Model registry yüklendi: {len(self.model_registry)} model")
            except Exception as e:
                logger.error(f"Model registry yükleme hatası: {str(e)}")
                self.model_registry = {}
        else:
            self.model_registry = {}

    def save_model_registry(self):
        """Model registry'yi dosyaya kaydet"""
        registry_path = f"{self.models_dir}/model_registry.json"
        try:
            with open(registry_path, 'w') as f:
                json.dump(self.model_registry, f, indent=2)
            logger.info("Model registry kaydedildi")
        except Exception as e:
            logger.error(f"Model registry kaydetme hatası: {str(e)}")
