"""
Lightweight Model Manager - Unified Interface for CPU Models

Tüm hafif modelleri yönetmek için birleşik interface.
Model factory, training orchestration, comparison utilities sağlar.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Model imports
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
    
    Özellikler:
    - Model registry ve factory
    - Training orchestration
    - Model comparison utilities
    - Performance metrics collection
    - Model persistence
    - Ensemble creation
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
        """
        Model instance oluşturma
        
        Args:
            model_type: Model tipi ('lightgbm', 'tabnet', vb.)
            mode: Model modu ('classification', 'regression', 'multiclass')
            config: Model konfigürasyonu
            model_id: Benzersiz model ID
            
        Returns:
            Model instance
        """
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
        
        if model_type == 'lightgbm':
            model = model_class(
                model_path=f"{self.models_dir}/{model_id}_model.txt",
                scaler_path=f"{self.models_dir}/{model_id}_scaler.pkl",
                mode=mode
            )
        elif model_type == 'tabnet':
            model = model_class(
                model_path=f"{self.models_dir}/{model_id}_model",
                scaler_path=f"{self.models_dir}/{model_id}_scaler.pkl"
            )
        elif model_type == 'autogluon':
            model = model_class(
                model_path=f"{self.models_dir}/{model_id}_model",
                scaler_path=f"{self.models_dir}/{model_id}_scaler.pkl"
            )
        elif model_type == 'catboost':
            model = model_class(
                model_path=f"{self.models_dir}/{model_id}_model",
                scaler_path=f"{self.models_dir}/{model_id}_scaler.pkl"
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
        """
        Model eğitimi
        
        Args:
            model_id: Model ID
            X: Feature matrix
            y: Target values
            training_config: Eğitim konfigürasyonu
            
        Returns:
            Training metrics
        """
        if model_id not in self.model_registry:
            raise ValueError(f"Model bulunamadı: {model_id}")
        
        model_info = self.model_registry[model_id]
        model_type = model_info['type']
        
        # Model instance oluştur
        model, _ = self.create_model(
            model_type=model_type,
            mode=model_info['mode'],
            config=model_info['config'],
            model_id=model_id
        )
        
        # Eğitim konfigürasyonu
        if training_config is None:
            training_config = {}
        
        # Model eğitimi
        logger.info(f"Model eğitimi başlıyor: {model_id}")
        
        try:
            metrics = model.train(X, y, **training_config)
            
            # Model kaydet
            model.save_model()
            
            # Registry güncelle
            self.model_registry[model_id].update({
                'status': 'trained',
                'trained_at': datetime.now().isoformat(),
                'metrics': metrics
            })
            
            # Trained models'e ekle
            self.trained_models[model_id] = model
            
            logger.info(f"Model eğitimi tamamlandı: {model_id}")
            return metrics
            
        except Exception as e:
            logger.error(f"Model eğitimi başarısız: {model_id} - {str(e)}")
            self.model_registry[model_id]['status'] = 'failed'
            raise
    
    def load_trained_model(self, model_id: str) -> Any:
        """
        Eğitilmiş model yükleme
        
        Args:
            model_id: Model ID
            
        Returns:
            Model instance
        """
        if model_id not in self.model_registry:
            raise ValueError(f"Model bulunamadı: {model_id}")
        
        model_info = self.model_registry[model_id]
        
        if model_info['status'] != 'trained':
            raise ValueError(f"Model henüz eğitilmemiş: {model_id}")
        
        # Model instance oluştur
        model, _ = self.create_model(
            model_type=model_info['type'],
            mode=model_info['mode'],
            config=model_info['config'],
            model_id=model_id
        )
        
        # Model yükle
        model.load_model()
        
        # Trained models'e ekle
        self.trained_models[model_id] = model
        
        logger.info(f"Model yüklendi: {model_id}")
        return model
    
    def compare_models(
        self,
        model_ids: List[str],
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Modelleri karşılaştırma
        
        Args:
            model_ids: Karşılaştırılacak model ID'leri
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Comparison DataFrame
        """
        results = []
        
        for model_id in model_ids:
            try:
                model = self.load_trained_model(model_id)
                model_info = self.model_registry[model_id]
                
                # Tahminler
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)
                    y_pred = model.predict(X_test)
                else:
                    y_pred = model.predict(X_test)
                    y_pred_proba = None
                
                # Metrics hesapla
                metrics = self._calculate_comparison_metrics(
                    y_test, y_pred, y_pred_proba, model_info['mode']
                )
                
                results.append({
                    'model_id': model_id,
                    'model_type': model_info['type'],
                    'mode': model_info['mode'],
                    **metrics
                })
                
            except Exception as e:
                logger.error(f"Model karşılaştırma hatası: {model_id} - {str(e)}")
                results.append({
                    'model_id': model_id,
                    'model_type': model_info['type'],
                    'mode': model_info['mode'],
                    'error': str(e)
                })
        
        return pd.DataFrame(results)
    
    def create_ensemble(
        self,
        model_ids: List[str],
        strategy: str = 'voting',
        weights: Optional[List[float]] = None
    ) -> Dict:
        """
        Ensemble oluşturma
        
        Args:
            model_ids: Ensemble'e dahil edilecek model ID'leri
            strategy: Ensemble stratejisi ('voting', 'stacking')
            weights: Model ağırlıkları (voting için)
            
        Returns:
            Ensemble configuration
        """
        if len(model_ids) < 2:
            raise ValueError("En az 2 model gerekli")
        
        # Modelleri yükle
        models = []
        for model_id in model_ids:
            model = self.load_trained_model(model_id)
            models.append(model)
        
        # Ensemble ID oluştur
        ensemble_id = f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Ensemble config
        ensemble_config = {
            'ensemble_id': ensemble_id,
            'strategy': strategy,
            'model_ids': model_ids,
            'weights': weights,
            'created_at': datetime.now().isoformat()
        }
        
        # Registry'ye kaydet
        self.model_registry[ensemble_id] = {
            'type': 'ensemble',
            'mode': 'ensemble',
            'config': ensemble_config,
            'status': 'created'
        }
        
        logger.info(f"Ensemble oluşturuldu: {ensemble_id}")
        return ensemble_config
    
    def get_model_list(self) -> pd.DataFrame:
        """Model listesini döndür"""
        models = []
        for model_id, info in self.model_registry.items():
            models.append({
                'model_id': model_id,
                'type': info['type'],
                'mode': info['mode'],
                'status': info['status'],
                'created_at': info['created_at'],
                'trained_at': info.get('trained_at', ''),
                'metrics': info.get('metrics', {})
            })
        
        return pd.DataFrame(models)
    
    def delete_model(self, model_id: str):
        """Model silme"""
        if model_id not in self.model_registry:
            raise ValueError(f"Model bulunamadı: {model_id}")
        
        # Model dosyalarını sil
        model_info = self.model_registry[model_id]
        model_type = model_info['type']
        
        if model_type != 'ensemble':
            # Model dosyalarını bul ve sil
            for ext in ['_model.txt', '_model', '_scaler.pkl', '_info.json']:
                file_path = f"{self.models_dir}/{model_id}{ext}"
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        # Registry'den sil
        del self.model_registry[model_id]
        
        # Trained models'den sil
        if model_id in self.trained_models:
            del self.trained_models[model_id]
        
        logger.info(f"Model silindi: {model_id}")
    
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
            
            # Virtual bankroll simulation
            if mode == 'classification':
                metrics['virtual_bankroll_roi'] = self._simulate_virtual_bankroll(y_true, y_pred)
        
        elif mode == 'regression':
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_true, y_pred)
        
        return metrics
    
    def _simulate_virtual_bankroll(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Virtual bankroll simulation"""
        initial = 10000
        wallet = initial
        
        for pred, actual in zip(y_pred, y_true):
            if pred == 1:  # Model 1.5 üstü dedi
                wallet -= 10
                if actual >= 1.5:
                    wallet += 15
        
        return ((wallet - initial) / initial) * 100
    
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

