"""
All Models Predictor - Tüm 5 modelin tahminlerini birleştiren sistem

Bu modül şu modelleri yönetir:
1. Progressive NN (Multi-Scale)
2. CatBoost Ensemble
3. AutoGluon AutoML
4. TabNet High-X Specialist
5. Consensus (Tüm modellerin birleşimi)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
import os

# Model imports
try:
    import tensorflow as tf
    from tensorflow.keras import models
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False

from category_definitions import FeatureEngineering, CategoryDefinitions
import joblib

logger = logging.getLogger(__name__)


class AllModelsPredictor:
    """Tüm 5 modeli yönetir ve tahminlerini birleştirir"""
    
    def __init__(self):
        """Initialize all models"""
        self.models = {}
        self.available_models = []
        
        # Model yolları
        self.paths = {
            'progressive_nn': 'models/progressive_multiscale',
            'catboost': 'models/catboost_multiscale',
            'autogluon': 'models/autogluon_model',
            'tabnet': 'models/tabnet_high_x.pkl'
        }
        
        # Window sizes for multi-scale models
        self.window_sizes = [500, 250, 100, 50, 20]
        
        logger.info("AllModelsPredictor başlatıldı")
    
    def load_progressive_nn(self) -> bool:
        """Progressive NN modellerini yükle"""
        if not TF_AVAILABLE:
            logger.warning("TensorFlow yüklü değil, Progressive NN atlanıyor")
            return False
        
        try:
            nn_models = {}
            nn_scalers = {}
            
            for window_size in self.window_sizes:
                model_path = os.path.join(self.paths['progressive_nn'], f'model_window_{window_size}.h5')
                scaler_path = os.path.join(self.paths['progressive_nn'], f'scaler_window_{window_size}.pkl')
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    nn_models[window_size] = models.load_model(model_path, compile=False)
                    nn_scalers[window_size] = joblib.load(scaler_path)
            
            if nn_models:
                self.models['progressive_nn'] = {
                    'models': nn_models,
                    'scalers': nn_scalers,
                    'loaded_windows': list(nn_models.keys())
                }
                self.available_models.append('progressive_nn')
                logger.info(f"✅ Progressive NN yüklendi ({len(nn_models)} window)")
                return True
            
        except Exception as e:
            logger.error(f"Progressive NN yükleme hatası: {e}")
        
        return False
    
    def load_catboost(self) -> bool:
        """CatBoost modellerini yükle"""
        if not CATBOOST_AVAILABLE:
            logger.warning("CatBoost yüklü değil, atlanıyor")
            return False
        
        try:
            cb_regressors = {}
            cb_classifiers = {}
            cb_scalers = {}
            
            for window_size in self.window_sizes:
                reg_path = os.path.join(self.paths['catboost'], f'regressor_window_{window_size}.cbm')
                cls_path = os.path.join(self.paths['catboost'], f'classifier_window_{window_size}.cbm')
                scaler_path = os.path.join(self.paths['catboost'], f'scaler_window_{window_size}.pkl')
                
                if os.path.exists(reg_path) and os.path.exists(cls_path) and os.path.exists(scaler_path):
                    regressor = CatBoostRegressor()
                    regressor.load_model(reg_path)
                    cb_regressors[window_size] = regressor
                    
                    classifier = CatBoostClassifier()
                    classifier.load_model(cls_path)
                    cb_classifiers[window_size] = classifier
                    
                    cb_scalers[window_size] = joblib.load(scaler_path)
            
            if cb_regressors:
                self.models['catboost'] = {
                    'regressors': cb_regressors,
                    'classifiers': cb_classifiers,
                    'scalers': cb_scalers,
                    'loaded_windows': list(cb_regressors.keys())
                }
                self.available_models.append('catboost')
                logger.info(f"✅ CatBoost yüklendi ({len(cb_regressors)} window)")
                return True
                
        except Exception as e:
            logger.error(f"CatBoost yükleme hatası: {e}")
        
        return False
    
    def load_autogluon(self) -> bool:
        """AutoGluon modelini yükle"""
        if not AUTOGLUON_AVAILABLE:
            logger.warning("AutoGluon yüklü değil, atlanıyor")
            return False
        
        try:
            if os.path.exists(self.paths['autogluon']):
                predictor = TabularPredictor.load(self.paths['autogluon'])
                self.models['autogluon'] = {
                    'predictor': predictor
                }
                self.available_models.append('autogluon')
                logger.info("✅ AutoGluon yüklendi")
                return True
        except Exception as e:
            logger.error(f"AutoGluon yükleme hatası: {e}")
        
        return False
    
    def load_tabnet(self) -> bool:
        """TabNet modelini yükle"""
        if not TABNET_AVAILABLE:
            logger.warning("TabNet yüklü değil, atlanıyor")
            return False
        
        try:
            model_path = self.paths['tabnet']
            scaler_path = 'models/tabnet_scaler.pkl'
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                import pickle
                with open(model_path, 'rb') as f:
                    tabnet_model = pickle.load(f)
                
                scaler = joblib.load(scaler_path)
                
                self.models['tabnet'] = {
                    'model': tabnet_model,
                    'scaler': scaler
                }
                self.available_models.append('tabnet')
                logger.info("✅ TabNet yüklendi")
                return True
        except Exception as e:
            logger.error(f"TabNet yükleme hatası: {e}")
        
        return False
    
    def load_all_models(self) -> Dict[str, bool]:
        """Tüm modelleri yüklemeyi dene"""
        results = {
            'progressive_nn': self.load_progressive_nn(),
            'catboost': self.load_catboost(),
            'autogluon': self.load_autogluon(),
            'tabnet': self.load_tabnet()
        }
        
        logger.info(f"Yüklenen modeller: {self.available_models}")
        return results
    
    def predict_progressive_nn(self, history: np.ndarray) -> Optional[Dict]:
        """Progressive NN tahmini"""
        if 'progressive_nn' not in self.models:
            return None
        
        try:
            model_data = self.models['progressive_nn']
            ensemble_reg = []
            ensemble_thr = []
            
            for window_size in model_data['loaded_windows']:
                # Feature extraction
                feats = FeatureEngineering.extract_all_features(history.tolist())
                features = np.array(list(feats.values())).reshape(1, -1)
                features = model_data['scalers'][window_size].transform(features)
                
                # Sequence
                sequence = history[-window_size:].reshape(1, window_size, 1)
                sequence = np.log10(sequence + 1e-8)
                
                # Predict
                pred = model_data['models'][window_size].predict([features, sequence], verbose=0)
                ensemble_reg.append(pred[0][0][0])
                ensemble_thr.append(pred[2][0][0])
            
            avg_reg = np.mean(ensemble_reg)
            avg_thr = np.mean(ensemble_thr)
            
            return {
                'prediction': float(avg_reg),
                'threshold_prob': float(avg_thr),
                'above_threshold': avg_thr >= 0.5,
                'confidence': float(abs(avg_thr - 0.5) * 2),  # 0-1 arası normalize
                'category': CategoryDefinitions.get_category_name(avg_reg)
            }
        except Exception as e:
            logger.error(f"Progressive NN tahmin hatası: {e}")
            return None
    
    def predict_catboost(self, history: np.ndarray) -> Optional[Dict]:
        """CatBoost tahmini"""
        if 'catboost' not in self.models:
            return None
        
        try:
            model_data = self.models['catboost']
            ensemble_reg = []
            ensemble_cls = []
            
            for window_size in model_data['loaded_windows']:
                # Feature extraction
                feats = FeatureEngineering.extract_all_features(history.tolist())
                features = np.array(list(feats.values())).reshape(1, -1)
                features = model_data['scalers'][window_size].transform(features)
                
                # Predictions
                p_reg = model_data['regressors'][window_size].predict(features)[0]
                p_cls = model_data['classifiers'][window_size].predict(features)[0]
                
                ensemble_reg.append(p_reg)
                ensemble_cls.append(p_cls)
            
            avg_reg = np.mean(ensemble_reg)
            avg_cls = np.round(np.mean(ensemble_cls)).astype(int)
            
            return {
                'prediction': float(avg_reg),
                'threshold_prob': float(avg_cls),
                'above_threshold': avg_cls == 1,
                'confidence': 0.75,  # CatBoost genelde güvenilir
                'category': CategoryDefinitions.get_category_name(avg_reg)
            }
        except Exception as e:
            logger.error(f"CatBoost tahmin hatası: {e}")
            return None
    
    def predict_autogluon(self, history: np.ndarray) -> Optional[Dict]:
        """AutoGluon tahmini"""
        if 'autogluon' not in self.models:
            return None
        
        try:
            # Feature extraction
            feats = FeatureEngineering.extract_all_features(history.tolist())
            X = pd.DataFrame([feats])
            
            # Predict
            pred = self.models['autogluon']['predictor'].predict(X)[0]
            pred_proba = self.models['autogluon']['predictor'].predict_proba(X)
            
            confidence = float(pred_proba.iloc[0].max())
            
            return {
                'prediction': 1.8 if pred == 1 else 1.2,  # Basitleştirilmiş
                'threshold_prob': float(pred),
                'above_threshold': pred == 1,
                'confidence': confidence,
                'category': 'Orta' if pred == 1 else 'Düşük'
            }
        except Exception as e:
            logger.error(f"AutoGluon tahmin hatası: {e}")
            return None
    
    def predict_tabnet(self, history: np.ndarray) -> Optional[Dict]:
        """TabNet tahmini"""
        if 'tabnet' not in self.models:
            return None
        
        try:
            # Feature extraction
            feats = FeatureEngineering.extract_all_features(history.tolist())
            features = np.array(list(feats.values())).reshape(1, -1)
            features = self.models['tabnet']['scaler'].transform(features)
            
            # Predict
            pred = self.models['tabnet']['model'].predict(features)[0]
            
            # 0: Düşük, 1: Orta, 2: Yüksek, 3: Mega
            category_map = {0: 'Düşük', 1: 'Orta', 2: 'Yüksek', 3: 'Mega'}
            value_map = {0: 1.2, 1: 1.8, 2: 5.0, 3: 20.0}
            
            return {
                'prediction': float(value_map.get(pred, 1.5)),
                'threshold_prob': 1.0 if pred >= 1 else 0.0,
                'above_threshold': pred >= 1,
                'confidence': 0.70,
                'category': category_map.get(pred, 'Orta')
            }
        except Exception as e:
            logger.error(f"TabNet tahmin hatası: {e}")
            return None
    
    def predict_all(self, history: np.ndarray) -> Dict:
        """Tüm modellerden tahmin al"""
        predictions = {}
        
        # Her modelden tahmin al
        if 'progressive_nn' in self.available_models:
            predictions['progressive_nn'] = self.predict_progressive_nn(history)
        
        if 'catboost' in self.available_models:
            predictions['catboost'] = self.predict_catboost(history)
        
        if 'autogluon' in self.available_models:
            predictions['autogluon'] = self.predict_autogluon(history)
        
        if 'tabnet' in self.available_models:
            predictions['tabnet'] = self.predict_tabnet(history)
        
        # Consensus hesapla
        valid_predictions = {k: v for k, v in predictions.items() if v is not None}
        
        if len(valid_predictions) >= 2:
            # Weighted consensus
            weights = {
                'progressive_nn': 0.30,
                'catboost': 0.35,
                'autogluon': 0.20,
                'tabnet': 0.15
            }
            
            total_weight = sum(weights[k] for k in valid_predictions.keys())
            
            weighted_pred = sum(
                valid_predictions[k]['prediction'] * weights.get(k, 0.25)
                for k in valid_predictions.keys()
            ) / total_weight
            
            above_count = sum(1 for v in valid_predictions.values() if v['above_threshold'])
            consensus_above = above_count >= len(valid_predictions) / 2
            
            avg_confidence = np.mean([v['confidence'] for v in valid_predictions.values()])
            
            predictions['consensus'] = {
                'prediction': float(weighted_pred),
                'threshold_prob': float(above_count / len(valid_predictions)),
                'above_threshold': consensus_above,
                'confidence': float(avg_confidence),
                'category': CategoryDefinitions.get_category_name(weighted_pred),
                'agreement': float(above_count / len(valid_predictions)),
                'models_agreed': above_count,
                'total_models': len(valid_predictions)
            }
        else:
            predictions['consensus'] = None
        
        return predictions
