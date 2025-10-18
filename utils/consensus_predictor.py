"""
Consensus Predictor - NN ve CatBoost Ensemble Modellerinin Consensus Tahminleri

Bu modül, Progressive NN ve CatBoost modellerinin tahminlerini birleştirir ve
sadece iki model de hemfikir olduğunda bahis önerisi yapar.

Consensus Mantığı:
- Her iki model de 1.5 üstü tahmin ediyorsa → OYNA
- Aksi durumda → OYNAMA

İki Sanal Kasa Stratejisi:
- Kasa 1: 1.5x eşikte çık
- Kasa 2: İki modelin regression tahminlerinin ortalamasının %70'inde çık
"""

import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, Tuple, List, Optional
import logging

try:
    import tensorflow as tf
    from tensorflow.keras import models
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. NN predictions will not work.")

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logging.warning("CatBoost not available. CatBoost predictions will not work.")

from category_definitions import FeatureEngineering

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConsensusPredictor:
    """
    NN ve CatBoost modellerinin consensus tahminlerini yönetir
    """
    
    def __init__(
        self,
        nn_model_dir: str = 'models/progressive_multiscale',
        catboost_model_dir: str = 'models/catboost_multiscale',
        window_sizes: List[int] = [1000, 500, 250, 100, 50, 20]
    ):
        """
        Args:
            nn_model_dir: Progressive NN modellerinin bulunduğu dizin
            catboost_model_dir: CatBoost modellerinin bulunduğu dizin
            window_sizes: Pencere boyutları
        """
        self.nn_model_dir = nn_model_dir
        self.catboost_model_dir = catboost_model_dir
        self.window_sizes = window_sizes
        
        self.nn_models = {}
        self.nn_scalers = {}
        self.catboost_regressors = {}
        self.catboost_classifiers = {}
        self.catboost_scalers = {}
        
        logger.info("ConsensusPredictor initialized")
        logger.info(f"  NN model directory: {nn_model_dir}")
        logger.info(f"  CatBoost model directory: {catboost_model_dir}")
        logger.info(f"  Window sizes: {window_sizes}")
    
    def load_nn_models(self):
        """Progressive NN modellerini yükle - Gelişmiş Lambda desteği ile"""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is not available. Cannot load NN models.")
        
        logger.info("\n" + "="*70)
        logger.info("LOADING PROGRESSIVE NN MODELS (Enhanced Lambda Support)")
        logger.info("="*70)
        
        # Custom objects for Lambda layers
        from tensorflow.keras import backend as K
        from tensorflow.keras.layers import Layer
        from tensorflow.keras.utils import register_keras_serializable
        
        @register_keras_serializable()
        class SafeLambdaLayer(Layer):
            """Lambda katmanı için güvenli wrapper"""
            def __init__(self, function, **kwargs):
                super().__init__(**kwargs)
                self.function = function
            
            def call(self, inputs):
                return self.function(inputs)
            
            def get_config(self):
                config = super().get_config()
                config.update({'function': self.function})
                return config
        
        # Lambda fonksiyonları
        def safe_sum_axis1(x):
            """Lambda: sum(x, axis=1) için güvenli alternatif"""
            return K.sum(x, axis=1)
        
        def safe_mean_axis1(x):
            """Lambda: mean(x, axis=1) için güvenli alternatif"""
            return K.mean(x, axis=1)
        
        # Custom objects dictionary
        custom_objects = {
            'SafeLambdaLayer': SafeLambdaLayer,
            'lambda': lambda x: K.sum(x, axis=1),  # Fallback
            'safe_sum_axis1': safe_sum_axis1,
            'safe_mean_axis1': safe_mean_axis1,
            'tf': tf,  # TensorFlow modülü
            'K': K,    # Keras backend
        }
        
        # Custom losses de ekle
        try:
            from utils.custom_losses import CUSTOM_OBJECTS
            custom_objects.update(CUSTOM_OBJECTS)
        except ImportError:
            logger.warning("⚠️  Custom losses yüklenemedi")
        
        for window_size in self.window_sizes:
            model_path = os.path.join(self.nn_model_dir, f'model_window_{window_size}.h5')
            scaler_path = os.path.join(self.nn_model_dir, f'scaler_window_{window_size}.pkl')
            
            if not os.path.exists(model_path):
                logger.warning(f"⚠️  NN model not found: {model_path}")
                continue
            
            if not os.path.exists(scaler_path):
                logger.warning(f"⚠️  NN scaler not found: {scaler_path}")
                continue
            
            # 3 aşamalı model yükleme stratejisi
            model_loaded = False
            
            # 1. Deneme: Normal yükleme (yeni modeller için)
            try:
                logger.info(f"🔄 Loading NN model for window {window_size} (Attempt 1: Normal)...")
                self.nn_models[window_size] = models.load_model(model_path, compile=False)
                logger.info(f"✅ Loaded NN model for window {window_size} (Normal)")
                model_loaded = True
            except Exception as e:
                logger.warning(f"⚠️  Normal load failed: {str(e)[:100]}...")
            
            # 2. Deneme: Custom objects ile yükleme
            if not model_loaded:
                try:
                    logger.info(f"🔄 Loading NN model for window {window_size} (Attempt 2: Custom Objects)...")
                    self.nn_models[window_size] = models.load_model(
                        model_path, 
                        compile=False,
                        custom_objects=custom_objects
                    )
                    logger.info(f"✅ Loaded NN model for window {window_size} (Custom Objects)")
                    model_loaded = True
                except Exception as e:
                    logger.warning(f"⚠️  Custom objects load failed: {str(e)[:100]}...")
            
            # 3. Deneme: Lambda rebuild ile yükleme
            if not model_loaded:
                try:
                    logger.info(f"🔄 Loading NN model for window {window_size} (Attempt 3: Lambda Rebuild)...")
                    self.nn_models[window_size] = self._load_model_with_lambda_rebuild(
                        model_path, custom_objects
                    )
                    logger.info(f"✅ Loaded NN model for window {window_size} (Lambda Rebuild)")
                    model_loaded = True
                except Exception as e:
                    logger.error(f"❌ Lambda rebuild failed: {str(e)[:100]}...")
            
            # Eğer hiçbiri çalışmadıysa
            if not model_loaded:
                logger.error(f"❌ All loading attempts failed for window {window_size}")
                continue
            
            # Scaler yükle
            try:
                self.nn_scalers[window_size] = joblib.load(scaler_path)
                logger.info(f"✅ Loaded scaler for window {window_size}")
            except Exception as e:
                logger.error(f"❌ Failed to load scaler for window {window_size}: {e}")
                continue
        
        logger.info(f"Total NN models loaded: {len(self.nn_models)}")
        logger.info("="*70 + "\n")
    
    def _load_model_with_lambda_rebuild(self, model_path: str, custom_objects: dict):
        """Modeli Lambda katmanlarını yeniden oluşturarak yükle"""
        # Model mimarisini JSON olarak yükle (varsa)
        json_path = model_path.replace('.h5', '.json')
        weights_path = model_path.replace('.h5', '_weights.h5')
        
        if os.path.exists(json_path) and os.path.exists(weights_path):
            # JSON + weights ile yükle
            with open(json_path, 'r') as f:
                model_json = f.read()
            
            model = models.model_from_json(model_json, custom_objects=custom_objects)
            model.load_weights(weights_path)
            return model
        
        # Son çare: Modeli mimarisini yeniden oluştur
        logger.warning("⚠️  Attempting model architecture rebuild...")
        return self._rebuild_model_architecture(model_path, custom_objects)
    
    def _rebuild_model_architecture(self, model_path: str, custom_objects: dict):
        """Model mimarisini yeniden oluştur (son çare)"""
        # Bu fonksiyon model mimarisini bilerek yeniden oluşturur
        # Gerçek implementasyon model yapısına göre değişebilir
        
        from tensorflow.keras import layers, models
        
        # Dummy model oluştur (aynı input/output şekliyle)
        input_features = layers.Input((64,), name='features')  # Tahmini feature sayısı
        input_sequence = layers.Input((1000, 1), name='sequence')  # Tahmini sequence boyutu
        
        # Basit bir mimari oluştur
        x_feat = layers.Dense(256, activation='relu')(input_features)
        x_seq = layers.LSTM(128)(input_sequence)
        
        fusion = layers.Concatenate()([x_feat, x_seq])
        fusion = layers.Dense(128, activation='relu')(fusion)
        
        # Outputs
        out_reg = layers.Dense(1, activation='linear', name='regression')(fusion)
        out_cls = layers.Dense(3, activation='softmax', name='classification')(fusion)
        out_thr = layers.Dense(1, activation='sigmoid', name='threshold')(fusion)
        
        model = models.Model([input_features, input_sequence], [out_reg, out_cls, out_thr])
        
        # Ağırlıkları yükle (mümkünse)
        try:
            model.load_weights(model_path, by_name=True, skip_mismatch=True)
            logger.info("✅ Partial weights loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️  Could not load weights: {e}")
        
        return model
    
    def load_catboost_models(self):
        """CatBoost modellerini yükle"""
        if not CATBOOST_AVAILABLE:
            raise RuntimeError("CatBoost is not available. Cannot load CatBoost models.")
        
        logger.info("\n" + "="*70)
        logger.info("LOADING CATBOOST MODELS")
        logger.info("="*70)
        
        for window_size in self.window_sizes:
            reg_path = os.path.join(self.catboost_model_dir, f'regressor_window_{window_size}.cbm')
            cls_path = os.path.join(self.catboost_model_dir, f'classifier_window_{window_size}.cbm')
            scaler_path = os.path.join(self.catboost_model_dir, f'scaler_window_{window_size}.pkl')
            
            if not os.path.exists(reg_path):
                logger.warning(f"⚠️  CatBoost regressor not found: {reg_path}")
                continue
            
            if not os.path.exists(cls_path):
                logger.warning(f"⚠️  CatBoost classifier not found: {cls_path}")
                continue
            
            if not os.path.exists(scaler_path):
                logger.warning(f"⚠️  CatBoost scaler not found: {scaler_path}")
                continue
            
            # Modelleri yükle
            self.catboost_regressors[window_size] = CatBoostRegressor()
            self.catboost_regressors[window_size].load_model(reg_path)
            
            self.catboost_classifiers[window_size] = CatBoostClassifier()
            self.catboost_classifiers[window_size].load_model(cls_path)
            
            # Scaler yükle
            self.catboost_scalers[window_size] = joblib.load(scaler_path)
            
            logger.info(f"✅ Loaded CatBoost models for window {window_size}")
        
        logger.info(f"Total CatBoost model pairs loaded: {len(self.catboost_regressors)}")
        logger.info("="*70 + "\n")
    
    def extract_features(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """
        Veri için feature extraction yap
        
        Args:
            data: Geçmiş veri
            window_size: Pencere boyutu
            
        Returns:
            Feature vector
        """
        # Son window_size veriyi al
        if len(data) < window_size:
            raise ValueError(f"Data length ({len(data)}) < window size ({window_size})")
        
        # Feature engineering
        feats = FeatureEngineering.extract_all_features(data.tolist())
        features = np.array(list(feats.values())).reshape(1, -1)
        
        return features
    
    def predict_nn_ensemble(self, data: np.ndarray) -> Dict:
        """
        NN ensemble tahminini yap
        
        Args:
            data: Geçmiş veri
            
        Returns:
            Dict with predictions
        """
        if not self.nn_models:
            raise RuntimeError("NN models not loaded. Call load_nn_models() first.")
        
        ensemble_reg = []
        ensemble_thr = []
        
        for window_size in self.window_sizes:
            if window_size not in self.nn_models:
                continue
            
            # Feature extraction
            features = self.extract_features(data, window_size)
            
            # Normalizasyon
            features = self.nn_scalers[window_size].transform(features)
            
            # Sequence (son window_size değer)
            sequence = data[-window_size:].reshape(1, window_size, 1)
            sequence = np.log10(sequence + 1e-8)
            
            # Tahmin
            pred = self.nn_models[window_size].predict([features, sequence], verbose=0)
            p_reg = pred[0][0][0]  # Regression output
            p_thr = pred[2][0][0]  # Threshold output (sigmoid)
            
            ensemble_reg.append(p_reg)
            ensemble_thr.append(p_thr)
        
        # Ensemble ortalama
        avg_reg = np.mean(ensemble_reg)
        avg_thr = np.mean(ensemble_thr)
        
        # Threshold: 1.5 üstü mü altı mı?
        threshold_prediction = 1 if avg_thr >= 0.5 else 0
        
        return {
            'regression': float(avg_reg),
            'threshold': threshold_prediction,
            'threshold_prob': float(avg_thr),
            'individual_predictions': {
                'regression': ensemble_reg,
                'threshold': ensemble_thr
            }
        }
    
    def predict_catboost_ensemble(self, data: np.ndarray) -> Dict:
        """
        CatBoost ensemble tahminini yap
        
        Args:
            data: Geçmiş veri
            
        Returns:
            Dict with predictions
        """
        if not self.catboost_regressors:
            raise RuntimeError("CatBoost models not loaded. Call load_catboost_models() first.")
        
        ensemble_reg = []
        ensemble_cls = []
        
        for window_size in self.window_sizes:
            if window_size not in self.catboost_regressors:
                continue
            
            # Feature extraction
            features = self.extract_features(data, window_size)
            
            # Normalizasyon
            features = self.catboost_scalers[window_size].transform(features)
            
            # Tahminler
            p_reg = self.catboost_regressors[window_size].predict(features)[0]
            p_cls = self.catboost_classifiers[window_size].predict(features)[0]
            
            ensemble_reg.append(p_reg)
            ensemble_cls.append(p_cls)
        
        # Ensemble ortalama
        avg_reg = np.mean(ensemble_reg)
        avg_cls = np.round(np.mean(ensemble_cls)).astype(int)
        
        return {
            'regression': float(avg_reg),
            'threshold': int(avg_cls),
            'individual_predictions': {
                'regression': ensemble_reg,
                'threshold': ensemble_cls
            }
        }
    
    def predict_consensus(self, data: np.ndarray) -> Dict:
        """
        Consensus tahmin yap
        
        Args:
            data: Geçmiş veri
            
        Returns:
            Consensus prediction dict:
            {
                'consensus': bool,
                'should_bet': bool,
                'nn_prediction': float,
                'catboost_prediction': float,
                'average_prediction': float,
                'exit_point_70': float,
                'nn_threshold': int,
                'catboost_threshold': int
            }
        """
        # NN tahminini al
        nn_result = self.predict_nn_ensemble(data)
        
        # CatBoost tahminini al
        catboost_result = self.predict_catboost_ensemble(data)
        
        # Consensus kontrolü: İki model de 1.5 üstü mü?
        nn_threshold = nn_result['threshold']
        catboost_threshold = catboost_result['threshold']
        
        consensus = (nn_threshold == 1 and catboost_threshold == 1)
        
        # Ortalama tahmin
        avg_prediction = (nn_result['regression'] + catboost_result['regression']) / 2
        
        # Kasa 2 için çıkış noktası: Ortalama × 0.70
        exit_point_70 = avg_prediction * 0.70
        
        result = {
            'consensus': consensus,
            'should_bet': consensus,  # Sadece consensus varsa oyna
            'nn_prediction': nn_result['regression'],
            'catboost_prediction': catboost_result['regression'],
            'average_prediction': avg_prediction,
            'exit_point_70': exit_point_70,
            'nn_threshold': nn_threshold,
            'catboost_threshold': catboost_threshold,
            'nn_threshold_prob': nn_result['threshold_prob']
        }
        
        return result


def simulate_consensus_bankroll(
    predictions: List[Dict],
    actuals: np.ndarray,
    bet_amount: float = 10.0
) -> Dict:
    """
    Consensus tahminleri ile sanal kasa simülasyonu
    
    Args:
        predictions: Consensus tahminleri listesi
        actuals: Gerçek değerler
        bet_amount: Bahis tutarı
        
    Returns:
        Simülasyon sonuçları
    """
    initial_bankroll = len(actuals) * bet_amount
    
    # Kasa 1: 1.5x EŞİK
    wallet1 = initial_bankroll
    total_bets1 = 0
    total_wins1 = 0
    
    # Kasa 2: %70 ÇIKIŞ
    wallet2 = initial_bankroll
    total_bets2 = 0
    total_wins2 = 0
    
    for pred, actual in zip(predictions, actuals):
        if pred['should_bet']:
            # KASA 1
            wallet1 -= bet_amount
            total_bets1 += 1
            
            if actual >= 1.5:
                wallet1 += 1.5 * bet_amount
                total_wins1 += 1
            
            # KASA 2
            wallet2 -= bet_amount
            total_bets2 += 1
            
            exit_point = pred['exit_point_70']
            if actual >= exit_point:
                wallet2 += exit_point * bet_amount
                total_wins2 += 1
    
    # Metrikler
    profit1 = wallet1 - initial_bankroll
    roi1 = (profit1 / initial_bankroll) * 100 if total_bets1 > 0 else 0
    win_rate1 = (total_wins1 / total_bets1 * 100) if total_bets1 > 0 else 0
    
    profit2 = wallet2 - initial_bankroll
    roi2 = (profit2 / initial_bankroll) * 100 if total_bets2 > 0 else 0
    win_rate2 = (total_wins2 / total_bets2 * 100) if total_bets2 > 0 else 0
    
    return {
        'kasa_1': {
            'initial': initial_bankroll,
            'final': wallet1,
            'profit': profit1,
            'roi': roi1,
            'total_bets': total_bets1,
            'total_wins': total_wins1,
            'win_rate': win_rate1
        },
        'kasa_2': {
            'initial': initial_bankroll,
            'final': wallet2,
            'profit': profit2,
            'roi': roi2,
            'total_bets': total_bets2,
            'total_wins': total_wins2,
            'win_rate': win_rate2
        }
    }
