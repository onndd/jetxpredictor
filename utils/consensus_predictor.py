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
        window_sizes: List[int] = [500, 250, 100, 50, 20]
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
        """Progressive NN modellerini yükle"""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is not available. Cannot load NN models.")
        
        logger.info("\n" + "="*70)
        logger.info("LOADING PROGRESSIVE NN MODELS")
        logger.info("="*70)
        
        for window_size in self.window_sizes:
            model_path = os.path.join(self.nn_model_dir, f'model_window_{window_size}.h5')
            scaler_path = os.path.join(self.nn_model_dir, f'scaler_window_{window_size}.pkl')
            
            if not os.path.exists(model_path):
                logger.warning(f"⚠️  NN model not found: {model_path}")
                continue
            
            if not os.path.exists(scaler_path):
                logger.warning(f"⚠️  NN scaler not found: {scaler_path}")
                continue
            
            # Model yükle - Lambda katmanı desteğiyle
            try:
                # İlk önce compile=False ile dene (yeni modeller için)
                self.nn_models[window_size] = models.load_model(model_path, compile=False)
                logger.info(f"✅ Loaded NN model for window {window_size}")
            except Exception as e:
                # Eğer Lambda hatası varsa, custom_objects ile tekrar dene
                logger.warning(f"⚠️  Initial load failed, trying with Lambda support...")
                try:
                    from tensorflow.keras import backend as K
                    custom_objects = {
                        'lambda': lambda x: K.sum(x, axis=1)  # Lambda fallback
                    }
                    self.nn_models[window_size] = models.load_model(
                        model_path, 
                        compile=False,
                        custom_objects=custom_objects
                    )
                    logger.info(f"✅ Loaded NN model for window {window_size} (with Lambda support)")
                except Exception as e2:
                    logger.error(f"❌ Failed to load NN model for window {window_size}: {e2}")
                    continue
            
            # Scaler yükle
            self.nn_scalers[window_size] = joblib.load(scaler_path)
        
        logger.info(f"Total NN models loaded: {len(self.nn_models)}")
        logger.info("="*70 + "\n")
    
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
