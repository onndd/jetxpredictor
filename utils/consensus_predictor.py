"""
Consensus Predictor - NN ve CatBoost Ensemble Modellerinin Consensus Tahminleri

Bu modül, Progressive NN ve CatBoost modellerinin tahminlerini birleştirir.
YENİ 2 MODLU YAPI:
- Normal Mod: Her iki model de ≥ 0.85 güven veriyorsa.
- Rolling Mod: Her iki model de ≥ 0.95 güven veriyorsa.

GÜNCELLEME:
- %85 Normal, %95 Rolling Eşikleri.
- Simülasyonlar bu modlara göre ayrıştırıldı.
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
        
        # KRITIK EŞİKLER
        self.THRESHOLD_NORMAL = 0.85
        self.THRESHOLD_ROLLING = 0.95
        
        logger.info(f"ConsensusPredictor initialized")
        logger.info(f"  Normal Threshold: {self.THRESHOLD_NORMAL}")
        logger.info(f"  Rolling Threshold: {self.THRESHOLD_ROLLING}")
    
    def load_nn_models(self):
        """Progressive NN modellerini yükle"""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is not available. Cannot load NN models.")
        
        logger.info("\nLOADING PROGRESSIVE NN MODELS")
        
        for window_size in self.window_sizes:
            model_path = os.path.join(self.nn_model_dir, f'model_window_{window_size}.h5')
            scaler_path = os.path.join(self.nn_model_dir, f'scaler_window_{window_size}.pkl')
            
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                logger.warning(f"⚠️  NN files not found for window {window_size}")
                continue
            
            try:
                self.nn_models[window_size] = models.load_model(model_path, compile=False)
                self.nn_scalers[window_size] = joblib.load(scaler_path)
                logger.info(f"✅ Loaded NN model for window {window_size}")
            except Exception as e:
                logger.error(f"❌ Failed to load NN model {window_size}: {e}")
    
    def load_catboost_models(self):
        """CatBoost modellerini yükle"""
        if not CATBOOST_AVAILABLE:
            raise RuntimeError("CatBoost is not available. Cannot load CatBoost models.")
        
        logger.info("\nLOADING CATBOOST MODELS")
        
        for window_size in self.window_sizes:
            reg_path = os.path.join(self.catboost_model_dir, f'regressor_window_{window_size}.cbm')
            cls_path = os.path.join(self.catboost_model_dir, f'classifier_window_{window_size}.cbm')
            scaler_path = os.path.join(self.catboost_model_dir, f'scaler_window_{window_size}.pkl')
            
            if not (os.path.exists(reg_path) and os.path.exists(cls_path) and os.path.exists(scaler_path)):
                logger.warning(f"⚠️  CatBoost files not found for window {window_size}")
                continue
            
            try:
                self.catboost_regressors[window_size] = CatBoostRegressor().load_model(reg_path)
                self.catboost_classifiers[window_size] = CatBoostClassifier().load_model(cls_path)
                self.catboost_scalers[window_size] = joblib.load(scaler_path)
                logger.info(f"✅ Loaded CatBoost models for window {window_size}")
            except Exception as e:
                logger.error(f"❌ Failed to load CatBoost models {window_size}: {e}")
    
    def extract_features(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Veri için feature extraction yap"""
        if len(data) < window_size:
            raise ValueError(f"Data length ({len(data)}) < window size ({window_size})")
        
        feats = FeatureEngineering.extract_all_features(data.tolist())
        features = np.array(list(feats.values())).reshape(1, -1)
        return features
    
    def predict_nn_ensemble(self, data: np.ndarray) -> Dict:
        """NN ensemble tahminini yap"""
        if not self.nn_models: return {'regression': 0, 'threshold_prob': 0}
        
        ensemble_reg = []
        ensemble_thr = []
        
        for window_size in self.window_sizes:
            if window_size not in self.nn_models: continue
            
            features = self.extract_features(data, window_size)
            features = self.nn_scalers[window_size].transform(features)
            
            sequence = data[-window_size:].reshape(1, window_size, 1)
            sequence = np.log10(sequence + 1e-8)
            
            pred = self.nn_models[window_size].predict([features, sequence], verbose=0)
            ensemble_reg.append(pred[0][0][0])
            ensemble_thr.append(pred[2][0][0])
        
        return {
            'regression': float(np.mean(ensemble_reg)),
            'threshold_prob': float(np.mean(ensemble_thr))
        }
    
    def predict_catboost_ensemble(self, data: np.ndarray) -> Dict:
        """CatBoost ensemble tahminini yap"""
        if not self.catboost_regressors: return {'regression': 0, 'threshold_prob': 0}
        
        ensemble_reg = []
        ensemble_cls_probs = []
        
        for window_size in self.window_sizes:
            if window_size not in self.catboost_regressors: continue
            
            features = self.extract_features(data, window_size)
            features = self.catboost_scalers[window_size].transform(features)
            
            p_reg = self.catboost_regressors[window_size].predict(features)[0]
            try:
                p_cls_prob = self.catboost_classifiers[window_size].predict_proba(features)[0][1]
            except:
                p_cls_prob = float(self.catboost_classifiers[window_size].predict(features)[0])
            
            ensemble_reg.append(p_reg)
            ensemble_cls_probs.append(p_cls_prob)
        
        return {
            'regression': float(np.mean(ensemble_reg)),
            'threshold_prob': float(np.mean(ensemble_cls_probs))
        }
    
    def predict_consensus(self, data: np.ndarray) -> Dict:
        """
        Consensus tahmin yap (Normal ve Rolling Mod)
        """
        # NN ve CatBoost tahminlerini al
        nn_result = self.predict_nn_ensemble(data)
        catboost_result = self.predict_catboost_ensemble(data)
        
        nn_prob = nn_result['threshold_prob']
        cb_prob = catboost_result['threshold_prob']
        
        # Normal Mod Consensus (Her ikisi de >= 0.85)
        consensus_normal = (nn_prob >= self.THRESHOLD_NORMAL) and (cb_prob >= self.THRESHOLD_NORMAL)
        
        # Rolling Mod Consensus (Her ikisi de >= 0.95)
        consensus_rolling = (nn_prob >= self.THRESHOLD_ROLLING) and (cb_prob >= self.THRESHOLD_ROLLING)
        
        avg_pred = (nn_result['regression'] + catboost_result['regression']) / 2
        
        return {
            'consensus_normal': consensus_normal,
            'consensus_rolling': consensus_rolling,
            'nn_confidence': nn_prob,
            'catboost_confidence': cb_prob,
            'nn_prediction': nn_result['regression'],
            'catboost_prediction': catboost_result['regression'],
            'average_prediction': avg_pred
        }


def simulate_consensus_bankroll(
    predictions: List[Dict],
    actuals: np.ndarray,
    bet_amount: float = 10.0
) -> Dict:
    """
    Consensus tahminleri ile sanal kasa simülasyonu (2 Modlu)
    """
    initial_bankroll = len(actuals) * bet_amount
    
    # Kasa 1: Normal Mod (Dinamik Çıkış)
    w1, b1, w_cnt1 = initial_bankroll, 0, 0
    
    # Kasa 2: Rolling Mod (Güvenli Çıkış)
    w2, b2, w_cnt2 = initial_bankroll, 0, 0
    
    for i, (pred, actual) in enumerate(zip(predictions, actuals)):
        
        # KASA 1: NORMAL MOD
        if pred['consensus_normal']:
            w1 -= bet_amount
            b1 += 1
            # Dinamik çıkış: Tahminin %80'i, min 1.5, max 2.5
            exit_pt = min(max(1.5, pred['average_prediction'] * 0.8), 2.5)
            
            if actual >= exit_pt:
                w1 += bet_amount * exit_pt
                w_cnt1 += 1
        
        # KASA 2: ROLLING MOD
        if pred['consensus_rolling']:
            w2 -= bet_amount
            b2 += 1
            # Güvenli çıkış: 1.5x Sabit
            if actual >= 1.5:
                w2 += bet_amount * 1.5
                w_cnt2 += 1
    
    # Metrikler
    def calc_metrics(wallet, bets, wins):
        profit = wallet - initial_bankroll
        roi = (profit / initial_bankroll) * 100 if bets > 0 else 0
        wr = (wins / bets * 100) if bets > 0 else 0
        return {'final': wallet, 'profit': profit, 'roi': roi, 'win_rate': wr, 'total_bets': bets, 'total_wins': wins}
    
    return {
        'kasa_1': {**calc_metrics(w1, b1, w_cnt1), 'initial': initial_bankroll},
        'kasa_2': {**calc_metrics(w2, b2, w_cnt2), 'initial': initial_bankroll}
    }
