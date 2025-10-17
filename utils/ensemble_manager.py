"""
JetX Predictor - Intelligent Ensemble Manager

Stacking Ensemble sistemi - 3 base modeli (Progressive, Ultra, CatBoost)
birleştirerek daha iyi tahminler yapar.

Meta-model ile hangi modele ne zaman güvenileceğini öğrenir.
"""

import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, List, Optional, Tuple
import os
import sys

# TensorFlow/Keras için
try:
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow bulunamadı. Neural Network modelleri yüklenemeyecek.")

# CatBoost için
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logging.warning("CatBoost bulunamadı. CatBoost modelleri yüklenemeyecek.")

# Custom losses
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.custom_losses import CUSTOM_OBJECTS
from category_definitions import CategoryDefinitions, FeatureEngineering

# Logging ayarla
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StackingEnsemble:
    """
    Stacking Ensemble - Meta-model ile akıllı ensemble
    
    3 Base Model:
    1. Progressive (Dengeli, 3 aşamalı)
    2. Ultra Aggressive (Agresif, yüksek doğruluk hedefi)
    3. CatBoost (Hızlı, feature-based)
    
    Meta-Model:
    - CatBoost classifier
    - Base modellerin tahminlerinden öğrenir
    - Hangi modele ne zaman güveneceğini bilir
    """
    
    def __init__(
        self,
        progressive_model_path: str = "models/jetx_progressive_final.h5",
        progressive_scaler_path: str = "models/scaler_progressive.pkl",
        ultra_model_path: str = "models/jetx_ultra_model.h5",
        ultra_scaler_path: str = "models/scaler_ultra.pkl",
        catboost_regressor_path: str = "models/catboost_regressor.cbm",
        catboost_classifier_path: str = "models/catboost_classifier.cbm",
        catboost_scaler_path: str = "models/catboost_scaler.pkl",
        meta_model_path: str = "models/meta_model.json"
    ):
        """
        Args:
            progressive_model_path: Progressive model dosya yolu
            progressive_scaler_path: Progressive scaler dosya yolu
            ultra_model_path: Ultra Aggressive model dosya yolu
            ultra_scaler_path: Ultra Aggressive scaler dosya yolu
            catboost_regressor_path: CatBoost regressor dosya yolu
            catboost_classifier_path: CatBoost classifier dosya yolu
            catboost_scaler_path: CatBoost scaler dosya yolu
            meta_model_path: Meta-model dosya yolu
        """
        self.models = {}
        self.scalers = {}
        self.meta_model = None
        
        # Model yolları
        self.model_paths = {
            'progressive': {
                'model': progressive_model_path,
                'scaler': progressive_scaler_path
            },
            'ultra': {
                'model': ultra_model_path,
                'scaler': ultra_scaler_path
            },
            'catboost': {
                'regressor': catboost_regressor_path,
                'classifier': catboost_classifier_path,
                'scaler': catboost_scaler_path
            }
        }
        self.meta_model_path = meta_model_path
        
        # Modelleri yükle
        self.load_all_models()
    
    def load_all_models(self):
        """Tüm base modelleri ve meta-modeli yükle"""
        logger.info("Base modelleri yükleniyor...")
        
        # Progressive Model
        if TENSORFLOW_AVAILABLE and os.path.exists(self.model_paths['progressive']['model']):
            try:
                self.models['progressive'] = keras.models.load_model(
                    self.model_paths['progressive']['model'],
                    custom_objects=CUSTOM_OBJECTS
                )
                self.scalers['progressive'] = joblib.load(self.model_paths['progressive']['scaler'])
                logger.info("✅ Progressive model yüklendi")
            except Exception as e:
                logger.warning(f"⚠️ Progressive model yüklenemedi: {e}")
                self.models['progressive'] = None
        else:
            logger.warning("⚠️ Progressive model dosyası bulunamadı")
            self.models['progressive'] = None
        
        # Ultra Aggressive Model
        if TENSORFLOW_AVAILABLE and os.path.exists(self.model_paths['ultra']['model']):
            try:
                self.models['ultra'] = keras.models.load_model(
                    self.model_paths['ultra']['model'],
                    custom_objects=CUSTOM_OBJECTS
                )
                self.scalers['ultra'] = joblib.load(self.model_paths['ultra']['scaler'])
                logger.info("✅ Ultra Aggressive model yüklendi")
            except Exception as e:
                logger.warning(f"⚠️ Ultra Aggressive model yüklenemedi: {e}")
                self.models['ultra'] = None
        else:
            logger.warning("⚠️ Ultra Aggressive model dosyası bulunamadı")
            self.models['ultra'] = None
        
        # CatBoost Models
        if CATBOOST_AVAILABLE and os.path.exists(self.model_paths['catboost']['regressor']):
            try:
                self.models['catboost_reg'] = CatBoostRegressor()
                self.models['catboost_reg'].load_model(self.model_paths['catboost']['regressor'])
                
                self.models['catboost_cls'] = CatBoostClassifier()
                self.models['catboost_cls'].load_model(self.model_paths['catboost']['classifier'])
                
                self.scalers['catboost'] = joblib.load(self.model_paths['catboost']['scaler'])
                logger.info("✅ CatBoost modelleri yüklendi")
            except Exception as e:
                logger.warning(f"⚠️ CatBoost modelleri yüklenemedi: {e}")
                self.models['catboost_reg'] = None
                self.models['catboost_cls'] = None
        else:
            logger.warning("⚠️ CatBoost model dosyaları bulunamadı")
            self.models['catboost_reg'] = None
            self.models['catboost_cls'] = None
        
        # Meta-Model
        if CATBOOST_AVAILABLE and os.path.exists(self.meta_model_path):
            try:
                self.meta_model = CatBoostClassifier()
                self.meta_model.load_model(self.meta_model_path)
                logger.info("✅ Meta-model yüklendi")
            except Exception as e:
                logger.warning(f"⚠️ Meta-model yüklenemedi: {e}")
                self.meta_model = None
        else:
            logger.warning("⚠️ Meta-model dosyası bulunamadı. Fallback: Weighted average kullanılacak.")
            self.meta_model = None
    
    def extract_features_for_model(
        self, 
        history: List[float], 
        model_type: str
    ) -> Dict:
        """
        Belirli bir model tipi için feature extraction
        
        Args:
            history: Geçmiş değerler
            model_type: 'progressive', 'ultra', veya 'catboost'
            
        Returns:
            Model input dictionary
        """
        # Feature extraction
        features_dict = FeatureEngineering.extract_all_features(history)
        feature_values = np.array(list(features_dict.values())).reshape(1, -1)
        
        # Scaler
        if model_type in self.scalers and self.scalers[model_type] is not None:
            feature_values = self.scalers[model_type].transform(feature_values)
        
        result = {'features': feature_values}
        
        # Neural Network modelleri için sequence'ler gerekli
        if model_type in ['progressive', 'ultra']:
            # Sequence 50
            if len(history) >= 50:
                seq_50 = np.array(history[-50:]).reshape(1, 50, 1)
                result['seq_50'] = np.log10(seq_50 + 1e-8)
            
            # Sequence 200
            if len(history) >= 200:
                seq_200 = np.array(history[-200:]).reshape(1, 200, 1)
                result['seq_200'] = np.log10(seq_200 + 1e-8)
            
            # Sequence 500
            if len(history) >= 500:
                seq_500 = np.array(history[-500:]).reshape(1, 500, 1)
                result['seq_500'] = np.log10(seq_500 + 1e-8)
            
            # Sequence 1000 (Progressive için)
            if len(history) >= 1000:
                seq_1000 = np.array(history[-1000:]).reshape(1, 1000, 1)
                result['seq_1000'] = np.log10(seq_1000 + 1e-8)
        
        return result
    
    def get_individual_predictions(
        self,
        history: List[float]
    ) -> Dict:
        """
        Her base modelden ayrı tahmin al
        
        Args:
            history: Geçmiş değerler
            
        Returns:
            Her modelin tahminleri
        """
        predictions = {}
        
        # Progressive
        if self.models.get('progressive') is not None:
            try:
                inputs = self.extract_features_for_model(history, 'progressive')
                input_list = [
                    inputs['features'],
                    inputs.get('seq_50'),
                    inputs.get('seq_200'),
                    inputs.get('seq_500'),
                    inputs.get('seq_1000')
                ]
                
                preds = self.models['progressive'].predict(input_list, verbose=0)
                predictions['progressive'] = {
                    'predicted_value': float(preds[0][0][0]),
                    'threshold_probability': float(preds[2][0][0]),
                    'above_threshold': float(preds[2][0][0]) >= 0.5
                }
            except Exception as e:
                logger.error(f"Progressive tahmin hatası: {e}")
                predictions['progressive'] = None
        
        # Ultra Aggressive
        if self.models.get('ultra') is not None:
            try:
                inputs = self.extract_features_for_model(history, 'ultra')
                # Ultra model 1000 yerine 500 kullanıyor, seq_1000 varsa kullan yoksa seq_500
                input_list = [
                    inputs['features'],
                    inputs.get('seq_50'),
                    inputs.get('seq_200'),
                    inputs.get('seq_500'),
                    inputs.get('seq_1000') if inputs.get('seq_1000') is not None else inputs.get('seq_500')
                ]
                
                preds = self.models['ultra'].predict(input_list, verbose=0)
                predictions['ultra'] = {
                    'predicted_value': float(preds[0][0][0]),
                    'threshold_probability': float(preds[2][0][0]),
                    'above_threshold': float(preds[2][0][0]) >= 0.5
                }
            except Exception as e:
                logger.error(f"Ultra tahmin hatası: {e}")
                predictions['ultra'] = None
        
        # CatBoost
        if self.models.get('catboost_reg') is not None and self.models.get('catboost_cls') is not None:
            try:
                inputs = self.extract_features_for_model(history, 'catboost')
                
                # Regression
                reg_pred = self.models['catboost_reg'].predict(inputs['features'])
                
                # Classification
                cls_proba = self.models['catboost_cls'].predict_proba(inputs['features'])
                
                predictions['catboost'] = {
                    'predicted_value': float(reg_pred[0]),
                    'threshold_probability': float(cls_proba[0][1]),  # 1.5 üstü olasılığı
                    'above_threshold': cls_proba[0][1] >= 0.5
                }
            except Exception as e:
                logger.error(f"CatBoost tahmin hatası: {e}")
                predictions['catboost'] = None
        
        return predictions
    
    def predict_with_stacking(
        self,
        history: List[float]
    ) -> Dict:
        """
        Stacking ensemble ile tahmin yap
        
        Args:
            history: Geçmiş değerler
            
        Returns:
            Ensemble tahmini ve detayları
        """
        # Minimum veri kontrolü
        if len(history) < 1000:
            return {
                'error': f'Ensemble için en az 1000 geçmiş veri gerekli (mevcut: {len(history)})',
                'predicted_value': None,
                'confidence': 0.0,
                'ensemble_method': 'none'
            }
        
        # Individual predictions
        individual_preds = self.get_individual_predictions(history)
        
        # Hiç model yüklü değilse
        valid_preds = {k: v for k, v in individual_preds.items() if v is not None}
        if len(valid_preds) == 0:
            return {
                'error': 'Hiçbir base model yüklü değil!',
                'predicted_value': None,
                'confidence': 0.0,
                'ensemble_method': 'none'
            }
        
        # Meta-model varsa stacking kullan
        if self.meta_model is not None:
            try:
                # Meta-model input: [prog_pred, ultra_pred, catboost_pred]
                meta_input = []
                for model_name in ['progressive', 'ultra', 'catboost']:
                    if individual_preds.get(model_name) is not None:
                        meta_input.append(individual_preds[model_name]['threshold_probability'])
                    else:
                        meta_input.append(0.5)  # Eksik model için neutral
                
                meta_input = np.array(meta_input).reshape(1, -1)
                
                # Meta-model tahmini
                meta_pred_proba = self.meta_model.predict_proba(meta_input)
                threshold_prob = float(meta_pred_proba[0][1])
                
                # Value prediction: Weighted average (meta-model confidence'a göre)
                weights = meta_pred_proba[0]  # [prob_below, prob_above]
                weighted_value = 0.0
                weight_sum = 0.0
                
                for model_name, pred_data in valid_preds.items():
                    # Model tahminini ağırlıklandır
                    weight = pred_data['threshold_probability'] if pred_data['above_threshold'] else (1 - pred_data['threshold_probability'])
                    weighted_value += pred_data['predicted_value'] * weight
                    weight_sum += weight
                
                if weight_sum > 0:
                    predicted_value = weighted_value / weight_sum
                else:
                    # Fallback: Simple average
                    predicted_value = np.mean([p['predicted_value'] for p in valid_preds.values()])
                
                return {
                    'predicted_value': round(predicted_value, 2),
                    'threshold_probability': round(threshold_prob, 2),
                    'above_threshold': threshold_prob >= 0.5,
                    'confidence': round(max(meta_pred_proba[0]), 2),
                    'ensemble_method': 'stacking',
                    'individual_predictions': individual_preds,
                    'meta_weights': meta_pred_proba[0].tolist(),
                    'active_models': list(valid_preds.keys())
                }
                
            except Exception as e:
                logger.error(f"Stacking hatası, weighted average'a geçiliyor: {e}")
                # Fallback to weighted average
        
        # Meta-model yoksa: Weighted Average (performansa göre)
        # Basit implementasyon: Eşit ağırlık
        avg_value = np.mean([p['predicted_value'] for p in valid_preds.values()])
        avg_threshold = np.mean([p['threshold_probability'] for p in valid_preds.values()])
        
        return {
            'predicted_value': round(avg_value, 2),
            'threshold_probability': round(avg_threshold, 2),
            'above_threshold': avg_threshold >= 0.5,
            'confidence': round(avg_threshold if avg_threshold >= 0.5 else 1 - avg_threshold, 2),
            'ensemble_method': 'weighted_average',
            'individual_predictions': individual_preds,
            'active_models': list(valid_preds.keys())
        }
    
    def predict(
        self,
        history: List[float],
        mode: str = 'ensemble'
    ) -> Dict:
        """
        Ana tahmin fonksiyonu
        
        Args:
            history: Geçmiş değerler
            mode: 'ensemble', 'progressive', 'ultra', veya 'catboost'
            
        Returns:
            Tahmin sonucu
        """
        if mode == 'ensemble':
            result = self.predict_with_stacking(history)
        else:
            # Tek model tahmini
            individual_preds = self.get_individual_predictions(history)
            if individual_preds.get(mode) is not None:
                result = individual_preds[mode]
                result['ensemble_method'] = 'single_model'
                result['model_used'] = mode
            else:
                result = {
                    'error': f'{mode} modeli yüklü değil',
                    'predicted_value': None,
                    'confidence': 0.0
                }
        
        # Kategori bilgisi ekle
        if result.get('predicted_value') is not None:
            result['category'] = CategoryDefinitions.get_category(result['predicted_value'])
            result['detailed_category'] = CategoryDefinitions.get_detailed_category(result['predicted_value'])
        
        return result
