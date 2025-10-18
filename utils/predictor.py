"""
JetX Predictor - Tahmin Motoru

Bu modül eğitilmiş modeli yükler ve tahmin yapar.
Hem kategorik hem de değer tahmini yapar.
CatBoost ve Neural Network modellerini destekler.
"""

import numpy as np
import joblib
from typing import Dict, Tuple, List, Optional
import os
import sys
import logging

# Kategori tanımlarını import et
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from category_definitions import (
    CategoryDefinitions,
    FeatureEngineering,
    CONFIDENCE_THRESHOLDS
)
from utils.custom_losses import CUSTOM_OBJECTS

# TensorFlow import'ı ekle (Lambda katmanları için)
try:
    import tensorflow as tf
except ImportError:
    tf = None
    logging.warning("TensorFlow not available. Lambda layer support will be limited.")

# Logging ayarla
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JetXPredictor:
    """JetX tahmin sınıfı - Neural Network ve CatBoost destekli"""
    
    def __init__(
        self,
        model_path: str = "models/jetx_model.h5",
        scaler_path: str = "models/scaler.pkl",
        model_type: str = 'neural_network'
    ):
        """
        Args:
            model_path: Eğitilmiş model dosyası yolu
            scaler_path: Scaler dosyası yolu
            model_type: Model tipi ('neural_network' veya 'catboost')
        """
        self.model_type = model_type
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        # CatBoost için ek modeller
        self.regressor = None
        self.classifier = None
        
        # Window sizes - model_info.json'dan okunacak veya default değer kullanılacak
        self.window_sizes = None  # load_model() fonksiyonunda ayarlanacak
        
        # CatBoost kullanılıyorsa dosya yollarını güncelle
        if model_type == 'catboost':
            self.model_path = "models/catboost_regressor.cbm"
            self.classifier_path = "models/catboost_classifier.cbm"
            self.scaler_path = "models/catboost_scaler.pkl"
        
        # Model varsa yükle
        if os.path.exists(model_path if model_type == 'neural_network' else self.model_path):
            self.load_model()
    
    def load_model(self):
        """Modeli ve scaler'ı yükler - Gelişmiş Lambda desteği ile"""
        try:
            # Model info JSON'u yükle (varsa)
            info_path = self.model_path.replace('.h5', '_info.json').replace('.cbm', '_info.json')
            if not os.path.exists(info_path):
                # Alternatif yolları dene
                model_dir = os.path.dirname(self.model_path)
                info_path = os.path.join(model_dir, 'model_info.json')
            
            if os.path.exists(info_path):
                try:
                    import json
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                        self.window_sizes = info.get('window_sizes', [1000, 500, 200, 50])
                        logger.info(f"✅ Model info yüklendi: {info_path}")
                        logger.info(f"📊 Window sizes: {self.window_sizes}")
                except Exception as e:
                    logger.warning(f"⚠️ Model info okunamadı: {e}")
                    self.window_sizes = [1000, 500, 200, 50]  # Default fallback
            else:
                # JSON yoksa default değerler
                self.window_sizes = [1000, 500, 200, 50]
                logger.warning(f"⚠️ Model info bulunamadı, default window sizes kullanılıyor: {self.window_sizes}")
            
            if self.model_type == 'neural_network':
                # TensorFlow/Keras modeli için - Gelişmiş yükleme
                self.model = self._load_neural_network_model()
                    
            elif self.model_type == 'catboost':
                # CatBoost modelleri için
                self._load_catboost_models()
            
            # Scaler'ı yükle
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"✅ Scaler yüklendi: {self.scaler_path}")
            else:
                logger.warning(f"⚠️ Scaler bulunamadı: {self.scaler_path}")
            
        except Exception as e:
            logger.error(f"⚠️ Model yükleme hatası: {e}")
            logger.info("Model henüz eğitilmemiş olabilir. Önce Google Colab'da eğitin.")
    
    def _load_neural_network_model(self):
        """Neural Network modelini gelişmiş Lambda desteği ile yükler"""
        from tensorflow import keras
        from tensorflow.keras import backend as K
        from tensorflow.keras.layers import Layer
        from tensorflow.keras.utils import register_keras_serializable
        
        # Custom Lambda Layer
        @register_keras_serializable()
        class SafeLambdaLayer(Layer):
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
            return K.sum(x, axis=1)
        
        def safe_mean_axis1(x):
            return K.mean(x, axis=1)
        
        # Gelişmiş custom objects
        custom_objects = {
            'SafeLambdaLayer': SafeLambdaLayer,
            'lambda': lambda x: K.sum(x, axis=1),
            'safe_sum_axis1': safe_sum_axis1,
            'safe_mean_axis1': safe_mean_axis1,
            'tf': tf,
            'K': K,
        }
        
        # Custom losses ekle
        custom_objects.update(CUSTOM_OBJECTS)
        
        # 3 aşamalı yükleme stratejisi
        model_loaded = False
        model = None
        
        # 1. Deneme: Normal yükleme
        try:
            logger.info(f"🔄 Loading NN model (Attempt 1: Normal)...")
            model = keras.models.load_model(self.model_path, compile=False)
            logger.info(f"✅ Neural Network modeli yüklendi (Normal): {self.model_path}")
            model_loaded = True
        except Exception as e:
            logger.warning(f"⚠️ Normal load failed: {str(e)[:100]}...")
        
        # 2. Deneme: Custom objects ile
        if not model_loaded:
            try:
                logger.info(f"🔄 Loading NN model (Attempt 2: Custom Objects)...")
                model = keras.models.load_model(
                    self.model_path, 
                    compile=False,
                    custom_objects=custom_objects
                )
                logger.info(f"✅ Neural Network modeli yüklendi (Custom Objects): {self.model_path}")
                model_loaded = True
            except Exception as e:
                logger.warning(f"⚠️ Custom objects load failed: {str(e)[:100]}...")
        
        # 3. Deneme: Lambda rebuild
        if not model_loaded:
            try:
                logger.info(f"🔄 Loading NN model (Attempt 3: Lambda Rebuild)...")
                model = self._load_model_with_lambda_rebuild(self.model_path, custom_objects)
                logger.info(f"✅ Neural Network modeli yüklendi (Lambda Rebuild): {self.model_path}")
                model_loaded = True
            except Exception as e:
                logger.error(f"❌ Lambda rebuild failed: {str(e)[:100]}...")
        
        if not model_loaded:
            raise RuntimeError(f"Failed to load NN model after 3 attempts: {self.model_path}")
        
        return model
    
    def _load_model_with_lambda_rebuild(self, model_path: str, custom_objects: dict):
        """Modeli Lambda katmanlarını yeniden oluşturarak yükler"""
        from tensorflow.keras import models
        
        # JSON + weights yükle
        json_path = model_path.replace('.h5', '.json')
        weights_path = model_path.replace('.h5', '_weights.h5')
        
        if os.path.exists(json_path) and os.path.exists(weights_path):
            with open(json_path, 'r') as f:
                model_json = f.read()
            
            model = models.model_from_json(model_json, custom_objects=custom_objects)
            model.load_weights(weights_path)
            return model
        
        # Son çare: weights-only yükle
        logger.warning("⚠️ Attempting weights-only loading...")
        model = models.load_model(model_path, compile=False, by_name=True, skip_mismatch=True)
        return model
    
    def _load_catboost_models(self):
        """CatBoost modellerini yükler"""
        from catboost import CatBoostRegressor, CatBoostClassifier
        
        try:
            self.regressor = CatBoostRegressor()
            self.regressor.load_model(self.model_path)
            logger.info(f"✅ CatBoost Regressor yüklendi: {self.model_path}")
        except Exception as e:
            logger.error(f"❌ CatBoost Regressor yüklenemedi: {e}")
            raise
        
        try:
            self.classifier = CatBoostClassifier()
            self.classifier.load_model(self.classifier_path)
            logger.info(f"✅ CatBoost Classifier yüklendi: {self.classifier_path}")
        except Exception as e:
            logger.error(f"❌ CatBoost Classifier yüklenemedi: {e}")
            raise
    
    def extract_features_from_history(self, history: List[float]) -> Dict:
        """
        Geçmiş verilerden özellik çıkarır ve sequence'ler oluşturur
        
        Args:
            history: Geçmiş değerler listesi (en yeni en sonda)
            
        Returns:
            Dictionary containing features and sequences
        """
        # Tüm özellikleri çıkar (geliştirilmiş feature engineering)
        features_dict = FeatureEngineering.extract_all_features(history)
        
        # Dictionary'yi array'e çevir
        feature_values = np.array(list(features_dict.values())).reshape(1, -1)
        
        # Scaler varsa normalize et
        if self.scaler is not None:
            feature_values = self.scaler.transform(feature_values)
        
        # Dinamik olarak sequence'leri hazırla
        # Model hangi window_sizes ile eğitildiyse onları kullan
        sequences = {}
        
        if self.window_sizes is None:
            # Fallback: Default değerler
            self.window_sizes = [1000, 500, 200, 50]
        
        # Her window size için sequence oluştur
        for window_size in self.window_sizes:
            if len(history) >= window_size:
                seq = np.array(history[-window_size:]).reshape(1, window_size, 1)
                seq = np.log10(seq + 1e-8)
                sequences[f'seq_{window_size}'] = seq
            else:
                sequences[f'seq_{window_size}'] = None
        
        # Sonucu dictionary olarak döndür
        result = {'features': feature_values}
        result.update(sequences)
        
        return result
    
    def predict(
        self,
        history: List[float],
        mode: str = 'normal'
    ) -> Dict:
        """
        Tahmin yapar (model tipine göre)
        
        Args:
            history: Geçmiş değerler listesi (en az 1000 değer gerekli)
            mode: Tahmin modu ('normal', 'rolling', 'aggressive')
            
        Returns:
            Tahmin sonuçları dictionary
        """
        # Model kontrolü
        if self.model_type == 'neural_network' and self.model is None:
            return {
                'error': 'Neural Network modeli yüklenmedi. Önce modeli Google Colab\'da eğitin.',
                'predicted_value': None,
                'confidence': 0.0,
                'above_threshold': None,
                'category': None,
                'recommendation': 'BEKLE'
            }
        elif self.model_type == 'catboost' and (self.regressor is None or self.classifier is None):
            return {
                'error': 'CatBoost modelleri yüklenmedi. Önce modelleri Google Colab\'da eğitin.',
                'predicted_value': None,
                'confidence': 0.0,
                'above_threshold': None,
                'category': None,
                'recommendation': 'BEKLE'
            }
        
        # Minimum veri kontrolü - en büyük window size kadar veri gerekli
        if self.window_sizes is None:
            self.window_sizes = [1000, 500, 200, 50]  # Fallback
        
        min_required = max(self.window_sizes)
        if len(history) < min_required:
            return {
                'error': f'En az {min_required} geçmiş veri gerekli (mevcut: {len(history)})',
                'predicted_value': None,
                'confidence': 0.0,
                'above_threshold': None,
                'category': None,
                'recommendation': 'BEKLE',
                'pattern_risk': 0.0
            }
        
        try:
            # Model tipine göre tahmin yap
            if self.model_type == 'neural_network':
                return self._predict_neural_network(history, mode)
            elif self.model_type == 'catboost':
                return self._predict_catboost(history, mode)
            else:
                return {
                    'error': f'Bilinmeyen model tipi: {self.model_type}',
                    'predicted_value': None,
                    'confidence': 0.0,
                    'above_threshold': None,
                    'category': None,
                    'recommendation': 'BEKLE'
                }
                
        except Exception as e:
            return {
                'error': f'Tahmin hatası: {str(e)}',
                'predicted_value': None,
                'confidence': 0.0,
                'above_threshold': None,
                'category': None,
                'recommendation': 'BEKLE',
                'pattern_risk': 0.0
            }
    
    def _predict_neural_network(
        self,
        history: List[float],
        mode: str
    ) -> Dict:
        """Neural Network ile tahmin yapar"""
        # Özellikleri ve sequence'leri çıkar
        model_inputs = self.extract_features_from_history(history)
        
        # Dinamik sequence kontrolü - model'in window_sizes'ına göre
        required_sequences = [f'seq_{ws}' for ws in self.window_sizes]
        missing_sequences = [seq for seq in required_sequences if model_inputs.get(seq) is None]
        
        if missing_sequences:
            # En büyük eksik window size'ı bul
            missing_sizes = [int(seq.split('_')[1]) for seq in missing_sequences]
            max_missing = max(missing_sizes)
            return {
                'error': f'Yetersiz veri: En az {max_missing} geçmiş veri gerekli (mevcut: {len(history)})',
                'predicted_value': None,
                'confidence': 0.0,
                'above_threshold': None,
                'category': None,
                'recommendation': 'BEKLE',
                'pattern_risk': 0.0
            }
        
        # Model inputları hazırla (dinamik olarak window_sizes'a göre)
        input_data = [model_inputs['features']]
        for window_size in self.window_sizes:
            input_data.append(model_inputs[f'seq_{window_size}'])
        
        # Tahmin yap (3 çıktı: regression, classification, threshold)
        predictions = self.model.predict(input_data, verbose=0)
        
        # Çıktıları ayır
        regression_pred = predictions[0]
        classification_pred = predictions[1]
        threshold_pred = predictions[2]
        
        predicted_value = float(regression_pred[0][0])
        threshold_prob = float(threshold_pred[0][0])
        
        # Model confidence
        model_confidence = max(threshold_prob, 1 - threshold_prob)
        
        # Güven skorunu hesapla (model confidence'a daha fazla ağırlık)
        confidence = model_confidence * 0.7 + self._calculate_confidence(history, predicted_value) * 0.3
        
        # 1.5x eşik kontrolü
        above_threshold = predicted_value >= CategoryDefinitions.CRITICAL_THRESHOLD
        
        # Kategori
        category = CategoryDefinitions.get_category(predicted_value)
        detailed_category = CategoryDefinitions.get_detailed_category(predicted_value)
        
        # Mod bazlı öneri
        recommendation = self._get_recommendation(confidence, mode, above_threshold)
        
        # Uyarılar
        warnings = self._generate_warnings(history, predicted_value, confidence)
        
        return {
            'predicted_value': round(predicted_value, 2),
            'confidence': round(confidence, 2),
            'above_threshold': above_threshold,
            'threshold_probability': round(threshold_prob, 2),
            'category': category,
            'detailed_category': detailed_category,
            'recommendation': recommendation,
            'pattern_risk': 0.0,
            'warnings': warnings,
            'mode': mode,
            'model_type': 'neural_network'
        }
    
    def _predict_catboost(
        self,
        history: List[float],
        mode: str
    ) -> Dict:
        """CatBoost ile tahmin yapar"""
        # Özellikleri çıkar
        model_inputs = self.extract_features_from_history(history)
        feature_values = model_inputs['features']
        
        # Regressor tahmin (değer)
        predicted_value = float(self.regressor.predict(feature_values)[0])
        
        # Classifier tahmin (1.5 eşik)
        threshold_proba = self.classifier.predict_proba(feature_values)[0]
        threshold_prob = float(threshold_proba[1])  # 1.5 üstü olma olasılığı
        
        # Confidence (model confidence'a daha fazla ağırlık)
        model_confidence = max(threshold_prob, 1 - threshold_prob)
        confidence = model_confidence * 0.7 + self._calculate_confidence(history, predicted_value) * 0.3
        
        # 1.5x eşik kontrolü
        above_threshold = predicted_value >= CategoryDefinitions.CRITICAL_THRESHOLD
        
        # Kategori
        category = CategoryDefinitions.get_category(predicted_value)
        detailed_category = CategoryDefinitions.get_detailed_category(predicted_value)
        
        # Öneri
        recommendation = self._get_recommendation(confidence, mode, above_threshold)
        
        # Uyarılar
        warnings = self._generate_warnings(history, predicted_value, confidence)
        
        return {
            'predicted_value': round(predicted_value, 2),
            'confidence': round(confidence, 2),
            'above_threshold': above_threshold,
            'threshold_probability': round(threshold_prob, 2),
            'category': category,
            'detailed_category': detailed_category,
            'recommendation': recommendation,
            'pattern_risk': 0.0,
            'warnings': warnings,
            'mode': mode,
            'model_type': 'catboost'
        }
    
    def _calculate_confidence(
        self,
        history: List[float],
        predicted_value: float
    ) -> float:
        """Güven skorunu hesaplar"""
        confidence = 0.65
        
        if len(history) >= 10:
            recent = history[-10:]
            volatility = np.std(recent)
            
            if volatility < 2.0:
                confidence += 0.10
            elif volatility > 5.0:
                confidence -= 0.10
        
        if 1.0 <= predicted_value <= 10.0:
            confidence += 0.10
        elif predicted_value > 50.0:
            confidence -= 0.15
        
        return max(0.0, min(1.0, confidence))
    
    def _get_recommendation(
        self,
        confidence: float,
        mode: str,
        above_threshold: bool
    ) -> str:
        """Mod bazlı öneri verir"""
        threshold = CONFIDENCE_THRESHOLDS.get(mode, 0.65)
        
        if confidence < threshold:
            return 'BEKLE'
        
        if not above_threshold:
            return 'BEKLE'
        
        if confidence >= threshold and above_threshold:
            if mode == 'rolling' and confidence >= 0.80:
                return 'OYNA'
            elif mode == 'normal' and confidence >= 0.65:
                return 'OYNA'
            elif mode == 'aggressive' and confidence >= 0.50:
                return 'RİSKLİ'
        
        return 'BEKLE'
    
    def _generate_warnings(
        self,
        history: List[float],
        predicted_value: float,
        confidence: float
    ) -> List[str]:
        """Uyarılar oluşturur"""
        warnings = []
        
        if confidence < 0.60:
            warnings.append(f"⚠️ Düşük güven seviyesi ({confidence:.0%})")
        
        if 1.45 <= predicted_value <= 1.55:
            warnings.append("🚨 KRİTİK BÖLGE: 1.45-1.55x arası çok riskli!")
        
        if predicted_value < CategoryDefinitions.CRITICAL_THRESHOLD:
            warnings.append(f"❌ TAHMİN 1.5x ALTINDA ({predicted_value:.2f}x) - OYNAMA!")
        
        try:
            features = FeatureEngineering.extract_all_features(history)
            
            if len(history) >= 50:
                distance_10x = features.get('distance_from_10x', 999)
                distance_20x = features.get('distance_from_20x', 999)
                
                if distance_10x < 15 or distance_20x < 20:
                    volatility = features.get('recent_volatility_pattern', 0)
                    if volatility > 0.5:
                        warnings.append("❄️ SOĞUMA DÖNEMİ OLABİLİR!")
                        warnings.append("📊 Tavsiye: Sonraki 10-15 eli oynama")
            
            volatility_norm = features.get('volatility_normalization', 0)
            if volatility_norm > 0.6:
                warnings.append("✅ TOPARLANMA İŞARETİ tespit edildi")
            
            z_score = features.get('z_score', 0)
            if abs(z_score) > 2.5:
                warnings.append(f"🔔 Anormal değer tespit edildi (Z-score: {z_score:.2f})")
        except:
            pass
        
        warnings.append("⚠️ Bu tahmin %100 doğru değildir, para kaybedebilirsiniz")
        
        return warnings
    
    def predict_threshold_only(self, history: List[float]) -> Dict:
        """Sadece 1.5x eşik tahmini yapar"""
        if len(history) < 10:
            return {
                'above_threshold_probability': 0.5,
                'recommendation': 'BEKLE'
            }
        
        recent_50 = history[-50:] if len(history) >= 50 else history
        above_count = sum(1 for v in recent_50 if v >= 1.5)
        probability = above_count / len(recent_50)
        
        return {
            'above_threshold_probability': round(probability, 2),
            'recommendation': 'OYNA' if probability > 0.65 else 'BEKLE'
        }
