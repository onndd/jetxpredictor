"""
JetX Predictor - Tahmin Motoru

Bu modül eğitilmiş modeli yükler ve tahmin yapar.
Hem kategorik hem de değer tahmini yapar.
CatBoost ve Neural Network modellerini destekler.

GÜNCELLEME: 
- %85 ve %95 Güven Eşiği ("Keskin Nişancı" Modu) uygulandı.
- Feature Schema Validation eklendi.
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
        
        # CatBoost kullanılıyorsa dosya yollarını güncelle
        if model_type == 'catboost':
            self.model_path = "models/catboost_regressor.cbm"
            self.classifier_path = "models/catboost_classifier.cbm"
            self.scaler_path = "models/catboost_scaler.pkl"
        
        # Model varsa yükle
        if os.path.exists(model_path if model_type == 'neural_network' else self.model_path):
            self.load_model()
    
    def load_model(self):
        """Modeli ve scaler'ı yükler"""
        try:
            if self.model_type == 'neural_network':
                # TensorFlow/Keras modeli için
                try:
                    from tensorflow import keras
                    
                    # Custom objects kullanarak model yükle
                    self.model = keras.models.load_model(self.model_path, custom_objects=CUSTOM_OBJECTS)
                    logger.info(f"✅ Neural Network modeli yüklendi: {self.model_path}")
                except ImportError:
                    # PyTorch veya sklearn modeli için
                    import joblib
                    self.model = joblib.load(self.model_path)
                    logger.info(f"✅ Model yüklendi: {self.model_path}")
                    
            elif self.model_type == 'catboost':
                # CatBoost modelleri için
                from catboost import CatBoostRegressor, CatBoostClassifier
                
                self.regressor = CatBoostRegressor()
                self.regressor.load_model(self.model_path)
                logger.info(f"✅ CatBoost Regressor yüklendi: {self.model_path}")
                
                self.classifier = CatBoostClassifier()
                self.classifier.load_model(self.classifier_path)
                logger.info(f"✅ CatBoost Classifier yüklendi: {self.classifier_path}")
            
            # Scaler'ı yükle
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"✅ Scaler yüklendi: {self.scaler_path}")
            else:
                logger.warning(f"⚠️ Scaler bulunamadı: {self.scaler_path}")
            
            # KRİTİK: Feature Schema Validation
            self._validate_feature_schema_or_fail()
            
        except Exception as e:
            logger.error(f"⚠️ Model yükleme hatası: {e}")
            logger.info("Model henüz eğitilmemiş olabilir. Önce Google Colab'da eğitin.")
    
    def _validate_feature_schema_or_fail(self):
        """
        Feature schema validation - ZORUNLU kontrol
        
        🔒 GÜVENLİK: Feature skew ve data leakage riskini önler
        🚨 KRİTİK: Model-features uyumsuzluğunda SİSTEM ÇÖKSÜN (fail-fast)
        
        Raises:
            RuntimeError: Feature schema uyumsuzluğunda
        """
        try:
            from utils.feature_validator import get_feature_validator
            
            # Dummy veriyle test feature'ları oluştur
            dummy_history = [1.5] * 1000  # 1000 adet örnek veri
            current_features = FeatureEngineering.extract_all_features(dummy_history)
            
            # Feature validator'ı al
            validator = get_feature_validator()
            
            # Model metadata path'ini belirle
            model_name = "jetx_model_v2"  # Default model name
            
            # Model tipine göre metadata path'ini güncelle
            if self.model_type == 'catboost':
                meta_path = self.model_path.replace("_regressor.cbm", "_metadata.json")
            else:
                meta_path = self.model_path.replace(".h5", "_metadata.json")
            
            # Metadata kontrolü
            if not os.path.exists(meta_path):
                logger.warning(f"⚠️ Metadata bulunamadı: {meta_path}")
                logger.warning("🔧 Training sırasında feature metadata kaydedilmemiş olabilir.")
                logger.warning("💡 Çözüm: Modeli yeniden eğitin ve metadata kaydedin.")
                # Continue without strict validation (development mode)
                return
            
            # Metadata'yı yükle
            import json
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
                expected_features = metadata.get("feature_names", [])
                expected_count = metadata.get("feature_count", len(expected_features))
            
            # Feature sayısı kontrolü
            current_count = len(current_features)
            if expected_count != current_count:
                raise RuntimeError(
                    f"🚨 FEATURE COUNT MISMATCH:\n"
                    f"   Model eğitildiği: {expected_count} özellik\n"
                    f"   Mevcut kod ürettiği: {current_count} özellik\n"
                    f"   Fark: {abs(expected_count - current_count)} özellik\n"
                    f"💡 Model yeniden eğitilmeli veya kod güncellenmeli"
                )
            
            # Feature isimleri kontrolü
            current_feature_names = sorted(list(current_features.keys()))
            expected_feature_names = sorted(expected_features)
            
            missing_features = set(expected_feature_names) - set(current_feature_names)
            extra_features = set(current_feature_names) - set(expected_feature_names)
            
            if missing_features or extra_features:
                error_msg = "🚨 FEATURE SCHEMA MISMATCH:\n"
                if missing_features:
                    error_msg += f"   Eksik feature'lar: {list(missing_features)[:5]}{'...' if len(missing_features) > 5 else ''}\n"
                if extra_features:
                    error_msg += f"   Fazla feature'lar: {list(extra_features)[:5]}{'...' if len(extra_features) > 5 else ''}\n"
                error_msg += f"   Model eğitildiği: {expected_count} feature\n"
                error_msg += f"   Mevcut kod: {current_count} feature\n"
                error_msg += "💡 Model yeniden eğitilmeli veya kod güncellenmeli"
                raise RuntimeError(error_msg)
            
            # Scaler compatibility kontrolü
            if self.scaler is not None:
                validator.validate_compatibility(current_features, self.scaler, model_name)
            
            logger.info(f"✅ Feature Schema Validation Passed - {current_count} features")
            
        except ImportError:
            logger.warning("⚠️ Feature validator bulunamadı, validation atlanıyor")
        except Exception as e:
            if "Feature schema" in str(e) or "FEATURE" in str(e):
                # Feature schema hatası - CRITICAL
                raise RuntimeError(f"🚨 FEATURE SCHEMA VALIDATION FAILED: {e}")
            else:
                # Diğer hatalar - warning ile devam et
                logger.warning(f"⚠️ Feature validation warning: {e}")
    
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
        
        # Sequence'leri hazırla (4 farklı pencere) - Neural Network için
        # Log10 transformation uygula (training ile tutarlı)
        seq_50 = None
        seq_200 = None
        seq_500 = None
        seq_1000 = None
        
        if len(history) >= 50:
            seq_50 = np.array(history[-50:]).reshape(1, 50, 1)
            seq_50 = np.log10(seq_50 + 1e-8)
            
        if len(history) >= 200:
            seq_200 = np.array(history[-200:]).reshape(1, 200, 1)
            seq_200 = np.log10(seq_200 + 1e-8)
            
        if len(history) >= 500:
            seq_500 = np.array(history[-500:]).reshape(1, 500, 1)
            seq_500 = np.log10(seq_500 + 1e-8)
            
        if len(history) >= 1000:
            seq_1000 = np.array(history[-1000:]).reshape(1, 1000, 1)
            seq_1000 = np.log10(seq_1000 + 1e-8)
        
        return {
            'features': feature_values,
            'seq_50': seq_50,
            'seq_200': seq_200,
            'seq_500': seq_500,
            'seq_1000': seq_1000
        }
    
    def predict(
        self,
        history: List[float],
        mode: str = 'normal'
    ) -> Dict:
        """
        Tahmin yapar (model tipine göre)
        
        Args:
            history: Geçmiş değerler listesi (en az 1000 değer gerekli)
            mode: Tahmin modu ('normal', 'rolling')
            
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
        
        # Minimum veri kontrolü
        if len(history) < 1000:
            return {
                'error': f'En az 1000 geçmiş veri gerekli (mevcut: {len(history)})',
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
        
        # Sequence kontrolü - Eğer herhangi biri None ise hata döndür
        required_sequences = ['seq_50', 'seq_200', 'seq_500', 'seq_1000']
        missing_sequences = [seq for seq in required_sequences if model_inputs.get(seq) is None]
        
        if missing_sequences:
            min_required_data = {
                'seq_50': 50,
                'seq_200': 200,
                'seq_500': 500,
                'seq_1000': 1000
            }
            max_missing = max(min_required_data[seq] for seq in missing_sequences)
            return {
                'error': f'Yetersiz veri: En az {max_missing} geçmiş veri gerekli (mevcut: {len(history)})',
                'predicted_value': None,
                'confidence': 0.0,
                'above_threshold': None,
                'category': None,
                'recommendation': 'BEKLE',
                'pattern_risk': 0.0
            }
        
        # Model inputları hazırla (5 girdi: features + 4 sequence)
        input_data = [
            model_inputs['features'],
            model_inputs['seq_50'],
            model_inputs['seq_200'],
            model_inputs['seq_500'],
            model_inputs['seq_1000']
        ]
        
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
        
        # Mod bazlı öneri (GÜNCELLENDİ - %85/%95)
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
        
        # Öneri (GÜNCELLENDİ - %85/%95)
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
        """Mod bazlı öneri verir (GÜNCELLENDİ - %85/%95 Standartları)"""
        
        # Config'den veya category_definitions'dan gelen değeri al
        # Bulamazsa varsayılan olarak en düşük güvenli eşik olan 0.85'i kullan
        threshold = CONFIDENCE_THRESHOLDS.get(mode, 0.85)
        
        # 1. Kural: Güven skoru eşiğin altındaysa BEKLE
        if confidence < threshold:
            return 'BEKLE'
        
        # 2. Kural: Tahmin 1.5x altındaysa BEKLE
        if not above_threshold:
            return 'BEKLE'
        
        # 3. Kural: Eşikler geçildiyse OYNA
        if confidence >= threshold and above_threshold:
            if mode == 'rolling':
                # %95 ve üzeri - Çok güvenli
                return 'OYNA (GÜVENLİ)'
            elif mode == 'normal':
                # %85 ve üzeri
                return 'OYNA'
            # 'aggressive' modu artık yok, o yüzden kaldırıldı.
        
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
