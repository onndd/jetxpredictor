"""
JetX Predictor - Tahmin Motoru

Bu modül eğitilmiş modeli yükler ve tahmin yapar.
Hem kategorik hem de değer tahmini yapar.
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
    """JetX tahmin sınıfı"""
    
    def __init__(
        self,
        model_path: str = "models/jetx_model.h5",
        scaler_path: str = "models/scaler.pkl"
    ):
        """
        Args:
            model_path: Eğitilmiş model dosyası yolu
            scaler_path: Scaler dosyası yolu
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        # Model varsa yükle
        if os.path.exists(model_path):
            self.load_model()
    
    def load_model(self):
        """Modeli ve scaler'ı yükler"""
        try:
            # TensorFlow/Keras modeli için
            try:
                from tensorflow import keras
                
                # Custom objects kullanarak model yükle
                self.model = keras.models.load_model(self.model_path, custom_objects=CUSTOM_OBJECTS)
                logger.info(f"✅ Model yüklendi: {self.model_path}")
            except ImportError:
                # PyTorch veya sklearn modeli için
                import joblib
                self.model = joblib.load(self.model_path)
                logger.info(f"✅ Model yüklendi: {self.model_path}")
            
            # Scaler'ı yükle
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"✅ Scaler yüklendi: {self.scaler_path}")
            else:
                logger.warning(f"⚠️ Scaler bulunamadı: {self.scaler_path}")
            
        except Exception as e:
            logger.error(f"⚠️ Model yükleme hatası: {e}")
            logger.info("Model henüz eğitilmemiş olabilir. Önce Google Colab'da eğitin.")
    
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
        
        # Sequence'leri hazırla (3 farklı pencere) - DOĞRU SHAPE: (1, length, 1)
        # Log10 transformation uygula (training ile tutarlı)
        seq_50 = None
        seq_200 = None
        seq_500 = None
        
        if len(history) >= 50:
            seq_50 = np.array(history[-50:]).reshape(1, 50, 1)
            seq_50 = np.log10(seq_50 + 1e-8)  # Training'deki gibi log transform
            
        if len(history) >= 200:
            seq_200 = np.array(history[-200:]).reshape(1, 200, 1)
            seq_200 = np.log10(seq_200 + 1e-8)
            
        if len(history) >= 500:
            seq_500 = np.array(history[-500:]).reshape(1, 500, 1)
            seq_500 = np.log10(seq_500 + 1e-8)
        
        return {
            'features': feature_values,
            'seq_50': seq_50,
            'seq_200': seq_200,
            'seq_500': seq_500
        }
    
    def predict(
        self,
        history: List[float],
        mode: str = 'normal'
    ) -> Dict:
        """
        Tahmin yapar
        
        Args:
            history: Geçmiş değerler listesi (en az 50 değer önerilir)
            mode: Tahmin modu ('normal', 'rolling', 'aggressive')
            
        Returns:
            Tahmin sonuçları dictionary
        """
        if self.model is None:
            return {
                'error': 'Model yüklenmedi. Önce modeli Google Colab\'da eğitin.',
                'predicted_value': None,
                'confidence': 0.0,
                'above_threshold': None,
                'category': None,
                'recommendation': 'BEKLE'
            }
        
        # Minimum veri kontrolü - artık 500 veri gerekiyor (uzun pencere için)
        if len(history) < 500:
            return {
                'error': f'En az 500 geçmiş veri gerekli (mevcut: {len(history)})',
                'predicted_value': None,
                'confidence': 0.0,
                'above_threshold': None,
                'category': None,
                'recommendation': 'BEKLE',
                'pattern_risk': 0.0
            }
        
        try:
            # Özellikleri ve sequence'leri çıkar
            model_inputs = self.extract_features_from_history(history)
            
            # Model inputları hazırla
            input_data = [
                model_inputs['features'],
                model_inputs['seq_50'],
                model_inputs['seq_200'],
                model_inputs['seq_500']
            ]
            
            # Tahmin yap (3 çıktı: regression, classification, threshold)
            predictions = self.model.predict(input_data, verbose=0)
            
            # Çıktıları ayır - MODEL 3 OUTPUT VERİYOR
            regression_pred = predictions[0]  # (batch, 1)
            classification_pred = predictions[1]  # (batch, 3)
            threshold_pred = predictions[2]  # (batch, 1) - sigmoid output
            
            predicted_value = float(regression_pred[0][0])
            threshold_prob = float(threshold_pred[0][0])  # 1.5x üstü olma olasılığı
            
            # Model confidence'ı threshold prediction'dan türet
            # Eğer tahmin kesin ise (0'a veya 1'e yakın) confidence yüksek
            model_confidence = max(threshold_prob, 1 - threshold_prob)
            
            # Pattern risk için basit bir hesaplama (opsiyonel)
            pattern_risk = 0.0  # Şimdilik kullanılmıyor
            
            # Güven skorunu hesapla (model çıktısı + heuristic)
            confidence = (model_confidence + self._calculate_confidence(history, predicted_value)) / 2
            
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
                'threshold_probability': round(threshold_prob, 2),  # Model'in 1.5x üstü tahmini
                'category': category,
                'detailed_category': detailed_category,
                'recommendation': recommendation,
                'pattern_risk': round(pattern_risk, 2),
                'warnings': warnings,
                'mode': mode
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
    
    def _calculate_confidence(
        self,
        history: List[float],
        predicted_value: float
    ) -> float:
        """
        Güven skorunu hesaplar - BIAS TEMİZLENDİ
        
        Args:
            history: Geçmiş veriler
            predicted_value: Tahmin edilen değer
            
        Returns:
            Güven skoru (0-1 arası)
        """
        # Model kendi güvenini hesaplıyor, bu sadece yardımcı
        confidence = 0.65  # Başlangıç değeri
        
        # Son değerlerin volatilitesine göre ayarla
        if len(history) >= 10:
            recent = history[-10:]
            volatility = np.std(recent)
            
            # Düşük volatilite = yüksek güven
            if volatility < 2.0:
                confidence += 0.10
            elif volatility > 5.0:
                confidence -= 0.10
        
        # Tahmin edilen değer makul aralıkta mı?
        if 1.0 <= predicted_value <= 10.0:
            confidence += 0.10
        elif predicted_value > 50.0:
            confidence -= 0.15
        
        # ÖNCEKİ BIAS KALDIRILDI: 1.45-1.55 arası güveni düşürmüyoruz
        # Model 1.5 altı tahmin yapabilmeli!
        
        # 0-1 aralığında tut
        return max(0.0, min(1.0, confidence))
    
    def _get_recommendation(
        self,
        confidence: float,
        mode: str,
        above_threshold: bool
    ) -> str:
        """
        Mod bazlı öneri verir - BİAS TEMİZLENDİ
        
        Args:
            confidence: Güven skoru
            mode: Tahmin modu
            above_threshold: 1.5x üstü mü
            
        Returns:
            Öneri ('OYNA', 'BEKLE', 'RİSKLİ', 'BEKLE_ALTI')
        """
        threshold = CONFIDENCE_THRESHOLDS.get(mode, 0.65)
        
        if confidence < threshold:
            return 'BEKLE'
        
        # 1.5 altı tahminler için uyarı
        if not above_threshold:
            # 1.5 altı tahmin - kullanıcı kesinlikle OYNAMAMALI
            return 'BEKLE'  # Risk çok yüksek
        
        # 1.5 üstü tahminler
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
        """
        Uyarılar oluşturur - GELİŞTİRİLMİŞ (pattern_risk dahil)
        
        Args:
            history: Geçmiş veriler
            predicted_value: Tahmin edilen değer
            confidence: Güven skoru
            
        Returns:
            Uyarı listesi
        """
        warnings = []
        
        # Düşük güven uyarısı
        if confidence < 0.60:
            warnings.append(f"⚠️ Düşük güven seviyesi ({confidence:.0%})")
        
        # Kritik bölge uyarısı
        if 1.45 <= predicted_value <= 1.55:
            warnings.append("🚨 KRİTİK BÖLGE: 1.45-1.55x arası çok riskli!")
        
        # 1.5x altı uyarısı
        if predicted_value < CategoryDefinitions.CRITICAL_THRESHOLD:
            warnings.append(f"❌ TAHMİN 1.5x ALTINDA ({predicted_value:.2f}x) - OYNAMA!")
        
        # GELİŞTİRİLMİŞ: Özelliklerden pattern kontrolü (model öğrenecek)
        try:
            features = FeatureEngineering.extract_all_features(history)
            
            # Soğuma dönemi pattern'leri (model öğreniyor, net kural yok)
            if len(history) >= 50:
                # Büyük çarpandan mesafe
                distance_10x = features.get('distance_from_10x', 999)
                distance_20x = features.get('distance_from_20x', 999)
                
                # Son 10-15 elde büyük çarpan olduysa
                if distance_10x < 15 or distance_20x < 20:
                    volatility = features.get('recent_volatility_pattern', 0)
                    if volatility > 0.5:  # Yüksek volatilite
                        warnings.append("❄️ SOĞUMA DÖNEMİ OLABİLİR!")
                        warnings.append("📊 Tavsiye: Sonraki 10-15 eli oynama")
                        warnings.append("🎲 JetX çok düzensiz olabilir")
                        warnings.append("📉 Tahmin doğruluğu düşebilir")
            
            # Recovery (toparlanma) işareti
            volatility_norm = features.get('volatility_normalization', 0)
            if volatility_norm > 0.6:  # Volatilite normalleşiyor
                warnings.append("✅ TOPARLANMA İŞARETİ tespit edildi")
                warnings.append("💚 Güvenli oynamaya başlayabilirsiniz")
            
            # Anomali tespiti
            z_score = features.get('z_score', 0)
            if abs(z_score) > 2.5:
                warnings.append(f"🔔 Anormal değer tespit edildi (Z-score: {z_score:.2f})")
            
        except:
            pass  # Özellik çıkarma hatası varsa devam et
        
        # Genel uyarı
        warnings.append("⚠️ Bu tahmin %100 doğru değildir, para kaybedebilirsiniz")
        
        return warnings
    
    def predict_threshold_only(self, history: List[float]) -> Dict:
        """
        Sadece 1.5x eşik tahmini yapar (basitleştirilmiş)
        
        Args:
            history: Geçmiş veriler
            
        Returns:
            Eşik tahmini
        """
        if len(history) < 10:
            return {
                'above_threshold_probability': 0.5,
                'recommendation': 'BEKLE'
            }
        
        # Basit istatistiksel yaklaşım
        recent_50 = history[-50:] if len(history) >= 50 else history
        above_count = sum(1 for v in recent_50 if v >= 1.5)
        probability = above_count / len(recent_50)
        
        return {
            'above_threshold_probability': round(probability, 2),
            'recommendation': 'OYNA' if probability > 0.65 else 'BEKLE'
        }
