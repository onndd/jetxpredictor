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

# Kategori tanımlarını import et
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from category_definitions import (
    CategoryDefinitions,
    FeatureEngineering,
    CONFIDENCE_THRESHOLDS
)


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
                self.model = keras.models.load_model(self.model_path)
                print(f"✅ Model yüklendi: {self.model_path}")
            except:
                # PyTorch veya sklearn modeli için
                import joblib
                self.model = joblib.load(self.model_path)
                print(f"✅ Model yüklendi: {self.model_path}")
            
            # Scaler'ı yükle
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                print(f"✅ Scaler yüklendi: {self.scaler_path}")
            
        except Exception as e:
            print(f"⚠️ Model yükleme hatası: {e}")
            print("Model henüz eğitilmemiş olabilir. Önce Google Colab'da eğitin.")
    
    def extract_features_from_history(self, history: List[float]) -> np.ndarray:
        """
        Geçmiş verilerden özellik çıkarır
        
        Args:
            history: Geçmiş değerler listesi
            
        Returns:
            Feature array
        """
        # Tüm özellikleri çıkar
        features_dict = FeatureEngineering.extract_all_features(history)
        
        # Dictionary'yi array'e çevir
        feature_values = np.array(list(features_dict.values())).reshape(1, -1)
        
        # Scaler varsa normalize et
        if self.scaler is not None:
            feature_values = self.scaler.transform(feature_values)
        
        return feature_values
    
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
        
        if len(history) < 50:
            return {
                'error': 'En az 50 geçmiş veri gerekli',
                'predicted_value': None,
                'confidence': 0.0,
                'above_threshold': None,
                'category': None,
                'recommendation': 'BEKLE'
            }
        
        try:
            # Özellikleri çıkar
            features = self.extract_features_from_history(history)
            
            # Tahmin yap
            prediction = self.model.predict(features, verbose=0)
            
            # Tahmin edilen değer
            predicted_value = float(prediction[0][0]) if len(prediction[0]) == 1 else float(prediction[0])
            
            # Güven skorunu hesapla (model çıktısına göre ayarlanabilir)
            confidence = self._calculate_confidence(history, predicted_value)
            
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
                'category': category,
                'detailed_category': detailed_category,
                'recommendation': recommendation,
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
                'recommendation': 'BEKLE'
            }
    
    def _calculate_confidence(
        self,
        history: List[float],
        predicted_value: float
    ) -> float:
        """
        Güven skorunu hesaplar
        
        Args:
            history: Geçmiş veriler
            predicted_value: Tahmin edilen değer
            
        Returns:
            Güven skoru (0-1 arası)
        """
        # Basit bir güven hesaplama
        # Gerçek uygulamada model çıktısından alınabilir veya daha karmaşık olabilir
        
        confidence = 0.65  # Başlangıç değeri
        
        # Son değerlerin volatilitesine göre ayarla
        if len(history) >= 10:
            recent = history[-10:]
            volatility = np.std(recent)
            
            # Düşük volatilite = yüksek güven
            if volatility < 2.0:
                confidence += 0.15
            elif volatility > 5.0:
                confidence -= 0.15
        
        # Tahmin edilen değer makul aralıkta mı?
        if 1.0 <= predicted_value <= 10.0:
            confidence += 0.10
        elif predicted_value > 50.0:
            confidence -= 0.20
        
        # Kritik bölgede ise güveni azalt
        if 1.45 <= predicted_value <= 1.55:
            confidence -= 0.10
        
        # 0-1 aralığında tut
        return max(0.0, min(1.0, confidence))
    
    def _get_recommendation(
        self,
        confidence: float,
        mode: str,
        above_threshold: bool
    ) -> str:
        """
        Mod bazlı öneri verir
        
        Args:
            confidence: Güven skoru
            mode: Tahmin modu
            above_threshold: 1.5x üstü mü
            
        Returns:
            Öneri ('OYNA', 'BEKLE', 'RİSKLİ')
        """
        threshold = CONFIDENCE_THRESHOLDS.get(mode, 0.65)
        
        if confidence < threshold:
            return 'BEKLE'
        
        if not above_threshold:
            return 'BEKLE'  # 1.5x altı tahmin ediliyorsa bekle
        
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
        Uyarılar oluşturur
        
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
        
        # Büyük çarpan sonrası soğuma
        if len(history) >= 10:
            recent = history[-10:]
            if max(recent) > 10.0:
                warnings.append("❄️ Son 10 elde büyük çarpan var - Soğuma dönemi olabilir")
        
        # Yüksek volatilite
        if len(history) >= 10:
            recent = history[-10:]
            if np.std(recent) > 5.0:
                warnings.append("📊 Yüksek volatilite tespit edildi")
        
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
