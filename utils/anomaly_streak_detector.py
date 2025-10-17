"""
JetX Predictor - Anormal Streak (Ardışıklık) Tespit Sistemi

Bu modül anormal ardışıklık pattern'lerini tespit eder.
Örnek: 10 el üst üste 1.5 üstü, 6-7 el ardışık 1.5 altı, vb.
"""

import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class AnomalyStreakDetector:
    """
    Anormal ardışıklık pattern'lerini tespit eden sınıf
    
    Tespit edilen pattern'ler:
    - Uzun ardışık 1.5 üstü serileri
    - Uzun ardışık 1.5 altı serileri
    - Anormal değişim pattern'leri
    - Extreme streak göstergeleri
    """
    
    def __init__(self, threshold: float = 1.5):
        """
        Args:
            threshold: Kritik eşik değeri (default: 1.5)
        """
        self.threshold = threshold
    
    def extract_streak_features(self, history: List[float]) -> Dict[str, float]:
        """
        Tüm streak özelliklerini çıkarır
        
        Args:
            history: Geçmiş değerler listesi (en yeni en sonda)
            
        Returns:
            Streak özellikleri dictionary'si
        """
        if len(history) < 5:
            return self._get_default_features()
        
        features = {}
        
        # 1) Mevcut streak analizi
        features.update(self._analyze_current_streak(history))
        
        # 2) Maksimum streak analizi (son 10, 20, 50 el)
        features.update(self._analyze_max_streaks(history))
        
        # 3) Extreme streak göstergeleri
        features.update(self._analyze_extreme_streaks(history))
        
        # 4) Streak kırılma olasılığı
        features['streak_break_probability'] = self._calculate_break_probability(history)
        
        # 5) Alternating pattern (dalgalı hareket)
        features.update(self._analyze_alternating_pattern(history))
        
        return features
    
    def _analyze_current_streak(self, history: List[float]) -> Dict[str, float]:
        """
        Mevcut streak'i analiz eder
        
        Returns:
            Current streak özellikleri
        """
        current_value = history[-1]
        current_above = current_value >= self.threshold
        
        # Geriden başlayarak mevcut streak'i say
        current_streak = 1
        for i in range(len(history) - 2, -1, -1):
            is_above = history[i] >= self.threshold
            if is_above == current_above:
                current_streak += 1
            else:
                break
        
        # Above veya below streak
        if current_above:
            current_above_streak = current_streak
            current_below_streak = 0
        else:
            current_above_streak = 0
            current_below_streak = current_streak
        
        return {
            'current_above_streak': float(current_above_streak),
            'current_below_streak': float(current_below_streak),
            'current_streak_length': float(current_streak)
        }
    
    def _analyze_max_streaks(self, history: List[float]) -> Dict[str, float]:
        """
        Farklı zaman dilimlerinde maksimum streak'leri analiz eder
        
        Returns:
            Maximum streak özellikleri
        """
        features = {}
        
        # Farklı pencere boyutları
        windows = {
            '10': 10,
            '20': 20,
            '50': 50
        }
        
        for name, size in windows.items():
            if len(history) >= size:
                recent = history[-size:]
                
                # Above threshold max streak
                max_above = self._find_max_streak(recent, lambda x: x >= self.threshold)
                features[f'max_above_streak_{name}'] = float(max_above)
                
                # Below threshold max streak
                max_below = self._find_max_streak(recent, lambda x: x < self.threshold)
                features[f'max_below_streak_{name}'] = float(max_below)
            else:
                features[f'max_above_streak_{name}'] = 0.0
                features[f'max_below_streak_{name}'] = 0.0
        
        return features
    
    def _find_max_streak(self, values: List[float], condition) -> int:
        """
        Bir condition için maksimum streak'i bulur
        
        Args:
            values: Değerler listesi
            condition: Lambda fonksiyonu (değer -> bool)
            
        Returns:
            Maksimum streak uzunluğu
        """
        max_streak = 0
        current_streak = 0
        
        for value in values:
            if condition(value):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _analyze_extreme_streaks(self, history: List[float]) -> Dict[str, float]:
        """
        Extreme (aşırı uzun) streak'leri tespit eder
        
        Returns:
            Extreme streak göstergeleri
        """
        if len(history) < 10:
            return {
                'has_extreme_above_streak': 0.0,
                'has_extreme_below_streak': 0.0,
                'extreme_streak_risk': 0.0
            }
        
        recent_20 = history[-20:] if len(history) >= 20 else history
        
        # Above threshold max streak
        max_above = self._find_max_streak(recent_20, lambda x: x >= self.threshold)
        
        # Below threshold max streak
        max_below = self._find_max_streak(recent_20, lambda x: x < self.threshold)
        
        # Extreme tanımı: 5+ ardışık
        has_extreme_above = 1.0 if max_above >= 5 else 0.0
        has_extreme_below = 1.0 if max_below >= 5 else 0.0
        
        # Extreme streak risk: Uzun streak'ler genellikle kırılır
        # 7+ streak varsa risk çok yüksek
        # 5-6 streak varsa risk yüksek
        if max_above >= 7 or max_below >= 7:
            extreme_risk = 0.9
        elif max_above >= 5 or max_below >= 5:
            extreme_risk = 0.7
        elif max_above >= 4 or max_below >= 4:
            extreme_risk = 0.5
        else:
            extreme_risk = 0.0
        
        return {
            'has_extreme_above_streak': has_extreme_above,
            'has_extreme_below_streak': has_extreme_below,
            'extreme_streak_risk': extreme_risk
        }
    
    def _calculate_break_probability(self, history: List[float]) -> float:
        """
        Mevcut streak'in kırılma olasılığını hesaplar
        
        Mantık: Uzun streak'ler daha kolay kırılır
        
        Returns:
            Kırılma olasılığı (0-1)
        """
        if len(history) < 2:
            return 0.5  # Belirsiz
        
        # Mevcut streak uzunluğu
        current_value = history[-1]
        current_above = current_value >= self.threshold
        
        current_streak = 1
        for i in range(len(history) - 2, -1, -1):
            is_above = history[i] >= self.threshold
            if is_above == current_above:
                current_streak += 1
            else:
                break
        
        # Streak ne kadar uzunsa, kırılma olasılığı o kadar yüksek
        # Formül: sigmoid-like function
        if current_streak >= 10:
            break_prob = 0.95
        elif current_streak >= 7:
            break_prob = 0.85
        elif current_streak >= 5:
            break_prob = 0.70
        elif current_streak >= 4:
            break_prob = 0.60
        elif current_streak >= 3:
            break_prob = 0.50
        else:
            break_prob = 0.30
        
        return float(break_prob)
    
    def _analyze_alternating_pattern(self, history: List[float]) -> Dict[str, float]:
        """
        Dalgalı hareket pattern'ini tespit eder
        
        Pattern: 1.5 üstü, 1.5 altı, 1.5 üstü, 1.5 altı... (alternating)
        
        Returns:
            Alternating pattern özellikleri
        """
        if len(history) < 5:
            return {
                'alternating_pattern_score': 0.0,
                'is_alternating': 0.0
            }
        
        recent_10 = history[-10:] if len(history) >= 10 else history[-5:]
        
        # Alternating sayısını say
        alternations = 0
        for i in range(1, len(recent_10)):
            prev_above = recent_10[i-1] >= self.threshold
            curr_above = recent_10[i] >= self.threshold
            if prev_above != curr_above:
                alternations += 1
        
        # Alternating score: Değişim oranı
        alternating_score = alternations / (len(recent_10) - 1) if len(recent_10) > 1 else 0.0
        
        # Eğer %70+ alternating ise, bu bir pattern
        is_alternating = 1.0 if alternating_score >= 0.7 else 0.0
        
        return {
            'alternating_pattern_score': float(alternating_score),
            'is_alternating': is_alternating
        }
    
    def _get_default_features(self) -> Dict[str, float]:
        """Default özellikler (yetersiz veri durumunda)"""
        return {
            'current_above_streak': 0.0,
            'current_below_streak': 0.0,
            'current_streak_length': 0.0,
            'max_above_streak_10': 0.0,
            'max_below_streak_10': 0.0,
            'max_above_streak_20': 0.0,
            'max_below_streak_20': 0.0,
            'max_above_streak_50': 0.0,
            'max_below_streak_50': 0.0,
            'has_extreme_above_streak': 0.0,
            'has_extreme_below_streak': 0.0,
            'extreme_streak_risk': 0.0,
            'streak_break_probability': 0.5,
            'alternating_pattern_score': 0.0,
            'is_alternating': 0.0
        }
    
    def get_streak_warning(self, features: Dict[str, float]) -> str:
        """
        Streak analiz sonucuna göre uyarı mesajı
        
        Args:
            features: Streak özellikleri
            
        Returns:
            Uyarı mesajı
        """
        warnings = []
        
        # Extreme streak uyarısı
        if features.get('has_extreme_above_streak', 0) > 0:
            current_above = int(features.get('current_above_streak', 0))
            if current_above >= 5:
                warnings.append(f"🔥 EXTREME STREAK! {current_above} ardışık 1.5 üstü! Kırılma riski yüksek.")
        
        if features.get('has_extreme_below_streak', 0) > 0:
            current_below = int(features.get('current_below_streak', 0))
            if current_below >= 5:
                warnings.append(f"❄️ EXTREME STREAK! {current_below} ardışık 1.5 altı! Toparlanma gelebilir.")
        
        # Kırılma olasılığı uyarısı
        break_prob = features.get('streak_break_probability', 0)
        if break_prob > 0.7:
            warnings.append(f"⚠️ Mevcut streak kırılma olasılığı yüksek: {break_prob:.0%}")
        
        # Alternating pattern uyarısı
        if features.get('is_alternating', 0) > 0:
            warnings.append("🔄 Dalgalı hareket pattern'i tespit edildi! (1.5 üstü/altı alternating)")
        
        if not warnings:
            return "✅ Streak pattern'leri normal görünüyor."
        
        return " | ".join(warnings)
    
    def detect_anomaly_streaks(self, history: List[float]) -> Dict[str, any]:
        """
        Ana analiz fonksiyonu - tüm anormal streak'leri tespit eder
        
        Args:
            history: Geçmiş değerler listesi
            
        Returns:
            Detaylı analiz raporu
        """
        features = self.extract_streak_features(history)
        warning = self.get_streak_warning(features)
        
        return {
            'features': features,
            'warning': warning,
            'has_anomaly': any([
                features.get('has_extreme_above_streak', 0) > 0,
                features.get('has_extreme_below_streak', 0) > 0,
                features.get('extreme_streak_risk', 0) > 0.7
            ])
        }


# Factory function
def create_anomaly_streak_detector(threshold: float = 1.5) -> AnomalyStreakDetector:
    """
    Anomaly streak detector oluştur
    
    Args:
        threshold: Kritik eşik değeri
        
    Returns:
        AnomalyStreakDetector instance
    """
    return AnomalyStreakDetector(threshold=threshold)
