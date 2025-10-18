"""
JetX Predictor - Psikolojik Analiz Sistemi

Bu modül JetX oyununun oyuncu psikolojisini manipüle etme pattern'lerini tespit eder.
Kumar oyunları oyuncuları kandırmak için çeşitli psikolojik taktikler kullanır.
"""

import numpy as np
from typing import List, Dict, Tuple
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


class PsychologicalAnalyzer:
    """
    Psikolojik pattern analiz sınıfı
    
    JetX gibi kumar oyunlarının oyuncu davranışlarını manipüle etme
    pattern'lerini tespit eder ve risk skorları hesaplar.
    """
    
    def __init__(self, threshold: float = 1.5):
        """
        Args:
            threshold: Kritik eşik değeri (default: 1.5)
        """
        self.threshold = threshold
    
    def analyze_psychological_patterns(self, history: List[float]) -> Dict[str, float]:
        """
        Tüm psikolojik pattern'leri analiz eder
        
        Args:
            history: Geçmiş değerler listesi (en yeni en sonda)
            
        Returns:
            Psikolojik özellikler dictionary'si
        """
        if len(history) < 20:
            # Yetersiz veri - default değerler
            return self._get_default_features()
        
        features = {}
        
        # A) Oyuncu Tuzakları
        features.update(self.detect_bait_and_switch(history))
        features.update(self.detect_false_confidence_pattern(history))
        
        # B) Soğuma/Heating Analizi
        features.update(self.detect_heating_up(history))
        features.update(self.detect_cooling_down(history))
        features.update(self.detect_volatility_shift(history))
        
        # C) Gambler's Fallacy
        features.update(self.calculate_desperation_score(history))
        features.update(self.calculate_gambler_fallacy_score(history))
        
        # D) Genel Manipülasyon - Recursive çağrıyı kaldır, doğrudan hesapla
        features['manipulation_score'] = self._calculate_manipulation_score_direct(features)
        
        return features
    
    def detect_bait_and_switch(self, history: List[float]) -> Dict[str, float]:
        """
        "Bait & Switch" pattern tespiti
        
        Pattern: 3-4 yüksek çarpan ile oyuncuya güven ver,
        sonra ani düşük serisi ile parayı al.
        
        Args:
            history: Geçmiş değerler
            
        Returns:
            Bait & Switch skorları
        """
        if len(history) < 10:
            return {'bait_switch_score': 0.0, 'trap_risk': 0.0}
        
        recent_20 = history[-20:] if len(history) >= 20 else history
        
        # Son 5-10 elde yüksek değerler varsa (bait)
        recent_5 = recent_20[-5:]
        high_count_recent = sum(1 for v in recent_5 if v >= 2.5)
        
        # Önceki 10 elde düşük değerler varsa (sonra switch geliyor olabilir)
        if len(recent_20) >= 15:
            previous_10 = recent_20[-15:-5]
            low_count_previous = sum(1 for v in previous_10 if v < self.threshold)
            
            # Bait score: Son 5 elde çok yüksek, önceki 10'da düşük
            bait_score = (high_count_recent / 5.0) * (low_count_previous / 10.0)
        else:
            bait_score = 0.0
        
        # Trap risk: Eğer son değerler yüksekse, sonra düşük gelebilir
        current_value = history[-1]
        if current_value >= 3.0 and high_count_recent >= 3:
            trap_risk = 0.8
        elif current_value >= 2.0 and high_count_recent >= 2:
            trap_risk = 0.5
        else:
            trap_risk = 0.0
        
        return {
            'bait_switch_score': float(bait_score),
            'trap_risk': float(trap_risk)
        }
    
    def detect_false_confidence_pattern(self, history: List[float]) -> Dict[str, float]:
        """
        Yanlış güven verme pattern'i tespiti
        
        Pattern: Sürekli 1.5 üstü vererek oyuncuya güven ver,
        sonra ani düşük serisi ile şok et.
        
        Args:
            history: Geçmiş değerler
            
        Returns:
            False confidence skorları
        """
        if len(history) < 15:
            return {'false_confidence_score': 0.0}
        
        recent_15 = history[-15:]
        
        # Son 10 elde kaç tanesi 1.5 üstü
        recent_10 = recent_15[-10:]
        above_count = sum(1 for v in recent_10 if v >= self.threshold)
        
        # Son 5 elde kaç tanesi 1.5 üstü
        recent_5 = recent_15[-5:]
        above_recent_5 = sum(1 for v in recent_5 if v >= self.threshold)
        
        # False confidence: Son 10'da çok fazla 1.5 üstü
        # Bu güven verdikten sonra düşük gelebilir
        if above_count >= 8:  # 10 elin 8'i üstü - çok yüksek
            false_conf = 0.9
        elif above_count >= 7:  # 10 elin 7'si üstü
            false_conf = 0.7
        elif above_count >= 6:  # 10 elin 6'sı üstü
            false_conf = 0.5
        else:
            false_conf = 0.0
        
        # Eğer son 5'te de hepsi üstü ise risk daha yüksek
        if above_recent_5 == 5:
            false_conf = min(1.0, false_conf + 0.2)
        
        return {'false_confidence_score': float(false_conf)}
    
    def detect_heating_up(self, history: List[float]) -> Dict[str, float]:
        """
        Isınma (heating up) pattern tespiti
        
        Pattern: Yavaş yavaş değerler artıyor (1.2 → 1.4 → 1.7 → 2.0)
        
        Args:
            history: Geçmiş değerler
            
        Returns:
            Heating up skoru
        """
        if len(history) < 5:
            return {'heating_score': 0.0}
        
        recent_5 = history[-5:]
        
        # Ardışık artışları say
        increases = 0
        for i in range(1, len(recent_5)):
            if recent_5[i] > recent_5[i-1]:
                increases += 1
        
        # Heating score: Ardışık artış oranı
        heating_score = increases / (len(recent_5) - 1)
        
        # Eğer değerler gerçekten artıyorsa (sadece direction değil)
        # İlk ve son değer arasındaki fark
        value_increase = recent_5[-1] - recent_5[0]
        if value_increase > 1.0:  # En az 1.0x artış
            heating_score = min(1.0, heating_score + 0.3)
        
        return {'heating_score': float(heating_score)}
    
    def detect_cooling_down(self, history: List[float]) -> Dict[str, float]:
        """
        Soğuma (cooling down) pattern tespiti
        
        Pattern: Yavaş yavaş değerler düşüyor
        
        Args:
            history: Geçmiş değerler
            
        Returns:
            Cooling down skoru
        """
        if len(history) < 5:
            return {'cooling_score': 0.0}
        
        recent_5 = history[-5:]
        
        # Ardışık düşüşleri say
        decreases = 0
        for i in range(1, len(recent_5)):
            if recent_5[i] < recent_5[i-1]:
                decreases += 1
        
        # Cooling score
        cooling_score = decreases / (len(recent_5) - 1)
        
        # Eğer değerler gerçekten düşüyorsa
        value_decrease = recent_5[0] - recent_5[-1]
        if value_decrease > 1.0:  # En az 1.0x düşüş
            cooling_score = min(1.0, cooling_score + 0.3)
        
        return {'cooling_score': float(cooling_score)}
    
    def detect_volatility_shift(self, history: List[float]) -> Dict[str, float]:
        """
        Ani volatilite değişimi tespiti
        
        Pattern: Volatilite ani değişirse manipülasyon olabilir
        
        Args:
            history: Geçmiş değerler
            
        Returns:
            Volatility shift skoru
        """
        if len(history) < 20:
            return {'volatility_shift': 0.0}
        
        # Son 10 elin volatilitesi
        recent_10 = history[-10:]
        recent_vol = np.std(recent_10)
        
        # Önceki 10 elin volatilitesi
        previous_10 = history[-20:-10]
        previous_vol = np.std(previous_10)
        
        # Volatilite farkı
        if previous_vol > 0:
            vol_ratio = abs(recent_vol - previous_vol) / previous_vol
            
            # Eğer volatilite 2x değiştiyse, manipülasyon olabilir
            if vol_ratio > 2.0:
                shift_score = 0.9
            elif vol_ratio > 1.5:
                shift_score = 0.7
            elif vol_ratio > 1.0:
                shift_score = 0.5
            else:
                shift_score = 0.0
        else:
            shift_score = 0.0
        
        return {'volatility_shift': float(shift_score)}
    
    def calculate_desperation_score(self, history: List[float]) -> Dict[str, float]:
        """
        "Desperation Mode" tespiti
        
        Pattern: Ardışık kayıptan sonra oyuncu umutsuzlaşır,
        "büyük çarpan gelmeli" illüzyonuna kapılır
        
        Args:
            history: Geçmiş değerler
            
        Returns:
            Desperation skoru
        """
        if len(history) < 10:
            return {'desperation_level': 0.0}
        
        recent_10 = history[-10:]
        
        # Son 10 elde kaç tanesi threshold altı
        below_count = sum(1 for v in recent_10 if v < self.threshold)
        
        # Desperation: Çok fazla düşük değer
        if below_count >= 8:  # 10 elin 8'i altı
            desperation = 0.9
        elif below_count >= 7:
            desperation = 0.7
        elif below_count >= 6:
            desperation = 0.5
        elif below_count >= 5:
            desperation = 0.3
        else:
            desperation = 0.0
        
        # Eğer son değer de düşükse, desperation daha yüksek
        if history[-1] < self.threshold:
            desperation = min(1.0, desperation + 0.1)
        
        return {'desperation_level': float(desperation)}
    
    def calculate_gambler_fallacy_score(self, history: List[float]) -> Dict[str, float]:
        """
        Gambler's Fallacy (Kumar Yanılgısı) skoru
        
        "5 kez düşük geldi, artık yüksek gelmeli" düşüncesi
        
        Args:
            history: Geçmiş değerler
            
        Returns:
            Gambler's fallacy risk skoru
        """
        if len(history) < 10:
            return {'gambler_fallacy_risk': 0.0}
        
        recent_10 = history[-10:]
        
        # Ardışık aynı yön (üstü veya altı) sayısı
        current_above = recent_10[-1] >= self.threshold
        
        consecutive_same = 1
        for i in range(len(recent_10) - 2, -1, -1):
            is_above = recent_10[i] >= self.threshold
            if is_above == current_above:
                consecutive_same += 1
            else:
                break
        
        # Gambler's fallacy riski: Uzun streak varsa
        if consecutive_same >= 7:
            fallacy_risk = 0.9
        elif consecutive_same >= 5:
            fallacy_risk = 0.7
        elif consecutive_same >= 4:
            fallacy_risk = 0.5
        elif consecutive_same >= 3:
            fallacy_risk = 0.3
        else:
            fallacy_risk = 0.0
        
        return {'gambler_fallacy_risk': float(fallacy_risk)}
    
    def _calculate_manipulation_score_direct(self, features: Dict[str, float]) -> float:
        """
        Manipülasyon skorunu doğrudan hesapla (recursive çağrı olmadan)
        
        Args:
            features: Zaten hesaplanmış psikolojik özellikler
            
        Returns:
            Genel manipülasyon skoru (0-1)
        """
        # Ağırlıklı ortalama
        weights = {
            'bait_switch_score': 0.25,
            'trap_risk': 0.20,
            'false_confidence_score': 0.20,
            'volatility_shift': 0.15,
            'desperation_level': 0.10,
            'gambler_fallacy_risk': 0.10
        }
        
        manipulation_score = 0.0
        for feature, weight in weights.items():
            if feature in features:
                manipulation_score += features[feature] * weight
        
        return float(manipulation_score)
    
    def detect_manipulation_pattern(self, history: List[float]) -> float:
        """
        Genel manipülasyon pattern skoru
        
        Tüm psikolojik faktörleri birleştiren genel skor
        
        Args:
            history: Geçmiş değerler
            
        Returns:
            Genel manipülasyon skoru (0-1)
        """
        if len(history) < 20:
            return 0.0
        
        # Tüm skorları al (recursive değil, direct)
        features = self.analyze_psychological_patterns(history)
        
        # Manipülasyon skorunu doğrudan hesapla
        return self._calculate_manipulation_score_direct(features)
    
    def _get_default_features(self) -> Dict[str, float]:
        """Default özellikler (yetersiz veri durumunda)"""
        return {
            'bait_switch_score': 0.0,
            'trap_risk': 0.0,
            'false_confidence_score': 0.0,
            'heating_score': 0.0,
            'cooling_score': 0.0,
            'volatility_shift': 0.0,
            'desperation_level': 0.0,
            'gambler_fallacy_risk': 0.0,
            'manipulation_score': 0.0
        }
    
    def get_psychological_warning(self, features: Dict[str, float]) -> str:
        """
        Psikolojik analiz sonucuna göre uyarı mesajı
        
        Args:
            features: Psikolojik özellikler
            
        Returns:
            Uyarı mesajı
        """
        warnings = []
        
        if features.get('trap_risk', 0) > 0.7:
            warnings.append("🚨 TUZAK RİSKİ YÜKSEK! Son değerler çok yüksek, ani düşüş gelebilir.")
        
        if features.get('false_confidence_score', 0) > 0.7:
            warnings.append("⚠️ Yanlış güven pattern'i! Sürekli 1.5 üstü geldi, dikkatli olun.")
        
        if features.get('desperation_level', 0) > 0.7:
            warnings.append("❄️ Çok fazla düşük değer! Umutsuzluğa kapılmayın.")
        
        if features.get('gambler_fallacy_risk', 0) > 0.7:
            warnings.append("🎲 Kumar yanılgısı riski! Ardışık pattern devam etmeyebilir.")
        
        if features.get('manipulation_score', 0) > 0.7:
            warnings.append("🎭 YÜKSEK MANİPÜLASYON RİSKİ! Oyun sizi kandırmaya çalışıyor olabilir.")
        
        if not warnings:
            return "✅ Psikolojik pattern'ler normal görünüyor."
        
        return " | ".join(warnings)


# Factory function
def create_psychological_analyzer(threshold: float = 1.5) -> PsychologicalAnalyzer:
    """
    Psikolojik analiz modülü oluştur
    
    Args:
        threshold: Kritik eşik değeri
        
    Returns:
        PsychologicalAnalyzer instance
    """
    return PsychologicalAnalyzer(threshold=threshold)
