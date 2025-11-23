"""
JetX Predictor - Psikolojik Analiz Sistemi

Bu modül JetX oyununun oyuncu psikolojisini manipüle etme pattern'lerini tespit eder.
Kumar oyunları oyuncuları kandırmak için çeşitli psikolojik taktikler kullanır.

GÜNCELLEME:
- Threshold Manager entegrasyonu.
- Pattern'ler 1.5x kritik eşiğine göre optimize edildi.
- 2 Modlu sisteme uygun uyarılar.
"""

import numpy as np
from typing import List, Dict, Tuple
import logging
from functools import lru_cache
from utils.threshold_manager import get_threshold_manager

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
        self.tm = get_threshold_manager()
    
    def analyze_psychological_patterns(self, history: List[float]) -> Dict[str, float]:
        """
        Tüm psikolojik pattern'leri analiz eder
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
        
        # D) Yeni Pattern Tespitleri
        features.update(self.detect_momentum_reversal(history))
        features.update(self.detect_sudden_spike(history))
        features.update(self.detect_pattern_repetition(history))
        features.update(self.detect_mean_reversion(history))
        features.update(self.detect_extreme_value_clustering(history))
        
        # E) Çoklu Zaman Dilimi Analizi
        features.update(self.multi_timeframe_analysis(history))
        
        # F) İstatistiksel İyileştirmeler
        features.update(self.calculate_statistical_indicators(history))
        
        # G) Genel Manipülasyon - Recursive çağrıyı kaldır, doğrudan hesapla
        features['manipulation_score'] = self._calculate_manipulation_score_direct(features)
        
        return features
    
    def detect_bait_and_switch(self, history: List[float]) -> Dict[str, float]:
        """
        "Bait & Switch" pattern tespiti (İyileştirilmiş)
        
        Pattern: 3-4 yüksek çarpan ile oyuncuya güven ver,
        sonra ani düşük serisi ile parayı al.
        """
        if len(history) < 10:
            return {'bait_switch_score': 0.0, 'trap_risk': 0.0}
        
        # Daha uzun pencereler kullan (30-50 el)
        window_size = min(50, len(history))
        recent_window = history[-window_size:] if len(history) >= window_size else history
        
        # Son 10 elde yüksek değerler varsa (bait)
        recent_10 = recent_window[-10:]
        high_count_recent = sum(1 for v in recent_10 if v >= 2.5)
        very_high_count = sum(1 for v in recent_10 if v >= 3.0)
        
        # Önceki 20 elde düşük değerler varsa (sonra switch geliyor olabilir)
        if len(recent_window) >= 30:
            previous_20 = recent_window[-30:-10]
            low_count_previous = sum(1 for v in previous_20 if v < self.threshold)
            
            # Bait score: Son 10 elde çok yüksek, önceki 20'de düşük
            bait_score = (high_count_recent / 10.0) * (low_count_previous / 20.0)
            
            # Çok yüksek değerler varsa risk daha yüksek
            if very_high_count >= 3:
                bait_score = min(1.0, bait_score + 0.3)
        else:
            bait_score = 0.0
        
        # Trap risk: Eğer son değerler yüksekse, sonra düşük gelebilir
        current_value = history[-1]
        if current_value >= 3.0 and high_count_recent >= 5:
            trap_risk = 0.9
        elif current_value >= 2.5 and high_count_recent >= 4:
            trap_risk = 0.7
        elif current_value >= 2.0 and high_count_recent >= 3:
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
        Pattern: Sürekli 1.5 üstü vererek oyuncuya güven ver, sonra ani düşük serisi.
        """
        if len(history) < 15:
            return {'false_confidence_score': 0.0}
        
        recent_30 = history[-30:] if len(history) >= 30 else history
        recent_15 = recent_30[-15:] if len(recent_30) >= 15 else recent_30
        recent_10 = recent_15[-10:] if len(recent_15) >= 10 else recent_15
        recent_5 = recent_10[-5:] if len(recent_10) >= 5 else recent_10
        
        # Kısa vadeli (5 el)
        above_5 = sum(1 for v in recent_5 if v >= self.threshold)
        
        # Orta vadeli (10 el)
        above_10 = sum(1 for v in recent_10 if v >= self.threshold)
        
        # Uzun vadeli (15 el)
        above_15 = sum(1 for v in recent_15 if v >= self.threshold)
        
        # Çok uzun vadeli (30 el)
        above_30 = sum(1 for v in recent_30 if v >= self.threshold) if len(recent_30) >= 30 else 0
        
        # False confidence: Çoklu zaman dilimlerinde tutarlı yüksek oran
        false_conf = 0.0
        
        if above_5 == 5: false_conf = 0.8
        elif above_5 >= 4: false_conf = 0.6
        
        if above_10 >= 8: false_conf = max(false_conf, 0.9)
        elif above_10 >= 7: false_conf = max(false_conf, 0.7)
        elif above_10 >= 6: false_conf = max(false_conf, 0.5)
        
        if above_15 >= 12: false_conf = min(1.0, false_conf + 0.2)
        
        if above_30 >= 25: false_conf = min(1.0, false_conf + 0.1)
        
        return {'false_confidence_score': float(false_conf)}
    
    def detect_heating_up(self, history: List[float]) -> Dict[str, float]:
        """Isınma (heating up) pattern tespiti"""
        if len(history) < 5:
            return {'heating_score': 0.0}
        
        recent_5 = history[-5:]
        increases = 0
        for i in range(1, len(recent_5)):
            if recent_5[i] > recent_5[i-1]:
                increases += 1
        
        heating_score = increases / (len(recent_5) - 1)
        
        value_increase = recent_5[-1] - recent_5[0]
        if value_increase > 1.0: 
            heating_score = min(1.0, heating_score + 0.3)
        
        return {'heating_score': float(heating_score)}
    
    def detect_cooling_down(self, history: List[float]) -> Dict[str, float]:
        """Soğuma (cooling down) pattern tespiti"""
        if len(history) < 5:
            return {'cooling_score': 0.0}
        
        recent_5 = history[-5:]
        decreases = 0
        for i in range(1, len(recent_5)):
            if recent_5[i] < recent_5[i-1]:
                decreases += 1
        
        cooling_score = decreases / (len(recent_5) - 1)
        
        value_decrease = recent_5[0] - recent_5[-1]
        if value_decrease > 1.0: 
            cooling_score = min(1.0, cooling_score + 0.3)
        
        return {'cooling_score': float(cooling_score)}
    
    def detect_volatility_shift(self, history: List[float]) -> Dict[str, float]:
        """Ani volatilite değişimi tespiti"""
        if len(history) < 30:
            return {'volatility_shift': 0.0}
        
        recent_10 = history[-10:]
        recent_20 = history[-20:]
        previous_10 = history[-20:-10] if len(history) >= 20 else []
        previous_20 = history[-40:-20] if len(history) >= 40 else []
        
        def calc_normalized_vol(values):
            if len(values) < 2: return 0.0
            mean_val = np.mean(values)
            if mean_val == 0: return 0.0
            return np.std(values) / mean_val
        
        recent_vol_10 = calc_normalized_vol(recent_10)
        recent_vol_20 = calc_normalized_vol(recent_20)
        
        shift_score = 0.0
        
        if len(previous_10) >= 10:
            previous_vol_10 = calc_normalized_vol(previous_10)
            if previous_vol_10 > 0:
                vol_ratio_10 = abs(recent_vol_10 - previous_vol_10) / previous_vol_10
                if vol_ratio_10 > 2.5: shift_score = 0.9
                elif vol_ratio_10 > 2.0: shift_score = 0.8
                elif vol_ratio_10 > 1.5: shift_score = 0.6
                elif vol_ratio_10 > 1.0: shift_score = 0.4
        
        if len(previous_20) >= 20:
            previous_vol_20 = calc_normalized_vol(previous_20)
            if previous_vol_20 > 0:
                vol_ratio_20 = abs(recent_vol_20 - previous_vol_20) / previous_vol_20
                if vol_ratio_20 > 2.0: shift_score = max(shift_score, 0.95)
                elif vol_ratio_20 > 1.5: shift_score = max(shift_score, 0.75)
        
        return {'volatility_shift': float(shift_score)}
    
    def calculate_desperation_score(self, history: List[float]) -> Dict[str, float]:
        """Desperation Mode tespiti"""
        if len(history) < 10:
            return {'desperation_level': 0.0}
        
        recent_10 = history[-10:]
        recent_20 = history[-20:] if len(history) >= 20 else history
        
        below_count_10 = sum(1 for v in recent_10 if v < self.threshold)
        below_count_20 = sum(1 for v in recent_20 if v < self.threshold)
        
        consecutive_losses = 0
        for i in range(len(recent_10) - 1, -1, -1):
            if recent_10[i] < self.threshold:
                consecutive_losses += 1
            else:
                break
        
        desperation = 0.0
        
        if below_count_10 >= 8: desperation = 0.9
        elif below_count_10 >= 7: desperation = 0.7
        elif below_count_10 >= 6: desperation = 0.5
        elif below_count_10 >= 5: desperation = 0.3
        
        if consecutive_losses >= 7: desperation = min(1.0, desperation + 0.3)
        elif consecutive_losses >= 5: desperation = min(1.0, desperation + 0.2)
        elif consecutive_losses >= 3: desperation = min(1.0, desperation + 0.1)
        
        if below_count_20 >= 15: desperation = min(1.0, desperation + 0.2)
        elif below_count_20 >= 12: desperation = min(1.0, desperation + 0.1)
        
        if history[-1] < self.threshold:
            desperation = min(1.0, desperation + 0.1)
        
        return {'desperation_level': float(desperation)}
    
    def calculate_gambler_fallacy_score(self, history: List[float]) -> Dict[str, float]:
        """Gambler's Fallacy skoru"""
        if len(history) < 10:
            return {'gambler_fallacy_risk': 0.0}
        
        recent_10 = history[-10:]
        recent_20 = history[-20:] if len(history) >= 20 else history
        
        current_above = recent_10[-1] >= self.threshold
        
        consecutive_same_10 = 1
        for i in range(len(recent_10) - 2, -1, -1):
            is_above = recent_10[i] >= self.threshold
            if is_above == current_above:
                consecutive_same_10 += 1
            else:
                break
        
        consecutive_same_20 = 1
        if len(recent_20) >= 20:
            for i in range(len(recent_20) - 2, -1, -1):
                is_above = recent_20[i] >= self.threshold
                if is_above == current_above:
                    consecutive_same_20 += 1
                else:
                    break
        
        alternating_count = 0
        if len(recent_10) >= 3:
            for i in range(len(recent_10) - 2):
                val1 = recent_10[i] >= self.threshold
                val2 = recent_10[i+1] >= self.threshold
                val3 = recent_10[i+2] >= self.threshold
                if val1 != val2 and val2 != val3:
                    alternating_count += 1
        
        fallacy_risk = 0.0
        
        if consecutive_same_10 >= 8: fallacy_risk = 0.95
        elif consecutive_same_10 >= 7: fallacy_risk = 0.85
        elif consecutive_same_10 >= 5: fallacy_risk = 0.7
        elif consecutive_same_10 >= 4: fallacy_risk = 0.5
        elif consecutive_same_10 >= 3: fallacy_risk = 0.3
        
        if consecutive_same_20 >= 12: fallacy_risk = max(fallacy_risk, 0.98)
        elif consecutive_same_20 >= 10: fallacy_risk = max(fallacy_risk, 0.9)
        elif consecutive_same_20 >= 8: fallacy_risk = max(fallacy_risk, 0.75)
        
        if alternating_count >= 3:
            fallacy_risk = max(0.0, fallacy_risk - 0.2)
        
        return {'gambler_fallacy_risk': float(fallacy_risk)}
    
    def detect_momentum_reversal(self, history: List[float]) -> Dict[str, float]:
        """Momentum tersine dönüş pattern'i"""
        if len(history) < 20:
            return {'momentum_reversal_score': 0.0, 'reversal_strength': 0.0}
        
        recent_20 = history[-20:]
        recent_10 = recent_20[-10:]
        previous_10 = recent_20[:10]
        
        def calc_momentum(values):
            if len(values) < 2: return 0.0
            changes = [values[i] - values[i-1] for i in range(1, len(values))]
            return np.mean(changes)
        
        momentum_prev = calc_momentum(previous_10)
        momentum_recent = calc_momentum(recent_10)
        
        reversal_score = 0.0
        reversal_strength = 0.0
        
        if momentum_prev > 0 and momentum_recent < 0:
            reversal_score = min(1.0, abs(momentum_prev) + abs(momentum_recent))
            reversal_strength = abs(momentum_prev - momentum_recent)
        elif momentum_prev < 0 and momentum_recent > 0:
            reversal_score = min(1.0, abs(momentum_prev) + abs(momentum_recent))
            reversal_strength = abs(momentum_prev - momentum_recent)
        
        return {
            'momentum_reversal_score': float(reversal_score),
            'reversal_strength': float(reversal_strength)
        }
    
    def detect_sudden_spike(self, history: List[float]) -> Dict[str, float]:
        """Ani yükseliş/düşüş tespiti"""
        if len(history) < 15:
            return {'sudden_spike_up': 0.0, 'sudden_spike_down': 0.0}
        
        recent_15 = history[-15:]
        current = history[-1]
        
        previous_10 = recent_15[:10]
        mean_prev = np.mean(previous_10)
        std_prev = np.std(previous_10)
        
        spike_up = 0.0
        spike_down = 0.0
        
        if std_prev > 0:
            z_score = (current - mean_prev) / std_prev
            
            if z_score > 3.0: spike_up = 0.9
            elif z_score > 2.5: spike_up = 0.7
            elif z_score > 2.0: spike_up = 0.5
            
            if z_score < -3.0: spike_down = 0.9
            elif z_score < -2.5: spike_down = 0.7
            elif z_score < -2.0: spike_down = 0.5
        
        return {
            'sudden_spike_up': float(spike_up),
            'sudden_spike_down': float(spike_down)
        }
    
    def detect_pattern_repetition(self, history: List[float]) -> Dict[str, float]:
        """Tekrarlayan pattern'ler tespiti"""
        if len(history) < 20:
            return {'pattern_repetition_score': 0.0, 'repetition_count': 0.0}
        
        recent_20 = history[-20:]
        pattern_length = 3
        repetition_count = 0
        max_repetitions = 0
        
        for start_idx in range(len(recent_20) - pattern_length * 2):
            pattern1 = recent_20[start_idx:start_idx + pattern_length]
            pattern2 = recent_20[start_idx + pattern_length:start_idx + pattern_length * 2]
            
            similarity = 0
            for i in range(pattern_length):
                cat1 = 1 if pattern1[i] >= self.threshold else 0
                cat2 = 1 if pattern2[i] >= self.threshold else 0
                if cat1 == cat2: similarity += 1
            
            if similarity == pattern_length:
                repetition_count += 1
                max_repetitions = max(max_repetitions, repetition_count)
            else:
                repetition_count = 0
        
        repetition_score = min(1.0, max_repetitions / 3.0)
        
        return {
            'pattern_repetition_score': float(repetition_score),
            'repetition_count': float(max_repetitions)
        }
    
    def detect_mean_reversion(self, history: List[float]) -> Dict[str, float]:
        """Ortalamaya dönüş pattern'i tespiti"""
        if len(history) < 30:
            return {'mean_reversion_score': 0.0, 'reversion_tendency': 0.0}
        
        recent_30 = history[-30:]
        long_term_mean = np.mean(recent_30)
        long_term_std = np.std(recent_30)
        
        recent_10 = recent_30[-10:]
        recent_mean = np.mean(recent_10)
        
        reversion_score = 0.0
        reversion_tendency = 0.0
        
        if long_term_std > 0:
            deviation = abs(recent_mean - long_term_mean) / long_term_std
            
            if deviation > 2.0:
                reversion_score = 0.8
                reversion_tendency = (recent_mean - long_term_mean) / long_term_std
            elif deviation > 1.5:
                reversion_score = 0.6
                reversion_tendency = (recent_mean - long_term_mean) / long_term_std
            elif deviation > 1.0:
                reversion_score = 0.4
                reversion_tendency = (recent_mean - long_term_mean) / long_term_std
        
        return {
            'mean_reversion_score': float(reversion_score),
            'reversion_tendency': float(reversion_tendency)
        }
    
    def detect_extreme_value_clustering(self, history: List[float]) -> Dict[str, float]:
        """Aşırı değer kümelenmesi tespiti"""
        if len(history) < 20:
            return {'extreme_high_cluster': 0.0, 'extreme_low_cluster': 0.0}
        
        recent_20 = history[-20:]
        mean_all = np.mean(recent_20)
        std_all = np.std(recent_20)
        
        recent_10 = recent_20[-10:]
        extreme_high = 0.0
        extreme_low = 0.0
        
        if std_all > 0:
            high_threshold = mean_all + 2 * std_all
            high_count = sum(1 for v in recent_10 if v >= high_threshold)
            
            if high_count >= 4: extreme_high = 0.9
            elif high_count >= 3: extreme_high = 0.7
            elif high_count >= 2: extreme_high = 0.5
            
            low_threshold = mean_all - 2 * std_all
            low_count = sum(1 for v in recent_10 if v <= low_threshold)
            
            if low_count >= 4: extreme_low = 0.9
            elif low_count >= 3: extreme_low = 0.7
            elif low_count >= 2: extreme_low = 0.5
        
        return {
            'extreme_high_cluster': float(extreme_high),
            'extreme_low_cluster': float(extreme_low)
        }
    
    def multi_timeframe_analysis(self, history: List[float]) -> Dict[str, float]:
        """Çoklu zaman dilimi analizi"""
        if len(history) < 50:
            return {
                'short_term_trend': 0.0, 'medium_term_trend': 0.0,
                'long_term_trend': 0.0, 'timeframe_divergence': 0.0
            }
        
        short_term = history[-10:] if len(history) >= 10 else history
        short_trend = np.mean([short_term[i] - short_term[i-1] for i in range(1, len(short_term))]) if len(short_term) > 1 else 0.0
        
        medium_term = history[-30:] if len(history) >= 30 else history
        medium_trend = np.mean([medium_term[i] - medium_term[i-1] for i in range(1, len(medium_term))]) if len(medium_term) > 1 else 0.0
        
        long_term = history[-50:] if len(history) >= 50 else history
        long_trend = np.mean([long_term[i] - long_term[i-1] for i in range(1, len(long_term))]) if len(long_term) > 1 else 0.0
        
        divergence = 0.0
        if (short_trend > 0 and medium_trend < 0) or (short_trend < 0 and medium_trend > 0):
            divergence += 0.5
        if (medium_trend > 0 and long_trend < 0) or (medium_trend < 0 and long_trend > 0):
            divergence += 0.5
        
        return {
            'short_term_trend': float(short_trend),
            'medium_term_trend': float(medium_trend),
            'long_term_trend': float(long_trend),
            'timeframe_divergence': float(divergence)
        }
    
    def calculate_statistical_indicators(self, history: List[float]) -> Dict[str, float]:
        """İstatistiksel göstergeler"""
        if len(history) < 30:
            return {'z_score_current': 0.0, 'macd_signal': 0.0, 'rsi_momentum': 0.0}
        
        recent_30 = history[-30:]
        current = history[-1]
        
        mean_30 = np.mean(recent_30)
        std_30 = np.std(recent_30)
        z_score = (current - mean_30) / std_30 if std_30 > 0 else 0.0
        
        if len(history) >= 26:
            ma_12 = np.mean(history[-12:])
            ma_26 = np.mean(history[-26:])
            macd = ma_12 - ma_26
            macd_signal = min(1.0, max(-1.0, macd / (mean_30 + 1e-8)))
        else:
            macd_signal = 0.0
        
        recent_14 = history[-14:] if len(history) >= 14 else history
        gains = [max(0, recent_14[i] - recent_14[i-1]) for i in range(1, len(recent_14))]
        losses = [max(0, recent_14[i-1] - recent_14[i]) for i in range(1, len(recent_14))]
        
        avg_gain = np.mean(gains) if gains else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            rsi_momentum = (rsi - 50) / 50
        else:
            rsi_momentum = 1.0 if avg_gain > 0 else 0.0
        
        return {
            'z_score_current': float(z_score),
            'macd_signal': float(macd_signal),
            'rsi_momentum': float(rsi_momentum)
        }
    
    def _calculate_manipulation_score_direct(self, features: Dict[str, float]) -> float:
        """Manipülasyon skorunu doğrudan hesapla"""
        weights = {
            'bait_switch_score': 0.15,
            'trap_risk': 0.12,
            'false_confidence_score': 0.12,
            'volatility_shift': 0.10,
            'desperation_level': 0.08,
            'gambler_fallacy_risk': 0.08,
            'momentum_reversal_score': 0.08,
            'sudden_spike_up': 0.05,
            'sudden_spike_down': 0.05,
            'pattern_repetition_score': 0.07,
            'extreme_high_cluster': 0.05,
            'extreme_low_cluster': 0.05
        }
        
        manipulation_score = 0.0
        total_weight = 0.0
        
        for feature, weight in weights.items():
            if feature in features:
                value = features[feature]
                manipulation_score += value * weight
                total_weight += weight
        
        if total_weight > 0:
            manipulation_score = manipulation_score / total_weight
        
        return float(manipulation_score)
    
    def detect_manipulation_pattern(self, history: List[float]) -> float:
        """Genel manipülasyon pattern skoru"""
        if len(history) < 20:
            return 0.0
        
        features = self.analyze_psychological_patterns(history)
        return self._calculate_manipulation_score_direct(features)
    
    def _get_default_features(self) -> Dict[str, float]:
        """Default özellikler"""
        return {
            'bait_switch_score': 0.0, 'trap_risk': 0.0, 'false_confidence_score': 0.0,
            'heating_score': 0.0, 'cooling_score': 0.0, 'volatility_shift': 0.0,
            'desperation_level': 0.0, 'gambler_fallacy_risk': 0.0, 'momentum_reversal_score': 0.0,
            'reversal_strength': 0.0, 'sudden_spike_up': 0.0, 'sudden_spike_down': 0.0,
            'pattern_repetition_score': 0.0, 'repetition_count': 0.0, 'mean_reversion_score': 0.0,
            'reversion_tendency': 0.0, 'extreme_high_cluster': 0.0, 'extreme_low_cluster': 0.0,
            'short_term_trend': 0.0, 'medium_term_trend': 0.0, 'long_term_trend': 0.0,
            'timeframe_divergence': 0.0, 'z_score_current': 0.0, 'macd_signal': 0.0,
            'rsi_momentum': 0.0, 'manipulation_score': 0.0
        }
    
    def get_psychological_warning(self, features: Dict[str, float]) -> str:
        """Psikolojik uyarı mesajı"""
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
    return PsychologicalAnalyzer(threshold=threshold)
