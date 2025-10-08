"""
JetX Predictor - Kategori Tanımları ve Özellik Çıkarma Fonksiyonları

Bu dosya hem Google Colab'da hem de lokal Streamlit uygulamasında kullanılacak.
Kategori tanımları ve özellik çıkarma fonksiyonlarını içerir.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


class CategoryDefinitions:
    """
    JetX için kategori tanımları.
    3 ana kategori seti: Basit, etkili ve anlaşılır
    """
    
    # Ana kategori eşikleri (1.5x kritik eşik)
    CRITICAL_THRESHOLD = 1.5
    HIGH_MULTIPLIER_THRESHOLD = 10.0  # Gerçek yüksek çarpan eşiği
    
    # 3 Ana Kategori
    CATEGORIES = {
        'LOSS_ZONE': 'Kayıp Bölgesi (< 1.5x)',
        'SAFE_ZONE': 'Güvenli Bölge (1.5x - 3.0x)',
        'HIGH_ZONE': 'Yüksek Çarpan (> 3.0x)'
    }
    
    @staticmethod
    def get_category(value: float) -> str:
        """Değeri kategoriye ata"""
        if value < CategoryDefinitions.CRITICAL_THRESHOLD:
            return 'LOSS_ZONE'
        elif value < CategoryDefinitions.HIGH_MULTIPLIER_THRESHOLD:
            return 'SAFE_ZONE'
        else:
            return 'HIGH_ZONE'
    
    @staticmethod
    def get_category_numeric(value: float) -> int:
        """Değeri sayısal kategoriye ata (0, 1, 2)"""
        if value < CategoryDefinitions.CRITICAL_THRESHOLD:
            return 0  # Kayıp
        elif value < CategoryDefinitions.HIGH_MULTIPLIER_THRESHOLD:
            return 1  # Güvenli
        else:
            return 2  # Yüksek
    
    @staticmethod
    def is_above_threshold(value: float) -> bool:
        """1.5x eşiğinin üstünde mi?"""
        return value >= CategoryDefinitions.CRITICAL_THRESHOLD
    
    @staticmethod
    def get_detailed_category(value: float) -> str:
        """Daha detaylı kategori (görselleştirme için)"""
        if value < 1.2:
            return 'Çok Düşük (< 1.2x)'
        elif value < 1.35:
            return 'Düşük (1.2x - 1.35x)'
        elif value < 1.5:
            return 'KRİTİK RİSK (1.35x - 1.49x)'
        elif value < 1.7:
            return 'Güvenli Başlangıç (1.5x - 1.7x)'
        elif value < 2.0:
            return 'İyi (1.7x - 2.0x)'
        elif value < 3.0:
            return 'Orta Yüksek (2.0x - 3.0x)'
        elif value < 5.0:
            return 'Yüksek (3.0x - 5.0x)'
        elif value < 10.0:
            return 'Çok Yüksek (5.0x - 10.0x)'
        elif value < 20.0:
            return 'Nadir (10.0x - 20.0x)'
        elif value < 50.0:
            return 'Çok Nadir (20.0x - 50.0x)'
        elif value < 100.0:
            return 'Mega (50.0x - 100.0x)'
        elif value < 200.0:
            return 'Süper Mega (100.0x - 200.0x)'
        else:
            return 'Ultra (200.0x+)'


class FeatureEngineering:
    """
    Özellik çıkarma fonksiyonları.
    Hem Colab'da eğitim sırasında hem de lokalde tahmin sırasında kullanılacak.
    """
    
    @staticmethod
    def extract_basic_features(values: List[float], window_sizes: List[int] = [25, 50, 100, 200, 500]) -> Dict[str, float]:
        """
        Temel özellikler: Ortalama, std, min, max
        
        Args:
            values: Geçmiş değerler listesi (en yeni en sonda)
            window_sizes: Pencere boyutları
            
        Returns:
            Özellik sözlüğü
        """
        features = {}
        
        for window in window_sizes:
            if len(values) >= window:
                recent = values[-window:]
                features[f'mean_{window}'] = np.mean(recent)
                features[f'std_{window}'] = np.std(recent)
                features[f'min_{window}'] = np.min(recent)
                features[f'max_{window}'] = np.max(recent)
                features[f'median_{window}'] = np.median(recent)
        
        return features
    
    @staticmethod
    def extract_threshold_features(values: List[float], threshold: float = 1.5) -> Dict[str, float]:
        """
        1.5x eşik özellikleri
        
        Args:
            values: Geçmiş değerler
            threshold: Eşik değeri (default 1.5)
            
        Returns:
            Eşik özellikleri
        """
        features = {}
        
        if len(values) >= 10:
            recent_10 = values[-10:]
            recent_50 = values[-50:] if len(values) >= 50 else values
            
            # Son 10 elde kaç tanesi eşik altı/üstü
            features['below_threshold_10'] = sum(1 for v in recent_10 if v < threshold)
            features['above_threshold_10'] = sum(1 for v in recent_10 if v >= threshold)
            features['threshold_ratio_10'] = features['above_threshold_10'] / 10
            
            # Son 50 elde oran
            if len(recent_50) > 0:
                features['threshold_ratio_50'] = sum(1 for v in recent_50 if v >= threshold) / len(recent_50)
            
            # Kritik bölge (1.45-1.55) analizi
            features['in_critical_zone_10'] = sum(1 for v in recent_10 if 1.45 <= v <= 1.55)
        
        return features
    
    @staticmethod
    def extract_distance_features(values: List[float], milestones: List[float] = [10.0, 20.0, 50.0, 100.0]) -> Dict[str, float]:
        """
        Büyük çarpanlardan bu yana geçen el sayısı
        
        Args:
            values: Geçmiş değerler
            milestones: Kilometre taşları
            
        Returns:
            Mesafe özellikleri
        """
        features = {}
        
        for milestone in milestones:
            distance = 0
            found = False
            
            # Geriden başlayarak ara
            for i in range(len(values) - 1, -1, -1):
                if values[i] >= milestone:
                    distance = len(values) - 1 - i
                    found = True
                    break
            
            if not found:
                distance = len(values)  # Hiç görülmemiş
            
            features[f'distance_from_{int(milestone)}x'] = distance
        
        return features
    
    @staticmethod
    def extract_streak_features(values: List[float]) -> Dict[str, float]:
        """
        Ardışık pattern özellikleri
        
        Args:
            values: Geçmiş değerler
            
        Returns:
            Ardışıklık özellikleri
        """
        features = {}
        
        if len(values) >= 2:
            # Ardışık yükseliş/düşüş
            rising_streak = 0
            falling_streak = 0
            
            for i in range(len(values) - 1, 0, -1):
                if values[i] > values[i - 1]:
                    rising_streak += 1
                    if falling_streak > 0:
                        break
                elif values[i] < values[i - 1]:
                    falling_streak += 1
                    if rising_streak > 0:
                        break
                else:
                    break
            
            features['rising_streak'] = rising_streak
            features['falling_streak'] = falling_streak
            
            # Son 10 elde aynı kategoride kaç el
            if len(values) >= 10:
                recent_categories = [CategoryDefinitions.get_category_numeric(v) for v in values[-10:]]
                current_cat = recent_categories[-1]
                same_category_count = sum(1 for c in recent_categories if c == current_cat)
                features['same_category_count_10'] = same_category_count
        
        return features
    
    @staticmethod
    def extract_volatility_features(values: List[float]) -> Dict[str, float]:
        """
        Volatilite özellikleri
        
        Args:
            values: Geçmiş değerler
            
        Returns:
            Volatilite özellikleri
        """
        features = {}
        
        if len(values) >= 10:
            recent = values[-10:]
            
            # Değişim oranları
            changes = [recent[i] - recent[i-1] for i in range(1, len(recent))]
            features['volatility_10'] = np.std(changes) if changes else 0
            features['mean_change_10'] = np.mean(changes) if changes else 0
            
            # Range
            features['range_10'] = np.max(recent) - np.min(recent)
        
        return features
    
    @staticmethod
    def extract_sequence_pattern_features(values: List[float]) -> Dict[str, float]:
        """
        Sequence pattern özellikleri - Model bu pattern'leri öğrenecek
        
        Args:
            values: Geçmiş değerler
            
        Returns:
            Sequence pattern özellikleri
        """
        features = {}
        
        if len(values) >= 10:
            # Son 10 elin kategori sequence'i encoding
            recent_10_categories = [CategoryDefinitions.get_category_numeric(v) for v in values[-10:]]
            
            # Pattern tekrarı skoru
            pattern_length = 3
            if len(values) >= pattern_length * 2:
                recent_pattern = values[-pattern_length:]
                previous_pattern = values[-pattern_length*2:-pattern_length]
                
                # Benzerlik skoru (0-1 arası)
                similarity = 0
                for i in range(pattern_length):
                    cat_recent = CategoryDefinitions.get_category_numeric(recent_pattern[i])
                    cat_prev = CategoryDefinitions.get_category_numeric(previous_pattern[i])
                    if cat_recent == cat_prev:
                        similarity += 1
                features['pattern_repetition_score'] = similarity / pattern_length
            
            # Kategorilerin dağılımı
            features['loss_zone_count_10'] = sum(1 for c in recent_10_categories if c == 0)
            features['safe_zone_count_10'] = sum(1 for c in recent_10_categories if c == 1)
            features['high_zone_count_10'] = sum(1 for c in recent_10_categories if c == 2)
        
        return features
    
    @staticmethod
    def extract_statistical_distribution_features(values: List[float]) -> Dict[str, float]:
        """
        İstatistiksel dağılım özellikleri
        
        Args:
            values: Geçmiş değerler
            
        Returns:
            Dağılım özellikleri
        """
        features = {}
        
        if len(values) >= 50:
            recent_50 = values[-50:]
            
            # Skewness (çarpıklık) ve Kurtosis (basıklık)
            from scipy import stats
            try:
                features['skewness_50'] = float(stats.skew(recent_50))
                features['kurtosis_50'] = float(stats.kurtosis(recent_50))
            except:
                features['skewness_50'] = 0.0
                features['kurtosis_50'] = 0.0
            
            # Percentile'lar
            features['percentile_25'] = np.percentile(recent_50, 25)
            features['percentile_50'] = np.percentile(recent_50, 50)
            features['percentile_75'] = np.percentile(recent_50, 75)
            features['percentile_90'] = np.percentile(recent_50, 90)
            
            # IQR (Interquartile Range)
            features['iqr'] = features['percentile_75'] - features['percentile_25']
        
        return features
    
    @staticmethod
    def extract_multi_timeframe_momentum(values: List[float]) -> Dict[str, float]:
        """
        Çoklu zaman dilimleri momentum özellikleri
        
        Args:
            values: Geçmiş değerler
            
        Returns:
            Momentum özellikleri
        """
        features = {}
        
        # Momentum hesaplama fonksiyonu
        def calc_momentum(window):
            if len(window) < 2:
                return 0.0
            # Son değer ile ortalama arasındaki fark
            return (window[-1] - np.mean(window)) / (np.std(window) + 1e-8)
        
        # Farklı zaman dilimleri için momentum
        timeframes = {'short_25': 25, 'medium_50': 50, 'medium_100': 100, 'long_200': 200}
        
        for name, size in timeframes.items():
            if len(values) >= size:
                recent = values[-size:]
                features[f'momentum_{name}'] = calc_momentum(recent)
                
                # Trend strength (yönlü hareketin gücü)
                changes = [recent[i] - recent[i-1] for i in range(1, len(recent))]
                positive_changes = sum(1 for c in changes if c > 0)
                features[f'trend_strength_{name}'] = (positive_changes / len(changes)) * 2 - 1  # -1 ile 1 arası
        
        # Acceleration (ivme) - momentum değişim hızı
        if len(values) >= 100:
            momentum_50_old = calc_momentum(values[-100:-50]) if len(values) >= 100 else 0
            momentum_50_new = calc_momentum(values[-50:])
            features['acceleration'] = momentum_50_new - momentum_50_old
        
        return features
    
    @staticmethod
    def extract_recovery_pattern_features(values: List[float]) -> Dict[str, float]:
        """
        Recovery (toparlanma) pattern özellikleri - Model öğrenecek
        
        Args:
            values: Geçmiş değerler
            
        Returns:
            Recovery pattern özellikleri
        """
        features = {}
        
        if len(values) >= 50:
            recent_50 = values[-50:]
            recent_10 = values[-10:]
            
            # Volatilite normalizasyonu (soğuma bitişi göstergesi)
            volatility_50 = np.std(recent_50)
            volatility_10 = np.std(recent_10)
            
            # Volatilite düşüyorsa toparlanma olabilir
            if volatility_50 > 0:
                features['volatility_normalization'] = 1 - (volatility_10 / volatility_50)
            else:
                features['volatility_normalization'] = 0.0
            
            # Büyük çarpandan sonra stabilizasyon
            max_in_50 = max(recent_50)
            if max_in_50 > 10.0:  # Büyük çarpan varsa
                # Son 10 elde ne kadar stabilize
                recent_10_std = np.std(recent_10)
                features['post_big_multiplier_stability'] = 1 / (1 + recent_10_std)
            else:
                features['post_big_multiplier_stability'] = 0.5  # Neutral
            
            # Trend değişimi (düşüşten yükselişe geçiş)
            if len(recent_50) >= 20:
                first_half_mean = np.mean(recent_50[:25])
                second_half_mean = np.mean(recent_50[25:])
                features['trend_reversal'] = (second_half_mean - first_half_mean) / (first_half_mean + 1e-8)
        
        return features
    
    @staticmethod
    def extract_anomaly_detection_features(values: List[float]) -> Dict[str, float]:
        """
        Anomali tespit özellikleri - Model için ipucu
        
        Args:
            values: Geçmiş değerler
            
        Returns:
            Anomali özellikleri
        """
        features = {}
        
        if len(values) >= 50:
            recent_50 = values[-50:]
            current_value = values[-1]
            
            # Z-score (standart sapma cinsinden sapma)
            mean_50 = np.mean(recent_50)
            std_50 = np.std(recent_50)
            
            if std_50 > 0:
                features['z_score'] = (current_value - mean_50) / std_50
                features['is_outlier'] = 1.0 if abs(features['z_score']) > 2.0 else 0.0
            else:
                features['z_score'] = 0.0
                features['is_outlier'] = 0.0
            
            # Median Absolute Deviation (MAD) - daha robust anomali tespiti
            median_50 = np.median(recent_50)
            mad = np.median([abs(v - median_50) for v in recent_50])
            
            if mad > 0:
                features['mad_score'] = (current_value - median_50) / (1.4826 * mad)
            else:
                features['mad_score'] = 0.0
            
            # Son değerin percentile'i
            features['current_value_percentile'] = sum(1 for v in recent_50 if v <= current_value) / len(recent_50)
        
        return features
    
    @staticmethod
    def extract_cooling_period_features(values: List[float]) -> Dict[str, float]:
        """
        Soğuma dönemi özellikleri - Model öğrenecek (net kural yok)
        
        Args:
            values: Geçmiş değerler
            
        Returns:
            Soğuma pattern özellikleri (model için)
        """
        features = {}
        
        if len(values) >= 50:
            # Büyük çarpanlardan mesafe
            features.update(FeatureEngineering.extract_distance_features(
                values, milestones=[10.0, 20.0, 50.0, 100.0, 200.0]
            ))
            
            # Son 10 elde volatilite pattern
            if len(values) >= 10:
                recent_10 = values[-10:]
                features['recent_volatility_pattern'] = np.std(recent_10) / (np.mean(recent_10) + 1e-8)
            
            # Ardışık düşük değer sayısı
            if len(values) >= 10:
                below_2x_count = sum(1 for v in values[-10:] if v < 2.0)
                features['low_value_streak_10'] = below_2x_count
        
        return features
    
    @staticmethod
    def extract_all_features(values: List[float]) -> Dict[str, float]:
        """
        Tüm özellikleri çıkar - Geliştirilmiş versiyon
        
        Args:
            values: Geçmiş değerler listesi
            
        Returns:
            Tüm özellikler
        """
        all_features = {}
        
        # Temel özellikler (güncellenen pencere boyutlarıyla)
        all_features.update(FeatureEngineering.extract_basic_features(values))
        
        # Eşik özellikleri
        all_features.update(FeatureEngineering.extract_threshold_features(values))
        
        # Mesafe özellikleri (genişletilmiş milestones)
        all_features.update(FeatureEngineering.extract_distance_features(
            values, milestones=[10.0, 20.0, 50.0, 100.0, 200.0]
        ))
        
        # Ardışıklık özellikleri
        all_features.update(FeatureEngineering.extract_streak_features(values))
        
        # Volatilite özellikleri
        all_features.update(FeatureEngineering.extract_volatility_features(values))
        
        # YENİ: Sequence pattern özellikleri
        all_features.update(FeatureEngineering.extract_sequence_pattern_features(values))
        
        # YENİ: İstatistiksel dağılım özellikleri
        all_features.update(FeatureEngineering.extract_statistical_distribution_features(values))
        
        # YENİ: Multi-timeframe momentum
        all_features.update(FeatureEngineering.extract_multi_timeframe_momentum(values))
        
        # YENİ: Recovery pattern
        all_features.update(FeatureEngineering.extract_recovery_pattern_features(values))
        
        # YENİ: Anomaly detection
        all_features.update(FeatureEngineering.extract_anomaly_detection_features(values))
        
        # YENİ: Soğuma dönemi pattern'leri (model için)
        all_features.update(FeatureEngineering.extract_cooling_period_features(values))
        
        # Son değer
        if len(values) > 0:
            all_features['last_value'] = values[-1]
            all_features['last_category'] = CategoryDefinitions.get_category_numeric(values[-1])
        
        return all_features


def create_sequences(data: List[float], sequence_length: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zaman serisi için sequence'ler oluştur (LSTM/TCN için)
    
    Args:
        data: Veri listesi
        sequence_length: Sequence uzunluğu
        
    Returns:
        X (sequences), y (hedefler)
    """
    X, y = [], []
    
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i])
    
    return np.array(X), np.array(y)


# Model eğitiminde kullanılacak sabitler
SEQUENCE_LENGTHS = {
    'short': 50,    # Kısa vadeli pattern'ler
    'medium': 200,  # Orta vadeli trend
    'long': 500     # Uzun vadeli davranış
}

# Risk yönetimi eşikleri
CONFIDENCE_THRESHOLDS = {
    'aggressive': 0.50,  # Agresif mod
    'normal': 0.65,      # Normal mod
    'rolling': 0.80      # Rolling (konservatif) mod
}
