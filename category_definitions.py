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
    HIGH_MULTIPLIER_THRESHOLD = 3.0
    
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
            return 'Yüksek (2.0x - 3.0x)'
        elif value < 5.0:
            return 'Çok Yüksek (3.0x - 5.0x)'
        elif value < 10.0:
            return 'Nadir (5.0x - 10.0x)'
        elif value < 20.0:
            return 'Çok Nadir (10.0x - 20.0x)'
        elif value < 50.0:
            return 'Mega (20.0x - 50.0x)'
        elif value < 100.0:
            return 'Süper Mega (50.0x - 100.0x)'
        else:
            return 'Ultra (100.0x+)'


class FeatureEngineering:
    """
    Özellik çıkarma fonksiyonları.
    Hem Colab'da eğitim sırasında hem de lokalde tahmin sırasında kullanılacak.
    """
    
    @staticmethod
    def extract_basic_features(values: List[float], window_sizes: List[int] = [5, 10, 20, 50]) -> Dict[str, float]:
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
    def extract_all_features(values: List[float]) -> Dict[str, float]:
        """
        Tüm özellikleri çıkar
        
        Args:
            values: Geçmiş değerler listesi
            
        Returns:
            Tüm özellikler
        """
        all_features = {}
        
        # Temel özellikler
        all_features.update(FeatureEngineering.extract_basic_features(values))
        
        # Eşik özellikleri
        all_features.update(FeatureEngineering.extract_threshold_features(values))
        
        # Mesafe özellikleri
        all_features.update(FeatureEngineering.extract_distance_features(values))
        
        # Ardışıklık özellikleri
        all_features.update(FeatureEngineering.extract_streak_features(values))
        
        # Volatilite özellikleri
        all_features.update(FeatureEngineering.extract_volatility_features(values))
        
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
