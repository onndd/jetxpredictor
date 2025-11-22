"""
JetX Predictor - Kategori Tanımları ve Özellik Çıkarma Fonksiyonları

Bu dosya hem Google Colab'da hem de lokal Streamlit uygulamasında kullanılacak.
Kategori tanımları ve özellik çıkarma fonksiyonlarını içerir.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import logging
import warnings
from functools import lru_cache

# Logging ayarla
logger = logging.getLogger(__name__)

# Döngüsel import'u önlemek için lazy loading kullanılıyor
# Import'lar extract_all_features fonksiyonu içinde yapılacak
ADVANCED_ANALYZERS_AVAILABLE = None  # İlk çağrıda kontrol edilecek

# Bağımlılık kontrol sistemi
MISSING_DEPENDENCIES = []
DEPENDENCY_WARNINGS_SHOWN = False

# Optional dependencies kontrolü
try:
    import scipy
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    MISSING_DEPENDENCIES.append('scipy')

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    MISSING_DEPENDENCIES.append('pywt')

try:
    import nolds
    NOLDS_AVAILABLE = True
except ImportError:
    NOLDS_AVAILABLE = False
    MISSING_DEPENDENCIES.append('nolds')

def _validate_critical_dependencies():
    """
    KRİTİK bağımlılıkları kontrol et - eksikse SİSTEM ÇÖKSÜN
    
    🔒 GÜVENLİK: Sessiz başarısızlık yok - fail-fast principle
    🚨 KRİTİK: Eksik dependency'ler data skew'e neden olur
    """
    critical_deps = ['numpy', 'pandas']  # Minimum çalışması gerekenler
    missing = []
    
    for dep in critical_deps:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    
    if missing:
        error_msg = f"🚨 KRİTİK BAĞIMLILIKLAR EKSİK: {missing}\n"
        error_msg += f"Lütfen kurun: pip install {' '.join(missing)}\n"
        error_msg += "JetX Predictor bu bağımlılıklar olmadan çalışamaz!"
        raise RuntimeError(error_msg)
    
    return True

def check_and_warn_dependencies():
    """
    Optional bağımlılıkları kontrol et ve uyar (sadece bir kez)
    
    ⚠️ UYARI: Eksik optional dependencies feature sayısını azaltır
    📊 ETKİ: 25+ advanced feature 0.0 olarak doldurulur
    """
    global DEPENDENCY_WARNINGS_SHOWN
    
    # Önce kritik bağımlılıkları kontrol et
    _validate_critical_dependencies()
    
    if MISSING_DEPENDENCIES and not DEPENDENCY_WARNINGS_SHOWN:
        logger.warning("=" * 70)
        logger.warning("EKSİK OPTİONEL BAĞIMLILIKLAR TESPİT EDİLDİ!")
        logger.warning("=" * 70)
        logger.warning("Aşağıdaki kütüphaneler bulunamadı:")
        for dep in MISSING_DEPENDENCIES:
            logger.warning(f"  ❌ {dep}")
        logger.warning("")
        logger.warning("📊 ETKİ: Bu, model performansını düşürebilir!")
        logger.warning("🔧 Özellikler: Eksik özelliklerin yerine 0.0 değerleri kullanılacak.")
        logger.warning(f"🎯 KAYIP: {len(MISSING_DEPENDENCIES)} kütüphane = 25+ advanced feature kaybı")
        logger.warning("")
        logger.warning("Kurulum için:")
        logger.warning(f"  pip install {' '.join(MISSING_DEPENDENCIES)}")
        logger.warning("")
        logger.warning("Alternatif: pip install scipy pywt nolds")
        logger.warning("=" * 70)
        DEPENDENCY_WARNINGS_SHOWN = True
    
    return len(MISSING_DEPENDENCIES) == 0


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
    
    # 15 KATEGORİ SETİ - Farklı perspektiflerden analiz için
    # Her set değerleri farklı granülarite ve odakta kategorilere ayırır
    # Model bu setleri özellik olarak kullanarak çok boyutlu pattern öğrenir
    
    CATEGORY_SET_1 = [  # Standart Dağılım
        (1.00, 1.20), (1.20, 1.35), (1.35, 1.49), (1.50, 1.70), (1.70, 2.00),
        (2.00, 3.00), (3.00, 5.00), (5.00, 10.00), (10.00, 20.00), (20.00, 50.00),
        (50.00, 100.00), (100.00, 200.00), (200.00, 500.00), (500.00, 1000.00), (1000.00, float('inf'))
    ]
    
    CATEGORY_SET_2 = [  # Alt Bölge Hassas
        (1.00, 1.10), (1.10, 1.20), (1.20, 1.30), (1.30, 1.40), (1.40, 1.45),
        (1.45, 1.49), (1.50, 1.55), (1.55, 1.65), (1.65, 1.80), (1.80, 2.20),
        (2.20, 3.50), (3.50, 7.00), (7.00, 15.00), (15.00, 50.00), (50.00, float('inf'))
    ]
    
    CATEGORY_SET_3 = [  # Üst Bölge Hassas
        (1.00, 1.49), (1.50, 2.00), (2.00, 3.00), (3.00, 4.50), (4.50, 6.50),
        (6.50, 9.00), (9.00, 12.00), (12.00, 18.00), (18.00, 25.00), (25.00, 40.00),
        (40.00, 70.00), (70.00, 150.00), (150.00, 300.00), (300.00, 700.00), (700.00, float('inf'))
    ]
    
    CATEGORY_SET_4 = [  # Psikolojik Bariyerler
        (1.00, 1.49), (1.50, 1.99), (2.00, 2.99), (3.00, 4.99), (5.00, 9.99),
        (10.00, 19.99), (20.00, 29.99), (30.00, 49.99), (50.00, 74.99), (75.00, 99.99),
        (100.00, 199.99), (200.00, 499.99), (500.00, 999.99), (1000.00, 1999.99), (2000.00, float('inf'))
    ]
    
    CATEGORY_SET_5 = [  # Kumar Stratejisi Odaklı
        (1.00, 1.30), (1.30, 1.45), (1.45, 1.49), (1.50, 1.60), (1.60, 1.85),
        (1.85, 2.30), (2.30, 3.20), (3.20, 5.50), (5.50, 11.00), (11.00, 25.00),
        (25.00, 60.00), (60.00, 180.00), (180.00, 450.00), (450.00, 1200.00), (1200.00, float('inf'))
    ]
    
    CATEGORY_SET_6 = [  # Mikro Hassas (1.5 civarı)
        (1.00, 1.25), (1.25, 1.38), (1.38, 1.44), (1.44, 1.47), (1.47, 1.49),
        (1.50, 1.52), (1.52, 1.56), (1.56, 1.62), (1.62, 1.75), (1.75, 2.10),
        (2.10, 3.00), (3.00, 6.00), (6.00, 20.00), (20.00, 100.00), (100.00, float('inf'))
    ]
    
    CATEGORY_SET_7 = [  # Logaritmik Ölçek
        (1.00, 1.49), (1.50, 2.00), (2.00, 2.70), (2.70, 3.70), (3.70, 5.00),
        (5.00, 7.40), (7.40, 10.00), (10.00, 15.00), (15.00, 22.00), (22.00, 33.00),
        (33.00, 50.00), (50.00, 100.00), (100.00, 220.00), (220.00, 500.00), (500.00, float('inf'))
    ]
    
    CATEGORY_SET_8 = [  # Risk Seviyeleri
        (1.00, 1.35), (1.35, 1.43), (1.43, 1.49), (1.50, 1.65), (1.65, 1.90),
        (1.90, 2.50), (2.50, 4.00), (4.00, 8.00), (8.00, 15.00), (15.00, 30.00),
        (30.00, 75.00), (75.00, 200.00), (200.00, 600.00), (600.00, 1500.00), (1500.00, float('inf'))
    ]
    
    CATEGORY_SET_9 = [  # Yüzde Tabanlı
        (1.00, 1.20), (1.20, 1.35), (1.35, 1.49), (1.50, 1.75), (1.75, 2.00),
        (2.00, 2.50), (2.50, 3.50), (3.50, 5.00), (5.00, 10.00), (10.00, 25.00),
        (25.00, 50.00), (50.00, 100.00), (100.00, 300.00), (300.00, 1000.00), (1000.00, float('inf'))
    ]
    
    CATEGORY_SET_10 = [  # Oyuncu Psikolojisi
        (1.00, 1.15), (1.15, 1.30), (1.30, 1.49), (1.50, 1.70), (1.70, 2.00),
        (2.00, 2.50), (2.50, 3.50), (3.50, 5.00), (5.00, 8.00), (8.00, 15.00),
        (15.00, 35.00), (35.00, 80.00), (80.00, 250.00), (250.00, 800.00), (800.00, float('inf'))
    ]
    
    CATEGORY_SET_11 = [  # Frekans Tabanlı
        (1.00, 1.40), (1.40, 1.49), (1.50, 1.80), (1.80, 2.20), (2.20, 3.00),
        (3.00, 4.50), (4.50, 7.00), (7.00, 12.00), (12.00, 20.00), (20.00, 40.00),
        (40.00, 100.00), (100.00, 300.00), (300.00, 700.00), (700.00, 2000.00), (2000.00, float('inf'))
    ]
    
    CATEGORY_SET_12 = [  # Kısa Vadeli Trader
        (1.00, 1.10), (1.10, 1.25), (1.25, 1.40), (1.40, 1.49), (1.50, 1.58),
        (1.58, 1.68), (1.68, 1.82), (1.82, 2.05), (2.05, 2.40), (2.40, 3.20),
        (3.20, 5.50), (5.50, 12.00), (12.00, 50.00), (50.00, 500.00), (500.00, float('inf'))
    ]
    
    CATEGORY_SET_13 = [  # Uzun Vadeli Pattern
        (1.00, 1.49), (1.50, 1.95), (1.95, 2.80), (2.80, 4.20), (4.20, 6.50),
        (6.50, 11.00), (11.00, 18.00), (18.00, 28.00), (28.00, 45.00), (45.00, 85.00),
        (85.00, 160.00), (160.00, 350.00), (350.00, 750.00), (750.00, 1800.00), (1800.00, float('inf'))
    ]
    
    CATEGORY_SET_14 = [  # Matematiksel Bölümler
        (1.00, 1.33), (1.33, 1.49), (1.50, 1.67), (1.67, 2.00), (2.00, 2.67),
        (2.67, 3.67), (3.67, 5.33), (5.33, 8.33), (8.33, 13.33), (13.33, 23.33),
        (23.33, 53.33), (53.33, 133.33), (133.33, 333.33), (333.33, 1333.33), (1333.33, float('inf'))
    ]
    
    CATEGORY_SET_15 = [  # Dinamik Hassas
        (1.00, 1.18), (1.18, 1.32), (1.32, 1.42), (1.42, 1.49), (1.50, 1.54),
        (1.54, 1.60), (1.60, 1.72), (1.72, 1.88), (1.88, 2.15), (2.15, 2.75),
        (2.75, 4.40), (4.40, 9.50), (9.50, 32.00), (32.00, 250.00), (250.00, float('inf'))
    ]
    
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
    def get_category_set_index(value: float, category_set: List[Tuple[float, float]]) -> int:
        """
        Verilen değerin kategori setindeki index'ini döndürür
        
        Args:
            value: Değer
            category_set: Kategori seti (tuple listesi)
            
        Returns:
            Kategori index'i (0-14 arası)
        """
        for idx, (min_val, max_val) in enumerate(category_set):
            if min_val <= value < max_val:
                return idx
        # Son kategoriye düşer (inf olan)
        return len(category_set) - 1
    
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
    def extract_basic_features(values: List[float], window_sizes: List[int] = [25, 50, 100, 200, 500, 1000]) -> Dict[str, float]:
        """
        Temel özellikler: Ortalama, std, min, max
        
        Args:
            values: Geçmiş değerler listesi (en yeni en sonda)
            window_sizes: Pencere boyutları (güncellenmiş: 1000 veri penceresi eklendi)
            
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
            else:
                # Yeterli veri yoksa varsayılan değerler
                features[f'mean_{window}'] = 0.0
                features[f'std_{window}'] = 0.0
                features[f'min_{window}'] = 0.0
                features[f'max_{window}'] = 0.0
                features[f'median_{window}'] = 0.0
        
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
            else:
                features['threshold_ratio_50'] = 0.0
            
            # Kritik bölge (1.45-1.55) analizi
            features['in_critical_zone_10'] = sum(1 for v in recent_10 if 1.45 <= v <= 1.55)
        else:
            # Yeterli veri yoksa varsayılan değerler
            features['below_threshold_10'] = 0.0
            features['above_threshold_10'] = 0.0
            features['threshold_ratio_10'] = 0.0
            features['threshold_ratio_50'] = 0.0
            features['in_critical_zone_10'] = 0.0
        
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
            else:
                features['same_category_count_10'] = 0.0
        else:
            # Yeterli veri yoksa varsayılan değerler
            features['rising_streak'] = 0.0
            features['falling_streak'] = 0.0
            features['same_category_count_10'] = 0.0
        
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
            features['volatility_10'] = np.std(changes) if changes else 0.0
            features['mean_change_10'] = np.mean(changes) if changes else 0.0
            
            # Range
            features['range_10'] = np.max(recent) - np.min(recent)
        else:
            # Yeterli veri yoksa varsayılan değerler
            features['volatility_10'] = 0.0
            features['mean_change_10'] = 0.0
            features['range_10'] = 0.0
        
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
            else:
                features['pattern_repetition_score'] = 0.0
            
            # Kategorilerin dağılımı
            features['loss_zone_count_10'] = sum(1 for c in recent_10_categories if c == 0)
            features['safe_zone_count_10'] = sum(1 for c in recent_10_categories if c == 1)
            features['high_zone_count_10'] = sum(1 for c in recent_10_categories if c == 2)
        else:
            # Yeterli veri yoksa varsayılan değerler
            features['pattern_repetition_score'] = 0.0
            features['loss_zone_count_10'] = 0.0
            features['safe_zone_count_10'] = 0.0
            features['high_zone_count_10'] = 0.0
        
        return features
    
    @staticmethod
    def extract_statistical_distribution_features(values: List[float]) -> Dict[str, float]:
        """
        İstatistiksel dağılım özellikleri - Hızlandırılmış versiyon
        
        Args:
            values: Geçmiş değerler
            
        Returns:
            Dağılım özellikleri
        """
        features = {}
        
        if len(values) >= 50:
            recent_50 = values[-50:]
            
            # Skewness (çarpıklık) ve Kurtosis (basıklık) - Hızlı manuel hesaplama
            try:
                # Skewness için hızlı hesaplama (RANSAC kullanmadan)
                mean_val = np.mean(recent_50)
                std_val = np.std(recent_50)
                
                if std_val > 0:
                    # Skewness: 3*(mean - median)/std
                    median_val = np.median(recent_50)
                    features['skewness_50'] = 3.0 * (mean_val - median_val) / std_val
                    
                    # Kurtosis için basit hesaplama
                    # Kurtosis: (mean - mode)/std (yaklaşık)
                    # Mode'u histogram ile tahmin et
                    hist, bins = np.histogram(recent_50, bins=10)
                    mode_idx = np.argmax(hist)
                    mode_val = (bins[mode_idx] + bins[mode_idx + 1]) / 2
                    features['kurtosis_50'] = (mean_val - mode_val) / std_val
                else:
                    features['skewness_50'] = 0.0
                    features['kurtosis_50'] = 0.0
                    
            except Exception as e:
                # Hata durumunda default değerler
                logger.warning(f"İstatistiksel özellik hesaplama hatası: {e}")
                features['skewness_50'] = 0.0
                features['kurtosis_50'] = 0.0
            
            # Percentile'lar
            features['percentile_25'] = np.percentile(recent_50, 25)
            features['percentile_50'] = np.percentile(recent_50, 50)
            features['percentile_75'] = np.percentile(recent_50, 75)
            features['percentile_90'] = np.percentile(recent_50, 90)
            
            # IQR (Interquartile Range)
            features['iqr'] = features['percentile_75'] - features['percentile_25']
        else:
            # Yeterli veri yoksa varsayılan değerler
            features['skewness_50'] = 0.0
            features['kurtosis_50'] = 0.0
            features['percentile_25'] = 0.0
            features['percentile_50'] = 0.0
            features['percentile_75'] = 0.0
            features['percentile_90'] = 0.0
            features['iqr'] = 0.0
        
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
            else:
                # Yeterli veri yoksa varsayılan değerler
                features[f'momentum_{name}'] = 0.0
                features[f'trend_strength_{name}'] = 0.0
        
        # Acceleration (ivme) - momentum değişim hızı
        if len(values) >= 100:
            momentum_50_old = calc_momentum(values[-100:-50]) if len(values) >= 100 else 0
            momentum_50_new = calc_momentum(values[-50:])
            features['acceleration'] = momentum_50_new - momentum_50_old
        else:
            features['acceleration'] = 0.0
        
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
            else:
                features['trend_reversal'] = 0.0
        else:
            # Yeterli veri yoksa varsayılan değerler
            features['volatility_normalization'] = 0.0
            features['post_big_multiplier_stability'] = 0.5
            features['trend_reversal'] = 0.0
        
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
        else:
            # Yeterli veri yoksa varsayılan değerler
            features['z_score'] = 0.0
            features['is_outlier'] = 0.0
            features['mad_score'] = 0.0
            features['current_value_percentile'] = 0.0
        
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
        
        # Büyük çarpanlardan mesafe (her zaman çalışır)
        features.update(FeatureEngineering.extract_distance_features(
            values, milestones=[10.0, 20.0, 50.0, 100.0, 200.0]
        ))
        
        if len(values) >= 10:
            # Son 10 elde volatilite pattern
            recent_10 = values[-10:]
            features['recent_volatility_pattern'] = np.std(recent_10) / (np.mean(recent_10) + 1e-8)
            
            # Ardışık düşük değer sayısı
            below_2x_count = sum(1 for v in values[-10:] if v < 2.0)
            features['low_value_streak_10'] = below_2x_count
        else:
            # Yeterli veri yoksa varsayılan değerler
            features['recent_volatility_pattern'] = 0.0
            features['low_value_streak_10'] = 0.0
        
        return features
    
    @staticmethod
    def extract_advanced_wavelet_features(values: List[float]) -> Dict[str, float]:
        """
        Wavelet Transform özellikleri (frequency domain analysis)
        
        Args:
            values: Geçmiş değerler
            
        Returns:
            Wavelet özellikleri
        """
        # Her zaman aynı sayıda özellik döndürmek için default değerler
        features = {
            'wavelet_energy_level_0': 0.0, 'wavelet_mean_level_0': 0.0, 'wavelet_std_level_0': 0.0,
            'wavelet_energy_level_1': 0.0, 'wavelet_mean_level_1': 0.0, 'wavelet_std_level_1': 0.0,
            'wavelet_energy_level_2': 0.0, 'wavelet_mean_level_2': 0.0, 'wavelet_std_level_2': 0.0,
            'wavelet_energy_level_3': 0.0, 'wavelet_mean_level_3': 0.0, 'wavelet_std_level_3': 0.0,
            'wavelet_total_energy': 0.0, 'wavelet_low_freq_ratio': 0.0
        }
        
        if len(values) >= 100:
            try:
                import pywt
                
                recent_100 = values[-100:]
                
                # Discrete Wavelet Transform (DWT)
                coeffs = pywt.wavedec(recent_100, 'db4', level=3)
                
                # Her level için energy hesapla
                for i, coeff in enumerate(coeffs):
                    energy = np.sum(coeff ** 2)
                    features[f'wavelet_energy_level_{i}'] = energy
                    features[f'wavelet_mean_level_{i}'] = np.mean(np.abs(coeff))
                    features[f'wavelet_std_level_{i}'] = np.std(coeff)
                
                # Total energy
                total_energy = sum(np.sum(c ** 2) for c in coeffs)
                features['wavelet_total_energy'] = total_energy
                
                # Energy ratios
                if total_energy > 0:
                    features['wavelet_low_freq_ratio'] = np.sum(coeffs[0] ** 2) / total_energy
                
            except ImportError:
                logger.warning("pywt bulunamadı, wavelet features atlanıyor")
            except Exception as e:
                logger.error(f"Wavelet transform hatası: {e}")
        
        return features
    
    @staticmethod
    def extract_advanced_dfa_features(values: List[float]) -> Dict[str, float]:
        """
        DFA (Detrended Fluctuation Analysis) özellikleri
        
        Args:
            values: Geçmiş değerler
            
        Returns:
            DFA özellikleri
        """
        features = {'dfa_alpha': 0.5, 'dfa_regime': 0.5} # Default to random walk
        
        if len(values) >= 200:
            try:
                import nolds
                
                recent_200 = values[-200:]
                dfa_alpha = nolds.dfa(recent_200)
                features['dfa_alpha'] = dfa_alpha
                
                if dfa_alpha < 0.5:
                    features['dfa_regime'] = 0  # Mean reverting
                elif dfa_alpha > 0.5:
                    features['dfa_regime'] = 1  # Trending
                
            except ImportError:
                logger.warning("nolds bulunamadı, DFA features atlanıyor")
            except Exception as e:
                logger.error(f"DFA hesaplama hatası: {e}")
        
        return features
    
    @staticmethod
    def extract_advanced_hurst_features(values: List[float]) -> Dict[str, float]:
        """
        Hurst Exponent özellikleri
        
        Args:
            values: Geçmiş değerler
            
        Returns:
            Hurst özellikleri
        """
        features = {'hurst_exponent': 0.5, 'hurst_classification': 0, 'trend_strength_hurst': 0.0}
        
        if len(values) >= 200:
            try:
                import nolds
                
                recent_200 = values[-200:]
                hurst = nolds.hurst_rs(recent_200)
                features['hurst_exponent'] = hurst
                
                if hurst < 0.45:
                    features['hurst_classification'] = -1
                elif hurst > 0.55:
                    features['hurst_classification'] = 1
                
                features['trend_strength_hurst'] = abs(hurst - 0.5) * 2
                
            except ImportError:
                logger.warning("nolds bulunamadı, Hurst features atlanıyor")
            except Exception as e:
                logger.error(f"Hurst exponent hesaplama hatası: {e}")
        
        return features
    
    @staticmethod
    def extract_advanced_fourier_features(values: List[float]) -> Dict[str, float]:
        """
        Fourier Transform özellikleri
        
        Args:
            values: Geçmiş değerler
            
        Returns:
            Fourier özellikleri
        """
        features = {
            'dominant_frequency_idx': 0.0, 'dominant_frequency_magnitude': 0.0,
            'spectral_energy': 0.0, 'low_freq_energy_ratio': 0.0,
            'mid_freq_energy_ratio': 0.0, 'high_freq_energy_ratio': 0.0,
            'spectral_centroid': 0.0
        }
        
        if len(values) >= 100:
            try:
                recent_100 = values[-100:]
                fft = np.fft.fft(recent_100)
                fft_abs = np.abs(fft[:50])
                
                dominant_freq_idx = np.argmax(fft_abs[1:]) + 1
                features['dominant_frequency_idx'] = dominant_freq_idx
                features['dominant_frequency_magnitude'] = fft_abs[dominant_freq_idx]
                
                spectral_energy = np.sum(fft_abs ** 2)
                features['spectral_energy'] = spectral_energy
                
                if spectral_energy > 0:
                    low_freq_energy = np.sum(fft_abs[1:10] ** 2)
                    mid_freq_energy = np.sum(fft_abs[10:25] ** 2)
                    high_freq_energy = np.sum(fft_abs[25:] ** 2)
                    
                    features['low_freq_energy_ratio'] = low_freq_energy / spectral_energy
                    features['mid_freq_energy_ratio'] = mid_freq_energy / spectral_energy
                    features['high_freq_energy_ratio'] = high_freq_energy / spectral_energy
                
                freqs = np.arange(len(fft_abs))
                if np.sum(fft_abs) > 0:
                    features['spectral_centroid'] = np.sum(freqs * fft_abs) / np.sum(fft_abs)
                
            except Exception as e:
                logger.error(f"Fourier transform hatası: {e}")
        
        return features
    
    @staticmethod
    def extract_advanced_autocorrelation_features(values: List[float]) -> Dict[str, float]:
        """
        Autocorrelation özellikleri
        
        Args:
            values: Geçmiş değerler
            
        Returns:
            Autocorrelation özellikleri
        """
        lags = [1, 2, 3, 5, 10, 20]
        features = {f'acf_lag_{lag}': 0.0 for lag in lags}
        features['acf_max_value'] = 0.0
        features['acf_max_lag'] = 0.0
        
        if len(values) >= 50:
            try:
                recent_50 = values[-50:]
                
                for lag in lags:
                    if len(recent_50) > lag:
                        series1 = recent_50[:-lag]
                        series2 = recent_50[lag:]
                        
                        if len(series1) > 1 and np.std(series1) > 0 and np.std(series2) > 0:
                            acf = np.corrcoef(series1, series2)[0, 1]
                            features[f'acf_lag_{lag}'] = acf
                
                acf_values = [features.get(f'acf_lag_{lag}', 0) for lag in lags]
                if acf_values:
                    max_acf = max(acf_values)
                    max_acf_lag = lags[acf_values.index(max_acf)]
                    features['acf_max_value'] = max_acf
                    features['acf_max_lag'] = max_acf_lag
                
            except Exception as e:
                logger.error(f"Autocorrelation hesaplama hatası: {e}")
        
        return features
    
    @staticmethod
    def extract_category_set_features(values: List[float]) -> Dict[str, float]:
        """
        15 Kategori Seti Özellikleri
        
        Args:
            values: Geçmiş değerler
            
        Returns:
            15 kategori seti özelliği
        """
        features = {f'cat_set_{i}': 0.0 for i in range(1, 16)}
        
        if len(values) > 0:
            current_value = values[-1]
            
            all_sets = [
                CategoryDefinitions.CATEGORY_SET_1, CategoryDefinitions.CATEGORY_SET_2,
                CategoryDefinitions.CATEGORY_SET_3, CategoryDefinitions.CATEGORY_SET_4,
                CategoryDefinitions.CATEGORY_SET_5, CategoryDefinitions.CATEGORY_SET_6,
                CategoryDefinitions.CATEGORY_SET_7, CategoryDefinitions.CATEGORY_SET_8,
                CategoryDefinitions.CATEGORY_SET_9, CategoryDefinitions.CATEGORY_SET_10,
                CategoryDefinitions.CATEGORY_SET_11, CategoryDefinitions.CATEGORY_SET_12,
                CategoryDefinitions.CATEGORY_SET_13, CategoryDefinitions.CATEGORY_SET_14,
                CategoryDefinitions.CATEGORY_SET_15
            ]
            
            for i, cat_set in enumerate(all_sets, 1):
                features[f'cat_set_{i}'] = CategoryDefinitions.get_category_set_index(current_value, cat_set)
        
        return features
    
    @staticmethod
    def _calculate_features_direct(values: List[float]) -> Dict[str, float]:
        """
        Doğrudan özellik hesaplama (cache olmadan - streaming için optimize)
        
        Args:
            values: Geçmiş değerler listesi
            
        Returns:
            Özellik sözlüğü
        """
        all_features = {}
        
        # Temel özellikler
        all_features.update(FeatureEngineering.extract_basic_features(values))
        all_features.update(FeatureEngineering.extract_threshold_features(values))
        all_features.update(FeatureEngineering.extract_distance_features(values, milestones=[10.0, 20.0, 50.0, 100.0, 200.0]))
        all_features.update(FeatureEngineering.extract_streak_features(values))
        all_features.update(FeatureEngineering.extract_volatility_features(values))
        all_features.update(FeatureEngineering.extract_sequence_pattern_features(values))
        all_features.update(FeatureEngineering.extract_statistical_distribution_features(values))
        all_features.update(FeatureEngineering.extract_multi_timeframe_momentum(values))
        all_features.update(FeatureEngineering.extract_recovery_pattern_features(values))
        all_features.update(FeatureEngineering.extract_cooling_period_features(values))
        all_features.update(FeatureEngineering.extract_category_set_features(values))
        all_features.update(FeatureEngineering.extract_advanced_wavelet_features(values))
        all_features.update(FeatureEngineering.extract_advanced_dfa_features(values))
        all_features.update(FeatureEngineering.extract_advanced_hurst_features(values))
        all_features.update(FeatureEngineering.extract_advanced_fourier_features(values))
        all_features.update(FeatureEngineering.extract_advanced_autocorrelation_features(values))
        
        # YENİ: Psikolojik Analiz Özellikleri (lazy loading ile)
        global ADVANCED_ANALYZERS_AVAILABLE
        if ADVANCED_ANALYZERS_AVAILABLE is None:
            # İlk kez kontrol ediliyor
            try:
                from utils.psychological_analyzer import PsychologicalAnalyzer
                from utils.anomaly_streak_detector import AnomalyStreakDetector
                ADVANCED_ANALYZERS_AVAILABLE = True
            except ImportError:
                ADVANCED_ANALYZERS_AVAILABLE = False
        
        if ADVANCED_ANALYZERS_AVAILABLE:
            try:
                from utils.psychological_analyzer import PsychologicalAnalyzer
                from utils.anomaly_streak_detector import AnomalyStreakDetector
                
                # Psikolojik analiz
                psychological_analyzer = PsychologicalAnalyzer(threshold=1.5)
                psych_features = psychological_analyzer.analyze_psychological_patterns(values)
                all_features.update(psych_features)
                
                # Anomaly streak analiz
                anomaly_detector = AnomalyStreakDetector(threshold=1.5)
                streak_features = anomaly_detector.extract_streak_features(values)
                all_features.update(streak_features)
            except Exception as e:
                logger.warning(f"Gelişmiş analiz hatası: {e}")
                # Default değerlerle devam et
                all_features.update({
                    'bait_switch_score': 0.0,
                    'trap_risk': 0.0,
                    'false_confidence_score': 0.0,
                    'heating_score': 0.0,
                    'cooling_score': 0.0,
                    'volatility_shift': 0.0,
                    'desperation_level': 0.0,
                    'gambler_fallacy_risk': 0.0,
                    'manipulation_score': 0.0,
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
                })
        else:
            # Advanced analyzers mevcut değil, default değerleri ekle
            all_features.update({
                'bait_switch_score': 0.0,
                'trap_risk': 0.0,
                'false_confidence_score': 0.0,
                'heating_score': 0.0,
                'cooling_score': 0.0,
                'volatility_shift': 0.0,
                'desperation_level': 0.0,
                'gambler_fallacy_risk': 0.0,
                'manipulation_score': 0.0,
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
            })
        
        # Son değer bilgileri
        if len(values) > 0:
            all_features['last_value'] = values[-1]
            all_features['last_category'] = CategoryDefinitions.get_category_numeric(values[-1])
        else:
            all_features['last_value'] = 0.0
            all_features['last_category'] = 0.0
        
        return all_features
    
    @staticmethod
    def extract_all_features(values: List[float]) -> Dict[str, float]:
        """
        Tüm özellikleri çıkar - Cache'siz versiyon (streaming için optimize)
        
        🔒 GÜVENLİK: LRU cache kaldırıldı - data skew ve leakage riski önledi
        ⚡ PERFORMANS: Direct calculation - tuple dönüşüm ve hashing maliyeti yok
        
        Args:
            values: Geçmiş değerler listesi (en yeni en sonda)
            
        Returns:
            Özellik sözlüğü
        """
        return FeatureEngineering._calculate_features_direct(values)


def create_sequences(data: List[float], sequence_length: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zaman serisi için sequence'ler oluştur (LSTM/TCN için)
    """
    X, y = [], []
    
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i])
    
    return np.array(X), np.array(y)


# Model eğitiminde kullanılacak sabitler
SEQUENCE_LENGTHS = {
    'short': 50,
    'medium': 200,
    'long': 500,
    'extra_long': 1000  # Yeni: 1000 veri penceresi
}

# Risk yönetimi eşikleri (GÜNCELLENDİ)
CONFIDENCE_THRESHOLDS = {
    'normal': 0.85,      # %85 Güven (Normal Mod)
    'rolling': 0.95      # %95 Güven (Kasa Katlama/Rolling)
}
