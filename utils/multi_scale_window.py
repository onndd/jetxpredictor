"""
Multi-Scale Window Training System for JetX Predictor

Bu modül, farklı pencere boyutlarında feature extraction ve model eğitimi sağlar.
VERİ SIRASI ASLA DEĞİŞTİRİLMEZ - Kronolojik bütünlük korunur.

Amaç:
- Uzun dönem desenleri (500, 250) → Genel trend
- Orta dönem desenleri (100, 50) → Orta vadeli davranış
- Kısa dönem desenleri (20) → Lokal volatilite

Ensemble ile tüm pencere boyutlarını birleştirerek hem mikro hem makro 
desenleri yakalayan bir sistem oluşturur.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
import pickle
from tqdm import tqdm

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiScaleWindowExtractor:
    """
    Multi-scale window feature extraction
    
    VERİ BÜTÜNLÜĞÜ GARANTİLERİ:
    - shuffle=False (HER ZAMAN)
    - Kronolojik sıra korunur
    - Augmentation YOK
    - Sequential blocks (sliding window DEĞİL)
    """
    
    def __init__(
        self,
        window_sizes: List[int] = [500, 250, 100, 50, 20],
        overlap: float = 0.0  # 0 = no overlap (sequential blocks)
    ):
        """
        Args:
            window_sizes: Pencere boyutları (büyükten küçüğe)
            overlap: Pencere overlap oranı (0.0 = sequential, 0.5 = %50 overlap)
        """
        self.window_sizes = sorted(window_sizes, reverse=True)
        self.overlap = overlap
        
        logger.info(f"MultiScaleWindowExtractor initialized:")
        logger.info(f"  Window sizes: {self.window_sizes}")
        logger.info(f"  Overlap: {self.overlap * 100:.1f}%")
        logger.info(f"  ⚠️  Data order preservation: ENABLED (shuffle=False)")
    
    def extract_windows(
        self,
        data: np.ndarray,
        window_size: int,
        preserve_order: bool = True
    ) -> List[np.ndarray]:
        """
        Belirli bir pencere boyutu için veriyi parçalara böl
        
        Args:
            data: Input data (N, features) veya (N,)
            window_size: Pencere boyutu
            preserve_order: Kronolojik sırayı koru (ZORUNLU True)
            
        Returns:
            List of windows (her biri window_size uzunluğunda)
        """
        if not preserve_order:
            raise ValueError("⚠️  preserve_order=False YASAK! Veri sırası değiştirilemez!")
        
        n_samples = len(data)
        
        if window_size > n_samples:
            logger.warning(
                f"Window size ({window_size}) > data size ({n_samples}). "
                f"Returning single window."
            )
            return [data]
        
        # Calculate step size
        if self.overlap == 0:
            step = window_size  # Sequential blocks (no overlap)
        else:
            step = int(window_size * (1 - self.overlap))
        
        windows = []
        start = 0
        
        while start + window_size <= n_samples:
            window = data[start:start + window_size]
            windows.append(window)
            start += step
        
        # Son kalan veri varsa, onu da al
        if start < n_samples and len(windows) > 0:
            # Son pencereyi uzat
            last_window = data[-window_size:]
            windows.append(last_window)
        
        logger.info(
            f"  Window {window_size}: {len(windows)} blocks created "
            f"(step={step}, sequential={'✓' if self.overlap == 0 else '✗'})"
        )
        
        return windows
    
    def extract_features_for_window(
        self,
        data: np.ndarray,
        window_size: int,
        feature_extractor: Optional[Any] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Belirli bir pencere boyutu için feature extraction
        
        Args:
            data: Raw data (multipliers)
            window_size: Pencere boyutu
            feature_extractor: Feature extraction fonksiyonu
            
        Returns:
            (features, targets) tuple
        """
        windows = self.extract_windows(data, window_size)
        
        all_features = []
        all_targets = []
        
        for window in tqdm(windows, desc=f"Window {window_size} features"):
            if feature_extractor is not None:
                # Custom feature extractor kullan
                features = feature_extractor(window)
            else:
                # Basit istatistiksel özellikler
                features = self._compute_basic_features(window)
            
            # Target: penceredeki son değer
            target = window[-1]
            
            all_features.append(features)
            all_targets.append(target)
        
        return np.array(all_features), np.array(all_targets)
    
    def _compute_basic_features(self, window: np.ndarray) -> np.ndarray:
        """
        Temel istatistiksel özellikler
        
        Args:
            window: Pencere verisi
            
        Returns:
            Feature vector
        """
        features = [
            np.mean(window),           # Ortalama
            np.std(window),            # Standart sapma
            np.min(window),            # Minimum
            np.max(window),            # Maximum
            np.median(window),         # Medyan
            np.percentile(window, 25), # Q1
            np.percentile(window, 75), # Q3
            window[-1],                # Son değer
            window[-1] - window[0],    # Değişim
            (window[-1] - window[0]) / (window[0] + 1e-10)  # Yüzde değişim
        ]
        
        return np.array(features)
    
    def create_multi_scale_features(
        self,
        data: np.ndarray,
        feature_extractor: Optional[Any] = None
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Tüm pencere boyutları için feature extraction
        
        Args:
            data: Raw data
            feature_extractor: Feature extraction fonksiyonu
            
        Returns:
            Dict {window_size: (features, targets)}
        """
        logger.info("\n" + "="*70)
        logger.info("MULTI-SCALE FEATURE EXTRACTION")
        logger.info("="*70)
        logger.info(f"Data size: {len(data)}")
        logger.info(f"Window sizes: {self.window_sizes}")
        logger.info("⚠️  DATA ORDER PRESERVED (shuffle=False)")
        logger.info("="*70 + "\n")
        
        results = {}
        
        for window_size in self.window_sizes:
            logger.info(f"\n📊 Processing window size: {window_size}")
            features, targets = self.extract_features_for_window(
                data, window_size, feature_extractor
            )
            results[window_size] = (features, targets)
            logger.info(f"  ✅ Features shape: {features.shape}")
            logger.info(f"  ✅ Targets shape: {targets.shape}")
        
        logger.info("\n" + "="*70)
        logger.info("✅ Multi-scale feature extraction complete!")
        logger.info("="*70 + "\n")
        
        return results


class MultiScaleEnsemble:
    """
    Multi-scale model ensemble
    
    Her pencere boyutu için ayrı model eğitir ve tahminleri birleştirir.
    """
    
    def __init__(
        self,
        window_sizes: List[int] = [500, 250, 100, 50, 20],
        weights: Optional[Dict[int, float]] = None
    ):
        """
        Args:
            window_sizes: Pencere boyutları
            weights: Her pencere için ağırlık {window_size: weight}
                    None ise equal weights
        """
        self.window_sizes = window_sizes
        self.models = {}  # {window_size: model}
        
        # Ağırlıklar
        if weights is None:
            # Equal weights
            self.weights = {ws: 1.0 / len(window_sizes) for ws in window_sizes}
        else:
            # Normalize weights
            total = sum(weights.values())
            self.weights = {ws: w / total for ws, w in weights.items()}
        
        logger.info("MultiScaleEnsemble initialized:")
        logger.info(f"  Window sizes: {self.window_sizes}")
        logger.info(f"  Weights: {self.weights}")
    
    def add_model(self, window_size: int, model: Any):
        """
        Bir pencere boyutu için model ekle
        
        Args:
            window_size: Pencere boyutu
            model: Trained model
        """
        if window_size not in self.window_sizes:
            logger.warning(
                f"Window size {window_size} not in configured sizes. "
                f"Adding anyway."
            )
        
        self.models[window_size] = model
        logger.info(f"✅ Model added for window size {window_size}")
    
    def predict(
        self,
        features_dict: Dict[int, np.ndarray],
        method: str = 'weighted_mean'
    ) -> np.ndarray:
        """
        Ensemble tahmin
        
        Args:
            features_dict: {window_size: features} dictionary
            method: Ensemble metodu
                - 'weighted_mean': Ağırlıklı ortalama
                - 'mean': Basit ortalama
                - 'median': Medyan
                - 'max_confidence': En yüksek confidence'a sahip tahmin
                
        Returns:
            Ensemble predictions
        """
        predictions = {}
        
        # Her modelden tahmin al
        for window_size, model in self.models.items():
            if window_size not in features_dict:
                logger.warning(
                    f"Window size {window_size} features not provided. Skipping."
                )
                continue
            
            features = features_dict[window_size]
            pred = model.predict(features)
            predictions[window_size] = pred
        
        if not predictions:
            raise ValueError("No predictions available!")
        
        # Ensemble
        if method == 'weighted_mean':
            # Ağırlıklı ortalama
            ensemble_pred = np.zeros(len(next(iter(predictions.values()))))
            
            for window_size, pred in predictions.items():
                weight = self.weights.get(window_size, 0.0)
                ensemble_pred += weight * pred
                
        elif method == 'mean':
            # Basit ortalama
            all_preds = np.array(list(predictions.values()))
            ensemble_pred = np.mean(all_preds, axis=0)
            
        elif method == 'median':
            # Medyan
            all_preds = np.array(list(predictions.values()))
            ensemble_pred = np.median(all_preds, axis=0)
            
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        return ensemble_pred
    
    def save(self, filepath: str):
        """
        Ensemble'ı kaydet
        
        Args:
            filepath: Kayıt yolu
        """
        save_dict = {
            'window_sizes': self.window_sizes,
            'weights': self.weights,
            'models': self.models
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"✅ Ensemble saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Ensemble'ı yükle
        
        Args:
            filepath: Dosya yolu
        """
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.window_sizes = save_dict['window_sizes']
        self.weights = save_dict['weights']
        self.models = save_dict['models']
        
        logger.info(f"✅ Ensemble loaded from {filepath}")
        logger.info(f"  Window sizes: {self.window_sizes}")
        logger.info(f"  Models loaded: {len(self.models)}")


def split_data_preserving_order(
    data: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Veriyi train/val/test olarak böl (KRONOLOJIK SIRAYI KORUYARAK)
    
    Args:
        data: Input data
        train_ratio: Train oranı
        val_ratio: Validation oranı (train'den alınır)
        
    Returns:
        (train, val, test) tuple
    """
    n = len(data)
    
    # Test: Son %20
    test_size = int(n * (1 - train_ratio))
    test_data = data[-test_size:]
    
    # Train + Val: İlk %80
    train_val_data = data[:-test_size]
    
    # Val: Train'in son %10'u
    val_size = int(len(train_val_data) * val_ratio)
    val_data = train_val_data[-val_size:]
    train_data = train_val_data[:-val_size]
    
    logger.info("\n" + "="*70)
    logger.info("TIME-SERIES SPLIT (Kronolojik)")
    logger.info("="*70)
    logger.info(f"Total data: {n}")
    logger.info(f"Train: {len(train_data)} ({len(train_data)/n*100:.1f}%)")
    logger.info(f"Val: {len(val_data)} ({len(val_data)/n*100:.1f}%)")
    logger.info(f"Test: {len(test_data)} ({len(test_data)/n*100:.1f}%)")
    logger.info("⚠️  SHUFFLE: DEVRE DIŞI (veri sırası korundu)")
    logger.info("="*70 + "\n")
    
    return train_data, val_data, test_data


# Kullanım örneği
if __name__ == "__main__":
    # Örnek veri
    np.random.seed(42)
    sample_data = np.random.randn(1000) + np.linspace(0, 10, 1000)
    
    # Multi-scale extractor
    extractor = MultiScaleWindowExtractor(
        window_sizes=[500, 250, 100, 50, 20],
        overlap=0.0  # Sequential blocks
    )
    
    # Feature extraction
    features_dict = extractor.create_multi_scale_features(sample_data)
    
    # Her pencere boyutu için
    for window_size, (features, targets) in features_dict.items():
        print(f"\nWindow {window_size}:")
        print(f"  Features: {features.shape}")
        print(f"  Targets: {targets.shape}")
    
    # Train/Val/Test split
    train, val, test = split_data_preserving_order(
        sample_data,
        train_ratio=0.8,
        val_ratio=0.1
    )
