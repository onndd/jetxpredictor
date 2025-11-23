"""
JetX Predictor - Data Augmentation (SAFE MODE)

GÜNCELLEME:
- "Strict No-Shuffle & No-Synthetic" politikası uygulandı.
- Zaman serisi yapısını bozan (time shift, warp, shuffle) tüm işlemler kaldırıldı.
- Sadece çok hafif Gaussian Noise (aşırı öğrenmeyi önlemek için opsiyonel) bırakıldı.
- Varsayılan olarak augmentation KAPALI.

Bu modül artık verinin orijinal yapısını ve sırasını %100 korur.
"""

import numpy as np
from typing import List, Tuple, Optional
import logging

# Logging ayarla
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SequenceAugmenter:
    """
    Time series sequence data için GÜVENLİ augmentation
    
    UYARI: Kullanıcı politikası gereği sentetik veri üretimi ve
    zaman ekseninde manipülasyon (shift, warp) KESİNLİKLE YASAKTIR.
    
    Bu sınıf sadece eğitim sırasında overfitting'i önlemek için
    çok hafif gürültü ekleme yeteneğine sahiptir (varsayılan kapalı).
    """
    
    def __init__(self, seed: int = None, enabled: bool = False):
        """
        Args:
            seed: Random seed
            enabled: Augmentation aktif mi? (Varsayılan: Hayır)
        """
        self.seed = seed
        self.enabled = enabled
        if seed is not None:
            np.random.seed(seed)
        
        if self.enabled:
            logger.warning("⚠️ Augmentation AKTİF! (Sadece hafif gürültü)")
        else:
            logger.info("✅ Augmentation KAPALI (Veri bütünlüğü korunuyor)")
    
    def add_gaussian_noise(
        self, 
        sequence: np.ndarray, 
        sigma: float = 0.001  # Çok düşük gürültü
    ) -> np.ndarray:
        """
        Gaussian noise ekle (Sadece enabled=True ise)
        Verinin trendini bozmaz, sadece aşırı öğrenmeyi (memarization) zorlaştırır.
        
        Args:
            sequence: Input sequence
            sigma: Noise seviyesi (Çok düşük tutulmalı)
            
        Returns:
            Sequence (Gürültülü veya Orijinal)
        """
        if not self.enabled:
            return sequence
            
        noise = np.random.normal(0, sigma, sequence.shape)
        return sequence + noise
    
    # ---------------------------------------------------------
    # TEHLİKELİ FONKSİYONLAR (DEVRE DIŞI BIRAKILDI / KALDIRILDI)
    # ---------------------------------------------------------
    # time_shift -> KALDIRILDI (Sırayı bozar)
    # time_warp -> KALDIRILDI (Zaman algısını bozar)
    # random_masking -> KALDIRILDI (Veri kaybı)
    # jitter -> KALDIRILDI (Noise ile benzer, gereksiz)
    # magnitude_warp -> KALDIRILDI (Değerleri saptırır)
    # ---------------------------------------------------------

    def augment(
        self,
        sequence: np.ndarray,
        method: str = 'none',  # Varsayılan: Hiçbir şey yapma
        num_augmentations: int = 0 # Varsayılan: Çoğaltma yapma
    ) -> List[np.ndarray]:
        """
        Augmentation fonksiyonu (GÜVENLİ MOD)
        
        Args:
            sequence: Input sequence
            method: 'noise' (hafif gürültü) veya 'none'
            num_augmentations: Kaç kopya (0 önerilir)
            
        Returns:
            Sadece orijinal veri veya (istenirse) gürültülü kopyalar
        """
        # Eğer augmentation kapalıysa veya method none ise direkt orijinali dön
        if not self.enabled or method == 'none' or num_augmentations <= 0:
            return [sequence] # Sadece orijinal
        
        augmented = []
        # Orijinal veriyi her zaman ekle (Veri kaybı olmasın)
        augmented.append(sequence)
        
        # İstenirse gürültülü kopyalar ekle (Sentetik veri sayılır, dikkat!)
        for _ in range(num_augmentations):
            if method == 'noise':
                aug_seq = self.add_gaussian_noise(sequence.copy(), sigma=0.005)
                augmented.append(aug_seq)
        
        return augmented
    
    def augment_batch(
        self,
        sequences: np.ndarray,
        method: str = 'none',
        augmentation_factor: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch augmentation (GÜVENLİ MOD)
        
        Varsayılan olarak sadece orijinal veriyi döndürür.
        """
        if not self.enabled or method == 'none' or augmentation_factor <= 0:
            # Hiçbir şey yapma, orijinali döndür
            original_indices = np.ones(len(sequences), dtype=bool)
            return sequences, original_indices

        # Eğer zorla açıldıysa (tavsiye edilmez)
        batch_list = []
        indices_list = []
        
        for seq in sequences:
            # Orijinal
            batch_list.append(seq)
            indices_list.append(True)
            
            # Kopyalar
            augs = self.augment(seq, method=method, num_augmentations=augmentation_factor)
            # İlk eleman orijinal olduğu için atla (zaten ekledik)
            for aug in augs[1:]:
                batch_list.append(aug)
                indices_list.append(False) # Sentetik
                
        return np.array(batch_list), np.array(indices_list)


class FeatureAugmenter:
    """
    Feature vector augmentation
    
    UYARI: Bu sınıf da varsayılan olarak pasif durumdadır.
    """
    
    def __init__(self, seed: int = None, enabled: bool = False):
        self.seed = seed
        self.enabled = enabled
        if seed is not None:
            np.random.seed(seed)
            
        if self.enabled:
            logger.warning("⚠️ Feature Augmentation AKTİF!")
        else:
            logger.info("✅ Feature Augmentation KAPALI")
    
    def add_noise(self, features: np.ndarray, noise_level: float = 0.001) -> np.ndarray:
        if not self.enabled: return features
        noise = np.random.normal(0, noise_level, features.shape)
        return features + noise
    
    def augment(self, features: np.ndarray, method: str = 'none', num_augmentations: int = 0) -> List[np.ndarray]:
        if not self.enabled or method == 'none':
            return [features]
            
        augmented = [features]
        for _ in range(num_augmentations):
            if method == 'noise':
                aug_feat = self.add_noise(features.copy())
                augmented.append(aug_feat)
                
        return augmented


# Kullanım örnekleri (Test)
if __name__ == "__main__":
    print("🛡️ SAFE AUGMENTATION TEST")
    
    # Varsayılan (Kapalı)
    seq_aug = SequenceAugmenter(seed=42, enabled=False)
    sample_seq = np.array([1.2, 1.5, 2.3])
    
    res = seq_aug.augment(sample_seq, method='all', num_augmentations=5)
    print(f"Kapalı mod çıktı sayısı: {len(res)} (Beklenen: 1)")
    print(f"Değişiklik var mı: {np.array_equal(res[0], sample_seq)}")
    
    # Açık (Sadece Noise)
    print("\n⚠️ Açık mod (Noise):")
    seq_aug_active = SequenceAugmenter(seed=42, enabled=True)
    res_active = seq_aug_active.augment(sample_seq, method='noise', num_augmentations=1)
    print(f"Çıktı sayısı: {len(res_active)} (1 Orijinal + 1 Kopya)")
