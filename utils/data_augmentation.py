"""
JetX Predictor - Data Augmentation

Sequence data için augmentation teknikleri.
Eğitim verisi çeşitliliğini artırarak model performansını iyileştirir.
"""

import numpy as np
from typing import List, Tuple
from scipy.interpolate import CubicSpline
import logging

# Logging ayarla
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SequenceAugmenter:
    """
    Time series sequence data için augmentation
    
    Teknikler:
    1. Gaussian Noise: Küçük rastgele gürültü ekle
    2. Time Shift: Zaman ekseninde kaydırma
    3. Scaling: Değerleri ölçeklendir
    4. Time Warping: Zaman ekseninde distortion
    5. Masking: Rastgele noktaları maskele
    6. Jittering: Küçük rastgele değişiklikler
    """
    
    def __init__(self, seed: int = None):
        """
        Args:
            seed: Random seed (reproducibility için)
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def add_gaussian_noise(
        self, 
        sequence: np.ndarray, 
        sigma: float = 0.02
    ) -> np.ndarray:
        """
        Gaussian noise ekle
        
        Args:
            sequence: Input sequence (N,) veya (N, features)
            sigma: Noise standard deviation
            
        Returns:
            Augmented sequence
        """
        noise = np.random.normal(0, sigma, sequence.shape)
        return sequence + noise
    
    def time_shift(
        self,
        sequence: np.ndarray,
        max_shift: int = 5
    ) -> np.ndarray:
        """
        Zaman ekseninde kaydırma
        
        Args:
            sequence: Input sequence
            max_shift: Maksimum kaydırma miktarı
            
        Returns:
            Shifted sequence
        """
        shift = np.random.randint(-max_shift, max_shift + 1)
        return np.roll(sequence, shift, axis=0)
    
    def scale(
        self,
        sequence: np.ndarray,
        scale_range: Tuple[float, float] = (0.95, 1.05)
    ) -> np.ndarray:
        """
        Değerleri ölçeklendir
        
        Args:
            sequence: Input sequence
            scale_range: (min_scale, max_scale)
            
        Returns:
            Scaled sequence
        """
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return sequence * scale
    
    def time_warp(
        self,
        sequence: np.ndarray,
        sigma: float = 0.2,
        knot: int = 4
    ) -> np.ndarray:
        """
        Zaman ekseninde distortion (warping)
        
        Args:
            sequence: Input sequence
            sigma: Warping strength
            knot: Knot points sayısı
            
        Returns:
            Warped sequence
        """
        length = len(sequence)
        
        # Original time steps
        orig_steps = np.arange(length)
        
        # Random warps at knot points
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2,))
        
        # Warp steps (evenly distributed)
        warp_steps = np.linspace(0, length - 1, num=knot + 2)
        
        # Cubic spline interpolation
        try:
            warper = CubicSpline(warp_steps, warp_steps * random_warps)
            new_steps = warper(orig_steps)
            
            # Clip to valid range
            new_steps = np.clip(new_steps, 0, length - 1)
            
            # Interpolate sequence values
            warped = np.interp(new_steps, orig_steps, sequence.flatten())
            
            return warped.reshape(sequence.shape)
        except Exception as e:
            logger.warning(f"Time warp başarısız, orijinal sequence döndürülüyor: {e}")
            return sequence
    
    def random_masking(
        self,
        sequence: np.ndarray,
        mask_ratio: float = 0.1
    ) -> np.ndarray:
        """
        Rastgele noktaları maskele (0 yap)
        
        Args:
            sequence: Input sequence
            mask_ratio: Maskelenecek noktaların oranı
            
        Returns:
            Masked sequence
        """
        mask = np.random.random(sequence.shape) > mask_ratio
        return sequence * mask
    
    def jitter(
        self,
        sequence: np.ndarray,
        sigma: float = 0.03
    ) -> np.ndarray:
        """
        Küçük rastgele değişiklikler (jittering)
        
        Args:
            sequence: Input sequence
            sigma: Jitter strength
            
        Returns:
            Jittered sequence
        """
        jitter_noise = np.random.normal(0, sigma, sequence.shape)
        return sequence + jitter_noise
    
    def magnitude_warp(
        self,
        sequence: np.ndarray,
        sigma: float = 0.2,
        knot: int = 4
    ) -> np.ndarray:
        """
        Magnitude warping (değer ekseninde distortion)
        
        Args:
            sequence: Input sequence
            sigma: Warping strength
            knot: Knot points sayısı
            
        Returns:
            Magnitude warped sequence
        """
        length = len(sequence)
        
        # Random warps at knot points
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2,))
        
        # Warp steps
        warp_steps = np.linspace(0, length - 1, num=knot + 2)
        
        try:
            # Cubic spline
            warper = CubicSpline(warp_steps, random_warps)
            warp_curve = warper(np.arange(length))
            
            return sequence * warp_curve.reshape(-1, 1) if sequence.ndim > 1 else sequence * warp_curve
        except Exception as e:
            logger.warning(f"Magnitude warp başarısız: {e}")
            return sequence
    
    def augment(
        self,
        sequence: np.ndarray,
        method: str = 'all',
        num_augmentations: int = 1
    ) -> List[np.ndarray]:
        """
        Ana augmentation fonksiyonu
        
        Args:
            sequence: Input sequence
            method: Augmentation metodu:
                - 'noise': Gaussian noise
                - 'shift': Time shift
                - 'scale': Scaling
                - 'warp': Time warping
                - 'mask': Random masking
                - 'jitter': Jittering
                - 'magnitude': Magnitude warping
                - 'all': Tüm metodları rastgele uygula
                - 'random': Rastgele bir metod seç
            num_augmentations: Kaç augmented sample üretilecek
            
        Returns:
            Augmented sequences listesi
        """
        augmented = []
        
        for _ in range(num_augmentations):
            # Kopyayı oluştur
            aug_seq = sequence.copy()
            
            if method == 'noise':
                aug_seq = self.add_gaussian_noise(aug_seq)
            elif method == 'shift':
                aug_seq = self.time_shift(aug_seq)
            elif method == 'scale':
                aug_seq = self.scale(aug_seq)
            elif method == 'warp':
                aug_seq = self.time_warp(aug_seq)
            elif method == 'mask':
                aug_seq = self.random_masking(aug_seq)
            elif method == 'jitter':
                aug_seq = self.jitter(aug_seq)
            elif method == 'magnitude':
                aug_seq = self.magnitude_warp(aug_seq)
            elif method == 'random':
                # Rastgele bir metod seç
                methods = ['noise', 'shift', 'scale', 'warp', 'jitter']
                chosen_method = np.random.choice(methods)
                aug_seq = self.augment(sequence, method=chosen_method, num_augmentations=1)[0]
            elif method == 'all':
                # Rastgele 2-3 metod uygula
                methods = ['noise', 'shift', 'scale', 'warp', 'jitter', 'magnitude']
                num_methods = np.random.randint(2, 4)
                chosen_methods = np.random.choice(methods, size=num_methods, replace=False)
                
                for m in chosen_methods:
                    if m == 'noise':
                        aug_seq = self.add_gaussian_noise(aug_seq, sigma=0.01)
                    elif m == 'shift':
                        aug_seq = self.time_shift(aug_seq, max_shift=3)
                    elif m == 'scale':
                        aug_seq = self.scale(aug_seq, scale_range=(0.97, 1.03))
                    elif m == 'warp':
                        aug_seq = self.time_warp(aug_seq, sigma=0.1)
                    elif m == 'jitter':
                        aug_seq = self.jitter(aug_seq, sigma=0.02)
                    elif m == 'magnitude':
                        aug_seq = self.magnitude_warp(aug_seq, sigma=0.1)
            
            augmented.append(aug_seq)
        
        return augmented
    
    def augment_batch(
        self,
        sequences: np.ndarray,
        method: str = 'random',
        augmentation_factor: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch augmentation
        
        Args:
            sequences: Input sequences (batch_size, seq_len, features)
            method: Augmentation metodu
            augmentation_factor: Her sequence için kaç augmentation
            
        Returns:
            (augmented_sequences, original_indices)
            - augmented_sequences: Augmented + original
            - original_indices: Hangi sequence'lerin original olduğu
        """
        batch_size = len(sequences)
        all_sequences = []
        original_indices = []
        
        for i, seq in enumerate(sequences):
            # Original sequence
            all_sequences.append(seq)
            original_indices.append(True)
            
            # Augmented versions
            aug_seqs = self.augment(seq, method=method, num_augmentations=augmentation_factor - 1)
            all_sequences.extend(aug_seqs)
            original_indices.extend([False] * (augmentation_factor - 1))
        
        return np.array(all_sequences), np.array(original_indices)


class FeatureAugmenter:
    """
    Feature vector augmentation (engineered features için)
    """
    
    def __init__(self, seed: int = None):
        """
        Args:
            seed: Random seed
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def add_noise(
        self,
        features: np.ndarray,
        noise_level: float = 0.01
    ) -> np.ndarray:
        """
        Feature'lara noise ekle
        
        Args:
            features: Feature vector (N, features)
            noise_level: Noise seviyesi
            
        Returns:
            Noisy features
        """
        noise = np.random.normal(0, noise_level, features.shape)
        return features + noise
    
    def dropout_features(
        self,
        features: np.ndarray,
        dropout_rate: float = 0.1
    ) -> np.ndarray:
        """
        Rastgele feature'ları 0 yap
        
        Args:
            features: Feature vector
            dropout_rate: Dropout oranı
            
        Returns:
            Dropped features
        """
        mask = np.random.random(features.shape) > dropout_rate
        return features * mask
    
    def augment(
        self,
        features: np.ndarray,
        method: str = 'noise',
        num_augmentations: int = 1
    ) -> List[np.ndarray]:
        """
        Feature augmentation
        
        Args:
            features: Input features
            method: 'noise' veya 'dropout'
            num_augmentations: Kaç sample
            
        Returns:
            Augmented features listesi
        """
        augmented = []
        
        for _ in range(num_augmentations):
            if method == 'noise':
                aug_feat = self.add_noise(features, noise_level=0.01)
            elif method == 'dropout':
                aug_feat = self.dropout_features(features, dropout_rate=0.1)
            elif method == 'both':
                aug_feat = self.add_noise(features, noise_level=0.005)
                aug_feat = self.dropout_features(aug_feat, dropout_rate=0.05)
            else:
                aug_feat = features.copy()
            
            augmented.append(aug_feat)
        
        return augmented


# Kullanım örnekleri
if __name__ == "__main__":
    # Sequence augmentation
    seq_aug = SequenceAugmenter(seed=42)
    
    # Örnek sequence
    sample_seq = np.array([1.2, 1.5, 2.3, 1.8, 3.4, 2.1, 1.6])
    
    print("Original sequence:", sample_seq)
    
    # Noise ekle
    noisy = seq_aug.add_gaussian_noise(sample_seq, sigma=0.05)
    print("Noisy:", noisy)
    
    # Time warp
    warped = seq_aug.time_warp(sample_seq, sigma=0.2)
    print("Warped:", warped)
    
    # Tüm augmentation
    augmented = seq_aug.augment(sample_seq, method='all', num_augmentations=3)
    print(f"\n{len(augmented)} augmented samples oluşturuldu")
    
    # Feature augmentation
    feat_aug = FeatureAugmenter(seed=42)
    sample_features = np.array([0.5, 1.2, 0.8, 2.1, 1.5])
    
    noisy_features = feat_aug.add_noise(sample_features, noise_level=0.02)
    print("\nOriginal features:", sample_features)
    print("Noisy features:", noisy_features)
