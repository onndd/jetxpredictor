"""
JetX Predictor - Balanced Batch Generator

Class imbalance sorununu çözmek için her batch'te dengeli sampling yapar.
Her batch'te %50 1.5 altı, %50 1.5 üstü değer olacak şekilde sampling.

Sentetik veri üretmez, sadece mevcut veriyi dengeli şekilde örnekler.
"""

import numpy as np
from tensorflow.keras.utils import Sequence
from typing import List, Dict, Union, Tuple


class BalancedBatchGenerator(Sequence):
    """
    Keras Sequence generator - Her batch'te dengeli class sampling
    
    Her batch'te:
    - %50 1.5 altı değerler (azınlık sınıfı - replace=True ile tekrar seçilebilir)
    - %50 1.5 üstü değerler (çoğunluk sınıfı - replace=False)
    
    Bu yaklaşım:
    - ✅ Veri değiştirmez (JetX doğasını korur)
    - ✅ Class weight'e fazla ihtiyaç kalmaz
    - ✅ Lazy learning'i önler
    - ✅ Model her batch'te her iki sınıfı da görür
    """
    
    def __init__(
        self,
        X: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
        y: Union[np.ndarray, Dict[str, np.ndarray]],
        batch_size: int = 32,
        threshold: float = 1.5,
        shuffle: bool = True,
        seed: int = None
    ):
        """
        Args:
            X: Input data
                - numpy array için tek girdi: (N, features)
                - List için çoklu girdi: [X_f, X_50, X_200, ...]
                - Dict için: {'features': X_f, 'seq_50': X_50, ...}
            y: Target data
                - numpy array için tek çıktı: (N,) veya (N, 1)
                - Dict için çoklu çıktı: {'regression': y_reg, 'threshold': y_thr, ...}
            batch_size: Batch boyutu (çift sayı olmalı - %50/%50 için)
            threshold: Class ayırma eşiği (default: 1.5)
            shuffle: Her epoch sonunda shuffle yapılsın mı
            seed: Random seed (reproducibility için)
        """
        self.batch_size = batch_size
        self.threshold = threshold
        self.shuffle = shuffle
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # X verisi formatını belirle
        if isinstance(X, dict):
            self.X = X
            self.X_is_dict = True
            self.X_is_list = False
            # İlk key'den sample sayısını al
            self.n_samples = len(list(X.values())[0])
        elif isinstance(X, list):
            self.X = X
            self.X_is_dict = False
            self.X_is_list = True
            self.n_samples = len(X[0])
        else:
            self.X = X
            self.X_is_dict = False
            self.X_is_list = False
            self.n_samples = len(X)
        
        # y verisi formatını belirle
        if isinstance(y, dict):
            self.y = y
            self.y_is_dict = True
            # Threshold için regression veya threshold output'u kullan
            if 'threshold' in y:
                y_for_split = y['threshold'].flatten()
            elif 'regression' in y:
                y_for_split = y['regression'].flatten()
            else:
                # İlk değeri kullan
                y_for_split = list(y.values())[0].flatten()
        else:
            self.y = y
            self.y_is_dict = False
            y_for_split = y.flatten() if len(y.shape) > 1 else y
        
        # 1.5 altı ve üstü indekslerini belirle
        self.below_idx = np.where(y_for_split < threshold)[0]
        self.above_idx = np.where(y_for_split >= threshold)[0]
        
        # Batch sayısını hesapla
        self.n_batches = int(np.floor(self.n_samples / batch_size))
        
        # Her batch'ten kaç tane altı/üstü alınacak
        self.n_below_per_batch = batch_size // 2
        self.n_above_per_batch = batch_size - self.n_below_per_batch
        
        print(f"✅ BalancedBatchGenerator oluşturuldu:")
        print(f"   • Toplam örnek: {self.n_samples}")
        print(f"   • 1.5 altı: {len(self.below_idx)} ({len(self.below_idx)/self.n_samples*100:.1f}%)")
        print(f"   • 1.5 üstü: {len(self.above_idx)} ({len(self.above_idx)/self.n_samples*100:.1f}%)")
        print(f"   • Batch size: {batch_size} (her batch'te {self.n_below_per_batch} altı, {self.n_above_per_batch} üstü)")
        print(f"   • Batch sayısı: {self.n_batches}")
    
    def __len__(self):
        """Batch sayısını döndür"""
        return self.n_batches
    
    def __getitem__(self, idx):
        """
        Belirtilen index'teki batch'i döndür
        
        Args:
            idx: Batch index
            
        Returns:
            (X_batch, y_batch) tuple
        """
        # Her batch için rastgele sampling yap
        # 1.5 altı: replace=True (az olduğu için tekrar seçilebilir)
        batch_below = np.random.choice(
            self.below_idx,
            size=self.n_below_per_batch,
            replace=True
        )
        
        # 1.5 üstü: replace=False (çok olduğu için her seferinde farklı)
        # Eğer 1.5 üstü sayısı batch için yetersizse replace=True yap
        replace_above = len(self.above_idx) < self.n_above_per_batch
        batch_above = np.random.choice(
            self.above_idx,
            size=self.n_above_per_batch,
            replace=replace_above
        )
        
        # İndeksleri birleştir ve shuffle et
        batch_idx = np.concatenate([batch_below, batch_above])
        np.random.shuffle(batch_idx)
        
        # X verilerini al
        if self.X_is_dict:
            X_batch = {key: val[batch_idx] for key, val in self.X.items()}
        elif self.X_is_list:
            X_batch = [x[batch_idx] for x in self.X]
        else:
            X_batch = self.X[batch_idx]
        
        # y verilerini al
        if self.y_is_dict:
            y_batch = {key: val[batch_idx] for key, val in self.y.items()}
        else:
            y_batch = self.y[batch_idx]
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        """Her epoch sonunda çağrılır - shuffle için"""
        if self.shuffle:
            # Sadece seed'i yenile (her epoch farklı sampling olsun)
            if self.seed is not None:
                 np.random.seed(self.seed + np.random.randint(0, 10000))
            else:
                 np.random.seed(np.random.randint(0, 10000))
    
    def get_stats(self):
        """Generator istatistiklerini döndür"""
        return {
            'total_samples': self.n_samples,
            'below_threshold_count': len(self.below_idx),
            'above_threshold_count': len(self.above_idx),
            'below_threshold_ratio': len(self.below_idx) / self.n_samples,
            'batch_size': self.batch_size,
            'batches_per_epoch': self.n_batches,
            'samples_per_batch_below': self.n_below_per_batch,
            'samples_per_batch_above': self.n_above_per_batch
        }

# Kullanım örneği
if __name__ == "__main__":
    # Örnek veri
    X_features = np.random.randn(1000, 50)
    X_seq = np.random.randn(1000, 100, 1)
    y_reg = np.random.rand(1000) * 10  # 0-10 arası değerler
    
    # %35 1.5 altı, %65 1.5 üstü olacak şekilde ayarla
    below_count = int(0.35 * 1000)
    y_reg[:below_count] = np.random.rand(below_count) * 1.4  # 1.5 altı
    y_reg[below_count:] = 1.5 + np.random.rand(1000 - below_count) * 8.5  # 1.5 üstü
    
    # Shuffle et
    idx = np.arange(1000)
    np.random.shuffle(idx)
    X_features = X_features[idx]
    X_seq = X_seq[idx]
    y_reg = y_reg[idx]
    
    print("Örnek veri oluşturuldu:")
    print(f"1.5 altı: {(y_reg < 1.5).sum()} ({(y_reg < 1.5).sum()/len(y_reg)*100:.1f}%)")
    print(f"1.5 üstü: {(y_reg >= 1.5).sum()} ({(y_reg >= 1.5).sum()/len(y_reg)*100:.1f}%)")
    
    # Generator oluştur - List formatında
    print("\n--- List formatında generator ---")
    gen = BalancedBatchGenerator(
        X=[X_features, X_seq],
        y=y_reg,
        batch_size=32,
        shuffle=True,
        seed=42
    )
    
    # İlk batch'i al
    X_batch, y_batch = gen[0]
    print(f"\nİlk batch:")
    print(f"X_batch[0] shape: {X_batch[0].shape}")
    print(f"X_batch[1] shape: {X_batch[1].shape}")
    print(f"y_batch shape: {y_batch.shape}")
    print(f"1.5 altı: {(y_batch < 1.5).sum()} ({(y_batch < 1.5).sum()/len(y_batch)*100:.1f}%)")
    print(f"1.5 üstü: {(y_batch >= 1.5).sum()} ({(y_batch >= 1.5).sum()/len(y_batch)*100:.1f}%)")
    
    # Generator istatistikleri
    stats = gen.get_stats()
    print(f"\nGenerator istatistikleri:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
