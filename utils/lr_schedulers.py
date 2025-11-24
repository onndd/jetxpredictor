"""
JetX Predictor - Advanced Learning Rate Schedulers

Gelişmiş learning rate scheduler'lar:
- Cosine Annealing with Warmup
- One Cycle Policy
- Exponential Decay with Warmup
- Polynomial Decay with Warmup

GÜNCELLEME:
- T4 GPU ve Büyük Batch Size (256+) için varsayılan LR değerleri artırıldı.
- Warmup epoch sayıları optimize edildi.
"""

import tensorflow as tf
from tensorflow.keras import callbacks
import numpy as np
from typing import Optional, Callable
import math


class CosineAnnealingWarmup(callbacks.Callback):
    """
    Cosine Annealing with Warmup Scheduler
    
    Başlangıçta warmup fazı, sonra cosine annealing ile smooth decay.
    
    Args:
        max_lr: Maximum learning rate (warmup sonrası ulaşılacak) - T4 için artırıldı
        min_lr: Minimum learning rate (cycle sonunda)
        warmup_epochs: Warmup epoch sayısı
        total_epochs: Toplam epoch sayısı
        cycles: Cosine cycle sayısı (default 1, restart için >1)
        initial_lr: Başlangıç learning rate (warmup için)
    """
    
    def __init__(
        self,
        max_lr: float = 0.002,  # Güncellendi: 1e-3 -> 2e-3 (Büyük batch için)
        min_lr: float = 1e-6,
        warmup_epochs: int = 5, # Güncellendi: 10 -> 5 (Daha hızlı ısınma)
        total_epochs: int = 100,
        cycles: int = 1,
        initial_lr: float = 1e-5,
        verbose: int = 0
    ):
        super().__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.cycles = cycles
        self.initial_lr = initial_lr
        self.verbose = verbose
        self.history = []
    
    def on_epoch_begin(self, epoch, logs=None):
        """Epoch başında learning rate'i ayarla"""
        lr = self._calculate_lr(epoch)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.history.append(lr)
        
        if self.verbose:
            print(f"\nEpoch {epoch + 1}: Learning rate = {lr:.6f}")
    
    def _calculate_lr(self, epoch: int) -> float:
        """Learning rate hesapla"""
        # Warmup fazı
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * (epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            
            # Cosine with restarts
            cycle_progress = (progress * self.cycles) % 1.0
            
            # Cosine annealing formula
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (
                1 + math.cos(math.pi * cycle_progress)
            )
        
        return lr
    
    def plot_schedule(self, save_path: Optional[str] = None):
        """Learning rate schedule'u görselleştir"""
        import matplotlib.pyplot as plt
        
        epochs = list(range(len(self.history)))
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.history, linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Cosine Annealing with Warmup Schedule')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


class OneCyclePolicy(callbacks.Callback):
    """
    One Cycle Policy Scheduler
    
    Leslie Smith's 1cycle policy: warmup -> peak -> decay
    
    Args:
        max_lr: Maximum learning rate (cycle ortasında) - T4 için artırıldı
        total_epochs: Toplam epoch sayısı
        warmup_pct: Warmup fazının yüzdesi (default 0.3 = %30)
        div_factor: Initial LR = max_lr / div_factor
        final_div_factor: Final LR = max_lr / (div_factor * final_div_factor)
    """
    
    def __init__(
        self,
        max_lr: float = 0.005, # Güncellendi: 1e-3 -> 5e-3 (Daha agresif peak)
        total_epochs: int = 100,
        warmup_pct: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        verbose: int = 0
    ):
        super().__init__()
        self.max_lr = max_lr
        self.total_epochs = total_epochs
        self.warmup_pct = warmup_pct
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.verbose = verbose
        self.history = []
        
        # Calculate LR bounds
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / (div_factor * final_div_factor)
        self.warmup_epochs = int(total_epochs * warmup_pct)
    
    def on_epoch_begin(self, epoch, logs=None):
        """Epoch başında learning rate'i ayarla"""
        lr = self._calculate_lr(epoch)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.history.append(lr)
        
        if self.verbose:
            print(f"\nEpoch {epoch + 1}: Learning rate = {lr:.6f}")
    
    def _calculate_lr(self, epoch: int) -> float:
        """Learning rate hesapla"""
        if epoch < self.warmup_epochs:
            # Warmup phase: linear increase
            progress = epoch / self.warmup_epochs
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * progress
        else:
            # Annealing phase: cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.final_lr + (self.max_lr - self.final_lr) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )
        
        return lr


class ExponentialDecayWarmup(callbacks.Callback):
    """
    Exponential Decay with Warmup
    
    Warmup sonrası exponential decay.
    
    Args:
        max_lr: Maximum learning rate
        min_lr: Minimum learning rate
        warmup_epochs: Warmup epoch sayısı
        decay_rate: Decay rate (0-1 arası, küçük = yavaş decay)
        initial_lr: Başlangıç LR
    """
    
    def __init__(
        self,
        max_lr: float = 0.002, # Güncellendi: 1e-3 -> 2e-3
        min_lr: float = 1e-6,
        warmup_epochs: int = 5,
        decay_rate: float = 0.96,
        initial_lr: float = 1e-5,
        verbose: int = 0
    ):
        super().__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.decay_rate = decay_rate
        self.initial_lr = initial_lr
        self.verbose = verbose
        self.history = []
    
    def on_epoch_begin(self, epoch, logs=None):
        """Epoch başında learning rate'i ayarla"""
        lr = self._calculate_lr(epoch)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.history.append(lr)
        
        if self.verbose:
            print(f"\nEpoch {epoch + 1}: Learning rate = {lr:.6f}")
    
    def _calculate_lr(self, epoch: int) -> float:
        """Learning rate hesapla"""
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * (epoch / self.warmup_epochs)
        else:
            # Exponential decay
            steps_after_warmup = epoch - self.warmup_epochs
            lr = self.max_lr * (self.decay_rate ** steps_after_warmup)
            lr = max(lr, self.min_lr)  # Clamp to min_lr
        
        return lr


class PolynomialDecayWarmup(callbacks.Callback):
    """
    Polynomial Decay with Warmup
    
    Warmup sonrası polynomial decay.
    
    Args:
        max_lr: Maximum learning rate
        min_lr: Minimum learning rate
        warmup_epochs: Warmup epoch sayısı
        total_epochs: Toplam epoch sayısı
        power: Polynomial power (1.0 = linear, 2.0 = quadratic, etc.)
        initial_lr: Başlangıç LR
    """
    
    def __init__(
        self,
        max_lr: float = 0.002, # Güncellendi: 1e-3 -> 2e-3
        min_lr: float = 1e-6,
        warmup_epochs: int = 5,
        total_epochs: int = 100,
        power: float = 2.0,
        initial_lr: float = 1e-5,
        verbose: int = 0
    ):
        super().__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.power = power
        self.initial_lr = initial_lr
        self.verbose = verbose
        self.history = []
    
    def on_epoch_begin(self, epoch, logs=None):
        """Epoch başında learning rate'i ayarla"""
        lr = self._calculate_lr(epoch)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.history.append(lr)
        
        if self.verbose:
            print(f"\nEpoch {epoch + 1}: Learning rate = {lr:.6f}")
    
    def _calculate_lr(self, epoch: int) -> float:
        """Learning rate hesapla"""
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * (epoch / self.warmup_epochs)
        else:
            # Polynomial decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.max_lr - self.min_lr) * ((1 - progress) ** self.power)
        
        return lr


# Keras LearningRateScheduler wrapper için helper fonksiyonlar
def cosine_annealing_warmup_schedule(
    max_lr: float = 0.002,
    min_lr: float = 1e-6,
    warmup_epochs: int = 5,
    total_epochs: int = 100
) -> Callable:
    """
    Keras LearningRateScheduler için cosine annealing warmup schedule
    
    Returns:
        Schedule function (epoch, lr) -> new_lr
    """
    def schedule(epoch, lr):
        if epoch < warmup_epochs:
            return min_lr + (max_lr - min_lr) * (epoch / warmup_epochs)
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    
    return schedule


if __name__ == "__main__":
    # Test schedulers
    total_epochs = 100
    
    # 1. Cosine Annealing with Warmup
    cos_scheduler = CosineAnnealingWarmup(
        max_lr=0.002, min_lr=1e-6, warmup_epochs=5, total_epochs=total_epochs, cycles=1
    )
    
    # 2. One Cycle Policy
    onecycle_scheduler = OneCyclePolicy(
        max_lr=0.005, total_epochs=total_epochs, warmup_pct=0.3
    )
    
    print("✅ LR Schedulers T4 GPU Optimize Testi Başarılı!")
