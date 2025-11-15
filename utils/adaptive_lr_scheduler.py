"""
JetX Predictor - Adaptive Learning Rate Scheduler

Bu modül, lazy learning problemini çözmek için adaptif learning rate
scheduler'ları içerir. Modelin tutarlı öğrenememesi durumunda
learning rate'i otomatik olarak ayarlar.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LearningRateScheduler(ABC):
    """Learning Rate Scheduler için abstract base class"""
    
    @abstractmethod
    def __call__(self, epoch: int, logs: Dict[str, Any]) -> float:
        """Epoch sonunda learning rate'i hesapla"""
        pass
    
    @abstractmethod
    def get_scheduler_info(self) -> Dict[str, Any]:
        """Scheduler bilgilerini getir"""
        pass


class CosineAnnealingSchedule(LearningRateScheduler):
    """
    Cosine annealing ile adaptif learning rate scheduler.
    Learning rate'i cosine dalgası şeklinde değiştirir.
    """
    
    def __init__(
        self,
        initial_lr: float = 0.001,
        max_lr: float = 0.01,
        min_lr: float = 0.0001,
        cycles: float = 3.0,
        warmup_epochs: int = 5
    ):
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.cycles = cycles
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        
        logger.info(f"Cosine Annealing Scheduler başlatıldı:")
        logger.info(f"  Initial LR: {initial_lr}")
        logger.info(f"  Max LR: {max_lr}")
        logger.info(f"  Min LR: {min_lr}")
        logger.info(f"  Cycles: {cycles}")
        logger.info(f"  Warmup Epochs: {warmup_epochs}")
    
    def __call__(self, epoch: int, logs: Dict[str, Any]) -> float:
        """Cosine annealing ile learning rate hesapla"""
        self.current_epoch = epoch
        
        # Warmup döneminde sabit learning rate
        if epoch < self.warmup_epochs:
            return self.initial_lr
        
        # Cosine annealing hesapla
        progress = (epoch - self.warmup_epochs) / self.cycles
        cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
        
        # Learning rate'i hesapla
        lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_factor
        
        logger.debug(f"Epoch {epoch}: Progress={progress:.3f}, Cosine Factor={cosine_factor:.3f}, LR={lr:.6f}")
        
        return lr
    
    def get_scheduler_info(self) -> Dict[str, Any]:
        """Scheduler bilgilerini getir"""
        return {
            'type': 'CosineAnnealing',
            'initial_lr': self.initial_lr,
            'max_lr': self.max_lr,
            'min_lr': self.min_lr,
            'cycles': self.cycles,
            'warmup_epochs': self.warmup_epochs,
            'current_epoch': self.current_epoch
        }


class AdaptiveLearningRateScheduler(LearningRateScheduler):
    """
    Model performansına göre learning rate'i adapte eden scheduler.
    Stability score'a göre learning rate'i artırır veya azaltır.
    """
    
    def __init__(
        self,
        initial_lr: float = 0.001,
        max_lr: float = 0.01,
        min_lr: float = 0.0001,
        patience: int = 5,
        factor: float = 0.5,
        improvement_threshold: float = 0.01,
        reduction_factor: float = 0.5,
        warmup_epochs: int = 3,
        stability_window: int = 10,
        min_lr_after_plateau: float = 0.0001
    ):
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.patience = patience
        self.factor = factor
        self.improvement_threshold = improvement_threshold
        self.reduction_factor = reduction_factor
        self.warmup_epochs = warmup_epochs
        self.stability_window = stability_window
        self.min_lr_after_plateau = min_lr_after_plateau
        
        # Internal state
        self.current_lr = initial_lr
        self.best_score = float('-inf')
        self.patience_counter = 0
        self.stability_scores = []
        self.plateau_count = 0
        self.in_plateau = False
        
        logger.info(f"Adaptive LR Scheduler başlatıldı:")
        logger.info(f"  Initial LR: {initial_lr}")
        logger.info(f"  Max LR: {max_lr}")
        logger.info(f"  Min LR: {min_lr}")
        logger.info(f"  Patience: {patience}")
        logger.info(f"  Factor: {factor}")
        logger.info(f"  Stability Window: {stability_window}")
    
    def calculate_stability_score(self, logs: Dict[str, Any]) -> float:
        """Son epoch'lar için stability score'u hesapla"""
        try:
            # Validation accuracy'leri al
            if 'val_accuracy' in logs:
                accuracies = [logs[f'val_accuracy'] for _ in range(max(0, len(logs) - self.stability_window))]
                if len(accuracies) >= 3:
                    # Son epoch'ların ortalama ve standart sapma
                    mean_accuracy = np.mean(accuracies)
                    std_accuracy = np.std(accuracies)
                    
                    # Stability score: düşük volatilite = yüksek stability
                    stability = 1.0 - (std_accuracy / (mean_accuracy + 1e-6))
                    
                    logger.debug(f"Stability hesapla: Mean={mean_accuracy:.4f}, Std={std_accuracy:.4f}, Score={stability:.4f}")
                    return stability
            
            return 0.5  # Varsayılan stability
            
        except Exception as e:
            logger.error(f"Stability score hesaplama hatası: {e}")
            return 0.5
    
    def __call__(self, epoch: int, logs: Dict[str, Any]) -> float:
        """Adaptif learning rate hesapla"""
        self.current_epoch = epoch
        
        # Stability score'u hesapla
        current_stability = self.calculate_stability_score(logs)
        
        # Stability score'u geçmişe ekle
        self.stability_scores.append(current_stability)
        if len(self.stability_scores) > self.stability_window:
            self.stability_scores.pop(0)  # En eski score'u çıkar
        
        # İyileşme varsa
        if current_stability > self.best_score + self.improvement_threshold:
            # İyileşme tespit edildi
            self.best_score = current_stability
            self.patience_counter = 0
            self.plateau_count = 0
            self.in_plateau = False
            
            # Learning rate'i artır
            new_lr = min(self.current_lr * self.factor, self.max_lr)
            logger.info(f"Epoch {epoch}: İyileşme tespit edildi! LR artırılıyor: {self.current_lr:.6f} -> {new_lr:.6f}")
            
        else:
            self.patience_counter += 1
            
            # Plateau kontrolü
            if self.patience_counter >= self.patience:
                if not self.in_plateau:
                    self.in_plateau = True
                    self.plateau_count += 1
                    logger.info(f"Epoch {epoch}: Plateau başladı (count: {self.plateau_count})")
                
                # Plateau'da learning rate'i azalt
                if self.in_plateau:
                    # Plateau süresine göre azaltma faktörü
                    plateau_factor = self.reduction_factor ** (self.plateau_count // 2)
                    plateau_factor = min(plateau_factor, 0.1)  # Minimum %10
                    
                    new_lr = max(self.current_lr * plateau_factor, self.min_lr_after_plateau)
                    
                    if new_lr != self.current_lr:
                        logger.info(f"Epoch {epoch}: Plateau'da LR azaltılıyor: {self.current_lr:.6f} -> {new_lr:.6f}")
                        
                        # Plateau'dan çıkış için reset
                        if self.plateau_count >= 3:  # 3 plateau sonrası
                            self.in_plateau = False
                            self.plateau_count = 0
                            self.patience_counter = 0
                            logger.info("Plateau resetlendi, normal iyileşme moduna geçildi")
                    
                    self.current_lr = new_lr
        
        logger.debug(f"Epoch {epoch}: Stability={current_stability:.4f}, Best={self.best_score:.4f}, Patience={self.patience_counter}, LR={self.current_lr:.6f}")
        
        return self.current_lr
    
    def get_scheduler_info(self) -> Dict[str, Any]:
        """Scheduler bilgilerini getir"""
        return {
            'type': 'AdaptiveLearningRateScheduler',
            'initial_lr': self.initial_lr,
            'max_lr': self.max_lr,
            'min_lr': self.min_lr,
            'patience': self.patience,
            'factor': self.factor,
            'improvement_threshold': self.improvement_threshold,
            'reduction_factor': self.reduction_factor,
            'warmup_epochs': self.warmup_epochs,
            'stability_window': self.stability_window,
            'min_lr_after_plateau': self.min_lr_after_plateau,
            'current_epoch': self.current_epoch,
            'current_lr': self.current_lr,
            'best_score': self.best_score,
            'patience_counter': self.patience_counter,
            'plateau_count': self.plateau_count,
            'in_plateau': self.in_plateau,
            'stability_scores': self.stability_scores[-5:] if self.stability_scores else []
        }


class PlateauDetectionScheduler(LearningRateScheduler):
    """
    Plateau detection'e odaklanmış basit scheduler.
    Model performansı düzelmediğinde learning rate'i düşürür.
    """
    
    def __init__(
        self,
        initial_lr: float = 0.001,
        reduction_factor: float = 0.5,
        patience: int = 10,
        min_lr: float = 0.0001,
        metric: str = 'val_loss',
        min_delta: float = 0.001,
        cooldown: int = 5,
    ):
        self.initial_lr = initial_lr
        self.reduction_factor = reduction_factor
        self.patience = patience
        self.min_lr = min_lr
        self.metric = metric
        self.min_delta = min_delta
        self.cooldown = cooldown
        
        # Internal state
        self.current_lr = initial_lr
        self.best_loss = float('inf')
        self.wait_count = 0
        self.last_reduction_epoch = 0
        self.cooldown_counter = 0
        
        logger.info(f"Plateau Detection Scheduler başlatıldı:")
        logger.info(f"  Initial LR: {initial_lr}")
        logger.info(f"  Reduction Factor: {reduction_factor}")
        logger.info(f"  Patience: {patience}")
        logger.info(f"  Min LR: {min_lr}")
    
    def __call__(self, epoch: int, logs: Dict[str, Any]) -> float:
        """Plateau detection ile learning rate ayarla"""
        self.current_epoch = epoch
        
        # Metric değerini al
        current_loss = logs.get(self.metric, float('inf'))
        
        # İyileşme kontrolü
        if current_loss < self.best_loss - self.min_delta:
            # İyileşme
            self.best_loss = current_loss
            self.wait_count = 0
            self.last_reduction_epoch = epoch
            self.cooldown_counter = 0
            
            logger.info(f"Epoch {epoch}: Loss iyileşti ({current_loss:.6f}), LR sabit tutuluyor: {self.current_lr:.6f}")
            
        else:
            self.wait_count += 1
            
            # Patience dolunca ve cooldown bittiyse LR azalt
            if (self.wait_count >= self.patience and 
                epoch - self.last_reduction_epoch >= self.cooldown and 
                self.cooldown_counter >= self.cooldown):
                
                new_lr = max(self.current_lr * self.reduction_factor, self.min_lr)
                
                if new_lr != self.current_lr:
                    self.current_lr = new_lr
                    self.last_reduction_epoch = epoch
                    self.cooldown_counter = 0
                    
                    logger.info(f"Epoch {epoch}: Plateau tespit edildi! LR azaltılıyor: {self.current_lr:.6f} -> {new_lr:.6f}")
        
        logger.debug(f"Epoch {epoch}: Loss={current_loss:.6f}, Best={self.best_loss:.6f}, Wait={self.wait_count}, LR={self.current_lr:.6f}")
        
        return self.current_lr
    
    def get_scheduler_info(self) -> Dict[str, Any]:
        """Scheduler bilgilerini getir"""
        return {
            'type': 'PlateauDetectionScheduler',
            'initial_lr': self.initial_lr,
            'reduction_factor': self.reduction_factor,
            'patience': self.patience,
            'min_lr': self.min_lr,
            'metric': self.metric,
            'min_delta': self.min_delta,
            'cooldown': self.cooldown,
            'current_epoch': self.current_epoch,
            'current_lr': self.current_lr,
            'best_loss': self.best_loss,
            'wait_count': self.wait_count,
            'last_reduction_epoch': self.last_reduction_epoch,
            'cooldown_counter': self.cooldown_counter
        }


class LearningRateSchedulerFactory:
    """Learning rate scheduler factory"""
    
    _schedulers = {
        'cosine': CosineAnnealingSchedule,
        'adaptive': AdaptiveLearningRateScheduler,
        'plateau': PlateauDetectionScheduler
    }
    
    @classmethod
    def create_scheduler(
        cls,
        scheduler_type: str,
        **kwargs
    ) -> LearningRateScheduler:
        """
        Scheduler tipine göre uygun scheduler'ı oluştur
        
        Args:
            scheduler_type: Scheduler tipi ('cosine', 'adaptive', 'plateau')
            **kwargs: Scheduler'a özel parametreler
            
        Returns:
            LearningRateScheduler instance
        """
        if scheduler_type not in cls._schedulers:
            raise ValueError(f"Bilinmeyen scheduler tipi: {scheduler_type}")
        
        scheduler_class = cls._schedulers[scheduler_type]
        return scheduler_class(**kwargs)
    
    @classmethod
    def get_available_schedulers(cls) -> List[str]:
        """Mevcut scheduler tiplerini getir"""
        return list(cls._schedulers.keys())
    
    @classmethod
    def get_scheduler_info(cls, scheduler: LearningRateScheduler) -> Dict[str, Any]:
        """Scheduler'ın detaylı bilgilerini getir"""
        return scheduler.get_scheduler_info()


# Easy-to-use functions
def create_cosine_scheduler(**kwargs) -> CosineAnnealingSchedule:
    """Cosine annealing scheduler oluştur"""
    return CosineAnnealingSchedule(**kwargs)


def create_adaptive_scheduler(**kwargs) -> AdaptiveLearningRateScheduler:
    """Adaptive learning rate scheduler oluştur"""
    return AdaptiveLearningRateScheduler(**kwargs)


def create_plateau_scheduler(**kwargs) -> PlateauDetectionScheduler:
    """Plateau detection scheduler oluştur"""
    return PlateauDetectionScheduler(**kwargs)


if __name__ == "__main__":
    # Test
    print("Adaptive Learning Rate Scheduler Test")
    
    # Cosine scheduler test
    cosine_scheduler = create_cosine_scheduler(
        initial_lr=0.001,
        max_lr=0.01,
        min_lr=0.0001,
        cycles=2.0
    )
    
    print("Cosine Scheduler Info:")
    info = cosine_scheduler.get_scheduler_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test epochs
    test_logs = [
        {'val_accuracy': 0.65, 'val_loss': 0.5},
        {'val_accuracy': 0.68, 'val_loss': 0.45},
        {'val_accuracy': 0.70, 'val_loss': 0.40},
        {'val_accuracy': 0.72, 'val_loss': 0.38},
        {'val_accuracy': 0.69, 'val_loss': 0.42},
    ]
    
    print("\nTest Epochs:")
    for epoch, logs in enumerate(test_logs):
        lr = cosine_scheduler(epoch, logs)
        print(f"  Epoch {epoch}: LR = {lr:.6f}")
    
    print("\nAdaptive Scheduler Test:")
    adaptive_scheduler = create_adaptive_scheduler(
        initial_lr=0.001,
        max_lr=0.01,
        min_lr=0.0001,
        patience=3,
        factor=0.5
    )
    
    print("Adaptive Scheduler Info:")
    info = adaptive_scheduler.get_scheduler_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test with stability scores
    test_logs_with_stability = [
        {'val_accuracy': 0.65, 'val_loss': 0.5, 'stability_score': 0.75},
        {'val_accuracy': 0.68, 'val_loss': 0.45, 'stability_score': 0.80},
        {'val_accuracy': 0.70, 'val_loss': 0.40, 'stability_score': 0.85},
        {'val_accuracy': 0.72, 'val_loss': 0.38, 'stability_score': 0.82},
        {'val_accuracy': 0.69, 'val_loss': 0.42, 'stability_score': 0.78},
    ]
    
    print("\nAdaptive Scheduler Test with Stability:")
    for epoch, logs in enumerate(test_logs_with_stability):
        lr = adaptive_scheduler(epoch, logs)
        print(f"  Epoch {epoch}: LR = {lr:.6f}")
