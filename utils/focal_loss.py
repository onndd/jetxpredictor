"""
Focal Loss - Zor Örneklere Odaklanan Loss Fonksiyonu
Lazy learning problemini çözmek için kullanılır.

Focal Loss, kolay tahmin edilen örneklerin loss'unu azaltırken,
zor örneklerin loss'unu artırır. Bu sayede model minority class'ı
(1.5 altı veya üstü) daha iyi öğrenir.

Referans: https://arxiv.org/abs/1708.02002
"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# TensorFlow varsa import et
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow yüklü değil. Focal Loss (TF versiyonu) kullanılamayacak.")

# CatBoost için numpy tabanlı Focal Loss
class CatBoostFocalLoss(object):
    '''
    CatBoost için Focal Loss implementasyonu.
    Gradyan ve Hessian'ı numpy kullanarak hesaplar.
    '''
    def __init__(self, alpha=0.75, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma

    def calc_ders_range(self, approxes, targets, weights):
        # approxes: modelin ham çıktısı (logits)
        # targets: gerçek etiketler (0 veya 1)
        
        # Sigmoid fonksiyonu ile olasılığa çevir
        p = 1. / (1. + np.exp(-approxes))
        
        # Hata
        error = p - targets
        
        # Gradyan (loss'un birinci türevi)
        # Standart LogLoss gradyanı: p - y
        # Focal Loss modülasyon faktörü: alpha * (1-p)^gamma veya (1-alpha) * p^gamma
        # ÖNEMLİ: targets=0 (azınlık sınıfı) alpha ile ağırlıklandırılır
        grad_modulator = np.where(targets == 0, self.alpha * np.power(1. - p, self.gamma), (1. - self.alpha) * np.power(p, self.gamma))
        
        # Odaklanma teriminin türevi
        focus_term_grad = np.where(targets == 1, -self.gamma * p * np.log(p), self.gamma * (1. - p) * np.log(1. - p))
        
        # Gradyan
        grad = grad_modulator * (error + focus_term_grad)

        # Hessian (loss'un ikinci türevi)
        # Hessian'ı basitleştirmek için p*(1-p) kullanıyoruz (LogLoss'tan gelir)
        # Bu, eğitimin stabilitesini artırır.
        hess_modulator = np.where(targets == 0, self.alpha * np.power(1. - p, self.gamma - 1.) * (p * self.gamma * (1. - p) * np.log(p) + p - 1.), (1. - self.alpha) * np.power(p, self.gamma - 1.) * (1. - p * self.gamma * np.log(1. - p) - p))
        hess = p * (1. - p) * grad_modulator * hess_modulator

        return list(zip(grad, hess))

if TF_AVAILABLE:
    class FocalLoss(keras.losses.Loss):
        """
        Focal Loss implementasyonu
        
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        
        Args:
            alpha: Minority class ağırlığı (0-1). Varsayılan: 0.75
            gamma: Odaklanma parametresi. Yüksek gamma = zor örneklere daha fazla odaklan. Varsayılan: 2.0
            from_logits: True ise model output logits, False ise probabilities
            label_smoothing: Label smoothing faktörü (0-1)
        """
        
        def __init__(
            self,
            alpha: float = 0.75,
            gamma: float = 2.0,
            from_logits: bool = False,
            label_smoothing: float = 0.0,
            name: str = 'focal_loss'
        ):
            super().__init__(name=name)
            self.alpha = alpha
            self.gamma = gamma
            self.from_logits = from_logits
            self.label_smoothing = label_smoothing
        
        def call(self, y_true, y_pred):
            """
            Loss hesapla
            """
            # Label smoothing uygula
            if self.label_smoothing > 0:
                y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            
            # Logits'ten probability'ye çevir
            if self.from_logits:
                y_pred = tf.sigmoid(y_pred)
            
            # Numerical stability için clip
            epsilon = keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            # p_t hesapla: doğru sınıfın olasılığı
            p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            
            # Alpha_t hesapla: sınıf ağırlığı
            # ÖNEMLİ: y_true=0 (azınlık sınıfı) alpha ile ağırlıklandırılır
            alpha_t = tf.where(tf.equal(y_true, 0), self.alpha, 1 - self.alpha)
            
            # Focal weight: (1 - p_t)^gamma
            focal_weight = tf.pow(1.0 - p_t, self.gamma)
            
            # Cross entropy
            cross_entropy = -tf.math.log(p_t)
            
            # Focal loss = alpha_t * focal_weight * cross_entropy
            focal_loss = alpha_t * focal_weight * cross_entropy
            
            return tf.reduce_mean(focal_loss)
        
        def get_config(self):
            """Konfigürasyonu döndür"""
            config = super().get_config()
            config.update({
                'alpha': self.alpha,
                'gamma': self.gamma,
                'from_logits': self.from_logits,
                'label_smoothing': self.label_smoothing
            })
            return config


    def focal_loss_function(
        alpha: float = 0.75,
        gamma: float = 2.0,
        from_logits: bool = False
    ):
        """
        Focal loss function factory
        """
        def loss(y_true, y_pred):
            return FocalLoss(alpha=alpha, gamma=gamma, from_logits=from_logits)(y_true, y_pred)
        return loss


    class BinaryFocalLoss(keras.losses.Loss):
        """
        Binary classification için optimize edilmiş Focal Loss
        Progressive NN'nin threshold output'u için kullanılır.
        """
        
        def __init__(
            self,
            alpha: float = 0.75,
            gamma: float = 2.0,
            name: str = 'binary_focal_loss'
        ):
            super().__init__(name=name)
            self.alpha = alpha
            self.gamma = gamma
        
        def call(self, y_true, y_pred):
            """Binary focal loss hesapla"""
            # Sigmoid uygula (eğer logits ise, ama burada proba varsayıyoruz genelde)
            # Eğer model output 'linear' ise burası sorun çıkarabilir. 
            # Progressive NN 'sigmoid' activation kullanıyor, yani probability.
            
            # Clip for stability
            epsilon = keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            # Binary cross entropy
            bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
            
            # Focal weight
            p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            focal_weight = tf.pow(1.0 - p_t, self.gamma)
            
            # Alpha weight
            alpha_t = tf.where(tf.equal(y_true, 0), self.alpha, 1 - self.alpha)
            
            # Final loss
            focal_loss = alpha_t * focal_weight * bce
            
            return tf.reduce_mean(focal_loss)
        
        def get_config(self):
            config = super().get_config()
            config.update({
                'alpha': self.alpha,
                'gamma': self.gamma
            })
            return config


    class AdaptiveFocalLoss(keras.losses.Loss):
        """
        Adaptive Focal Loss - Eğitim sırasında parametreleri otomatik ayarlar
        
        Gamma parametresini epoch'lara göre ayarlar:
        - Başlangıç: Düşük gamma (kolay örneklere de odaklan)
        - Ortalar: Orta gamma (dengeli)
        - Son: Yüksek gamma (sadece zor örneklere odaklan)
        """
        
        def __init__(
            self,
            alpha: float = 0.75,
            gamma_start: float = 1.0,
            gamma_end: float = 3.0,
            name: str = 'adaptive_focal_loss'
        ):
            super().__init__(name=name)
            self.alpha = alpha
            self.gamma_start = gamma_start
            self.gamma_end = gamma_end
            self.current_gamma = tf.Variable(gamma_start, trainable=False, dtype=tf.float32)
        
        def update_gamma(self, epoch: int, total_epochs: int):
            """Gamma'yı epoch'a göre güncelle"""
            # Linear interpolation
            progress = epoch / max(1, total_epochs - 1)
            new_gamma = self.gamma_start + (self.gamma_end - self.gamma_start) * progress
            self.current_gamma.assign(new_gamma)
        
        def call(self, y_true, y_pred):
            """Adaptive focal loss hesapla"""
            epsilon = keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            alpha_t = tf.where(tf.equal(y_true, 0), self.alpha, 1 - self.alpha)
            
            focal_weight = tf.pow(1.0 - p_t, self.current_gamma)
            cross_entropy = -tf.math.log(p_t)
            
            focal_loss = alpha_t * focal_weight * cross_entropy
            return tf.reduce_mean(focal_loss)
        
        def get_config(self):
            config = super().get_config()
            config.update({
                'alpha': self.alpha,
                'gamma_start': self.gamma_start,
                'gamma_end': self.gamma_end
            })
            return config

    # Utility functions
    def create_focal_loss(
        loss_type: str = 'focal',
        alpha: float = 0.75,
        gamma: float = 2.0,
        **kwargs
    ) -> keras.losses.Loss:
        """
        Focal loss factory function
        """
        if loss_type == 'focal':
            return FocalLoss(alpha=alpha, gamma=gamma, **kwargs)
        elif loss_type == 'binary_focal':
            return BinaryFocalLoss(alpha=alpha, gamma=gamma)
        elif loss_type == 'adaptive_focal':
            gamma_start = kwargs.get('gamma_start', 1.0)
            gamma_end = kwargs.get('gamma_end', 3.0)
            return AdaptiveFocalLoss(alpha=alpha, gamma_start=gamma_start, gamma_end=gamma_end)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
