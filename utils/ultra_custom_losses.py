"""
JetX Predictor - Ultra Aggressive Custom Loss Fonksiyonları

Bu modül "Ultra Aggressive" model için özel olarak tasarlanmış çok agresif
loss fonksiyonlarını içerir. Bu fonksiyonlar para kaybını önlemek için
maksimum ceza sistemleri kullanır.

Farkları:
- 12x false positive ceza (balanced_threshold_killer_loss'ta 5x)
- gamma=3.0 focal loss (balanced_focal_loss'ta gamma=2.0)
- Daha agresif parametreler
"""

import tensorflow as tf
from tensorflow.keras import backend as K


def ultra_threshold_killer_loss(y_true, y_pred):
    """
    BALANCED Threshold Killer Loss - DENGELİ CEZA!
    
    Bu loss fonksiyonu para kaybını önlerken lazy learning'i de önler:
    - 1.5 altıyken üstü tahmin = 2.5x ceza (DÜZELTME: 12x → 2.5x)
    - 1.5 üstüyken altı tahmin = 1.5x ceza (D��ZELTME: 6x → 1.5x)
    - Kritik bölge (1.4-1.6) = 3.0x ceza (DÜZELTME: 10x → 3.0x)
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin edilen değerler
        
    Returns:
        Ultra weighted MAE loss
    """
    mae = K.abs(y_true - y_pred)
    
    # 1.5 altıyken üstü tahmin = 2.5x ceza (DÜZELTME: Lazy learning'i önle)
    false_positive = K.cast(
        tf.logical_and(y_true < 1.5, y_pred >= 1.5),
        'float32'
    ) * 2.5
    
    # 1.5 üstüyken altı tahmin = 1.5x ceza (DÜZELTME: Dengeli)
    false_negative = K.cast(
        tf.logical_and(y_true >= 1.5, y_pred < 1.5),
        'float32'
    ) * 1.5
    
    # Kritik bölge (1.4-1.6) = 3.0x ceza (DÜZELTME: Hassas bölge)
    critical_zone = K.cast(
        tf.logical_and(y_true >= 1.4, y_true <= 1.6),
        'float32'
    ) * 3.0
    
    # Maksimum cezayı uygula
    weight = K.maximum(K.maximum(false_positive, false_negative), critical_zone)
    weight = K.maximum(weight, 1.0)  # Minimum 1.0 (normal MAE)
    
    return K.mean(mae * weight)


def ultra_focal_loss(gamma=2.0, alpha=0.75):
    """
    BALANCED Focal Loss - gamma=2.0 (DÜZELTME)
    
    Args:
        gamma: Focal loss parametresi (2.0 - dengeli)
        alpha: Class balancing parametresi (0.75 - dengeli)
        
    Returns:
        Ultra focal loss fonksiyonu
    """
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = alpha * K.pow(1 - pt, gamma)
        return -K.mean(focal_weight * K.log(pt))
    return loss


def ultra_weighted_binary_crossentropy(weight_0, weight_1):
    """
    Ultra sınıf ağırlıklı binary crossentropy
    
    Args:
        weight_0: 1.5 altı (class 0) için ağırlık
        weight_1: 1.5 üstü (class 1) için ağırlık
    
    Returns:
        Ultra weighted binary crossentropy loss fonksiyonu
    """
    def loss(y_true, y_pred):
        # Binary crossentropy hesapla
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        bce = -(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        
        # Class weight'leri uygula
        weights = y_true * weight_1 + (1 - y_true) * weight_0
        
        # Ağırlıklı loss'u döndür
        return K.mean(bce * weights)
    
    return loss


# =============================================================================
# CUSTOM OBJECTS - Model Yükleme İçin
# =============================================================================

ULTRA_CUSTOM_OBJECTS = {
    'ultra_threshold_killer_loss': ultra_threshold_killer_loss,
    'ultra_focal_loss': ultra_focal_loss(),
    'ultra_weighted_binary_crossentropy': ultra_weighted_binary_crossentropy,
}
