"""
JetX Predictor - Ultra Aggressive Custom Loss Fonksiyonları

Bu modül "Ultra Aggressive" model için özel olarak tasarlanmış loss fonksiyonlarını içerir.
Para kaybını önlemek (False Positive cezası) ile öğrenme kapasitesini (Lazy Learning önleme)
arasındaki dengeyi kurar.

GÜNCELLEME:
- Lazy Learning'i önlemek için aşırı cezalar (12x) optimize edildi (2.5x).
- Threshold Manager ile uyumlu yapı.
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from utils.threshold_manager import get_loss_penalty

def ultra_threshold_killer_loss(y_true, y_pred):
    """
    ULTRA BALANCED Threshold Killer Loss
    
    Para kaybını önlemek için tasarlanmıştır ancak modelin "hiçbir zaman oynama" 
    tuzağına (Lazy Learning) düşmemesi için cezalar dengelenmiştir.
    
    Katsayılar:
    - False Positive (Para Kaybı): 2.5x (Eski: 12.0x) - Yeterince caydırıcı ama öğrenmeye izin verir.
    - False Negative (Fırsat): 1.5x (Eski: 6.0x)
    - Kritik Bölge (1.4-1.6): 3.0x (Eski: 10.0x)
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin edilen değerler
        
    Returns:
        Weighted MAE loss
    """
    mae = K.abs(y_true - y_pred)
    
    # Katsayılar (Optimize Edilmiş Sabitler)
    # Config'den almak yerine bu modelin karakteristiği olan değerleri koruyoruz
    FP_PENALTY = 2.5
    FN_PENALTY = 1.5
    CRITICAL_PENALTY = 3.0
    
    # 1.5 altıyken üstü tahmin = PARA KAYBI
    false_positive = K.cast(
        tf.logical_and(y_true < 1.5, y_pred >= 1.5),
        'float32'
    ) * FP_PENALTY
    
    # 1.5 üstüyken altı tahmin = FIRSAT KAÇIRMA
    false_negative = K.cast(
        tf.logical_and(y_true >= 1.5, y_pred < 1.5),
        'float32'
    ) * FN_PENALTY
    
    # Kritik bölge (1.4-1.6)
    critical_zone = K.cast(
        tf.logical_and(y_true >= 1.4, y_true <= 1.6),
        'float32'
    ) * CRITICAL_PENALTY
    
    # Maksimum cezayı uygula
    weight = K.maximum(K.maximum(false_positive, false_negative), critical_zone)
    weight = K.maximum(weight, 1.0)  # Minimum 1.0 (normal MAE)
    
    return K.mean(mae * weight)


def ultra_focal_loss(gamma=2.0, alpha=0.75):
    """
    ULTRA Focal Loss
    
    Dengesiz veri setleri için optimize edilmiştir.
    gamma=2.0 (Daha stabil gradyanlar için 3.0'dan düşürüldü)
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
    """
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        bce = -(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        
        weights = y_true * weight_1 + (1 - y_true) * weight_0
        
        return K.mean(bce * weights)
    
    return loss


# =============================================================================
# CUSTOM OBJECTS - Model Yükleme İçin
# =============================================================================

ULTRA_CUSTOM_OBJECTS = {
    'ultra_threshold_killer_loss': ultra_threshold_killer_loss,
    'ultra_focal_loss': ultra_focal_loss(),
    'ultra_weighted_binary_crossentropy': ultra_weighted_binary_crossentropy,
    # Geriye dönük uyumluluk (eski isimler)
    'balanced_threshold_killer_loss': ultra_threshold_killer_loss
}
