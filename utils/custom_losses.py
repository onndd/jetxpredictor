"""
JetX Predictor - Custom Loss Fonksiyonları

Bu modül eğitim ve tahmin sırasında kullanılan özel loss fonksiyonlarını içerir.
Hem model eğitiminde hem de model yüklemede kullanılır.
"""

import tensorflow as tf
from tensorflow.keras import backend as K


def threshold_killer_loss(y_true, y_pred):
    """
    1.5 altı yanlış tahmine DENGELI CEZA
    
    Bu loss fonksiyonu modelin 1.5 altında yanlış tahmin yapmasını önlemek için tasarlanmıştır.
    Para kaybı riskini minimize eder. YUMUŞATILMIŞ versiyonu - lazy learning'i önler.
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin edilen değerler
        
    Returns:
        Weighted MAE loss
    """
    mae = K.abs(y_true - y_pred)
    
    # 1.5 altıyken üstü tahmin = 2x ceza (PARA KAYBI - yumuşatıldı: 4→2)
    false_positive = K.cast(
        tf.logical_and(y_true < 1.5, y_pred >= 1.5),
        'float32'
    ) * 2.0
    
    # 1.5 üstüyken altı tahmin = 1.5x ceza (yumuşatıldı: 2→1.5)
    false_negative = K.cast(
        tf.logical_and(y_true >= 1.5, y_pred < 1.5),
        'float32'
    ) * 1.5
    
    # Kritik bölge (1.4-1.6) = 2.5x ceza (yumuşatıldı: 3→2.5)
    critical_zone = K.cast(
        tf.logical_and(y_true >= 1.4, y_true <= 1.6),
        'float32'
    ) * 2.5
    
    weight = K.maximum(K.maximum(false_positive, false_negative), critical_zone)
    weight = K.maximum(weight, 1.0)
    
    return K.mean(mae * weight)


def ultra_focal_loss(gamma=2.5, alpha=0.75):
    """
    Focal loss - yanlış tahminlere dengeli ceza (YUMUŞATILMIŞ)
    
    Args:
        gamma: Focal loss parametresi (yumuşatıldı: 5.0→2.5, daha dengeli)
        alpha: Class balancing parametresi (yumuşatıldı: 0.85→0.75)
        
    Returns:
        Loss fonksiyonu
    """
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = alpha * K.pow(1 - pt, gamma)
        return -K.mean(focal_weight * K.log(pt))
    return loss


# Custom objects dictionary for model loading
CUSTOM_OBJECTS = {
    'threshold_killer_loss': threshold_killer_loss,
    'ultra_focal_loss': ultra_focal_loss()
}
