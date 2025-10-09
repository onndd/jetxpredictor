"""
JetX Predictor - Custom Loss Fonksiyonları

Bu modül eğitim ve tahmin sırasında kullanılan özel loss fonksiyonlarını içerir.
Hem model eğitiminde hem de model yüklemede kullanılır.
"""

import tensorflow as tf
from tensorflow.keras import backend as K


def threshold_killer_loss(y_true, y_pred):
    """
    1.5 altı yanlış tahmine ÇOK BÜYÜK CEZA
    
    Bu loss fonksiyonu modelin 1.5 altında yanlış tahmin yapmasını önlemek için tasarlanmıştır.
    Para kaybı riskini minimize eder.
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin edilen değerler
        
    Returns:
        Weighted MAE loss
    """
    mae = K.abs(y_true - y_pred)
    
    # 1.5 altıyken üstü tahmin = 12x ceza (PARA KAYBI!) - 3. Tur: 15→12 (Dengeli)
    false_positive = K.cast(
        tf.logical_and(y_true < 1.5, y_pred >= 1.5),
        'float32'
    ) * 12.0
    
    # 1.5 üstüyken altı tahmin = 6x ceza - 3. Tur: 8→6 (Dengeli)
    false_negative = K.cast(
        tf.logical_and(y_true >= 1.5, y_pred < 1.5),
        'float32'
    ) * 6.0
    
    # Kritik bölge (1.4-1.6) = 10x ceza - 3. Tur: 12→10 (Hassas Bölge)
    critical_zone = K.cast(
        tf.logical_and(y_true >= 1.4, y_true <= 1.6),
        'float32'
    ) * 10.0
    
    weight = K.maximum(K.maximum(false_positive, false_negative), critical_zone)
    weight = K.maximum(weight, 1.0)
    
    return K.mean(mae * weight)


def ultra_focal_loss(gamma=5.0, alpha=0.85):
    """
    Focal loss - yanlış tahminlere çok büyük ceza
    
    Args:
        gamma: Focal loss parametresi (yüksek = daha agresif)
        alpha: Class balancing parametresi
        
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