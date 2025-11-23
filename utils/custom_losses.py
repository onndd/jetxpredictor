"""
JetX Predictor - Custom Loss Fonksiyonları (v2.0)

Bu modül eğitim ve tahmin sırasında kullanılan özel loss fonksiyonlarını içerir.
Hem model eğitiminde hem de model yüklemede kullanılır.

GÜNCELLEME:
- Threshold Manager entegrasyonu: Penalty değerleri config'den dinamik okunur.
- Lazy learning önleyici dengeli fonksiyonlar.
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from utils.threshold_manager import get_loss_penalty

# =============================================================================
# DİNAMİK LOSS FONKSİYONLARI (Threshold Manager Destekli)
# =============================================================================

def percentage_aware_regression_loss(y_true, y_pred):
    """
    YÜZDE HATAYA DAYALI REGRESSION LOSS
    
    Gerçek değerleri (2.7x, 3.09x, 10x, 15.77x vb.) daha doğru tahmin etmek için
    mutlak hata yerine yüzde hata kullanır.
    
    Mantık:
    - Gerçek: 10x, Tahmin: 5x → %50 hata → Yüksek ceza
    - Gerçek: 2x, Tahmin: 1.8x → %10 hata → Düşük ceza
    """
    epsilon = K.epsilon()
    
    # Yüzde hatayı hesapla: |gerçek - tahmin| / gerçek
    percentage_error = K.abs(y_true - y_pred) / (K.abs(y_true) + epsilon)
    
    # Yüksek değerler (5x+) için hafif ağırlık artışı
    high_value_weight = tf.where(y_true >= 5.0, 1.2, 1.0)
    
    weighted_percentage_error = percentage_error * high_value_weight
    
    return K.mean(weighted_percentage_error)


def balanced_threshold_killer_loss(y_true, y_pred):
    """
    DENGELI ve TUTARLI ceza sistemi - Lazy learning'i önler
    
    Penalty değerleri Threshold Manager üzerinden config dosyasından dinamik olarak alınır.
    Varsayılan: FP=5.0, FN=2.0, Critical=4.0
    """
    mae = K.abs(y_true - y_pred)
    
    # Config'den penalty değerlerini al (Yoksa varsayılanları kullanır)
    FALSE_POSITIVE_PENALTY = get_loss_penalty('false_positive_penalty') # Para Kaybı
    FALSE_NEGATIVE_PENALTY = get_loss_penalty('false_negative_penalty') # Fırsat Kaçırma
    # Critical zone penalty genellikle FP ve FN arasında bir değerdir veya config'den alınabilir
    # Şimdilik FP'ye yakın tutuyoruz.
    CRITICAL_ZONE_PENALTY = FALSE_POSITIVE_PENALTY * 0.8 
    
    # 1.5 altıyken üstü tahmin (PARA KAYBI)
    false_positive = K.cast(
        tf.logical_and(y_true < 1.5, y_pred >= 1.5),
        'float32'
    ) * FALSE_POSITIVE_PENALTY
    
    # 1.5 üstüyken altı tahmin (FIRSAT KAÇIRMA)
    false_negative = K.cast(
        tf.logical_and(y_true >= 1.5, y_pred < 1.5),
        'float32'
    ) * FALSE_NEGATIVE_PENALTY
    
    # Kritik bölge (1.4-1.6)
    critical_zone = K.cast(
        tf.logical_and(y_true >= 1.4, y_true <= 1.6),
        'float32'
    ) * CRITICAL_ZONE_PENALTY
    
    # Maksimum cezayı uygula
    weight = K.maximum(K.maximum(false_positive, false_negative), critical_zone)
    weight = K.maximum(weight, 1.0) # Minimum 1.0 (normal MAE)
    
    return K.mean(mae * weight)


def balanced_focal_loss(gamma=2.0, alpha=0.7):
    """
    DENGELI focal loss - Aşırı agresif değil, zor örneklere odaklanır.
    """
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = alpha * K.pow(1 - pt, gamma)
        return -K.mean(focal_weight * K.log(pt))
    return loss


def create_weighted_binary_crossentropy(weight_0, weight_1):
    """
    Sınıf ağırlıklarını doğrudan içeren weighted binary crossentropy
    Lazy learning'i önlemek için azınlık sınıfına (genellikle 1.5 altı veya üstü) daha fazla ağırlık verilir.
    """
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        bce = -(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        
        # Class weight'leri uygula
        weights = y_true * weight_1 + (1 - y_true) * weight_0
        
        return K.mean(bce * weights)
    
    return loss


# =============================================================================
# CUSTOM OBJECTS - Model Yükleme İçin
# =============================================================================
# Model yüklerken bu sözlük `custom_objects` parametresine verilmelidir.

CUSTOM_OBJECTS = {
    'percentage_aware_regression_loss': percentage_aware_regression_loss,
    'balanced_threshold_killer_loss': balanced_threshold_killer_loss,
    'balanced_focal_loss': balanced_focal_loss(), # Fonksiyon çağrısı ile instance döner
    'create_weighted_binary_crossentropy': create_weighted_binary_crossentropy,
    # Geriye dönük uyumluluk için eski isimler
    'threshold_killer_loss': balanced_threshold_killer_loss,
    'ultra_focal_loss': balanced_focal_loss()
}
