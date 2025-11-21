"""
JetX Predictor - Custom Loss Fonksiyonları

Bu modül eğitim ve tahmin sırasında kullanılan özel loss fonksiyonlarını içerir.
Hem model eğitiminde hem de model yüklemede kullanılır.

Yeni dengeli loss functions lazy learning'i önler ve tutarlı ceza sistemi sunar.
"""

import tensorflow as tf
from tensorflow.keras import backend as K


# =============================================================================
# YENİ: DENGELI LOSS FUNCTIONS (Lazy Learning Çözümü)
# =============================================================================

def percentage_aware_regression_loss(y_true, y_pred):
    """
    YÜZDE HATAYA DAYALI REGRESSION LOSS
    
    Gerçek değerleri (2.7x, 3.09x, 10x, 15.77x vb.) daha doğru tahmin etmek için
    mutlak hata yerine yüzde hata kullanır.
    
    Mantık:
    - Gerçek: 10x, Tahmin: 5x → %50 hata → Yüksek ceza
    - Gerçek: 2x, Tahmin: 1.8x → %10 hata → Düşük ceza
    - Gerçek: 3.09x, Tahmin: 3.12x → %1 hata → Minimal ceza
    
    Bu şekilde model tüm değer aralıklarında eşit doğrulukla tahmin yapmaya zorlanır.
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin edilen değerler
        
    Returns:
        Yüzde hata bazlı loss (Mean Absolute Percentage Error benzeri)
    """
    # Sıfıra bölme hatasını önlemek için epsilon ekle
    epsilon = K.epsilon()
    
    # Yüzde hatayı hesapla: |gerçek - tahmin| / gerçek
    # Örnek: gerçek=10, tahmin=8 → |10-8|/10 = 0.2 = %20 hata
    percentage_error = K.abs(y_true - y_pred) / (K.abs(y_true) + epsilon)
    
    # Ekstra ağırlıklandırma: Yüksek değerler için biraz daha fazla önem
    # 5x altı: normal
    # 5x+ değerler: 1.2x ağırlık (hafif artış)
    high_value_weight = tf.where(
        y_true >= 5.0,
        1.2,  # Yüksek değerler için %20 daha fazla önem
        1.0   # Normal değerler için standart
    )
    
    # Weighted percentage error
    weighted_percentage_error = percentage_error * high_value_weight
    
    return K.mean(weighted_percentage_error)


def balanced_threshold_killer_loss(y_true, y_pred, 
                                   fp_penalty=5.0, fn_penalty=3.0, critical_penalty=4.0):
    """
    DENGELI ve TUTARLI ceza sistemi - Lazy learning'i önler
    
    GÜNCELLEME: Penalty değerleri artık parametrik - config'den alınıyor
    "Raporlama vs. Eylem" tutarsızlıkları önleniyor
    
    Bu loss fonksiyonu:
    - Tutarlı ceza değerleri kullanır (config'den gelir)
    - Lazy learning'i önler
    - Para kaybı riskini minimize eder
    - Balanced batch generator ile birlikte çalışır
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin edilen değerler
        fp_penalty: False positive penalty (PARA KAYBI)
        fn_penalty: False negative penalty (fırsat kaçırma)
        critical_penalty: Kritik bölge penalty
        
    Returns:
        Weighted MAE loss
    """
    mae = K.abs(y_true - y_pred)
    
    # Parametrik ceza çarpanları - ARTIK CONFIG'DEN GELİYOR
    FALSE_POSITIVE_PENALTY = fp_penalty  # 1.5 altıyken üstü tahmin (PARA KAYBI)
    FALSE_NEGATIVE_PENALTY = fn_penalty  # 1.5 üstüyken altı tahmin
    CRITICAL_ZONE_PENALTY = critical_penalty   # 1.4-1.6 arası (kritik bölge)
    
    # 1.5 altıyken üstü tahmin = 5x ceza (PARA KAYBI - en önemli)
    false_positive = K.cast(
        tf.logical_and(y_true < 1.5, y_pred >= 1.5),
        'float32'
    ) * FALSE_POSITIVE_PENALTY
    
    # 1.5 üstüyken altı tahmin = 3x ceza (fırsat kaçırma)
    false_negative = K.cast(
        tf.logical_and(y_true >= 1.5, y_pred < 1.5),
        'float32'
    ) * FALSE_NEGATIVE_PENALTY
    
    # Kritik bölge (1.4-1.6) = 4x ceza (hassas bölge)
    critical_zone = K.cast(
        tf.logical_and(y_true >= 1.4, y_true <= 1.6),
        'float32'
    ) * CRITICAL_ZONE_PENALTY
    
    # Maksimum cezayı uygula
    weight = K.maximum(K.maximum(false_positive, false_negative), critical_zone)
    weight = K.maximum(weight, 1.0)  # Minimum 1.0 (normal MAE)
    
    return K.mean(mae * weight)


def balanced_focal_loss(gamma=2.0, alpha=0.7):
    """
    DENGELI focal loss - Aşırı agresif değil
    
    Args:
        gamma: Focal loss parametresi (2.0 - standart, dengeli)
        alpha: Class balancing parametresi (0.7 - hafif baskı)
        
    Returns:
        Loss fonksiyonu
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
    
    Balanced batch generator ile birlikte kullanıldığında:
    - weight_0 = 1.0-2.0 (hafif - çünkü batch zaten dengeli)
    - weight_1 = 1.0 (baseline)
    
    Args:
        weight_0: 1.5 altı (class 0) için ağırlık
        weight_1: 1.5 üstü (class 1) için ağırlık
    
    Returns:
        Ağırlıklı binary crossentropy loss fonksiyonu
    """
    def loss(y_true, y_pred):
        # Binary crossentropy hesapla
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        bce = -(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        
        # Class weight'leri uygula
        # y_true = 1 ise weight_1, y_true = 0 ise weight_0 kullan
        weights = y_true * weight_1 + (1 - y_true) * weight_0
        
        # Ağırlıklı loss'u döndür
        return K.mean(bce * weights)
    
    return loss


# =============================================================================
# ESKİ: MEVCUT LOSS FUNCTIONS (Geriye Uyumluluk İçin)
# =============================================================================
# NOT: Artık balanced_* fonksiyonlarını kullanmanız önerilir

def threshold_killer_loss(y_true, y_pred):
    """
    [DEPRECATED] Eski versiyon - balanced_threshold_killer_loss kullanın
    
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
    [DEPRECATED] Eski versiyon - balanced_focal_loss kullanın
    
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


# =============================================================================
# CUSTOM OBJECTS - Model Yükleme İçin
# =============================================================================
# Hem yeni hem eski fonksiyonlar dahil (geriye uyumluluk)

CUSTOM_OBJECTS = {
    # Yeni dengeli fonksiyonlar
    'percentage_aware_regression_loss': percentage_aware_regression_loss,
    'balanced_threshold_killer_loss': balanced_threshold_killer_loss,
    'balanced_focal_loss': balanced_focal_loss(),
    'create_weighted_binary_crossentropy': create_weighted_binary_crossentropy,
    
    # Eski fonksiyonlar (geriye uyumluluk)
    'threshold_killer_loss': threshold_killer_loss,
    'ultra_focal_loss': ultra_focal_loss()
}
