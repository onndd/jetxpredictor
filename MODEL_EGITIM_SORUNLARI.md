# 🚨 Model Eğitim İnstabilite Sorunu - Analiz ve Çözümler

**Tarih:** 2025-10-09  
**Sorun:** Model eğitimi sırasında threshold prediction'lar bir uçtan diğerine savrulıyor

---

## 📊 Gözlemlenen Sorun

### Epoch İlerleme Analizi

| Epoch | 1.5 Altı Doğruluğu | 1.5 Üstü Doğruluğu | Para Kaybı Riski | Durum |
|-------|--------------------|--------------------|------------------|-------|
| 1 | 0.0% | 100.0% | 100.0% | ❌ Çok kötü |
| **6** | **54.3%** | **42.9%** | **45.7%** | **✅ İYİ!** |
| 11 | 0.0% | 99.9% | 100.0% | ❌ Çok kötü |
| 16 | 13.2% | 90.9% | 86.8% | ❌ Kötü |
| 21 | 100.0% | 0.0% | 0.0% | ❌ Ters uçta kötü |
| 26 | 0.0% | 100.0% | 100.0% | ❌ Çok kötü |
| 31 | 0.0% | 99.7% | 100.0% | ❌ Çok kötü |
| 36 | 100.0% | 0.0% | 0.0% | ❌ Ters uçta kötü |
| 41 | 100.0% | 0.0% | 0.0% | ❌ Ters uçta kötü |
| 46 | 100.0% | 0.0% | 0.0% | ❌ Ters uçta kötü |

### 🔴 Kritik Bulgular

1. **Epoch 6'da Peak Performans** - Model dengelenmiş tahminler yapıyordu
2. **Epoch 11'den sonra bozulma** - Model bir sınıfa kilitleniyor
3. **Epoch 21'den sonra ters tarafa kayma** - Tamamen 1.5 altı tahmin ediyor
4. **Sürekli savrulma** - Sağlıklı bir öğrenme curve'u yok

---

## 🔍 Kök Nedenleri

### 1. ❌ Yanlış Training Strategy (AŞAMA 1)

```python
# MEVCUT KOD (notebooks/jetx_PROGRESSIVE_TRAINING.py)
loss_weights={
    'regression': 1.0, 
    'classification': 0.0,  # ❌ KAPALI!
    'threshold': 0.0        # ❌ KAPALI!
}
```

**Sorun:** AŞAMA 1'de sadece regression eğitiliyor, threshold output hiç kullanılmıyor!

**Sonuç:** Model threshold prediction yapmayı öğrenemiyor, sonraki aşamalarda instabil oluyor.

---

### 2. ❌ Çok Agresif Class Weights

```python
# AŞAMA 2 & 3
TARGET_MULTIPLIER = 25.0  # AŞAMA 2
TARGET_MULTIPLIER = 30.0  # AŞAMA 3

w0 = (len(y_thr_tr) / (2 * c0)) * TARGET_MULTIPLIER
```

**Sorun:** 25x-30x class weight çok fazla! Model azınlık sınıfına aşırı odaklanıyor.

**Beklenen:** 3-7x arası dengeli bir ağırlık olmalı.

---

### 3. ❌ Yanlış Monitoring Metric

```python
# AŞAMA 1
callbacks.ModelCheckpoint(
    'stage1_best.h5',
    monitor='val_regression_mae',  # ❌ YANLIŞ!
    save_best_only=True,
    mode='min'
)
```

**Sorun:** Regression MAE monitör ediliyor, ama asıl hedef threshold accuracy!

**Olması gereken:** `val_threshold_accuracy` monitör edilmeli.

---

### 4. ❌ Çok Düşük Batch Size

```python
# AŞAMA 1: batch_size=16
# AŞAMA 2: batch_size=8
# AŞAMA 3: batch_size=4
```

**Sorun:** Batch size 4-8 çok düşük! Gradient tahminleri gürültülü oluyor.

**Beklenen:** En az 32-64 batch size.

---

### 5. ❌ Aşırı Patience

```python
callbacks.EarlyStopping(
    patience=50   # AŞAMA 1
    patience=40   # AŞAMA 2
    patience=50   # AŞAMA 3
)
```

**Sorun:** Model epoch 6'da peak yapmış ama 50 epoch daha devam ediyor ve bozuluyor!

**Beklenen:** Patience 10-20 arası olmalı.

---

### 6. ❌ Learning Rate Schedule Çok Hızlı

```python
def lr_schedule(epoch, lr):
    if epoch < 50:    # Öne çekildi: 200 → 50
        return initial_lr
    elif epoch < 150: # Öne çekildi: 500 → 150
        return initial_lr * 0.5
```

**Sorun:** LR çok erken düşüyor, model stabilize olamıyor.

---

## ✅ ÖNERİLEN ÇÖZÜMLER

### Çözüm 1: AŞAMA 1'i Düzelt - Threshold'u Dahil Et

```python
# YENİ AŞAMA 1: Regression + Threshold Birlikte
model.compile(
    optimizer=Adam(0.0001),  # Daha düşük LR
    loss={
        'regression': threshold_killer_loss,
        'classification': 'categorical_crossentropy',
        'threshold': 'binary_crossentropy'  # ✅ AKTİF!
    },
    loss_weights={
        'regression': 0.60,      # ✅ Ana odak regression
        'classification': 0.10,  # ✅ Hafif classification
        'threshold': 0.30        # ✅ Threshold öğrenmeye başla!
    },
    metrics={
        'regression': ['mae'], 
        'classification': ['accuracy'], 
        'threshold': ['accuracy', 'binary_crossentropy']
    }
)
```

**Neden:** Model baştan threshold prediction öğrenmeye başlamalı!

---

### Çözüm 2: Class Weight'leri Yumuşat

```python
# DENGELI CLASS WEIGHTS
# AŞAMA 1
TARGET_MULTIPLIER = 3.0  # ✅ Yumuşak başlangıç

# AŞAMA 2
TARGET_MULTIPLIER = 5.0  # ✅ Orta seviye

# AŞAMA 3
TARGET_MULTIPLIER = 7.0  # ✅ Final odaklanma
```

**Neden:** Aşırı ağırlık model instabilitesine neden oluyor.

---

### Çözüm 3: Doğru Metric'i Monitör Et

```python
# AŞAMA 1
callbacks.ModelCheckpoint(
    'stage1_best.h5',
    monitor='val_threshold_accuracy',  # ✅ DOĞRU!
    save_best_only=True,
    mode='max'  # ✅ Max accuracy
)

callbacks.EarlyStopping(
    monitor='val_threshold_accuracy',  # ✅ DOĞRU!
    patience=15,  # ✅ Daha kısa patience
    mode='max',
    restore_best_weights=True
)
```

**Neden:** Asıl hedefimiz threshold accuracy!

---

### Çözüm 4: Batch Size'ı Artır

```python
# AŞAMA 1: batch_size=64   # ✅ Daha stabil
# AŞAMA 2: batch_size=32   # ✅ Orta seviye
# AŞAMA 3: batch_size=16   # ✅ Fine-tuning için
```

**Neden:** Daha büyük batch size = daha stabil gradient.

---

### Çözüm 5: Learning Rate Schedule Değiştir

```python
def lr_schedule(epoch, lr):
    # ✅ Daha yavaş düşüş
    if epoch < 100:
        return initial_lr
    elif epoch < 300:
        return initial_lr * 0.7  # Daha yumuşak
    elif epoch < 500:
        return initial_lr * 0.5
    else:
        return initial_lr * 0.3
```

**Neden:** Model stabilize olmak için daha fazla zamana ihtiyaç duyuyor.

---

### Çözüm 6: Focal Loss Parametrelerini Yumuşat

```python
# MEVCUT: gamma=4.0, alpha=0.85
# ÖNERİLEN: gamma=2.0, alpha=0.75

def ultra_focal_loss(gamma=2.0, alpha=0.75):  # ✅ Daha yumuşak
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = alpha * K.pow(1 - pt, gamma)
        return -K.mean(focal_weight * K.log(pt))
    return loss
```

**Neden:** Gamma=4 çok agresif, model instabil oluyor.

---

### Çözüm 7: Epoch 6'dan Devam Et (Hızlı Çözüm)

```python
# Eğer epoch 6'da iyi performans varsa:
# 1. stage1_best.h5 dosyasını epoch 6'dan kaydet
# 2. AŞAMA 2'ye geç
# 3. Daha düşük class weight ile devam et (5x)
```

---

## 🎯 YENİ PROGRESSIVE TRAINING STRATEJİSİ

### AŞAMA 1: Balanced Foundation (150 epoch)
- **Hedef:** Regression + Threshold birlikte öğren
- **Loss Weights:** `regression: 0.60, classification: 0.10, threshold: 0.30`
- **Batch Size:** 64
- **LR:** 0.0001
- **Class Weight:** 3x (yumuşak)
- **Patience:** 15
- **Monitor:** `val_threshold_accuracy`

### AŞAMA 2: Threshold Focus (100 epoch)
- **Hedef:** Threshold accuracy'yi artır
- **Loss Weights:** `regression: 0.40, classification: 0.10, threshold: 0.50`
- **Batch Size:** 32
- **LR:** 0.00005
- **Class Weight:** 5x (orta)
- **Patience:** 12
- **Monitor:** `val_threshold_accuracy`

### AŞAMA 3: Final Polish (100 epoch)
- **Hedef:** Tüm output'ları optimize et
- **Loss Weights:** `regression: 0.30, classification: 0.15, threshold: 0.55`
- **Batch Size:** 16
- **LR:** 0.00003
- **Class Weight:** 7x (final)
- **Patience:** 10
- **Monitor:** `val_threshold_accuracy`

---

## 📝 UYGULAMA PLANI

### 1. Yeni Training Script Oluştur

**Dosya:** `notebooks/jetx_STABLE_PROGRESSIVE_TRAINING.py`

**Değişiklikler:**
- [ ] AŞAMA 1'de threshold loss'u aktif et
- [ ] Class weight'leri 3x-5x-7x yap
- [ ] Batch size'ları 64-32-16 yap
- [ ] Patience'leri 15-12-10 yap
- [ ] Monitor metric'i `val_threshold_accuracy` yap
- [ ] LR schedule'u yumuşat
- [ ] Focal loss gamma'yı 2.0'a düşür

### 2. Test Et

```bash
# Google Colab'da çalıştır
python notebooks/jetx_STABLE_PROGRESSIVE_TRAINING.py
```

### 3. Beklenen Sonuçlar

**AŞAMA 1 Sonrası (Epoch ~50-100):**
- 1.5 altı doğruluğu: %50-60
- 1.5 üstü doğruluğu: %60-70
- **STABİL** - Bir uçtan diğerine savr kullanımı yok

**AŞAMA 2 Sonrası:**
- 1.5 altı doğruluğu: %60-70
- 1.5 üstü doğruluğu: %70-80

**AŞAMA 3 Sonrası:**
- 1.5 altı doğruluğu: %70-80+ (HEDEF!)
- 1.5 üstü doğruluğu: %75-85+

---

## 🔧 HIZLI FİX (Şu Anki Eğitimi Kurtar)

Eğer eğitim hala devam ediyorsa:

1. **HEMEN DURDUR** - Epoch 6'dan sonra modeli kaybet
2. **Epoch 6 Model'ini Yükle:**
   ```python
   model.load_weights('stage1_best.h5')
   ```
3. **AŞAMA 2'ye yumuşak geç:**
   ```python
   # Class weight 5x (30x değil!)
   TARGET_MULTIPLIER = 5.0
   
   # Batch size 32 (8 değil!)
   batch_size = 32
   
   # Patience 15 (40 değil!)
   patience = 15
   ```

---

## 📊 İZLEME METRİKLERİ

Her epoch'ta izle:
1. **1.5 Altı Doğruluğu** - %50+ olmalı, sürekli
2. **1.5 Üstü Doğruluğu** - %60+ olmalı, sürekli
3. **Para Kaybı Riski** - %50 altında olmalı, sürekli
4. **Stabilite** - Bir uçtan diğerine savrulmama

**Uyarı İşaretleri:**
- Bir sınıf doğruluğu %100'e çıkıyorsa → DURDUR!
- Bir sınıf doğruluğu %0'a düşüyorsa → DURDUR!
- Para kaybı riski %80+ → DURDUR!

---

## 🎓 DERS ÇIKARILANLAR

1. **Progressive training** = Aşamalı ağırlık artışı, hepsi baştan kapalı DEĞİL!
2. **Class weight** çok yüksek olursa model instabil olur
3. **Batch size** çok küçük olursa gradient gürültülü olur
4. **Patience** çok yüksek olursa overfitting olur
5. **Doğru metric monitör etmek kritik!**

---

## 📁 REFERANSLAR

- Mevcut kod: [`notebooks/jetx_PROGRESSIVE_TRAINING.py`](notebooks/jetx_PROGRESSIVE_TRAINING.py)
- Loss fonksiyonları: [`utils/custom_losses.py`](utils/custom_losses.py)
- Kategori tanımları: [`category_definitions.py`](category_definitions.py)

---

**Sonuç:** Model eğitimi şu anda sağlıksız. Yukarıdaki düzeltmeler uygulanmalı!