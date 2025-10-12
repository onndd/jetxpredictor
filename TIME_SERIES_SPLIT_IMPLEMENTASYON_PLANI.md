# 🔧 TIME-SERIES SPLIT İMPLEMENTASYON PLANI

**Tarih**: 2025-10-12  
**Amaç**: Modelin zaman serisi üzerindeki sıralı paternleri ezberlemesini sağlamak

---

## 📋 TEMEL PRENSİPLER

### 🎯 Hedef
Modelin **kronolojik veri akışına** ne kadar iyi adapte olduğunu test etmek ve **ezberleme (overfitting) eğilimlerini** netleştirmek.

### ⚠️ KRİTİK KISITLAR

1. **RASTGELELIK YASAK**: Veri setinin hiçbir aşamasında rastgele karıştırma (shuffle) veya rastgele örnekleme (random sampling) yapılmayacak.

2. **KRONOLOJİK BÖLME**: 
   - Test Seti: Son 1,000 kayıt (kesinlikle)
   - Eğitim Seti: İlk 5,000 kayıt (geri kalan)

3. **MODEL.FIT() KISITLARI**:
   - `shuffle=False` (validation split'te de)
   - Batch akışı kronolojik sırayla (en eskiden en yeniye)

4. **VALİDASYON SETİ**: Eğitim setinin son %20'si (kronolojik olarak)

---

## 🔄 MEVCUT DURUM vs HEDEF DURUM

### ❌ Mevcut Durum (Yanlış)
```python
# Progressive Training
tr_idx, te_idx = train_test_split(idx, test_size=0.2, 
                                   shuffle=True,      # ❌ YANLIŞ!
                                   stratify=y_cls,    # ❌ Zaman serisini bozuyor!
                                   random_state=42)

model.fit(..., validation_split=0.2, ...)  # Varsayılan shuffle=True
```

**Sorunlar**:
- Veriler karıştırılıyor (shuffle=True)
- Gelecekten geçmişe "sızıntı" oluşuyor (data leakage)
- Model ezberliyor, gerçek dünya performansı kötü
- Stratified sampling zaman serisi yapısını bozuyor

### ✅ Hedef Durum (Doğru)
```python
# Kronolojik split (shuffle yok!)
test_size = 1000
train_end = len(X) - test_size

X_train = X[:train_end]
X_test = X[train_end:]
y_train = y[:train_end]
y_test = y[train_end:]

# Validation split de kronolojik
val_size = int(len(X_train) * 0.2)
val_start = len(X_train) - val_size

X_tr = X_train[:val_start]
X_val = X_train[val_start:]
y_tr = y_train[:val_start]
y_val = y_train[val_start:]

model.fit(X_tr, y_tr, 
          validation_data=(X_val, y_val),  # Manuel validation
          shuffle=False,                     # ✅ Kronolojik!
          ...)
```

**Avantajlar**:
- Zaman serisi yapısı korunuyor
- Gerçek dünya senaryosunu simüle ediyor
- Model ezberleme eğilimi net görünüyor
- Data leakage yok

---

## 📊 VERİ BÖLME STRATEJİSİ

### Veri Seti Yapısı
```
Toplam: 5,090 örnek (window_size=1000 sonrası)

├─ EĞİTİM SETİ: 4,090 örnek (ilk %80.3)
│  ├─ Train: 3,272 örnek (eğitim setinin %80'i)
│  └─ Validation: 818 örnek (eğitim setinin %20'si, kronolojik)
│
└─ TEST SETİ: 1,000 örnek (son %19.7, kesinlikle test!)
```

### Zaman Serisi Görselleştirme
```
[Örnek 0] ─────► [Örnek 3,271] | [Örnek 3,272] ─────► [Örnek 4,089] | [Örnek 4,090] ─────► [Örnek 5,089]
     TRAIN (3,272)                    VALIDATION (818)                      TEST (1,000)
     ◄───────────────────────────────────────────────────────────────────────────────────────────────►
                                    Zaman Akışı (Kronolojik)
```

### Kod İmplementasyonu
```python
# 1. Feature extraction (kronolojik sırada)
window_size = 1000
X, y_reg, y_cls = [], [], []

for i in range(window_size, len(all_values)-1):  # Kronolojik sıra
    hist = all_values[:i].tolist()
    target = all_values[i]
    
    # Features
    feats = FeatureEngineering.extract_all_features(hist)
    X.append(list(feats.values()))
    y_reg.append(target)
    y_cls.append(1 if target >= 1.5 else 0)

X = np.array(X)
y_reg = np.array(y_reg)
y_cls = np.array(y_cls)

# 2. Normalizasyon (sadece train set üzerinde fit!)
test_size = 1000
train_end = len(X) - test_size

scaler = StandardScaler()
X_train = scaler.fit_transform(X[:train_end])  # Sadece train'de fit
X_test = scaler.transform(X[train_end:])        # Test'te sadece transform

y_train = y_reg[:train_end]
y_test = y_reg[train_end:]

# 3. Validation split (kronolojik)
val_size = int(len(X_train) * 0.2)
val_start = len(X_train) - val_size

X_tr = X_train[:val_start]
X_val = X_train[val_start:]
y_tr = y_train[:val_start]
y_val = y_train[val_start:]

print(f"✅ Train: {len(X_tr):,}, Validation: {len(X_val):,}, Test: {len(X_test):,}")
print(f"📊 Toplam: {len(X_tr) + len(X_val) + len(X_test):,}")
```

---

## 🧠 NEURAL NETWORK DEĞİŞİKLİKLERİ

### 1. Data Split Değişiklikleri

**Dosya**: `notebooks/jetx_PROGRESSIVE_TRAINING.py`

**Değiştirilecek Satırlar**: 323-332

#### Eski Kod (Yanlış):
```python
# Train/Test split - STRATIFIED SAMPLING EKLENDI
idx = np.arange(len(X_f))
tr_idx, te_idx = train_test_split(idx, test_size=0.2, shuffle=True, 
                                   stratify=y_cls, random_state=42)

X_f_tr, X_50_tr, X_200_tr, X_500_tr, X_1000_tr = X_f[tr_idx], X_50[tr_idx], ...
y_reg_tr, y_cls_tr, y_thr_tr = y_reg[tr_idx], y_cls[tr_idx], y_thr[tr_idx]

X_f_te, X_50_te, X_200_te, X_500_te, X_1000_te = X_f[te_idx], X_50[te_idx], ...
y_reg_te, y_cls_te, y_thr_te = y_reg[te_idx], y_cls[te_idx], y_thr[te_idx]
```

#### Yeni Kod (Doğru):
```python
# =============================================================================
# TIME-SERIES SPLIT (KRONOLOJIK) - SHUFFLE YOK!
# =============================================================================
print("\n📊 TIME-SERIES SPLIT (Kronolojik Bölme)...")
print("⚠️  UYARI: Shuffle devre dışı - Zaman serisi yapısı korunuyor!")

# Test seti: Son 1000 kayıt
test_size = 1000
train_end = len(X_f) - test_size

# Train/Test split (kronolojik)
X_f_train = X_f[:train_end]
X_50_train = X_50[:train_end]
X_200_train = X_200[:train_end]
X_500_train = X_500[:train_end]
X_1000_train = X_1000[:train_end]
y_reg_train = y_reg[:train_end]
y_cls_train = y_cls[:train_end]
y_thr_train = y_thr[:train_end]

X_f_te = X_f[train_end:]
X_50_te = X_50[train_end:]
X_200_te = X_200[train_end:]
X_500_te = X_500[train_end:]
X_1000_te = X_1000[train_end:]
y_reg_te = y_reg[train_end:]
y_cls_te = y_cls[train_end:]
y_thr_te = y_thr[train_end:]

# Validation split (eğitim setinin son %20'si, kronolojik)
val_size = int(len(X_f_train) * 0.2)
val_start = len(X_f_train) - val_size

X_f_tr = X_f_train[:val_start]
X_50_tr = X_50_train[:val_start]
X_200_tr = X_200_train[:val_start]
X_500_tr = X_500_train[:val_start]
X_1000_tr = X_1000_train[:val_start]
y_reg_tr = y_reg_train[:val_start]
y_cls_tr = y_cls_train[:val_start]
y_thr_tr = y_thr_train[:val_start]

X_f_val = X_f_train[val_start:]
X_50_val = X_50_train[val_start:]
X_200_val = X_200_train[val_start:]
X_500_val = X_500_train[val_start:]
X_1000_val = X_1000_train[val_start:]
y_reg_val = y_reg_train[val_start:]
y_cls_val = y_cls_train[val_start:]
y_thr_val = y_thr_train[val_start:]

print(f"✅ Train: {len(X_f_tr):,}")
print(f"✅ Validation: {len(X_f_val):,} (eğitim setinin son %20'si)")
print(f"✅ Test: {len(X_f_te):,} (tüm verinin son {test_size} kaydı)")
print(f"📊 Toplam: {len(X_f_tr) + len(X_f_val) + len(X_f_te):,}")
```

### 2. Model.fit() Değişiklikleri

**Her 3 aşamada da** (AŞAMA 1, 2, 3):

#### Eski Kod:
```python
hist1 = model.fit(
    [X_f_tr, X_50_tr, X_200_tr, X_500_tr, X_1000_tr],
    {'regression': y_reg_tr, 'classification': y_cls_tr, 'threshold': y_thr_tr},
    epochs=100,
    batch_size=64,
    validation_split=0.2,  # ❌ Otomatik split (shuffle yapıyor!)
    callbacks=cb1,
    verbose=1
)
```

#### Yeni Kod:
```python
hist1 = model.fit(
    [X_f_tr, X_50_tr, X_200_tr, X_500_tr, X_1000_tr],
    {'regression': y_reg_tr, 'classification': y_cls_tr, 'threshold': y_thr_tr},
    epochs=100,
    batch_size=64,
    validation_data=(  # ✅ Manuel validation (kronolojik!)
        [X_f_val, X_50_val, X_200_val, X_500_val, X_1000_val],
        {'regression': y_reg_val, 'classification': y_cls_val, 'threshold': y_thr_val}
    ),
    shuffle=False,  # ✅ KRITIK: Shuffle devre dışı!
    callbacks=cb1,
    verbose=1
)
```

### 3. Class Weight Artırımı

**Tüm aşamalarda class weight'i artır**:

```python
# AŞAMA 1
w0_stage1 = 15.0  # 1.2 → 15.0 (12.5x artış!)
w1_stage1 = 1.0

# AŞAMA 2
w0 = 20.0  # 1.5 → 20.0 (13.3x artış!)
w1 = 1.0

# AŞAMA 3
w0_final = 25.0  # 2.0 → 25.0 (12.5x artış!)
w1_final = 1.0
```

### 4. Adaptive Weight Scheduler Güncelleme

`max_weight` parametresini artır:

```python
adaptive_scheduler_2 = AdaptiveWeightScheduler(
    initial_weight=20.0,   # 1.5 → 20.0
    min_weight=10.0,       # 1.0 → 10.0
    max_weight=50.0,       # 4.0 → 50.0 (lazy learning için yeterli!)
    target_below_acc=0.70,
    target_above_acc=0.75,
    test_data=([X_f_te, X_50_te, X_200_te, X_500_te, X_1000_te], y_reg_te),
    threshold=1.5,
    check_interval=5
)
```

---

## 🤖 CATBOOST DEĞİŞİKLİKLERİ

### 1. Data Split Değişiklikleri

**Dosya**: `notebooks/jetx_CATBOOST_TRAINING.py`

**Değiştirilecek Satırlar**: 134-138

#### Eski Kod:
```python
# Train/Test split - STRATIFIED SAMPLING
X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
    X, y_reg, y_cls, test_size=0.2, shuffle=True, stratify=y_cls, random_state=42
)
```

#### Yeni Kod:
```python
# =============================================================================
# TIME-SERIES SPLIT (KRONOLOJIK) - SHUFFLE YOK!
# =============================================================================
print("\n📊 TIME-SERIES SPLIT (Kronolojik Bölme)...")
print("⚠️  UYARI: Shuffle devre dışı - Zaman serisi yapısı korunuyor!")

# Test seti: Son 1000 kayıt
test_size = 1000
train_end = len(X) - test_size

# Train/Test split (kronolojik)
X_train = X[:train_end]
X_test = X[train_end:]
y_reg_train = y_reg[:train_end]
y_reg_test = y_reg[train_end:]
y_cls_train = y_cls[:train_end]
y_cls_test = y_cls[train_end:]

print(f"✅ Train: {len(X_train):,}")
print(f"✅ Test: {len(X_test):,} (tüm verinin son {test_size} kaydı)")
print(f"📊 Toplam: {len(X_train) + len(X_test):,}")

# Validation için train setini böl (kronolojik)
val_size = int(len(X_train) * 0.2)
val_start = len(X_train) - val_size

X_tr = X_train[:val_start]
X_val = X_train[val_start:]
y_reg_tr = y_reg_train[:val_start]
y_reg_val = y_reg_train[val_start:]
y_cls_tr = y_cls_train[:val_start]
y_cls_val = y_cls_train[val_start:]

print(f"   ├─ Actual Train: {len(X_tr):,}")
print(f"   └─ Validation: {len(X_val):,} (train'in son %20'si)")
```

### 2. GPU Callback Hatası Düzeltmesi

**3 Seçenek**:

#### Seçenek 1: CPU Kullan (En Kolay)
```python
regressor = CatBoostRegressor(
    iterations=1500,
    depth=10,
    learning_rate=0.03,
    l2_leaf_reg=5,
    bootstrap_type='Bernoulli',
    subsample=0.8,
    loss_function='MAE',
    eval_metric='MAE',
    task_type='CPU',  # ✅ GPU → CPU
    verbose=100,
    random_state=42,
    early_stopping_rounds=100
)

# Callback kullanılabilir
regressor.fit(
    X_tr, y_reg_tr,
    eval_set=(X_val, y_reg_val),  # ✅ Manuel eval set (kronolojik!)
    verbose=100,
    callbacks=[virtual_bankroll_reg]  # ✅ CPU'da çalışır
)
```

#### Seçenek 2: Callback'i Kaldır (GPU Hızını Korur)
```python
regressor = CatBoostRegressor(
    iterations=1500,
    task_type='GPU',  # GPU korunur
    ...
)

# Callback YOK
regressor.fit(
    X_tr, y_reg_tr,
    eval_set=(X_val, y_reg_val),  # ✅ Manuel eval set (kronolojik!)
    verbose=100
    # callbacks parametresi YOK
)
```

#### Seçenek 3: Koşullu Callback (Önerilen!)
```python
# GPU kontrolü
try:
    import GPUtil
    has_gpu = len(GPUtil.getGPUs()) > 0
except:
    has_gpu = False

task_type = 'GPU' if has_gpu else 'CPU'

regressor = CatBoostRegressor(
    iterations=1500,
    task_type=task_type,
    ...
)

# Callback sadece CPU'da
callbacks_list = [virtual_bankroll_reg] if task_type == 'CPU' else []

regressor.fit(
    X_tr, y_reg_tr,
    eval_set=(X_val, y_reg_val),
    verbose=100,
    callbacks=callbacks_list  # ✅ Koşullu
)
```

### 3. Class Weight Artırımı

```python
# CatBoost için class_weights
class_weights = {0: 20.0, 1: 1.0}  # 2.0 → 20.0 (10x artış!)

classifier = CatBoostClassifier(
    iterations=1500,
    depth=9,
    learning_rate=0.03,
    l2_leaf_reg=5,
    bootstrap_type='Bernoulli',
    subsample=0.8,
    loss_function='Logloss',
    eval_metric='Accuracy',
    task_type='CPU',  # veya koşullu
    class_weights=class_weights,  # ✅ 20x weight
    verbose=100,
    random_state=42,
    early_stopping_rounds=100
)
```

---

## 📋 İMPLEMENTASYON ADIMLARI

### Adım 1: Neural Network Scripti Güncelle
```bash
# Dosya: notebooks/jetx_PROGRESSIVE_TRAINING.py
```

**Değişiklikler**:
1. ✅ Data split: Kronolojik bölme (satır 323-332)
2. ✅ Model.fit(): shuffle=False, manuel validation (satır 907-916, 1018-1027, 1125-1134)
3. ✅ Class weights: 15x, 20x, 25x (satır 866-868, 965-967, 1072-1074)
4. ✅ Adaptive scheduler: max_weight=50.0 (satır 987-997, 1092-1101)

### Adım 2: CatBoost Scripti Güncelle
```bash
# Dosya: notebooks/jetx_CATBOOST_TRAINING.py
```

**Değişiklikler**:
1. ✅ Data split: Kronolojik bölme (satır 134-138)
2. ✅ GPU callback hatası: Koşullu callback veya CPU (satır 189-194, 280-285)
3. ✅ Class weights: 20x (satır 232)
4. ✅ Manual eval set: Kronolojik validation (satır 189-194, 280-285)

### Adım 3: Test ve Karşılaştırma
```bash
# 1. Neural Network eğit
python notebooks/jetx_PROGRESSIVE_TRAINING.py

# 2. CatBoost eğit
python notebooks/jetx_CATBOOST_TRAINING.py

# 3. Sonuçları karşılaştır
```

---

## 📊 BEKLENEN SONUÇLAR

### Başarı Kriterleri

#### Neural Network:
- **1.5 Altı Doğruluk**: %70+  (şu an: %0.0 ❌)
- **1.5 Üstü Doğruluk**: %70-80 (şu an: %99.85 ama yanıltıcı)
- **Para Kaybı Riski**: <%30  (şu an: %100 ❌)
- **Sanal Kasa ROI**: >0%  (şu an: -2.21% ❌)
- **Kazanma Oranı**: >66.7%  (şu an: 65.2% ⚠️)

#### CatBoost:
- **1.5 Altı Doğruluk**: %65+
- **1.5 Üstü Doğruluk**: %70+
- **Para Kaybı Riski**: <%35
- **MAE**: <10.0

### ⚠️ Önemli Notlar

1. **Daha Kötü Metrikler Beklenebilir**:
   - Shuffle=False olduğu için metrikler düşebilir
   - Bu NORMALDIR - gerçek dünya performansını gösterir!

2. **Lazy Learning Azalmalı**:
   - Model artık her şeyi "1.5 üstü" diye tahmin etmemeli
   - 1.5 altı doğruluk %0'dan yukarı çıkmalı

3. **Ezberleme Testi**:
   - Time-series split modelin ezberleme eğilimini netleştirir
   - Train vs Test performans farkı büyükse → overfitting var

---

## 🔍 TEST SENARYOLARI

### Senaryo 1: Baseline Test (Şu Anki Kod)
```python
# Shuffle=True (mevcut)
ROI: -2.21%
1.5 Altı: %0.0
Para Kaybı: %100
```

### Senaryo 2: Time-Series Split (Hedef)
```python
# Shuffle=False + Kronolojik split
Beklenen ROI: -5% ile +5% arası (daha gerçekçi)
Beklenen 1.5 Altı: %40-70 (lazy learning azalır)
Beklenen Para Kaybı: %25-40 (düzelir ama hala yüksek)
```

### Senaryo 3: Time-Series + 20x Class Weight
```python
# Shuffle=False + 20x weight
Beklenen ROI: -2% ile +8% arası
Beklenen 1.5 Altı: %60-80 (hedef!)
Beklenen Para Kaybı: %20-30 (hedef: <%20)
```

---

## 📝 DOKÜMANTASYON

Eğitim sonrası oluşturulacak rapor:

```markdown
# TIME-SERIES SPLIT EĞİTİM RAPORU

## Yapılandırma
- Shuffle: False
- Train Size: 3,272
- Validation Size: 818 (kronolojik)
- Test Size: 1,000 (kronolojik)
- Class Weight: 15x → 20x → 25x

## Sonuçlar
| Metrik | Shuffle=True (Eski) | Shuffle=False (Yeni) | Fark |
|--------|---------------------|----------------------|------|
| 1.5 Altı Doğruluk | 0.0% | XX.X% | +XX.X% |
| ROI | -2.21% | +X.XX% | +X.XX% |
| Para Kaybı Riski | 100% | XX% | -XX% |

## durumu: [Azaldı/Devam ediyor]
- Ezberleme: [Var/Yok]
- Öneriler: [...]
```

---

## ⚡ HIZLI BAŞLANGIÇ

Değişiklikleri uygulamak için:

```bash
# 1. Code mode'a geç
switch_mode code

# 2. Her iki scripti güncelle
# - notebooks/jetx_PROGRESSIVE_TRAINING.py
# - notebooks/jetx_CATBOOST_TRAINING.py

# 3. Colab'da test et
# - Upload updated scripts
# - Run training
# - Compare results
```

---

**Hazırlayan**: AI Architect  
**Tarih**: 2025-10-12 21:23  
**Durum**: İmplementasyona hazır ✅