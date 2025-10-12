# 🚀 FAZ 2: COLAB ENTEGRASYONU - İmplementasyon Planı

**Tarih:** 2025-10-12  
**Durum:** Hazır - İmplementasyona başlanabilir  
**Bağlantı:** YENI_MODEL_GELISTIRME_PLANI.md

---

## 📊 ÖZET

Faz 1 tamamlandı (Kod altyapısı YENI_MODEL_GELISTIRME_PLANI.md'de hazır).
Şimdi Faz 2'de bu kodları Progressive training ve XGBoost scriptlerine entegre edeceğiz.

---

## 🎯 ANA DEĞİŞİKLİKLER

### 1. Progressive Training Script ([`notebooks/jetx_PROGRESSIVE_TRAINING.py`](notebooks/jetx_PROGRESSIVE_TRAINING.py))

**Eklenecekler:**
- ✅ Transformer sınıfları (PositionalEncoding + LightweightTransformerEncoder)
- ✅ build_progressive_model fonksiyonuna Transformer branch
- ✅ Çift sanal kasa simülasyonu (Kasa 1: 1.5x + Kasa 2: %80 çıkış)
- ✅ ZIP paketleme ve indirme sistemi

**Satır Konumları:**
- Transformer sınıfları: ~73 satır sonrası (import'lardan sonra)
- build_progressive_model güncellemesi: ~220-310 satır arası
- Çift sanal kasa: ~995 satır sonrası (mevcut sanal kasa yerine)
- ZIP paketi: ~1056 satır sonrası (model kaydetme bölümü)

### 2. XGBoost → CatBoost Dönüşümü

**Yeni Dosya:** [`notebooks/jetx_CATBOOST_TRAINING.py`](notebooks/jetx_CATBOOST_TRAINING.py)

**İçerik:**
- CatBoost regressor (değer tahmini)
- CatBoost classifier (1.5 eşik tahmini)
- Çift sanal kasa simülasyonu
- ZIP paketleme sistemi

### 3. Colab Notebook ([`notebooks/JetX_PROGRESSIVE_TRAINING_Colab.ipynb`](notebooks/JetX_PROGRESSIVE_TRAINING_Colab.ipynb))

**Güncellemeler:**
- Kütüphane listesine `catboost` ekleme (satır ~65)
- XGBoost referansını CatBoost'a çevirme (satır ~27-50)

---

## 📋 DETAYLI ADIMLAR

### ADIM 1: Progressive Script - Transformer Sınıfları Ekleme

**Konum:** [`notebooks/jetx_PROGRESSIVE_TRAINING.py`](notebooks/jetx_PROGRESSIVE_TRAINING.py) ~73 satır sonrası

**Eklenecek Kod:** (YENI_MODEL_GELISTIRME_PLANI.md ADIM 1.1'den)

```python
import tensorflow as tf
from tensorflow.keras import layers

class PositionalEncoding(layers.Layer):
    # ... (tam kod YENI_MODEL_GELISTIRME_PLANI.md satır 65-99)

class LightweightTransformerEncoder(layers.Layer):
    # ... (tam kod YENI_MODEL_GELISTIRME_PLANI.md satır 102-228)
```

**Gerekçe:** Progressive NN'ye Transformer branch eklemek için önce sınıflar tanımlanmalı.

---

### ADIM 2: build_progressive_model - Transformer Entegrasyonu

**Konum:** [`notebooks/jetx_PROGRESSIVE_TRAINING.py`](notebooks/jetx_PROGRESSIVE_TRAINING.py) ~220-310 satır arası

**Değişiklik:** (YENI_MODEL_GELISTIRME_PLANI.md ADIM 1.2'den)

```python
def build_progressive_model(n_features):
    # ... (mevcut kod)
    
    # YENİ: Transformer branch (TCN'den sonra, ~280 satır civarı)
    transformer = LightweightTransformerEncoder(
        d_model=256,
        num_layers=4,
        num_heads=8,
        dff=1024,
        dropout=0.2
    )(inp_1000)
    
    # Fusion güncelleme (~287 satır)
    fus = layers.Concatenate()([inp_f, nb_all, tcn, transformer])  # transformer eklendi
    
    # ... (geri kalan kod aynı)
```

**Gerekçe:** Model mimarisine Transformer branch ekleyerek daha derin zaman serisi analizi.

---

### ADIM 3: Progressive Script - Çift Sanal Kasa Simülasyonu

**Konum:** [`notebooks/jetx_PROGRESSIVE_TRAINING.py`](notebooks/jetx_PROGRESSIVE_TRAINING.py) ~995 satır sonrası

**Değişiklik:** Mevcut "Gelişmiş Sanal Kasa" bölümünü kaldırıp YENI_MODEL_GELISTIRME_PLANI.md ADIM 1.4'teki çift kasa sistemini ekle.

**Eklenecek Kod:** (YENI_MODEL_GELISTIRME_PLANI.md satır 363-523)

```python
# =============================================================================
# ÇİFT SANAL KASA SİMÜLASYONU
# =============================================================================
print("\n" + "="*80)
print("💰 ÇİFT SANAL KASA SİMÜLASYONU")
print("="*80)

# Dinamik kasa miktarı
test_count = len(y_reg_te)
initial_bankroll = test_count * 10
bet_amount = 10.0

# KASA 1: 1.5x EŞİK SİSTEMİ
# ... (tam kod)

# KASA 2: %80 ÇIKIŞ SİSTEMİ
# ... (tam kod)

# KARŞILAŞTIRMA
# ... (tam kod)
```

**Gerekçe:** İki farklı strateji ile kar/zarar analizi.

---

### ADIM 4: Progressive Script - ZIP Paketleme Sistemi

**Konum:** [`notebooks/jetx_PROGRESSIVE_TRAINING.py`](notebooks/jetx_PROGRESSIVE_TRAINING.py) ~1056 satır sonrası

**Değişiklik:** Mevcut model kaydetme bölümünü genişletip ZIP paketi ekleme.

**Eklenecek Kod:** (YENI_MODEL_GELISTIRME_PLANI.md satır 593-697)

```python
# =============================================================================
# MODEL KAYDETME + ZIP PAKETI
# =============================================================================
import shutil

# models/ klasörünü oluştur
os.makedirs('models', exist_ok=True)

# 1. Progressive NN modeli
model.save('models/jetx_progressive_transformer.h5')

# 2. Scaler
joblib.dump(scaler, 'models/scaler_progressive_transformer.pkl')

# 3. Model bilgileri (JSON)
# ... (tam kod)

# ZIP dosyası oluştur
zip_filename = 'jetx_models_progressive_v2.0.zip'
shutil.make_archive(
    '/content/jetx_models_progressive_v2.0', 
    'zip', 
    'models'
)

# Google Colab'da indirme
from google.colab import files
files.download(f'/content/{zip_filename}')
```

**Gerekçe:** Tüm dosyaları tek ZIP'te indirme kolaylığı.

---

### ADIM 5: Yeni CatBoost Training Script Oluşturma

**Dosya:** [`notebooks/jetx_CATBOOST_TRAINING.py`](notebooks/jetx_CATBOOST_TRAINING.py) (YENİ)

**İçerik:** [`notebooks/jetx_XGBOOST_TRAINING.py`](notebooks/jetx_XGBOOST_TRAINING.py) dosyasını temel alıp:

1. **XGBoost → CatBoost dönüşümü:**
   - `import xgboost as xgb` → `from catboost import CatBoostRegressor, CatBoostClassifier`
   - `xgb.XGBRegressor()` → `CatBoostRegressor()`
   - `xgb.XGBClassifier()` → `CatBoostClassifier()`

2. **CatBoost parametreleri:** (YENI_MODEL_GELISTIRME_PLANI.md satır 290-343)
   ```python
   regressor = CatBoostRegressor(
       iterations=500,
       depth=8,
       learning_rate=0.05,
       loss_function='MAE',
       task_type='GPU',
       verbose=50
   )
   
   classifier = CatBoostClassifier(
       iterations=500,
       depth=7,
       learning_rate=0.05,
       loss_function='Logloss',
       class_weights={0: 2.0, 1: 1.0},
       task_type='GPU',
       verbose=50
   )
   ```

3. **Model kaydetme:**
   ```python
   regressor.save_model('models/catboost_regressor.cbm')
   classifier.save_model('models/catboost_classifier.cbm')
   ```

**Gerekçe:** CatBoost XGBoost'a göre daha iyi category handling ve class imbalance yönetimi.

---

### ADIM 6: CatBoost Script - Çift Sanal Kasa

**Konum:** [`notebooks/jetx_CATBOOST_TRAINING.py`](notebooks/jetx_CATBOOST_TRAINING.py) içinde

**Ekleme:** ADIM 3'teki çift sanal kasa sisteminin aynısını ekle.

**Gerekçe:** Progressive ve CatBoost modellerinin karşılaştırılabilir kasa simülasyonları.

---

### ADIM 7: Colab Notebook - Kütüphane Güncellemesi

**Konum:** [`notebooks/JetX_PROGRESSIVE_TRAINING_Colab.ipynb`](notebooks/JetX_PROGRESSIVE_TRAINING_Colab.ipynb) ~65 satır

**Değişiklik:**
```python
# ESKI:
!pip install -q tensorflow scikit-learn xgboost pandas numpy scipy joblib matplotlib seaborn tqdm

# YENİ:
!pip install -q tensorflow scikit-learn pandas numpy scipy joblib matplotlib seaborn tqdm catboost
```

**Gerekçe:** CatBoost kütüphanesini Colab ortamına ekleme.

---

### ADIM 8: Colab Notebook - Referans Güncellemesi

**Konum:** [`notebooks/JetX_PROGRESSIVE_TRAINING_Colab.ipynb`](notebooks/JetX_PROGRESSIVE_TRAINING_Colab.ipynb) ~27-50 satır

**Değişiklik:**
```markdown
# ESKI:
### 2️⃣ XGBoost (Gradient Boosting) ~30-60 dakika

# YENİ:
### 2️⃣ CatBoost (Gradient Boosting) ~30-60 dakika
```

Ve script referansını güncelle:
```python
# ESKI:
!python notebooks/jetx_XGBOOST_TRAINING.py

# YENİ:
!python notebooks/jetx_CATBOOST_TRAINING.py
```

**Gerekçe:** XGBoost artık CatBoost ile değiştirildi.

---

## 🔄 İMPLEMENTASYON SIRASI

1. ✅ Progressive script - Transformer sınıfları ekleme
2. ✅ Progressive script - build_progressive_model Transformer entegrasyonu
3. ✅ Progressive script - Çift sanal kasa simülasyonu
4. ✅ Progressive script - ZIP paketleme sistemi
5. ✅ Yeni CatBoost training script oluşturma
6. ✅ CatBoost script - Çift sanal kasa simülasyonu
7. ✅ Colab notebook - Kütüphane güncellemesi
8. ✅ Colab notebook - Referans güncellemesi
9. ✅ Final test ve doğrulama

---

## 📊 BEKLENEN SONUÇLAR

### Progressive NN (Transformer ile):
- **1.5 Altı Doğruluk:** %70-80 (Hedef: %75)
- **1.5 Üstü Doğruluk:** %75-85 (Hedef: %75)
- **Para Kaybı Riski:** <%20 (Hedef: <%20)
- **MAE:** < 2.0

### CatBoost:
- **1.5 Altı Doğruluk:** %70-80 (Hedef: %75)
- **MAE:** < 2.0

### Çift Sanal Kasa:
- **Kasa 1 (1.5x):** ROI +%5 - +%15
- **Kasa 2 (%80 çıkış):** ROI +%10 - +%25 (potansiyel daha yüksek)

---

## ⚠️ NOTLAR

1. **Backup:** Değişikliklerden önce mevcut dosyaların yedeğini al
2. **Test:** Her adımdan sonra syntax hatalarını kontrol et
3. **Colab:** GPU runtime kullan (Runtime → Change runtime type → GPU)
4. **Süre:** Toplam ~2-3 saat eğitim süresi (Progressive NN + CatBoost)

---

## ✅ KONTROL LİSTESİ

- [ ] Progressive script - Transformer sınıfları eklendi
- [ ] Progressive script - build_progressive_model güncellendi
- [ ] Progressive script - Çift sanal kasa eklendi
- [ ] Progressive script - ZIP paketleme eklendi
- [ ] CatBoost training script oluşturuldu
- [ ] CatBoost script - Çift sanal kasa eklendi
- [ ] Colab notebook - Kütüphane güncellendi
- [ ] Colab notebook - Referanslar güncellendi
- [ ] Tüm değişiklikler test edildi

---

**Sonraki Adım:** Code moduna geçerek implementasyona başla
