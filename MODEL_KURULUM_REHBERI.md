# 🚀 JetX Model Kurulum Rehberi

## 📦 Google Colab'dan Model İndirme ve Lokal Kurulum

Bu rehber, Google Colab'da eğittiğiniz JetX modellerini lokal bilgisayarınıza indirip Streamlit uygulamasında kullanmanız için adım adım talimatlar içerir.

---

## 📋 İÇİNDEKİLER

1. [Model Eğitimi (Google Colab)](#1-model-eğitimi-google-colab)
2. [Model İndirme](#2-model-indirme)
3. [Lokal Kurulum](#3-lokal-kurulum)
4. [Streamlit Başlatma](#4-streamlit-başlatma)
5. [Sorun Giderme](#5-sorun-giderme)

---

## 1. Model Eğitimi (Google Colab)

### Adımlar:

1. Google Colab'da `notebooks/JetX_PROGRESSIVE_TRAINING_Colab.ipynb` dosyasını açın
2. Runtime > Change runtime type > **GPU** seçin
3. Tüm hücreleri sırayla çalıştırın
4. Eğitim tamamlandığında ZIP dosyası otomatik olarak indirilecek

### ⏱️ Beklenen Süre:
- Progressive NN: ~10-12 saat
- CatBoost: ~3-4 saat  
- AutoGluon: ~1-2 saat
- TabNet: ~2-3 saat
- **TOPLAM: ~16-21 saat**

---

## 2. Model İndirme

### Seçenek A: Otomatik İndirme (Önerilen)

Eğitim tamamlandığında, Colab otomatik olarak bir ZIP dosyası indirecek:
```
jetx_5models_v7_YYYYMMDD_HHMM.zip
```

### Seçenek B: Google Drive'dan İndirme

Eğer otomatik indirme çalışmazsa:

1. Google Drive'ınıza gidin
2. `My Drive/JetX_Models_v7/` klasörüne gidin
3. ZIP dosyasını bulun ve indirin

---

## 3. Lokal Kurulum

### 🔧 Adım Adım Kurulum:

#### 1️⃣ ZIP Dosyasını Açın

```bash
# macOS/Linux
unzip jetx_5models_v7_*.zip -d ~/Desktop/jetxpredictor/models/

# Windows (PowerShell)
Expand-Archive jetx_5models_v7_*.zip -DestinationPath C:\Users\YourName\Desktop\jetxpredictor\models\
```

Veya ZIP dosyasına çift tıklayarak açın.

#### 2️⃣ Dosya Yapısını Kontrol Edin

Kurulum sonrası dosya yapınız şöyle olmalı:

```
jetxpredictor/
└── models/
    ├── jetx_model.h5                    ✅ Ana Progressive NN modeli
    ├── scaler.pkl                       ✅ Feature scaler
    ├── catboost_regressor.cbm          ✅ CatBoost regressor
    ├── catboost_classifier.cbm         ✅ CatBoost classifier
    ├── catboost_scaler.pkl             ✅ CatBoost scaler
    ├── autogluon_model/                ✅ AutoGluon model klasörü
    ├── tabnet_high_x.pkl               ✅ TabNet model
    ├── ✅ TabNet scaler
    ├── progressive_multiscale/         📁 Progressive model detayları
    ├── catboost_multiscale/            📁 CatBoost model detayları
    └── all_models_results_v7.json      📊 Tüm sonuçlar
```

#### 3️⃣ Dosya İsimlendirmesini Doğrulayın

**ÖNEMLİ:** Streamlit uygulaması şu dosya isimlerini bekliyor:

| Beklenen Dosya | Açıklama |
|----------------|----------|
| `jetx_model.h5` | Ana Neural Network modeli |
| `scaler.pkl` | Feature normalization için scaler |
| `catboost_regressor.cbm` | CatBoost değer tahmini |
| `catboost_classifier.cbm` | CatBoost sınıflandırma |

Eğer dosya isimleri farklıysa, yeniden adlandırın:

```bash
cd ~/Desktop/jetxpredictor/models/

# Progressive NN modeli varsa
mv progressive_multiscale/ensemble_model.h5 jetx_model.h5
mv progressive_multiscale/scaler.pkl scaler.pkl
```

---

## 4. Streamlit Başlatma

### Terminal'de:

```bash
cd ~/Desktop/jetxpredictor
streamlit run app.py
```

### ✅ Başarılı Kurulum Kontrol Listesi:

Streamlit açıldığında şunları görmelisiniz:

- [x] "✅ Model yüklendi ve hazır!" mesajı
- [x] Sol sidebar'da eksik model uyarısı YOK
- [x] "🔮 YENİ TAHMİN YAP" butonu aktif
- [x] Grafik görüntüleniyor

### ❌ Eğer Hata Görüyorsanız:

**Hata: "⚠️ X model dosyası eksik!"**
```bash
# Eksik dosyaları kontrol edin
ls -la ~/Desktop/jetxpredictor/models/
```

**Hata: "Model yüklenmedi!"**
```bash
# Dosya izinlerini kontrol edin
chmod 644 ~/Desktop/jetxpredictor/models/*.h5
chmod 644 ~/Desktop/jetxpredictor/models/*.pkl
chmod 644 ~/Desktop/jetxpredictor/models/*.cbm
```

---

## 5. Sorun Giderme

### 🔴 Problem 1: ZIP dosyası indirilmedi

**Çözüm:**
1. Google Colab'da son hücreyi tekrar çalıştırın
2. Manuel olarak Google Drive'dan indirin
3. Tarayıcı indirme ayarlarını kontrol edin

### 🔴 Problem 2: Modeller yüklenmiyor

**Olası Sebepler:**
- Dosya isimleri yanlış
- Dosyalar yanlış klasörde
- Dosyalar bozuk (eğitim tamamlanmadan kesildi)

**Çözüm:**
```bash
# 1. Dosya isimlerini kontrol et
ls -la ~/Desktop/jetxpredictor/models/

# 2. Dosya boyutlarını kontrol et (çok küçükse bozuktur)
du -h ~/Desktop/jetxpredictor/models/*.h5

# 3. Gerekirse modeli Colab'da yeniden eğitin
```

### 🔴 Problem 3: TensorFlow hatası

**Hata:** `ImportError: cannot import name 'X' from 'tensorflow'`

**Çözüm:**
```bash
# TensorFlow'u yeniden yükle
pip install --upgrade tensorflow

# Alternatif: CPU versiyonu kullan
pip install tensorflow-cpu
```

### 🔴 Problem 4: CatBoost hatası

**Hata:** `CatBoostError: Cannot load model`

**Çözüm:**
```bash
# CatBoost versiyonunu kontrol et
pip show catboost

# Güncelle
pip install --upgrade catboost
```

---

## 📊 Model Bilgilerini Kontrol Etme

```bash
# JSON sonuçlarını oku
cat ~/Desktop/jetxpredictor/models/all_models_results_v7.json | jq
```

Bu dosyada:
- Eğitim süresi
- Model performans metrikleri
- 1.5x altı/üstü doğruluk oranları
- ROI ve kazanç istatistikleri

bulunur.

---

## 🎯 Model Performans Beklentileri

| Metrik | Hedef | Açıklama |
|--------|-------|----------|
| 1.5 Altı Doğruluk | **75%+** | Model 1.5x altını doğru tahmin ediyor mu? |
| 1.5 Üstü Doğruluk | **75%+** | Model 1.5x üstünü doğru tahmin ediyor mu? |
| Para Kaybı Riski | **<20%** | Yanlış "1.5 üstü" tahmini oranı |
| ROI | **Pozitif** | Sanal kasa simülasyonunda kar |

---

## 🔄 Model Güncelleme

Yeni veri ile modeli güncellemek için:

1. Yeni verileri `jetx_data.db`'ye ekleyin
2. Google Colab'da notebook'u yeniden çalıştırın
3. Yeni ZIP'i indirin ve `models/` klasörüne çıkartın
4. Streamlit'i yeniden başlatın

---

## 💡 İpuçları

✅ **DO:**
- Her eğitim sonrası ZIP'i Google Drive'a yedekleyin
- Eski modelleri `models/backup/` klasörüne taşıyın
- JSON sonuç dosyalarını saklayın (performans karşılaştırması için)

❌ **DON'T:**
- Eğitim tamamlanmadan Colab'ı kapatmayın
- Farklı versiyonların modellerini karıştırmayın
- Scaler olmadan model kullanmayın

---

## 📞 Destek

Sorun yaşıyorsanız:

1. `MODEL_EGITIM_SONUCLARI.md` dosyasını kontrol edin
2. GitHub Issues'da arayın
3. Yeni bir Issue açın

---

## 📚 İlgili Dökümanlar

- [GPU_OPTIMIZATION_SUMMARY.md](GPU_OPTIMIZATION_SUMMARY.md) - GPU ayarları
- [JetX_Progressive_Training_Guide.md](JetX_Progressive_Training_Guide.md) - Progressive NN detayları
- [CPU_MODELS_INSTALLATION_GUIDE.md](CPU_MODELS_INSTALLATION_GUIDE.md) - CPU modelleri

---

**Son Güncelleme:** 2025-10-20  
**Versiyon:** 7.0
