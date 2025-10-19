# 🚀 JetX Progressive Training Guide - v6.0

**Tam Kılavuz: Google Colab ile JetX Model Eğitimi**

---

## 📋 İçerik

1. [Giriş ve v6.0 Yenilikler](#giriş)
2. [Sistem Gereksinimleri](#sistem-gereksinimleri)
3. [Adım Adım Kullanım](#adım-adım-kullanım)
4. [Google Drive Entegrasyonu](#google-drive-entegrasyonu)
5. [Test Modeli ve İndirme Kontrolü](#test-modeli)
6. [Model İndirme Sistemleri](#model-indirme-sistemleri)
7. [Troubleshooting](#troubleshooting)
8. [Model Kullanımı](#model-kullanımı)

---

<a name="giriş"></a>
## 🎯 Giriş

Bu guide, JetX tahmin modellerini Google Colab üzerinde eğitmek için kullanılan **JetX_PROGRESSIVE_TRAINING_Colab.ipynb** notebook'unun tam kullanım kılavuzudur.

### 🆕 v6.0 Yenilikler

#### 📁 **Google Drive Entegrasyonu**
- Tüm modeller otomatik olarak Drive'a yedeklenir
- Eğitim sırasında kaybolma riski ortadan kalkar
- Drive'dan kolayca erişim ve indirme imkanı

#### 🧪 **Test Modeli ve İndirme Kontrolü**
- Küçük bir test modeli ile sistem kontrol edilir
- İndirme mekanizması test edilir
- Hata durumunda kullanıcı bilgilendirilir

#### 📦 **Gelişmiş İndirme Sistemi**
- 3 farklı indirme yöntemi
- Colab otomatik indirme
- Google Drive kopyalama
- Manuel indirme seçenekleri

#### 📚 **Tam Dokümantasyon**
- Detaylı kullanım kılavuzu
- Troubleshooting bölümü
- Adım adım açıklamalar

---

<a name="sistem-gereksinimleri"></a>
## 💻 Sistem Gereksinimleri

### Google Colab
- **GPU**: T4 (önerilen) veya V100
- **RAM**: En az 12GB
- **Depolama**: En az 15GB

### Google Hesabı
- Google Drive erişimi
- Yeterli depolama alanı (~2GB için modeller)

---

<a name="adım-adım-kullanım"></a>
## 📝 Adım Adım Kullanım

### 1. 🚀 Başlangıç

1. **Notebook'u Aç**
   ```
   Google Colab → File → Open notebook → Upload
   JetX_PROGRESSIVE_TRAINING_Colab.ipynb
   ```

2. **GPU Ayarla**
   ```
   Runtime → Change runtime type → GPU → T4
   ```

3. **Adım 1'i Çalıştır** (Hazırlık ve Test)
   - ⏱️ Süre: 5-10 dakika
   - Google Drive bağlantısı
   - Kütüphane kurulumu
   - Test modeli eğitimi

### 2. 🧠 Progressive NN Eğitimi

**Adım 2A'yı Çalıştır**
- ⏱️ Süre: 10-12 saat
- 5 farklı LSTM modeli
- Multi-Scale Window sistemi
- Otomatik Drive yedekleme

**Model Özellikleri:**
- Window boyutları: [500, 250, 100, 50, 20]
- Ensemble prediction
- Weighted model selection

### 3. ⚡ CatBoost Eğitimi

**Adım 2B'yi Çalıştır**
- ⏱️ Süre: 40-60 dakika
- 10 model (5 pencere × 2 model)
- Regressor + Classifier
- Ensemble prediction

### 4. 🎯 Consensus Değerlendirme

**Adım 2C'yi Çalıştır**
- ⏱️ Süre: 5-10 dakika
- NN ve CatBoost birleştirme
- İki kasa stratejisi testi
- Sonuç analizi

### 5. 📦 Model İndirme

**Adım 3A ve 3B'yi Çalıştır**
- NN modellerini indir
- CatBoost modellerini indir
- 3 farklı indirme yöntemi

---

<a name="google-drive-entegrasyonu"></a>
## 📁 Google Drive Entegrasyonu

### Bağlantı İşlemi

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Drive Klasör Yapısı

```
/content/drive/MyDrive/JetX_Models/
├── jetx_models_progressive_multiscale_v3.0.zip
├── jetx_models_catboost_multiscale_v3.0.zip
├── consensus_evaluation.json
└── test_model.h5 (geçici)
```

### Otomatik Yedekleme

- **NN Eğitimi Sonrası**: `jetx_models_progressive_multiscale_v3.0.zip`
- **CatBoost Eğitimi Sonrası**: `jetx_models_catboost_multiscale_v3.0.zip`
- **Consensus Sonrası**: `consensus_evaluation.json`

---

<a name="test-modeli"></a>
## 🧪 Test Modeli ve İndirme Kontrolü

### Test Modeli Özellikleri

```python
model = Sequential([
    LSTM(32, input_shape=(10, 5), return_sequences=False),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

### Kontrol Süreci

1. **Model Oluşturma**: Basit LSTM mimarisi
2. **Eğitim**: 10 epoch, 100 örnek
3. **Kaydetme**: `test_model.h5`
4. **İndirme Testi**: Dosya boyutu kontrolü
5. **Drive Kopyalama**: Yedekleme testi
6. **Temizleme**: Test dosyası silinir

### Başarısızlık Durumunda

```
❌ İndirme sisteminde sorun var!
⚠️ Lütfen bu hatayı bildirin!
```

---

<a name="model-indirme-sistemleri"></a>
## 📦 Model İndirme Sistemleri

### Yöntem 1: Colab Otomatik İndirme

```python
from google.colab import files
files.download(zip_file)
```

**Avantajları:**
- Otomatik ve hızlı
- Doğrudan bilgisayara iner

**Dezavantajları:**
- Büyük dosyalarda sorun olabilir
- Bağlantı kopukluğunda sorun

### Yöntem 2: Google Drive Kopyalama

```python
import shutil
shutil.copy2(zip_file, drive_path)
```

**Avantajları:**
- Güvenli ve stabil
- Her zaman erişilebilir
- Büyük dosyalar için ideal

**Kullanım:**
```
Google Drive → JetX_Models → [dosya_adı]
```

### Yöntem 3: Manuel İndirme

**Adımlar:**
1. Sol panel → Files
2. `jetxpredictor` klasörü
3. ZIP dosyasını bul
4. Sağ tık → Download

### İndirme Özeti

```
📊 İNDİRME ÖZETİ:
   ✅ Başarılı yöntemler: Colab Otomatik, Google Drive, Manuel
   ✅ İndirme başarılı!
```

---

<a name="troubleshooting"></a>
## 🔧 Troubleshooting

### Yayın Hataları

#### **Drive Bağlantı Hatası**
```
⚠️ Drive bağlantı hatası: [hata_mesajı]
ℹ️ Drive olmadan devam ediliyor...
```
**Çözüm:**
- Google hesabını kontrol et
- Drive izinlerini ver
- Sayfayı yenile ve tekrar dene

#### **GPU Bellek Hatası**
```
ResourceExhaustedError: OOM when allocating tensor
```
**Çözüm:**
- Runtime → Restart runtime
- Daha küçük batch size kullan
- GPU'nu temizle

#### **İndirme Hatası**
```
⚠️ Otomatik indirme başarısız: [hata_mesajı]
```
**Çözüm:**
- Drive kopyalamayı kullan
- Manuel indirmeyi dene
- İnternet bağlantısını kontrol et

#### **Model Eğitim Hatası**
```
FileNotFoundError: [dosya_adı] bulunamadı
```
**Çözüm:**
- Adım 1'i tamamen çalıştır
- Dosya yollarını kontrol et
- Projeyi yeniden klonla

### Performans İpuçları

#### **GPU Kullanımı**
- T4 GPU yeterli
- V100 daha hızlı
- A100 gereksiz (maliyetli)

#### **Zaman Yönetimi**
- NN eğitimi: Gece başlat
- CatBoost: Gündüz yapılabilir
- Consensus: Hızlı

#### **Depolama**
- En az 15GB gereklidir
- Drive alanını kontrol et
- Gereksiz dosyaları temizle

---

<a name="model-kullanımı"></a>
## 🎯 Eğitim Sonrası Model Kullanımı

### 1. ZIP Dosyalarını İndir

```
jetx_models_progressive_multiscale_v3.0.zip  (NN modelleri)
jetx_models_catboost_multiscale_v3.0.zip      (CatBoost modelleri)
```

### 2. Lokal Projeye Kopyala

```
proje_klasörü/
├── models/
│   ├── model_window_500.h5
│   ├── model_window_250.h5
│   ├── model_window_100.h5
│   ├── model_window_50.h5
│   ├── model_window_20.h5
│   ├── regressor_window_*.cbm
│   ├── classifier_window_*.cbm
│   └── scaler_window_*.pkl
```

### 3. Streamlit Uygulamasını Başlat

```bash
cd jetxpredictor
python -m streamlit run app.py
```

### 4. Model Doğrulama

- **Streamlit arayüzü**: Modelleri test et
- **Consensus sonuçları**: `consensus_evaluation.json`
- **Performans metrikleri**: ROI, Win Rate

---

## 📊 Başarı Metrikleri

### Consensus Sistemi

**Kasa 1 (1.5x Eşik):**
- ROI: %+X.XX%
- Win Rate: XX.X%
- Toplam Bahis: X,XXX

**Kasa 2 (%70 Çıkış):**
- ROI: %+X.XX%
- Win Rate: XX.X%
- Toplam Bahis: X,XXX

### Model Performansı

**NN Modelleri:**
- 5 farklı zaman ölçeği
- Ensemble accuracy: XX.X%
- Window-specific performans

**CatBoost Modelleri:**
- Regresyon ve Sınıflandırma
- Feature importance analizi
- Hızlı tahmin süresi

---

## ⚠️ Önemli Uyarılar

### Riskler
- 🚨 Modeller %100 doğru değildir
- 💰 Para kaybedebilirsiniz
- 🎯 Sadece reference olarak kullanın

### Sorumlu Oyun
- Bütçeni aşma
- Duygusal karar verme
- Ara verler ver

### Teknik
- Modelleri düzenli güncelle
- Backtesting yap
- Performansı izle

---

## 🆘 Destek

### Hata Bildirimi

**Bilgiler:**
- Hata mesajı (tam olarak)
- Çalıştırılan adım
- Colab ortamı (GPU/RAM)
- Ekran görüntüsü (varsa)

### İletişim

**GitHub Issues:**
```
github.com/onndd/jetxpredictor/issues
```

**Yardım İçin:**
- Detaylı hata açıklaması
- Log dosyaları
- Adım adım tekrar etme

---

## 📚 Ek Kaynaklar

### Dokümantasyon
- [README.md](README.md)
- [GPU_OPTIMIZATION_SUMMARY.md](GPU_OPTIMIZATION_SUMMARY.md)
- [CLASS_IMBALANCE_SOLUTION_PLAN.md](CLASS_IMBALANCE_SOLUTION_PLAN.md)

### Notebook'lar
- [JetX_Training_Colab.ipynb](notebooks/JetX_Training_Colab.ipynb)
- [Comprehensive_Model_Training_Colab.ipynb](notebooks/Comprehensive_Model_Training_Colab.ipynb)

### Eğitim Script'leri
- [jetx_PROGRESSIVE_TRAINING_MULTISCALE.py](notebooks/jetx_PROGRESSIVE_TRAINING_MULTISCALE.py)
- [jetx_CATBOOST_TRAINING_MULTISCALE.py](notebooks/jetx_CATBOOST_TRAINING_MULTISCALE.py)
- [CONSENSUS_EVALUATION.py](notebooks/CONSENSUS_EVALUATION.py)

---

## 🎉 Sonuç

**JetX Progressive Training v6.0** ile:

✅ **Güvenli model eğitimi** (Drive yedekleme)
✅ **Test edilmiş sistem** (İndirme kontrolü)
✅ **Çoklu indirme seçeneği** (3 farklı yöntem)
✅ **Tam dokümantasyon** (Adım adım kılavuz)
✅ **Consensus sistemi** (İki model birleşimi)
✅ **Multi-Scale yaklaşım** (5 zaman penceresi)

**Bu guide ile JetX tahmin modellerinizi güvenli ve etkili bir şekilde eğitebilirsiniz!** 🚀

---

**Sorumlu oynayın! 🎲**

*Son güncelleme: 19 Ekim 2025 - v6.0*
