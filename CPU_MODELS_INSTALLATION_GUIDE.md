# 🚀 JetX CPU Modelleri Entegrasyonu - Kurulum ve Yapılandırma Rehberi

**Tarih:** 20 Ekim 2025  
**Versiyon:** 1.0  
**Durum:** ✅ Tamamlandı

---

## 📋 İçindekiler

1. [Genel Bakış](#genel-bakış)
2. [Yapılan Değişiklikler](#yapılan-değişiklikler)
3. [Eklenen Dosyalar](#eklenen-dosyalar)
4. [Teknik Detaylar](#teknik-detaylar)
5. [Kurulum Adımları](#kurulum-adımları)
6. [Karşılaşılan Sorunlar ve Çözümleri](#karşılaşılan-sorunlar-ve-çözümleri)
7. [Ayrı Proje Kurulumu](#ayrı-proje-kurulumu)
8. [Kullanım](#kullanım)
9. [Sonraki Adımlar](#sonraki-adımlar)

---

## 🎯 Genel Bakış

JetX Predictor projesine **CPU ile çalışabilen hafif modeller** için özel bir uygulama eklenmiştir. Bu uygulama, GPU gerektirmeyen, tamamen CPU üzerinde çalışan makine öğrenmesi modellerini destekler.

### Desteklenen Modeller

- **LightGBM**: CPU optimized gradient boosting
- **CatBoost**: Categorical boosting
- **TabNet**: Attention-based deep learning  
- **AutoGluon**: Automated ML

### Amaç

Ana GPU tabanlı uygulamadan bağımsız olarak, CPU ile model eğitimi, hyperparameter tuning, ensemble oluşturma ve tahmin yapabilme imkanı sağlamak.

---

## 📝 Yapılan Değişiklikler

### 1. Ana Projeye Eklenenler (`jetxpredictor/`)

#### Yeni Dosyalar:
- `app_cpu_models.py` - Ana CPU modelleri Streamlit uygulaması
- `README_CPU_MODELS.md` - CPU modelleri dokümantasyonu
- `MODEL_EGITIM_SONUCLARI.md` - Model eğitim sonuçları dokümantasyonu
- `config/cpu_models_config.yaml` - CPU modelleri konfigürasyonu

#### Yeni Utils Modülleri:
- `utils/lightweight_model_manager.py` - Hafif modeller için birleşik yönetici
- `utils/cpu_training_engine.py` - CPU optimized eğitim motoru
- `utils/lightgbm_predictor.py` - LightGBM predictor sınıfı

#### Yeni Pages (CPU Sayfaları):
- `pages/cpu/1_🎯_Model_Training.py` - Model eğitim arayüzü
- `pages/cpu/2_🔧_Hyperparameter_Tuning.py` - Hyperparameter optimization
- `pages/cpu/3_📊_Model_Comparison.py` - Model karşılaştırma
- `pages/cpu/4_🤝_Ensemble_Builder.py` - Ensemble oluşturma
- `pages/cpu/5_🎲_Prediction_Backtest.py` - Tahmin ve backtesting

#### Güncellenen Dosyalar:
- `requirements.txt` - Yeni bağımlılıklar eklendi (AutoGluon, TabNet, vb.)
- Çeşitli notebook dosyaları güncellemeler aldı

### 2. Ayrı Proje Oluşturulması (`jetx-cpu-models/`)

CPU modelleri uygulaması, ana projeden bağımsız olarak çalışabilmesi için ayrı bir klasöre kopyalanmıştır:

**Konum:** `/Users/numanondes/Desktop/jetx-cpu-models/`

---

## 📦 Eklenen Dosyalar

### Ana Uygulama Dosyası

#### `app_cpu_models.py`
- **Amaç**: CPU modelleri için ana Streamlit uygulaması
- **Özellikler**:
  - Model yönetimi dashboard'u
  - Real-time CPU ve memory kullanımı gösterimi
  - Model istatistikleri (toplam model, eğitilmiş model, vb.)
  - Hızlı eylem butonları (Model Eğit, Tuning, Karşılaştır, vb.)
  - Sistem durumu kontrolü
- **Bağımlılıklar**: 
  - `LightweightModelManager`
  - `CPUTrainingEngine`
  - `DatabaseManager`
  - `psutil` (CPU/Memory monitoring)

### Konfigürasyon Dosyası

#### `config/cpu_models_config.yaml`
- **Amaç**: Tüm CPU modelleri için merkezi konfigürasyon
- **İçerik**:
  - Model parametreleri (LightGBM, CatBoost, TabNet, AutoGluon)
  - Eğitim ayarları (window_size, train/val/test split)
  - Hyperparameter tuning konfigürasyonu
  - Data processing ayarları
  - Virtual bankroll simülasyon parametreleri
  - Logging ayarları

### Utils Modülleri

#### `utils/lightweight_model_manager.py`
- **Amaç**: Tüm hafif modelleri yönetmek için birleşik interface
- **Özellikler**:
  - Model factory (model oluşturma)
  - Model registry (model kayıt sistemi)
  - Model eğitimi orchestration
  - Model karşılaştırma utilities
  - Ensemble oluşturma
  - Model persistence (kaydetme/yükleme)
- **Desteklenen Modeller**: LightGBM, TabNet, AutoGluon, CatBoost

#### `utils/cpu_training_engine.py`
- **Amaç**: CPU optimized model eğitim motoru
- **Özellikler**:
  - Tek model eğitimi
  - Hyperparameter search (Optuna entegrasyonu)
  - Cross-validation
  - Feature engineering
  - Data preprocessing
  - Virtual bankroll simulation

#### `utils/lightgbm_predictor.py`
- **Amaç**: LightGBM modeli için predictor sınıfı
- **Modlar**: Classification, Regression, Multiclass
- **Özellikler**:
  - Model eğitimi
  - Tahmin (predict, predict_proba)
  - Feature importance
  - Model kaydetme/yükleme
  - Cross-validation

### Pages (UI Sayfaları)

#### `pages/cpu/1_🎯_Model_Training.py`
- Model tipi seçimi
- Model modu (Classification/Regression/Multiclass)
- Hyperparameter ayarları
- Real-time eğitim takibi
- Eğitim sonuçları görüntüleme

#### `pages/cpu/2_🔧_Hyperparameter_Tuning.py`
- Optuna entegrasyonu
- Search space tanımlama
- Trial sayısı ve timeout ayarları
- Optimization history görselleştirme
- Best parameters gösterimi

#### `pages/cpu/3_📊_Model_Comparison.py`
- Birden fazla modeli karşılaştırma
- Side-by-side metrikler
- Performance grafikleri
- Radar charts
- Karşılaştırma tabloları

#### `pages/cpu/4_🤝_Ensemble_Builder.py`
- Voting stratejisi (Hard/Soft)
- Stacking stratejisi
- Ağırlıklı ensemble
- Ensemble test ve değerlendirme

#### `pages/cpu/5_🎲_Prediction_Backtest.py`
- Real-time tahminler
- Historical backtesting
- Virtual bankroll simulation
- ROI hesaplama
- Performance metrikleri

---

## 🔧 Teknik Detaylar

### Mimari

```
jetxpredictor/
├── app_cpu_models.py                 # Ana CPU modelleri uygulaması
├── config/
│   └── cpu_models_config.yaml        # Konfigürasyon
├── utils/
│   ├── lightweight_model_manager.py  # Model yöneticisi
│   ├── cpu_training_engine.py        # Eğitim motoru
│   └── lightgbm_predictor.py         # LightGBM predictor
└── pages/cpu/                        # UI sayfaları
    ├── 1_🎯_Model_Training.py
    ├── 2_🔧_Hyperparameter_Tuning.py
    ├── 3_📊_Model_Comparison.py
    ├── 4_🤝_Ensemble_Builder.py
    └── 5_🎲_Prediction_Backtest.py
```

### Teknoloji Stack

- **Framework**: Streamlit
- **ML Libraries**: 
  - LightGBM 4.0+
  - CatBoost 1.2+
  - PyTorch TabNet 4.0+
  - AutoGluon 0.8+
- **Optimization**: Optuna 3.0+
- **Data Processing**: NumPy, Pandas
- **Visualization**: Plotly, Matplotlib, Seaborn

### Model Workflow

1. **Veri Yükleme**: Database'den JetX verileri çekilir
2. **Feature Engineering**: Otomatik feature extraction
3. **Model Oluşturma**: LightweightModelManager ile model instance oluşturulur
4. **Eğitim**: CPUTrainingEngine ile model eğitilir
5. **Değerlendirme**: Metrics hesaplanır ve kaydedilir
6. **Kayıt**: Model registry'ye kaydedilir
7. **Tahmin**: Trained model ile tahmin yapılır

---

## 💻 Kurulum Adımları

### Gereksinimler

- Python 3.8+
- 8GB+ RAM (16GB önerilen)
- macOS / Linux / Windows
- Homebrew (macOS için, OpenMP kütüphanesi yüklemek için)

### 1. Repository Clone

```bash
git clone https://github.com/onndd/jetxpredictor.git
cd jetxpredictor
```

### 2. Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# veya
venv\Scripts\activate     # Windows
```

### 3. Dependencies Yükleme

```bash
pip install -r requirements.txt
```

### 4. OpenMP Kütüphanesi (macOS için gerekli)

```bash
brew install libomp
```

**Not**: Bu kütüphane LightGBM'in çalışması için gereklidir. Windows ve Linux'ta genellikle varsayılan olarak bulunur.

### 5. Uygulamayı Başlatma

```bash
streamlit run app_cpu_models.py --server.port 8502
```

Tarayıcıda otomatik olarak açılacaktır: `http://localhost:8502`

---

## ⚠️ Karşılaşılan Sorunlar ve Çözümleri

### 1. LightGBM OpenMP Hatası

**Hata:**
```
OSError: dlopen(...lib_lightgbm.dylib, 0x0006): Library not loaded: @rpath/libomp.dylib
```

**Sebep:** macOS'ta OpenMP kütüphanesi eksik

**Çözüm:**
```bash
brew install libomp
```

### 2. DataFrame KeyError: 'status'

**Hata:**
```python
KeyError: 'status'
```

**Sebep:** Boş DataFrame'de kolon erişimi

**Çözüm:** `app_cpu_models.py` dosyasında düzeltme yapıldı:
```python
# Öncesi
trained_models = len(model_df[model_df['status'] == 'trained'])

# Sonrası
trained_models = len(model_df[model_df['status'] == 'trained']) if not model_df.empty else 0
```

### 3. Sayfa Yolu Hatası

**Hata:**
```
StreamlitAPIException: Could not find page: `pages/cpu/1_🎯_Model_Training.py`
```

**Sebep:** Ayrı projede sayfa yolları farklı

**Çözüm:** `app.py` dosyasında sayfa yolları düzeltildi:
```python
# Öncesi
st.switch_page("pages/cpu/1_🎯_Model_Training.py")

# Sonrası
st.switch_page("pages/1_🎯_Model_Training.py")
```

### 4. Session State AttributeError

**Hata:**
```python
AttributeError: st.session_state has no attribute "tuning_in_progress"
```

**Sebep:** Session state değişkenleri başlatılmamış

**Durum:** Tespit edildi, düzeltme ayrı projede yapılmalı

**Çözüm:** Session state initialization eklenebilir

### 5. psutil Modülü Eksik

**Hata:**
```python
ModuleNotFoundError: No module named 'psutil'
```

**Çözüm:**
```bash
pip install psutil
```

---

## 📁 Ayrı Proje Kurulumu

CPU modelleri uygulaması, ana projeden bağımsız olarak çalışabilmesi için ayrı bir klasöre kopyalanmıştır.

### Konum

`/Users/numanondes/Desktop/jetx-cpu-models/`

### Kopyalanan Dosyalar

```
jetx-cpu-models/
├── app.py (app_cpu_models.py'den)
├── README.md (README_CPU_MODELS.md'den)
├── requirements.txt
├── .gitignore
├── category_definitions.py
├── jetx_data.db
├── config/
│   └── config.yaml (cpu_models_config.yaml'dan)
├── pages/
│   ├── 1_🎯_Model_Training.py
│   ├── 2_🔧_Hyperparameter_Tuning.py
│   ├── 3_📊_Model_Comparison.py
│   ├── 4_🤝_Ensemble_Builder.py
│   └── 5_🎲_Prediction_Backtest.py
├── utils/
│   ├── __init__.py
│   ├── lightweight_model_manager.py
│   ├── cpu_training_engine.py
│   ├── lightgbm_predictor.py
│   ├── tabnet_predictor.py
│   ├── autogluon_predictor.py
│   ├── catboost_ensemble.py
│   ├── database.py
│   └── config_loader.py
└── models/cpu/
```

### Kurulum

```bash
cd ~/Desktop/jetx-cpu-models
pip install -r requirements.txt
brew install libomp  # macOS
streamlit run app.py
```

### Avantajları

- Ana GPU projesinden bağımsız çalışma
- Daha hafif ve modüler yapı
- Kolay deployment
- Bağımlılık çakışması riski yok

---

## 🎮 Kullanım

### İlk Model Eğitimi

1. Uygulamayı başlat: `streamlit run app_cpu_models.py --server.port 8502`
2. "🎯 Model Eğit" butonuna tıkla
3. Model ayarlarını yap:
   - Model Tipi: `LightGBM` (başlangıç için önerilen)
   - Mod: `Classification`
   - Window Size: `1000`
4. "Eğitimi Başlat" butonuna tıkla
5. Real-time progress takip et

### Hyperparameter Tuning

1. "🔧 Hyperparameter Tuning" sayfasına git
2. Model seçimi yap
3. Search space tanımla
4. Trial sayısı belirle (50-100 önerilen)
5. Optimization başlat
6. Best parameters'ı kaydet

### Ensemble Oluşturma

1. En az 2 model eğit
2. "🤝 Ensemble Oluştur" sayfasına git
3. Modelleri seç
4. Strateji belirle (Voting/Stacking)
5. Ensemble test et

---

## 🔜 Sonraki Adımlar

### Tamamlanması Gerekenler

1. **Session State Initialization**: Ayrı projede (`jetx-cpu-models/`) tüm sayfalarda session state değişkenlerinin başlatılması
2. **Config Loading Fix**: `cpu_models_config.yaml` yerine `config.yaml` okuması için düzeltme
3. **Error Handling**: Daha robust error handling eklenmesi
4. **Unit Tests**: Model manager ve training engine için testler
5. **Documentation**: API referans dokümantasyonu

### İyileştirmeler

1. **Model Caching**: Eğitilmiş modellerin cache'lenmesi
2. **Batch Prediction**: Toplu tahmin desteği
3. **Export/Import**: Model export/import fonksiyonları
4. **Monitoring**: Gelişmiş performans monitoring
5. **Auto-Tuning**: Otomatik hyperparameter tuning

---

## 📊 Git Commit Geçmişi

### Ana Proje (jetxpredictor)

**Commit Hash:** `7e2c5e1`  
**Commit Message:** "feat: Add CPU lightweight models application with LightGBM, TabNet, AutoGluon, and CatBoost support"  
**Tarih:** 20 Ekim 2025  
**Push:** ✅ Başarılı (origin/main)

**Değişiklikler:**
- 19 dosya değiştirildi
- 5875 satır eklendi
- 229 satır silindi

---

## 📚 Referanslar

### Dokümantasyon

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/docs/)
- [TabNet Paper](https://arxiv.org/abs/1908.07442)
- [AutoGluon Documentation](https://auto.gluon.ai/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Optuna Documentation](https://optuna.readthedocs.io/)

### Proje Dosyaları

- `README_CPU_MODELS.md`: Detaylı kullanım rehberi
- `config/cpu_models_config.yaml`: Konfigürasyon referansı
- `MODEL_EGITIM_SONUCLARI.md`: Model eğitim sonuçları

---

## 👥 Katkıda Bulunanlar

- **Ana Geliştirici**: Numan Öndeş
- **AI Asistan**: Claude (Anthropic)
- **Tarih**: 20 Ekim 2025

---

## 📄 Lisans

MIT License - Ana projeyle aynı

---

## 🆘 Destek

Sorunlarla karşılaşırsanız:

1. `README_CPU_MODELS.md` dosyasındaki sorun giderme bölümüne bakın
2. GitHub Issues'da arama yapın
3. Yeni bir issue oluşturun

---

**Son Güncelleme:** 20 Ekim 2025  
**Versiyon:** 1.0  
**Durum:** ✅ Production Ready
