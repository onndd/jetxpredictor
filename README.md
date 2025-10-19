# 🚀 JetX Predictor - AI Tahmin Sistemi

**Gelişmiş makine öğrenimi teknolojileriyle JetX çarpan tahmini**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14+-orange.svg)](https://www.tensorflow.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2+-green.svg)](https://catboost.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Educational-yellow.svg)](#)

## 📋 İçindekiler

- [Genel Bakış](#-genel-bakış)
- [Özellikler](#-özellikler)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Model Eğitimi](#-model-eğitimi)
- [Proje Yapısı](#-proje-yapısı)
- [Teknik Detaylar](#-teknik-detaylar)
- [Risk Yönetimi](#-risk-yönetimi)
- [Katkıda Bulunma](#-katkıda-bulunma)
- [Feragatname](#️-feragatname)

## 🎯 Genel Bakış

JetX Predictor, gelişmiş makine öğrenimi algoritmaları kullanarak JetX oyununda bir sonraki çarpan değerini tahmin etmeye çalışan deneysel bir AI sistemidir. Sistem, geçmiş oyun verilerindeki pattern'leri analiz ederek gelecek tahmininde bulunur.

### Ana Hedef
**1.5x kritik eşiği** doğru tahmin etmek:
- ✅ **1.5x ve üstü** = Kazanç (Güvenli bölge)
- ❌ **1.5x altı** = Kayıp (Riskli bölge)

### Temel Özellikler
- 🤖 **Hybrid Model Sistemi**: Neural Network + CatBoost
- 🎯 **Ensemble Predictor**: Birden fazla modeli birleştirerek güvenilirlik artırma
- 📊 **Adaptive Threshold**: Güven skoruna göre dinamik eşik ayarlama
- 🔬 **Backtesting**: Geçmiş verilerle performans testi
- 🛡️ **Risk Yönetimi**: Üç farklı mod (Rolling, Normal, Aggressive)
- 📈 **Gerçek Zamanlı Analiz**: Canlı tahmin ve görselleştirme

## ✨ Özellikler

### Model Özellikleri

#### 1. Progressive Neural Network
- Multi-input architecture (features + 4 time sequences)
- Multi-output prediction (regression, classification, threshold)
- Custom loss functions (Focal Loss, Threshold Killer Loss)
- Advanced feature engineering (150+ özellik)
- Time-series split validation
- Class imbalance handling

#### 2. CatBoost Models
- Gradient boosting regressor (değer tahmini)
- Binary classifier (1.5x eşik tahmini)
- Auto class weighting
- GPU acceleration support
- Fast inference (<1ms)

#### 3. Ensemble Predictor
Birden fazla modeli birleştirerek daha güvenilir tahminler:
- **Weighted Strategy**: Model güvenilirliklerine göre ağırlıklı ortalama
- **Unanimous Strategy**: Tüm modeller aynı yönde tahmin yapmalı
- **Confidence-Based**: Yüksek güvenli modele daha fazla ağırlık
- **Majority Strategy**: Basit çoğunluk oylaması

#### 4. Adaptive Threshold System
Güven skoruna ve performansa göre dinamik threshold ayarlama:
- Confidence-based: Güven skoruna göre (0.90+ → 1.5x, 0.50- → bahse girme)
- Performance-based: Geçmiş performansa göre (kazanma oranı bazlı)
- Hybrid: Her ikisinin kombinasyonu

### Özellik Mühendisliği

**Temel Özellikler:**
- Hareketli ortalamalar (25, 50, 100, 200, 500 pencere)
- 1.5x eşik analizi (oran, frekans, kritik bölge)
- Büyük çarpan mesafeleri (10x, 20x, 50x, 100x, 200x)
- Ardışık pattern'ler (yükseliş/düşüş streak)
- Volatilite metrikleri

**Gelişmiş Özellikler:**
- İstatistiksel dağılım (skewness, kurtosis, percentiles)
- Multi-timeframe momentum (kısa/orta/uzun vadeli trend)
- Recovery pattern detection
- Anomaly detection (Z-score, MAD)
- 15 farklı kategori seti (çok boyutlu analiz)

**Frequency Domain Analysis:**
- Wavelet Transform (time-frequency localization)
- Fourier Transform (periyodik pattern detection)
- Autocorrelation (lag-based patterns)

**Advanced Time Series:**
- DFA (Detrended Fluctuation Analysis)
- Hurst Exponent (trend persistence)

### Risk Yönetimi

#### Tahmin Modları

**🛡️ Rolling Mod (Konservatif)**
- Minimum güven: %80
- Önerilen çıkış: 1.5x
- Sermaye koruma odaklı
- En güvenli mod

**🎯 Normal Mod (Dengeli)**
- Minimum güven: %65
- Dengeli risk/getiri
- Standart kullanım için ideal

**⚡ Agresif Mod (Riskli)**
- Minimum güven: %50
- Yüksek risk, yüksek getiri
- Sadece deneyimli kullanıcılar için

#### Risk Kontrolleri
- Ardışık kayıp limiti (3 kayıp → uyarı)
- Kritik bölge filtreleme (1.45-1.55x)
- Güven skoru bazlı karar verme
- Performans bazlı uyarılar

### Backtesting & Analiz

**Backtesting Engine:**
- Historical data üzerinde performans testi
- ROI, kazanma oranı, Sharpe ratio hesaplama
- Maximum drawdown analizi
- Equity curve görselleştirme

**Model Karşılaştırma:**
- Farklı modellerin performans analizi
- Confusion matrix ve metrikler
- Feature importance analizi

## 🚀 Kurulum

### Gereksinimler

- Python 3.9 veya üzeri
- pip (Python package manager)
- GPU (opsiyonel, CatBoost ve TensorFlow için)

### Adım 1: Repository'yi Klonlayın

```bash
git clone https://github.com/onndd/jetxpredictor.git
cd jetxpredictor
```

### Adım 2: Sanal Ortam Oluşturun (Önerilen)

```bash
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Adım 3: Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

### Adım 4: Veritabanını Oluşturun

```bash
python -c "from utils.database import DatabaseManager; db = DatabaseManager(); print('✅ Veritabanı oluşturuldu')"
```

## 📊 Kullanım

### Streamlit Uygulamasını Başlatın

```bash
streamlit run app.py
```

Uygulama `http://localhost:8501` adresinde açılacaktır.

### Ana Özellikler

#### 1. Tahmin Yapma
- Mod seçimi (Rolling/Normal/Aggressive)
- Tahmin butonu ile yeni tahmin
- Güven skoru ve risk seviyesi gösterimi
- Dinamik threshold önerisi (aktifse)

#### 2. Veri Analizi
- Son N el grafiği (50-200 arası)
- İstatistiksel metrikler
- 1.5x eşik analizi

#### 3. Veri Girişi
- Manuel oyun sonucu ekleme
- Otomatik tahmin değerlendirme
- Geçmiş takibi

#### 4. Model Karşılaştırma (Gelişmiş)
- Farklı modellerin performans karşılaştırması
- Confusion matrix
- ROI ve kazanma oranı analizi

### Gelişmiş Özellikler

#### Ensemble Predictor Kullanımı

Sidebar'dan etkinleştirin:
1. ✅ "Ensemble Predictor" checkbox'ını işaretleyin
2. Oylama stratejisi seçin:
   - Weighted (Önerilen): CatBoost %60, NN %40
   - Unanimous: Her iki model de aynı tahminde
   - Confidence: En güvenli modele öncelik
   - Majority: Basit çoğunluk

#### Adaptive Threshold Kullanımı

Sidebar'dan etkinleştirin:
1. ✅ "Dinamik Threshold" checkbox'ını işaretleyin
2. Strateji seçin:
   - Hybrid (Önerilen): Güven + Performans
   - Confidence: Sadece güven skoru
   - Performance: Geçmiş performans

#### Backtesting

1. "Backtesting" bölümünü açın
2. Parametreleri ayarlayın:
   - Test veri sayısı (50-500)
   - Başlangıç sermayesi
   - Bahis tutarı
3. "Backtest Çalıştır" butonuna tıklayın
4. Sonuçları inceleyin (ROI, kazanma oranı, equity curve)

## 🎓 Model Eğitimi

### Google Colab'da Eğitim (Önerilen)

Modeller Google Colab'da eğitilir (ücretsiz GPU/TPU):

#### 1. CatBoost Model Eğitimi

```bash
# notebooks/jetx_CATBOOST_TRAINING.py dosyasını Colab'a yükleyin
```

Colab notebook özellikleri:
- Otomatik veri yükleme (jetx_data.db)
- GPU accelerated training
- Time-series split validation
- Model kaydetme ve indirme
- Performans raporlama

#### 2. Progressive Neural Network Eğitimi

```bash
# notebooks/JetX_PROGRESSIVE_TRAINING_Colab.ipynb dosyasını açın
```

Eğitim özellikleri:
- 3 aşamalı progressive training
- Custom loss functions (Focal Loss)
- Class imbalance handling
- Virtual bankroll callback
- Early stopping

### Eğitilmiş Modelleri İndirme

Colab'da eğitim tamamlandıktan sonra:

1. **Model dosyalarını indirin:**
   - `catboost_regressor.cbm`
   - `catboost_classifier.cbm`
   - `catboost_scaler.pkl`
   - `jetx_progressive_transformer.h5` (NN için)
   - `scaler_progressive_transformer.pkl` (NN için)

2. **`models/` klasörüne kopyalayın:**
   ```bash
   cp ~/Downloads/catboost*.* models/
   cp ~/Downloads/*progressive*.* models/
   ```

3. **Uygulamayı yeniden başlatın:**
   ```bash
   streamlit run app.py
   ```

## 📁 Proje Yapısı

```
jetxpredictor/
├── app.py                          # Ana Streamlit uygulaması
├── category_definitions.py         # Kategori tanımları ve feature engineering
├── requirements.txt                # Python bağımlılıkları
├── README.md                       # Bu dosya
│
├── config/
│   └── config.yaml                 # Konfigürasyon ayarları
│
├── data/
│   └── jetx_data.db               # SQLite veritabanı
│
├── models/                         # Eğitilmiş modeller (Colab'dan)
│   ├── catboost_regressor.cbm
│   ├── catboost_classifier.cbm
│   ├── catboost_scaler.pkl
│   ├── jetx_progressive_transformer.h5
│   └── scaler_progressive_transformer.pkl
│
├── notebooks/                      # Model eğitim scriptleri (Colab)
│   ├── jetx_CATBOOST_TRAINING.py
│   ├── jetx_PROGRESSIVE_TRAINING.py
│   ├── JetX_PROGRESSIVE_TRAINING_Colab.ipynb
│   ├── OPTUNA_HYPERPARAMETER_SEARCH.py
│   └── TRAIN_META_MODEL.py
│
├── pages/                          # Streamlit sayfaları
│   ├── 1_📊_Analiz.py
│   └── 2_🔬_Model_Karsilastirma.py
│
└── utils/                          # Yardımcı modüller
    ├── __init__.py
    ├── predictor.py               # Tahmin motoru (NN + CatBoost)
    ├── database.py                # Veritabanı yönetimi
    ├── risk_manager.py            # Risk yönetimi ve karar verme
    ├── ensemble_predictor.py      # Ensemble tahmin sistemi
    ├── adaptive_threshold.py      # Dinamik threshold yönetimi
    ├── backtesting.py             # Backtesting motoru
    ├── ensemble_manager.py        # Ensemble model yönetimi
    ├── config_loader.py           # Konfigürasyon yükleyici
    ├── custom_losses.py           # Özel loss fonksiyonları
    ├── focal_loss.py              # Focal loss implementasyonu
    ├── balanced_batch_generator.py # Dengeli batch üretimi
    ├── data_augmentation.py       # Veri augmentation
    ├── adaptive_weight_scheduler.py # Dinamik weight scheduler
    ├── lr_schedulers.py           # Learning rate schedulers
    ├── attention_layers.py        # Attention mekanizmaları
    ├── advanced_bankroll.py       # Gelişmiş bankroll yönetimi
    ├── dual_bankroll_system.py    # Dual bankroll sistemi
    └── virtual_bankroll_callback.py # Virtual bankroll callback
```

## 🔧 Teknik Detaylar

### Model Mimarileri

#### Progressive Neural Network

```python
# Multi-input architecture
inputs = {
    'features': Dense features (150+ özellik),
    'seq_50': LSTM(50 timesteps),
    'seq_200': LSTM(200 timesteps),
    'seq_500': LSTM(500 timesteps),
    'seq_1000': LSTM(1000 timesteps)
}

# Multi-output
outputs = {
    'regression': Değer tahmini (continuous),
    'classification': Kategori tahmini (3 class),
    'threshold': 1.5x eşik tahmini (binary)
}

# Custom losses
losses = {
    'regression': MAE,
    'classification': Categorical Crossentropy,
    'threshold': Focal Loss + Threshold Killer Loss
}
```

#### CatBoost Models

**Regressor:**
```python
CatBoostRegressor(
    iterations=1500,
    depth=10,
    learning_rate=0.03,
    loss_function='MAE',
    task_type='GPU'
)
```

**Classifier:**
```python
CatBoostClassifier(
    iterations=1500,
    depth=9,
    learning_rate=0.03,
    loss_function='Logloss',
    auto_class_weights='Balanced',
    task_type='GPU'
)
```

### Feature Engineering

**150+ özellik çıkarımı:**

1. **Temel İstatistikler** (25 özellik)
   - Mean, std, min, max, median (5 pencere)

2. **Threshold Özellikleri** (10 özellik)
   - 1.5x altı/üstü oranları
   - Kritik bölge analizi

3. **Distance Features** (5 özellik)
   - 10x, 20x, 50x, 100x, 200x'ten mesafe

4. **Streak Features** (5 özellik)
   - Ardışık yükseliş/düşüş
   - Kategori tekrarı

5. **Volatility** (10 özellik)
   - Farklı pencerelerde std, range, değişim

6. **Statistical Distribution** (10 özellik)
   - Skewness, kurtosis, percentiles, IQR

7. **Multi-timeframe Momentum** (20 özellik)
   - Kısa/orta/uzun vadeli momentum
   - Trend strength, acceleration

8. **Recovery Patterns** (10 özellik)
   - Volatilite normalizasyonu
   - Post-big-multiplier stability
   - Trend reversal

9. **Anomaly Detection** (10 özellik)
   - Z-score, MAD score
   - Outlier detection

10. **15 Kategori Setleri** (15 özellik)
    - Çok boyutlu kategorizasyon

11. **Advanced Analysis** (30+ özellik)
    - Wavelet transform
    - Fourier transform
    - Autocorrelation
    - DFA, Hurst exponent

### Veritabanı Şeması

**jetx_results:**
```sql
CREATE TABLE jetx_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    value REAL NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

**predictions:**
```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    predicted_value REAL,
    confidence_score REAL,
    above_threshold INTEGER,
    actual_value REAL,
    was_correct INTEGER,
    mode TEXT DEFAULT 'normal',
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Konfigürasyon

[`config/config.yaml`](config/config.yaml) dosyasında özelleştirilebilir:

```yaml
# Model ayarları
model:
  path: "models/jetx_model.h5"
  scaler_path: "models/scaler.pkl"

# Tahmin ayarları
prediction:
  critical_threshold: 1.5
  confidence_thresholds:
    aggressive: 0.50
    normal: 0.65
    rolling: 0.80
  max_consecutive_losses: 3

# Risk yönetimi
ui:
  alerts:
    sound_enabled: true
    critical_zone_alert: true
    loss_streak_alert: true
```

## 🛡️ Risk Yönetimi

### Kritik Kurallar

**Asla Unutmayın:**
1. ⚠️ Bu sistem %100 doğru **DEĞİLDİR**
2. 💰 Para kaybedebilirsiniz
3. 🎯 1.5x kritik eşiktir (altı = kayıp, üstü = kazanç)
4. 🛡️ Rolling modu en güvenlidir (%80+ güven)
5. 📊 Düşük güvende **OYNAMAYIN**
6. ⚡ 3 ardışık yanlış tahmin → **DUR**

### Güvenlik Önlemleri

**Maximum Bet Limits:**
- Günlük max: Sermayenin %20'si
- El başı max: Sermayenin %2'si

**Stop-Loss:**
- Günlük %10 kayıp → Dur
- 5 ardışık kayıp → Dur

**Cool-down Period:**
- Her kayıptan sonra 2 el bekle
- Büyük kayıp (>%5) → 5 el bekle

### Performans Metrikleri

**Hedef Performans:**
- ✅ 1.5x eşik doğruluğu: **%75+**
- ✅ Rolling mod doğruluğu: **%85+**
- ✅ Para kaybı riski: **<%20**
- ✅ ROI: **+3-5%**

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen şu alanlarda yardımcı olabilirsiniz:

1. **Model İyileştirmeleri**
   - Yeni özellik önerileri
   - Hyperparameter tuning
   - Yeni model mimarileri

2. **Risk Yönetimi**
   - Daha iyi risk filtreleme algoritmaları
   - Bankroll yönetimi stratejileri

3. **UI/UX Geliştirmeleri**
   - Daha iyi görselleştirmeler
   - Yeni analiz araçları

4. **Dokümantasyon**
   - Tutorial'lar
   - Video anlatımları
   - Örnek kullanım senaryoları

### Katkı Süreci

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit edin (`git commit -m 'feat: add amazing feature'`)
4. Push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## ⚖️ Feragatname

**ÖNEMLİ: LÜTFEN DİKKATLE OKUYUN**

Bu yazılım **eğitim ve araştırma amaçlıdır**. 

### Sorumluluk Reddi

- ❌ **HİÇBİR GARANTİ VERİLMEZ**: Bu sistem %100 doğru değildir ve para kaybedebilirsiniz.
- ❌ **YASAL SORUMLULUK**: Yazılımı kullanarak oluşabilecek kayıplardan geliştirici sorumlu değildir.
- ❌ **KUMAR BAĞIMLILIĞI**: Kumar ciddi bir sorundur. Yardım için: [Kumar Bağımlılığı Danışma Hattı]

### Kullanım Koşulları

✅ **Yapabilirsiniz:**
- Kişisel eğitim ve araştırma amaçlı kullanım
- Akademik çalışmalarda referans verme
- Açık kaynak katkıları

❌ **Yapamazsınız:**
- Ticari amaçlı kullanım (izin olmadan)
- Garantili kazanç vaat etme
- Başkalarını maddi zarara uğratma

### Etik Kullanım

- 🎓 Sadece kaybetmeyi göze alabileceğiniz parayla oynayın
- 🛡️ Sorumlu oyun ilkelerine uyun
- 📊 Sistemin sınırlamalarını bilin
- ⚠️ Risk yönetimi kurallarına uyun

### İletişim ve Destek

- 📧 GitHub Issues: Teknik sorular ve bug raporları
- 💬 Discussions: Genel sorular ve tartışmalar
- 📚 Wiki: Detaylı dokümantasyon

---

## 📄 Lisans

Bu proje eğitim ve araştırma amaçlıdır. Ticari kullanım için lütfen iletişime geçin.

---

## 🙏 Teşekkürler

- TensorFlow ve Keras ekibi
- CatBoost geliştiricileri
- Streamlit ekibi
- Açık kaynak topluluğu

---

**Son Güncelleme:** 12 Ekim 2025

**Versiyon:** 2.0

**Geliştirici:** [onndd](https://github.com/onndd)

---

**⚠️ Hatırlatma: Bu bir tahmin sistemidir, garanti değildir. Sorumlu oynayın!**
