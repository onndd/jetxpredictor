# JetX PREDICTOR - DETAYLI PROJE PLANI

## 📊 PROJE GENEL BİLGİLERİ

**Proje Adı:** JetX Predictor - AI-Powered Prediction System  
**Amaç:** 7000+ geçmiş veri analizi ile pattern tabanlı tahmin sistemi  
**Kritik Eşik:** 1.5x (Altı kayıp, üstü kazanç)  
**Mevcut Veri:** 6091 kayıt (1.0x - 1808.04x arası)

### Veri Dağılımı
- **1.5x Altı (Risk Bölgesi):** %35.05 (2135 kayıt)
- **1.5x Üstü (Güvenli Bölge):** %64.95 (3956 kayıt)

## 🎯 PROJE MİMARİSİ

### 1. ORTAMLAR
- **Google Colab:** Model eğitimi, ağır hesaplamalar, GPU kullanımı
- **Lokal Streamlit:** Kullanıcı arayüzü, tahmin gösterimi, veri girişi

### 2. MODEL STRATEJİSİ
- **Ana Motor:** N-BEATS + TCN Hibrit
- **Pencereler:** Kısa (50), Orta (200), Uzun (500), Derin (1000)
- **Kategori Setleri:** 15 farklı perspektif
- **Psikolojik Analiz:** Tuzak pattern tespiti
- **Soğuma Analizi:** Model tarafından öğrenilecek

### 3. MODLAR
- **Normal Mod:** %65+ güven, orta risk
- **Rolling Mod:** %80+ güven, sermaye koruma odaklı
- **Agresif Mod:** %50+ güven, yüksek risk-getiri

## 📋 DETAYLI UYGULAMA ADIMLARI

### FAZA 1: VERİ ANALİZİ VE HAZİRLIK (Google Colab)

#### 1.1 Veri Keşfi ve Analiz
- [ ] SQLite veritabanını Colab'a yükleme
- [ ] Temel istatistiksel analiz (ortalama, medyan, std, varyans)
- [ ] Kategorik dağılım analizi (15 farklı kategori seti)
- [ ] Kritik 1.5x eşiği detaylı analizi
- [ ] Aykırı değer (outlier) tespiti
- [ ] Zaman serisi görselleştirmeleri (Plotly/Matplotlib)
- [ ] Ardışıklık pattern analizi

#### 1.2 Özellik Mühendisliği (Feature Engineering)
- [ ] **Temel Özellikler:**
  - Son N elin ortalaması (N=5,10,20,50,100)
  - Hareketli ortalamalar (moving averages)
  - Volatilite metrikleri (std, range)
  
- [ ] **Mesafe Özellikleri:**
  - Son 10x'ten bu yana geçen el sayısı
  - Son 20x'ten bu yana geçen el sayısı
  - Son 50x, 100x, 500x, 1000x mesafeleri
  
- [ ] **Ardışık Pattern Özellikleri:**
  - Ardışık yükseliş/düşüş sayısı
  - Aynı kategoride ardışık tekrar sayısı
  - Zigzag pattern tespiti
  
- [ ] **Bölge Yoğunluk Özellikleri:**
  - Son 50 elde kategori dağılımı
  - Son 100 elde risk/güvenli bölge oranı
  - Trend göstergeleri
  
- [ ] **Kritik Eşik Özellikleri:**
  - 1.5x üstü/altı geçiş sıklığı
  - 1.45-1.55 kritik bölgede kalma süresi
  - Eşik yakınlığı skorları
  
- [ ] **15 Kategori Seti Encoding:**
  - Her set için one-hot/label encoding
  - Kategori geçiş matrisleri
  - Kategori momentum skorları

#### 1.3 Veri Hazırlama
- [ ] Train/Validation/Test split (%70/%15/%15)
- [ ] Zaman serisi cross-validation stratejisi
- [ ] Normalizasyon/Standardizasyon
- [ ] Sequence oluşturma (LSTM/TCN için)
- [ ] Data augmentation stratejileri (opsiyonel)

### FAZA 2: MODEL GELİŞTİRME (Google Colab)

#### 2.1 N-BEATS Model Geliştirme
- [ ] Temel N-BEATS mimarisi kurulumu
- [ ] Üç pencere sistemi implementasyonu:
  - Kısa pencere (50 el) modülü
  - Orta pencere (200 el) modülü
  - Uzun pencere (500 el) modülü
- [ ] Stack ve block konfigürasyonları
- [ ] Trend ve mevsimsellik decomposition
- [ ] Hyperparameter tuning (Optuna kullanarak)

#### 2.2 TCN Model Geliştirme
- [ ] Temporal Convolutional Network mimarisi
- [ ] Dilated convolution yapılandırması
- [ ] Residual connections implementasyonu
- [ ] 1000 el derin analiz penceresi
- [ ] Receptive field optimizasyonu
- [ ] Hyperparameter tuning

#### 2.3 Hibrit Model Oluşturma
- [ ] N-BEATS ve TCN çıktılarını birleştirme
- [ ] Attention mekanizması ekleme (opsiyonel)
- [ ] Ensemble stratejisi geliştirme
- [ ] Ağırlıklandırma optimizasyonu
- [ ] Multi-task learning (kategori + değer tahmini)

#### 2.4 Psikolojik Analiz Modülü
- [ ] Tuzak pattern tespit algoritması:
  - Honeypot pattern
  - Recovery trap pattern
  - False momentum pattern
- [ ] Pattern encoding ve feature integration
- [ ] Anomaly detection modülü
- [ ] Risk skoru hesaplama

#### 2.5 Soğuma Dönemi Öğrenme
- [ ] Unsupervised learning yaklaşımı
- [ ] Büyük çarpan sonrası davranış analizi
- [ ] Clustering tabanlı soğuma tespiti
- [ ] Dinamik soğuma pattern öğrenme
- [ ] Soğuma süresi ve karakteristik tahmini

#### 2.6 Model Eğitimi ve Optimizasyonu
- [ ] Loss fonksiyonu tasarımı:
  - 1.5x eşik doğruluğu için özel loss
  - Multi-objective loss (kategori + değer + eşik)
  - Weighted loss (kritik bölgeler ağır ceza)
- [ ] Learning rate scheduling
- [ ] Early stopping ve checkpoint stratejisi
- [ ] Gradient clipping ve regularizasyon
- [ ] Cross-validation ile model değerlendirme

#### 2.7 Model Değerlendirme
- [ ] Başarı metrikleri hesaplama:
  - 1.5x altı/üstü doğruluk oranı
  - Kategori tahmin doğruluğu
  - RMSE, MAE, MAPE
  - Güven kalibrasyonu metrikleri
- [ ] Backtesting sistemi
- [ ] Confusion matrix ve analizi
- [ ] Mod bazlı performans değerlendirmesi
- [ ] Feature importance analizi

#### 2.8 Model Export ve Optimizasyon
- [ ] Model quantization (boyut küçültme)
- [ ] ONNX veya TensorFlow Lite dönüşümü
- [ ] Model dosyalarını Google Drive'a kaydetme
- [ ] Metadata ve versiyon bilgisi ekleme
- [ ] Inference hızı optimizasyonu

### FAZA 3: STREAMLIT ARAYÜZÜ GELİŞTİRME (Lokal)

#### 3.1 Proje Yapısı Kurulumu
```
jetxpredictor/
├── app.py                      # Ana Streamlit uygulaması
├── pages/
│   ├── 1_📊_Analiz.py         # Veri analiz sayfası
│   ├── 2_📈_Tahminler.py       # Tahmin geçmişi
│   ├── 3_⚙️_Ayarlar.py        # Ayarlar ve konfigürasyon
│   └── 4_📚_Yardım.py         # Kullanım kılavuzu
├── models/
│   ├── nbeats_model.h5        # N-BEATS modeli
│   ├── tcn_model.h5           # TCN modeli
│   ├── ensemble_model.h5      # Ensemble model
│   └── metadata.json          # Model bilgileri
├── utils/
│   ├── data_processor.py      # Veri işleme fonksiyonları
│   ├── model_loader.py        # Model yükleme
│   ├── predictor.py           # Tahmin motoru
│   ├── database.py            # SQLite işlemleri
│   ├── visualizer.py          # Grafik fonksiyonları
│   └── risk_manager.py        # Risk yönetimi
├── data/
│   ├── jetx_data.db           # Ana veritabanı
│   └── cache/                 # Geçici cache dosyaları
├── config/
│   └── config.yaml            # Konfigürasyon
├── requirements.txt
└── README.md
```

- [ ] Klasör yapısını oluşturma
- [ ] requirements.txt hazırlama
- [ ] Config dosyası oluşturma

#### 3.2 Backend Fonksiyonları

**3.2.1 Veri İşleme (data_processor.py)**
- [ ] Veri okuma ve preprocessing
- [ ] Özellik çıkarma pipeline'ı
- [ ] Real-time feature engineering
- [ ] Veri validasyon
- [ ] Cache mekanizması

**3.2.2 Model Yükleyici (model_loader.py)**
- [ ] Model dosyalarını yükleme
- [ ] Model warm-up (ilk tahmin)
- [ ] Model versiyon kontrolü
- [ ] Google Drive entegrasyonu (yeni model indirme)
- [ ] Model fallback mekanizması

**3.2.3 Tahmin Motoru (predictor.py)**
- [ ] Tahmin pipeline'ı
- [ ] Güven skoru hesaplama
- [ ] Kategori tahmini
- [ ] Eşik (1.5x altı/üstü) tahmini
- [ ] Risk analizi
- [ ] Soğuma dönemi tespiti
- [ ] Tuzak pattern uyarıları

**3.2.4 Veritabanı Yönetimi (database.py)**
- [ ] SQLite bağlantı yönetimi
- [ ] Veri okuma fonksiyonları
- [ ] Yeni veri ekleme
- [ ] Tahmin geçmişi kaydetme
- [ ] Performans metrikleri kaydetme
- [ ] Backup mekanizması

**3.2.5 Görselleştirme (visualizer.py)**
- [ ] Son N elin grafiği (Plotly)
- [ ] Kategori dağılım grafikleri
- [ ] Tahmin vs gerçek karşılaştırma
- [ ] Güven skoru göstergeleri
- [ ] Pattern görselleştirme
- [ ] Performans dashboard'u

**3.2.6 Risk Yönetimi (risk_manager.py)**
- [ ] Mod bazlı risk hesaplama (Normal/Rolling/Agresif)
- [ ] Sermaye yönetimi önerileri
- [ ] Ardışık kayıp uyarısı
- [ ] Kritik bölge uyarıları
- [ ] Oyun önerisi (oyna/bekle)

#### 3.3 Frontend Arayüzü

**3.3.1 Ana Sayfa (app.py)**
- [ ] Sidebar tasarımı:
  - Mod seçimi (Normal/Rolling/Agresif)
  - Son tahmin bilgileri
  - Performans özeti
  - Hızlı ayarlar
  
- [ ] Ana panel:
  - Tahmin kartı (büyük, belirgin)
  - Güven göstergesi (progress bar/gauge)
  - Kategori tahmini
  - 1.5x eşik tahmini (Altı/Üstü)
  - Önerilen aksiyon (OYNA/BEKLE)
  
- [ ] Grafik bölümü:
  - Son 100 elin trend grafiği
  - Real-time güncellenebilir
  - Tahmin noktası işaretleme
  
- [ ] Uyarı bölümü:
  - Risk uyarıları
  - Soğuma dönemi bildirimi
  - Tuzak pattern uyarısı
  - Kritik bölge uyarısı
  
- [ ] Veri girişi:
  - Manuel veri girişi formu
  - Otomatik veri çekme (ileride)
  - Girilen verinin otomatik kaydı

**3.3.2 Analiz Sayfası (1_📊_Analiz.py)**
- [ ] Veritabanı istatistikleri
- [ ] Kategori dağılım grafikleri
- [ ] 1.5x eşik analizi
- [ ] Pattern frekans analizi
- [ ] Büyük çarpan geçmişi
- [ ] Soğuma dönemleri görselleştirme

**3.3.3 Tahmin Geçmişi (2_📈_Tahminler.py)**
- [ ] Tahmin geçmişi tablosu
- [ ] Doğru/Yanlış filtreleme
- [ ] Mod bazında filtreleme
- [ ] Performans metrikleri:
  - Genel doğruluk oranı
  - 1.5x eşik doğruluğu
  - Kategori doğruluğu
  - Mod bazında performans
- [ ] Zaman bazlı analiz
- [ ] Export fonksiyonu (CSV/Excel)

**3.3.4 Ayarlar Sayfası (3_⚙️_Ayarlar.py)**
- [ ] Model seçimi/güncelleme
- [ ] Güven eşiği ayarları (mod bazında)
- [ ] Görselleştirme ayarları
- [ ] Uyarı ayarları (sesli/sessiz)
- [ ] Veritabanı yönetimi:
  - Backup alma
  - Veri temizleme
  - İstatistik sıfırlama
- [ ] Tema seçimi (Dark/Light)
- [ ] Dil seçimi (TR/EN)

**3.3.5 Yardım Sayfası (4_📚_Yardım.py)**
- [ ] Kullanım kılavuzu
- [ ] Model açıklaması
- [ ] Mod açıklamaları
- [ ] Risk yönetimi tavsiyeleri
- [ ] SSS (Sık Sorulan Sorular)
- [ ] Feragatname ve uyarılar

#### 3.4 Özel Özellikler

**3.4.1 Sesli Uyarı Sistemi**
- [ ] Kritik durum sesleri
- [ ] Özelleştirilebilir uyarı sesleri
- [ ] Sessiz mod desteği

**3.4.2 Real-time Güncellemeler**
- [ ] Auto-refresh mekanizması
- [ ] Websocket desteği (opsiyonel)
- [ ] Canlı grafik güncellemeleri

**3.4.3 Cache ve Performans**
- [ ] Streamlit cache kullanımı
- [ ] Model tahmin cache'i
- [ ] Veritabanı sorgu optimizasyonu

**3.4.4 Responsive Tasarım**
- [ ] Mobil uyumlu layout
- [ ] Farklı ekran çözünürlükleri desteği

### FAZA 4: ENTEGRASYON VE TEST

#### 4.1 Model-Streamlit Entegrasyonu
- [ ] Google Drive'dan model indirme scripti
- [ ] Model yükleme ve test
- [ ] Tahmin pipeline testi
- [ ] Performans benchmark

#### 4.2 Veritabanı Entegrasyonu
- [ ] Mevcut veritabanı bağlantı testi
- [ ] CRUD operasyonları testi
- [ ] Transaction yönetimi
- [ ] Hata yönetimi

#### 4.3 Kapsamlı Test

**4.3.1 Unit Tests**
- [ ] Data processor testleri
- [ ] Model loader testleri
- [ ] Predictor testleri
- [ ] Database testleri
- [ ] Visualizer testleri
- [ ] Risk manager testleri

**4.3.2 Integration Tests**
- [ ] End-to-end tahmin pipeline
- [ ] Veri girişi ve tahmin akışı
- [ ] Mod geçişleri
- [ ] Database işlemleri

**4.3.3 Performance Tests**
- [ ] Tahmin hızı (target: <1 saniye)
- [ ] Arayüz yükleme hızı
- [ ] Grafik render hızı
- [ ] Veritabanı sorgu hızı

**4.3.4 User Acceptance Tests**
- [ ] Kullanıcı senaryoları
- [ ] Hata mesajları kontrolü
- [ ] Uyarı sistemi kontrolü
- [ ] Görsel kalite kontrolü

#### 4.4 Backtesting
- [ ] Son 1000 veri ile test
- [ ] Farklı modlarda performans
- [ ] Rolling mod sermaye simülasyonu
- [ ] Worst-case senaryolar

### FAZA 5: DOKÜMANTASYON VE İYİLEŞTİRME

#### 5.1 Dokümantasyon
- [ ] README.md yazma
- [ ] Kurulum kılavuzu
- [ ] Kullanım kılavuzu
- [ ] Model dokümantasyonu
- [ ] API dokümantasyonu (internal)
- [ ] Troubleshooting kılavuzu

#### 5.2 Code Quality
- [ ] Code review
- [ ] Refactoring
- [ ] Type hints ekleme
- [ ] Docstring'ler
- [ ] Code formatting (Black/autopep8)
- [ ] Linting (pylint/flake8)

#### 5.3 Güvenlik ve Stabilite
- [ ] Input validation
- [ ] Error handling
- [ ] Logging sistemi
- [ ] Exception handling
- [ ] Data backup stratejisi

#### 5.4 Optimizasyon
- [ ] Model inference optimizasyonu
- [ ] Cache stratejisi optimizasyonu
- [ ] Veritabanı indeksleme
- [ ] Gereksiz hesaplama eliminasyonu

### FAZA 6: GELİŞMİŞ ÖZELLİKLER (Opsiyonel)

#### 6.1 Otomatik Veri Çekme
- [ ] JetX API/Web scraping araştırması
- [ ] Otomatik veri toplama botu
- [ ] Real-time veri entegrasyonu

#### 6.2 Gelişmiş Analitik
- [ ] A/B testing farklı modeller
- [ ] Multi-model ensemble
- [ ] Online learning (model güncelleme)
- [ ] Reinforcement learning entegrasyonu

#### 6.3 Bildirim Sistemi
- [ ] Email bildirimleri
- [ ] Telegram/Discord bot entegrasyonu
- [ ] Mobil push notification

#### 6.4 API Geliştirme
- [ ] REST API (FastAPI)
- [ ] API authentication
- [ ] Rate limiting
- [ ] API documentation (Swagger)

## 🎯 BAŞARI KRİTERLERİ

### Model Performans Hedefleri
- [ ] **1.5x Eşik Doğruluğu:** Minimum %75
- [ ] **Kategori Doğruluğu:** Minimum %60
- [ ] **Güven Kalibrasyonu:** %80 güven = %80 doğruluk
- [ ] **Rolling Mod:** %85+ doğruluk
- [ ] **Ardışık Yanlış Maksimum:** 5 el

### Teknik Hedefler
- [ ] **Tahmin Hızı:** <1 saniye
- [ ] **Arayüz Yükleme:** <3 saniye
- [ ] **CPU Kullanımı:** <%50 (idle)
- [ ] **RAM Kullanımı:** <2GB

### Kullanıcı Deneyimi
- [ ] **Sezgisel Arayüz:** Açıklamaya gerek kalmadan kullanılabilir
- [ ] **Anlaşılır Uyarılar:** Net ve zamanında
- [ ] **Hızlı Yanıt:** Tüm işlemler akıcı
- [ ] **Güvenilir:** Crash olmayan, stabil sistem

## 📊 BAŞARI METRİKLERİ

### Her Tahmin İçin
- Tahmin edilen değer/kategori
- Gerçek değer
- Güven skoru
- 1.5x eşik tahmini (Doğru/Yanlış)
- Kategori tahmini (Doğru/Yanlış)
- Mod (Normal/Rolling/Agresif)
- Timestamp

### Genel Metrikler
- Toplam tahmin sayısı
- Genel doğruluk oranı
- Mod bazında doğruluk
- 1.5x eşik spesifik doğruluk
- Ortalama güven skoru
- Güven-doğruluk korelasyonu

## ⚠️ KRİTİK UYARILAR

### Geliştirme Sırasında Dikkat Edilecekler
1. **1.5x Eşiği:** Tüm modellerde bu eşik merkezi öneme sahip
2. **Güven Kalibrasyonu:** %80 güven = %80 doğruluk olmalı
3. **Overfitting:** Validation set ile sürekli kontrol
4. **Data Leakage:** Gelecek verisi geçmişe sızmamalı
5. **Rolling Mod:** Yüksek güven seviyesi şart
6. **Soğuma Dönemleri:** Sabit değil, öğrenilmeli
7. **Psikolojik Pattern'ler:** Sürekli güncellenmeli

### Kullanıcıya Gösterilecek Uyarılar
1. **"Bu tahmin %100 doğru değildir"** - Her tahminle
2. **"Düşük güven, oynamayın"** - Eşik altında
3. **"Kritik bölge: 1.45-1.55x"** - Risk bölgesi
4. **"Soğuma dönemi tespit edildi"** - Bekle uyarısı
5. **"3 eldir yanlış tahmin"** - Dikkat uyarısı

## 🔄 SÜREKLI İYİLEŞTİRME

### Haftalık
- [ ] Model performans analizi
- [ ] Yanlış tahminleri inceleme
- [ ] Yeni pattern keşfi

### Aylık
- [ ] Model yeniden eğitimi (yeni verilerle)
- [ ] Hyperparameter optimizasyonu
- [ ] Feature engineering iyileştirmesi

### Gerektiğinde
- [ ] Mimari değişiklikleri
- [ ] Yeni mod ekleme
- [ ] Yeni feature'lar

## 📞 YARDIM VE DESTEK

Proje geliştirme sürecinde:
- Google Colab notebook'ları korunmalı
- Model versiyonları kayıt altında
- Performans logları düzenli tutulmalı
- Code repository (Git) kullanılmalı

---

**NOT:** Bu plan yaşayan bir dokümandır. Geliştirme sürecinde ihtiyaca göre güncellenebilir ve genişletilebilir.
