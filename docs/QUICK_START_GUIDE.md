# JetX Predictor - Hızlı Başlangıç Rehberi

## Google Colab → Lokal Kullanım Döngüsü

### Adım 1: Google Colab'da Model Eğitimi

1. **Colab Notebook'u Açın**:
   - `notebooks/jetx_PROGRESSIVE_TRAINING_MULTISCALE.py` (Neural Network)
   - `notebooks/jetx_CATBOOST_TRAINING_MULTISCALE.py` (CatBoost)

2. **Eğitimi Başlatın**:
   ```python
   # Colab'da çalıştır
   python notebooks/jetx_PROGRESSIVE_TRAINING_MULTISCALE.py
   ```

3. **Model Dosyaları Oluşturulur**:
   - Script otomatik olarak ZIP oluşturur
   - ZIP adı: `jetx_models_progressive_multiscale_v3.0.zip`

4. **ZIP'i İndirin**:
   - Colab otomatik olarak indirir
   - Veya sol panelden Files → ZIP'e sağ tık → Download

### Adım 2: Lokal Projeye Kopyalama

1. **ZIP'i Açın**:
   ```bash
   unzip jetx_models_progressive_multiscale_v3.0.zip
   ```

2. **Dosyaları Kopyalayın**:
   ```bash
   # ZIP içindeki tüm dosyaları models/ klasörüne kopyala
   cp -r progressive_multiscale/* /path/to/jetxpredictor/models/progressive_multiscale/
   ```

3. **Klasör Yapısı**:
   ```
   models/
   ├── progressive_multiscale/
   │   ├── model_window_500.h5
   │   ├── model_window_250.h5
   │   ├── model_window_100.h5
   │   ├── model_window_50.h5
   │   ├── model_window_20.h5
   │   ├── scaler_window_*.pkl
   │   └── model_info.json
   └── catboost_multiscale/
       ├── regressor_window_*.cbm
       ├── classifier_window_*.cbm
       ├── scaler_window_*.pkl
       └── model_info.json
   ```

### Adım 3: Uygulamayı Başlatma

1. **Streamlit'i Başlatın**:
   ```bash
   streamlit run app.py
   ```

2. **Model Durumunu Kontrol Edin**:
   - Sidebar'da model durumu gösterilir
   - Eksik modeller için uyarı verilir
   - "Kurulum Rehberi" butonuna tıklayarak detaylı bilgi alın

### Adım 4: Model Doğrulama

Uygulama otomatik olarak:
- ✅ Tüm model dosyalarını kontrol eder
- ✅ Eksik dosyaları tespit eder
- ✅ Model versiyonlarını gösterir
- ✅ Model durumunu sidebar'da gösterir

## Sanal Kasa Sistemi

### Sistemler

#### 1. VirtualBankrollCallback (Eğitim Sırasında)
- **Kullanım**: Model eğitimi sırasında her epoch'ta
- **Kasa 1**: 1.5x eşik sistemi (Model "1.5 üstü" dedi → 1.5x'te çıkış)
- **Kasa 2**: %70 çıkış sistemi (Model 2.0x+ dedi → Tahmin×0.70'de çıkış)
- **Metrikler**: ROI, Win Rate, Profit/Loss

#### 2. DualBankrollSystem (Test/Değerlendirme)
- **Kullanım**: Eğitilmiş modellerin performansını test etmek
- **Özellikler**: Güven skoru filtresi, dinamik kasa miktarı
- **Dosya**: `utils/dual_bankroll_system.py`

#### 3. AdvancedBankrollManager (Production)
- **Kullanım**: Gerçek kullanımda bankroll yönetimi
- **Özellikler**: 
  - Kelly Criterion (optimal bahis hesaplama)
  - Risk stratejileri (conservative, moderate, aggressive)
  - Stop-loss & Take-profit
  - Streak tracking
- **Dosya**: `utils/advanced_bankroll.py`

### Kasa Stratejileri

**Conservative (Konservatif)**:
- Max bahis: %2 bankroll
- Min güven: %75
- Stop-loss: %20

**Moderate (Dengeli)**:
- Max bahis: %5 bankroll
- Min güven: %65
- Stop-loss: %30

**Aggressive (Agresif)**:
- Max bahis: %10 bankroll
- Min güven: %50
- Stop-loss: %40

## Model Seçim Mekanizması

### Otomatik Model Tespiti

1. **Config Dosyası** (`config/config.yaml`):
   ```yaml
   model:
     path: "models/jetx_model.h5"
     scaler_path: "models/scaler.pkl"
   ```

2. **AllModelsPredictor**:
   - Uygulama başlatıldığında otomatik olarak tüm modelleri yükler
   - `models/` klasörünü tarar
   - Mevcut modelleri tespit eder

3. **Model Yükleme Sırası**:
   - Progressive NN (progressive_multiscale/)
   - CatBoost (catboost_multiscale/)
   - AutoGluon (autogluon_model/)
   - TabNet (tabnet_high_x.pkl)

### Model Seçim Stratejileri

**Tek Model**:
- Config'de belirtilen model kullanılır
- Varsayılan: `models/jetx_model.h5`

**Multi-Scale Ensemble**:
- Tüm pencere boyutlarından tahminler alınır
- Weighted averaging ile birleştirilir
- Consensus tahmin döner

**Model Versiyonlama**:
- Production modeli belirlenebilir
- Versiyon karşılaştırması yapılabilir
- Sidebar'dan yönetilebilir

## Yeni Özellikler

### Model Versiyonlama
- Model versiyonlarını kaydetme
- Production modeli belirleme
- Versiyon karşılaştırma

### A/B Testing
- İki modeli karşılaştırma
- İstatistiksel analiz
- Kazanan model belirleme

### Tüm Modellerin Çıktılarını Görme
- "Tüm Modellerin Çıktılarını Göster" checkbox'ı
- Progressive NN, CatBoost, AutoGluon, TabNet, Consensus
- Model karşılaştırma grafiği

### Model Loader
- Otomatik model tespiti
- Model doğrulama
- Kurulum rehberi

## Sorun Giderme

### Model Yüklenemiyor
1. Model dosyalarının `models/` klasöründe olduğundan emin olun
2. Dosya yollarını kontrol edin
3. Sidebar'dan "Kurulum Rehberi"ne bakın

### Eksik Model Dosyaları
1. Colab'da eğitimi tekrar çalıştırın
2. ZIP'i tekrar indirin
3. Dosyaları doğru klasöre kopyalayın

### Model Versiyonlama Çalışmıyor
1. `models/model_registry.json` dosyasının oluşturulduğundan emin olun
2. Model versiyonlarını manuel kaydedin

---

**Detaylı Bilgi**: `docs/WORKFLOW_AND_SYSTEMS.md`

