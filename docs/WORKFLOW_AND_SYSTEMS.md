# JetX Predictor - İş Akışı ve Sistemler Dokümantasyonu

## 1. Google Colab → Lokal Kullanım Döngüsü

### Mevcut İş Akışı

```
1. Google Colab'da Model Eğitimi (GPU)
   ↓
2. Model Dosyalarını ZIP'leme
   ↓
3. ZIP'i İndirme
   ↓
4. Lokal Projeye Kopyalama
   ↓
5. Streamlit Uygulamasında Kullanım
```

### Model Eğitim Dosyaları (Colab)

#### Progressive NN Multi-Scale
**Dosya**: `notebooks/jetx_PROGRESSIVE_TRAINING_MULTISCALE.py`

**Eğitim Süreci**:
1. Veri yükleme (SQLite'dan)
2. Time-series split (70/15/15)
3. Her pencere boyutu için model eğitimi (500, 250, 100, 50, 20)
4. Model kaydetme (H5 + PKL)
5. ZIP oluşturma

**Oluşturulan Dosyalar**:
```
models/progressive_multiscale/
├── model_window_500.h5
├── model_window_250.h5
├── model_window_100.h5
├── model_window_50.h5
├── model_window_20.h5
├── scaler_window_500.pkl
├── scaler_window_250.pkl
├── scaler_window_100.pkl
├── scaler_window_50.pkl
├── scaler_window_20.pkl
└── model_info.json
```

**ZIP İndirme**:
- Script otomatik olarak ZIP oluşturur
- Google Colab'da `files.download()` ile indirilir
- ZIP adı: `jetx_models_progressive_multiscale_v3.0.zip`

#### CatBoost Multi-Scale
**Dosya**: `notebooks/jetx_CATBOOST_TRAINING_MULTISCALE.py`

**Eğitim Süreci**:
1. Veri yükleme
2. Time-series split
3. Her pencere boyutu için:
   - Regressor eğitimi (1500 iterations)
   - Classifier eğitimi (1500 iterations)
4. Model kaydetme (CBM + PKL)
5. ZIP oluşturma

**Oluşturulan Dosyalar**:
```
models/catboost_multiscale/
├── regressor_window_500.cbm
├── regressor_window_250.cbm
├── regressor_window_100.cbm
├── regressor_window_50.cbm
├── regressor_window_20.cbm
├── classifier_window_500.cbm
├── classifier_window_250.cbm
├── classifier_window_100.cbm
├── classifier_window_50.cbm
├── classifier_window_20.cbm
├── scaler_window_500.pkl
├── scaler_window_250.pkl
├── scaler_window_100.pkl
├── scaler_window_50.pkl
├── scaler_window_20.pkl
└── model_info.json
```

**ZIP İndirme**:
- ZIP adı: `jetx_models_catboost_multiscale_v3.0.zip`

### Lokal Model Yükleme Mekanizması

#### Otomatik Model Tespiti

**AllModelsPredictor** (`utils/all_models_predictor.py`):
- Uygulama başlatıldığında otomatik olarak `models/` klasörünü tarar
- Mevcut modelleri tespit eder ve yükler
- Eksik modeller için uyarı verir

**Model Yükleme Sırası**:
1. Progressive NN modelleri (`models/progressive_multiscale/`)
2. CatBoost modelleri (`models/catboost_multiscale/`)
3. AutoGluon modeli (`models/autogluon_model/`)
4. TabNet modeli (`models/tabnet_high_x.pkl`)

**Yükleme Kontrolü**:
```python
# Her model için dosya varlık kontrolü
if os.path.exists(model_path) and os.path.exists(scaler_path):
    # Model yükle
    model = keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)
    scaler = joblib.load(scaler_path)
    # Başarılı
else:
    # Model eksik - atla
    logger.warning(f"Model bulunamadı: {model_path}")
```

#### Model Seçim Mekanizması

**Ana Predictor** (`utils/predictor.py` - `JetXPredictor`):
- Varsayılan: `models/jetx_model.h5` (tek model)
- CatBoost: `models/catboost_regressor.cbm` + `models/catboost_classifier.cbm`
- Config dosyasından model yolu alınır: `config/config.yaml`

**Config Yapısı**:
```yaml
model:
  path: "models/jetx_model.h5"
  scaler_path: "models/scaler.pkl"
```

**Model Tipi Seçimi**:
- `model_type='neural_network'`: TensorFlow/Keras modeli
- `model_type='catboost'`: CatBoost modelleri

**Multi-Scale Model Seçimi**:
- `AllModelsPredictor`: Tüm multi-scale modelleri otomatik yükler
- Her pencere boyutu için ayrı model
- Ensemble: Tüm pencere boyutlarının ağırlıklı ortalaması

**Model Versiyonlama** (Yeni):
- `ModelVersionManager`: Model versiyonlarını yönetir
- Production modeli belirleme
- Versiyon karşılaştırma

### İyileştirme Önerileri

1. **Otomatik Model Kontrolü**:
   - Uygulama başlatıldığında tüm modelleri kontrol et
   - Eksik modeller için kullanıcıya bilgi ver
   - Model durumunu sidebar'da göster

2. **Model Versiyonlama Entegrasyonu**:
   - Colab'da eğitim sonrası otomatik versiyon kaydı
   - Lokal'de versiyon kontrolü
   - Production modeli otomatik seçimi

3. **Model İndirme Yardımcısı**:
   - ZIP dosyalarını otomatik açma
   - Model dosyalarını doğru klasöre kopyalama
   - Model doğrulama (dosya bütünlüğü kontrolü)

## 2. Sanal Kasa Sistemi

### Sistemlerin Genel Bakışı

Uygulamada **3 farklı sanal kasa sistemi** bulunmaktadır:

1. **VirtualBankrollCallback** - Eğitim sırasında kullanım
2. **DualBankrollSystem** - Test/değerlendirme için
3. **AdvancedBankrollManager** - Production kullanımı için

### 2.1 VirtualBankrollCallback (Eğitim Sırasında)

**Dosya**: `utils/virtual_bankroll_callback.py`

**Kullanım Amacı**: Model eğitimi sırasında her epoch'ta performans ölçümü

**İki Kasa Sistemi**:

#### Kasa 1: 1.5x Eşik Sistemi
- **Strateji**: Model "1.5x üstü" tahmin ederse → 1.5x'te çıkış
- **Bahis**: 10 TL (sabit)
- **Kazanç**: 1.5 × 10 = 15 TL
- **Başabaş Noktası**: %66.7 kazanma oranı gerekli

**Simülasyon Mantığı**:
```python
for her tahmin:
    if model_pred == 1:  # 1.5 üstü dedi
        wallet -= 10  # Bahis yap
        if actual_value >= 1.5:
            wallet += 15  # Kazandık
        else:
            # Kaybettik (bahis zaten kesildi)
```

#### Kasa 2: %70 Çıkış Sistemi
- **Strateji**: Model 2.0x+ tahmin ederse → Tahmin × 0.70'de çıkış
- **Bahis**: 10 TL (sabit)
- **Kazanç**: exit_point × 10 TL
- **Örnek**: Tahmin 3.0x → 2.1x'te çıkış → 21 TL kazanç

**Simülasyon Mantığı**:
```python
for her tahmin:
    if model_pred_value >= 2.0:
        wallet -= 10  # Bahis yap
        exit_point = model_pred_value * 0.70
        if actual_value >= exit_point:
            wallet += exit_point * 10  # Kazandık
        else:
            # Kaybettik
```

**Metrikler**:
- ROI (Return on Investment)
- Win Rate (Kazanma Oranı)
- Total Bets (Toplam Bahis)
- Profit/Loss (Net Kar/Zarar)

### 2.2 DualBankrollSystem (Test/Değerlendirme)

**Dosya**: `utils/dual_bankroll_system.py`

**Kullanım Amacı**: Eğitilmiş modellerin performansını test etmek

**Özellikler**:
- Güven skoru filtresi (opsiyonel)
- Dinamik kasa miktarı (test veri sayısı × bahis tutarı)
- Detaylı raporlama

**Kasa 1: 1.5x Eşik (Güven Filtreli)**
```python
# Güven skoru filtresi ile
if confidence >= confidence_threshold:  # Örn: %90
    if model_pred == 1:  # 1.5 üstü
        # Bahis yap
```

**Kasa 2: %70 Çıkış (Güven Filtreli)**
```python
# Sadece 2.0x+ tahminlerde
if model_pred_value >= 2.0:
    if confidence >= confidence_threshold:
        exit_point = model_pred_value * 0.70
        # Bahis yap
```

**Kullanım Örneği**:
```python
from utils.dual_bankroll_system import DualBankrollSystem

system = DualBankrollSystem(
    test_predictions=predictions,
    actual_values=actuals,
    bet_amount=10.0,
    confidence_scores=confidences  # Opsiyonel
)

# Kasa 1: %90 güven filtresi ile
kasa1 = system.run_kasa1_simulation(
    threshold_predictions=threshold_preds,
    confidence_threshold=0.90
)

# Kasa 2: %70 çıkış, %90 güven filtresi ile
kasa2 = system.run_kasa2_simulation(
    exit_multiplier=0.70,
    confidence_threshold=0.90
)

# Detaylı rapor
system.print_detailed_report()
```

### 2.3 AdvancedBankrollManager (Production)

**Dosya**: `utils/advanced_bankroll.py`

**Kullanım Amacı**: Gerçek kullanımda bankroll yönetimi

**Özellikler**:
- Kelly Criterion (optimal bahis hesaplama)
- Risk tolerance seviyeleri (conservative, moderate, aggressive)
- Stop-loss & Take-profit otomasyonu
- Streak tracking

**Risk Stratejileri**:

#### Conservative (Konservatif)
- Max bahis: %2 bankroll
- Kelly fraction: 1/4 (çok güvenli)
- Stop-loss: %20 kayıp
- Take-profit: %50 kar
- Min güven: %75

#### Moderate (Dengeli)
- Max bahis: %5 bankroll
- Kelly fraction: 1/2 (dengeli)
- Stop-loss: %30 kayıp
- Take-profit: %100 kar
- Min güven: %65

#### Aggressive (Agresif)
- Max bahis: %10 bankroll
- Kelly fraction: Full Kelly (riskli)
- Stop-loss: %40 kayıp
- Take-profit: %200 kar
- Min güven: %50

**Kelly Criterion Formülü**:
```
f = (p * b - q) / b
- f: Bahis oranı (bankroll'un kaçta kaçı)
- p: Kazanma olasılığı (confidence)
- b: Kazanç oranı (1.5x için 0.5)
- q: Kaybetme olasılığı (1 - p)
```

**Kullanım Örneği**:
```python
from utils.advanced_bankroll import AdvancedBankrollManager

manager = AdvancedBankrollManager(
    initial_bankroll=1000.0,
    risk_tolerance='moderate'
)

# Optimal bahis hesapla
bet_size = manager.calculate_bet_size(
    confidence=0.75,
    predicted_value=2.0
)

# Bahis yap
result = manager.place_bet(
    bet_size=bet_size,
    predicted_value=2.0,
    actual_value=2.5,
    confidence=0.75
)

# Stop-loss kontrolü
should_stop, reason = manager.should_stop()
if should_stop:
    print(f"Durdur: {reason}")

# Rapor
report = manager.get_report()
```

### Sanal Kasa Sistemlerinin Karşılaştırması

| Özellik | VirtualBankrollCallback | DualBankrollSystem | AdvancedBankrollManager |
|---------|------------------------|-------------------|------------------------|
| **Kullanım** | Eğitim sırasında | Test/değerlendirme | Production |
| **Bahis Hesaplama** | Sabit (10 TL) | Sabit (ayarlanabilir) | Kelly Criterion |
| **Güven Filtresi** | ❌ | ✅ | ✅ |
| **Stop-Loss** | ❌ | ❌ | ✅ |
| **Take-Profit** | ❌ | ❌ | ✅ |
| **Streak Tracking** | ❌ | ❌ | ✅ |
| **Risk Stratejileri** | ❌ | ❌ | ✅ |

## 3. Model Seçim Mekanizması

### 3.1 Otomatik Model Tespiti

**Uygulama Başlatıldığında**:

1. **Config Dosyası Kontrolü** (`config/config.yaml`):
   ```yaml
   model:
     path: "models/jetx_model.h5"
     scaler_path: "models/scaler.pkl"
   ```

2. **Model Dosyası Kontrolü**:
   ```python
   if os.path.exists(model_path):
       predictor = JetXPredictor(model_path, scaler_path)
   else:
       # Model eksik - uyarı göster
   ```

3. **AllModelsPredictor Otomatik Yükleme**:
   ```python
   all_models_predictor = AllModelsPredictor()
   all_models_predictor.load_all_models()
   # Tüm mevcut modelleri yükler
   ```

### 3.2 Model Yükleme Önceliği

**Sıralama**:
1. **Config'de belirtilen model** (varsayılan)
2. **Multi-scale modeller** (progressive_multiscale, catboost_multiscale)
3. **CPU modelleri** (lightgbm, tabnet, autogluon)
4. **Eksik modeller atlanır** (uyarı verilir)

### 3.3 Model Seçim Stratejileri

#### Tek Model Kullanımı
```python
# Varsayılan model
predictor = JetXPredictor(
    model_path="models/jetx_model.h5",
    scaler_path="models/scaler.pkl"
)
```

#### Multi-Scale Ensemble
```python
# Tüm pencere boyutlarını kullan
all_models_predictor = AllModelsPredictor()
all_models_predictor.load_all_models()
predictions = all_models_predictor.predict_all(history)
# Consensus tahmin döner
```

#### Model Versiyonlama ile Seçim
```python
# Production modelini al
version_manager = get_version_manager()
production_model = version_manager.get_production_model('progressive_nn')
# Production model bilgilerini kullan
```

### 3.4 Model Seçim İyileştirmeleri

**Önerilen İyileştirmeler**:

1. **Otomatik En İyi Model Seçimi**:
   - Tüm modellerin performans metriklerini karşılaştır
   - En yüksek ROI'ye sahip modeli otomatik seç
   - Model versiyonlama ile entegre et

2. **Model Durum Göstergesi**:
   - Sidebar'da model durumunu göster
   - Hangi modellerin yüklü olduğunu listele
   - Model performans metriklerini göster

3. **Dinamik Model Seçimi**:
   - Kullanıcı model seçebilsin
   - A/B testi ile model karşılaştırması
   - Otomatik model değiştirme (performans düşerse)

## 4. İyileştirme Planı

### 4.1 Colab → Lokal Döngüsü İyileştirmeleri

1. **Model Doğrulama Scripti**:
   - ZIP açıldıktan sonra model dosyalarını kontrol et
   - Dosya bütünlüğü kontrolü
   - Model uyumluluk kontrolü

2. **Otomatik Model Kurulumu**:
   - ZIP'i otomatik açma
   - Dosyaları doğru klasöre kopyalama
   - Model versiyonunu otomatik kaydetme

3. **Model Durum Raporu**:
   - Hangi modellerin yüklü olduğunu göster
   - Model versiyonlarını listele
   - Eksik modeller için rehber

### 4.2 Sanal Kasa Sistemi İyileştirmeleri

1. **Gerçek Zamanlı Kasa Takibi**:
   - Streamlit'te canlı kasa durumu
   - Grafik ile kasa değişimi
   - Uyarılar (stop-loss, take-profit)

2. **Kasa Stratejisi Seçimi**:
   - Kullanıcı risk tolerance seçebilsin
   - Strateji karşılaştırması
   - Otomatik strateji önerisi

3. **Detaylı İstatistikler**:
   - Kasa geçmişi
   - Performans grafikleri
   - ROI trend analizi

### 4.3 Model Seçim İyileştirmeleri

1. **Akıllı Model Seçimi**:
   - Geçmiş performansa göre otomatik seçim
   - Model drift tespiti
   - Otomatik model değiştirme

2. **Model Karşılaştırma Dashboard**:
   - Tüm modellerin performansını göster
   - Karşılaştırma grafikleri
   - En iyi model önerisi

3. **A/B Test Entegrasyonu**:
   - İki modeli karşılaştır
   - İstatistiksel analiz
   - Kazanan modeli otomatik seç

---

**Son Güncelleme**: 2025-01-XX  
**Versiyon**: 1.0

