# JetX Predictor - Genel Yapı ve Model Eğitim Analizi

## 1. Uygulama Mimarisi

### Ana Uygulama Dosyaları

- **app.py**: Ana Streamlit uygulaması (Neural Network + CatBoost hybrid)
- **app_cpu_models.py**: CPU modelleri için özel Streamlit uygulaması (TabNet, AutoGluon, LightGBM, CatBoost)
- **app_5_models.py**: 5 model ensemble sistemi

### Veri Akışı

```
SQLite Database (jetx_data.db)
    ↓
DatabaseManager (utils/database.py)
    ↓
Feature Engineering (category_definitions.py - 150+ features)
    ↓
Predictor (utils/predictor.py)
    ├─ Neural Network (TensorFlow/Keras)
    ├─ CatBoost (Regressor + Classifier)
    └─ Multi-scale Ensemble
    ↓
Risk Manager (utils/risk_manager.py)
    ↓
Streamlit UI (app.py)
```

## 2. Notebook Klasöründeki Model Eğitim Dosyaları

### 2.1 Progressive Neural Network Eğitimi

**Dosya**: `notebooks/jetx_PROGRESSIVE_TRAINING_MULTISCALE.py`

**Özellikler**:

- **Multi-Scale Window Ensemble**: 5 farklı pencere boyutu (500, 250, 100, 50, 20)
- Her pencere boyutu için ayrı model eğitimi
- **Model Mimarisi**: Multi-input (features + sequences) / Multi-output (regression, classification, threshold)
- **Loss Functions**: 
  - Percentage-aware regression loss
  - Weighted binary crossentropy (para kaybı cezası: 10x, fırsat kaçırma: 1x)
- **Callbacks**:
  - `DetailedMetricsCallback`: Her epoch'ta detaylı metrikler
  - `WeightedModelCheckpoint`: Profit-focused weighted score (50% ROI, 30% Precision, 20% Win Rate)
- **Eğitim Süresi**: ~10-12 saat (GPU ile, 5 model × ~2 saat)
- **Model Kayıt**: `models/progressive_multiscale/model_window_{size}.h5`

**Veri İşleme**:

- Kronolojik sıra korunuyor (shuffle=False)
- Time-series split (70% train, 15% val, 15% test)
- Log10 transformation sequences için
- StandardScaler features için

**Model Mimarisi Detayları**:

```python
# Küçük pencere (≤50): Basit LSTM
x_seq = layers.LSTM(64, return_sequences=False)(inp_sequence)

# Orta pencere (≤100): 2-layer LSTM
x_seq = layers.LSTM(128, return_sequences=True)(inp_sequence)
x_seq = layers.LSTM(64, return_sequences=False)(x_seq)

# Büyük pencere (>100): 3-layer LSTM + Attention
x_seq = layers.LSTM(256, return_sequences=True)(inp_sequence)
x_seq = layers.LSTM(128, return_sequences=True)(x_seq)
# Attention mechanism
x_seq = layers.GlobalAveragePooling1D()(x_seq_attended)
```

### 2.2 CatBoost Multi-Scale Eğitimi

**Dosya**: `notebooks/jetx_CATBOOST_TRAINING_MULTISCALE.py`

**Özellikler**:

- Aynı multi-scale yaklaşım (5 pencere boyutu)
- **İki Model**: Regressor (değer tahmini) + Classifier (1.5x eşik)
- **GPU/CPU Otomatik Seçim**: GPU varsa GPU, yoksa CPU
- **Class Weights**: Window boyutuna göre dinamik (10-25x)
- **Eğitim Süresi**: ~40-60 dakika (5 model × ~8-12 dk)
- **Model Kayıt**: `models/catboost_multiscale/regressor_window_{size}.cbm` ve `classifier_window_{size}.cbm`

**Metrikler**:

- MAE, RMSE (regression)
- Accuracy, Below/Above 1.5 accuracy (classification)
- Money Loss Risk (False Positive Rate)
- Virtual Bankroll Simulation (2 kasa: 1.5x eşik + %80 çıkış)

**CatBoost Parametreleri**:

```python
# Regressor
CatBoostRegressor(
    iterations=1500,
    depth=10,
    learning_rate=0.03,
    l2_leaf_reg=5,
    bootstrap_type='Bernoulli',
    subsample=0.8,
    loss_function='MAE',
    task_type='GPU'  # veya 'CPU'
)

# Classifier
CatBoostClassifier(
    iterations=1500,
    depth=9,
    learning_rate=0.03,
    l2_leaf_reg=5,
    bootstrap_type='Bernoulli',
    subsample=0.8,
    loss_function='Logloss',
    auto_class_weights='Balanced',
    task_type='GPU'  # veya 'CPU'
)
```

### 2.3 CatBoost Tek Pencere Eğitimi

**Dosya**: `notebooks/jetx_CATBOOST_TRAINING.py`

**Özellikler**:

- Tek pencere boyutu (1000)
- Daha hızlı eğitim (~30-60 dakika)
- Focal Loss kullanımı (dengesiz veri için)
- Feature importance analizi
- Dual bankroll simulation

### 2.4 Diğer Eğitim Dosyaları

- `jetx_PROGRESSIVE_TRAINING.py`: Tek pencere Progressive NN
- `jetx_CATBOOST_ULTRA_TRAINING.py`: Ultra agresif CatBoost
- `jetx_model_training_ULTRA_AGGRESSIVE.py`: Ultra agresif NN
- `OPTUNA_HYPERPARAMETER_SEARCH.py`: Hyperparameter optimizasyonu
- `TRAIN_META_MODEL.py`: Meta-learner (stacking)
- `CONSENSUS_EVALUATION.py`: Consensus predictor değerlendirmesi

## 3. Eğitilen Modellerin Çalışma Şekli

### 3.1 Model Yükleme

**Dosya**: `utils/predictor.py` - `JetXPredictor` sınıfı

**Yükleme Mekanizması**:

```python
# Neural Network için
model = keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)
scaler = joblib.load(scaler_path)

# CatBoost için
regressor = CatBoostRegressor()
regressor.load_model(model_path)
classifier = CatBoostClassifier()
classifier.load_model(classifier_path)
scaler = joblib.load(scaler_path)
```

**Multi-Scale Model Yükleme**:

```python
# Progressive NN Multi-Scale
models = {}
scalers = {}
for window_size in [500, 250, 100, 50, 20]:
    model_path = f'models/progressive_multiscale/model_window_{window_size}.h5'
    scaler_path = f'models/progressive_multiscale/scaler_window_{window_size}.pkl'
    models[window_size] = keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)
    scalers[window_size] = joblib.load(scaler_path)
```

### 3.2 Tahmin Yapma Süreci

**Adım 1: Feature Extraction**

- `FeatureEngineering.extract_all_features(history)` → 150+ feature
- Multi-scale windows (50, 200, 500, 1000)
- Statistical features, volatility, streaks, patterns

**Adım 2: Model Input Hazırlama**

- Features → StandardScaler ile normalize
- Sequences → Log10 transformation
- Multi-input hazırlama (features + 4 sequence)

**Adım 3: Tahmin**

- **Neural Network**: Multi-output (regression, classification, threshold)
- **CatBoost**: Regressor (değer) + Classifier (1.5x eşik)

**Adım 4: Sonuç İşleme**

- Confidence hesaplama (model confidence + pattern confidence)
- Kategori belirleme (CRASH, SAFE, JACKPOT)
- Risk analizi (RiskManager)
- Uyarılar oluşturma

### 3.3 Ensemble Sistemleri

**Ensemble Predictor** (`utils/ensemble_predictor.py`):

- **Weighted**: CatBoost %60, NN %40
- **Unanimous**: Her iki model aynı tahminde
- **Confidence-based**: En güvenli modele öncelik
- **Majority**: Basit çoğunluk

**Multi-Scale Ensemble**:

- 5 farklı pencere boyutundan tahminler
- Weighted averaging (window boyutuna göre ağırlık)
- Final prediction = ensemble çıktısı

**Ensemble Ağırlıkları** (Progressive NN):

```python
window_weights = {
    20: 0.10,   # Kısa dönem (lokal volatilite)
    50: 0.15,   # Kısa-orta dönem
    100: 0.30,  # Orta dönem (en yüksek ağırlık)
    250: 0.25,  # Orta-uzun dönem
    500: 0.20   # Uzun dönem (genel trend)
}
```

**All Models Predictor** (`utils/all_models_predictor.py`):

- Progressive NN, CatBoost, AutoGluon, TabNet birleştirme
- Consensus voting

### 3.4 CPU Modelleri Yönetimi

**LightweightModelManager** (`utils/lightweight_model_manager.py`):

- Model factory ve registry
- Training orchestration
- Model comparison utilities
- Ensemble creation
- Model persistence

**Desteklenen Modeller**:

- LightGBM (CPU optimized)
- TabNet (attention-based)
- AutoGluon (automated ML)
- CatBoost (CPU mode)

## 4. Model Eğitim Pipeline Özeti

### Progressive NN Multi-Scale

1. Veri yükleme (SQLite, kronolojik sıra)
2. Time-series split (70/15/15)
3. Her pencere boyutu için:
   - Feature extraction
   - Sequence hazırlama
   - Model oluşturma (pencere boyutuna göre adapte)
   - Eğitim (150 epochs, early stopping)
   - Weighted checkpoint (profit-focused)
4. Ensemble değerlendirme
5. Model kaydetme (H5 + PKL + JSON)

**Eğitim Detayları**:

```python
# Model compile
model.compile(
    optimizer=Adam(0.0001),
    loss={
        'regression': percentage_aware_regression_loss,
        'classification': 'categorical_crossentropy',
        'threshold': create_weighted_binary_crossentropy(w0=10.0, w1=1.0)
    },
    loss_weights={
        'regression': 0.50,
        'classification': 0.25,
        'threshold': 0.25
    }
)

# Eğitim
model.fit(
    [X_features, X_sequences],
    {
        'regression': y_regression,
        'classification': y_classification,
        'threshold': y_threshold
    },
    epochs=150,
    batch_size=32,
    shuffle=False,  # KRİTİK: Kronolojik sıra korunmalı
    callbacks=[...]
)
```

### CatBoost Multi-Scale

1. Veri yükleme
2. Time-series split
3. Her pencere boyutu için:
   - Feature extraction
   - Regressor eğitimi (1500 iterations)
   - Classifier eğitimi (1500 iterations, class weights)
4. Ensemble değerlendirme
5. Model kaydetme (CBM + PKL + JSON)

**Eğitim Detayları**:

```python
# Regressor
regressor = CatBoostRegressor(
    iterations=1500,
    depth=10,
    learning_rate=0.03,
    loss_function='MAE',
    task_type='GPU'  # veya 'CPU'
)
regressor.fit(X_train, y_reg_train, eval_set=(X_val, y_reg_val))

# Classifier
classifier = CatBoostClassifier(
    iterations=1500,
    depth=9,
    learning_rate=0.03,
    loss_function='Logloss',
    auto_class_weights='Balanced',
    task_type='GPU'  # veya 'CPU'
)
classifier.fit(X_train, y_cls_train, eval_set=(X_val, y_cls_val))
```

## 5. Önemli Notlar

### Veri Bütünlüğü

- **Shuffle YASAK**: Kronolojik sıra korunmalı
- **Augmentation YASAK**: Gerçek zaman serisi yapısı korunmalı
- **Time-series split**: Train/Val/Test kronolojik olarak ayrılmalı

**Neden Önemli?**:

- Zaman serisi verilerinde gelecek bilgisi geçmişe sızarsa (data leakage) model gerçekçi olmayan performans gösterir
- Shuffle yapılırsa model gelecekteki desenleri öğrenir (impossible in production)
- Augmentation yapılırsa gerçek zaman serisi yapısı bozulur

### Model Seçim Kriteri

- **Eski Metrikler**: Balanced Accuracy, F1 Score (yanıltıcı)
- **Yeni Metrik**: Profit-Focused Weighted Score
  - 50% ROI (para kazandırma)
  - 30% Precision (1.5 üstü dediğinde ne kadar haklı)
  - 20% Win Rate (kazanan tahmin oranı)

**Neden Değişti?**:

- Balanced Accuracy: Model hep "1.5 üstü" dediğinde yüksek çıkıyordu (yanıltıcı)
- F1 Score: Dengesiz veride işe yaramıyordu
- ROI odaklı metrik: Para kazandırmayan model işe yaramaz

### Risk Yönetimi

- **Rolling Mode**: %80+ güven (konservatif)
- **Normal Mode**: %65+ güven (dengeli)
- **Aggressive Mode**: %50+ güven (riskli)

### Model Dosya Yapısı

```
models/
├── progressive_multiscale/
│   ├── model_window_500.h5
│   ├── model_window_250.h5
│   ├── model_window_100.h5
│   ├── model_window_50.h5
│   ├── model_window_20.h5
│   ├── scaler_window_500.pkl
│   ├── scaler_window_250.pkl
│   ├── scaler_window_100.pkl
│   ├── scaler_window_50.pkl
│   ├── scaler_window_20.pkl
│   └── model_info.json
├── catboost_multiscale/
│   ├── regressor_window_500.cbm
│   ├── regressor_window_250.cbm
│   ├── regressor_window_100.cbm
│   ├── regressor_window_50.cbm
│   ├── regressor_window_20.cbm
│   ├── classifier_window_500.cbm
│   ├── classifier_window_250.cbm
│   ├── classifier_window_100.cbm
│   ├── classifier_window_50.cbm
│   ├── classifier_window_20.cbm
│   ├── scaler_window_500.pkl
│   ├── scaler_window_250.pkl
│   ├── scaler_window_100.pkl
│   ├── scaler_window_50.pkl
│   ├── scaler_window_20.pkl
│   └── model_info.json
├── catboost_regressor.cbm
├── catboost_classifier.cbm
├── catboost_scaler.pkl
└── cpu/
    └── (CPU modelleri)
```

## 6. Multi-Scale Ensemble Çalışma Prensibi

### Pencere Boyutlarının Anlamı

- **500**: Uzun dönem trend (genel piyasa davranışı)
- **250**: Orta-uzun dönem (sezonluk desenler)
- **100**: Orta dönem (aylık desenler)
- **50**: Kısa-orta dönem (haftalık desenler)
- **20**: Kısa dönem (günlük volatilite)

### Ensemble Stratejisi

```python
# Her pencere boyutu için tahmin
predictions = {}
for window_size in [500, 250, 100, 50, 20]:
    model = models[window_size]
    features = extract_features(history, window_size)
    sequence = history[-window_size:]
    pred = model.predict([features, sequence])
    predictions[window_size] = pred

# Weighted ensemble
final_prediction = sum(
    weights[ws] * predictions[ws] 
    for ws in window_sizes
)
```

### Avantajları

1. **Farklı zaman ölçeklerinde desen yakalama**: Hem mikro hem makro desenler
2. **Robustness**: Bir pencere yanılsa diğerleri telafi eder
3. **Adaptive**: Farklı piyasa koşullarında farklı pencereler öne çıkar

## 7. Model Performans Metrikleri

### Eğitim Metrikleri

- **MAE** (Mean Absolute Error): Ortalama mutlak hata
- **RMSE** (Root Mean Square Error): Kök ortalama kare hata
- **Threshold Accuracy**: 1.5x eşik doğruluğu
- **Below 1.5 Accuracy**: 1.5 altı tahmin doğruluğu (para kaybı önleme)
- **Above 1.5 Accuracy**: 1.5 üstü tahmin doğruluğu (fırsat yakalama)
- **Money Loss Risk**: False Positive Rate (para kaybı riski)
- **ROI**: Return on Investment (sanal kasa simülasyonu)

### Production Metrikleri

- **Win Rate**: Kazanan tahmin oranı
- **Sharpe Ratio**: Risk-ayarlı getiri
- **Maximum Drawdown**: Maksimum düşüş
- **Profit Factor**: Kazanç/Zarar oranı

## 8. Öneriler ve İyileştirme Noktaları

1. **Model Versiyonlama**: Model kayıtlarında versiyon kontrolü
2. **Otomatik Model Seçimi**: En iyi performans gösteren modeli otomatik seçme
3. **Model Monitoring**: Production'da model drift tespiti
4. **A/B Testing**: Farklı modelleri canlıda test etme
5. **Model Compression**: Model boyutlarını küçültme (quantization, pruning)
6. **Incremental Learning**: Yeni verilerle model güncelleme
7. **Feature Selection**: En önemli feature'ları otomatik seçme
8. **Hyperparameter Auto-tuning**: Optuna ile otomatik hyperparameter optimizasyonu

## 9. Kullanım Örnekleri

### Model Eğitimi (Google Colab)

```python
# Progressive NN Multi-Scale
python notebooks/jetx_PROGRESSIVE_TRAINING_MULTISCALE.py

# CatBoost Multi-Scale
python notebooks/jetx_CATBOOST_TRAINING_MULTISCALE.py
```

### Model Yükleme ve Tahmin

```python
from utils.predictor import JetXPredictor

# Neural Network
predictor = JetXPredictor(
    model_path="models/jetx_model.h5",
    scaler_path="models/scaler.pkl",
    model_type='neural_network'
)

# CatBoost
predictor = JetXPredictor(
    model_path="models/catboost_regressor.cbm",
    scaler_path="models/catboost_scaler.pkl",
    model_type='catboost'
)

# Tahmin
history = [1.2, 1.5, 2.1, 1.8, ...]  # Son 1000+ değer
prediction = predictor.predict(history, mode='normal')
```

### Multi-Scale Ensemble Kullanımı

```python
from utils.consensus_predictor import ConsensusPredictor

predictor = ConsensusPredictor()
predictor.load_nn_models()
predictor.load_catboost_models()

history = np.array([1.2, 1.5, 2.1, ...])
result = predictor.predict_consensus(history)
```

## 10. Sorun Giderme

### Model Yüklenemiyor

- Model dosyalarının `models/` klasöründe olduğundan emin olun
- Custom objects'lerin yüklü olduğundan emin olun (`CUSTOM_OBJECTS`)
- Model versiyonunun uyumlu olduğundan emin olun

### Tahmin Hataları

- En az 1000 geçmiş veri gerekli
- Veri formatının doğru olduğundan emin olun (float listesi)
- Feature extraction'ın başarılı olduğundan emin olun

### Eğitim Hataları

- GPU bellek yetersizliği: Batch size'ı küçültün
- Veri yetersizliği: En az 5000 veri gerekli
- Kronolojik sıra: Shuffle=False olduğundan emin olun

---

**Son Güncelleme**: 2025-01-XX  
**Versiyon**: 1.0

