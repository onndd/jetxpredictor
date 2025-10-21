# JetX Predictor

## Proje Mimarisi

### Ana Uygulama Yapısı

```
jetxpredictor/
├── app.py                          # Streamlit ana UI
├── category_definitions.py         # Kategori tanımları ve feature extraction
├── requirements.txt                # Python dependencies
│
├── config/
│   ├── config.yaml                # Ana konfigürasyon
│   └── cpu_models_config.yaml     # CPU model ayarları
│
├── data/
│   ├── jetx_data.db              # SQLite veritabanı
│   └── cache/                     # Önbellek dizini
│
├── models/                         # Eğitilmiş modeller
│   ├── progressive_multiscale/    # Multi-scale NN modelleri
│   └── cpu/                       # CPU ağırlıklı modeller
│
├── notebooks/                      # Google Colab eğitim scriptleri
│   ├── JetX_PROGRESSIVE_TRAINING_Colab.ipynb
│   ├── jetx_PROGRESSIVE_TRAINING_MULTISCALE.py
│   ├── jetx_CATBOOST_TRAINING_MULTISCALE.py
│   ├── OPTUNA_HYPERPARAMETER_SEARCH.py
│   └── TRAIN_META_MODEL.py
│
├── pages/                          # Streamlit sayfaları
│   ├── 1_📊_Analiz.py
│   ├── 2_🔬_Model_Karsilastirma.py
│   └── cpu/                       # CPU model sayfaları
│
└── utils/                          # Core utility modülleri
    ├── predictor.py               # Ana tahmin motoru
    ├── ensemble_predictor.py      # Ensemble sistemi
    ├── risk_manager.py            # Risk yönetimi
    ├── database.py                # Veritabanı yönetimi
    └── ...                        # Diğer utility modülleri
```

## Veritabanı Şeması

### jetx_results Tablosu
```sql
CREATE TABLE jetx_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    value REAL NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### predictions Tablosu
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

## Core Utils Modülleri

### Tahmin Motorları
- **predictor.py** - Ana tahmin motoru (Neural Network + CatBoost hybrid)
- **ensemble_predictor.py** - Multi-model ensemble sistemi (weighted, unanimous, confidence, majority voting)
- **consensus_predictor.py** - Consensus based prediction
- **all_models_predictor.py** - Tüm model tahminlerini birleştirme

### Model Öğrenimi
- **cpu_training_engine.py** - CPU üzerinde model eğitim motoru
- **lightweight_model_manager.py** - Hafif model yönetim sistemi
- **autogluon_predictor.py** - AutoGluon AutoML entegrasyonu
- **tabnet_predictor.py** - TabNet attention-based predictor
- **lightgbm_predictor.py** - LightGBM gradient boosting
- **catboost_ensemble.py** - CatBoost ensemble wrapper

### Feature Engineering
- **multi_scale_window.py** - Multi-scale window extraction (500, 250, 100, 50, 20 pencere boyutları)
- **data_augmentation.py** - Sequence ve feature augmentation
- **balanced_batch_generator.py** - Class-balanced batch generation

### Custom Loss Functions
- **custom_losses.py** - Percentage-aware regression, threshold killer, balanced focal loss
- **focal_loss.py** - CatBoost ve Keras focal loss implementasyonları
- **adaptive_threshold.py** - Dinamik threshold yönetimi (confidence, performance, hybrid)

### Risk & Bankroll Yönetimi
- **risk_manager.py** - Risk analizi ve karar verme (rolling, normal, aggressive modları)
- **advanced_bankroll.py** - Kelly criterion, bet sizing, stop-loss
- **dual_bankroll_system.py** - İki ayrı kasa simülasyonu (1.5x threshold & %80 exit)

### Backtesting & Monitoring
- **backtesting.py** - Historical performance testing (fixed, kelly, confidence-based stratejiler)
- **ensemble_monitor.py** - Ensemble performans izleme ve logging
- **psychological_analyzer.py** - Pattern analizi (bait-and-switch, heating/cooling detection)
- **anomaly_streak_detector.py** - Streak ve anomali tespiti

### Deep Learning Components
- **attention_layers.py** - Positional encoding, multi-head attention, temporal attention, transformer encoder
- **lr_schedulers.py** - Cosine annealing, one-cycle, exponential/polynomial decay schedulers
- **adaptive_weight_scheduler.py** - Dinamik loss weight ayarlama (Keras callback)
- **virtual_bankroll_callback.py** - Training sırasında sanal kasa simülasyonu

### Database & Configuration
- **database.py** - SQLite veritabanı yönetimi (CRUD, stats, backup)
- **database_setup.py** - Veritabanı kurulum ve initialization
- **config_loader.py** - YAML konfigürasyon yükleme (singleton pattern)
- **gpu_config.py** - TensorFlow ve CatBoost GPU konfigürasyonu

### Ensemble & Stacking
- **ensemble_manager.py** - Stacking ensemble (meta-learner)
- **consensus_predictor.py** - NN + CatBoost consensus voting

## Model Pipeline Akışı

```
Veri Girişi (database.py)
    ↓
SQLite Database (jetx_results table)
    ↓
Feature Extraction (category_definitions.py)
    ├─ 150+ statistical features
    ├─ Multi-scale windows
    ├─ Wavelet & Fourier transforms
    └─ Psychological patterns
    ↓
Tahmin Motorları
    ├─ Progressive NN (multi-input/multi-output)
    ├─ CatBoost (regressor + classifier)
    ├─ AutoGluon (50+ model ensemble)
    └─ TabNet (attention mechanism)
    ↓
Ensemble Predictor (ensemble_predictor.py)
    ├─ Weighted voting (CatBoost 60%, NN 40%)
    ├─ Unanimous strategy
    ├─ Confidence-based
    └─ Majority voting
    ↓
Risk Analizi (risk_manager.py)
    ├─ Confidence threshold check
    ├─ Consecutive loss tracking
    ├─ Warning level assessment
    └─ Betting suggestion
    ↓
Adaptive Threshold (adaptive_threshold.py)
    ├─ Confidence-based (0.90+ → 1.5x)
    ├─ Performance-based (win rate)
    └─ Hybrid (combined approach)
    ↓
Streamlit UI (app.py)
    ├─ Prediction visualization
    ├─ Real-time charts
    ├─ Backtesting interface
    └─ Performance metrics
```

## Multi-Scale Architecture

### Window Sizes
```python
window_sizes = [500, 250, 100, 50, 20]
```

### Her Window İçin
- Ayrı model eğitimi
- LSTM derinliği window size'a göre adapte
- Attention mechanism (büyük windowlar için)
- Time-series split validation
- Kronolojik sıra korunması (shuffle=False)

### Ensemble Stratejisi
- Her model tahmin yapar
- Weighted averaging (basit ortalama veya ağırlıklı)
- Final prediction = ensemble çıktısı

## Feature Engineering Pipeline

### Temel İstatistikler
- Hareketli ortalamalar (5 farklı pencere)
- Min, max, median, std, variance
- Percentiles (25th, 75th, 90th)

### Threshold Özellikleri
- 1.5x altı/üstü oranları
- Son N elde kritik bölge frekansı
- Threshold'dan uzaklık

### Distance Features
- 10x, 20x, 50x, 100x, 200x'ten son geçiş mesafesi

### Streak Features
- Ardışık yükseliş/düşüş
- Maksimum streak uzunluğu
- Pattern tekrarı

### Volatility
- Multi-timeframe std
- Range, coefficient of variation
- Bollinger bands

### Advanced Analysis
- **Wavelet Transform**: Time-frequency localization
- **Fourier Transform**: Periyodik pattern detection
- **Autocorrelation**: Lag-based patterns
- **DFA (Detrended Fluctuation Analysis)**
- **Hurst Exponent**: Trend persistence

### Psychological Patterns
- Bait-and-switch detection
- False confidence patterns
- Heating up / Cooling down
- Gambler's fallacy score
- Manipulation detection

## Model Mimarileri

### Progressive Neural Network

```python
# Multi-input
inputs = {
    'features': Input(150+),        # Dense features
    'seq_50': Input(50, 1),         # LSTM 50 timesteps
    'seq_200': Input(200, 1),       # LSTM 200 timesteps
    'seq_500': Input(500, 1),       # LSTM 500 timesteps
    'seq_1000': Input(1000, 1)      # LSTM 1000 timesteps
}

# Multi-output
outputs = {
    'regression': Dense(1),          # Değer tahmini
    'classification': Dense(3),      # 3-class kategorization
    'threshold': Dense(1, sigmoid)   # 1.5x binary prediction
}

# Loss weights
loss_weights = {
    'regression': 0.50,
    'classification': 0.15,
    'threshold': 0.35
}
```

### CatBoost Models

#### Regressor
```python
CatBoostRegressor(
    iterations=1500,
    depth=10,
    learning_rate=0.03,
    loss_function='MAE',
    task_type='GPU'
)
```

#### Classifier
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

### AutoGluon
```python
TabularPredictor.fit(
    time_limit=3600,
    presets='best_quality',
    eval_metric='roc_auc',
    # 50+ model ensemble (LightGBM, CatBoost, XGBoost, Neural Network, etc.)
)
```

### TabNet
```python
TabNetClassifier(
    n_d=64,
    n_a=64,
    n_steps=5,
    gamma=1.5,
    n_independent=2,
    n_shared=2,
    # Attention mechanism ile yüksek çarpan tespiti
)
```

## Training Pipeline (Google Colab)

### 1. Progressive NN Training
```
notebooks/jetx_PROGRESSIVE_TRAINING_MULTISCALE.py
├─ Multi-scale window extraction
├─ Feature engineering (150+ features)
├─ 5 ayrı model eğitimi (her window size için)
├─ Custom callbacks
│   ├─ DetailedMetricsCallback (below/above accuracy)
│   ├─ WeightedModelCheckpoint (50% below, 40% above, 10% ROI)
│   └─ VirtualBankrollCallback
├─ Early stopping (patience=20)
└─ Model kaydetme (H5 + PKL)
```

### 2. CatBoost Training
```
notebooks/jetx_CATBOOST_TRAINING_MULTISCALE.py
├─ Regressor training (değer tahmini)
├─ Classifier training (1.5x threshold)
├─ Auto class weighting
├─ GPU acceleration
└─ Model kaydetme (CBM + PKL)
```

### 3. AutoGluon Training
```python
AutoGluonPredictor.train(
    time_limit=3600,
    presets='best_quality'
)
# → 50+ model otomatik denenir, en iyisi seçilir
```

### 4. TabNet Training
```python
TabNetHighXPredictor.train(
    max_epochs=200,
    patience=20,
    batch_size=256
)
# → Attention ile yüksek çarpan uzmanlaşması
```

## Prediction Strategies

### Ensemble Voting Strategies

#### Weighted
```python
final_prediction = (
    0.60 * catboost_prediction +
    0.40 * nn_prediction
)
```

#### Unanimous
```python
if catboost_pred == nn_pred:
    final_prediction = catboost_pred
else:
    skip_bet = True
```

#### Confidence
```python
if catboost_confidence > nn_confidence:
    final_prediction = catboost_prediction
else:
    final_prediction = nn_prediction
```

#### Majority
```python
final_prediction = mode([
    catboost_pred,
    nn_pred,
    autogluon_pred,
    tabnet_pred
])
```

## Risk Management Modes

### Rolling Mode (Konservatif)
```python
confidence_threshold = 0.80
suggested_multiplier = 1.5
risk_level = 'LOW'
```

### Normal Mode (Dengeli)
```python
confidence_threshold = 0.65
suggested_multiplier = dynamic
risk_level = 'MEDIUM'
```

### Aggressive Mode (Riskli)
```python
confidence_threshold = 0.50
suggested_multiplier = predicted * 0.80
risk_level = 'HIGH'
```

## Backtesting Strategies

### Fixed Betting
```python
bet_size = constant (örn: 10 TL)
```

### Kelly Criterion
```python
bet_size = bankroll * (
    (win_prob * win_multiplier - loss_prob) / 
    win_multiplier
)
```

### Confidence-Based
```python
if confidence >= 0.8:
    bet_size = base_bet * 2
elif confidence >= 0.6:
    bet_size = base_bet
else:
    skip_bet = True
```

## Performance Metrics

### Model Evaluation
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- Threshold Accuracy (1.5x binary)
- Below 1.5 Accuracy
- Above 1.5 Accuracy
- Confusion Matrix

### Backtesting Metrics
- ROI (Return on Investment)
- Win Rate
- Sharpe Ratio
- Maximum Drawdown
- Profit Factor
- Equity Curve

### Risk Metrics
- Para Kaybı Riski (False Positive Rate)
- Ardışık Kayıp Streaks
- Confidence Distribution
- Betting Frequency

## Konfigürasyon Yapısı

### config/config.yaml
```yaml
model:
  path: "models/jetx_model.h5"
  scaler_path: "models/scaler.pkl"

prediction:
  critical_threshold: 1.5
  confidence_thresholds:
    aggressive: 0.50
    normal: 0.65
    rolling: 0.80
  max_consecutive_losses: 3

database:
  path: "data/jetx_data.db"
  backup_dir: "data/backups"

ui:
  alerts:
    sound_enabled: true
    critical_zone_alert: true
    loss_streak_alert: true
```

## Veri Akışı

### Veri Girişi
```
Manuel Giriş (Streamlit UI) → database.add_result() → SQLite
```

### Tahmin Yapma
```
database.get_recent_results(500) → 
predictor.predict(history) →
{
    'predicted_value': float,
    'confidence': float,
    'above_threshold': bool,
    'category': str,
    'warnings': list
}
```

### Risk Değerlendirme
```
risk_manager.should_play(prediction) →
{
    'should_play': bool,
    'risk_level': str,
    'reasons': list
}
```

### Tahmin Kaydı
```
database.add_prediction(
    predicted_value,
    confidence,
    above_threshold,
    mode
)
```

### Sonuç Güncelleme
```
database.update_prediction_result(
    prediction_id,
    actual_value,
    was_correct
)
```

## Class Definitions

### CategoryDefinitions
```python
CRITICAL_THRESHOLD = 1.5

CATEGORIES = {
    'CRASH': [1.00, 1.49],
    'SAFE': [1.50, 2.99],
    'JACKPOT': [3.00, float('inf')]
}

@staticmethod
def float) -> int:
    # 0: CRASH, 1: SAFE, 2: JACKPOT
```

### FeatureEngineering
```python
@staticmethod
def extract_all_features(history: List[float]) -> Dict:
    # 150+ features extraction
    # Returns: OrderedDict of features
```

## Kurulum

### Gereksinimler
```bash
pip install -r requirements.txt
```

### Veritabanı Kurulumu
```python
from utils.database import DatabaseManager
db = DatabaseManager()
# Otomatik tablo oluşturma
```

### Streamlit Başlatma
```bash
streamlit run app.py
```

## Model Eğitimi (Google Colab)

### Progressive NN
```bash
python notebooks/jetx_PROGRESSIVE_TRAINING_MULTISCALE.py
```

### CatBoost
```bash
python notebooks/jetx_CATBOOST_TRAINING_MULTISCALE.py
```

### Model İndirme
```
models/ klasörüne kopyala:
├── progressive_multiscale/
│   ├── model_window_*.h5
│   ├── scaler_window_*.pkl
│   └── model_info.json
├── catboost_regressor.cbm
├── catboost_classifier.cbm
└── catboost_scaler.pkl
```

---

**Proje Tipi:** Machine Learning Prediction System  
**Framework:** TensorFlow, CatBoost, Streamlit  
**Database:** SQLite  
**Deployment:** Local / Google Colab Training
