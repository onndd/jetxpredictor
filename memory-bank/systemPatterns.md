# JetX Predictor - Sistem Patternleri

## Mimari Genel Bakış

### 1. Katmanlı Mimari

#### Presentation Layer (Streamlit)
```
app.py (Ana UI)
├── pages/
│   ├── 1_📊_Analiz.py (Veri analizi)
│   ├── 2_🔬_Model_Karsilastirma.py (Model karşılaştırma)
│   └── cpu/ (CPU modelleri için özel sayfalar)
└── components/
    ├── Prediction Display
    ├── Risk Analysis
    └── Performance Charts
```

#### Business Logic Layer (Utils)
```
utils/
├── Core Prediction
│   ├── predictor.py (Ana tahmin motoru)
│   ├── ensemble_predictor.py (Ensemble sistemi)
│   └── consensus_predictor.py (Consensus voting)
├── Model Management
│   ├── all_models_predictor.py (Tüm modeller)
│   ├── model_loader.py (Model yükleme)
│   └── model_versioning.py (Versiyon yönetimi)
├── Risk Management
│   ├── risk_manager.py (Risk analizi)
│   ├── advanced_bankroll.py (Kelly Criterion)
│   └── dual_bankroll_system.py (İkili kasa)
├── Data Processing
│   ├── database.py (Veritabanı yönetimi)
│   ├── category_definitions.py (Feature extraction)
│   └── multi_scale_window.py (Multi-scale windows)
└── Analysis & Monitoring
    ├── backtesting.py (Geçmiş performans)
    ├── psychological_analyzer.py (Psikolojik analiz)
    └── anomaly_streak_detector.py (Anomali tespiti)
```

#### Data Layer (SQLite + Models)
```
data/
├── jetx_data.db (SQLite veritabanı)
└── cache/ (Önbellek)
models/
├── progressive_multiscale/ (NN modelleri)
├── catboost_multiscale/ (CatBoost modelleri)
└── cpu/ (CPU optimize modelleri)
```

## 2. Model Ensemble Patterni

### Multi-Model Stratejisi

#### 1. Weighted Voting (Varsayılan)
```python
final_prediction = (
    0.60 * catboost_prediction +  # %60 ağırlık
    0.40 * neural_network_prediction  # %40 ağırlık
)
```

#### 2. Unanimous Voting
```python
if catboost_pred == neural_pred:
    final_prediction = catboost_pred
else:
    skip_bet = True  # Fikir ayrılığı varsa bekle
```

#### 3. Confidence-Based Voting
```python
if catboost_confidence > neural_confidence:
    final_prediction = catboost_prediction
else:
    final_prediction = neural_network_prediction
```

#### 4. Majority Voting
```python
predictions = [catboost, neural, autogluon, tabnet]
final_prediction = mode(predictions)  # En çok oylanmış
```

### Multi-Scale Ensemble Patterni

#### Window Size Stratejisi
```python
window_weights = {
    20: 0.10,   # Kısa dönem (lokal volatilite)
    50: 0.15,   # Kısa-orta dönem
    100: 0.30,  # Orta dönem (en yüksek ağırlık)
    250: 0.25,  # Orta-uzun dönem
    500: 0.20   # Uzun dönem (genel trend)
}

final_prediction = sum(weights[ws] * predictions[ws] for ws in window_sizes)
```

#### Model Adaptasyon Patterni
- **Küçük Pencereler (≤50)**: Basit LSTM, hızlı öğrenme
- **Orta Pencereler (≤100)**: 2-layer LSTM, attention
- **Büyük Pencereler (>100)**: 3-layer LSTM + attention mechanism

## 3. Veri Akış Patternleri

### Time Series Data Flow
```
1. Manuel Input → SQLite Database
2. Database Query → Feature Extraction (150+ features)
3. Feature Processing → Model Input Preparation
4. Model Prediction → Confidence Scoring
5. Risk Analysis → User Interface
6. Result Input → Database Update → Model Learning
```

### Feature Engineering Pipeline
```python
# 1. Temel İstatistikler
basic_stats = extract_basic_statistics(history)

# 2. Multi-Scale Windows
windows = create_multi_scale_windows(history, [20, 50, 100, 250, 500])

# 3. Advanced Features
advanced_features = {
    'volatility': calculate_volatility(history),
    'streaks': extract_streak_patterns(history),
    'threshold_analysis': analyze_threshold_patterns(history, 1.5),
    'psychological': analyze_psychological_patterns(history)
}

# 4. Feature Birleştirme
all_features = combine_features(basic_stats, windows, advanced_features)
```

### Data Validation Pattern
```python
def validate_input_data(data):
    # Minimum veri kontrolü
    if len(data) < 50:
        raise ValueError("En az 50 veri noktası gerekli")
    
    # Veri aralığı kontrolü
    if not all(1.0 <= x <= 10000.0 for x in data):
        raise ValueError("Veri 1.0-10000.0 aralığında olmalı")
    
    # Kronolojik sıra kontrolü
    if not is_chronological(data):
        raise ValueError("Veri kronolojik sıralı olmalı")
    
    return True
```

## 4. Risk Management Patternleri

### Üç Katmanlı Risk Sistemi

#### Level 1: Prediction Confidence
```python
confidence_thresholds = {
    'aggressive': 0.50,  # Yüksek risk, yüksek potansiyel
    'normal': 0.65,     # Dengeli risk-getiri oranı
    'rolling': 0.80      # Düşük risk, konservatif
}

if confidence < confidence_thresholds[mode]:
    return {'should_play': False, 'reason': 'Düşük güven skoru'}
```

#### Level 2: Consecutive Loss Tracking
```python
max_consecutive_losses = {
    'aggressive': 5,
    'normal': 3,
    'rolling': 2
}

if consecutive_losses >= max_consecutive_losses[mode]:
    return {'should_play': False, 'reason': 'Ardışık kayıp limiti'}
```

#### Level 3: Bankroll Management
```python
# Kelly Criterion
kelly_fraction = (win_prob * win_multiplier - loss_prob) / win_multiplier
optimal_bet = bankroll * max(kelly_fraction, 0.25)  # Max %25

# Stop-Loss / Take-Profit
if cumulative_loss > stop_loss_threshold:
    force_stop_trading()
if cumulative_profit > take_profit_threshold:
    secure_profits()
```

## 5. Model Training Patternleri

### Google Colab → Lokal Pipeline

#### Colab Training Pattern
```python
# 1. Veri Hazırlığı
data = load_from_sqlite()
X_train, X_val, X_test = time_series_split(data, [0.7, 0.15, 0.15])

# 2. Multi-Scale Training
for window_size in [500, 250, 100, 50, 20]:
    model = create_model_for_window(window_size)
    train_model(model, X_train, X_val)
    save_model(model, f'model_window_{window_size}.h5')

# 3. Ensemble Değerlendirmesi
ensemble_score = evaluate_ensemble(all_models, X_test)
save_best_models(ensemble_score)

# 4. ZIP ve İndirme
create_model_zip()
files.download('jetx_models_v3.0.zip')
```

#### Lokal Loading Pattern
```python
# 1. Otomatik Model Tespiti
available_models = scan_models_directory()
model_registry = create_model_registry(available_models)

# 2. Versiyon Yönetimi
production_models = get_production_models()
latest_version = get_latest_version('progressive_nn')

# 3. Dinamik Yükleme
if production_models:
    load_production_models()
else:
    load_available_models_with_fallback()
```

### Training Optimization Patternleri

#### Learning Rate Scheduling
```python
# Cosine Annealing
lr_schedule = CosineAnnealingSchedule(
    initial_lr=0.001,
    max_lr=0.01,
    min_lr=0.0001,
    steps_per_epoch=len(X_train)//batch_size
)

# Adaptive Weight Scheduling
weight_scheduler = AdaptiveWeightScheduler(
    metrics=['val_accuracy', 'val_roi'],
    patience=10,
    factor=0.5
)
```

#### Early Stopping Pattern
```python
early_stopping = EarlyStopping(
    monitor='val_stability_score',  # Sadece loss değil
    patience=15,
    restore_best_weights=True,
    min_delta=0.01,
    mode='max'
)
```

## 6. Error Handling Patternleri

### Katmanlı Error Management

#### Level 1: Input Validation
```python
try:
    prediction = predict(history)
except ValueError as e:
    logger.error(f"Input validation hatası: {e}")
    return {'error': 'Geçersiz veri', 'suggestion': 'Veri formatını kontrol edin'}
except IndexError as e:
    logger.error(f"Veri uzunluğu hatası: {e}")
    return {'error': 'Yetersiz veri', 'suggestion': 'En az 50 veri noktası gerekli'}
```

#### Level 2: Model Error Recovery
```python
try:
    result = model.predict(input_data)
except ModelLoadError:
    logger.warning("Ana model yüklenemedi, fallback model deneniyor")
    result = fallback_model.predict(input_data)
except PredictionError:
    logger.error("Tahmin hatası, ensemble deneniyor")
    result = ensemble_predictor.predict(input_data)
```

#### Level 3: Graceful Degradation
```python
if primary_model.confidence < 0.5:
    # Ana model düşük güven veriyorsa
    if secondary_model.available:
        result = secondary_model.predict(input_data)
    else:
        result = conservative_strategy.default_prediction()
```

## 7. Performance Monitoring Patternleri

### Real-time Monitoring
```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics_history = []
        self.alert_thresholds = {
            'accuracy_drop': 0.10,  # %10 düşüş
            'confidence_drop': 0.15,  # %15 düşüş
            'consecutive_losses': 5
        }
    
    def check_performance(self, current_metrics):
        alerts = []
        for metric, threshold in self.alert_thresholds.items():
            if self.detect_drop(metric, current_metrics[metric], threshold):
                alerts.append(f"{metric} düşüş tespiti")
        return alerts
```

### Model Drift Detection
```python
def detect_model_drift(recent_predictions, historical_performance):
    current_accuracy = calculate_accuracy(recent_predictions)
    historical_avg = historical_performance['avg_accuracy']
    
    drift_threshold = 0.15  # %15 fark
    
    if abs(current_accuracy - historical_avg) > drift_threshold:
        return {
            'drift_detected': True,
            'severity': 'high' if abs(current_accuracy - historical_avg) > 0.25 else 'medium',
            'suggestion': 'Model yeniden eğitimi öneriliyor'
        }
    
    return {'drift_detected': False}
```

## 8. Configuration Management Patternleri

### Hierarchical Configuration
```yaml
# config/config.yaml (Ana konfigürasyon)
database:
  path: "data/jetx_data.db"
  
model:
  path: "models/jetx_model.h5"
  scaler_path: "models/scaler.pkl"
  
prediction:
  confidence_thresholds:
    aggressive: 0.50
    normal: 0.65
    rolling: 0.80

# config/cpu_models_config.yaml (CPU modelleri)
models:
  lightgbm:
    enabled: true
    parameters: {...}
  tabnet:
    enabled: true
    parameters: {...}
```

### Environment-Specific Configuration
```python
class ConfigLoader:
    def __init__(self, environment='development'):
        self.config = self.load_config()
        self.environment = environment
        self.apply_environment_overrides()
    
    def apply_environment_overrides(self):
        if self.environment == 'production':
            self.config['logging']['level'] = 'WARNING'
            self.config['debug']['enabled'] = False
        elif self.environment == 'development':
            self.config['logging']['level'] = 'DEBUG'
            self.config['debug']['enabled'] = True
```

## 9. Testing Patternleri

### Multi-Level Testing Strategy

#### Unit Tests
```python
def test_feature_extraction():
    # Feature extraction doğruluğu
    sample_data = [1.2, 1.5, 2.1, 1.8, 3.2]
    features = extract_features(sample_data)
    assert len(features) == 150, "150 feature bekleniyordu"
    assert 'volatility' in features, "Volatilite feature eksik"

def test_risk_management():
    # Risk yönetimi mantığı
    assert risk_manager.should_play({'confidence': 0.3}) == False
    assert risk_manager.should_play({'confidence': 0.8}) == True
```

#### Integration Tests
```python
def test_full_pipeline():
    # Tam pipeline testi
    # 1. Veri girişi
    db.add_result(1.5)
    
    # 2. Tahmin
    prediction = predictor.predict(db.get_recent_results(100))
    
    # 3. Risk analizi
    risk = risk_manager.should_play(prediction)
    
    # 4. Sonuç güncelleme
    db.update_prediction_result(prediction['id'], 2.0, risk['was_correct'])
    
    assert prediction['predicted_value'] > 0
    assert 'confidence' in prediction
```

#### Performance Tests
```python
def test_prediction_speed():
    # Tahmin hızı testi
    start_time = time.time()
    for _ in range(100):
        predictor.predict(large_history_dataset)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    assert avg_time < 1.0, "Tahmin <1 saniye olmalı"
```

---

*Bu belge sistemin temel tasarım patternlerini, mimari kararlarını ve en iyi uygulamalarını tanımlar. Tüm geliştirme bu patternlere uygun olmalıdır.*
