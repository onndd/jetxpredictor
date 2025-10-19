# 🚀 JetX CPU Lightweight Models

CPU ile eğitilebilen hafif modeller için özelleştirilmiş Streamlit uygulaması. TabNet, AutoGluon, LightGBM, CatBoost gibi hafif modelleri destekler.

## 📋 İçindekiler

- [Özellikler](#-özellikler)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Model Özellikleri](#-model-özellikleri)
- [Eğitim Rehberi](#-eğitim-rehberi)
- [Hyperparameter Tuning](#-hyperparameter-tuning)
- [Ensemble Oluşturma](#-ensemble-oluşturma)
- [API Referansı](#-api-referansı)
- [Best Practices](#-best-practices)
- [Sorun Giderme](#-sorun-giderme)

## 🎯 Özellikler

### 🚀 Desteklenen Modeller
- **LightGBM**: CPU optimized gradient boosting
- **CatBoost**: Categorical boosting
- **TabNet**: Attention-based deep learning
- **AutoGluon**: Automated ML

### 🔧 Gelişmiş Özellikler
- **Model Eğitimi**: Gelişmiş eğitim arayüzü
- **Hyperparameter Tuning**: Optuna entegrasyonu
- **Model Karşılaştırma**: Side-by-side metrikler
- **Ensemble Builder**: Voting ve stacking
- **Prediction & Backtesting**: Real-time tahmin ve historical backtesting
- **Virtual Bankroll Simulation**: ROI hesaplama

### 💻 CPU Optimized
- Tüm modeller CPU üzerinde optimize edilmiş
- Memory efficient training
- Fast inference
- Cross-platform compatibility

## 🛠️ Kurulum

### Gereksinimler
- Python 3.8+
- 8GB+ RAM (16GB önerilen)
- 2GB+ disk alanı

### Adım 1: Repository Clone
```bash
git clone https://github.com/yourusername/jetxpredictor.git
cd jetxpredictor
```

### Adım 2: Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

### Adım 3: Dependencies
```bash
pip install -r requirements.txt
```

### Adım 4: Veri Hazırlığı
```bash
# jetx_data.db dosyasının mevcut olduğundan emin olun
# Eğer yoksa, ana uygulamadan veri yükleyin
```

### Adım 5: Uygulamayı Başlat
```bash
streamlit run app_cpu_models.py --server.port 8502
```

## 🎮 Kullanım

### Ana Dashboard
Ana sayfa model durumunu, sistem bilgilerini ve hızlı eylemleri gösterir.

### Model Eğitimi
1. **Model Tipi Seçin**: LightGBM, CatBoost, TabNet, AutoGluon
2. **Model Modu**: Classification, Regression, Multiclass
3. **Hyperparameters**: Model-specific parametreler
4. **Class Weights**: Imbalanced data için ağırlıklar
5. **Eğitimi Başlat**: Real-time progress tracking

### Hyperparameter Tuning
1. **Model Seçin**: Optimize edilecek model
2. **Search Space**: Parametre aralıkları tanımlayın
3. **Optimization**: Trial sayısı ve timeout
4. **Sonuçlar**: Best parameters ve visualization

### Model Karşılaştırma
1. **Modelleri Seçin**: Karşılaştırılacak modeller
2. **Test Verisi**: Test set boyutu
3. **Karşılaştır**: Side-by-side metrikler
4. **Görselleştirme**: Charts ve tablolar

### Ensemble Builder
1. **Modelleri Seçin**: Ensemble'e dahil edilecek modeller
2. **Strateji**: Voting (hard/soft) veya Stacking
3. **Ağırlıklar**: Weighted voting için
4. **Test**: Ensemble performance

### Prediction & Backtesting
1. **Model Seçin**: Tahmin yapacak model
2. **Mod**: Real-time veya Historical backtesting
3. **Tahmin**: Live prediction veya backtest
4. **Analiz**: Performance metrics ve visualization

## 🤖 Model Özellikleri

### LightGBM
- **Tip**: Gradient Boosting
- **Modlar**: Classification, Regression, Multiclass
- **Avantajlar**: Hızlı eğitim, memory efficient
- **Hyperparameters**: num_leaves, max_depth, learning_rate, feature_fraction

### CatBoost
- **Tip**: Categorical Boosting
- **Modlar**: Classification, Regression
- **Avantajlar**: Categorical features, overfitting resistance
- **Hyperparameters**: iterations, depth, learning_rate, l2_leaf_reg

### TabNet
- **Tip**: Attention-based Deep Learning
- **Modlar**: Classification, Multiclass
- **Avantajlar**: Interpretable, attention mechanism
- **Hyperparameters**: n_d, n_a, n_steps, gamma

### AutoGluon
- **Tip**: Automated ML
- **Modlar**: Classification, Regression
- **Avantajlar**: Auto-tuning, ensemble, no hyperparameter tuning needed
- **Hyperparameters**: time_limit, presets, eval_metric

## 📚 Eğitim Rehberi

### Temel Eğitim
1. **Veri Hazırlığı**: Database'den veri yükleme
2. **Feature Engineering**: Otomatik feature extraction
3. **Model Seçimi**: Use case'e göre model seçimi
4. **Eğitim**: Parametrelerle model eğitimi
5. **Değerlendirme**: Metrics ve visualization

### İleri Seviye Eğitim
1. **Hyperparameter Tuning**: Optuna ile optimization
2. **Cross-Validation**: Robust evaluation
3. **Ensemble**: Multiple models combination
4. **Backtesting**: Historical performance analysis

### Best Practices
- **Veri Split**: 70% train, 15% validation, 15% test
- **Class Weights**: Imbalanced data için
- **Early Stopping**: Overfitting prevention
- **Cross-Validation**: Model stability
- **Feature Importance**: Model interpretability

## 🔧 Hyperparameter Tuning

### Optuna Integration
```python
# Search space tanımlama
search_space = {
    'num_leaves': (10, 100),
    'max_depth': (3, 15),
    'learning_rate': (0.01, 0.3)
}

# Optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

### Visualization
- **Optimization History**: Trial progress
- **Parameter Importances**: Feature importance
- **Parallel Coordinate**: Parameter relationships

### Best Practices
- **Trial Sayısı**: 50-100 trials
- **Timeout**: 1-2 hours
- **CV Folds**: 5-fold cross-validation
- **Pruning**: Early stopping for bad trials

## 🤝 Ensemble Oluşturma

### Voting Strategy
- **Hard Voting**: Majority vote
- **Soft Voting**: Average probabilities
- **Weighted Voting**: Custom weights

### Stacking Strategy
- **Meta Model**: Secondary model
- **Cross-Validation**: Out-of-fold predictions
- **Probability/Class**: Input type

### Best Practices
- **Model Diversity**: Different algorithms
- **Performance**: Individual model quality
- **Correlation**: Low correlation between models
- **Validation**: Proper validation strategy

## 📖 API Referansı

### LightweightModelManager
```python
# Model oluşturma
model, model_id = manager.create_model(
    model_type='lightgbm',
    mode='classification',
    config=config
)

# Model eğitimi
metrics = manager.train_model(
    model_id=model_id,
    X=X_train,
    y=y_train
)

# Model karşılaştırma
results = manager.compare_models(
    model_ids=['model1', 'model2'],
    X_test=X_test,
    y_test=y_test
)
```

### CPUTrainingEngine
```python
# Tek model eğitimi
model_id, metrics = engine.train_single_model(
    model_type='lightgbm',
    X=X_train,
    y=y_train,
    mode='classification'
)

# Hyperparameter search
results = engine.hyperparameter_search(
    model_type='lightgbm',
    X=X_train,
    y=y_train,
    n_trials=50
)
```

### LightGBMPredictor
```python
# Model eğitimi
predictor = LightGBMPredictor(mode='classification')
metrics = predictor.train(X_train, y_train)

# Tahmin
predictions = predictor.predict(X_test)
probabilities = predictor.predict_proba(X_test)

# Feature importance
importance = predictor.get_feature_importance()
```

## 🏆 Best Practices

### Model Seçimi
- **LightGBM**: Hızlı eğitim, memory efficient
- **CatBoost**: Categorical features, robust
- **TabNet**: Interpretable, attention
- **AutoGluon**: No tuning, automated

### Hyperparameter Tuning
- **Search Space**: Reasonable ranges
- **Trial Count**: 50-100 trials
- **Validation**: Cross-validation
- **Early Stopping**: Prevent overfitting

### Ensemble
- **Diversity**: Different algorithms
- **Quality**: Good individual models
- **Validation**: Proper evaluation
- **Weighting**: Performance-based weights

### Performance
- **CPU Usage**: Monitor resource usage
- **Memory**: Efficient data handling
- **Speed**: Fast inference
- **Accuracy**: Balanced performance

## 🔍 Sorun Giderme

### Yaygın Sorunlar

#### Model Eğitimi Başarısız
- **Veri Kontrolü**: Yeterli veri var mı?
- **Feature Kontrolü**: Feature extraction çalışıyor mu?
- **Memory**: Yeterli RAM var mı?
- **Dependencies**: Tüm paketler yüklü mü?

#### Hyperparameter Tuning Yavaş
- **Trial Sayısı**: Azaltın
- **Timeout**: Artırın
- **CV Folds**: Azaltın
- **Search Space**: Küçültün

#### Ensemble Performance Düşük
- **Model Quality**: Individual model performance
- **Diversity**: Model çeşitliliği
- **Validation**: Proper validation
- **Weighting**: Optimal weights

#### Memory Issues
- **Batch Size**: Azaltın
- **Data Size**: Küçültün
- **Model Size**: Simpler models
- **Cleanup**: Memory cleanup

### Log Dosyaları
- **App Log**: `data/cpu_models.log`
- **Training Log**: Console output
- **Error Log**: Exception details

### Debug Mode
```bash
# Debug mode ile başlat
streamlit run app_cpu_models.py --logger.level debug
```

## 📞 Destek

### Dokümantasyon
- **GitHub**: [Repository](https://github.com/yourusername/jetxpredictor)
- **Issues**: [GitHub Issues](https://github.com/yourusername/jetxpredictor/issues)
- **Wiki**: [Project Wiki](https://github.com/yourusername/jetxpredictor/wiki)

### Topluluk
- **Discord**: [Community Server](https://discord.gg/yourinvite)
- **Reddit**: [r/jetxpredictor](https://reddit.com/r/jetxpredictor)
- **Stack Overflow**: `jetx-predictor` tag

### Geliştirici
- **Email**: developer@jetxpredictor.com
- **Twitter**: [@jetxpredictor](https://twitter.com/jetxpredictor)
- **LinkedIn**: [JetX Predictor](https://linkedin.com/company/jetxpredictor)

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🙏 Teşekkürler

- **Streamlit**: Web framework
- **LightGBM**: Gradient boosting
- **CatBoost**: Categorical boosting
- **TabNet**: Attention-based models
- **AutoGluon**: Automated ML
- **Optuna**: Hyperparameter optimization
- **Plotly**: Visualization

---

**🚀 JetX CPU Lightweight Models v1.0** - CPU Optimized Training & Prediction
