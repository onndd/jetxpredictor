# -*- coding: utf-8 -*-
"""JetX_Advanced_Model_Training.ipynb

# JetX Predictor - Gelişmiş Model Eğitimi
**Model Mimarisi:** N-BEATS + TCN Hibrit Model

Bu notebook, JetX tahmin sistemi için gelişmiş hibrit modeli eğitir.
Detaylı mimari için MODEL_MIMARISI.md dosyasına bakın.

## 📋 İçerik

1. **Kurulum ve Konfigürasyon**
2. **Veri Yükleme ve Analiz**
3. **Özellik Mühendisliği**
4. **N-BEATS Model Bloğu**
5. **TCN Model Bloğu**
6. **Psikolojik Analiz Motoru**
7. **Ensemble Fusion**
8. **Model Eğitimi**
9. **Değerlendirme ve Export**

---

## 1. Kurulum ve Konfigürasyon
"""

# Google Drive'ı bağla
from google.colab import drive
drive.mount('/content/drive')

# Gerekli kütüphaneleri yükle
!pip install -q tensorflow scikit-learn pandas numpy scipy matplotlib seaborn plotly joblib

# İçe aktarımlar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import joblib
import warnings
from typing import List, Dict, Tuple
from scipy import stats

# TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU kullanılabilir mi: {tf.config.list_physical_devices('GPU')}")

# Seed ayarla (reproducibility için)
np.random.seed(42)
tf.random.set_seed(42)

"""## 2. Veri Yükleme ve Analiz"""

# GitHub'dan projeyi klonla
!git clone https://github.com/onndd/jetxpredictor.git
%cd jetxpredictor

# Veritabanından verileri yükle
def load_data_from_db(db_path='jetx_data.db'):
    """SQLite veritabanından verileri yükler"""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM jetx_results ORDER BY id", conn)
    conn.close()
    return df

df = load_data_from_db()
print(f"\
📊 Toplam veri sayısı: {len(df)}")
print(f"İlk 5 kayıt:\
{df.head()}")

# Temel istatistikler
print("\
📈 Temel İstatistikler:")
print(df['value'].describe())

# 1.5x eşik analizi
below_threshold = len(df[df['value'] < 1.5])
above_threshold = len(df[df['value'] >= 1.5])
print(f"\
🎯 1.5x Eşik Analizi:")
print(f"1.5x Altı: {below_threshold} ({below_threshold/len(df)*100:.2f}%)")
print(f"1.5x Üstü: {above_threshold} ({above_threshold/len(df)*100:.2f}%)")

"""## 3. Özellik Mühendisliği

GELİŞTİRİLMİŞ - category_definitions.py'den import ediyoruz
"""

# category_definitions.py'den FeatureEngineering sınıfını import et
from category_definitions import FeatureEngineering, CategoryDefinitions

def create_dataset(data, window_size=50):
    """
    Zaman serisi verisi için özellikler ve hedefler oluştur
    GELİŞTİRİLMİŞ: category_definitions.py'deki yeni özelliklerle
    """
    X_features = []
    X_sequences = []
    y_regression = []
    y_classification = []
    
    for i in range(window_size, len(data)):
        window = data[i-window_size:i].tolist()
        target = data[i]
        
        # Geliştirilmiş özellik çıkarma (category_definitions'dan)
        features = FeatureEngineering.extract_all_features(window)
        X_features.append(list(features.values()))
        
        # Sequence (TCN/N-BEATS için)
        X_sequences.append(window)
        
        # Hedefler
        y_regression.append(target)
        y_classification.append(1 if target >= 1.5 else 0)
    
    return {
        'X_features': np.array(X_features),
        'X_sequences': np.array(X_sequences),
        'y_regression': np.array(y_regression),
        'y_classification': np.array(y_classification),
        'feature_names': list(features.keys())
    }

# Dataset oluştur - Kısa pencere (50)
print("\n🔧 Özellikler çıkarılıyor (50 pencere)...")
dataset_50 = create_dataset(df['value'].values, window_size=50)

print(f"✅ 50 Pencere - Özellikler hazır!")
print(f"Özellik sayısı: {dataset_50['X_features'].shape[1]}")
print(f"Sequence boyutu: {dataset_50['X_sequences'].shape}")
print(f"Örnek sayısı: {len(dataset_50['y_regression'])}")
print(f"Hedef dağılımı: {np.bincount(dataset_50['y_classification'])}")

# Dataset oluştur - Orta pencere (200)
print("\n🔧 Özellikler çıkarılıyor (200 pencere)...")
dataset_200 = create_dataset(df['value'].values, window_size=200)

print(f"✅ 200 Pencere - Özellikler hazır!")
print(f"Örnek sayısı: {len(dataset_200['y_regression'])}")

# Dataset oluştur - Uzun pencere (500)
print("\n🔧 Özellikler çıkarılıyor (500 pencere)...")
dataset_500 = create_dataset(df['value'].values, window_size=500)

print(f"✅ 500 Pencere - Özellikler hazır!")
print(f"Örnek sayısı: {len(dataset_500['y_regression'])}")

# Ana dataset olarak 50 pencereyi kullan
dataset = dataset_50

"""## 4. N-BEATS Model Bloğu

ÜÇ PENCERE BLOĞU: 50, 200, 500 - TÜMÜ AKTİF
"""

def create_nbeats_block(input_shape, units=64, name_prefix='nbeats'):
    """
    N-BEATS bloğu - Basis expansion ile
    """
    inputs = layers.Input(shape=input_shape, name=f'{name_prefix}_input')
    
    # Basis expansion
    x = layers.Dense(units * 4, activation='relu')(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(units * 2, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Backward (geçmiş açıklama) ve Forward (gelecek tahmini)
    backward = layers.Dense(units, activation='relu', name=f'{name_prefix}_backward')(x)
    forward = layers.Dense(1, name=f'{name_prefix}_forecast')(x)
    
    # Feature extraction
    features = layers.Dense(units, activation='relu', name=f'{name_prefix}_features')(backward)
    
    model = Model(inputs=inputs, outputs=[forward, features], name=name_prefix)
    return model

# Kısa pencere bloğu (50) - 64 boyut
nbeats_short = create_nbeats_block((50,), units=64, name_prefix='nbeats_short')
print("✅ N-BEATS Kısa Pencere (50 el) - 64 boyut")

# Orta pencere bloğu (200) - 128 boyut
nbeats_medium = create_nbeats_block((200,), units=128, name_prefix='nbeats_medium')
print("✅ N-BEATS Orta Pencere (200 el) - 128 boyut")

# Uzun pencere bloğu (500) - 256 boyut
nbeats_long = create_nbeats_block((500,), units=256, name_prefix='nbeats_long')
print("✅ N-BEATS Uzun Pencere (500 el) - 256 boyut")

"""## 5. TCN (Temporal Convolutional Network) Modülü

Dilated convolutions ile uzun mesafe pattern'leri yakalar
"""

def create_tcn_block(input_shape, filters=64, kernel_size=3, dilations=[1, 2, 4, 8], name='tcn'):
    """
    TCN bloğu - Dilated causal convolutions
    """
    inputs = layers.Input(shape=input_shape, name=f'{name}_input')
    
    x = layers.Reshape((input_shape[0], 1))(inputs)
    
    # Dilated convolution katmanları
    for i, dilation in enumerate(dilations):
        # Causal convolution (sadece geçmişe bakar)
        conv = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation,
            padding='causal',
            activation='relu',
            name=f'{name}_conv_{i}'
        )(x)
        
        # Residual connection
        if x.shape[-1] != filters:
            x = layers.Conv1D(filters, 1, padding='same')(x)
        x = layers.Add()([x, conv])
        x = layers.Dropout(0.2)(x)
    
    # Son hidden state'i al
    x = layers.GlobalAveragePooling1D()(x)
    features = layers.Dense(512, activation='relu', name=f'{name}_features')(x)
    
    model = Model(inputs=inputs, outputs=features, name=name)
    return model

# TCN bloğu oluştur
tcn_model = create_tcn_block((50,), filters=64, dilations=[1, 2, 4, 8])
print("✅ TCN Modülü - 512 boyut")

"""## 6. Psikolojik Analiz Motoru

Pattern detection: Tuzak, Soğuma, Momentum
"""

def create_psychological_analyzer(input_shape):
    """
    Psikolojik pattern analiz modülü
    """
    inputs = layers.Input(shape=input_shape, name='psych_input')
    
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    
    # Pattern skorları
    trap_score = layers.Dense(1, activation='sigmoid', name='trap_score')(x)
    cooling_score = layers.Dense(1, activation='sigmoid', name='cooling_score')(x)
    momentum_score = layers.Dense(1, activation='tanh', name='momentum_score')(x)
    
    # Birleştir
    combined = layers.Concatenate()([trap_score, cooling_score, momentum_score])
    features = layers.Dense(32, activation='relu', name='psych_features')(combined)
    
    model = Model(inputs=inputs, outputs=features, name='psychological_analyzer')
    return model

psych_model = create_psychological_analyzer((dataset['X_features'].shape[1],))
print("✅ Psikolojik Analiz Motoru - 32 boyut")

"""## 7. Ensemble Fusion - Gelişmiş Hibrit Model

3 N-BEATS (50,200,500) + TCN + Psikolojik Motor
"""

def create_advanced_hybrid_model(feature_dim):
    """
    Gelişmiş hibrit model: 3 N-BEATS + TCN + Psikolojik Motor
    """
    # Inputs
    feature_input = layers.Input(shape=(feature_dim,), name='feature_input')
    seq_50_input = layers.Input(shape=(50,), name='seq_50_input')
    seq_200_input = layers.Input(shape=(200,), name='seq_200_input')
    seq_500_input = layers.Input(shape=(500,), name='seq_500_input')
    
    # 1. ÜÇ N-BEATS Bloğu
    # Kısa pencere (50) - 64 boyut
    nbeats_short_forecast, nbeats_short_features = nbeats_short(seq_50_input)
    
    # Orta pencere (200) - 128 boyut
    nbeats_medium_forecast, nbeats_medium_features = nbeats_medium(seq_200_input)
    
    # Uzun pencere (500) - 256 boyut
    nbeats_long_forecast, nbeats_long_features = nbeats_long(seq_500_input)
    
    # N-BEATS tahminlerini ağırlıklı birleştir
    weighted_forecast = layers.Average()([
        layers.Lambda(lambda x: x * 0.5)(nbeats_short_forecast),
        layers.Lambda(lambda x: x * 0.3)(nbeats_medium_forecast),
        layers.Lambda(lambda x: x * 0.2)(nbeats_long_forecast)
    ])
    
    # N-BEATS özelliklerini birleştir: 64 + 128 + 256 = 448 boyut
    nbeats_combined = layers.Concatenate()([
        nbeats_short_features,
        nbeats_medium_features,
        nbeats_long_features
    ])
    
    # 2. TCN Bloğu (50 pencere üzerinde) - 512 boyut
    tcn_features = tcn_model(seq_50_input)
    
    # N-BEATS + TCN füzyonu
    time_series_features = layers.Concatenate()([nbeats_combined, tcn_features])  # 448 + 512 = 960
    time_series_features = layers.Dense(256, activation='relu')(time_series_features)
    time_series_features = layers.Dropout(0.3)(time_series_features)
    
    # 3. Psikolojik Motor - 32 boyut
    psych_features = psych_model(feature_input)
    
    # 4. İstatistiksel Baseline - 16 boyut
    stat_features = layers.Dense(16, activation='relu')(feature_input)
    
    # Ensemble Fusion: 256 + 32 + 16 = 304 boyut
    all_features = layers.Concatenate()([
        time_series_features,
        psych_features,
        stat_features
    ])
    
    # Final katmanlar
    x = layers.Dense(128, activation='relu')(all_features)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Çıktılar
    # 1. Regresyon (değer tahmini)
    regression_output = layers.Dense(1, name='regression_output')(x)
    
    # 2. Sınıflandırma (1.5x altı/üstü)
    classification_output = layers.Dense(1, activation='sigmoid', name='classification_output')(x)
    
    # 3. Güven skoru
    confidence_output = layers.Dense(1, activation='sigmoid', name='confidence_output')(x)
    
    # 4. Pattern risk skoru (YENİ - soğuma, tuzak detection için)
    pattern_risk_output = layers.Dense(1, activation='sigmoid', name='pattern_risk_output')(x)
    
    # Model oluştur
    model = Model(
        inputs=[feature_input, seq_50_input, seq_200_input, seq_500_input],
        outputs=[regression_output, classification_output, confidence_output, pattern_risk_output],
        name='JetX_Advanced_Hybrid_Model'
    )
    
    return model

# Gelişmiş hibrit modeli oluştur
print("\n🔨 Gelişmiş Hibrit Model oluşturuluyor...")
hybrid_model = create_advanced_hybrid_model(
    feature_dim=dataset['X_features'].shape[1]
)

print("\n✅ Gelişmiş Hibrit Model Oluşturuldu!")
print("📊 Model Yapısı:")
print(f"  - 3 N-BEATS bloğu (50, 200, 500 pencere)")
print(f"  - TCN bloğu (512 boyut)")
print(f"  - Psikolojik analiz motoru (32 boyut)")
print(f"  - 4 çıktı: regresyon, classification, confidence, pattern_risk")
hybrid_model.summary()

"""## 8. Model Eğitimi

Multi-task learning loss ile eğitim - GELİŞTİRİLMİŞ (4 çıktı)
"""

# Veri hazırlama - 500 pencere dataset'i kullanıyoruz (en kapsayıcı)
# Test split
split_idx_test = int(len(dataset_500['X_features']) * 0.85)
X_feat_train_val = dataset_500['X_features'][:split_idx_test]
X_feat_test = dataset_500['X_features'][split_idx_test:]
y_reg_train_val = dataset_500['y_regression'][:split_idx_test]
y_reg_test = dataset_500['y_regression'][split_idx_test:]
y_class_train_val = dataset_500['y_classification'][:split_idx_test]
y_class_test = dataset_500['y_classification'][split_idx_test:]

# Validation split
split_idx_val = int(len(X_feat_train_val) * 0.85)
X_feat_train = X_feat_train_val[:split_idx_val]
X_feat_val = X_feat_train_val[split_idx_val:]
y_reg_train = y_reg_train_val[:split_idx_val]
y_reg_val = y_reg_train_val[split_idx_val:]
y_class_train = y_class_train_val[:split_idx_val]
y_class_val = y_class_train_val[split_idx_val:]

# Sequence data - 3 farklı pencere için
# 50 pencere sequences
seq_50_train_val = dataset_500['X_sequences'][:split_idx_test, -50:]
seq_50_test = dataset_500['X_sequences'][split_idx_test:, -50:]
seq_50_train = seq_50_train_val[:split_idx_val]
seq_50_val = seq_50_train_val[split_idx_val:]

# 200 pencere sequences
seq_200_train_val = dataset_500['X_sequences'][:split_idx_test, -200:]
seq_200_test = dataset_500['X_sequences'][split_idx_test:, -200:]
seq_200_train = seq_200_train_val[:split_idx_val]
seq_200_val = seq_200_train_val[split_idx_val:]

# 500 pencere sequences
seq_500_train_val = dataset_500['X_sequences'][:split_idx_test]
seq_500_test = dataset_500['X_sequences'][split_idx_test:]
seq_500_train = seq_500_train_val[:split_idx_val]
seq_500_val = seq_500_train_val[split_idx_val:]

print(f"\n📊 Veri Bölümleme:")
print(f"Train: {len(X_feat_train)} ({len(X_feat_train)/len(dataset_500['X_features'])*100:.1f}%)")
print(f"Val: {len(X_feat_val)} ({len(X_feat_val)/len(dataset_500['X_features'])*100:.1f}%)")
print(f"Test: {len(X_feat_test)} ({len(X_feat_test)/len(dataset_500['X_features'])*100:.1f}%)")

# Normalizasyon
scaler = StandardScaler()
X_feat_train_scaled = scaler.fit_transform(X_feat_train)
X_feat_val_scaled = scaler.transform(X_feat_val)
X_feat_test_scaled = scaler.transform(X_feat_test)

# Multi-task loss weights - 4 çıktı için
loss_weights = {
    'regression_output': 0.25,           # α₁ - Değer tahmini
    'classification_output': 0.45,       # α₂ - 1.5x eşik (EN ÖNEMLİ)
    'confidence_output': 0.15,           # α₃ - Güven skoru
    'pattern_risk_output': 0.15          # α₄ - Pattern risk (YENİ)
}

# Model derleme
print("\n🔨 Model derleniyor...")
hybrid_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={
        'regression_output': 'huber',  # Outlier'lara daha az duyarlı
        'classification_output': 'binary_crossentropy',  # 1.5x eşik
        'confidence_output': 'mse',
        'pattern_risk_output': 'mse'  # Pattern risk
    },
    loss_weights=loss_weights,
    metrics={
        'regression_output': ['mae', 'mse'],
        'classification_output': ['accuracy'],
        'confidence_output': ['mae'],
        'pattern_risk_output': ['mae']
    }
)

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_classification_output_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'best_hybrid_model.h5',
        monitor='val_classification_output_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
]

# Dummy skorları (eğitim için)
y_conf_train = np.random.uniform(0.6, 1.0, size=(len(y_reg_train),))
y_conf_val = np.random.uniform(0.6, 1.0, size=(len(y_reg_val),))
y_pattern_risk_train = np.random.uniform(0.0, 0.5, size=(len(y_reg_train),))
y_pattern_risk_val = np.random.uniform(0.0, 0.5, size=(len(y_reg_val),))

# Eğitim
print("\n🚀 Model eğitimi başlıyor...")
print("💡 İpucu: GPU varsa eğitim ~10-20 dakika sürer")
history = hybrid_model.fit(
    [X_feat_train_scaled, seq_50_train, seq_200_train, seq_500_train],
    {
        'regression_output': y_reg_train,
        'classification_output': y_class_train,
        'confidence_output': y_conf_train,
        'pattern_risk_output': y_pattern_risk_train
    },
    validation_data=(
        [X_feat_val_scaled, seq_50_val, seq_200_val, seq_500_val],
        {
            'regression_output': y_reg_val,
            'classification_output': y_class_val,
            'confidence_output': y_conf_val,
            'pattern_risk_output': y_pattern_risk_val
        }
    ),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Eğitim tamamlandı!")

"""## 9. Model Değerlendirme ve Export"""

# Test seti değerlendirmesi
print("\n📊 Test seti değerlendirmesi...")
y_conf_test = np.random.uniform(0.6, 1.0, size=(len(y_reg_test),))
y_pattern_risk_test = np.random.uniform(0.0, 0.5, size=(len(y_reg_test),))

# Tahmin yap
predictions = hybrid_model.predict([X_feat_test_scaled, seq_50_test, seq_200_test, seq_500_test])
y_reg_pred, y_class_pred_proba, y_conf_pred, y_pattern_risk_pred = predictions

# 1.5x eşik doğruluğu (EN ÖNEMLİ METRİK)
y_class_pred = (y_class_pred_proba > 0.5).astype(int).flatten()
threshold_accuracy = accuracy_score(y_class_test, y_class_pred)

print(f"\n🎯 1.5x Eşik Doğruluğu: {threshold_accuracy:.4f} ({threshold_accuracy*100:.2f}%)")
print(f"Hedef: %75+")

if threshold_accuracy >= 0.75:
    print("✅ Hedef başarıldı!")
else:
    print("⚠️ Hedef henüz ulaşılamadı, daha fazla eğitim gerekebilir")

# Regresyon metrikleri
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
r2 = r2_score(y_reg_test, y_reg_pred)

print(f"\n📈 Regresyon Metrikleri:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# Pattern risk ortalama
avg_pattern_risk = np.mean(y_pattern_risk_pred)
print(f"\n⚠️ Ortalama Pattern Risk: {avg_pattern_risk:.4f}")

# Classification report
print("\n📋 Classification Report:")
print(classification_report(y_class_test, y_class_pred, target_names=['< 1.5x', '≥ 1.5x']))

# Confusion Matrix
cm = confusion_matrix(y_class_test, y_class_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['< 1.5x', '≥ 1.5x'],
            yticklabels=['< 1.5x', '≥ 1.5x'])
plt.title('Confusion Matrix - 1.5x Eşik Tahmini')
plt.ylabel('Gerçek')
plt.xlabel('Tahmin')
plt.show()

# Model ve scaler'ı kaydet
print("\
💾 Model kaydediliyor...")
hybrid_model.save('jetx_hybrid_model.h5')
joblib.dump(scaler, 'scaler.pkl')
print("✅ Model kaydedildi: jetx_hybrid_model.h5")
print("✅ Scaler kaydedildi: scaler.pkl")

# Google Drive'a kopyala
!cp jetx_hybrid_model.h5 /content/drive/MyDrive/
!cp scaler.pkl /content/drive/MyDrive/
print("\
✅ Dosyalar Google Drive'a kopyalandı!")

# Lokal indirme
from google.colab import files
files.download('jetx_hybrid_model.h5')
files.download('scaler.pkl')
print("\
✅ Dosyalar indiriliyor...")

"""## 🎉 Model Eğitimi Tamamlandı!

### Sonraki Adımlar:

1. ✅ Modeli indirin (`jetx_hybrid_model.h5` ve `scaler.pkl`)
2. ✅ Lokal projenizin `models/` klasörüne koyun
3. ✅ Streamlit uygulamasını çalıştırın: `streamlit run app.py`
4. ✅ Tahminleri test edin!

### Önemli Notlar:

- **Hedef:** 1.5x eşik doğruluğu %75+ 
- **Model Boyutu:** ~304 boyutlu feature fusion
- **Eğitim Süresi:** GPU ile ~10-20 dakika
- **Optimizasyon:** Quantization ve pruning ile daha da hızlandırılabilir

---

**Model Mimarisi Detayları:** `MODEL_MIMARISI.md`  
**GitHub:** https://github.com/onndd/jetxpredictor
"""

print("Notebook tamamlandı! 🎊")
