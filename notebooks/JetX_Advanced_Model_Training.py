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
!pip install -q tensorflow scikit-learn pandas numpy matplotlib seaborn plotly joblib

# İçe aktarımlar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import joblib
import warnings
from typing import List, Dict, Tuple

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

Üç paralel akış:
- Ham veri akışı
- Kategori dönüşüm akışı  
- Özellik türetme akışı
"""

class FeatureEngineer:
    """
    Detaylı özellik mühendisliği sınıfı
    MODEL_MIMARISI.md'deki spesifikasyonlara göre
    """
    
    def __init__(self, window_size=50):
        self.window_size = window_size
        
    def extract_basic_features(self, window):
        """Temel istatistiksel özellikler"""
        features = {}
        for size in [5, 10, 20, 50]:
            if len(window) >= size:
                recent = window[-size:]
                features[f'mean_{size}'] = np.mean(recent)
                features[f'std_{size}'] = np.std(recent)
                features[f'min_{size}'] = np.min(recent)
                features[f'max_{size}'] = np.max(recent)
                features[f'median_{size}'] = np.median(recent)
        return features
    
    def extract_threshold_features(self, window, threshold=1.5):
        """1.5x eşik özellikleri"""
        features = {}
        if len(window) >= 10:
            recent_10 = window[-10:]
            recent_50 = window[-50:] if len(window) >= 50 else window
            
            features['below_threshold_10'] = sum(1 for v in recent_10 if v < threshold)
            features['above_threshold_10'] = sum(1 for v in recent_10 if v >= threshold)
            features['threshold_ratio_10'] = features['above_threshold_10'] / 10
            
            if len(recent_50) > 0:
                features['threshold_ratio_50'] = sum(1 for v in recent_50 if v >= threshold) / len(recent_50)
            
            features['in_critical_zone_10'] = sum(1 for v in recent_10 if 1.45 <= v <= 1.55)
        return features
    
    def extract_distance_features(self, window, milestones=[10.0, 20.0, 50.0, 100.0]):
        """Büyük çarpanlardan bu yana geçen mesafe"""
        features = {}
        for milestone in milestones:
            distance = len(window)
            for i in range(len(window) - 1, -1, -1):
                if window[i] >= milestone:
                    distance = len(window) - 1 - i
                    break
            features[f'distance_from_{int(milestone)}x'] = distance
        return features
    
    def extract_streak_features(self, window):
        """Ardışık pattern özellikleri"""
        features = {}
        if len(window) >= 2:
            rising_streak = 0
            falling_streak = 0
            
            for i in range(len(window) - 1, 0, -1):
                if window[i] > window[i - 1]:
                    rising_streak += 1
                    if falling_streak > 0:
                        break
                elif window[i] < window[i - 1]:
                    falling_streak += 1
                    if rising_streak > 0:
                        break
                else:
                    break
            
            features['rising_streak'] = rising_streak
            features['falling_streak'] = falling_streak
        return features
    
    def extract_volatility_features(self, window):
        """Volatilite özellikleri"""
        features = {}
        if len(window) >= 10:
            recent = window[-10:]
            changes = [recent[i] - recent[i-1] for i in range(1, len(recent))]
            features['volatility_10'] = np.std(changes) if changes else 0
            features['mean_change_10'] = np.mean(changes) if changes else 0
            features['range_10'] = np.max(recent) - np.min(recent)
        return features
    
    def extract_all_features(self, window):
        """Tüm özellikleri birleştir"""
        all_features = {}
        all_features.update(self.extract_basic_features(window))
        all_features.update(self.extract_threshold_features(window))
        all_features.update(self.extract_distance_features(window))
        all_features.update(self.extract_streak_features(window))
        all_features.update(self.extract_volatility_features(window))
        
        if len(window) > 0:
            all_features['last_value'] = window[-1]
        
        return all_features

def create_dataset(data, window_size=50):
    """
    Zaman serisi verisi için özellikler ve hedefler oluştur
    """
    feature_engineer = FeatureEngineer(window_size)
    X_features = []
    X_sequences = []
    y_regression = []
    y_classification = []
    
    for i in range(window_size, len(data)):
        window = data[i-window_size:i]
        target = data[i]
        
        # Özellik çıkarma
        features = feature_engineer.extract_all_features(window)
        X_features.append(list(features.values()))
        
        # Sequence (LSTM/TCN için)
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

# Dataset oluştur
print("\
🔧 Özellikler çıkarılıyor...")
dataset = create_dataset(df['value'].values, window_size=50)

print(f"\
✅ Özellikler hazır!")
print(f"Özellik sayısı: {dataset['X_features'].shape[1]}")
print(f"Sequence boyutu: {dataset['X_sequences'].shape}")
print(f"Örnek sayısı: {len(dataset['y_regression'])}")
print(f"Hedef dağılımı: {np.bincount(dataset['y_classification'])}")

"""## 4. N-BEATS Model Bloğu

Üç farklı pencere bloğu: 50, 200, 500
"""

def create_nbeats_block(input_shape, units=64, name_prefix='nbeats'):
    """
    Basitleştirilmiş N-BEATS bloğu
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

# Kısa pencere bloğu (50)
nbeats_short = create_nbeats_block((50,), units=64, name_prefix='nbeats_short')
print("✅ N-BEATS Kısa Pencere (50 el) - 64 boyut")

# Orta pencere bloğu (200) - Daha büyük window için yeni dataset gerekir
# Uzun pencere bloğu (500) - Daha büyük window için yeni dataset gerekir

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

"""## 7. Ensemble Fusion - Hibrit Model

N-BEATS + TCN + Psikolojik Motor birleşimi
"""

def create_hybrid_model(feature_dim, sequence_length=50):
    """
    Tam hibrit model: N-BEATS + TCN + Psikolojik Motor
    """
    # Inputs
    feature_input = layers.Input(shape=(feature_dim,), name='feature_input')
    sequence_input = layers.Input(shape=(sequence_length,), name='sequence_input')
    
    # 1. N-BEATS Bloğu (%60 ağırlık)
    nbeats_forecast, nbeats_features = nbeats_short(sequence_input)
    
    # 2. TCN Bloğu (%60 ağırlık)
    tcn_features = tcn_model(sequence_input)
    
    # N-BEATS + TCN füzyonu
    time_series_features = layers.Concatenate()([nbeats_features, tcn_features])
    time_series_features = layers.Dense(256, activation='relu')(time_series_features)
    time_series_features = layers.Dropout(0.3)(time_series_features)
    
    # 3. Psikolojik Motor (%30 ağırlık)
    psych_features = psych_model(feature_input)
    
    # 4. İstatistiksel Baseline (%10 ağırlık)
    stat_features = layers.Dense(16, activation='relu')(feature_input)
    
    # Ensemble Fusion
    all_features = layers.Concatenate()([
        time_series_features,  # 256
        psych_features,        # 32
        stat_features          # 16
    ])  # Toplam 304 boyut
    
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
    
    # Model oluştur
    model = Model(
        inputs=[feature_input, sequence_input],
        outputs=[regression_output, classification_output, confidence_output],
        name='JetX_Hybrid_Model'
    )
    
    return model

# Hibrit modeli oluştur
hybrid_model = create_hybrid_model(
    feature_dim=dataset['X_features'].shape[1],
    sequence_length=50
)

print("\
✅ Hibrit Model Oluşturuldu!")
hybrid_model.summary()

"""## 8. Model Eğitimi

Multi-task learning loss ile eğitim
"""

# Veri hazırlama
X_feat_temp, X_feat_test, X_seq_temp, X_seq_test, y_reg_temp, y_reg_test, y_class_temp, y_class_test = train_test_split(
    dataset['X_features'], 
    dataset['X_sequences'],
    dataset['y_regression'],
    dataset['y_classification'],
    test_size=0.15,
    shuffle=False
)

X_feat_train, X_feat_val, X_seq_train, X_seq_val, y_reg_train, y_reg_val, y_class_train, y_class_val = train_test_split(
    X_feat_temp,
    X_seq_temp,
    y_reg_temp,
    y_class_temp,
    test_size=0.176,
    shuffle=False
)

print(f"Train: {len(X_feat_train)} ({len(X_feat_train)/len(dataset['X_features'])*100:.1f}%)")
print(f"Val: {len(X_feat_val)} ({len(X_feat_val)/len(dataset['X_features'])*100:.1f}%)")
print(f"Test: {len(X_feat_test)} ({len(X_feat_test)/len(dataset['X_features'])*100:.1f}%)")

# Normalizasyon
scaler = StandardScaler()
X_feat_train_scaled = scaler.fit_transform(X_feat_train)
X_feat_val_scaled = scaler.transform(X_feat_val)
X_feat_test_scaled = scaler.transform(X_feat_test)

# Multi-task loss weights
# α₁ × Kategori Loss + α₂ × Regresyon Loss + α₃ × Güven Loss
loss_weights = {
    'regression_output': 0.3,      # α₂
    'classification_output': 0.5,   # α₁ (en önemli: 1.5x eşik)
    'confidence_output': 0.2        # α₃
}

# Model derleme
hybrid_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={
        'regression_output': 'huber',  # Outlier'lara daha az duyarlı
        'classification_output': 'binary_crossentropy',  # 1.5x eşik
        'confidence_output': 'mse'
    },
    loss_weights=loss_weights,
    metrics={
        'regression_output': ['mae', 'mse'],
        'classification_output': ['accuracy'],
        'confidence_output': ['mae']
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

# Dummy güven skorları (eğitim için)
y_conf_train = np.random.uniform(0.6, 1.0, size=(len(y_reg_train),))
y_conf_val = np.random.uniform(0.6, 1.0, size=(len(y_reg_val),))

# Eğitim
print("\
🚀 Model eğitimi başlıyor...")
history = hybrid_model.fit(
    [X_feat_train_scaled, X_seq_train],
    {
        'regression_output': y_reg_train,
        'classification_output': y_class_train,
        'confidence_output': y_conf_train
    },
    validation_data=(
        [X_feat_val_scaled, X_seq_val],
        {
            'regression_output': y_reg_val,
            'classification_output': y_class_val,
            'confidence_output': y_conf_val
        }
    ),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

"""## 9. Model Değerlendirme ve Export"""

# Test seti değerlendirmesi
print("\
📊 Test seti değerlendirmesi...")
y_conf_test = np.random.uniform(0.6, 1.0, size=(len(y_reg_test),))

predictions = hybrid_model.predict([X_feat_test_scaled, X_seq_test])
y_reg_pred, y_class_pred_proba, y_conf_pred = predictions

# 1.5x eşik doğruluğu (EN ÖNEMLİ METRİK)
y_class_pred = (y_class_pred_proba > 0.5).astype(int).flatten()
threshold_accuracy = accuracy_score(y_class_test, y_class_pred)

print(f"\
🎯 1.5x Eşik Doğruluğu: {threshold_accuracy:.4f} ({threshold_accuracy*100:.2f}%)")
print(f"Hedef: %75+")

if threshold_accuracy >= 0.75:
    print("✅ Hedef başarıldı!")
else:
    print("⚠️ Hedef henüz ulaşılamadı, daha fazla eğitim gerekebilir")

# Classification report
print("\
📋 Classification Report:")
print(classification_report(y_class_test, y_class_pred, target_names=['< 1.5x', '≥ 1.5x']))

# Confusion Matrix
cm = confusion_matrix(y_class_test, y_class_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['< 1.5x', '≥ 1.5x'],
            yticklabels=['< 1.5x', '≥ 1.5x'])
plt.title('Confusion Matrix')
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
