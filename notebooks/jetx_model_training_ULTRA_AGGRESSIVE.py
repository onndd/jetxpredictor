#!/usr/bin/env python3
"""
🔥 JetX ULTRA AGGRESSIVE Model - 1.5 ALTI TAHMİN MAKİNESİ (v4.0 MONOLITHIC)

BU DOSYA TEK BAŞINA ÇALIŞIR. TÜM YARDIMCI SINIFLAR İÇİNE GÖMÜLMÜŞTÜR.

HEDEFLER:
- Normal Mod Doğruluk: %80+ (Eşik 0.85)
- Rolling Mod Doğruluk: %90+ (Eşik 0.95)
- Para Kaybı Riski: %15 altı

ULTRA AGGRESSIVE İYİLEŞTİRMELER:
- ✅ 1000 EPOCH (Uzun süreli eğitim)
- ✅ Dynamic Batch Size (GPU Memory Optimizasyonu)
- ✅ Ultra Deep Mimari (N-Beats + TCN + Fusion)
- ✅ Threshold Killer Loss (Para kaybına ağır ceza)
- ✅ 2 Modlu Yapı Entegrasyonu

Süre: ~3-5 saat (GPU ile)
"""

import subprocess
import sys
import os
import time
from datetime import datetime
import json
import shutil
import pickle
import warnings
import math
import random

# Uyarıları kapat
warnings.filterwarnings('ignore')

print("="*80)
print("🔥 JetX ULTRA AGGRESSIVE TRAINING (v4.0 MONOLITHIC)")
print("="*80)
print(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()
print("🔧 SİSTEM KONFIGURASYONU:")
print("   Normal Mod Eşik: 0.85")
print("   Rolling Mod Eşik: 0.95")
print("   Mimari: Ultra Deep Hybrid")
print()

# -----------------------------------------------------------------------------
# 1. KÜTÜPHANE KURULUMU VE İMPORTLAR
# -----------------------------------------------------------------------------
print("📦 Kütüphaneler kontrol ediliyor...")
required_packages = [
    "tensorflow", "scikit-learn", "pandas", "numpy", 
    "scipy", "joblib", "matplotlib", "seaborn", "tqdm"
]

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"   ⬇️ {package} kuruluyor...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

import numpy as np
import pandas as pd
import joblib
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report, confusion_matrix
from tqdm.auto import tqdm

# Proje kök dizinini ayarla
if not os.path.exists('jetxpredictor') and not os.path.exists('jetx_data.db'):
    print("\n📥 Proje klonlanıyor...")
    subprocess.check_call(["git", "clone", "https://github.com/onndd/jetxpredictor.git"])
    os.chdir('jetxpredictor')

sys.path.append(os.getcwd())

# Feature Engineering (Veritabanı yapısı karmaşık olduğu için import ediyoruz)
try:
    from category_definitions import CategoryDefinitions, FeatureEngineering
except ImportError:
    # Fallback Class (Eğer import edilemezse)
    class FeatureEngineering:
        @staticmethod
        def extract_all_features(history):
            features = {}
            if not history: return features
            features['mean_50'] = np.mean(history[-50:]) if len(history) >= 50 else np.mean(history)
            features['std_50'] = np.std(history[-50:]) if len(history) >= 50 else np.std(history)
            return features
            
    class CategoryDefinitions:
        CRITICAL_THRESHOLD = 1.5
        @staticmethod
        def get_category_numeric(val): return 0 if val < 1.5 else 1

# GPU Ayarları
print("\n🚀 GPU Ayarları Yapılandırılıyor...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        print(f"✅ GPU Aktif: {len(gpus)} adet (Mixed Precision)")
    except RuntimeError as e:
        print(f"⚠️ GPU Hatası: {e}")
else:
    print("⚠️ GPU Bulunamadı! CPU modunda devam ediliyor.")

# Kritik Eşikler
THRESHOLD_NORMAL = 0.85
THRESHOLD_ROLLING = 0.95

# =============================================================================
# 2. GÖMÜLÜ YARDIMCI MODÜLLER (TEK DOSYA MANTIGI)
# =============================================================================

# --- A. CUSTOM LOSS FUNCTIONS ---
def percentage_aware_regression_loss(y_true, y_pred):
    """Yüzde hataya dayalı regression loss"""
    epsilon = K.epsilon()
    percentage_error = K.abs(y_true - y_pred) / (K.abs(y_true) + epsilon)
    high_value_weight = tf.where(y_true >= 5.0, 1.2, 1.0)
    weighted_percentage_error = percentage_error * high_value_weight
    return K.mean(weighted_percentage_error)

def balanced_threshold_killer_loss(y_true, y_pred):
    """1.5 altı yanlış tahmine DENGELI CEZA"""
    mae = K.abs(y_true - y_pred)
    # 1.5 altıyken üstü tahmin = 12x ceza (PARA KAYBI!)
    false_positive = K.cast(tf.logical_and(y_true < 1.5, y_pred >= 1.5), 'float32') * 12.0
    # 1.5 üstüyken altı tahmin = 6x ceza
    false_negative = K.cast(tf.logical_and(y_true >= 1.5, y_pred < 1.5), 'float32') * 6.0
    # Kritik bölge (1.4-1.6) = 10x ceza
    critical_zone = K.cast(tf.logical_and(y_true >= 1.4, y_true <= 1.6), 'float32') * 10.0
    
    weight = K.maximum(K.maximum(false_positive, false_negative), critical_zone)
    weight = K.maximum(weight, 1.0)
    return K.mean(mae * weight)

def balanced_focal_loss(gamma=3.0, alpha=0.85):
    """Ultra Aggressive Focal Loss"""
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = alpha * K.pow(1 - pt, gamma)
        return -K.mean(focal_weight * K.log(pt))
    return loss

def create_weighted_binary_crossentropy(weight_0, weight_1):
    """Ağırlıklı Binary Crossentropy"""
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        bce = -(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        weights = y_true * weight_1 + (1 - y_true) * weight_0
        return K.mean(bce * weights)
    return loss

# --- B. CUSTOM CALLBACKS ---
class UltraMetricsCallback(callbacks.Callback):
    """2 Modlu (Normal/Rolling) Performans Raporu"""
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.best_below_acc = 0
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 != 0: return
        
        # Tahmin (Validasyon Seti)
        # Not: X_val bir liste: [X_f, X_50, X_200, X_500, X_1000]
        p = self.model.predict(self.X_val, verbose=0)[2].flatten()
        t = self.y_val.flatten().astype(int) # y_thr
        
        # Normal Mod (0.85)
        p_norm = (p >= THRESHOLD_NORMAL).astype(int)
        acc_norm = accuracy_score(t, p_norm)
        
        # Rolling Mod (0.95)
        p_roll = (p >= THRESHOLD_ROLLING).astype(int)
        acc_roll = accuracy_score(t, p_roll)
        
        # 1.5 Altı Doğruluğu (Normal Mod için)
        below_mask = t == 0
        below_acc = accuracy_score(t[below_mask], p_norm[below_mask]) if below_mask.sum() > 0 else 0
        
        # Para Kaybı Riski (Normal Mod)
        false_positive = ((p_norm == 1) & (t == 0)).sum()
        risk = false_positive / below_mask.sum() if below_mask.sum() > 0 else 0
        
        print(f"\n📊 Epoch {epoch+1} ULTRA REPORT:")
        print(f"   🎯 Normal Mod ({THRESHOLD_NORMAL}): {acc_norm:.2%} (1.5 Altı Acc: {below_acc:.2%})")
        print(f"   🚀 Rolling Mod ({THRESHOLD_ROLLING}): {acc_roll:.2%}")
        print(f"   💰 Para Kaybı Riski: {risk:.2%} (Hedef: <15%)")
        
        if below_acc > self.best_below_acc:
            self.best_below_acc = below_acc
            print(f"   ✨ YENİ REKOR! En iyi 1.5 altı: {below_acc:.2%}")

def lr_schedule(epoch, lr):
    """Learning Rate Schedule"""
    if epoch < 50: return 0.00005
    elif epoch < 150: return 0.000025
    elif epoch < 300: return 0.000005
    else: return 0.0000025

# -----------------------------------------------------------------------------
# 3. VERİ YÜKLEME VE HAZIRLIK
# -----------------------------------------------------------------------------
print("\n📊 Veri yükleniyor...")
conn = sqlite3.connect('jetx_data.db')
data = pd.read_sql_query("SELECT value FROM jetx_results ORDER BY id", conn)
conn.close()

all_values = data['value'].values
cleaned_values = []
for val in all_values:
    try:
        val_str = str(val).replace('\u2028', '').replace('\u2029', '').strip()
        if ' ' in val_str: val_str = val_str.split()[0]
        cleaned_values.append(float(val_str))
    except:
        continue
all_values = np.array(cleaned_values)

print(f"✅ {len(all_values):,} veri yüklendi")

# Feature Extraction Loop
print("\n🔧 Feature extraction (Ultra)...")
window_size = 1000 
X_f, X_50, X_200, X_500, X_1000 = [], [], [], [], []
y_reg, y_cls, y_thr = [], [], []

for i in tqdm(range(window_size, len(all_values)-1), desc='Features'):
    hist = all_values[:i].tolist()
    target = all_values[i]
    
    feats = FeatureEngineering.extract_all_features(hist)
    X_f.append(list(feats.values()))
    
    X_50.append(all_values[i-50:i])
    X_200.append(all_values[i-200:i])
    X_500.append(all_values[i-500:i])
    X_1000.append(all_values[i-1000:i])
    
    y_reg.append(target)
    
    cat = CategoryDefinitions.get_category_numeric(target)
    onehot = np.zeros(3)
    onehot[cat] = 1
    y_cls.append(onehot)
    
    y_thr.append(1.0 if target >= 1.5 else 0.0)

# Numpy dönüşümü
X_f = np.array(X_f)
X_50 = np.array(X_50).reshape(-1, 50, 1)
X_200 = np.array(X_200).reshape(-1, 200, 1)
X_500 = np.array(X_500).reshape(-1, 500, 1)
X_1000 = np.array(X_1000).reshape(-1, 1000, 1)
y_reg = np.array(y_reg)
y_cls = np.array(y_cls)
y_thr = np.array(y_thr).reshape(-1, 1)

print(f"✅ {len(X_f):,} örnek hazırlandı")

# Normalizasyon
scaler = StandardScaler()
X_f = scaler.fit_transform(X_f)
X_50 = np.log10(X_50 + 1e-8)
X_200 = np.log10(X_200 + 1e-8)
X_500 = np.log10(X_500 + 1e-8)
X_1000 = np.log10(X_1000 + 1e-8)

# Kronolojik Split
test_size = 1500
val_size = 1000
train_size = len(X_f) - test_size - val_size

# Train
X_f_tr = X_f[:train_size]
X_50_tr = X_50[:train_size]
X_200_tr = X_200[:train_size]
X_500_tr = X_500[:train_size]
X_1000_tr = X_1000[:train_size]
y_reg_tr = y_reg[:train_size]
y_cls_tr = y_cls[:train_size]
y_thr_tr = y_thr[:train_size]

# Validation
X_f_val = X_f[train_size:train_size+val_size]
X_50_val = X_50[train_size:train_size+val_size]
X_200_val = X_200[train_size:train_size+val_size]
X_500_val = X_500[train_size:train_size+val_size]
X_1000_val = X_1000[train_size:train_size+val_size]
y_reg_val = y_reg[train_size:train_size+val_size]
y_cls_val = y_cls[train_size:train_size+val_size]
y_thr_val = y_thr[train_size:train_size+val_size]

# Test
X_f_te = X_f[train_size+val_size:]
X_50_te = X_50[train_size+val_size:]
X_200_te = X_200[train_size+val_size:]
X_500_te = X_500[train_size+val_size:]
X_1000_te = X_1000[train_size+val_size:]
y_reg_te = y_reg[train_size+val_size:]
y_cls_te = y_cls[train_size+val_size:]
y_thr_te = y_thr[train_size+val_size:]

# -----------------------------------------------------------------------------
# 4. ULTRA DEEP MODEL MİMARİSİ (GÖMÜLÜ)
# -----------------------------------------------------------------------------
def ultra_nbeats(x, units, blocks):
    for _ in range(blocks):
        x = layers.Dense(units, activation='relu', kernel_regularizer='l2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
    return x

def ultra_tcn_block(x, filters, dilation):
    conv = layers.Conv1D(filters, 3, dilation_rate=dilation, padding='causal', activation='relu', kernel_regularizer='l2')(x)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Dropout(0.2)(conv)
    residual = layers.Conv1D(filters, 1, padding='same')(x) if x.shape[-1] != filters else x
    return layers.Add()([conv, residual])

inp_f = layers.Input((X_f.shape[1],), name='features')
inp_50 = layers.Input((50, 1), name='seq50')
inp_200 = layers.Input((200, 1), name='seq200')
inp_500 = layers.Input((500, 1), name='seq500')
inp_1000 = layers.Input((1000, 1), name='seq1000')

# N-BEATS
nb_s = ultra_nbeats(layers.Flatten()(inp_50), 128, 7)
nb_m = ultra_nbeats(layers.Flatten()(inp_200), 256, 10)
nb_l = ultra_nbeats(layers.Flatten()(inp_500), 512, 12)
nb_xl = ultra_nbeats(layers.Flatten()(inp_1000), 512, 12)
nb_all = layers.Concatenate()([nb_s, nb_m, nb_l, nb_xl])

# TCN
tcn = inp_500
tcn = ultra_tcn_block(tcn, 128, 1)
tcn = ultra_tcn_block(tcn, 256, 4)
tcn = ultra_tcn_block(tcn, 512, 16)
tcn = ultra_tcn_block(tcn, 512, 64)
tcn = ultra_tcn_block(tcn, 1024, 256)
tcn = layers.GlobalAveragePooling1D()(tcn)
tcn = layers.Dense(1024, activation='relu')(tcn)
tcn = layers.Dropout(0.25)(tcn)

# FUSION (3x Derin)
fus = layers.Concatenate()([inp_f, nb_all, tcn])
fus = layers.Dense(1024, activation='relu', kernel_regularizer='l2')(fus)
fus = layers.BatchNormalization()(fus)
fus = layers.Dropout(0.3)(fus)
fus = layers.Dense(512, activation='relu')(fus)
fus = layers.BatchNormalization()(fus)
fus = layers.Dropout(0.3)(fus)
fus = layers.Dense(256, activation='relu')(fus)
fus = layers.Dropout(0.25)(fus)
fus = layers.Dense(128, activation='relu')(fus)

# OUTPUTS
reg_branch = layers.Dense(128, activation='relu')(fus)
out_reg = layers.Dense(1, activation='linear', name='regression')(reg_branch)

cls_branch = layers.Dense(128, activation='relu')(fus)
out_cls = layers.Dense(3, activation='softmax', name='classification')(cls_branch)

thr_branch = layers.Dense(64, activation='relu')(fus)
out_thr = layers.Dense(1, activation='sigmoid', name='threshold')(thr_branch)

model = models.Model([inp_f, inp_50, inp_200, inp_500, inp_1000], [out_reg, out_cls, out_thr])
print(f"✅ ULTRA DEEP Model: {model.count_params():,} parametre")

# -----------------------------------------------------------------------------
# 5. EĞİTİM KONFIGURASYONU
# -----------------------------------------------------------------------------
# Class Weight Hesaplama
c0 = (y_thr_tr.flatten() == 0).sum()
c1 = (y_thr_tr.flatten() == 1).sum()
TARGET_MULTIPLIER = 7.0 
w0 = (len(y_thr_tr) / (2 * c0)) * TARGET_MULTIPLIER
w1 = len(y_thr_tr) / (2 * c1)

print(f"\n🎯 CLASS WEIGHTS: 1.5 altı (0): {w0:.2f}x | 1.5 üstü (1): {w1:.2f}x")

# Dynamic Batch Size
try:
    if gpus:
        # GPU Memory test
        start = time.time()
        model.predict([X_f_tr[:32], X_50_tr[:32], X_200_tr[:32], X_500_tr[:32], X_1000_tr[:32]], verbose=0)
        if time.time() - start < 0.5:
            optimal_batch_size = 64
        else:
            optimal_batch_size = 32
    else:
        optimal_batch_size = 32
except:
    optimal_batch_size = 16

print(f"🎯 Optimal Batch Size: {optimal_batch_size}")

# Compile
model.compile(
    optimizer=Adam(0.00005),
    loss={
        'regression': balanced_threshold_killer_loss,
        'classification': 'categorical_crossentropy',
        'threshold': balanced_focal_loss()
    },
    loss_weights={'regression': 0.25, 'classification': 0.15, 'threshold': 0.60},
    metrics={'threshold': ['accuracy']}
)

# Callbacks
callbacks_list = [
    callbacks.ModelCheckpoint('jetx_ultra_best.h5', monitor='val_threshold_accuracy', save_best_only=True, mode='max', verbose=1),
    callbacks.EarlyStopping(monitor='val_threshold_accuracy', patience=50, mode='max', restore_best_weights=True),
    callbacks.LearningRateScheduler(lr_schedule),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-8),
    UltraMetricsCallback([X_f_val, X_50_val, X_200_val, X_500_val, X_1000_val], y_thr_val)
]

# -----------------------------------------------------------------------------
# 6. EĞİTİM (1000 Epoch)
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("🔥 ULTRA AGGRESSIVE TRAINING BAŞLIYOR (1000 Epoch)")
print("="*70)

hist = model.fit(
    [X_f_tr, X_50_tr, X_200_tr, X_500_tr, X_1000_tr],
    {'regression': y_reg_tr, 'classification': y_cls_tr, 'threshold': y_thr_tr},
    validation_data=(
        [X_f_val, X_50_val, X_200_val, X_500_val, X_1000_val],
        {'regression': y_reg_val, 'classification': y_cls_val, 'threshold': y_thr_val}
    ),
    epochs=1000,
    batch_size=optimal_batch_size,
    shuffle=False,
    callbacks=callbacks_list,
    verbose=1
)

# -----------------------------------------------------------------------------
# 7. FİNAL DEĞERLENDİRME VE KASA SİMÜLASYONU
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("📊 TEST SETİ DEĞERLENDİRMESİ")
print("="*70)

model.load_weights('jetx_ultra_best.h5')
pred = model.predict([X_f_te, X_50_te, X_200_te, X_500_te, X_1000_te], verbose=0)
p_reg = pred[0].flatten()
p_thr = pred[2].flatten()

# 2 Modlu Analiz
y_true_cls = (y_reg_te >= 1.5).astype(int)
p_norm = (p_thr >= THRESHOLD_NORMAL).astype(int)
p_roll = (p_thr >= THRESHOLD_ROLLING).astype(int)

acc_norm = accuracy_score(y_true_cls, p_norm)
acc_roll = accuracy_score(y_true_cls, p_roll)
mae = mean_absolute_error(y_reg_te, p_reg)

print(f"\n📈 Regression MAE: {mae:.4f}")
print(f"🎯 Normal Mod ({THRESHOLD_NORMAL}) Accuracy: {acc_norm:.2%}")
print(f"🚀 Rolling Mod ({THRESHOLD_ROLLING}) Accuracy: {acc_roll:.2%}")

# Çift Kasa Simülasyonu
initial_bankroll = 1000.0
bet_amount = 10.0

# Kasa 1 (Normal - Dinamik)
w1, b1, w_cnt1 = initial_bankroll, 0, 0
for i in range(len(y_reg_te)):
    if p_thr[i] >= THRESHOLD_NORMAL:
        w1 -= bet_amount
        b1 += 1
        exit_pt = min(max(1.5, p_reg[i] * 0.8), 2.5)
        if y_reg_te[i] >= exit_pt:
            w1 += exit_pt * bet_amount
            w_cnt1 += 1
roi1 = (w1 - initial_bankroll) / initial_bankroll * 100
wr1 = (w_cnt1 / b1 * 100) if b1 > 0 else 0
print(f"\n💰 KASA 1 (NORMAL): ROI {roi1:+.2f}% | Win Rate {wr1:.1f}%")

# Kasa 2 (Rolling - Güvenli)
w2, b2, w_cnt2 = initial_bankroll, 0, 0
for i in range(len(y_reg_te)):
    if p_thr[i] >= THRESHOLD_ROLLING:
        w2 -= bet_amount
        b2 += 1
        if y_reg_te[i] >= 1.5:
            w2 += 1.5 * bet_amount
            w_cnt2 += 1
roi2 = (w2 - initial_bankroll) / initial_bankroll * 100
wr2 = (w_cnt2 / b2 * 100) if b2 > 0 else 0
print(f"💰 KASA 2 (ROLLING): ROI {roi2:+.2f}% | Win Rate {wr2:.1f}%")

# -----------------------------------------------------------------------------
# 8. KAYDET VE İNDİR
# -----------------------------------------------------------------------------
print("\n💾 Dosyalar kaydediliyor...")
os.makedirs('models', exist_ok=True)
model.save('jetx_ultra_model.h5')
joblib.dump(scaler, 'scaler_ultra.pkl')

info = {
    'model': 'ULTRA_AGGRESSIVE_NBEATS_TCN_MONOLITH',
    'version': '4.0',
    'metrics': {'normal_acc': float(acc_norm), 'rolling_acc': float(acc_roll), 'mae': float(mae)},
    'simulation': {'normal_roi': float(roi1), 'rolling_roi': float(roi2)}
}
with open('ultra_model_info.json', 'w') as f: json.dump(info, f, indent=2)

# Zip ve İndir
zip_name = 'jetx_ultra_model_v4.0.zip'
shutil.make_archive('jetx_ultra_model_v4.0', 'zip', '.')

try:
    from google.colab import files
    files.download(zip_name)
except:
    print(f"⚠️ Manuel indirin: {zip_name}")

print("\n🎉 ULTRA AGGRESSIVE TRAINING TAMAMLANDI!")
print("="*80)
