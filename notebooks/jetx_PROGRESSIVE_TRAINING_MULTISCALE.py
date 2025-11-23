#!/usr/bin/env python3
"""
🎯 JetX PROGRESSIVE TRAINING - MULTI-SCALE WINDOW ENSEMBLE (v3.1)

YENİ YAKLAŞIM: Multi-Scale Window Ensemble
- Her pencere boyutu için ayrı model eğitimi
- Window boyutları: [500, 250, 100, 50, 20]
- Her model farklı zaman ölçeğinde desen öğrenir
- Final: Tüm modellerin ensemble'ı

GÜNCELLEME (v3.1):
- 2 MODLU YAPI: Normal (0.85) ve Rolling (0.95)
- Sanal kasalar bu modlara göre optimize edildi.

HEDEFLER:
- Normal Mod Doğruluk: %80+
- Rolling Mod Doğruluk: %90+
- MAE: < 2.0

⚠️  VERİ BÜTÜNLİĞİ:
- Shuffle: YASAK
- Augmentation: YASAK
- Kronolojik sıra: KORUNUYOR
"""

import subprocess
import sys
import os
import time
from datetime import datetime
import json
import shutil
from pathlib import Path
import pickle

# XLA optimizasyonu devre dışı (stabilite için)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("="*80)
print("🎯 JetX PROGRESSIVE TRAINING - MULTI-SCALE WINDOW ENSEMBLE (v3.1)")
print("="*80)
print(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()
print("🔧 SİSTEM KONFIGURASYONU:")
print("   Normal Mod Eşik: 0.85")
print("   Rolling Mod Eşik: 0.95")
print("   Window Boyutları: [500, 250, 100, 50, 20]")
print()

# Kütüphaneleri yükle
print("📦 Kütüphaneler yükleniyor...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                      "tensorflow", "scikit-learn", "pandas", "numpy", 
                      "scipy", "joblib", "matplotlib", "seaborn", "tqdm",
                      "PyWavelets", "nolds"])

import numpy as np
import pandas as pd
import joblib
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, backend as K
from tensorflow.keras.optimizers import Adam
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# EŞİKLER
THRESHOLD_NORMAL = 0.85
THRESHOLD_ROLLING = 0.95

# =============================================================================
# GPU OPTIMIZER ENTEGRASYONU
# =============================================================================
try:
    # GPU optimizer'ı import et
    from utils.gpu_optimizer import setup_colab_gpu_optimization, get_gpu_optimizer
    
    print("\n🚀 GPU OPTİMİZASYONU BAŞLATILIYOR...")
    gpu_results = setup_colab_gpu_optimization()
    
    # GPU optimizer instance
    gpu_optimizer = get_gpu_optimizer()
    
    # GPU monitoring
    print("📊 GPU performansı izleniyor...")
    gpu_optimizer.monitor_gpu_usage(duration_seconds=3)
    
except ImportError as e:
    print(f"⚠️ GPU optimizer import edilemedi: {e}")
    gpu_optimizer = None
except Exception as e:
    print(f"⚠️ GPU optimizasyonu başarısız: {e}")
    gpu_optimizer = None

# GPU konfigürasyonu
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Memory growth ayarla - GPU belleğini dinamik kullan
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Mixed precision training - GPU performansını artırır
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        
        print(f"✅ TensorFlow: {tf.__version__}")
        print(f"✅ GPU: {len(gpus)} GPU bulundu ve yapılandırıldı")
        print(f"   - Memory growth: Aktif")
        print(f"   - Mixed precision: Aktif (float16)")
        print(f"   - GPU'lar: {[gpu.name for gpu in gpus]}")
        
        # GPU optimizer entegrasyonu
        if gpu_optimizer:
            try:
                gpu_optimizer.optimize_tensorflow()
            except Exception as e:
                print(f"⚠️ TensorFlow GPU optimizasyonu başarısız: {e}")
        
    except RuntimeError as e:
        print(f"⚠️ GPU konfigürasyon hatası: {e}")
        print(f"✅ TensorFlow: {tf.__version__}")
        print(f"✅ GPU: Mevcut ama CPU modunda çalışacak")
else:
    print(f"✅ TensorFlow: {tf.__version__}")
    print(f"⚠️ GPU: Bulunamadı - CPU modunda çalışacak")
    # CPU fallback için gpu optimizer'ı hala çağırabiliriz
    if gpu_optimizer:
        print("ℹ️ GPU optimizer CPU fallback mekanizmalarını çalıştırıyor...")

# Proje yükle ve kök dizini tespit et
PROJECT_ROOT = None

# Önce mevcut dizini kontrol et
if os.path.exists('jetx_data.db'):
    PROJECT_ROOT = os.getcwd()
    print("\n✅ Proje kök dizini tespit edildi (mevcut dizin)")
elif os.path.exists('jetxpredictor/jetx_data.db'):
    PROJECT_ROOT = os.path.join(os.getcwd(), 'jetxpredictor')
    print(f"\n✅ Proje kök dizini tespit edildi: {PROJECT_ROOT}")
else:
    # Yoksa klonla
    print("\n📥 Proje klonlanıyor...")
    subprocess.check_call(["git", "clone", "https://github.com/onndd/jetxpredictor.git"])
    PROJECT_ROOT = os.path.join(os.getcwd(), 'jetxpredictor')
    print(f"✅ Proje klonlandı: {PROJECT_ROOT}")

# sys.path'e ekle (chdir YAPMA!)
sys.path.insert(0, PROJECT_ROOT)
print(f"📂 Çalışma dizini: {os.getcwd()}")
print(f"📂 Proje kök dizini: {PROJECT_ROOT}")

from category_definitions import CategoryDefinitions, FeatureEngineering
from utils.multi_scale_window import MultiScaleWindowExtractor, MultiScaleEnsemble, split_data_preserving_order
from utils.custom_losses import percentage_aware_regression_loss, balanced_focal_loss, create_weighted_binary_crossentropy
from utils.adaptive_lr_scheduler import AdaptiveLearningRateScheduler, CosineAnnealingSchedule, LearningRateSchedulerFactory
print(f"✅ Proje yüklendi - Kritik eşik: {CategoryDefinitions.CRITICAL_THRESHOLD}x\n")

# =============================================================================
# VERİ YÜKLEME (SIRA KORUNARAK)
# =============================================================================
print("📊 Veri yükleniyor...")
db_path = os.path.join(PROJECT_ROOT, 'jetx_data.db')
conn = sqlite3.connect(db_path)
data = pd.read_sql_query("SELECT value FROM jetx_results ORDER BY id", conn)
conn.close()

# String verileri float'a çevir - Unicode karakterleri temizle (DÜZELTME: Index kayması önlendi)
all_values = data['value'].values

# Unicode karakterlerini ve bozuk verileri temizle - DÜZELTME: Index korunuyor
cleaned_values = []
skipped_indices = []  # Atlanan indexleri takip et
for i, val in enumerate(all_values):
    try:
        # String'i temizle - Unicode satır ayırıcılarını ve diğer bozuk karakterleri kaldır
        val_str = str(val).replace('\u2028', '').replace('\u2029', '').strip()
        # Birden fazla sayı varsa (örn: "2.29 1.29") ilkini al
        if ' ' in val_str:
            val_str = val_str.split()[0]
        # Float'a çevir
        cleaned_values.append(float(val_str))
    except (ValueError, TypeError) as e:
        skipped_indices.append(i)  # Index'i kaydet
        print(f"⚠️ Satır {i} atlandı - bozuk veri: '{val}' - Hata: {e}")
        continue

all_values = np.array(cleaned_values)
print(f"✅ {len(all_values):,} veri yüklendi", end="")
if len(skipped_indices) > 0:
    print(f" ({len(skipped_indices)} bozuk satır atlandı - indexler: {skipped_indices[:5]}{'...' if len(skipped_indices) > 5 else ''})")
else:
    print()
print(f"Aralık: {all_values.min():.2f}x - {all_values.max():.2f}x")

below = (all_values < 1.5).sum()
above = (all_values >= 1.5).sum()
print(f"\n📊 CLASS DAĞILIMI:")
print(f"  1.5 altı: {below:,} ({below/len(all_values)*100:.1f}%)")
print(f"  1.5 üstü: {above:,} ({above/len(all_values)*100:.1f}%)")
print(f"  Dengesizlik: 1:{above/below:.2f}")

# =============================================================================
# TIME-SERIES SPLIT (SHUFFLE YOK!)
# =============================================================================
print("\n📊 TIME-SERIES SPLIT (Kronolojik)...")
train_data, val_data, test_data = split_data_preserving_order(
    all_values,
    train_ratio=0.70,
    val_ratio=0.15
)

# =============================================================================
# MULTI-SCALE FEATURE ENGINEERING
# =============================================================================
print("\n🔧 MULTI-SCALE FEATURE EXTRACTION...")
print("🔹 Her pencere boyutu için feature engineering")

window_sizes = [500, 250, 100, 50, 20]

def extract_features_for_window(data, window_size, start_idx=None, end_idx=None):
    """
    Belirli bir pencere boyutu için feature extraction
    
    Args:
        data: Input veri
        window_size: Pencere boyutu
        start_idx: Başlangıç indeksi (None ise window_size'den başlar)
        end_idx: Bitiş indeksi (None ise veri sonuna kadar)
    """
    X_features = []
    X_sequences = []
    y_regression = []
    y_classification = []
    y_threshold = []
    
    # Başlangıç ve bitiş indekslerini belirle
    if start_idx is None:
        start_idx = window_size
    if end_idx is None:
        end_idx = len(data) - 1
    
    for i in tqdm(range(start_idx, end_idx), desc=f'Window {window_size}'):
        hist = data[:i].tolist()
        target = data[i]
        
        # Feature engineering
        feats = FeatureEngineering.extract_all_features(hist)
        X_features.append(list(feats.values()))
        
        # Sequence (son window_size değer)
        sequence = data[i-window_size:i]
        X_sequences.append(sequence)
        
        # Targets
        y_regression.append(target)
        
        # Classification (3 sınıf)
        cat = CategoryDefinitions.get_category_numeric(target)
        onehot = np.zeros(3)
        onehot[cat] = 1
        y_classification.append(onehot)
        
        # Threshold (1.5 altı/üstü)
        y_threshold.append(1.0 if target >= 1.5 else 0.0)
    
    X_features = np.array(X_features)
    X_sequences = np.array(X_sequences).reshape(-1, window_size, 1)
    y_regression = np.array(y_regression)
    y_classification = np.array(y_classification)
    y_threshold = np.array(y_threshold).reshape(-1, 1)
    
    return X_features, X_sequences, y_regression, y_classification, y_threshold

# Her window boyutu için feature extraction
all_data_by_window = {}

# En büyük pencere boyutu (500) için test başlangıç indeksini hesapla
max_window = max(window_sizes)
test_start_idx = max_window  # En büyük pencere boyutu kadar offset

for window_size in window_sizes:
    print(f"\n🔧 Window {window_size} için feature extraction...")
    
    # Train data
    X_f_train, X_seq_train, y_reg_train, y_cls_train, y_thr_train = extract_features_for_window(
        train_data, window_size
    )
    
    # Val data
    X_f_val, X_seq_val, y_reg_val, y_cls_val, y_thr_val = extract_features_for_window(
        val_data, window_size
    )
    
    # Test data - TÜM MODELLER İÇİN AYNI BAŞLANGIÇ İNDEKSİ
    # Bu, ensemble için tutarlı tahmin uzunlukları sağlar
    X_f_test, X_seq_test, y_reg_test, y_cls_test, y_thr_test = extract_features_for_window(
        test_data, window_size, start_idx=test_start_idx
    )
    
    # Normalizasyon
    scaler = StandardScaler()
    X_f_train = scaler.fit_transform(X_f_train)
    X_f_val = scaler.transform(X_f_val)
    X_f_test = scaler.transform(X_f_test)
    
    # Log-scale sequences
    X_seq_train = np.log10(X_seq_train + 1e-8)
    X_seq_val = np.log10(X_seq_val + 1e-8)
    X_seq_test = np.log10(X_seq_test + 1e-8)
    
    all_data_by_window[window_size] = {
        'train': (X_f_train, X_seq_train, y_reg_train, y_cls_train, y_thr_train),
        'val': (X_f_val, X_seq_val, y_reg_val, y_cls_val, y_thr_val),
        'test': (X_f_test, X_seq_test, y_reg_test, y_cls_test, y_thr_test),
        'scaler': scaler
    }
    
    print(f"✅ Window {window_size}: {len(X_f_train):,} train, {len(X_f_val):,} val, {len(X_f_test):,} test")

# =============================================================================
# MODEL MİMARİSİ (HER PENCERE İÇİN AYRI)
# =============================================================================
def build_model_for_window(window_size, n_features):
    """
    Belirli bir pencere boyutu için model oluştur
    Her pencere boyutu kendi modeline sahip
    """
    # Inputs
    inp_features = layers.Input((n_features,), name='features')
    inp_sequence = layers.Input((window_size, 1), name='sequence')
    
    # Feature branch
    x_feat = layers.Dense(256, activation='relu', kernel_regularizer='l2')(inp_features)
    x_feat = layers.BatchNormalization()(x_feat)
    x_feat = layers.Dropout(0.3)(x_feat)
    x_feat = layers.Dense(128, activation='relu')(x_feat)
    x_feat = layers.Dropout(0.2)(x_feat)
    
    # Sequence branch - pencere boyutuna göre adapte
    if window_size <= 50:
        # Küçük pencere: Basit LSTM
        x_seq = layers.LSTM(64, return_sequences=False)(inp_sequence)
        x_seq = layers.Dropout(0.2)(x_seq)
    elif window_size <= 100:
        # Orta pencere: 2-layer LSTM
        x_seq = layers.LSTM(128, return_sequences=True)(inp_sequence)
        x_seq = layers.Dropout(0.2)(x_seq)
        x_seq = layers.LSTM(64, return_sequences=False)(x_seq)
        x_seq = layers.Dropout(0.2)(x_seq)
    else:
        # Büyük pencere: 3-layer LSTM + Attention
        x_seq = layers.LSTM(256, return_sequences=True)(inp_sequence)
        x_seq = layers.Dropout(0.2)(x_seq)
        x_seq = layers.LSTM(128, return_sequences=True)(x_seq)
        x_seq = layers.Dropout(0.2)(x_seq)
        
        # Attention - Lambda yerine GlobalAveragePooling1D kullan
        attention = layers.Dense(1, activation='tanh')(x_seq)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(128)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        x_seq_attended = layers.Multiply()([x_seq, attention])
        x_seq = layers.GlobalAveragePooling1D()(x_seq_attended)
        x_seq = layers.Dense(128, activation='linear', use_bias=False)(x_seq)
        x_seq = layers.Dropout(0.2)(x_seq)
    
    # Fusion
    fusion = layers.Concatenate()([x_feat, x_seq])
    fusion = layers.Dense(256, activation='relu', kernel_regularizer='l2')(fusion)
    fusion = layers.BatchNormalization()(fusion)
    fusion = layers.Dropout(0.3)(fusion)
    fusion = layers.Dense(128, activation='relu')(fusion)
    fusion = layers.Dropout(0.2)(fusion)
    
    # Outputs
    # Regression
    reg_branch = layers.Dense(64, activation='relu')(fusion)
    reg_branch = layers.Dropout(0.2)(reg_branch)
    out_reg = layers.Dense(1, activation='linear', name='regression')(reg_branch)
    
    # Classification (3 sınıf)
    cls_branch = layers.Dense(64, activation='relu')(fusion)
    cls_branch = layers.Dropout(0.2)(cls_branch)
    out_cls = layers.Dense(3, activation='softmax', name='classification')(cls_branch)
    
    # Threshold (1.5 altı/üstü)
    thr_branch = layers.Dense(32, activation='relu')(fusion)
    thr_branch = layers.Dropout(0.2)(thr_branch)
    out_thr = layers.Dense(1, activation='sigmoid', name='threshold')(thr_branch)
    
    model = models.Model([inp_features, inp_sequence], [out_reg, out_cls, out_thr])
    
    return model

# =============================================================================
# DETAYLI EPOCH CALLBACK (2 MODLU)
# =============================================================================
class DetailedMetricsCallback(callbacks.Callback):
    """
    Her epoch sonunda detaylı metrikler gösterir (Normal ve Rolling Mod için)
    """
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
    
    def simulate_bankroll(self, predictions, actuals, threshold):
        """Basit kasa simülasyonu"""
        initial = 10000
        wallet = initial
        wins = 0
        total_bets = 0
        for pred, actual in zip(predictions, actuals):
            if pred >= threshold:
                wallet -= 10
                total_bets += 1
                if actual >= 1.5:
                    wallet += 15
                    wins += 1
        roi = ((wallet - initial) / initial) * 100 if total_bets > 0 else 0
        win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
        return roi, win_rate, wins, total_bets
    
    def on_epoch_end(self, epoch, logs=None):
        # Tahminler yap
        preds = self.model.predict(self.X_val, verbose=0)
        threshold_preds = preds[2].flatten()
        
        # Gerçek Değerler
        y_true = (self.y_val >= 1.5).astype(int)
        
        # NORMAL MOD (0.85) Analizi
        y_pred_normal = (threshold_preds >= THRESHOLD_NORMAL).astype(int)
        acc_normal = accuracy_score(y_true, y_pred_normal) * 100
        
        # ROLLING MOD (0.95) Analizi
        y_pred_rolling = (threshold_preds >= THRESHOLD_ROLLING).astype(int)
        acc_rolling = accuracy_score(y_true, y_pred_rolling) * 100
        
        # ROI Hesapla (Normal Mod üzerinden)
        roi, win_rate, wins, total_bets = self.simulate_bankroll(threshold_preds, self.y_val, THRESHOLD_NORMAL)
        
        # Detaylı çıktı
        print(f"\n{'='*80}")
        print(f"📊 EPOCH {epoch+1} - PERFORMANS RAPORU")
        print(f"⚖️  Normal Mod ({THRESHOLD_NORMAL}):   {acc_normal:6.2f}%")
        print(f"🚀 Rolling Mod ({THRESHOLD_ROLLING}):   {acc_rolling:6.2f}%")
        print(f"💵 ROI (Normal):         {roi:+7.2f}%")
        print(f"📈 Win Rate (Normal):    {win_rate:6.2f}%  ({wins}/{total_bets})")
        print(f"📉 Loss:                 val_loss={logs.get('val_loss', 0):.4f}")
        print(f"{'='*80}\n")

# =============================================================================
# WEIGHTED MODEL CHECKPOINT CALLBACK (NORMAL MOD ODAKLI)
# =============================================================================
class WeightedModelCheckpoint(callbacks.Callback):
    """
    Modeli kaydederken Normal Mod (0.85) performansına odaklanır.
    """
    def __init__(self, filepath, X_val, y_val):
        super().__init__()
        self.filepath = filepath
        self.X_val = X_val
        self.y_val = y_val
        self.best_score = -float('inf')
    
    def normalize_roi(self, roi):
        if roi < 0:
            return max(0, 40 + roi * 0.4) 
        else:
            return min(100, 50 + roi * 0.5)
    
    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.X_val, verbose=0)
        threshold_preds = preds[2].flatten()
        
        y_true = (self.y_val >= 1.5).astype(int)
        
        # Normal Mod (0.85) Metrikleri
        y_pred = (threshold_preds >= THRESHOLD_NORMAL).astype(int)
        
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        TP = np.sum((y_true == 1) & (y_pred == 1))
        
        precision = (TP / (TP + FP) * 100) if (TP + FP) > 0 else 0
        
        # ROI Hesapla
        initial = 10000
        wallet = initial
        total_bets = 0
        wins = 0
        
        for pred, actual in zip(threshold_preds, self.y_val):
            if pred >= THRESHOLD_NORMAL:
                total_bets += 1
                wallet -= 10
                if actual >= 1.5:
                    wallet += 15
                    wins += 1
        
        roi = ((wallet - initial) / initial) * 100 if total_bets > 0 else 0
        win_rate = (wins / total_bets) * 100 if total_bets > 0 else 0
        
        normalized_roi = self.normalize_roi(roi)
        
        # Skorlama (Normal Mod Performansı)
        weighted_score = (
            0.50 * normalized_roi +
            0.30 * precision +
            0.20 * win_rate
        )
        
        if weighted_score > self.best_score:
            self.best_score = weighted_score
            self.model.save(self.filepath)
            print(f"\n✨ YENİ EN İYİ MODEL! (Score: {weighted_score:.2f})")
            print(f"   ROI: {roi:.2f}% | Precision: {precision:.2f}%")

# =============================================================================
# HER PENCERE İÇİN MODEL EĞİTİMİ
# =============================================================================
print("\n" + "="*80)
print("🔥 MULTI-SCALE MODEL EĞİTİMİ BAŞLIYOR")
print("="*80)

trained_models = {}
training_times = {}

for window_size in window_sizes:
    print("\n" + "="*80)
    print(f"🎯 WINDOW {window_size} - MODEL EĞİTİMİ")
    print("="*80)
    
    window_start_time = time.time()
    
    # Veriyi al
    data_dict = all_data_by_window[window_size]
    X_f_tr, X_seq_tr, y_reg_tr, y_cls_tr, y_thr_tr = data_dict['train']
    X_f_val, X_seq_val, y_reg_val, y_cls_val, y_thr_val = data_dict['val']
    
    # Model oluştur
    model = build_model_for_window(window_size, X_f_tr.shape[1])
    print(f"✅ Model oluşturuldu: {model.count_params():,} parametre")
    
    # Class weights - DENGELI
    w0, w1 = 1.5, 1.0
    
    # Adaptive Learning Rate Scheduler oluştur
    adaptive_scheduler = AdaptiveLearningRateScheduler(
        initial_lr=0.001,
        max_lr=0.01,
        min_lr=0.0001,
        patience=5
    )
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'regression': percentage_aware_regression_loss,
            'classification': 'categorical_crossentropy',
            'threshold': create_weighted_binary_crossentropy(w0, w1)
        },
        loss_weights={
            'regression': 0.50,
            'classification': 0.25,
            'threshold': 0.25
        },
        metrics={
            'regression': ['mae'],
            'classification': ['accuracy'],
            'threshold': ['accuracy']
        }
    )
    
    # Callbacks
    checkpoint_path = os.path.join(PROJECT_ROOT, f'models/progressive_window_{window_size}_best.h5')
    os.makedirs(os.path.join(PROJECT_ROOT, 'models'), exist_ok=True)
    
    # Detaylı metrikler callback'i
    detailed_metrics = DetailedMetricsCallback(
        X_val=[X_f_val, X_seq_val],
        y_val=y_reg_val
    )
    
    # Weighted model checkpoint
    weighted_checkpoint = WeightedModelCheckpoint(
        filepath=checkpoint_path,
        X_val=[X_f_val, X_seq_val],
        y_val=y_reg_val
    )
    
    # Custom Learning Rate Callback
    class AdaptiveLRCallback(callbacks.Callback):
        def __init__(self, scheduler):
            super().__init__()
            self.scheduler = scheduler
            
        def on_epoch_end(self, epoch, logs=None):
            if logs is None: logs = {}
            current_lr = self.scheduler(epoch, logs)
            K.set_value(self.model.optimizer.learning_rate, current_lr)
    
    adaptive_lr_callback = AdaptiveLRCallback(adaptive_scheduler)
    
    cbs = [
        detailed_metrics,
        weighted_checkpoint,
        adaptive_lr_callback,
        callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1)
    ]
    
    # Eğitim
    hist = model.fit(
        [X_f_tr, X_seq_tr],
        {
            'regression': y_reg_tr,
            'classification': y_cls_tr,
            'threshold': y_thr_tr
        },
        epochs=150,
        batch_size=32,
        validation_data=(
            [X_f_val, X_seq_val],
            {
                'regression': y_reg_val,
                'classification': y_cls_val,
                'threshold': y_thr_val
            }
        ),
        shuffle=False,  # KRITIK: Shuffle devre dışı!
        callbacks=cbs,
        verbose=1
    )
    
    window_time = time.time() - window_start_time
    training_times[window_size] = window_time
    
    # En iyi modeli yükle
    model.load_weights(checkpoint_path)
    
    # Test performansı
    X_f_test, X_seq_test, y_reg_test, y_cls_test, y_thr_test = data_dict['test']
    pred = model.predict([X_f_test, X_seq_test], verbose=0)
    p_thr = pred[2].flatten()
    
    # 2 Modlu Analiz
    y_true = (y_reg_test >= 1.5).astype(int)
    y_pred_normal = (p_thr >= THRESHOLD_NORMAL).astype(int)
    y_pred_rolling = (p_thr >= THRESHOLD_ROLLING).astype(int)
    
    acc_normal = accuracy_score(y_true, y_pred_normal)
    acc_rolling = accuracy_score(y_true, y_pred_rolling)
    
    print(f"\n📊 WINDOW {window_size} SONUÇLARI:")
    print(f"  Normal Mod Acc: {acc_normal*100:.2f}%")
    print(f"  Rolling Mod Acc: {acc_rolling*100:.2f}%")
    
    trained_models[window_size] = {
        'model': model,
        'scaler': data_dict['scaler'],
        'acc_normal': float(acc_normal),
        'acc_rolling': float(acc_rolling),
        'training_time': window_time
    }

total_training_time = sum(training_times.values())
print(f"\n✅ TÜM MODELLER EĞİTİLDİ! (Toplam: {total_training_time/60:.1f} dk)")

# =============================================================================
# ENSEMBLE PERFORMANS DEĞERLENDİRMESİ
# =============================================================================
print("\n" + "="*80)
print("🎯 ENSEMBLE PERFORMANS DEĞERLENDİRMESİ")
print("="*80)

X_f_test_500, X_seq_test_500, y_reg_test, _, y_thr_test = all_data_by_window[500]['test']

ensemble_predictions_reg = []
ensemble_predictions_thr = []

for window_size in window_sizes:
    model_dict = trained_models[window_size]
    model = model_dict['model']
    X_f_test_w, X_seq_test_w, _, _, _ = all_data_by_window[window_size]['test']
    pred = model.predict([X_f_test_w, X_seq_test_w], verbose=0)
    ensemble_predictions_reg.append(pred[0].flatten())
    ensemble_predictions_thr.append(pred[2].flatten())

# Ağırlıklı Ortalama
weights = [0.10, 0.15, 0.30, 0.25, 0.20] # 20, 50, 100, 250, 500
ensemble_reg = np.average(ensemble_predictions_reg, axis=0, weights=weights)
ensemble_thr = np.average(ensemble_predictions_thr, axis=0, weights=weights)

# Ensemble Metrics
y_true = (y_reg_test >= 1.5).astype(int)
y_pred_normal = (ensemble_thr >= THRESHOLD_NORMAL).astype(int)
y_pred_rolling = (ensemble_thr >= THRESHOLD_ROLLING).astype(int)

acc_ensemble_normal = accuracy_score(y_true, y_pred_normal)
acc_ensemble_rolling = accuracy_score(y_true, y_pred_rolling)

print(f"\n📊 ENSEMBLE PERFORMANSI:")
print(f"  Normal Mod ({THRESHOLD_NORMAL}): {acc_ensemble_normal*100:.2f}%")
print(f"  Rolling Mod ({THRESHOLD_ROLLING}): {acc_ensemble_rolling*100:.2f}%")

# =============================================================================
# 2 MODLU SANAL KASA SİMÜLASYONU (ENSEMBLE)
# =============================================================================
print("\n" + "="*80)
print("💰 SANAL KASA SİMÜLASYONU (ENSEMBLE)")
print("="*80)

initial_bankroll = len(y_reg_test) * 10
bet_amount = 10.0

# KASA 1: NORMAL MOD (0.85)
wallet1 = initial_bankroll
bets1 = 0
wins1 = 0

for i in range(len(y_reg_test)):
    if ensemble_thr[i] >= THRESHOLD_NORMAL:
        wallet1 -= bet_amount
        bets1 += 1
        # Dinamik Çıkış
        exit_point = min(max(1.5, ensemble_reg[i] * 0.8), 2.5)
        if y_reg_test[i] >= exit_point:
            wallet1 += exit_point * bet_amount
            wins1 += 1

roi1 = (wallet1 - initial_bankroll) / initial_bankroll * 100
win_rate1 = (wins1 / bets1 * 100) if bets1 > 0 else 0

print(f"💰 KASA 1 (NORMAL - {THRESHOLD_NORMAL}):")
print(f"  ROI: {roi1:+.2f}% | Win Rate: {win_rate1:.1f}% | Bets: {bets1}")

# KASA 2: ROLLING MOD (0.95)
wallet2 = initial_bankroll
bets2 = 0
wins2 = 0

for i in range(len(y_reg_test)):
    if ensemble_thr[i] >= THRESHOLD_ROLLING:
        wallet2 -= bet_amount
        bets2 += 1
        # Sabit Güvenli Çıkış
        if y_reg_test[i] >= 1.5:
            wallet2 += 1.5 * bet_amount
            wins2 += 1

roi2 = (wallet2 - initial_bankroll) / initial_bankroll * 100
win_rate2 = (wins2 / bets2 * 100) if bets2 > 0 else 0

print(f"💰 KASA 2 (ROLLING - {THRESHOLD_ROLLING}):")
print(f"  ROI: {roi2:+.2f}% | Win Rate: {win_rate2:.1f}% | Bets: {bets2}")

# =============================================================================
# MODEL KAYDETME & ZIP
# =============================================================================
print("\n" + "="*80)
print("💾 MODELLER KAYDEDİLİYOR")
print("="*80)

try:
    models_dir = os.path.join(PROJECT_ROOT, 'models/progressive_multiscale')
    os.makedirs(models_dir, exist_ok=True)
    
    for window_size in window_sizes:
        model_dict = trained_models[window_size]
        model_path = os.path.join(models_dir, f'model_window_{window_size}.h5')
        model_dict['model'].save(model_path)
        scaler_path = os.path.join(models_dir, f'scaler_window_{window_size}.pkl')
        joblib.dump(model_dict['scaler'], scaler_path)
        
    # Info JSON
    info = {
        'model': 'Progressive_NN_MultiScale_Ensemble',
        'version': '3.1',
        'thresholds': {'normal': THRESHOLD_NORMAL, 'rolling': THRESHOLD_ROLLING},
        'metrics': {
            'normal_acc': float(acc_ensemble_normal),
            'rolling_acc': float(acc_ensemble_rolling)
        },
        'simulation': {
            'normal_roi': float(roi1),
            'rolling_roi': float(roi2)
        }
    }
    with open(os.path.join(models_dir, 'model_info.json'), 'w') as f:
        json.dump(info, f, indent=2)
        
    print("✅ Modeller kaydedildi.")
    
except Exception as e:
    print(f"❌ Kaydetme hatası: {e}")

# ZIP
zip_filename = 'jetx_models_progressive_multiscale_v3.1'
shutil.make_archive(zip_filename, 'zip', models_dir)
print(f"✅ ZIP oluşturuldu: {zip_filename}.zip")

# Colab Download
try:
    import google.colab
    from google.colab import files
    files.download(f'{zip_filename}.zip')
except:
    pass

print(f"\n{'='*80}")
print(f"Bitiş: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*80}")
