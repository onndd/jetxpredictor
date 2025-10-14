#!/usr/bin/env python3
"""
🎯 JetX PROGRESSIVE TRAINING - MULTI-SCALE WINDOW ENSEMBLE

YENİ YAKLAŞIM: Multi-Scale Window Ensemble
- Her pencere boyutu için ayrı model eğitimi
- Window boyutları: [500, 250, 100, 50, 20]
- Her model farklı zaman ölçeğinde desen öğrenir
- Final: Tüm modellerin ensemble'ı

HEDEFLER:
- 1.5 ALTI Doğruluk: %70-80%+
- 1.5 ÜSTÜ Doğruluk: %75-85%+
- Para kaybı riski: %20 altı
- MAE: < 2.0

⚠️  VERİ BÜTÜNLİĞİ:
- Shuffle: YASAK
- Augmentation: YASAK
- Kronolojik sıra: KORUNUYOR

SÜRE: ~10-12 saat (GPU ile, 5 model × ~2 saat)
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

# XLA optimizasyonu devre dışı
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("="*80)
print("🎯 JetX PROGRESSIVE TRAINING - MULTI-SCALE WINDOW ENSEMBLE")
print("="*80)
print(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()
print("🔧 YENİ SİSTEM: Her pencere boyutu için ayrı model")
print("   Window boyutları: [500, 250, 100, 50, 20]")
print("   ⚠️  Veri sırası KORUNUYOR (shuffle=False)")
print("   ⚠️  Data augmentation KAPALI")
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

print(f"✅ TensorFlow: {tf.__version__}")
print(f"✅ GPU: {'Mevcut' if len(tf.config.list_physical_devices('GPU')) > 0 else 'Yok (CPU)'}")

# Proje yükle
if not os.path.exists('jetxpredictor'):
    print("\n📥 Proje klonlanıyor...")
    subprocess.check_call(["git", "clone", "https://github.com/onndd/jetxpredictor.git"])

os.chdir('jetxpredictor')
sys.path.append(os.getcwd())

from category_definitions import CategoryDefinitions, FeatureEngineering
from utils.multi_scale_window import MultiScaleWindowExtractor, MultiScaleEnsemble, split_data_preserving_order
from utils.custom_losses import percentage_aware_regression_loss, balanced_focal_loss, create_weighted_binary_crossentropy
print(f"✅ Proje yüklendi - Kritik eşik: {CategoryDefinitions.CRITICAL_THRESHOLD}x\n")

# =============================================================================
# VERİ YÜKLEME (SIRA KORUNARAK)
# =============================================================================
print("📊 Veri yükleniyor...")
conn = sqlite3.connect('jetx_data.db')
data = pd.read_sql_query("SELECT value FROM jetx_results ORDER BY id", conn)
conn.close()

all_values = data['value'].values
print(f"✅ {len(all_values):,} veri yüklendi")
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
print("📌 Her pencere boyutu için feature engineering")

window_sizes = [500, 250, 100, 50, 20]

def extract_features_for_window(data, window_size):
    """
    Belirli bir pencere boyutu için feature extraction
    """
    X_features = []
    X_sequences = []
    y_regression = []
    y_classification = []
    y_threshold = []
    
    for i in tqdm(range(window_size, len(data)-1), desc=f'Window {window_size}'):
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
    
    # Test data
    X_f_test, X_seq_test, y_reg_test, y_cls_test, y_thr_test = extract_features_for_window(
        test_data, window_size
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
    # Küçük pencereler için basit, büyük pencereler için karmaşık
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
        x_seq = layers.LSTM(128, return_sequences=True)(inp_sequence)
        x_seq = layers.Dropout(0.2)(x_seq)
        
        # Attention
        attention = layers.Dense(1, activation='tanh')(x_seq)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(128)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        x_seq_attended = layers.Multiply()([x_seq, attention])
        x_seq = layers.Lambda(lambda x: K.sum(x, axis=1))(x_seq_attended)
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
# WEIGHTED MODEL CHECKPOINT CALLBACK
# =============================================================================
class WeightedModelCheckpoint(callbacks.Callback):
    """
    Weighted model selection based on:
    - 50% Below 15 accuracy
    - 40% Above 15 accuracy  
    - 10% ROI (normalized)
    """
    def __init__(self, filepath, X_val, y_val):
        super().__init__()
        self.filepath = filepath
        self.X_val = X_val
        self.y_val = y_val
        self.best_score = -1
    
    def normalize_roi(self, roi):
        """Kademeli lineer normalizasyon (Seçenek 2)"""
        if roi < 0:
            # Negatif ROI: 0-40 arası
            return max(0, (roi + 100) / 100 * 40)
        else:
            # Pozitif ROI: 40-100 arası
            return min(100, 40 + (roi / 200 * 60))
    
    def simulate_bankroll(self, predictions, actuals):
        """1.5x eşikte sanal kasa simülasyonu"""
        initial = 10000
        wallet = initial
        for pred, actual in zip(predictions, actuals):
            if pred >= 0.5:  # Model 1.5 üstü dedi
                wallet -= 10
                if actual >= 1.5:
                    wallet += 15
        roi = ((wallet - initial) / initial) * 100
        return roi
    
    def on_epoch_end(self, epoch, logs=None):
        # Tahminler yap (threshold output kullan)
        preds = self.model.predict(self.X_val, verbose=0)
        threshold_preds = preds[2].flatten()  # threshold output (sigmoid)
        
        # Below/Above accuracy hesapla
        below_mask = self.y_val < 1.5
        above_mask = self.y_val >= 1.5
        
        below_correct = 0
        below_total = 0
        for pred, actual in zip(threshold_preds[below_mask], self.y_val[below_mask]):
            below_total += 1
            if pred < 0.5 and actual < 1.5:  # Doğru tahmin (1.5 altı)
                below_correct += 1
        
        above_correct = 0
        above_total = 0
        for pred, actual in zip(threshold_preds[above_mask], self.y_val[above_mask]):
            above_total += 1
            if pred >= 0.5 and actual >= 1.5:  # Doğru tahmin (1.5 üstü)
                above_correct += 1
        
        below_acc = (below_correct / below_total * 100) if below_total > 0 else 0
        above_acc = (above_correct / above_total * 100) if above_total > 0 else 0
        
        # ROI hesapla
        roi = self.simulate_bankroll(threshold_preds, self.y_val)
        normalized_roi = self.normalize_roi(roi)
        
        # Weighted score
        weighted_score = (0.5 * below_acc) + (0.4 * above_acc) + (0.1 * normalized_roi)
        
        # En iyi modeli kaydet
        if weighted_score > self.best_score:
            self.best_score = weighted_score
            self.model.save(self.filepath)
            print(f"\n✨ Yeni en iyi model! Weighted Score: {weighted_score:.2f}")
            print(f"   Below 15: {below_acc:.1f}% | Above 15: {above_acc:.1f}% | ROI: {roi:+.1f}% (Normalized: {normalized_roi:.1f})")

# =============================================================================
# HER PENCERE İÇİN MODEL EĞİTİMİ
# =============================================================================
print("\n" + "="*80)
print("🔥 MULTI-SCALE MODEL EĞİTİMİ BAŞLIYOR")
print("="*80)
print(f"Window boyutları: {window_sizes}")
print(f"Her window için ayrı model eğitilecek")
print(f"📊 Model Seçim Kriteri: Weighted Score")
print(f"   - 50% Below 15 Accuracy")
print(f"   - 40% Above 15 Accuracy")
print(f"   - 10% ROI (Normalized)")
print("="*80 + "\n")

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
    
    # Class weights - window boyutuna göre ayarla
    # Küçük pencereler daha agresif (daha yüksek weight)
    # Büyük pencereler daha dengeli (daha düşük weight)
    if window_size <= 50:
        w0 = 25.0  # Çok agresif
    elif window_size <= 100:
        w0 = 20.0  # Agresif
    elif window_size <= 250:
        w0 = 15.0  # Orta
    else:
        w0 = 10.0  # Dengeli
    
    w1 = 1.0
    
    print(f"📊 CLASS WEIGHTS (Window {window_size}):")
    print(f"  1.5 altı: {w0:.1f}x")
    print(f"  1.5 üstü: {w1:.1f}x")
    
    # Compile
    model.compile(
        optimizer=Adam(0.0001),
        loss={
            'regression': percentage_aware_regression_loss,
            'classification': 'categorical_crossentropy',
            'threshold': create_weighted_binary_crossentropy(w0, w1)
        },
        loss_weights={
            'regression': 0.50,
            'classification': 0.15,
            'threshold': 0.35
        },
        metrics={
            'regression': ['mae'],
            'classification': ['accuracy'],
            'threshold': ['accuracy']
        }
    )
    
    # Callbacks
    checkpoint_path = f'models/progressive_window_{window_size}_best.h5'
    os.makedirs('models', exist_ok=True)
    
    # Weighted model checkpoint - 50% Below 15, 40% Above 15, 10% ROI
    weighted_checkpoint = WeightedModelCheckpoint(
        filepath=checkpoint_path,
        X_val=[X_f_val, X_seq_val],
        y_val=y_reg_val
    )
    
    cbs = [
        weighted_checkpoint,  # Weighted model selection
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            min_delta=0.001,
            mode='min',
            restore_best_weights=False,  # Weighted checkpoint handles this
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Eğitim
    print(f"\n🔥 Window {window_size} eğitimi başlıyor...")
    print(f"⏱️  Tahmini süre: ~2 saat")
    
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
    
    print(f"\n✅ Window {window_size} eğitimi tamamlandı!")
    print(f"⏱️  Süre: {window_time/60:.1f} dakika ({window_time/3600:.2f} saat)")
    
    # En iyi modeli yükle
    model.load_weights(checkpoint_path)
    
    # Test performansı
    X_f_test, X_seq_test, y_reg_test, y_cls_test, y_thr_test = data_dict['test']
    
    pred = model.predict([X_f_test, X_seq_test], verbose=0)
    p_reg = pred[0].flatten()
    p_thr = pred[2].flatten()
    
    # Metrics
    mae = mean_absolute_error(y_reg_test, p_reg)
    
    thr_true = (y_reg_test >= 1.5).astype(int)
    thr_pred = (p_thr >= 0.5).astype(int)
    thr_acc = accuracy_score(thr_true, thr_pred)
    
    below_mask = thr_true == 0
    above_mask = thr_true == 1
    below_acc = accuracy_score(thr_true[below_mask], thr_pred[below_mask]) if below_mask.sum() > 0 else 0
    above_acc = accuracy_score(thr_true[above_mask], thr_pred[above_mask]) if above_mask.sum() > 0 else 0
    
    print(f"\n📊 WINDOW {window_size} TEST PERFORMANSI:")
    print(f"  MAE: {mae:.4f}")
    print(f"  Threshold Accuracy: {thr_acc*100:.2f}%")
    print(f"  🔴 1.5 Altı: {below_acc*100:.2f}%")
    print(f"  🟢 1.5 Üstü: {above_acc*100:.2f}%")
    
    # Modeli kaydet
    trained_models[window_size] = {
        'model': model,
        'scaler': data_dict['scaler'],
        'mae': float(mae),
        'threshold_acc': float(thr_acc),
        'below_acc': float(below_acc),
        'above_acc': float(above_acc),
        'training_time': window_time
    }
    
    print("="*80)

total_training_time = sum(training_times.values())
print(f"\n✅ TÜM MODELLER EĞİTİLDİ!")
print(f"⏱️  Toplam Süre: {total_training_time/60:.1f} dakika ({total_training_time/3600:.2f} saat)")

# =============================================================================
# ENSEMBLE PERFORMANS DEĞERLENDİRMESİ
# =============================================================================
print("\n" + "="*80)
print("🎯 ENSEMBLE PERFORMANS DEĞERLENDİRMESİ")
print("="*80)

# Her modelden tahminleri al
X_f_test_500, X_seq_test_500, y_reg_test, _, y_thr_test = all_data_by_window[500]['test']

ensemble_predictions_reg = []
ensemble_predictions_thr = []

for window_size in window_sizes:
    model_dict = trained_models[window_size]
    model = model_dict['model']
    
    # Bu window için test data
    X_f_test_w, X_seq_test_w, _, _, _ = all_data_by_window[window_size]['test']
    
    # Tahmin
    pred = model.predict([X_f_test_w, X_seq_test_w], verbose=0)
    p_reg = pred[0].flatten()
    p_thr = pred[2].flatten()
    
    ensemble_predictions_reg.append(p_reg)
    ensemble_predictions_thr.append(p_thr)

# Ensemble: Basit ortalama
ensemble_reg = np.mean(ensemble_predictions_reg, axis=0)
ensemble_thr = np.mean(ensemble_predictions_thr, axis=0)

# Metrics
mae_ensemble = mean_absolute_error(y_reg_test, ensemble_reg)
rmse_ensemble = np.sqrt(mean_squared_error(y_reg_test, ensemble_reg))

thr_true = (y_reg_test >= 1.5).astype(int)
thr_pred_ensemble = (ensemble_thr >= 0.5).astype(int)
thr_acc_ensemble = accuracy_score(thr_true, thr_pred_ensemble)

below_mask = thr_true == 0
above_mask = thr_true == 1
below_acc_ensemble = accuracy_score(thr_true[below_mask], thr_pred_ensemble[below_mask]) if below_mask.sum() > 0 else 0
above_acc_ensemble = accuracy_score(thr_true[above_mask], thr_pred_ensemble[above_mask]) if above_mask.sum() > 0 else 0

print(f"\n📊 ENSEMBLE PERFORMANSI:")
print(f"  MAE: {mae_ensemble:.4f}")
print(f"  RMSE: {rmse_ensemble:.4f}")
print(f"  Threshold Accuracy: {thr_acc_ensemble*100:.2f}%")
print(f"  🔴 1.5 Altı: {below_acc_ensemble*100:.2f}%")
print(f"  🟢 1.5 Üstü: {above_acc_ensemble*100:.2f}%")

# Confusion matrix
cm = confusion_matrix(thr_true, thr_pred_ensemble)
print(f"\n📋 CONFUSION MATRIX (ENSEMBLE):")
print(f"                Tahmin")
print(f"Gerçek   1.5 Altı | 1.5 Üstü")
print(f"1.5 Altı {cm[0,0]:6d}   | {cm[0,1]:6d}  ⚠️ PARA KAYBI")
print(f"1.5 Üstü {cm[1,0]:6d}   | {cm[1,1]:6d}")

if cm[0,0] + cm[0,1] > 0:
    fpr = cm[0,1] / (cm[0,0] + cm[0,1])
    print(f"\n💰 PARA KAYBI RİSKİ: {fpr*100:.1f}%", end="")
    if fpr < 0.20:
        print(" ✅ HEDEF AŞILDI!")
    else:
        print(f" (Hedef: <20%)")

# =============================================================================
# MODEL KARŞILAŞTIRMASI
# =============================================================================
print("\n" + "="*80)
print("📊 WINDOW BAZINDA PERFORMANS KARŞILAŞTIRMASI")
print("="*80)

print(f"\n{'Window':<10} {'MAE':<10} {'Thr Acc':<12} {'Below':<12} {'Above':<12} {'Süre':<12}")
print("-"*70)
for window_size in window_sizes:
    model_dict = trained_models[window_size]
    print(
        f"{window_size:<10} "
        f"{model_dict['mae']:<10.4f} "
        f"{model_dict['threshold_acc']*100:<12.2f}% "
        f"{model_dict['below_acc']*100:<12.2f}% "
        f"{model_dict['above_acc']*100:<12.2f}% "
        f"{model_dict['training_time']/60:<12.1f} dk"
    )
print("-"*70)
print(
    f"{'ENSEMBLE':<10} "
    f"{mae_ensemble:<10.4f} "
    f"{thr_acc_ensemble*100:<12.2f}% "
    f"{below_acc_ensemble*100:<12.2f}% "
    f"{above_acc_ensemble*100:<12.2f}%"
)
print("="*80)

# =============================================================================
# SANAL KASA SİMÜLASYONU (ENSEMBLE)
# =============================================================================
print("\n" + "="*80)
print("💰 SANAL KASA SİMÜLASYONU (ENSEMBLE)")
print("="*80)

test_count = len(y_reg_test)
initial_bankroll = test_count * 10
bet_amount = 10.0

print(f"📊 Test Veri Sayısı: {test_count:,}")
print(f"💰 Başlangıç Kasası: {initial_bankroll:,.2f} TL")
print(f"💵 Bahis Tutarı: {bet_amount:.2f} TL\n")

# KASA 1: 1.5x EŞİK
wallet1 = initial_bankroll
total_bets1 = 0
total_wins1 = 0

for i in range(len(y_reg_test)):
    model_pred = thr_pred_ensemble[i]
    actual_value = y_reg_test[i]
    
    if model_pred == 1:
        wallet1 -= bet_amount
        total_bets1 += 1
        
        if actual_value >= 1.5:
            wallet1 += 1.5 * bet_amount
            total_wins1 += 1

profit1 = wallet1 - initial_bankroll
roi1 = (profit1 / initial_bankroll) * 100
win_rate1 = (total_wins1 / total_bets1 * 100) if total_bets1 > 0 else 0

print(f"💰 KASA 1 (1.5x EŞİK):")
print(f"  Toplam Oyun: {total_bets1:,}")
print(f"  Kazanan: {total_wins1:,} ({win_rate1:.1f}%)")
print(f"  Final Kasa: {wallet1:,.2f} TL")
print(f"  Net Kar/Zarar: {profit1:+,.2f} TL")
print(f"  ROI: {roi1:+.2f}%")

# KASA 2: %80 ÇIKIŞ
wallet2 = initial_bankroll
total_bets2 = 0
total_wins2 = 0

for i in range(len(y_reg_test)):
    model_pred_value = ensemble_reg[i]
    actual_value = y_reg_test[i]
    
    if model_pred_value >= 2.0:
        wallet2 -= bet_amount
        total_bets2 += 1
        
        exit_point = model_pred_value * 0.80
        if actual_value >= exit_point:
            wallet2 += exit_point * bet_amount
            total_wins2 += 1

profit2 = wallet2 - initial_bankroll
roi2 = (profit2 / initial_bankroll) * 100
win_rate2 = (total_wins2 / total_bets2 * 100) if total_bets2 > 0 else 0

print(f"\n💰 KASA 2 (%80 ÇIKIŞ):")
print(f"  Toplam Oyun: {total_bets2:,}")
print(f"  Kazanan: {total_wins2:,} ({win_rate2:.1f}%)")
print(f"  Final Kasa: {wallet2:,.2f} TL")
print(f"  Net Kar/Zarar: {profit2:+,.2f} TL")
print(f"  ROI: {roi2:+.2f}%")

print("\n" + "="*80)

# =============================================================================
# MODEL KAYDETME
# =============================================================================
print("\n" + "="*80)
print("💾 MODELLER KAYDEDİLİYOR")
print("="*80)

os.makedirs('models/progressive_multiscale', exist_ok=True)

# Her window için model kaydet
for window_size in window_sizes:
    model_dict = trained_models[window_size]
    
    # Model
    model_path = f'models/progressive_multiscale/model_window_{window_size}.h5'
    model_dict['model'].save(model_path)
    
    # Scaler
    scaler_path = f'models/progressive_multiscale/scaler_window_{window_size}.pkl'
    joblib.dump(model_dict['scaler'], scaler_path)
    
    print(f"✅ Window {window_size} kaydedildi")

# Model bilgileri
info = {
    'model': 'Progressive_NN_MultiScale_Ensemble',
    'version': '3.0',
    'date': datetime.now().strftime('%Y-%m-%d'),
    'window_sizes': window_sizes,
    'total_training_time_hours': round(total_training_time/3600, 2),
    'ensemble_metrics': {
        'mae': float(mae_ensemble),
        'rmse': float(rmse_ensemble),
        'threshold_accuracy': float(thr_acc_ensemble),
        'below_15_accuracy': float(below_acc_ensemble),
        'above_15_accuracy': float(above_acc_ensemble),
        'money_loss_risk': float(fpr) if cm[0,0] + cm[0,1] > 0 else 0.0
    },
    'individual_models': {
        str(ws): {
            'mae': trained_models[ws]['mae'],
            'threshold_acc': trained_models[ws]['threshold_acc'],
            'below_acc': trained_models[ws]['below_acc'],
            'above_acc': trained_models[ws]['above_acc'],
            'training_time_minutes': round(trained_models[ws]['training_time']/60, 1)
        } for ws in window_sizes
    },
    'bankroll_performance': {
        'kasa_1_15x': {
            'roi': float(roi1),
            'win_rate': float(win_rate1),
            'total_bets': int(total_bets1),
            'profit_loss': float(profit1)
        },
        'kasa_2_80percent': {
            'roi': float(roi2),
            'win_rate': float(win_rate2),
            'total_bets': int(total_bets2),
            'profit_loss': float(profit2)
        }
    }
}

with open('models/progressive_multiscale/model_info.json', 'w') as f:
    json.dump(info, f, indent=2)

print(f"✅ Model bilgileri kaydedildi")

# ZIP oluştur
zip_filename = 'jetx_models_progressive_multiscale_v3.0'
shutil.make_archive(zip_filename, 'zip', 'models/progressive_multiscale')

print(f"\n✅ ZIP dosyası oluşturuldu: {zip_filename}.zip")
print(f"📦 Boyut: {os.path.getsize(f'{zip_filename}.zip') / (1024*1024):.2f} MB")

# Google Colab'da indirme
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    try:
        from google.colab import files
        files.download(f'{zip_filename}.zip')
        print(f"✅ {zip_filename}.zip indiriliyor...")
    except Exception as e:
        print(f"⚠️ İndirme hatası: {e}")
else:
    print(f"📁 ZIP dosyası mevcut: {zip_filename}.zip")

print("="*80)

# =============================================================================
# FINAL RAPOR
# =============================================================================
print("\n" + "="*80)
print("🎉 MULTI-SCALE PROGRESSIVE TRAINING TAMAMLANDI!")
print("="*80)
print(f"Toplam Süre: {total_training_time/60:.1f} dakika ({total_training_time/3600:.2f} saat)")
print()

if below_acc_ensemble >= 0.75 and fpr < 0.20:
    print("✅ ✅ ✅ TÜM HEDEFLER BAŞARIYLA AŞILDI!")
    print(f"  🔴 1.5 ALTI: {below_acc_ensemble*100:.1f}% (Hedef: 75%+)")
    print(f"  💰 Para kaybı: {fpr*100:.1f}% (Hedef: <20%)")
elif below_acc_ensemble >= 0.70:
    print("✅ ✅ İYİ PERFORMANS!")
    print(f"  🔴 1.5 ALTI: {below_acc_ensemble*100:.1f}%")
else:
    print("⚠️ Hedefin altında")
    print(f"  🔴 1.5 ALTI: {below_acc_ensemble*100:.1f}% (Hedef: 75%+)")

print(f"\n{'='*80}")
print(f"Bitiş: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*80}")
