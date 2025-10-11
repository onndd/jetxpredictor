#!/usr/bin/env python3
"""
🔍 JetX Hyperparameter Optimization with Optuna

Optuna ile otomatik hyperparameter search. En iyi kombinasyonu bulur:
- Class weights
- Learning rate
- Dropout rates
- Model architecture parameters
- Loss function parameters

KULLANIM:
1. Google Colab'da çalıştır
2. 50-100 trial ile en iyi parametreleri bul
3. En iyi parametrelerle model eğit

Süre: ~2-4 saat (GPU ile, 50 trial)
"""

import subprocess
import sys
import os

print("📦 Kütüphaneler yükleniyor...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                      "tensorflow", "scikit-learn", "pandas", "numpy", 
                      "optuna", "joblib", "matplotlib", "seaborn", "plotly"])

import numpy as np
import pandas as pd
import joblib
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import json
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# GPU ayarları
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU bulundu: {len(gpus)} adet")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("⚠️ GPU bulunamadı, CPU kullanılacak (çok daha yavaş!)")

# Proje yükle
if not os.path.exists('jetxpredictor'):
    print("📥 Proje klonlanıyor...")
    subprocess.check_call(["git", "clone", "https://github.com/onndd/jetxpredictor.git"])

os.chdir('jetxpredictor')
sys.path.append(os.getcwd())

from category_definitions import CategoryDefinitions, FeatureEngineering

print("✅ Proje yüklendi")

# =============================================================================
# VERİ HAZIRLIK
# =============================================================================
print("\n📊 Veri hazırlanıyor...")
conn = sqlite3.connect('jetx_data.db')
data = pd.read_sql_query("SELECT value FROM jetx_results ORDER BY id", conn)
conn.close()

all_values = data['value'].values
print(f"✅ {len(all_values)} veri yüklendi")

# Feature extraction
window_size = 500
X_f, X_50, X_200, X_500 = [], [], [], []
y_reg, y_thr = [], []

print("🔧 Features extraction...")
from tqdm.auto import tqdm

for i in tqdm(range(window_size, min(len(all_values)-1, window_size+5000)), desc='Features'):
    hist = all_values[:i].tolist()
    target = all_values[i]
    
    feats = FeatureEngineering.extract_all_features(hist)
    X_f.append(list(feats.values()))
    X_50.append(all_values[i-50:i])
    X_200.append(all_values[i-200:i])
    X_500.append(all_values[i-500:i])
    
    y_reg.append(target)
    y_thr.append(1.0 if target >= 1.5 else 0.0)

# Arrays
X_f = np.array(X_f)
X_50 = np.array(X_50).reshape(-1, 50, 1)
X_200 = np.array(X_200).reshape(-1, 200, 1)
X_500 = np.array(X_500).reshape(-1, 500, 1)
y_reg = np.array(y_reg)
y_thr = np.array(y_thr).reshape(-1, 1)

print(f"✅ {len(y_thr)} sample hazırlandı")

# Normalizasyon
scaler = StandardScaler()
X_f = scaler.fit_transform(X_f)
X_50 = np.log10(X_50 + 1e-8)
X_200 = np.log10(X_200 + 1e-8)
X_500 = np.log10(X_500 + 1e-8)

# Train/Validation split (Optuna için validation gerekli)
idx = np.arange(len(X_f))
tr_idx, val_idx = train_test_split(idx, test_size=0.2, shuffle=False)

X_f_tr, X_50_tr, X_200_tr, X_500_tr = X_f[tr_idx], X_50[tr_idx], X_200[tr_idx], X_500[tr_idx]
y_thr_tr = y_thr[tr_idx]

X_f_val, X_50_val, X_200_val, X_500_val = X_f[val_idx], X_50[val_idx], X_200[val_idx], X_500[val_idx]
y_thr_val = y_thr[val_idx]

print(f"Train: {len(X_f_tr)}, Validation: {len(X_f_val)}")

# =============================================================================
# OPTUNA OBJECTIVE FUNCTION
# =============================================================================

# Global best trial tracker
best_trial_accuracy = 0.0
best_trial_params = {}

def create_model(trial: optuna.Trial, n_features: int) -> models.Model:
    """
    Optuna trial'dan model oluştur
    
    Trial parametreleri:
    - n_blocks: N-BEATS blok sayısı
    - block_units: N-BEATS blok unit sayısı
    - tcn_filters: TCN filter sayısı
    - dropout: Dropout rate
    - fusion_units: Fusion layer unit sayısı
    """
    # Hyperparameters
    n_blocks = trial.suggest_int('n_blocks', 3, 10)
    block_units = trial.suggest_int('block_units', 64, 256, step=64)
    tcn_filters = trial.suggest_int('tcn_filters', 128, 512, step=128)
    dropout = trial.suggest_float('dropout', 0.1, 0.4)
    fusion_units = trial.suggest_int('fusion_units', 256, 1024, step=256)
    
    # Inputs
    inp_f = layers.Input((n_features,), name='features')
    inp_50 = layers.Input((50, 1), name='seq50')
    inp_200 = layers.Input((200, 1), name='seq200')
    inp_500 = layers.Input((500, 1), name='seq500')
    
    # N-BEATS branches (simplified)
    def nbeats_branch(x, units, blocks, name):
        for i in range(blocks):
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout)(x)
        return x
    
    # Short
    nb_s = layers.Flatten()(inp_50)
    nb_s = nbeats_branch(nb_s, block_units, n_blocks, 's')
    
    # Medium
    nb_m = layers.Flatten()(inp_200)
    nb_m = nbeats_branch(nb_m, block_units, n_blocks, 'm')
    
    # Long
    nb_l = layers.Flatten()(inp_500)
    nb_l = nbeats_branch(nb_l, block_units, n_blocks, 'l')
    
    nb_all = layers.Concatenate()([nb_s, nb_m, nb_l])
    
    # TCN (simplified)
    tcn = inp_500
    for i, dilation in enumerate([1, 2, 4, 8]):
        tcn = layers.Conv1D(tcn_filters, 3, dilation_rate=dilation, 
                           padding='causal', activation='relu')(tcn)
        tcn = layers.BatchNormalization()(tcn)
        tcn = layers.Dropout(dropout)(tcn)
    
    tcn = layers.GlobalAveragePooling1D()(tcn)
    
    # Fusion
    fus = layers.Concatenate()([inp_f, nb_all, tcn])
    fus = layers.Dense(fusion_units, activation='relu')(fus)
    fus = layers.BatchNormalization()(fus)
    fus = layers.Dropout(dropout)(fus)
    fus = layers.Dense(fusion_units // 2, activation='relu')(fus)
    fus = layers.Dropout(dropout)(fus)
    
    # Output (threshold only for speed)
    thr_branch = layers.Dense(64, activation='relu')(fus)
    thr_branch = layers.Dropout(dropout)(thr_branch)
    out_thr = layers.Dense(1, activation='sigmoid', name='threshold')(thr_branch)
    
    model = models.Model([inp_f, inp_50, inp_200, inp_500], out_thr)
    
    return model


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function
    
    Returns:
        Validation threshold accuracy (maximize edilecek)
    """
    global best_trial_accuracy, best_trial_params
    
    print(f"\n{'='*70}")
    print(f"TRIAL #{trial.number}")
    print(f"{'='*70}")
    
    # Hyperparameters to optimize
    class_weight_multiplier = trial.suggest_float('class_weight', 1.5, 5.0)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    focal_gamma = trial.suggest_float('focal_gamma', 1.0, 4.0)
    focal_alpha = trial.suggest_float('focal_alpha', 0.6, 0.9)
    
    # Model architecture
    n_features = X_f_tr.shape[1]
    model = create_model(trial, n_features)
    
    # Loss fonksiyonları
    def threshold_killer_loss(y_true, y_pred):
        mae = K.abs(y_true - y_pred)
        false_positive = K.cast(tf.logical_and(y_true < 0.5, y_pred >= 0.5), 'float32') * 2.0
        false_negative = K.cast(tf.logical_and(y_true >= 0.5, y_pred < 0.5), 'float32') * 1.5
        weight = K.maximum(false_positive, false_negative)
        weight = K.maximum(weight, 1.0)
        return K.mean(mae * weight)
    
    def focal_loss(gamma, alpha):
        def loss(y_true, y_pred):
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            focal_weight = alpha * K.pow(1 - pt, gamma)
            return -K.mean(focal_weight * K.log(pt))
        return loss
    
    # Class weights
    c0 = (y_thr_tr.flatten() == 0).sum()
    c1 = (y_thr_tr.flatten() == 1).sum()
    w0 = (len(y_thr_tr) / (2 * c0)) * class_weight_multiplier
    w1 = len(y_thr_tr) / (2 * c1)
    
    class_weight = {0: w0, 1: w1}
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate),
        loss=focal_loss(focal_gamma, focal_alpha),
        metrics=['accuracy']
    )
    
    print(f"\n🎯 Trial Parameters:")
    print(f"  Class weight: {class_weight_multiplier:.2f}x → w0={w0:.2f}, w1={w1:.2f}")
    print(f"  Learning rate: {learning_rate:.6f}")
    print(f"  Focal gamma: {focal_gamma:.2f}")
    print(f"  Focal alpha: {focal_alpha:.2f}")
    print(f"  Model params: {model.count_params():,}")
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        mode='max'
    )
    
    # Train (hızlı trial için 20 epoch)
    try:
        history = model.fit(
            [X_f_tr, X_50_tr, X_200_tr, X_500_tr],
            y_thr_tr,
            epochs=20,
            batch_size=32,
            validation_data=([X_f_val, X_50_val, X_200_val, X_500_val], y_thr_val),
            callbacks=[early_stop],
            class_weight=class_weight,
            verbose=0
        )
        
        # Validation accuracy
        val_acc = max(history.history['val_accuracy'])
        
        # Below/Above threshold accuracy
        y_pred = model.predict([X_f_val, X_50_val, X_200_val, X_500_val], verbose=0)
        y_pred_class = (y_pred >= 0.5).astype(int).flatten()
        y_true_class = y_thr_val.flatten().astype(int)
        
        below_mask = y_true_class == 0
        above_mask = y_true_class == 1
        
        below_acc = (y_pred_class[below_mask] == y_true_class[below_mask]).mean() if below_mask.sum() > 0 else 0
        above_acc = (y_pred_class[above_mask] == y_true_class[above_mask]).mean() if above_mask.sum() > 0 else 0
        
        print(f"\n📊 Results:")
        print(f"  Val Accuracy: {val_acc*100:.2f}%")
        print(f"  Below 1.5: {below_acc*100:.2f}%")
        print(f"  Above 1.5: {above_acc*100:.2f}%")
        
        # Update best trial
        if val_acc > best_trial_accuracy:
            best_trial_accuracy = val_acc
            best_trial_params = {
                'class_weight': class_weight_multiplier,
                'learning_rate': learning_rate,
                'focal_gamma': focal_gamma,
                'focal_alpha': focal_alpha,
                'n_blocks': trial.params['n_blocks'],
                'block_units': trial.params['block_units'],
                'tcn_filters': trial.params['tcn_filters'],
                'dropout': trial.params['dropout'],
                'fusion_units': trial.params['fusion_units']
            }
            print(f"\n🎉 NEW BEST TRIAL! Accuracy: {val_acc*100:.2f}%")
        
        # Cleanup
        K.clear_session()
        del model
        
        return val_acc
        
    except Exception as e:
        print(f"\n❌ Trial failed: {e}")
        K.clear_session()
        return 0.0


# =============================================================================
# OPTUNA STUDY
# =============================================================================
print("\n" + "="*70)
print("🔍 OPTUNA HYPERPARAMETER SEARCH BAŞLIYOR")
print("="*70)

# Kullanıcıdan trial sayısı al
n_trials = 50  # Default
print(f"\n📊 {n_trials} trial ile search başlayacak")
print("⏱️ Tahmini süre: ~2-4 saat (GPU ile)")

# Optuna study oluştur
study = optuna.create_study(
    direction='maximize',
    study_name='jetx_hyperparameter_search',
    sampler=optuna.samplers.TPESampler(seed=42)
)

# Optimize
try:
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
except KeyboardInterrupt:
    print("\n⚠️ Search interrupted by user")

# =============================================================================
# RESULTS
# =============================================================================
print("\n" + "="*70)
print("📊 OPTUNA SEARCH RESULTS")
print("="*70)

print(f"\n🏆 BEST TRIAL:")
print(f"  Trial #: {study.best_trial.number}")
print(f"  Accuracy: {study.best_value*100:.2f}%")

print(f"\n🎯 BEST PARAMETERS:")
for key, value in study.best_params.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.6f}")
    else:
        print(f"  {key}: {value}")

# Statistical summary
print(f"\n📈 STATISTICS:")
print(f"  Total trials: {len(study.trials)}")
print(f"  Best accuracy: {study.best_value*100:.2f}%")
print(f"  Mean accuracy: {np.mean([t.value for t in study.trials if t.value is not None])*100:.2f}%")
print(f"  Std accuracy: {np.std([t.value for t in study.trials if t.value is not None])*100:.2f}%")

# =============================================================================
# VISUALIZATION
# =============================================================================
print("\n📊 Görselleştirmeler oluşturuluyor...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Optimization History
ax = axes[0, 0]
values = [t.value for t in study.trials if t.value is not None]
ax.plot(values, marker='o', linestyle='-', alpha=0.6)
ax.axhline(y=study.best_value, color='r', linestyle='--', label=f'Best: {study.best_value*100:.2f}%')
ax.set_xlabel('Trial')
ax.set_ylabel('Accuracy')
ax.set_title('Optimization History')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Parameter Importances
try:
    importance = optuna.importance.get_param_importances(study)
    params = list(importance.keys())
    values = list(importance.values())
    
    ax = axes[0, 1]
    ax.barh(params, values, color='#3498db')
    ax.set_xlabel('Importance')
    ax.set_title('Parameter Importances')
    ax.grid(True, alpha=0.3)
except:
    axes[0, 1].text(0.5, 0.5, 'Not enough trials\nfor importance', 
                    ha='center', va='center', transform=axes[0, 1].transAxes)

# 3. Learning Rate vs Accuracy
ax = axes[1, 0]
lrs = [t.params['learning_rate'] for t in study.trials if t.value is not None]
accs = [t.value for t in study.trials if t.value is not None]
ax.scatter(lrs, accs, alpha=0.6, c=accs, cmap='viridis')
ax.set_xlabel('Learning Rate (log scale)')
ax.set_ylabel('Accuracy')
ax.set_xscale('log')
ax.set_title('Learning Rate vs Accuracy')
ax.colorbar(ax.scatter(lrs, accs, alpha=0.6, c=accs, cmap='viridis'), ax=ax, label='Accuracy')
ax.grid(True, alpha=0.3)

# 4. Class Weight vs Accuracy
ax = axes[1, 1]
weights = [t.params['class_weight'] for t in study.trials if t.value is not None]
ax.scatter(weights, accs, alpha=0.6, c=accs, cmap='viridis')
ax.set_xlabel('Class Weight Multiplier')
ax.set_ylabel('Accuracy')
ax.set_title('Class Weight vs Accuracy')
ax.colorbar(ax.scatter(weights, accs, alpha=0.6, c=accs, cmap='viridis'), ax=ax, label='Accuracy')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optuna_results.png', dpi=300, bbox_inches='tight')
print("✅ Görselleştirme kaydedildi: optuna_results.png")

# Interactive plots (Plotly)
try:
    import plotly.io as pio
    
    # Optimization history
    fig1 = plot_optimization_history(study)
    pio.write_html(fig1, 'optuna_history.html')
    
    # Parameter importances
    fig2 = plot_param_importances(study)
    pio.write_html(fig2, 'optuna_importance.html')
    
    print("✅ Interactive plots kaydedildi: optuna_history.html, optuna_importance.html")
except:
    print("⚠️ Plotly plots oluşturulamadı")

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n💾 Sonuçlar kaydediliyor...")

# Best parameters JSON
results = {
    'best_trial': {
        'number': study.best_trial.number,
        'accuracy': float(study.best_value),
        'params': study.best_params
    },
    'statistics': {
        'total_trials': len(study.trials),
        'best_accuracy': float(study.best_value),
        'mean_accuracy': float(np.mean([t.value for t in study.trials if t.value is not None])),
        'std_accuracy': float(np.std([t.value for t in study.trials if t.value is not None]))
    },
    'all_trials': [
        {
            'number': t.number,
            'accuracy': float(t.value) if t.value is not None else None,
            'params': t.params
        }
        for t in study.trials
    ]
}

with open('optuna_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✅ Dosyalar kaydedildi:")
print("- optuna_results.json")
print("- optuna_results.png")
print("- optuna_history.html (interaktif)")
print("- optuna_importance.html (interaktif)")

# Google Colab'da ise indir
try:
    from google.colab import files
    files.download('optuna_results.json')
    files.download('optuna_results.png')
    files.download('optuna_history.html')
    files.download('optuna_importance.html')
    print("\n✅ Dosyalar indirildi!")
except:
    print("\n⚠️ Colab dışında - dosyalar sadece kaydedildi")

# =============================================================================
# RECOMMENDATION
# =============================================================================
print("\n" + "="*70)
print("🎉 OPTUNA HYPERPARAMETER SEARCH TAMAMLANDI!")
print("="*70)

print(f"\n📊 BEST CONFIGURATION:")
print(f"✅ Accuracy: {study.best_value*100:.2f}%")
print(f"\n🎯 Önerilen parametreler:")
print(json.dumps(study.best_params, indent=2))

print(f"\n📁 Sonraki adımlar:")
print(f"1. optuna_results.json dosyasındaki best_params'ı kullanın")
print(f"2. Bu parametrelerle full training yapın (Progressive veya Ultra)")
print(f"3. Full epoch ile daha iyi sonuçlar bekleyebilirsiniz")
print(f"4. optuna_importance.html'de hangi parametrelerin önemli olduğunu görün")

print("\n💡 ÖNEMLİ NOT:")
print("Bu search sadece 20 epoch ile yapıldı (hızlı test için).")
print("Full training'de 100-300 epoch ile daha iyi sonuçlar alabilirsiniz!")

print("\n" + "="*70)
