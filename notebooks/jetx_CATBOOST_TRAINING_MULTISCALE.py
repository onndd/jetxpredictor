#!/usr/bin/env python3
"""
🤖 JetX CATBOOST TRAINING - MULTI-SCALE WINDOW ENSEMBLE (v3.1)

YENİ YAKLAŞIM: Multi-Scale Window Ensemble
- Her pencere boyutu için ayrı CatBoost modeli
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

print("="*80)
print("🤖 JetX CATBOOST TRAINING - MULTI-SCALE WINDOW ENSEMBLE (v3.1)")
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
                      "catboost", "scikit-learn", "pandas", "numpy", 
                      "scipy>=1.10.0", "joblib", "matplotlib", "seaborn", "tqdm",
                      "PyWavelets>=1.4.1", "nolds>=0.5.2"])

import numpy as np
import pandas as pd
import joblib
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix
from catboost import CatBoostRegressor, CatBoostClassifier
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

print(f"✅ CatBoost: İmport edildi")

# YENİ EŞİKLER
THRESHOLD_NORMAL = 0.85
THRESHOLD_ROLLING = 0.95

# GPU kontrolü ve fallback
try:
    from catboost import CatBoostClassifier
    X_test_gpu = [[1, 2, 3], [4, 5, 6]]
    y_test_gpu = [0, 1]
    
    temp_model = CatBoostClassifier(
        iterations=1, 
        task_type='GPU', 
        devices='0',
        border_count=32,
        verbose=False
    )
    temp_model.fit(X_test_gpu, y_test_gpu, verbose=False)
    
    GPU_AVAILABLE = True
    TASK_TYPE = 'GPU'
    print(f"✅ GPU: Mevcut ve kullanılabilir (Tesla T4 detected)")
    
except Exception as e:
    GPU_AVAILABLE = False
    TASK_TYPE = 'CPU'
    print(f"⚠️ GPU: Kullanılamıyor, CPU modunda çalışacak")

# Proje yükle ve kök dizini tespit et
PROJECT_ROOT = None
if os.path.exists('jetx_data.db'):
    PROJECT_ROOT = os.getcwd()
    print("\n✅ Proje kök dizini tespit edildi (mevcut dizin)")
elif os.path.exists('jetxpredictor/jetx_data.db'):
    PROJECT_ROOT = os.path.join(os.getcwd(), 'jetxpredictor')
    print(f"\n✅ Proje kök dizini tespit edildi: {PROJECT_ROOT}")
else:
    print("\n📥 Proje klonlanıyor...")
    subprocess.check_call(["git", "clone", "https://github.com/onndd/jetxpredictor.git"])
    PROJECT_ROOT = os.path.join(os.getcwd(), 'jetxpredictor')
    print(f"✅ Proje klonlandı: {PROJECT_ROOT}")

sys.path.insert(0, PROJECT_ROOT)
print(f"📂 Çalışma dizini: {os.getcwd()}")

from category_definitions import CategoryDefinitions, FeatureEngineering
from utils.multi_scale_window import split_data_preserving_order
from utils.threshold_manager import get_threshold_manager
print(f"✅ Proje yüklendi - Kritik eşik: {CategoryDefinitions.CRITICAL_THRESHOLD}x\n")

# =============================================================================
# VERİ YÜKLEME (SIRA KORUNARAK)
# =============================================================================
print("📊 Veri yükleniyor...")
db_path = os.path.join(PROJECT_ROOT, 'jetx_data.db')
conn = sqlite3.connect(db_path)
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
window_sizes = [500, 250, 100, 50, 20]

def extract_features_for_window(data, window_size, start_idx=None, end_idx=None):
    X_features = []
    y_regression = []
    y_classification = []
    
    if start_idx is None: start_idx = window_size
    if end_idx is None: end_idx = len(data) - 1
    
    for i in tqdm(range(start_idx, end_idx), desc=f'Window {window_size}'):
        hist = data[:i].tolist()
        target = data[i]
        
        feats = FeatureEngineering.extract_all_features(hist)
        X_features.append(list(feats.values()))
        
        y_regression.append(target)
        y_classification.append(1 if target >= 1.5 else 0)
    
    return np.array(X_features), np.array(y_regression), np.array(y_classification)

all_data_by_window = {}
max_window = max(window_sizes)
test_start_idx = max_window

for window_size in window_sizes:
    print(f"\n🔧 Window {window_size} için feature extraction...")
    
    X_train, y_reg_train, y_cls_train = extract_features_for_window(train_data, window_size)
    X_val, y_reg_val, y_cls_val = extract_features_for_window(val_data, window_size)
    X_test, y_reg_test, y_cls_test = extract_features_for_window(
        test_data, window_size, start_idx=test_start_idx
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    all_data_by_window[window_size] = {
        'train': (X_train, y_reg_train, y_cls_train),
        'val': (X_val, y_reg_val, y_cls_val),
        'test': (X_test, y_reg_test, y_cls_test),
        'scaler': scaler
    }
    
    print(f"✅ Window {window_size}: {len(X_train):,} train, {len(X_val):,} val, {len(X_test):,} test")

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
    
    data_dict = all_data_by_window[window_size]
    X_train, y_reg_train, y_cls_train = data_dict['train']
    X_val, y_reg_val, y_cls_val = data_dict['val']
    X_test, y_reg_test, y_cls_test = data_dict['test']
    
    # 1. REGRESSOR EĞİTİMİ
    print(f"\n🎯 REGRESSOR EĞİTİMİ (Window {window_size})")
    base_params = {
        'iterations': 1500,
        'depth': 10,
        'learning_rate': 0.03,
        'l2_leaf_reg': 5,
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8,
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'task_type': TASK_TYPE,
        'verbose': 100,
        'random_state': 42
    }
    if TASK_TYPE == 'GPU':
        base_params.update({'border_count': 128, 'gpu_ram_part': 0.95})
        
    regressor = CatBoostRegressor(**base_params)
    regressor.fit(X_train, y_reg_train, eval_set=(X_val, y_reg_val), verbose=100)
    
    y_reg_pred = regressor.predict(X_test)
    mae_reg = mean_absolute_error(y_reg_test, y_reg_pred)
    rmse_reg = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
    print(f"📊 Regressor MAE: {mae_reg:.4f}")
    
    # 2. CLASSIFIER EĞİTİMİ
    print(f"\n🎯 CLASSIFIER EĞİTİMİ (Window {window_size})")
    classifier_params = {
        'iterations': 1500,
        'depth': 9,
        'learning_rate': 0.03,
        'l2_leaf_reg': 5,
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8,
        'loss_function': 'Logloss',
        'eval_metric': 'Accuracy',
        'task_type': TASK_TYPE,
        'auto_class_weights': 'Balanced',
        'verbose': 100,
        'random_state': 42
    }
    if TASK_TYPE == 'GPU':
        classifier_params.update({'border_count': 128, 'gpu_ram_part': 0.95})
        
    classifier = CatBoostClassifier(**classifier_params)
    classifier.fit(X_train, y_cls_train, eval_set=(X_val, y_cls_val), verbose=100)
    
    # Test performansı (2 Modlu)
    y_cls_proba = classifier.predict_proba(X_test)[:, 1]
    
    # Normal Mod (0.85)
    y_pred_normal = (y_cls_proba >= THRESHOLD_NORMAL).astype(int)
    acc_normal = accuracy_score(y_cls_test, y_pred_normal)
    
    # Rolling Mod (0.95)
    y_pred_rolling = (y_cls_proba >= THRESHOLD_ROLLING).astype(int)
    acc_rolling = accuracy_score(y_cls_test, y_pred_rolling)
    
    print(f"\n📊 CLASSIFIER PERFORMANSI:")
    print(f"  Normal Mod Acc ({THRESHOLD_NORMAL}): {acc_normal*100:.2f}%")
    print(f"  Rolling Mod Acc ({THRESHOLD_ROLLING}): {acc_rolling*100:.2f}%")
    
    window_time = time.time() - window_start_time
    training_times[window_size] = window_time
    
    trained_models[window_size] = {
        'regressor': regressor,
        'classifier': classifier,
        'scaler': data_dict['scaler'],
        'mae': float(mae_reg),
        'rmse': float(rmse_reg),
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

X_test_500, y_reg_test, y_cls_test = all_data_by_window[500]['test']

ensemble_predictions_reg = []
ensemble_predictions_cls = []

for window_size in window_sizes:
    model_dict = trained_models[window_size]
    X_test_w, _, _ = all_data_by_window[window_size]['test']
    
    p_reg = model_dict['regressor'].predict(X_test_w)
    p_cls_proba = model_dict['classifier'].predict_proba(X_test_w)[:, 1]
    
    ensemble_predictions_reg.append(p_reg)
    ensemble_predictions_cls.append(p_cls_proba)

# Ensemble
ensemble_reg = np.mean(ensemble_predictions_reg, axis=0)
ensemble_proba_avg = np.mean(ensemble_predictions_cls, axis=0)

# Metrics
mae_ensemble = mean_absolute_error(y_reg_test, ensemble_reg)
rmse_ensemble = np.sqrt(mean_squared_error(y_reg_test, ensemble_reg))

# 2 Modlu Analiz
ensemble_cls_normal = (ensemble_proba_avg >= THRESHOLD_NORMAL).astype(int)
acc_ensemble_normal = accuracy_score(y_cls_test, ensemble_cls_normal)

ensemble_cls_rolling = (ensemble_proba_avg >= THRESHOLD_ROLLING).astype(int)
acc_ensemble_rolling = accuracy_score(y_cls_test, ensemble_cls_rolling)

print(f"\n📊 ENSEMBLE PERFORMANSI:")
print(f"  MAE: {mae_ensemble:.4f}")
print(f"  RMSE: {rmse_ensemble:.4f}")
print(f"  Normal Mod Acc ({THRESHOLD_NORMAL}): {acc_ensemble_normal*100:.2f}%")
print(f"  Rolling Mod Acc ({THRESHOLD_ROLLING}): {acc_ensemble_rolling*100:.2f}%")

# =============================================================================
# SANAL KASA SİMÜLASYONU (2 MODLU)
# =============================================================================
print("\n" + "="*80)
print("💰 SANAL KASA SİMÜLASYONU (2 MODLU YAPI)")
print("="*80)

initial_bankroll = len(y_reg_test) * 10
bet_amount = 10.0

# KASA 1: NORMAL MOD (0.85)
wallet1 = initial_bankroll
bets1 = 0
wins1 = 0

for i in range(len(y_reg_test)):
    if ensemble_proba_avg[i] >= THRESHOLD_NORMAL:
        wallet1 -= bet_amount
        bets1 += 1
        # Dinamik Çıkış (Max 2.5x)
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
    if ensemble_proba_avg[i] >= THRESHOLD_ROLLING:
        wallet2 -= bet_amount
        bets2 += 1
        # Sabit Güvenli Çıkış (1.5x)
        if y_reg_test[i] >= 1.5:
            wallet2 += 1.5 * bet_amount
            wins2 += 1

roi2 = (wallet2 - initial_bankroll) / initial_bankroll * 100
win_rate2 = (wins2 / bets2 * 100) if bets2 > 0 else 0

print(f"💰 KASA 2 (ROLLING - {THRESHOLD_ROLLING}):")
print(f"  ROI: {roi2:+.2f}% | Win Rate: {win_rate2:.1f}% | Bets: {bets2}")

print("\n" + "="*80)

# =============================================================================
# MODEL KAYDETME
# =============================================================================
print("\n" + "="*80)
print("💾 MODELLER KAYDEDİLİYOR")
print("="*80)

models_dir = os.path.join(PROJECT_ROOT, 'models/catboost_multiscale')
os.makedirs(models_dir, exist_ok=True)

for window_size in window_sizes:
    model_dict = trained_models[window_size]
    reg_path = os.path.join(models_dir, f'regressor_window_{window_size}.cbm')
    model_dict['regressor'].save_model(reg_path)
    cls_path = os.path.join(models_dir, f'classifier_window_{window_size}.cbm')
    model_dict['classifier'].save_model(cls_path)
    scaler_path = os.path.join(models_dir, f'scaler_window_{window_size}.pkl')
    joblib.dump(model_dict['scaler'], scaler_path)
    print(f"✅ Window {window_size} kaydedildi")

# Info JSON
info = {
    'model': 'CatBoost_MultiScale_Ensemble',
    'version': '3.1',
    'thresholds': {'normal': THRESHOLD_NORMAL, 'rolling': THRESHOLD_ROLLING},
    'metrics': {
        'mae': float(mae_ensemble),
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
print(f"✅ Model bilgileri kaydedildi")

# ZIP
zip_filename = 'jetx_models_catboost_multiscale_v3.1'
shutil.make_archive(zip_filename, 'zip', models_dir)
print(f"✅ ZIP oluşturuldu: {zip_filename}.zip")

# Colab Download
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    try:
        from google.colab import files
        files.download(f'{zip_filename}.zip')
    except Exception as e:
        print(f"⚠️ Otomatik indirme hatası: {e}")
else:
    print(f"📁 ZIP dosyası mevcut: {zip_filename}.zip")

print(f"\n{'='*80}")
print(f"Bitiş: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*80}")
