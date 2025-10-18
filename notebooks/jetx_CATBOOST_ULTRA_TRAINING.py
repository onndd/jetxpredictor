#!/usr/bin/env python3
"""
🚀 JetX CATBOOST ULTRA TRAINING - Maksimum Performans (v3.0)

SEÇENEK C: ULTRA AGGRESSIVE
- 10,000 iterations (1,500 → 10,000, 6.5x artış!)
- 10 Model Ensemble (farklı seed/subsample ile)
- GPU desteği
- Advanced hyperparameters
- 5-Fold Cross-Validation (opsiyonel)
- 3 Sanal Kasa Sistemi
- Data Augmentation
- Extensive performance tracking

HEDEF: %85-90 accuracy, MAE < 1.2, ROI > %40

SÜRE: 4-6 saat (GPU ile)
"""

import subprocess
import sys
import os
import time
from datetime import datetime
import json
import shutil
from pathlib import Path

print("="*80)
print("🚀 JetX CATBOOST ULTRA TRAINING (v3.0 - SEÇENEK C)")
print("="*80)
print(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Kütüphaneleri yükle
print("📦 Kütüphaneler yükleniyor...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                      "catboost", "scikit-learn", "pandas", "numpy", 
                      "scipy", "joblib", "matplotlib", "seaborn", "tqdm",
                      "PyWavelets", "nolds"])

import numpy as np
import pandas as pd
import joblib
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix, classification_report
from catboost import CatBoostRegressor, CatBoostClassifier
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

print(f"✅ CatBoost & Dependencies: Import edildi")

# Proje yükle
if not os.path.exists('jetxpredictor'):
    print("\n📥 Proje klonlanıyor...")
    subprocess.check_call(["git", "clone", "https://github.com/onndd/jetxpredictor.git"])

os.chdir('jetxpredictor')
sys.path.append(os.getcwd())

# GPU Konfigürasyonunu yükle ve uygula
from utils.gpu_config import setup_catboost_gpu, print_gpu_status
print_gpu_status()
catboost_gpu_config = setup_catboost_gpu()
print()

from category_definitions import CategoryDefinitions, FeatureEngineering
from utils.catboost_ensemble import CatBoostEnsemble, CrossValidatedEnsemble
print(f"✅ Proje yüklendi - Kritik eşik: {CategoryDefinitions.CRITICAL_THRESHOLD}x\n")

# =============================================================================
# VERİ YÜKLEME
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
# FEATURE ENGINEERING
# =============================================================================
print("\n🔧 Feature extraction (Gelişmiş)...")
print("📌 Wavelet, DFA, Hurst, Fourier, Autocorrelation özellikleri dahil")

window_size = 1000
X_features = []
y_regression = []
y_classification = []

for i in tqdm(range(window_size, len(all_values)-1), desc='Features'):
    hist = all_values[:i].tolist()
    target = all_values[i]
    
    # TÜM gelişmiş özellikleri çıkar
    feats = FeatureEngineering.extract_all_features(hist)
    X_features.append(list(feats.values()))
    
    # Regression target
    y_regression.append(target)
    
    # Classification target (1.5 altı/üstü)
    y_classification.append(1 if target >= 1.5 else 0)

X = np.array(X_features)
y_reg = np.array(y_regression)
y_cls = np.array(y_classification)

print(f"✅ {len(X):,} örnek hazırlandı")
print(f"✅ Feature sayısı: {X.shape[1]} (Gelişmiş feature engineering ile)")

# =============================================================================
# NORMALIZASYON
# =============================================================================
print("\n📊 Normalizasyon...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

# =============================================================================
# TIME-SERIES SPLIT
# =============================================================================
print("\n📊 TIME-SERIES SPLIT (Kronolojik Bölme)...")
test_size = 1000
train_end = len(X) - test_size

# Train/Test split
X_train = X[:train_end]
X_test = X[train_end:]
y_reg_train = y_reg[:train_end]
y_reg_test = y_reg[train_end:]
y_cls_train = y_cls[:train_end]
y_cls_test = y_cls[train_end:]

print(f"✅ Train: {len(X_train):,}")
print(f"✅ Test: {len(X_test):,}")

# Validation split
val_size = int(len(X_train) * 0.2)
val_start = len(X_train) - val_size

X_tr = X_train[:val_start]
X_val = X_train[val_start:]
y_reg_tr = y_reg_train[:val_start]
y_reg_val = y_reg_train[val_start:]
y_cls_tr = y_cls_train[:val_start]
y_cls_val = y_cls_train[val_start:]

print(f"   ├─ Actual Train: {len(X_tr):,}")
print(f"   └─ Validation: {len(X_val):,}")
print()
print("⚠️  DATA AUGMENTATION: DEVRE DIŞI (Veri bütünlüğü korunuyor)")
print("⚠️  VERİ SIRASI: KORUNDU (shuffle=False)")

# =============================================================================
# REGRESSOR ENSEMBLE (10 MODEL)
# =============================================================================
print("\n" + "="*80)
print("🎯 CATBOOST REGRESSOR ENSEMBLE (10 Model)")
print("="*80)

reg_start = time.time()

# Base parametreler (ULTRA AGGRESSIVE)
base_reg_params = {
    'iterations': 10000,  # 1500 → 10000 (6.5x)
    'depth': 14,  # 10 → 14 (daha derin)
    'learning_rate': 0.05,  # 0.03 → 0.05
    'l2_leaf_reg': 3,  # 5 → 3 (daha az regularization)
    'random_strength': 1.5,  # YENİ
    'border_count': 254,  # Maksimum feature splits
    'leaf_estimation_iterations': 10,  # YENİ
    'loss_function': 'MAE',
    'eval_metric': 'MAE',
    'bootstrap_type': 'Bernoulli',  # Bernoulli (subsample ile uyumlu)
    'subsample': 0.8,  # YENİ - Bernoulli ile uyumlu
    'verbose': 100,
    **catboost_gpu_config  # GPU konfigürasyonunu ekle
}

print("📊 ULTRA AGGRESSIVE Parametreler:")
print(f"  iterations: 10,000 (1,500 → 10,000, 6.5x artış!)")
print(f"  depth: 14 (10 → 14)")
print(f"  learning_rate: 0.05 (0.03 → 0.05)")
print(f"  l2_leaf_reg: 3 (5 → 3, daha az regularization)")
print(f"  random_strength: 1.5 (YENİ)")
print(f"  border_count: 254 (maksimum)")
print(f"  leaf_estimation_iterations: 10 (YENİ)")
print(f"  task_type: GPU (AKTIF!)")
print(f"  bootstrap_type: Bernoulli (subsample ile uyumlu)")
print(f"  subsample: 0.8 (YENİ)")
print(f"  ⚠️  bagging_temperature: KALDIRILDI (Bernoulli ile uyumsuz)")
print()

# Ensemble oluştur
regressor_ensemble = CatBoostEnsemble(
    model_type='regressor',
    n_models=10,
    base_params=base_reg_params
)

# Eğit
print("🔥 10 Model Regressor Ensemble eğitimi başlıyor...")
print("⚠️  Bu 2-3 saat sürebilir (GPU ile)\n")

reg_results = regressor_ensemble.train_ensemble(
    X_tr, y_reg_tr,
    X_val, y_reg_val,
    verbose=True
)

reg_time = time.time() - reg_start
print(f"\n✅ Regressor Ensemble eğitimi tamamlandı! Süre: {reg_time/60:.1f} dakika")

# Test performansı
y_reg_pred, y_reg_variance = regressor_ensemble.predict(X_test, return_variance=True)
reg_confidence = regressor_ensemble.get_confidence(X_test)

mae_reg = mean_absolute_error(y_reg_test, y_reg_pred)
rmse_reg = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))

print(f"\n📊 REGRESSOR ENSEMBLE PERFORMANSI:")
print(f"  MAE: {mae_reg:.4f} (Hedef: < 1.2)")
print(f"  RMSE: {rmse_reg:.4f}")
print(f"  Ortalama Ensemble Confidence: {reg_confidence.mean():.4f}")
print(f"  Ensemble Agreement (std): {reg_results['std_score']:.4f}")

# Feature importance (ilk modelden)
feature_names = list(FeatureEngineering.extract_all_features(all_values[:1000].tolist()).keys())
importances = regressor_ensemble.models[0].feature_importances_
top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:20]

print(f"\n📊 TOP 20 ÖNEMLİ ÖZELLIKLER:")
for i, (feat, imp) in enumerate(top_features, 1):
    print(f"  {i:2d}. {feat:35s}: {imp:.4f}")

# =============================================================================
# CLASSIFIER ENSEMBLE (10 MODEL)
# =============================================================================
print("\n" + "="*80)
print("🎯 CATBOOST CLASSIFIER ENSEMBLE (10 Model)")
print("="*80)

cls_start = time.time()

# Base parametreler
base_cls_params = {
    'iterations': 10000,  # 1500 → 10000
    'depth': 12,  # 9 → 12
    'learning_rate': 0.05,
    'l2_leaf_reg': 3,
    'random_strength': 1.5,
    'border_count': 254,
    'leaf_estimation_iterations': 10,
    'loss_function': 'Logloss',
    'eval_metric': 'Accuracy',
    'bootstrap_type': 'Bernoulli',  # Bernoulli (subsample ile uyumlu)
    'subsample': 0.8,  # YENİ - Bernoulli ile uyumlu
    'auto_class_weights': 'Balanced',
    'verbose': 100,
    **catboost_gpu_config  # GPU konfigürasyonunu ekle
}

print("📊 ULTRA AGGRESSIVE Parametreler:")
print(f"  iterations: 10,000 (classifier için)")
print(f"  depth: 12 (9 → 12)")
print(f"  bootstrap_type: Bernoulli (subsample ile uyumlu)")
print(f"  subsample: 0.8 (YENİ)")
print(f"  auto_class_weights: Balanced")
print()

# Ensemble oluştur
classifier_ensemble = CatBoostEnsemble(
    model_type='classifier',
    n_models=10,
    base_params=base_cls_params
)

# Eğit
print("🔥 10 Model Classifier Ensemble eğitimi başlıyor...")
print("⚠️  Bu 2-3 saat sürebilir (GPU ile)\n")

cls_results = classifier_ensemble.train_ensemble(
    X_tr, y_cls_tr,
    X_val, y_cls_val,
    verbose=True
)

cls_time = time.time() - cls_start
print(f"\n✅ Classifier Ensemble eğitimi tamamlandı! Süre: {cls_time/60:.1f} dakika")

# Test performansı
y_cls_pred = classifier_ensemble.predict(X_test)
y_cls_proba, y_cls_proba_variance = classifier_ensemble.predict_proba(X_test, return_variance=True)
cls_confidence = classifier_ensemble.get_confidence(X_test)

cls_acc = accuracy_score(y_cls_test, y_cls_pred)

# Sınıf bazında accuracy
below_mask = y_cls_test == 0
above_mask = y_cls_test == 1

below_acc = accuracy_score(y_cls_test[below_mask], y_cls_pred[below_mask]) if below_mask.sum() > 0 else 0
above_acc = accuracy_score(y_cls_test[above_mask], y_cls_pred[above_mask]) if above_mask.sum() > 0 else 0

print(f"\n📊 CLASSIFIER ENSEMBLE PERFORMANSI:")
print(f"  Genel Accuracy: {cls_acc*100:.2f}% (Hedef: > 85%)")
print(f"  🔴 1.5 Altı Doğruluk: {below_acc*100:.2f}%")
print(f"  🟢 1.5 Üstü Doğruluk: {above_acc*100:.2f}%")
print(f"  Ortalama Ensemble Confidence: {cls_confidence.mean():.4f}")

# Confusion Matrix
cm = confusion_matrix(y_cls_test, y_cls_pred)
print(f"\n📋 CONFUSION MATRIX:")
print(f"                Tahmin")
print(f"Gerçek   1.5 Altı | 1.5 Üstü")
print(f"1.5 Altı {cm[0,0]:6d}   | {cm[0,1]:6d}  ⚠️ PARA KAYBI")
print(f"1.5 Üstü {cm[1,0]:6d}   | {cm[1,1]:6d}")

if cm[0,0] + cm[0,1] > 0:
    fpr = cm[0,1] / (cm[0,0] + cm[0,1])
    print(f"\n💰 PARA KAYBI RİSKİ: {fpr*100:.1f}%", end="")
    if fpr < 0.15:
        print(" ✅✅ MÜKEMMEL!")
    elif fpr < 0.20:
        print(" ✅ ÇOK İYİ!")
    else:
        print(f" (Hedef: <20%)")

# =============================================================================
# 3 SANAL KASA SİMÜLASYONU
# =============================================================================
print("\n" + "="*80)
print("💰 3 SANAL KASA SİMÜLASYONU (ULTRA)")
print("="*80)

test_count = len(y_reg_test)
initial_bankroll = test_count * 10
bet_amount = 10.0

print(f"📊 Test Veri Sayısı: {test_count:,}")
print(f"💰 Başlangıç Kasası: {initial_bankroll:,.2f} TL")
print(f"💵 Bahis Tutarı: {bet_amount:.2f} TL\n")

# =============================================================================
# KASA 1: 1.5x EŞİK SİSTEMİ
# =============================================================================
print("="*80)
print("💰 KASA 1: 1.5x EŞİK SİSTEMİ")
print("="*80)

kasa1_wallet = initial_bankroll
kasa1_total_bets = 0
kasa1_total_wins = 0
kasa1_total_losses = 0

for i in range(len(y_reg_test)):
    model_pred_cls = y_cls_pred[i]
    actual_value = y_reg_test[i]
    
    if model_pred_cls == 1:
        kasa1_wallet -= bet_amount
        kasa1_total_bets += 1
        
        exit_point = 1.5
        
        if actual_value >= exit_point:
            kasa1_wallet += exit_point * bet_amount
            kasa1_total_wins += 1
        else:
            kasa1_total_losses += 1

kasa1_profit_loss = kasa1_wallet - initial_bankroll
kasa1_roi = (kasa1_profit_loss / initial_bankroll) * 100
kasa1_win_rate = (kasa1_total_wins / kasa1_total_bets * 100) if kasa1_total_bets > 0 else 0

print(f"\n📊 KASA 1 SONUÇLARI:")
print(f"{'='*70}")
print(f"Toplam Oyun: {kasa1_total_bets:,}")
print(f"✅ Kazanan: {kasa1_total_wins:,} ({kasa1_win_rate:.1f}%)")
print(f"❌ Kaybeden: {kasa1_total_losses:,}")
print(f"💰 Final Kasa: {kasa1_wallet:,.2f} TL")
print(f"📈 Net Kar/Zarar: {kasa1_profit_loss:+,.2f} TL")
print(f"📊 ROI: {kasa1_roi:+.2f}%")

# =============================================================================
# KASA 2: %80 ÇIKIŞ SİSTEMİ
# =============================================================================
print("\n" + "="*80)
print("💰 KASA 2: %80 ÇIKIŞ SİSTEMİ")
print("="*80)

kasa2_wallet = initial_bankroll
kasa2_total_bets = 0
kasa2_total_wins = 0
kasa2_total_losses = 0

for i in range(len(y_reg_test)):
    model_pred_value = y_reg_pred[i]
    actual_value = y_reg_test[i]
    
    if model_pred_value >= 2.0:
        kasa2_wallet -= bet_amount
        kasa2_total_bets += 1
        
        exit_point = model_pred_value * 0.80
        
        if actual_value >= exit_point:
            kasa2_wallet += exit_point * bet_amount
            kasa2_total_wins += 1
        else:
            kasa2_total_losses += 1

kasa2_profit_loss = kasa2_wallet - initial_bankroll
kasa2_roi = (kasa2_profit_loss / initial_bankroll) * 100
kasa2_win_rate = (kasa2_total_wins / kasa2_total_bets * 100) if kasa2_total_bets > 0 else 0

print(f"\n📊 KASA 2 SONUÇLARI:")
print(f"{'='*70}")
print(f"Toplam Oyun: {kasa2_total_bets:,}")
print(f"✅ Kazanan: {kasa2_total_wins:,} ({kasa2_win_rate:.1f}%)")
print(f"❌ Kaybeden: {kasa2_total_losses:,}")
print(f"💰 Final Kasa: {kasa2_wallet:,.2f} TL")
print(f"📈 Net Kar/Zarar: {kasa2_profit_loss:+,.2f} TL")
print(f"📊 ROI: {kasa2_roi:+.2f}%")

# =============================================================================
# KASA 3: ENSEMBLE CONFIDENCE-BASED (YENİ!)
# =============================================================================
print("\n" + "="*80)
print("💰 KASA 3: ENSEMBLE CONFIDENCE-BASED (YENİ!)")
print("="*80)
print("Strateji: Sadece model agreement > %80 olduğunda bahis")
print("Çıkış: Ensemble tahmininin ortalaması\n")

kasa3_wallet = initial_bankroll
kasa3_total_bets = 0
kasa3_total_wins = 0
kasa3_total_losses = 0
confidence_threshold = 0.80  # %80 güven eşiği

for i in range(len(y_reg_test)):
    # Hem regressor hem classifier confidence'ı kullan
    combined_confidence = (reg_confidence[i] + cls_confidence[i]) / 2
    
    # Sadece yüksek güvende bahis yap
    if combined_confidence >= confidence_threshold:
        # Classifier 1.5 üstü tahmin ediyorsa
        if y_cls_proba[i, 1] > 0.5:
            kasa3_wallet -= bet_amount
            kasa3_total_bets += 1
            
            # Exit point: regressor tahmini × 0.85 (güvenli)
            exit_point = max(1.5, y_reg_pred[i] * 0.85)
            actual_value = y_reg_test[i]
            
            if actual_value >= exit_point:
                kasa3_wallet += exit_point * bet_amount
                kasa3_total_wins += 1
            else:
                kasa3_total_losses += 1

kasa3_profit_loss = kasa3_wallet - initial_bankroll
kasa3_roi = (kasa3_profit_loss / initial_bankroll) * 100
kasa3_win_rate = (kasa3_total_wins / kasa3_total_bets * 100) if kasa3_total_bets > 0 else 0

print(f"\n📊 KASA 3 SONUÇLARI:")
print(f"{'='*70}")
print(f"Toplam Oyun: {kasa3_total_bets:,} (sadece yüksek güven)")
print(f"✅ Kazanan: {kasa3_total_wins:,} ({kasa3_win_rate:.1f}%)")
print(f"❌ Kaybeden: {kasa3_total_losses:,}")
print(f"💰 Final Kasa: {kasa3_wallet:,.2f} TL")
print(f"📈 Net Kar/Zarar: {kasa3_profit_loss:+,.2f} TL")
print(f"📊 ROI: {kasa3_roi:+.2f}%")
print(f"🎯 Ortalama Confidence: {np.mean([reg_confidence[i] + cls_confidence[i] for i in range(len(reg_confidence))])/2:.2f}")

# =============================================================================
# KARŞILAŞTIRMA
# =============================================================================
print("\n" + "="*80)
print("📊 KASA KARŞILAŞTIRMASI")
print("="*80)
print(f"{'Metrik':<25} {'Kasa 1':<15} {'Kasa 2':<15} {'Kasa 3':<15}")
print(f"{'-'*70}")
print(f"{'Toplam Oyun':<25} {kasa1_total_bets:<15,} {kasa2_total_bets:<15,} {kasa3_total_bets:<15,}")
print(f"{'Kazanma Oranı':<25} {kasa1_win_rate:<15.1f}% {kasa2_win_rate:<15.1f}% {kasa3_win_rate:<15.1f}%")
print(f"{'Net Kar/Zarar':<25} {kasa1_profit_loss:<15,.2f} {kasa2_profit_loss:<15,.2f} {kasa3_profit_loss:<15,.2f}")
print(f"{'ROI':<25} {kasa1_roi:<15.2f}% {kasa2_roi:<15.2f}% {kasa3_roi:<15.2f}%")
print(f"{'-'*70}")

# En karlı kasa
profits = [
    ('KASA 1 (1.5x)', kasa1_profit_loss),
    ('KASA 2 (%80)', kasa2_profit_loss),
    ('KASA 3 (Confidence)', kasa3_profit_loss)
]
best_kasa = max(profits, key=lambda x: x[1])
print(f"🏆 EN KARLI: {best_kasa[0]} (+{best_kasa[1]:,.2f} TL)")

# =============================================================================
# MODEL KAYDETME
# =============================================================================
print("\n" + "="*80)
print("💾 ENSEMBLE MODELLER KAYDEDİLİYOR")
print("="*80)

# Dizinler oluştur
os.makedirs('models', exist_ok=True)

# Regressor ensemble kaydet
reg_ensemble_dir = 'models/catboost_ultra_regressor_ensemble'
regressor_ensemble.save_ensemble(reg_ensemble_dir)
print(f"✅ Regressor Ensemble kaydedildi: {reg_ensemble_dir}/")

# Classifier ensemble kaydet
cls_ensemble_dir = 'models/catboost_ultra_classifier_ensemble'
classifier_ensemble.save_ensemble(cls_ensemble_dir)
print(f"✅ Classifier Ensemble kaydedildi: {cls_ensemble_dir}/")

# Scaler kaydet
joblib.dump(scaler, 'models/catboost_ultra_scaler.pkl')
print(f"✅ Scaler kaydedildi: catboost_ultra_scaler.pkl")

# Model info kaydet
total_time = reg_time + cls_time
info = {
    'model': 'CatBoost_Ultra_Ensemble',
    'version': '3.0',
    'date': datetime.now().strftime('%Y-%m-%d'),
    'architecture': {
        'regressor': '10-Model Ensemble',
        'classifier': '10-Model Ensemble',
        'type': 'CatBoost'
    },
    'training_time_minutes': round(total_time/60, 1),
    'training_time_hours': round(total_time/3600, 1),
    'model_times': {
        'regressor': round(reg_time/60, 1),
        'classifier': round(cls_time/60, 1)
    },
    'feature_count': X.shape[1],
    'metrics': {
        'regression': {
            'mae': float(mae_reg),
            'rmse': float(rmse_reg),
            'ensemble_agreement': float(reg_results['std_score'])
        },
        'classification': {
            'accuracy': float(cls_acc),
            'below_15_accuracy': float(below_acc),
            'above_15_accuracy': float(above_acc),
            'money_loss_risk': float(fpr) if cm[0,0] + cm[0,1] > 0 else 0.0,
            'ensemble_agreement': float(cls_results['std_score'])
        }
    },
    'hyperparameters': {
        'regressor': base_reg_params,
        'classifier': base_cls_params
    },
    'ensemble_config': {
        'n_models': 10,
        'ensemble_type': 'weighted_average',
        'confidence_based': True
    },
    'triple_bankroll_performance': {
        'kasa_1_15x': {
            'roi': float(kasa1_roi),
            'win_rate': float(kasa1_win_rate),
            'total_bets': int(kasa1_total_bets),
            'profit_loss': float(kasa1_profit_loss)
        },
        'kasa_2_80percent': {
            'roi': float(kasa2_roi),
            'win_rate': float(kasa2_win_rate),
            'total_bets': int(kasa2_total_bets),
            'profit_loss': float(kasa2_profit_loss)
        },
        'kasa_3_confidence': {
            'roi': float(kasa3_roi),
            'win_rate': float(kasa3_win_rate),
            'total_bets': int(kasa3_total_bets),
            'profit_loss': float(kasa3_profit_loss),
            'confidence_threshold': confidence_threshold
        }
    },
    'top_features': [{'name': feat, 'importance': float(imp)} for feat, imp in top_features]
}

with open('models/catboost_ultra_model_info.json', 'w') as f:
    json.dump(info, f, indent=2)
print(f"✅ Model bilgileri kaydedildi: catboost_ultra_model_info.json")

# =============================================================================
# ZIP OLUŞTUR
# =============================================================================
print("\n" + "="*80)
print("📦 MODELLER ZIP'LENIYOR")
print("="*80)

zip_filename = 'jetx_models_catboost_ultra_v3.0'
shutil.make_archive(zip_filename, 'zip', 'models')

print(f"✅ ZIP dosyası oluşturuldu: {zip_filename}.zip")
print(f"📦 Boyut: {os.path.getsize(f'{zip_filename}.zip') / (1024*1024):.2f} MB")

# Google Colab'da indir ve Google Drive'a yedekle
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    # Google Drive'a otomatik yedekleme
    try:
        from google.colab import drive
        import os.path
        
        # Drive mount edilmiş mi kontrol et
        if not os.path.exists('/content/drive'):
            print("\n📦 Google Drive bağlanıyor...")
            drive.mount('/content/drive')
        
        # Yedekleme dizini oluştur
        backup_dir = '/content/drive/MyDrive/JetX_Models_Backup'
        os.makedirs(backup_dir, exist_ok=True)
        
        # ZIP'i kopyala
        import shutil
        backup_path = f'{backup_dir}/{zip_filename}.zip'
        shutil.copy(f'{zip_filename}.zip', backup_path)
        print(f"✅ Google Drive'a yedeklendi: {backup_path}")
        print(f"📁 Drive klasörü: MyDrive/JetX_Models_Backup/")
    except Exception as e:
        print(f"⚠️ Google Drive yedekleme hatası: {e}")
    
    # Manuel indirme
    try:
        from google.colab import files
        print(f"\n📥 {zip_filename}.zip tarayıcınıza indiriliyor...")
        files.download(f'{zip_filename}.zip')
        print(f"✅ İndirme başlatıldı!")
    except Exception as e:
        print(f"\n⚠️ Otomatik indirme hatası: {e}")
        print(f"\n{'='*80}")
        print("📥 MANUEL İNDİRME TALİMATLARI")
        print("="*80)
        print("1. Sol panelden 'Files' (📁) ikonuna tıklayın")
        print(f"2. '{zip_filename}.zip' dosyasını bulun")
        print("3. Dosyaya sağ tıklayıp 'Download' seçin")
        print(f"4. İndirilen ZIP'i lokal projenizin models/ klasörüne çıkartın")
        print("="*80)
else:
    print("\n⚠️ Google Colab ortamı değil - dosyalar kaydedildi")
    print(f"📁 ZIP: {zip_filename}.zip")

# =============================================================================
# FINAL RAPOR
# =============================================================================
print("\n" + "="*80)
print("🎉 CATBOOST ULTRA TRAINING TAMAMLANDI!")
print("="*80)
print(f"Toplam Süre: {total_time/60:.1f} dakika ({total_time/3600:.1f} saat)")
print()

# Hedef kontrolü
targets_met = []
if mae_reg < 1.2:
    targets_met.append(f"✅ MAE < 1.2: {mae_reg:.4f}")
else:
    targets_met.append(f"⚠️ MAE: {mae_reg:.4f} (Hedef: < 1.2)")

if cls_acc >= 0.85:
    targets_met.append(f"✅ Accuracy ≥ 85%: {cls_acc*100:.1f}%")
else:
    targets_met.append(f"⚠️ Accuracy: {cls_acc*100:.1f}% (Hedef: ≥ 85%)")

best_roi = max(kasa1_roi, kasa2_roi, kasa3_roi)
if best_roi >= 40:
    targets_met.append(f"✅ ROI ≥ 40%: {best_roi:.1f}%")
else:
    targets_met.append(f"⚠️ En İyi ROI: {best_roi:.1f}% (Hedef: ≥ 40%)")

print("📊 HEDEF KONTROL:")
for target in targets_met:
    print(f"  {target}")

print("\n📁 Çıktılar:")
print(f"  • {reg_ensemble_dir}/ (10 regressor model)")
print(f"  • {cls_ensemble_dir}/ (10 classifier model)")
print(f"  • catboost_ultra_scaler.pkl")
print(f"  • catboost_ultra_model_info.json")
print(f"  • {zip_filename}.zip")

print("\n🚀 Kullanım:")
print("  1. ZIP'i lokal projeye kopyalayın")
print("  2. models/ klasörüne çıkartın")
print("  3. Predictor'da model_type='catboost_ultra' ile kullanın")
print("  4. Ensemble confidence için get_confidence() metodunu kullanın")

print(f"\n{'='*80}")
print(f"Bitiş: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*80}")
