#!/usr/bin/env python3
"""
🎯 JetX Meta-Model Training Script (v2.0)

GÜNCELLEME:
- Meta-model ve Base modellerin performansı %85 güven eşiğine göre değerlendirilir.
- 2 Modlu Yapı (Normal/Rolling) entegre edildi.

Meta-model, base modellerin (Progressive, Ultra, XGBoost, AutoGluon, TabNet) 
tahminlerini input olarak alır ve final kararı verir.

Süre: ~30 dakika
"""

import subprocess
import sys
import os

print("📦 Kütüphaneler yükleniyor...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                      "tensorflow", "scikit-learn", "pandas", "numpy", 
                      "xgboost", "joblib", "matplotlib", "seaborn"])

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Kritik Güven Eşikleri
THRESHOLD_NORMAL = 0.85
THRESHOLD_ROLLING = 0.95

# Proje yükle
if not os.path.exists('jetxpredictor'):
    print("📥 Proje klonlanıyor...")
    subprocess.check_call(["git", "clone", "https://github.com/onndd/jetxpredictor.git"])
    
os.chdir('jetxpredictor')
sys.path.append(os.getcwd())

from category_definitions import CategoryDefinitions, FeatureEngineering

# CUSTOM_OBJECTS tanımı - Meta model yükleme için gerekli
from utils.custom_losses import (
    balanced_threshold_killer_loss,
    balanced_focal_loss,
    percentage_aware_regression_loss
)

CUSTOM_OBJECTS = {
    'balanced_threshold_killer_loss': balanced_threshold_killer_loss,
    'balanced_focal_loss': balanced_focal_loss,
    'percentage_aware_regression_loss': percentage_aware_regression_loss
}

print(f"✅ Proje yüklendi - Normal Eşik: {THRESHOLD_NORMAL}")

# =============================================================================
# VERİ YÜKLE
# =============================================================================
print("\n📊 Veri yükleniyor...")
conn = sqlite3.connect('jetx_data.db')
data = pd.read_sql_query("SELECT value FROM jetx_results ORDER BY id", conn)
conn.close()

all_values = data['value'].values
print(f"✅ {len(all_values)} veri yüklendi")

# =============================================================================
# BASE MODEL TAHMİNLERİNİ TOPLA
# =============================================================================
print("\n🤖 Base modelleri yükleniyor...")

# Model yolları
model_paths = {
    'progressive': {
        'model': 'models/jetx_progressive_final.h5',
        'scaler': 'models/scaler_progressive.pkl'
    },
    'ultra': {
        'model': 'models/jetx_ultra_model.h5',
        'scaler': 'models/scaler_ultra.pkl'
    },
    'xgboost': {
        'regressor': 'models/xgboost_regressor.json',
        'classifier': 'models/xgboost_classifier.json',
        'scaler': 'models/xgboost_scaler.pkl'
    },
    'autogluon': {
        'model': 'models/autogluon_model',
        'scaler': 'models/autogluon_scaler.pkl'
    },
    'tabnet': {
        'model': 'models/tabnet_high_x.pkl',
        'scaler': 'models/tabnet_scaler.pkl'
    }
}

# Modelleri yükle
loaded_models = {}

# Progressive
try:
    from tensorflow import keras
    if os.path.exists(model_paths['progressive']['model']):
        loaded_models['progressive'] = {
            'model': keras.models.load_model(
                model_paths['progressive']['model'],
                custom_objects=CUSTOM_OBJECTS
            ),
            'scaler': joblib.load(model_paths['progressive']['scaler'])
        }
        print("✅ Progressive model yüklendi")
    else:
        print("⚠️ Progressive model bulunamadı, atlanıyor")
except Exception as e:
    print(f"⚠️ Progressive model yüklenemedi: {e}")

# Ultra Aggressive
try:
    from tensorflow import keras
    if os.path.exists(model_paths['ultra']['model']):
        loaded_models['ultra'] = {
            'model': keras.models.load_model(
                model_paths['ultra']['model'],
                custom_objects=CUSTOM_OBJECTS
            ),
            'scaler': joblib.load(model_paths['ultra']['scaler'])
        }
        print("✅ Ultra Aggressive model yüklendi")
    else:
        print("⚠️ Ultra model bulunamadı, atlanıyor")
except Exception as e:
    print(f"⚠️ Ultra model yüklenemedi: {e}")

# XGBoost
try:
    if os.path.exists(model_paths['xgboost']['regressor']):
        xgb_reg = xgb.XGBRegressor()
        xgb_reg.load_model(model_paths['xgboost']['regressor'])
        xgb_cls = xgb.XGBClassifier()
        xgb_cls.load_model(model_paths['xgboost']['classifier'])
        
        loaded_models['xgboost'] = {
            'regressor': xgb_reg,
            'classifier': xgb_cls,
            'scaler': joblib.load(model_paths['xgboost']['scaler'])
        }
        print("✅ XGBoost modelleri yüklendi")
    else:
        print("⚠️ XGBoost modelleri bulunamadı, atlanıyor")
except Exception as e:
    print(f"⚠️ XGBoost modelleri yüklenemedi: {e}")

# AutoGluon
try:
    if os.path.exists(model_paths['autogluon']['model']):
        from autogluon.tabular import TabularPredictor
        loaded_models['autogluon'] = {
            'predictor': TabularPredictor.load(model_paths['autogluon']['model']),
            'scaler': joblib.load(model_paths['autogluon']['scaler']) if os.path.exists(model_paths['autogluon']['scaler']) else None
        }
        print("✅ AutoGluon modeli yüklendi")
    else:
        print("⚠️ AutoGluon modeli bulunamadı, atlanıyor")
except Exception as e:
    print(f"⚠️ AutoGluon modeli yüklenemedi: {e}")

# TabNet
try:
    if os.path.exists(model_paths['tabnet']['model']):
        from pytorch_tabnet.tab_model import TabNetClassifier
        tabnet_model = TabNetClassifier()
        tabnet_model.load_model(model_paths['tabnet']['model'])
        
        loaded_models['tabnet'] = {
            'model': tabnet_model,
            'scaler': joblib.load(model_paths['tabnet']['scaler']) if os.path.exists(model_paths['tabnet']['scaler']) else None
        }
        print("✅ TabNet modeli yüklendi")
    else:
        print("⚠️ TabNet modeli bulunamadı, atlanıyor")
except Exception as e:
    print(f"⚠️ TabNet modeli yüklenemedi: {e}")

if len(loaded_models) == 0:
    print("\n❌ HATA: Hiçbir base model yüklenemedi!")
    sys.exit(1)

print(f"\n✅ {len(loaded_models)} base model yüklendi: {list(loaded_models.keys())}")

# =============================================================================
# FEATURE EXTRACTION & BASE MODEL PREDICTIONS
# =============================================================================
print("\n🔧 Base model tahminleri toplanıyor...")

window_size = 1000
X_features = []
X_progressive = []
X_ultra = []
X_xgboost = []
X_autogluon = []
X_tabnet = []
y_true = []

from tqdm.auto import tqdm

for i in tqdm(range(window_size, len(all_values)-1), desc='Tahminler'):
    history = all_values[:i].tolist()
    target = all_values[i]
    
    # Features
    features = FeatureEngineering.extract_all_features(history)
    feature_values = np.array(list(features.values()))
    
    predictions = []
    
    # Progressive prediction
    if 'progressive' in loaded_models:
        try:
            scaler = loaded_models['progressive']['scaler']
            scaled_features = scaler.transform(feature_values.reshape(1, -1))
            
            seq_50 = np.log10(np.array(history[-50:]).reshape(1, 50, 1) + 1e-8)
            seq_200 = np.log10(np.array(history[-200:]).reshape(1, 200, 1) + 1e-8)
            seq_500 = np.log10(np.array(history[-500:]).reshape(1, 500, 1) + 1e-8)
            seq_1000 = np.log10(np.array(history[-1000:]).reshape(1, 1000, 1) + 1e-8)
            
            pred = loaded_models['progressive']['model'].predict(
                [scaled_features, seq_50, seq_200, seq_500, seq_1000],
                verbose=0
            )
            predictions.append(float(pred[2][0][0]))
        except:
            predictions.append(0.5)
    else:
        predictions.append(0.5)
    
    # Ultra prediction
    if 'ultra' in loaded_models:
        try:
            scaler = loaded_models['ultra']['scaler']
            scaled_features = scaler.transform(feature_values.reshape(1, -1))
            
            seq_50 = np.log10(np.array(history[-50:]).reshape(1, 50, 1) + 1e-8)
            seq_200 = np.log10(np.array(history[-200:]).reshape(1, 200, 1) + 1e-8)
            seq_500 = np.log10(np.array(history[-500:]).reshape(1, 500, 1) + 1e-8)
            seq_1000 = np.log10(np.array(history[-1000:]).reshape(1, 1000, 1) + 1e-8)
            
            pred = loaded_models['ultra']['model'].predict(
                [scaled_features, seq_50, seq_200, seq_500, seq_1000],
                verbose=0
            )
            predictions.append(float(pred[2][0][0]))
        except:
            predictions.append(0.5)
    else:
        predictions.append(0.5)
    
    # XGBoost prediction
    if 'xgboost' in loaded_models:
        try:
            scaler = loaded_models['xgboost']['scaler']
            scaled_features = scaler.transform(feature_values.reshape(1, -1))
            pred_proba = loaded_models['xgboost']['classifier'].predict_proba(scaled_features)
            predictions.append(float(pred_proba[0][1]))
        except:
            predictions.append(0.5)
    else:
        predictions.append(0.5)
    
    # AutoGluon prediction
    if 'autogluon' in loaded_models:
        try:
            feature_df = pd.DataFrame([feature_values])
            if loaded_models['autogluon']['scaler'] is not None:
                feature_df = loaded_models['autogluon']['scaler'].transform(feature_df)
            pred_proba = loaded_models['autogluon']['predictor'].predict_proba(feature_df)
            predictions.append(float(pred_proba.iloc[0, 1]))
        except:
            predictions.append(0.5)
    else:
        predictions.append(0.5)
    
    # TabNet prediction
    if 'tabnet' in loaded_models:
        try:
            if loaded_models['tabnet']['scaler'] is not None:
                scaled_features = loaded_models['tabnet']['scaler'].transform(feature_values.reshape(1, -1))
            else:
                scaled_features = feature_values.reshape(1, -1)
            
            pred_proba = loaded_models['tabnet']['model'].predict_proba(scaled_features)
            high_x_prob = float(pred_proba[0][2] + pred_proba[0][3])
            predictions.append(high_x_prob)
        except:
            predictions.append(0.5)
    else:
        predictions.append(0.5)
    
    # Kaydet
    X_features.append(feature_values)
    X_progressive.append(predictions[0])
    X_ultra.append(predictions[1])
    X_xgboost.append(predictions[2])
    X_autogluon.append(predictions[3])
    X_tabnet.append(predictions[4])
    y_true.append(1 if target >= 1.5 else 0)

# Array'lere çevir
X_progressive = np.array(X_progressive).reshape(-1, 1)
X_ultra = np.array(X_ultra).reshape(-1, 1)
X_xgboost = np.array(X_xgboost).reshape(-1, 1)
X_autogluon = np.array(X_autogluon).reshape(-1, 1)
X_tabnet = np.array(X_tabnet).reshape(-1, 1)
y_true = np.array(y_true)

print(f"\n✅ {len(y_true)} tahmin toplandı")

# =============================================================================
# META-MODEL INPUT OLUŞTUR
# =============================================================================
print("\n📊 Meta-model input oluşturuluyor...")

X_meta = np.concatenate([X_progressive, X_ultra, X_xgboost, X_autogluon, X_tabnet], axis=1)
print(f"Meta-model input shape: {X_meta.shape}")

# Kronolojik Split
test_size = int(len(X_meta) * 0.2)
train_end = len(X_meta) - test_size

X_train = X_meta[:train_end]
X_test = X_meta[train_end:]
y_train = y_true[:train_end]
y_test = y_true[train_end:]

print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

# =============================================================================
# META-MODEL TRAINING (XGBoost)
# =============================================================================
print("\n🎯 Meta-model eğitiliyor...")

meta_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    random_state=42,
    eval_metric='logloss'
)

cv_scores = cross_val_score(meta_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

meta_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

# =============================================================================
# EVALUATION (2 MODLU)
# =============================================================================
print("\n" + "="*70)
print(f"📊 META-MODEL EVALUATION")
print("="*70)

# Tahminler (Olasılıklar)
y_pred_proba = meta_model.predict_proba(X_test)[:, 1]

# 1. Normal Mod (0.85)
y_pred_normal = (y_pred_proba >= THRESHOLD_NORMAL).astype(int)
acc_normal = accuracy_score(y_test, y_pred_normal)

# 2. Rolling Mod (0.95)
y_pred_rolling = (y_pred_proba >= THRESHOLD_ROLLING).astype(int)
acc_rolling = accuracy_score(y_test, y_pred_rolling)

print(f"🎯 Normal Mod Accuracy ({THRESHOLD_NORMAL}): {acc_normal*100:.2f}%")
print(f"🚀 Rolling Mod Accuracy ({THRESHOLD_ROLLING}): {acc_rolling*100:.2f}%")

# Confusion Matrix (Normal Mod)
cm = confusion_matrix(y_test, y_pred_normal)
if cm.sum() > 0:
    money_loss_risk = cm[0,1] / (cm[0,0] + cm[0,1])
    print(f"💰 Para Kaybı Riski (Normal): {money_loss_risk*100:.1f}%")

# =============================================================================
# BASE MODELS vs META-MODEL COMPARISON
# =============================================================================
print("\n" + "="*70)
print("📊 BASE MODELS vs META-MODEL (Normal Mod)")
print("="*70)

prog_acc = accuracy_score(y_test, (X_test[:, 0] >= THRESHOLD_NORMAL).astype(int))
ultra_acc = accuracy_score(y_test, (X_test[:, 1] >= THRESHOLD_NORMAL).astype(int))
xgb_acc = accuracy_score(y_test, (X_test[:, 2] >= THRESHOLD_NORMAL).astype(int))

print(f"Progressive:  {prog_acc*100:.2f}%")
print(f"Ultra:        {ultra_acc*100:.2f}%")
print(f"XGBoost:      {xgb_acc*100:.2f}%")
print(f"Meta-Model:   {acc_normal*100:.2f}% ⭐")

# =============================================================================
# VISUALIZATION
# =============================================================================
print("\n📊 Görselleştirmeler oluşturuluyor...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=axes[0])
axes[0].set_title('Meta-Model Confusion Matrix (Normal Mod)')

# Prediction Distribution
axes[1].hist(y_pred_proba, bins=50, alpha=0.7, color='#3498db')
axes[1].set_title('Meta-Model Prediction Distribution')
axes[1].axvline(x=THRESHOLD_NORMAL, color='red', linestyle='--', label='Normal (0.85)')
axes[1].axvline(x=THRESHOLD_ROLLING, color='green', linestyle='--', label='Rolling (0.95)')
axes[1].legend()

plt.tight_layout()
plt.savefig('meta_model_evaluation.png')

# =============================================================================
# SAVE META-MODEL
# =============================================================================
print("\n💾 Meta-model kaydediliyor...")
os.makedirs('models', exist_ok=True)
meta_model.save_model('models/meta_model.json')

feature_names = ['Progressive Prob', 'Ultra Prob', 'XGBoost Prob', 'AutoGluon Prob', 'TabNet High X Prob']
importance = meta_model.feature_importances_

model_info = {
    'model': 'XGBoost Meta-Model',
    'version': '2.0',
    'thresholds': {'normal': THRESHOLD_NORMAL, 'rolling': THRESHOLD_ROLLING},
    'metrics': {
        'normal_acc': float(acc_normal),
        'rolling_acc': float(acc_rolling),
        'money_loss_risk': float(money_loss_risk) if cm.sum() > 0 else 0.0
    },
    'feature_importance': {name: float(imp) for name, imp in zip(feature_names, importance)}
}

with open('models/meta_model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

# Google Colab Download
try:
    from google.colab import files
    files.download('models/meta_model.json')
    files.download('models/meta_model_info.json')
    files.download('meta_model_evaluation.png')
except:
    print("⚠️ Colab dışında - dosyalar sadece kaydedildi")

print("\n🎉 META-MODEL TRAINING TAMAMLANDI!")
print("="*70)
