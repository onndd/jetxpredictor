#!/usr/bin/env python3
"""
🎯 JetX Meta-Model Training Script - GPU ENHANCED VERSION (v2.1)

GÜNCELLEME:
- 2 Modlu Yapı (Normal/Rolling) entegrasyonu.
- Güven Eşiği: %85 (Normal Mod).
- GPU Optimizasyonları korundu.

Meta-model, base modellerin (Progressive, Ultra, XGBoost) tahminlerini input olarak alır
ve final kararı verir. Hangi modele ne zaman güveneceğini öğrenir.

Süre: ~30 dakika
"""

import subprocess
import sys
import os
import time
from datetime import datetime

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

# =============================================================================
# GPU OPTIMIZER ENTEGRASYONU
# =============================================================================
try:
    # GPU optimizer'ı import et
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.gpu_optimizer import setup_colab_gpu_optimization, get_gpu_optimizer
    
    print("\n🚀 GPU OPTİMİZASYONU BAŞLATILIYOR...")
    gpu_results = setup_colab_gpu_optimization()
    
    # GPU optimizer instance
    gpu_optimizer = get_gpu_optimizer()
    
    # Performance monitoring
    print("📊 GPU performansı izleniyor...")
    gpu_optimizer.monitor_gpu_usage(duration_seconds=3)
    
except ImportError as e:
    print(f"⚠️ GPU optimizer import edilemedi: {e}")
    gpu_optimizer = None
except Exception as e:
    print(f"⚠️ GPU optimizasyonu başarısız: {e}")
    gpu_optimizer = None

# =============================================================================
# PYTORCH GPU OPTİMİZASYONU
# =============================================================================
try:
    import torch
    
    # GPU device detection
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ PyTorch GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # Memory management
        torch.cuda.empty_cache()
        print("✅ PyTorch GPU memory cache temizlendi")
    else:
        device = torch.device('cpu')
        print("⚠️ PyTorch GPU not available, using CPU")
        
except ImportError:
    print("⚠️ PyTorch not installed")
    device = None

# =============================================================================
# TENSORFLOW GPU OPTİMİZASYONU
# =============================================================================
try:
    import tensorflow as tf
    
    # TensorFlow GPU configuration
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Mixed precision
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            
            print(f"✅ TensorFlow GPU: {len(gpus)} GPU detected and optimized")
            print(f"   - Memory growth: Aktif")
            print(f"   - Mixed precision: Aktif")
        except Exception as e:
            print(f"⚠️ TensorFlow GPU configuration failed: {e}")
    else:
        print("⚠️ TensorFlow GPU not detected")
        
except ImportError:
    print("⚠️ TensorFlow not installed")

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

print(f"✅ Proje yüklendi - Eşik: {THRESHOLD_NORMAL}")

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

# TabNet (Yüksek X Specialist) - GPU ENHANCED
try:
    if os.path.exists(model_paths['tabnet']['model']):
        from pytorch_tabnet.tab_model import TabNetClassifier
        
        tabnet_model = TabNetClassifier()
        
        # GPU optimizasyonu
        if device and device.type == 'cuda':
            print("🔥 TabNet GPU optimizasyonu aktif...")
            try:
                tabnet_model.device = device
                print(f"✅ TabNet GPU device: {device}")
            except:
                print("⚠️ TabNet GPU taşınamadı, CPU ile devam ediliyor")
        
        tabnet_model.load_model(model_paths['tabnet']['model'])
        
        loaded_models['tabnet'] = {
            'model': tabnet_model,
            'scaler': joblib.load(model_paths['tabnet']['scaler']) if os.path.exists(model_paths['tabnet']['scaler']) else None
        }
        print("✅ TabNet modeli yüklendi (Yüksek X Specialist)")
        if device and device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ TabNet model bulunamadı, atlanıyor")
except Exception as e:
    print(f"⚠️ TabNet model yüklenemedi: {e}")

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

# Performance tracking
start_time = time.time()
prediction_times = []

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
            
            # Sequences
            seq_50 = np.log10(np.array(history[-50:]).reshape(1, 50, 1) + 1e-8)
            seq_200 = np.log10(np.array(history[-200:]).reshape(1, 200, 1) + 1e-8)
            seq_500 = np.log10(np.array(history[-500:]).reshape(1, 500, 1) + 1e-8)
            seq_1000 = np.log10(np.array(history[-1000:]).reshape(1, 1000, 1) + 1e-8)
            
            pred = loaded_models['progressive']['model'].predict(
                [scaled_features, seq_50, seq_200, seq_500, seq_1000],
                verbose=0
            )
            
            # Threshold probability (3rd output)
            threshold_prob = float(pred[2][0][0])
            predictions.append(threshold_prob)
        except:
            predictions.append(0.5)  # Neutral
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
            
            # Threshold output (3. output)
            threshold_prob = float(pred[2][0][0])
            predictions.append(threshold_prob)
        except Exception as e:
            print(f"⚠️ Ultra prediction hatası: {e}")
            predictions.append(0.5)
    else:
        predictions.append(0.5)
    
    # XGBoost prediction
    if 'xgboost' in loaded_models:
        try:
            scaler = loaded_models['xgboost']['scaler']
            scaled_features = scaler.transform(feature_values.reshape(1, -1))
            
            pred_proba = loaded_models['xgboost']['classifier'].predict_proba(scaled_features)
            threshold_prob = float(pred_proba[0][1])  # 1.5 üstü probability
            predictions.append(threshold_prob)
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
            threshold_prob = float(pred_proba.iloc[0, 1])  # 1.5 üstü probability
            predictions.append(threshold_prob)
        except:
            predictions.append(0.5)
    else:
        predictions.append(0.5)
    
    # TabNet prediction (yüksek X specialist) - GPU ENHANCED
    if 'tabnet' in loaded_models:
        try:
            pred_start = time.time()
            
            if loaded_models['tabnet']['scaler'] is not None:
                scaled_features = loaded_models['tabnet']['scaler'].transform(feature_values.reshape(1, -1))
            else:
                scaled_features = feature_values.reshape(1, -1)
            
            # GPU optimizasyonlu tahmin
            if device and device.type == 'cuda':
                # GPU memory tracking
                with torch.cuda.device(device):
                    pred_proba = loaded_models['tabnet']['model'].predict_proba(scaled_features)
            else:
                pred_proba = loaded_models['tabnet']['model'].predict_proba(scaled_features)
            
            # Yüksek X olasılığı (kategori 2 ve 3'ün toplamı: 10x+)
            high_x_prob = float(pred_proba[0][2] + pred_proba[0][3])
            predictions.append(high_x_prob)
            
            # Performance tracking
            pred_time = time.time() - pred_start
            prediction_times.append(pred_time)
            
        except Exception as e:
            print(f"⚠️ TabNet prediction hatası: {e}")
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
    
    # Target: 1.5 eşik
    y_true.append(1 if target >= 1.5 else 0)

# Array'lere çevir
X_features = np.array(X_features)
X_progressive = np.array(X_progressive).reshape(-1, 1)
X_ultra = np.array(X_ultra).reshape(-1, 1)
X_xgboost = np.array(X_xgboost).reshape(-1, 1)
X_autogluon = np.array(X_autogluon).reshape(-1, 1)
X_tabnet = np.array(X_tabnet).reshape(-1, 1)
y_true = np.array(y_true)

# Performance metrics
total_prediction_time = time.time() - start_time
avg_prediction_time = total_prediction_time / len(y_true)

print(f"\n✅ {len(y_true)} tahmin toplandı")
print(f"⏱️  Toplam tahmin süresi: {total_prediction_time:.2f}s")
print(f"⏱️  Ortalama tahmin süresi: {avg_prediction_time*1000:.2f}ms")

# =============================================================================
# META-MODEL INPUT OLUŞTUR
# =============================================================================
print("\n📊 Meta-model input oluşturuluyor...")

X_meta = np.concatenate([X_progressive, X_ultra, X_xgboost, X_autogluon, X_tabnet], axis=1)
print(f"Meta-model input shape: {X_meta.shape}")

# KRONOLOJİK SPLIT
print(f"\n⚠️ UYARI: Shuffle KAPALI - Meta model kronolojik split kullanıyor!")

test_size = int(len(X_meta) * 0.2)
train_end = len(X_meta) - test_size

X_train = X_meta[:train_end]
X_test = X_meta[train_end:]
y_train = y_true[:train_end]
y_test = y_true[train_end:]

print(f"\n✅ Kronolojik Split Tamamlandı:")
print(f"Train: {len(X_train):,}")
print(f"Test: {len(X_test):,}")

# =============================================================================
# META-MODEL TRAINING (XGBoost) - GPU ENHANCED
# =============================================================================
print("\n🎯 Meta-model eğitiliyor...")

# XGBoost Classifier
meta_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    random_state=42,
    eval_metric='logloss',
    tree_method='hist',  # GPU optimizasyonu için
    predictor='gpu_predictor' if torch.cuda.is_available() else 'cpu_predictor'
)

# GPU kontrolü
if torch.cuda.is_available():
    print("🔥 XGBoost GPU optimizasyonu aktif:")
    print(f"   - tree_method: hist")
    print(f"   - predictor: gpu_predictor")
else:
    print("⚠️ XGBoost CPU modunda çalışıyor")

# Cross-validation
print("\n📊 Cross-validation yapılıyor...")
cv_start = time.time()
cv_scores = cross_val_score(
    meta_model, X_train, y_train,
    cv=5, scoring='accuracy'
)
cv_time = time.time() - cv_start

print(f"CV Accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
print(f"CV Süresi: {cv_time:.2f}s")

# Train
print("\n🚀 Final training başlıyor...")
train_start = time.time()
meta_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)
train_time = time.time() - train_start

print(f"✅ Training tamamlandı!")
print(f"⏱️  Training süresi: {train_time:.2f}s")

# =============================================================================
# EVALUATION
# =============================================================================
print("\n" + "="*70)
print("📊 META-MODEL EVALUATION")
print("="*70)

# Test predictions
test_start = time.time()
y_pred_proba = meta_model.predict_proba(X_test)
# GÜNCELLEME: 0.85 EŞİĞİ
y_pred = (y_pred_proba[:, 1] >= THRESHOLD_NORMAL).astype(int)
test_time = time.time() - test_start

# Accuracy
test_acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy (Normal Mod - 0.85): {test_acc*100:.2f}%")

# Below/Above threshold accuracy
below_mask = y_test == 0
above_mask = y_test == 1

below_acc = accuracy_score(y_test[below_mask], y_pred[below_mask]) if below_mask.sum() > 0 else 0
above_acc = accuracy_score(y_test[above_mask], y_pred[above_mask]) if above_mask.sum() > 0 else 0

print(f"\n🔴 1.5 ALTI Accuracy: {below_acc*100:.2f}%")
print(f"🟢 1.5 ÜSTÜ Accuracy: {above_acc*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n📋 CONFUSION MATRIX (Normal Mod):")
print(f"                Tahmin")
print(f"Gerçek    1.5 Altı | 1.5 Üstü")
print(f"1.5 Altı  {cm[0,0]:6d}   | {cm[0,1]:6d}  ⚠️ PARA KAYBI")
print(f"1.5 Üstü  {cm[1,0]:6d}   | {cm[1,1]:6d}")

# Para kaybı riski
if cm[0,0] + cm[0,1] > 0:
    money_loss_risk = cm[0,1] / (cm[0,0] + cm[0,1])
    print(f"\n💰 PARA KAYBI RİSKİ: {money_loss_risk*100:.1f}%")

# Feature Importance
print(f"\n🎯 MODEL FEATURE IMPORTANCE:")
feature_names = ['Progressive Prob', 'Ultra Prob', 'XGBoost Prob', 'AutoGluon Prob', 'TabNet High X Prob']
importance = meta_model.feature_importances_

for name, imp in zip(feature_names, importance):
    print(f"  {name}: {imp:.3f}")

# =============================================================================
# BASE MODELS vs META-MODEL COMPARISON
# =============================================================================
print("\n" + "="*70)
print("📊 BASE MODELS vs META-MODEL KARŞILAŞTIRMASI (Eşik: 0.85)")
print("="*70)

# Individual model predictions (threshold = 0.85)
prog_pred = (X_test[:, 0] >= THRESHOLD_NORMAL).astype(int)
ultra_pred = (X_test[:, 1] >= THRESHOLD_NORMAL).astype(int)
xgb_pred = (X_test[:, 2] >= THRESHOLD_NORMAL).astype(int)

prog_acc = accuracy_score(y_test, prog_pred)
ultra_acc = accuracy_score(y_test, ultra_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)

print(f"\n📊 Test Set Accuracy:")
print(f"Progressive:  {prog_acc*100:.2f}%")
print(f"Ultra:        {ultra_acc*100:.2f}%")
print(f"XGBoost:      {xgb_acc*100:.2f}%")
print(f"Meta-Model:   {test_acc*100:.2f}% ⭐")

# =============================================================================
# SAVE META-MODEL
# =============================================================================
print("\n💾 Meta-model kaydediliyor...")

os.makedirs('models', exist_ok=True)
meta_model.save_model('models/meta_model_gpu_enhanced.json')

# GPU optimizasyon raporu
gpu_info = {}
if gpu_optimizer:
    gpu_info = gpu_optimizer.gpu_info

# Model bilgilerini kaydet
import json

model_info = {
    'model': 'XGBoost Meta-Model - GPU Enhanced',
    'version': '2.1-GPU',
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'thresholds': {'normal': THRESHOLD_NORMAL, 'rolling': THRESHOLD_ROLLING},
    'performance_metrics': {
        'test_accuracy': float(test_acc),
        'below_15_accuracy': float(below_acc),
        'above_15_accuracy': float(above_acc),
        'money_loss_risk': float(money_loss_risk) if cm[0,0] + cm[0,1] > 0 else 0.0
    },
    'base_models': list(loaded_models.keys()),
    'gpu_info': gpu_info
}

with open('models/meta_model_gpu_enhanced_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("✅ Dosyalar kaydedildi.")

# Google Colab'da ise indir
try:
    from google.colab import files
    files.download('models/meta_model_gpu_enhanced.json')
    files.download('models/meta_model_gpu_enhanced_info.json')
    print("\n✅ Dosyalar indirildi!")
except:
    print("\n⚠️ Colab dışında - dosyalar sadece kaydedildi")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*70)
print("🎉 GPU-ENHANCED META-MODEL TRAINING TAMAMLANDI!")
print("="*70)
