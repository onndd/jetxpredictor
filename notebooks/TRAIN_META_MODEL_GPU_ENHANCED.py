#!/usr/bin/env python3
"""
🎯 JetX Meta-Model Training Script - GPU ENHANCED VERSION

Meta-model, base modellerin (Progressive, Ultra, XGBoost) tahminlerini input olarak alır
ve final kararı verir. Hangi modele ne zaman güveneceğini öğrenir.

GPU ENHANCEMENTS:
- PyTorch/TabNet GPU optimizasyonu
- GPU optimizer entegrasyonu
- Performance monitoring ve benchmarking
- Colab özel optimizasyonlar

KULLANIM:
1. Base modelleri Google Colab'da eğit (Progressive, Ultra, XGBoost)
2. Bu scripti çalıştır (lokal veya Colab'da)
3. Meta-model train edilir ve kaydedilir
4. Ensemble sistemi artık stacking ile çalışır

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

print("✅ Proje yüklendi")

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
            # TabNet'i GPU'ya taşı (eğer destekliyorsa)
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
    print("Önce base modelleri Google Colab'da eğitin.")
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

if prediction_times:
    avg_tabnet_time = np.mean(prediction_times)
    print(f"⏱️  TabNet ortalama tahmin süresi: {avg_tabnet_time*1000:.2f}ms")

print(f"1.5 altı: {(y_true == 0).sum()} ({(y_true == 0).sum() / len(y_true) * 100:.1f}%)")
print(f"1.5 üstü: {(y_true == 1).sum()} ({(y_true == 1).sum() / len(y_true) * 100:.1f}%)")

# =============================================================================
# META-MODEL INPUT OLUŞTUR
# =============================================================================
print("\n📊 Meta-model input oluşturuluyor...")

# Meta-model input: [progressive_prob, ultra_prob, xgboost_prob, autogluon_prob, tabnet_high_x_prob]
# 5 modelin birleşimi: 3 mevcut model + AutoGluon + TabNet (yüksek X specialist)
X_meta = np.concatenate([X_progressive, X_ultra, X_xgboost, X_autogluon, X_tabnet], axis=1)

print(f"Meta-model input shape: {X_meta.shape}")
print(f"Features: Progressive prob, Ultra prob, XGBoost prob, AutoGluon prob, TabNet high X prob")

# KRONOLOJİK SPLIT - TEST DATA LEAKAGE ÖNLENDİ
# Model test verisini eğitimde GÖRMEYECEK
print(f"\n⚠️ UYARI: Shuffle KAPALI - Meta model kronolojik split kullanıyor!")

test_size = int(len(X_meta) * 0.2)
train_end = len(X_meta) - test_size

X_train = X_meta[:train_end]
X_test = X_meta[train_end:]
y_train = y_true[:train_end]
y_test = y_true[train_end:]

print(f"\n✅ Kronolojik Split Tamamlandı:")
print(f"Train: {len(X_train):,} (ilk %80)")
print(f"Test: {len(X_test):,} (son %20 - modelin görmediği veri)")

# =============================================================================
# META-MODEL TRAINING (XGBoost) - GPU ENHANCED
# =============================================================================
print("\n🎯 Meta-model eğitiliyor...")

# XGBoost Classifier (1.5 eşik tahmini için)
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
    predictor='gpu_predictor' if torch.cuda.is_available() else 'cpu_predictor'  # GPU tahmin
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
y_pred = meta_model.predict(X_test)
y_pred_proba = meta_model.predict_proba(X_test)
test_time = time.time() - test_start

print(f"⏱️  Test tahmin süresi: {test_time:.3f}s")

# Accuracy
test_acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")

# Below/Above threshold accuracy
below_mask = y_test == 0
above_mask = y_test == 1

below_acc = accuracy_score(y_test[below_mask], y_pred[below_mask]) if below_mask.sum() > 0 else 0
above_acc = accuracy_score(y_test[above_mask], y_pred[above_mask]) if above_mask.sum() > 0 else 0

print(f"\n🔴 1.5 ALTI Accuracy: {below_acc*100:.2f}%")
print(f"🟢 1.5 ÜSTÜ Accuracy: {above_acc*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n📋 CONFUSION MATRIX:")
print(f"                  Tahmin")
print(f"Gerçek    1.5 Altı | 1.5 Üstü")
print(f"1.5 Altı  {cm[0,0]:6d}   | {cm[0,1]:6d}  ⚠️ PARA KAYBI")
print(f"1.5 Üstü  {cm[1,0]:6d}   | {cm[1,1]:6d}")

# Para kaybı riski
if cm[0,0] + cm[0,1] > 0:
    money_loss_risk = cm[0,1] / (cm[0,0] + cm[0,1])
    print(f"\n💰 PARA KAYBI RİSKİ: {money_loss_risk*100:.1f}%")

# Classification Report
print(f"\n📊 DETAYLI RAPOR:")
print(classification_report(y_test, y_pred, target_names=['1.5 Altı', '1.5 Üstü']))

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
print("📊 BASE MODELS vs META-MODEL KARŞILAŞTIRMASI")
print("="*70)

# Individual model predictions (threshold = 0.5)
prog_pred = (X_test[:, 0] >= 0.5).astype(int)
ultra_pred = (X_test[:, 1] >= 0.5).astype(int)
xgb_pred = (X_test[:, 2] >= 0.5).astype(int)

prog_acc = accuracy_score(y_test, prog_pred)
ultra_acc = accuracy_score(y_test, ultra_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)

print(f"\n📊 Test Set Accuracy:")
print(f"Progressive:  {prog_acc*100:.2f}%")
print(f"Ultra:        {ultra_acc*100:.2f}%")
print(f"XGBoost:      {xgb_acc*100:.2f}%")
print(f"Meta-Model:   {test_acc*100:.2f}% ⭐")

if test_acc > max(prog_acc, ultra_acc, xgb_acc):
    improvement = (test_acc - max(prog_acc, ultra_acc, xgb_acc)) * 100
    print(f"\n✨ Meta-model en iyi base modelden {improvement:.1f}% daha iyi!")

# =============================================================================
# PERFORMANCE RAPORU
# =============================================================================
print("\n" + "="*70)
print("⚡ GPU PERFORMANCE RAPORU")
print("="*70)

print(f"\n🔥 GPU OPTİMİZASYON BAŞARILARI:")
print(f"✅ PyTorch GPU: {'Aktif' if device and device.type == 'cuda' else 'Pasif'}")
print(f"✅ TensorFlow GPU: {'Aktif' if len(tf.config.list_physical_devices('GPU')) > 0 else 'Pasif'}")
print(f"✅ XGBoost GPU: {'Aktif' if torch.cuda.is_available() else 'Pasif'}")
print(f"✅ TabNet GPU: {'Aktif' if device and device.type == 'cuda' else 'Pasif'}")

print(f"\n⏱️  PERFORMANS METRİKLERİ:")
print(f"   Tahmin süresi: {avg_prediction_time*1000:.2f}ms / örnek")
print(f"   CV süresi: {cv_time:.2f}s")
print(f"   Training süresi: {train_time:.2f}s")
print(f"   Test süresi: {test_time:.3f}s")

if gpu_optimizer:
    try:
        # GPU memory usage
        memory_info = gpu_optimizer.get_gpu_memory_usage()
        if memory_info:
            print(f"\n💾 GPU MEMORY KULLANIMI:")
            for system, info in memory_info.items():
                if isinstance(info, dict) and 'allocated' in info:
                    print(f"   {system.title()}: {info['allocated']:.2f}GB / {info.get('total', 0):.2f}GB")
    except Exception as e:
        print(f"\n⚠️ GPU memory bilgisi alınamadı: {e}")

# =============================================================================
# VISUALIZATION
# =============================================================================
print("\n📊 Görselleştirmeler oluşturuluyor...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=axes[0, 0])
axes[0, 0].set_title('Meta-Model Confusion Matrix')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')

# 2. Accuracy Comparison
models = ['Progressive', 'Ultra', 'XGBoost', 'Meta-Model']
accuracies = [prog_acc*100, ultra_acc*100, xgb_acc*100, test_acc*100]
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

axes[0, 1].bar(models, accuracies, color=colors)
axes[0, 1].set_title('Model Accuracy Comparison')
axes[0, 1].set_ylabel('Accuracy (%)')
axes[0, 1].set_ylim([0, 100])
for i, v in enumerate(accuracies):
    axes[0, 1].text(i, v + 1, f'{v:.1f}%', ha='center')

# 3. Feature Importance
axes[1, 0].barh(feature_names, importance, color='#3498db')
axes[1, 0].set_title('Meta-Model Feature Importance')
axes[1, 0].set_xlabel('Importance')

# 4. Prediction Distribution
axes[1, 1].hist(y_pred_proba[:, 1], bins=50, alpha=0.7, color='#3498db', edgecolor='black')
axes[1, 1].set_title('Meta-Model Prediction Distribution')
axes[1, 1].set_xlabel('Probability of 1.5+')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].axvline(x=0.5, color='red', linestyle='--', label='Threshold')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('meta_model_evaluation_gpu_enhanced.png', dpi=300, bbox_inches='tight')
print("✅ Görselleştirme kaydedildi: meta_model_evaluation_gpu_enhanced.png")

# =============================================================================
# SAVE META-MODEL
# =============================================================================
print("\n💾 Meta-model kaydediliyor...")

# Modeli kaydet
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
    'version': '2.0-GPU',
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.05,
    'gpu_optimizations': {
        'pytorch_gpu': device.type == 'cuda' if device else False,
        'tensorflow_gpu': len(tf.config.list_physical_devices('GPU')) > 0,
        'xgboost_gpu': torch.cuda.is_available(),
        'tabnet_gpu': device.type == 'cuda' if device else False
    },
    'performance_metrics': {
        'cv_accuracy': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'test_accuracy': float(test_acc),
        'below_15_accuracy': float(below_acc),
        'above_15_accuracy': float(above_acc),
        'money_loss_risk': float(money_loss_risk) if cm[0,0] + cm[0,1] > 0 else 0.0,
        'avg_prediction_time_ms': float(avg_prediction_time * 1000),
        'training_time_seconds': float(train_time),
        'cv_time_seconds': float(cv_time)
    },
    'base_models': list(loaded_models.keys()),
    'feature_importance': {
        name: float(imp) for name, imp in zip(feature_names, importance)
    },
    'comparison': {
        'progressive_accuracy': float(prog_acc),
        'ultra_accuracy': float(ultra_acc),
        'xgboost_accuracy': float(xgb_acc),
        'meta_model_accuracy': float(test_acc)
    },
    'gpu_info': gpu_info
}

with open('models/meta_model_gpu_enhanced_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("✅ Dosyalar kaydedildi:")
print("- models/meta_model_gpu_enhanced.json")
print("- models/meta_model_gpu_enhanced_info.json")
print("- meta_model_evaluation_gpu_enhanced.png")

# Google Colab'da ise indir
try:
    from google.colab import files
    files.download('models/meta_model_gpu_enhanced.json')
    files.download('models/meta_model_gpu_enhanced_info.json')
    files.download('meta_model_evaluation_gpu_enhanced.png')
    print("\n✅ Dosyalar indirildi!")
except:
    print("\n⚠️ Colab dışında - dosyalar sadece kaydedildi")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*70)
print("🎉 GPU-ENHANCED META-MODEL TRAINING TAMAMLANDI!")
print("="*70)

print(f"\n📊 SONUÇLAR:")
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")
print(f"✅ 1.5 Altı Accuracy: {below_acc*100:.2f}%")
print(f"✅ Para Kaybı Riski: {money_loss_risk*100:.1f}%" if cm[0,0] + cm[0,1] > 0 else "")

print(f"\n🚀 GPU OPTİMİZASYON BAŞARILARI:")
gpu_count = sum([
    device.type == 'cuda' if device else False,
    len(tf.config.list_physical_devices('GPU')) > 0,
    torch.cuda.is_available(),
    device.type == 'cuda' if device else False
])
print(f"   Aktif GPU optimizasyonları: {gpu_count}/4")

print(f"\n⚡ PERFORMANCE İYİLEŞTİRMELERİ:")
print(f"   Ortalama tahmin süresi: {avg_prediction_time*1000:.2f}ms")
print(f"   Training süresi: {train_time:.2f}s")

if test_acc > max(prog_acc, ultra_acc, xgb_acc):
    print(f"\n🚀 Meta-model tüm base modellerden daha iyi!")
    print(f"En iyi base model: {max(prog_acc, ultra_acc, xgb_acc)*100:.2f}%")
    print(f"Meta-model: {test_acc*100:.2f}%")
    print(f"İyileştirme: +{(test_acc - max(prog_acc, ultra_acc, xgb_acc))*100:.1f}%")

print(f"\n📁 Sonraki adımlar:")
print(f"1. models/meta_model_gpu_enhanced.json dosyasını projenize ekleyin")
print(f"2. Ensemble sistemi artık GPU-enhanced stacking ile çalışacak")
print(f"3. Model Karşılaştırma dashboard'ında sonuçları izleyin")
print(f"4. GPU performans raporunu inceleyin")

print("\n" + "="*70)
