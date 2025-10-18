#!/usr/bin/env python3
"""
🤖 JetX CATBOOST TRAINING - Feature Engineering Bazlı Model (YENİ - FAZ 2)

AMAÇ: CatBoost ile hızlı ve etkili tahmin modeli eğitmek

AVANTAJLAR:
- Çok daha hızlı eğitim (~30-60 dakika vs 2-3 saat)
- Feature importance analizi yapılabilir
- Daha az bellek kullanımı
- Overfitting'e daha dirençli
- Class imbalance için native destek

DEĞİŞİKLİKLER (XGBoost → CatBoost):
- XGBoost → CatBoost kütüphanesi
- scale_pos_weight → class_weights (native support)
- .json → .cbm model formatı
- Çift sanal kasa simülasyonu (Kasa 1: 1.5x + Kasa 2: %80 çıkış)

STRATEJI:
- CatBoostRegressor: Değer tahmini için
- CatBoostClassifier: 1.5 eşik tahmini için (class_weights ile dengeleme)

HEDEFLER:
- 1.5 ALTI Doğruluk: %70-80%+
- 1.5 ÜSTÜ Doğruluk: %70-80%+
- MAE: < 2.0

SÜRE: ~30-60 dakika (GPU ile)
"""

import subprocess
import sys
import os
import time
from datetime import datetime

print("="*80)
print("🤖 JetX CATBOOST TRAINING (FAZ 2)")
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

print(f"✅ CatBoost: İmport edildi")

# Google Drive mount (Colab için)
try:
    from google.colab import drive
    
    if not os.path.exists('/content/drive'):
        print("\n📦 Google Drive bağlanıyor...")
        drive.mount('/content/drive')
    
    # Model kayıt dizini
    DRIVE_MODEL_DIR = '/content/drive/MyDrive/JetX_Models/CatBoost/'
    os.makedirs(DRIVE_MODEL_DIR, exist_ok=True)
    print(f"✅ Google Drive bağlandı: {DRIVE_MODEL_DIR}")
    USE_DRIVE = True
except ImportError:
    print("⚠️ Google Colab dışında - lokal kayıt kullanılacak")
    DRIVE_MODEL_DIR = 'models/'
    USE_DRIVE = False
except Exception as e:
    print(f"⚠️ Google Drive mount hatası: {e}")
    DRIVE_MODEL_DIR = 'models/'
    USE_DRIVE = False

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
# from utils.virtual_bankroll_callback import CatBoostBankrollCallback # Bu callback hatalı ve kullanılmıyor.
from utils.focal_loss import CatBoostFocalLoss
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
print("\n🔧 Feature extraction...")
window_size = 1000  # Progressive NN ile aynı
X_features = []
y_regression = []
y_classification = []

for i in tqdm(range(window_size, len(all_values)-1), desc='Features'):
    hist = all_values[:i].tolist()
    target = all_values[i]
    
    # Tüm özellikleri çıkar
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
print(f"✅ Feature sayısı: {X.shape[1]}")

# =============================================================================
# NORMALIZASYON
# =============================================================================
print("\n📊 Normalizasyon...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

# =============================================================================
# TIME-SERIES SPLIT (KRONOLOJIK) - SHUFFLE YOK!
# =============================================================================
print("\n📊 TIME-SERIES SPLIT (Kronolojik Bölme)...")
print("⚠️  UYARI: Shuffle devre dışı - Zaman serisi yapısı korunuyor!")

# Test seti: Son 1000 kayıt
test_size = 1000
train_end = len(X) - test_size

# Train/Test split (kronolojik)
X_train = X[:train_end]
X_test = X[train_end:]
y_reg_train = y_reg[:train_end]
y_reg_test = y_reg[train_end:]
y_cls_train = y_cls[:train_end]
y_cls_test = y_cls[train_end:]

print(f"✅ Train: {len(X_train):,}")
print(f"✅ Test: {len(X_test):,} (tüm verinin son {test_size} kaydı)")
print(f"📊 Toplam: {len(X_train) + len(X_test):,}")

# Validation için train setini böl (kronolojik)
val_size = int(len(X_train) * 0.2)
val_start = len(X_train) - val_size

X_tr = X_train[:val_start]
X_val = X_train[val_start:]
y_reg_tr = y_reg_train[:val_start]
y_reg_val = y_reg_train[val_start:]
y_cls_tr = y_cls_train[:val_start]
y_cls_val = y_cls_train[val_start:]

print(f"   ├─ Actual Train: {len(X_tr):,}")
print(f"   └─ Validation: {len(X_val):,} (train'in son %20'si)")

# =============================================================================
# CATBOOST REGRESSOR (Değer Tahmini)
# =============================================================================
print("\n" + "="*80)
print("🎯 CATBOOST REGRESSOR EĞİTİMİ (Değer Tahmini)")
print("="*80)

reg_start = time.time()

# CatBoost parametreleri - OPTIMIZE EDİLDİ + EARLY STOPPING KALDIRILDI
regressor_params = {
    'iterations': 1500,           # 500 → 1500 (3x artış)
    'depth': 10,                  # 8 → 10 (daha derin ağaçlar)
    'learning_rate': 0.03,        # 0.05 → 0.03 (daha stabil)
    'l2_leaf_reg': 5,             # YENİ: Overfitting önleme
    'bootstrap_type': 'Bernoulli',  # YENİ: subsample için gerekli
    'subsample': 0.8,             # YENİ: Stochastic gradient
    'loss_function': 'MAE',
    'eval_metric': 'MAE',
    'verbose': 100,               # 50 → 100 (daha az log)
    'random_state': 42,
    **catboost_gpu_config  # GPU konfigürasyonunu ekle
}
regressor = CatBoostRegressor(**regressor_params)

print("📊 Model Parametreleri (Optimize):")
print(f"  iterations: 1500 (500 → 1500)")
print(f"  depth: 10 (8 → 10)")
print(f"  learning_rate: 0.03 (0.05 → 0.03)")
print(f"  l2_leaf_reg: 5 (YENİ)")
print(f"  bootstrap_type: Bernoulli (YENİ - subsample için)")
print(f"  subsample: 0.8 (YENİ)")
print(f"  loss_function: MAE")
print(f"  task_type: GPU (varsa)")
print(f"  early_stopping_rounds: Yok (Tüm 1500 iterasyon tamamlanacak) ✅\n")

# Hatalı Virtual Bankroll Callback kaldırıldı.
# Eğitim sonunda zaten daha kapsamlı bir simülasyon yapılıyor.

# Eğitim
print("🔥 CatBoost Regressor eğitimi başlıyor...")
regressor.fit(
    X_tr, y_reg_tr,
    eval_set=(X_val, y_reg_val),  # ✅ KRONOLOJIK VALIDATION!
    verbose=100
)

reg_time = time.time() - reg_start
print(f"\n✅ Regressor eğitimi tamamlandı! Süre: {reg_time/60:.1f} dakika")

# Değerlendirme
y_reg_pred = regressor.predict(X_test)
mae_reg = mean_absolute_error(y_reg_test, y_reg_pred)
rmse_reg = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))

print(f"\n📊 REGRESSOR PERFORMANSI:")
print(f"  MAE: {mae_reg:.4f}")
print(f"  RMSE: {rmse_reg:.4f}")

# Feature importance (Top 15)
feature_names = list(FeatureEngineering.extract_all_features(all_values[:1000].tolist()).keys())
importances = regressor.feature_importances_
top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:15]

print(f"\n📊 TOP 15 ÖNEMLİ ÖZELLIKLER:")
for i, (feat, imp) in enumerate(top_features, 1):
    print(f"  {i:2d}. {feat:30s}: {imp:.4f}")

# =============================================================================
# CATBOOST CLASSIFIER (Eşik Tahmini)
# =============================================================================
print("\n" + "="*80)
print("🎯 CATBOOST CLASSIFIER EĞİTİMİ (1.5 Eşik Tahmini)")
print("="*80)

cls_start = time.time()

# Class weights - CatBoost native support
below_count = (y_cls_train == 0).sum()
above_count = (y_cls_train == 1).sum()

# CatBoost için class_weights parametresi (native)
# Focal Loss ile birlikte daha güçlü bir etki için manuel ağırlıklandırma deniyoruz.
class_weights = [20.0, 1.0]  # CatBoost class_weights'i liste olarak bekler [class_0_weight, class_1_weight]

print(f"📊 CLASS WEIGHTS (CatBoost Native - TIME-SERIES SPLIT):")
print(f"  1.5 altı (class 0): {class_weights[0]:.1f}x (agresif - lazy learning önleme)")
print(f"  1.5 üstü (class 1): {class_weights[1]:.1f}x")
print(f"  Toplam 1.5 altı: {below_count:,} örnek")
print(f"  Toplam 1.5 üstü: {above_count:,} örnek\n")

# CatBoost parametreleri - OPTIMIZE EDİLDİ + EARLY STOPPING KALDIRILDI
classifier_params = {
    'iterations': 1500,           # 500 → 1500 (3x artış)
    'depth': 9,                   # 7 → 9 (daha derin ağaçlar)
    'learning_rate': 0.03,        # 0.05 → 0.03 (daha stabil)
    'l2_leaf_reg': 5,             # YENİ: Overfitting önleme
    'bootstrap_type': 'Bernoulli',  # YENİ: subsample için gerekli
    'subsample': 0.8,             # YENİ: Stochastic gradient
    'loss_function': CatBoostFocalLoss(),  # Logloss -> Focal Loss
    'eval_metric': 'Accuracy',
    'class_weights': class_weights, # Manuel sınıf ağırlıklarını etkinleştir
    # auto_class_weights='Balanced', # Focal Loss ile birlikte kullanılmaz
    'verbose': 100,               # 50 → 100 (daha az log)
    'random_state': 42,
    **catboost_gpu_config  # GPU konfigürasyonunu ekle
}
classifier = CatBoostClassifier(**classifier_params)

print("📊 Model Parametreleri (Optimize):")
print(f"  iterations: 1500 (500 → 1500)")
print(f"  depth: 9 (7 → 9)")
print(f"  learning_rate: 0.03 (0.05 → 0.03)")
print(f"  l2_leaf_reg: 5 (YENİ)")
print(f"  bootstrap_type: Bernoulli (YENİ - subsample için)")
print(f"  subsample: 0.8 (YENİ)")
print(f"  loss_function: Focal Loss (Dengesiz Veri İçin)")
print(f"  auto_class_weights: Devre Dışı (Focal Loss kullanılıyor)")
print(f"  early_stopping_rounds: Yok (Tüm 1500 iterasyon tamamlanacak) ✅\n")

# Hatalı Virtual Bankroll Callback kaldırıldı.
# Eğitim sonunda zaten daha kapsamlı bir simülasyon yapılıyor.

# Eğitim
print("🔥 CatBoost Classifier eğitimi başlıyor...")
classifier.fit(
    X_tr, y_cls_tr,
    eval_set=(X_val, y_cls_val),  # ✅ KRONOLOJIK VALIDATION!
    verbose=100
)

cls_time = time.time() - cls_start
print(f"\n✅ Classifier eğitimi tamamlandı! Süre: {cls_time/60:.1f} dakika")

# Değerlendirme
y_cls_pred = classifier.predict(X_test)
y_cls_proba = classifier.predict_proba(X_test)[:, 1]  # 1.5 üstü olma olasılığı

cls_acc = accuracy_score(y_cls_test, y_cls_pred)

# Sınıf bazında accuracy
below_mask = y_cls_test == 0
above_mask = y_cls_test == 1

below_acc = accuracy_score(y_cls_test[below_mask], y_cls_pred[below_mask]) if below_mask.sum() > 0 else 0
above_acc = accuracy_score(y_cls_test[above_mask], y_cls_pred[above_mask]) if above_mask.sum() > 0 else 0

print(f"\n📊 CLASSIFIER PERFORMANSI:")
print(f"  Genel Accuracy: {cls_acc*100:.2f}%")
print(f"  🔴 1.5 Altı Doğruluk: {below_acc*100:.2f}%")
print(f"  🟢 1.5 Üstü Doğruluk: {above_acc*100:.2f}%")

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
    if fpr < 0.20:
        print(" ✅ HEDEF AŞILDI!")
    else:
        print(f" (Hedef: <20%)")
    
    # Classification Report
    print(f"\n📊 DETAYLI RAPOR:")
    print(classification_report(y_cls_test, y_cls_pred, target_names=['1.5 Altı', '1.5 Üstü']))

# =============================================================================
# ÇİFT SANAL KASA SİMÜLASYONU (YENİ - FAZ 2)
# =============================================================================
print("\n" + "="*80)
print("💰 ÇİFT SANAL KASA SİMÜLASYONU")
print("="*80)

# Dinamik kasa miktarı hesapla
test_count = len(y_reg_test)
initial_bankroll = test_count * 10  # Her test verisi için 10 TL
bet_amount = 10.0

print(f"📊 Test Veri Sayısı: {test_count:,}")
print(f"💰 Başlangıç Kasası: {initial_bankroll:,.2f} TL (dinamik)")
print(f"💵 Bahis Tutarı: {bet_amount:.2f} TL (sabit)")
print()

# =============================================================================
# KASA 1: 1.5x EŞİK SİSTEMİ
# =============================================================================
print("="*80)
print("💰 KASA 1: 1.5x EŞİK SİSTEMİ")
print("="*80)
print("Strateji: Model 1.5x üstü tahmin ederse → 1.5x'te çıkış")
print()

kasa1_wallet = initial_bankroll
kasa1_total_bets = 0
kasa1_total_wins = 0
kasa1_total_losses = 0

for i in range(len(y_reg_test)):
    model_pred_cls = y_cls_pred[i]  # 0 veya 1
    actual_value = y_reg_test[i]
    
    # Model "1.5 üstü" (1) tahmin ediyorsa bahis yap
    if model_pred_cls == 1:
        kasa1_wallet -= bet_amount  # Bahis yap
        kasa1_total_bets += 1
        
        # 1.5x'te çıkış yap
        exit_point = 1.5
        
        # Gerçek değer çıkış noktasından büyük veya eşitse kazandık
        if actual_value >= exit_point:
            # Kazandık! 1.5x × 10 TL = 15 TL geri al
            kasa1_wallet += exit_point * bet_amount
            kasa1_total_wins += 1
        else:
            # Kaybettik (bahis zaten kesildi)
            kasa1_total_losses += 1

# Kasa 1 sonuçları
kasa1_profit_loss = kasa1_wallet - initial_bankroll
kasa1_roi = (kasa1_profit_loss / initial_bankroll) * 100
kasa1_win_rate = (kasa1_total_wins / kasa1_total_bets * 100) if kasa1_total_bets > 0 else 0
kasa1_accuracy = kasa1_win_rate

print(f"\n📊 KASA 1 SONUÇLARI:")
print(f"{'='*70}")
print(f"Toplam Oyun: {kasa1_total_bets:,} el")
print(f"✅ Kazanan: {kasa1_total_wins:,} oyun ({kasa1_win_rate:.1f}%)")
print(f"❌ Kaybeden: {kasa1_total_losses:,} oyun ({100-kasa1_win_rate:.1f}%)")
print(f"")
print(f"💰 Başlangıç Kasası: {initial_bankroll:,.2f} TL")
print(f"💰 Final Kasa: {kasa1_wallet:,.2f} TL")
print(f"📈 Net Kar/Zarar: {kasa1_profit_loss:+,.2f} TL")
print(f"📊 ROI: {kasa1_roi:+.2f}%")
print(f"🎯 Doğruluk (Kazanma Oranı): {kasa1_accuracy:.1f}%")
print(f"{'='*70}\n")

# =============================================================================
# KASA 2: %80 ÇIKIŞ SİSTEMİ (Yeni)
# =============================================================================
print("="*80)
print("💰 KASA 2: %80 ÇIKIŞ SİSTEMİ (Yüksek Tahminler)")
print("="*80)
print("Strateji: Model 2.0x+ tahmin ederse → Tahmin × 0.80'de çıkış")
print()

kasa2_wallet = initial_bankroll
kasa2_total_bets = 0
kasa2_total_wins = 0
kasa2_total_losses = 0
kasa2_exit_points = []  # Çıkış noktalarını kaydet

for i in range(len(y_reg_test)):
    model_pred_value = y_reg_pred[i]  # Tahmin edilen değer
    actual_value = y_reg_test[i]
    
    # SADECE 2.0x ve üzeri tahminlerde oyna
    if model_pred_value >= 2.0:
        kasa2_wallet -= bet_amount  # Bahis yap
        kasa2_total_bets += 1
        
        # Çıkış noktası: Tahmin × 0.80
        exit_point = model_pred_value * 0.80
        kasa2_exit_points.append(exit_point)
        
        # Gerçek değer çıkış noktasından büyük veya eşitse kazandık
        if actual_value >= exit_point:
            # Kazandık! exit_point × 10 TL geri al
            kasa2_wallet += exit_point * bet_amount
            kasa2_total_wins += 1
        else:
            # Kaybettik (bahis zaten kesildi)
            kasa2_total_losses += 1

# Kasa 2 sonuçları
kasa2_profit_loss = kasa2_wallet - initial_bankroll
kasa2_roi = (kasa2_profit_loss / initial_bankroll) * 100
kasa2_win_rate = (kasa2_total_wins / kasa2_total_bets * 100) if kasa2_total_bets > 0 else 0
kasa2_accuracy = kasa2_win_rate
kasa2_avg_exit = np.mean(kasa2_exit_points) if kasa2_exit_points else 0

print(f"\n📊 KASA 2 SONUÇLARI:")
print(f"{'='*70}")
print(f"Toplam Oyun: {kasa2_total_bets:,} el")
print(f"✅ Kazanan: {kasa2_total_wins:,} oyun ({kasa2_win_rate:.1f}%)")
print(f"❌ Kaybeden: {kasa2_total_losses:,} oyun ({100-kasa2_win_rate:.1f}%)")
print(f"")
print(f"💰 Başlangıç Kasası: {initial_bankroll:,.2f} TL")
print(f"💰 Final Kasa: {kasa2_wallet:,.2f} TL")
print(f"📈 Net Kar/Zarar: {kasa2_profit_loss:+,.2f} TL")
print(f"📊 ROI: {kasa2_roi:+.2f}%")
print(f"🎯 Doğruluk (Kazanma Oranı): {kasa2_accuracy:.1f}%")
print(f"📊 Ortalama Çıkış Noktası: {kasa2_avg_exit:.2f}x")
print(f"{'='*70}\n")

# =============================================================================
# KARŞILAŞTIRMA
# =============================================================================
print("="*80)
print("📊 KASA KARŞILAŞTIRMASI")
print("="*80)
print(f"{'Metrik':<30} {'Kasa 1 (1.5x)':<20} {'Kasa 2 (%80)':<20}")
print(f"{'-'*70}")
print(f"{'Toplam Oyun':<30} {kasa1_total_bets:<20,} {kasa2_total_bets:<20,}")
print(f"{'Kazanan Oyun':<30} {kasa1_total_wins:<20,} {kasa2_total_wins:<20,}")
print(f"{'Kazanma Oranı':<30} {kasa1_win_rate:<20.1f}% {kasa2_win_rate:<20.1f}%")
print(f"{'Net Kar/Zarar':<30} {kasa1_profit_loss:<20,.2f} TL {kasa2_profit_loss:<20,.2f} TL")
print(f"{'ROI':<30} {kasa1_roi:<20.2f}% {kasa2_roi:<20.2f}%")
print(f"{'-'*70}")

# Hangi kasa daha karlı?
if kasa1_profit_loss > kasa2_profit_loss:
    print(f"🏆 KASA 1 daha karlı (+{kasa1_profit_loss - kasa2_profit_loss:,.2f} TL fark)")
elif kasa2_profit_loss > kasa1_profit_loss:
    print(f"🏆 KASA 2 daha karlı (+{kasa2_profit_loss - kasa1_profit_loss:,.2f} TL fark)")
else:
    print(f"⚖️ Her iki kasa eşit karlılıkta")

print(f"{'='*80}\n")

# =============================================================================
# MODEL KAYDETME + ZIP PAKETI
# =============================================================================
print("\n" + "="*80)
print("💾 MODELLER KAYDEDİLİYOR")
print("="*80)

import json
import shutil

# models/ klasörünü oluştur
os.makedirs('models', exist_ok=True)

# 1. CatBoost Regressor (.cbm formatı)
regressor.save_model(f'{DRIVE_MODEL_DIR}catboost_regressor.cbm')
print("✅ CatBoost Regressor kaydedildi: catboost_regressor.cbm")

# 2. CatBoost Classifier (.cbm formatı)
classifier.save_model(f'{DRIVE_MODEL_DIR}catboost_classifier.cbm')
print("✅ CatBoost Classifier kaydedildi: catboost_classifier.cbm")

# 3. Scaler
joblib.dump(scaler, f'{DRIVE_MODEL_DIR}catboost_scaler.pkl')
print("✅ Scaler kaydedildi: catboost_scaler.pkl")

# 4. Model bilgileri (JSON)
total_time = reg_time + cls_time
info = {
    'model': 'CatBoost_Dual_Model',
    'version': '2.0',
    'date': '2025-10-12',
    'architecture': {
        'regressor': 'CatBoost',
        'classifier': 'CatBoost'
    },
    'training_time_minutes': round(total_time/60, 1),
    'model_times': {
        'regressor': round(reg_time/60, 1),
        'classifier': round(cls_time/60, 1)
    },
    'feature_count': X.shape[1],
    'metrics': {
        'regression': {
            'mae': float(mae_reg),
            'rmse': float(rmse_reg)
        },
        'classification': {
            'accuracy': float(cls_acc),
            'below_15_accuracy': float(below_acc),
            'above_15_accuracy': float(above_acc),
            'money_loss_risk': float(fpr) if cm[0,0] + cm[0,1] > 0 else 0.0
        }
    },
    'hyperparameters': {
        'regressor': {
            'iterations': 1500,
            'depth': 10,
            'learning_rate': 0.03,
            'l2_leaf_reg': 5,
            'subsample': 0.8,
            'loss_function': 'MAE',
            'early_stopping_rounds': None  # Tüm iterasyonlar tamamlanacak
        },
        'classifier': {
            'iterations': 1500,
            'depth': 9,
            'learning_rate': 0.03,
            'l2_leaf_reg': 5,
            'subsample': 0.8,
            'loss_function': 'Logloss',
            'auto_class_weights': 'Balanced',
            'early_stopping_rounds': None  # Tüm iterasyonlar tamamlanacak
        }
    },
    'dual_bankroll_performance': {
        'kasa_1_15x': {
            'roi': float(kasa1_roi),
            'accuracy': float(kasa1_accuracy),
            'total_bets': int(kasa1_total_bets),
            'profit_loss': float(kasa1_profit_loss)
        },
        'kasa_2_80percent': {
            'roi': float(kasa2_roi),
            'accuracy': float(kasa2_accuracy),
            'total_bets': int(kasa2_total_bets),
            'profit_loss': float(kasa2_profit_loss),
            'avg_exit_point': float(kasa2_avg_exit)
        }
    },
    'top_features': [{'name': feat, 'importance': float(imp)} for feat, imp in top_features]
}

with open(f'{DRIVE_MODEL_DIR}catboost_model_info.json', 'w') as f:
    json.dump(info, f, indent=2)
print("✅ Model bilgileri kaydedildi: catboost_model_info.json")

print("\n📁 Kaydedilen dosyalar:")
print("  • catboost_regressor.cbm (CatBoost Regressor)")
print("  • catboost_classifier.cbm (CatBoost Classifier)")
print("  • catboost_scaler.pkl (Scaler)")
print("  • catboost_model_info.json (Model bilgileri)")
print("="*80)

# =============================================================================
# MODELLERİ ZIP'LE VE İNDİR
# =============================================================================
print("\n" + "="*80)
print("📦 MODELLER ZIP'LENIYOR")
print("="*80)

# ZIP dosyası oluştur
zip_filename = 'jetx_models_catboost_v2.0.zip'
shutil.make_archive(
    'jetx_models_catboost_v2.0', 
    'zip', 
    'models'
)

print(f"✅ ZIP dosyası oluşturuldu: {zip_filename}")
print(f"📦 Boyut: {os.path.getsize(f'{zip_filename}') / (1024*1024):.2f} MB")

# Google Colab'da indir
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    try:
        from google.colab import files
        files.download(f'{zip_filename}')
        print(f"✅ {zip_filename} indiriliyor...")
        print("\n📌 İNDİRDİĞİNİZ DOSYAYI AÇIP models/ KLASÖRÜNE KOPYALAYIN:")
        print("  1. ZIP'i açın")
        print("  2. Tüm dosyaları lokal projenizin models/ klasörüne kopyalayın")
        print("  3. Streamlit uygulamasını yeniden başlatın")
    except Exception as e:
        print(f"⚠️ İndirme hatası: {e}")
        print(f"⚠️ Manuel indirme gerekli: {zip_filename}")
else:
    print("\n⚠️ Google Colab ortamı algılanamadı - dosyalar sadece kaydedildi")
    print(f"📁 ZIP dosyası mevcut: {zip_filename}")
    print("\n💡 Not: Bu script Google Colab'da çalıştırıldığında dosyalar otomatik indirilir.")

print("="*80)

print(f"\n📊 Model Bilgisi:")
print(json.dumps(info, indent=2))

# Final rapor
print("\n" + "="*80)
print("🎉 CATBOOST TRAINING TAMAMLANDI!")
print("="*80)
print(f"Toplam Süre: {total_time/60:.1f} dakika ({total_time/3600:.1f} saat)")
print()

if below_acc >= 0.70 and fpr < 0.25:
    print("✅ ✅ İYİ PERFORMANS!")
    print(f"  🔴 1.5 ALTI: {below_acc*100:.1f}%")
    print(f"  💰 Para kaybı: {fpr*100:.1f}%")
    print("\n🚀 Model kullanıma hazır!")
else:
    print("⚠️ Orta performans")
    print(f"  🔴 1.5 ALTI: {below_acc*100:.1f}%")
    print(f"  💰 Para kaybı: {fpr*100:.1f}%")
    print("\nCatBoost XGBoost'a göre daha iyi class imbalance yönetimi sağlar.")

print("\n📁 Sonraki adım:")
print("  1. CatBoost modellerini lokal projeye kopyalayın")
print("  2. Predictor'da model_type='catboost' ile kullanın")
print("  3. Progressive NN ile karşılaştırın")
print("="*80)
print(f"Bitiş: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
