#!/usr/bin/env python3
"""
🤖 JetX CATBOOST TRAINING - MULTI-SCALE WINDOW ENSEMBLE

YENİ YAKLAŞIM: Multi-Scale Window Ensemble
- Her pencere boyutu için ayrı CatBoost modeli
- Window boyutları: [500, 250, 100, 50, 20]
- Her model farklı zaman ölçeğinde desen öğrenir
- Final: Tüm modellerin ensemble'ı

AVANTAJLAR:
- Çok hızlı eğitim (~40-60 dakika, 5 model × ~8-12 dk)
- Feature importance analizi yapılabilir
- Daha az bellek kullanımı
- Class imbalance için native destek
- Multi-scale desenler yakalanır

HEDEFLER:
- 1.5 ALTI Doğruluk: %70-80%+
- 1.5 ÜSTÜ Doğruluk: %75-85%+
- Para kaybı riski: %20 altı
- MAE: < 2.0

⚠️  VERİ BÜTÜNLİĞİ:
- Shuffle: YASAK
- Augmentation: YASAK
- Kronolojik sıra: KORUNUYOR

SÜRE: ~40-60 dakika (5 model × ~8-12 dk)
"""

import subprocess
import sys
import os
import time
from datetime import datetime
import json
import shutil

print("="*80)
print("🤖 JetX CATBOOST TRAINING - MULTI-SCALE WINDOW ENSEMBLE")
print("="*80)
print(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()
print("🔧 YENİ SİSTEM: Her pencere boyutu için ayrı CatBoost modeli")
print("   Window boyutları: [500, 250, 100, 50, 20]")
print("   ⚠️  Veri sırası KORUNUYOR (shuffle=False)")
print("   ⚠️  Data augmentation KAPALI")
print()

# Kütüphaneleri yükle
print("📦 Kütüphaneler yükleniyor...")
print("   ⚠️  Advanced features için scipy, PyWavelets, nolds yükleniyor...")
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

# GPU kontrolü ve fallback
try:
    # GPU test için geçici model oluştur
    from catboost import CatBoostClassifier
    temp_model = CatBoostClassifier(iterations=1, task_type='GPU', devices='0')
    temp_model.fit([[1]], [0], verbose=False)
    GPU_AVAILABLE = True
    TASK_TYPE = 'GPU'
    print(f"✅ GPU: Mevcut ve kullanılabilir")
except Exception as e:
    GPU_AVAILABLE = False
    TASK_TYPE = 'CPU'
    print(f"⚠️ GPU: Kullanılamıyor, CPU modunda çalışacak")
    print(f"   Sebep: {str(e)[:100]}")

# Proje yükle
if not os.path.exists('jetxpredictor'):
    print("\n📥 Proje klonlanıyor...")
    subprocess.check_call(["git", "clone", "https://github.com/onndd/jetxpredictor.git"])

os.chdir('jetxpredictor')
sys.path.append(os.getcwd())

from category_definitions import CategoryDefinitions, FeatureEngineering
from utils.multi_scale_window import split_data_preserving_order
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
    y_regression = []
    y_classification = []
    
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
        
        # Targets
        y_regression.append(target)
        y_classification.append(1 if target >= 1.5 else 0)
    
    X_features = np.array(X_features)
    y_regression = np.array(y_regression)
    y_classification = np.array(y_classification)
    
    return X_features, y_regression, y_classification

# Her window boyutu için feature extraction
all_data_by_window = {}

# En büyük pencere boyutu (500) için test başlangıç indeksini hesapla
max_window = max(window_sizes)
test_start_idx = max_window  # En büyük pencere boyutu kadar offset

for window_size in window_sizes:
    print(f"\n🔧 Window {window_size} için feature extraction...")
    
    # Train data
    X_train, y_reg_train, y_cls_train = extract_features_for_window(train_data, window_size)
    
    # Val data
    X_val, y_reg_val, y_cls_val = extract_features_for_window(val_data, window_size)
    
    # Test data - TÜM MODELLER İÇİN AYNI BAŞLANGIÇ İNDEKSİ
    # Bu, ensemble için tutarlı tahmin uzunlukları sağlar
    X_test, y_reg_test, y_cls_test = extract_features_for_window(
        test_data, window_size, start_idx=test_start_idx
    )
    
    # FEATURE VALIDATION - CatBoost için kritik!
    print(f"\n🔍 Feature validation (Window {window_size})...")
    print(f"  Feature sayısı: {X_train.shape[1]}")
    print(f"  Feature ortalaması: {np.mean(X_train):.4f}")
    print(f"  Feature std: {np.std(X_train):.4f}")
    print(f"  Sıfır olmayan feature sayısı: {np.count_nonzero(np.std(X_train, axis=0))}/{X_train.shape[1]}")
    
    # Constant features kontrolü
    feature_stds = np.std(X_train, axis=0)
    constant_features = np.sum(feature_stds < 1e-10)
    if constant_features > X_train.shape[1] * 0.5:  # %50'den fazla constant varsa uyar
        print(f"  ⚠️  UYARI: {constant_features} feature constant veya sıfıra çok yakın!")
        print(f"  ⚠️  Bu CatBoost performansını düşürebilir!")
    else:
        print(f"  ✅ Feature kalitesi iyi: {constant_features} constant feature")
    
    # Normalizasyon
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
# WEIGHTED SCORE HELPER FUNCTIONS
# =============================================================================
def normalize_roi(roi):
    """Kademeli lineer normalizasyon (Seçenek 2)"""
    if roi < 0:
        # Negatif ROI: 0-40 arası
        return max(0, (roi + 100) / 100 * 40)
    else:
        # Pozitif ROI: 40-100 arası
        return min(100, 40 + (roi / 200 * 60))

def simulate_bankroll(predictions, actuals):
    """1.5x eşikte sanal kasa simülasyonu"""
    initial = 10000
    wallet = initial
    for pred, actual in zip(predictions, actuals):
        if pred == 1:  # Model 1.5 üstü dedi
            wallet -= 10
            if actual >= 1.5:
                wallet += 15
    roi = ((wallet - initial) / initial) * 100
    return roi

def calculate_weighted_score(y_true, y_pred):
    """
    Weighted score hesaplama:
    - 50% Below 15 accuracy
    - 40% Above 15 accuracy  
    - 10% ROI (normalized)
    """
    # Below/Above accuracy hesapla
    below_mask = y_true < 1.5
    above_mask = y_true >= 1.5
    
    below_correct = 0
    below_total = 0
    for pred, actual in zip(y_pred[below_mask], y_true[below_mask]):
        below_total += 1
        if pred == 0 and actual < 1.5:  # Doğru tahmin (1.5 altı)
            below_correct += 1
    
    above_correct = 0
    above_total = 0
    for pred, actual in zip(y_pred[above_mask], y_true[above_mask]):
        above_total += 1
        if pred == 1 and actual >= 1.5:  # Doğru tahmin (1.5 üstü)
            above_correct += 1
    
    below_acc = (below_correct / below_total * 100) if below_total > 0 else 0
    above_acc = (above_correct / above_total * 100) if above_total > 0 else 0
    
    # ROI hesapla
    roi = simulate_bankroll(y_pred, y_true)
    normalized_roi = normalize_roi(roi)
    
    # Weighted score
    weighted_score = (0.5 * below_acc) + (0.4 * above_acc) + (0.1 * normalized_roi)
    
    return weighted_score, below_acc, above_acc, roi, normalized_roi

# =============================================================================
# HER PENCERE İÇİN MODEL EĞİTİMİ
# =============================================================================
print("\n" + "="*80)
print("🔥 MULTI-SCALE MODEL EĞİTİMİ BAŞLIYOR")
print("="*80)
print(f"Window boyutları: {window_sizes}")
print(f"Her window için ayrı Regressor + Classifier eğitilecek")
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
    X_train, y_reg_train, y_cls_train = data_dict['train']
    X_val, y_reg_val, y_cls_val = data_dict['val']
    X_test, y_reg_test, y_cls_test = data_dict['test']
    
    # =============================================================================
    # REGRESSOR EĞİTİMİ
    # =============================================================================
    print(f"\n🎯 REGRESSOR EĞİTİMİ (Window {window_size})")
    
    regressor = CatBoostRegressor(
        iterations=1500,
        depth=10,
        learning_rate=0.03,
        l2_leaf_reg=5,
        bootstrap_type='Bernoulli',
        subsample=0.8,
        loss_function='MAE',
        eval_metric='MAE',
        task_type=TASK_TYPE,  # GPU veya CPU (otomatik seçildi)
        verbose=100,
        random_state=42
    )
    
    print(f"📊 Regressor parametreleri:")
    print(f"  iterations: 1500")
    print(f"  depth: 10")
    print(f"  learning_rate: 0.03")
    
    regressor.fit(
        X_train, y_reg_train,
        eval_set=(X_val, y_reg_val),
        verbose=100
    )
    
    # Test performansı
    y_reg_pred = regressor.predict(X_test)
    mae_reg = mean_absolute_error(y_reg_test, y_reg_pred)
    rmse_reg = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
    
    print(f"\n📊 REGRESSOR FINAL PERFORMANSI:")
    print(f"  MAE: {mae_reg:.4f}")
    print(f"  RMSE: {rmse_reg:.4f}")
    
    # =============================================================================
    # CLASSIFIER EĞİTİMİ
    # =============================================================================
    print(f"\n🎯 CLASSIFIER EĞİTİMİ (Window {window_size})")
    
    # Class weights - window boyutuna göre ayarla
    if window_size <= 50:
        class_weight_0 = 25.0
    elif window_size <= 100:
        class_weight_0 = 20.0
    elif window_size <= 250:
        class_weight_0 = 15.0
    else:
        class_weight_0 = 10.0
    
    print(f"📊 CLASS WEIGHTS (Window {window_size}):")
    print(f"  1.5 altı: {class_weight_0:.1f}x")
    print(f"  1.5 üstü: 1.0x")
    
    classifier = CatBoostClassifier(
        iterations=1500,
        depth=9,
        learning_rate=0.03,
        l2_leaf_reg=5,
        bootstrap_type='Bernoulli',
        subsample=0.8,
        loss_function='Logloss',
        eval_metric='Accuracy',
        task_type=TASK_TYPE,  # GPU veya CPU (otomatik seçildi)
        auto_class_weights='Balanced',
        verbose=100,
        random_state=42
    )
    
    classifier.fit(
        X_train, y_cls_train,
        eval_set=(X_val, y_cls_val),
        verbose=100
    )
    
    # Test performansı
    y_cls_pred = classifier.predict(X_test)
    cls_acc = accuracy_score(y_cls_test, y_cls_pred)
    
    below_mask = y_cls_test == 0
    above_mask = y_cls_test == 1
    below_acc = accuracy_score(y_cls_test[below_mask], y_cls_pred[below_mask]) if below_mask.sum() > 0 else 0
    above_acc = accuracy_score(y_cls_test[above_mask], y_cls_pred[above_mask]) if above_mask.sum() > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"📊 CLASSIFIER FINAL PERFORMANSI")
    print(f"{'='*80}")
    print(f"🎯 Genel Accuracy:     {cls_acc*100:6.2f}%")
    print(f"🔴 1.5 Altı Doğruluk:  {below_acc*100:6.2f}%")
    print(f"🟢 1.5 Üstü Doğruluk:  {above_acc*100:6.2f}%")
    
    # Validation setinde Weighted Score hesapla
    y_cls_pred_val = classifier.predict(X_val)
    weighted_score, below_acc_val, above_acc_val, roi_val, normalized_roi_val = calculate_weighted_score(
        y_reg_val, y_cls_pred_val
    )
    
    # Sanal kasa simülasyonu detayları
    wins_val = 0
    total_bets_val = 0
    for pred, actual in zip(y_cls_pred_val, y_reg_val):
        if pred == 1:
            total_bets_val += 1
            if actual >= 1.5:
                wins_val += 1
    win_rate_val = (wins_val / total_bets_val * 100) if total_bets_val > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"✨ VALIDATION DETAYLI METRİKLER")
    print(f"{'='*80}")
    print(f"📊 Weighted Score:     {weighted_score:6.2f}")
    print(f"🔴 Below 1.5 Acc:      {below_acc_val:6.1f}%")
    print(f"🟢 Above 1.5 Acc:      {above_acc_val:6.1f}%")
    print(f"💰 ROI:                {roi_val:+7.2f}%  (Normalized: {normalized_roi_val:6.1f})")
    print(f"📈 Win Rate:           {win_rate_val:6.2f}%  ({wins_val}/{total_bets_val})")
    print(f"{'='*80}\n")
    
    window_time = time.time() - window_start_time
    training_times[window_size] = window_time
    
    print(f"\n✅ Window {window_size} eğitimi tamamlandı!")
    print(f"⏱️  Süre: {window_time/60:.1f} dakika")
    
    # Modeli kaydet
    trained_models[window_size] = {
        'regressor': regressor,
        'classifier': classifier,
        'scaler': data_dict['scaler'],
        'mae': float(mae_reg),
        'rmse': float(rmse_reg),
        'cls_acc': float(cls_acc),
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
X_test_500, y_reg_test, y_cls_test = all_data_by_window[500]['test']

ensemble_predictions_reg = []
ensemble_predictions_cls = []

for window_size in window_sizes:
    model_dict = trained_models[window_size]
    regressor = model_dict['regressor']
    classifier = model_dict['classifier']
    
    # Bu window için test data
    X_test_w, _, _ = all_data_by_window[window_size]['test']
    
    # Tahmin
    p_reg = regressor.predict(X_test_w)
    p_cls = classifier.predict(X_test_w)
    
    ensemble_predictions_reg.append(p_reg)
    ensemble_predictions_cls.append(p_cls)

# Ensemble: Basit ortalama
ensemble_reg = np.mean(ensemble_predictions_reg, axis=0)
ensemble_cls = np.round(np.mean(ensemble_predictions_cls, axis=0)).astype(int)

# Metrics
mae_ensemble = mean_absolute_error(y_reg_test, ensemble_reg)
rmse_ensemble = np.sqrt(mean_squared_error(y_reg_test, ensemble_reg))
cls_acc_ensemble = accuracy_score(y_cls_test, ensemble_cls)

below_mask = y_cls_test == 0
above_mask = y_cls_test == 1
below_acc_ensemble = accuracy_score(y_cls_test[below_mask], ensemble_cls[below_mask]) if below_mask.sum() > 0 else 0
above_acc_ensemble = accuracy_score(y_cls_test[above_mask], ensemble_cls[above_mask]) if above_mask.sum() > 0 else 0

print(f"\n📊 ENSEMBLE PERFORMANSI:")
print(f"  MAE: {mae_ensemble:.4f}")
print(f"  RMSE: {rmse_ensemble:.4f}")
print(f"  Classifier Accuracy: {cls_acc_ensemble*100:.2f}%")
print(f"  🔴 1.5 Altı: {below_acc_ensemble*100:.2f}%")
print(f"  🟢 1.5 Üstü: {above_acc_ensemble*100:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_cls_test, ensemble_cls)
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

print(f"\n{'Window':<10} {'MAE':<10} {'RMSE':<10} {'Cls Acc':<12} {'Below':<12} {'Above':<12} {'Süre':<12}")
print("-"*80)
for window_size in window_sizes:
    model_dict = trained_models[window_size]
    print(
        f"{window_size:<10} "
        f"{model_dict['mae']:<10.4f} "
        f"{model_dict['rmse']:<10.4f} "
        f"{model_dict['cls_acc']*100:<12.2f}% "
        f"{model_dict['below_acc']*100:<12.2f}% "
        f"{model_dict['above_acc']*100:<12.2f}% "
        f"{model_dict['training_time']/60:<12.1f} dk"
    )
print("-"*80)
print(
    f"{'ENSEMBLE':<10} "
    f"{mae_ensemble:<10.4f} "
    f"{rmse_ensemble:<10.4f} "
    f"{cls_acc_ensemble*100:<12.2f}% "
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
    model_pred = ensemble_cls[i]
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

os.makedirs('models/catboost_multiscale', exist_ok=True)

# Her window için model kaydet
for window_size in window_sizes:
    model_dict = trained_models[window_size]
    
    # Regressor
    reg_path = f'models/catboost_multiscale/regressor_window_{window_size}.cbm'
    model_dict['regressor'].save_model(reg_path)
    
    # Classifier
    cls_path = f'models/catboost_multiscale/classifier_window_{window_size}.cbm'
    model_dict['classifier'].save_model(cls_path)
    
    # Scaler
    scaler_path = f'models/catboost_multiscale/scaler_window_{window_size}.pkl'
    joblib.dump(model_dict['scaler'], scaler_path)
    
    print(f"✅ Window {window_size} kaydedildi")

# Model bilgileri
info = {
    'model': 'CatBoost_MultiScale_Ensemble',
    'version': '3.0',
    'date': datetime.now().strftime('%Y-%m-%d'),
    'window_sizes': window_sizes,
    'total_training_time_minutes': round(total_training_time/60, 1),
    'ensemble_metrics': {
        'mae': float(mae_ensemble),
        'rmse': float(rmse_ensemble),
        'classifier_accuracy': float(cls_acc_ensemble),
        'below_15_accuracy': float(below_acc_ensemble),
        'above_15_accuracy': float(above_acc_ensemble),
        'money_loss_risk': float(fpr) if cm[0,0] + cm[0,1] > 0 else 0.0
    },
    'individual_models': {
        str(ws): {
            'mae': trained_models[ws]['mae'],
            'rmse': trained_models[ws]['rmse'],
            'cls_acc': trained_models[ws]['cls_acc'],
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

with open('models/catboost_multiscale/model_info.json', 'w') as f:
    json.dump(info, f, indent=2)

print(f"✅ Model bilgileri kaydedildi")

# ZIP oluştur
zip_filename = 'jetx_models_catboost_multiscale_v3.0'
shutil.make_archive(zip_filename, 'zip', 'models/catboost_multiscale')

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
        print(f"✅ {zip_filename}.zip tarayıcınıza indiriliyor...")
        print(f"   Eğer otomatik indirme başlamazsa, sol panelden Files sekmesine gidin")
        print(f"   ve '{zip_filename}.zip' dosyasına sağ tıklayıp 'Download' seçin.")
        files.download(f'{zip_filename}.zip')
    except Exception as e:
        print(f"⚠️ Otomatik indirme hatası: {e}")
        print(f"📁 Manuel indirme: Sol panelden Files → '{zip_filename}.zip' → Download")
else:
    print(f"📁 ZIP dosyası mevcut: {zip_filename}.zip")

print("="*80)

# =============================================================================
# FINAL RAPOR
# =============================================================================
print("\n" + "="*80)
print("🎉 MULTI-SCALE CATBOOST TRAINING TAMAMLANDI!")
print("="*80)
print(f"Toplam Süre: {total_training_time/60:.1f} dakika ({total_training_time/3600:.2f} saat)")
print()

if below_acc_ensemble >= 0.75 and fpr < 0.20:
    print("✅ ✅ ✅ TÜM BAŞARIYLA AŞILDI!")
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
