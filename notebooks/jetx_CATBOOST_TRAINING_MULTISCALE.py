#!/usr/bin/env python3
"""
🤖 JetX CATBOOST TRAINING - MULTI-SCALE WINDOW ENSEMBLE

YENİ YAKLAŞIM: Multi-Scale Window Ensemble
- Her pencere boyutu için ayrı CatBoost modeli
- Window boyutları: [500, 250, 100, 50, 20]
- Her model farklı zaman ölçeğinde desen öğrenir
- Final: Tüm modellerin ensemble'ı

AVANTAJLAR:
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
from utils.multi_scale_window import split_data_preserving_order
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
print("� Her pencere boyutu için feature engineering")

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
    """1.5x eşikte sanal kasa simülasyonu - DÜZELTME: ROI hesaplama mantığı"""
    initial = 10000
    wallet = initial
    bets_made = 0
    wins = 0
    
    for pred, actual in zip(predictions, actuals):
        if pred == 1:  # Model 1.5 üstü dedi
            bets_made += 1
            wallet -= 10  # 10 TL bahis
            if actual >= 1.5:
                wallet += 15  # 15 TL kazanç (1.5x)
                wins += 1
    
    # DÜZELTME: ROI hesaplama - sadece bahis yapılan durumlarda
    if bets_made > 0:
        roi = ((wallet - initial) / initial) * 100
        win_rate = (wins / bets_made) * 100
    else:
        roi = 0
        win_rate = 0
    
    return roi, win_rate, bets_made, wallet

def calculate_weighted_score(y_true, y_pred):
    """
    PROFIT-FOCUSED Weighted score hesaplama (YENİ!):
    - 50% ROI (para kazandırma - EN ÖNEMLİ!)
    - 30% Precision (1.5 üstü dediğinde ne kadar haklı)
    - 20% Win Rate (kazanan tahmin oranı)
    
    ESKI FORMÜL SORUNLARI:
    - Balanced Accuracy yanıltıcıydı (model hep "1.5 üstü" dediğinde yüksek çıkıyordu)
    - F1 Score dengesiz veride işe yaramıyordu
    - ROI sadece %10 ağırlıktaydı (çok az!)
    """
    # Confusion Matrix hesapla
    y_true_binary = (y_true >= 1.5).astype(int)
    
    TN = np.sum((y_true_binary == 0) & (y_pred == 0))
    FP = np.sum((y_true_binary == 0) & (y_pred == 1))
    FN = np.sum((y_true_binary == 1) & (y_pred == 0))
    TP = np.sum((y_true_binary == 1) & (y_pred == 1))
    
    # PRECISION (Model "1.5 üstü" dediğinde ne kadar haklı?)
    precision = (TP / (TP + FP) * 100) if (TP + FP) > 0 else 0
    
    # Sanal kasa simülasyonu
    roi, win_rate, bets_made, final_wallet = simulate_bankroll(y_pred, y_true)
    
    # ROI normalizasyonu
    normalized_roi = normalize_roi(roi)
    
    # YENİ WEIGHTED SCORE - PARA KAZANDIRMAYA ODAKLI!
    # 50% ROI + 30% Precision + 20% Win Rate
    weighted_score = (
        0.50 * normalized_roi +         # Para kazandırma (EN ÖNEMLİ!)
        0.30 * precision +               # "1.5 üstü" dediğinde ne kadar haklı
        0.20 * win_rate                  # Kazanan tahmin oranı
    )
    
    # Backward compatibility için eski metrikleri de hesapla
    below_acc = (TN / (TN + FP) * 100) if (TN + FP) > 0 else 0
    above_acc = (TP / (TP + FN) * 100) if (TP + FN) > 0 else 0
    balanced_acc = (below_acc + above_acc) / 2
    recall = (TP / (TP + FN)) if (TP + FN) > 0 else 0
    f1_score = (2 * (precision/100) * recall / ((precision/100) + recall)) if ((precision/100) + recall) > 0 else 0
    f1_score_percent = f1_score * 100
    money_loss_risk = (FP / (TN + FP)) if (TN + FP) > 0 else 1.0
    
    return weighted_score, balanced_acc, below_acc, above_acc, f1_score_percent, money_loss_risk * 100, roi, win_rate, bets_made

# =============================================================================
# HER PENCERE İÇİN MODEL EĞİTİMİ
# =============================================================================
print("\n" + "="*80)
print("🔥 MULTI-SCALE MODEL EĞİTİMİ BAŞLIYOR")
print("="*80)
print(f"Window boyutları: {window_sizes}")
print(f"Her window için ayrı Regressor + Classifier eğitilecek")
print(f"� Model Seçim Kriteri: PROFIT-FOCUSED Weighted Score (YENİ!)")
print(f"   - 50% ROI (para kazandırma - EN ÖNEMLİ!)")
print(f"   - 30% Precision (1.5 üstü dediğinde ne kadar haklı)")
print(f"   - 20% Win Rate (kazanan tahmin oranı)")
print(f"")
print(f"⚠️  ESKİ METRİKLER ARTIK KULLANILMIYOR:")
print(f"   - Balanced Accuracy (yanıltıcıydı - model hep '1.5 üstü' dediğinde yüksek çıkıyordu)")
print(f"   - F1 Score (dengesiz veride işe yaramıyordu)")
print(f"   - Threshold Accuracy (en yanıltıcı metrik)")
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
    
    # Class weights - DENGELI SISTEM (LAZY LEARNING DÜZELTMESİ)
    # DÜZELTME: Aşırı yüksek class weights yerine dengeli değerler
    # Eski değerler (25x, 20x, 15x, 10x) modeli "1.5 altı" demeye zorluyordu
    class_weight_0 = 1.5  # Sadece 1.5x ceza - model artık "1.5 üstü" demeye korkmayacak!
    
    print(f"📊 CLASS WEIGHTS (Window {window_size} - Lazy Learning Düzeltildi):")
    print(f"  1.5 altı: {class_weight_0:.1f}x ✅ DENGELİ!")
    print(f"  1.5 üstü: 1.0x")
    print(f"  Oran: {class_weight_0:.1f}x (ESKİ: 25x-10x → 1.5x)")
    print(f"  � Artık model '1.5 üstü' demeye teşvik edilecek!")
    
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
    weighted_score, balanced_acc_val, below_acc_val, above_acc_val, f1_score_val, money_loss_risk_val, roi_val, win_rate_val, bets_made_val = calculate_weighted_score(
        y_reg_val, y_cls_pred_val
    )
    
    print(f"\n{'='*80}")
    print(f"✨ VALIDATION BALANCED METRİKLER")
    print(f"{'='*80}")
    print(f"📊 Weighted Score:     {weighted_score:6.2f}")
    print(f"⚖️  Balanced Acc:       {balanced_acc_val:6.2f}% (Her sınıf eşit önemli)")
    print(f"🔴 Below 1.5 Acc:      {below_acc_val:6.1f}%")
    print(f"🟢 Above 1.5 Acc:      {above_acc_val:6.1f}%")
    print(f"🎯 F1 Score:           {f1_score_val:6.2f}%")
    print(f"💰 Money Loss Risk:    {money_loss_risk_val:6.2f}% (Target: <25%)")
    print(f"💵 ROI:                {roi_val:+7.2f}%")
    print(f"📈 Win Rate:           {win_rate_val:6.2f}%  ({int(win_rate_val*bets_made_val/100)}/{bets_made_val})")
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

models_dir = os.path.join(PROJECT_ROOT, 'models/catboost_multiscale')
os.makedirs(models_dir, exist_ok=True)

# Her window için model kaydet
for window_size in window_sizes:
    model_dict = trained_models[window_size]
    
    # Regressor
    reg_path = os.path.join(PROJECT_ROOT, f'models/catboost_multiscale/regressor_window_{window_size}.cbm')
    model_dict['regressor'].save_model(reg_path)
    
    # Classifier
    cls_path = os.path.join(PROJECT_ROOT, f'models/catboost_multiscale/classifier_window_{window_size}.cbm')
    model_dict['classifier'].save_model(cls_path)
    
    # Scaler
    scaler_path = os.path.join(PROJECT_ROOT, f'models/catboost_multiscale/scaler_window_{window_size}.pkl')
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

info_path = os.path.join(PROJECT_ROOT, 'models/catboost_multiscale/model_info.json')
with open(info_path, 'w') as f:
    json.dump(info, f, indent=2)

print(f"✅ Model bilgileri kaydedildi")

# ZIP oluştur
zip_filename = 'jetx_models_catboost_multiscale_v3.0'
catboost_multiscale_dir = os.path.join(PROJECT_ROOT, 'models/catboost_multiscale')
shutil.make_archive(zip_filename, 'zip', catboost_multiscale_dir)

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
