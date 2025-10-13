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
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                      "catboost", "scikit-learn", "pandas", "numpy", 
                      "scipy", "joblib", "matplotlib", "seaborn", "tqdm",
                      "PyWavelets", "nolds"])

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
    train_ratio=0.8,
    val_ratio=0.1
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
    y_regression = []
    y_classification = []
    
    for i in tqdm(range(window_size, len(data)-1), desc=f'Window {window_size}'):
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

for window_size in window_sizes:
    print(f"\n🔧 Window {window_size} için feature extraction...")
    
    # Train data
    X_train, y_reg_train, y_cls_train = extract_features_for_window(train_data, window_size)
    
    # Val data
    X_val, y_reg_val, y_cls_val = extract_features_for_window(val_data, window_size)
    
    # Test data
    X_test, y_reg_test, y_cls_test = extract_features_for_window(test_data, window_size)
    
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
# HER PENCERE İÇİN MODEL EĞİTİMİ
# =============================================================================
print("\n" + "="*80)
print("🔥 MULTI-SCALE MODEL EĞİTİMİ BAŞLIYOR")
print("="*80)
print(f"Window boyutları: {window_sizes}")
print(f"Her window için ayrı Regressor + Classifier eğitilecek")
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
        task_type='GPU',
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
    
    print(f"\n📊 REGRESSOR PERFORMANSI:")
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
        task_type='GPU',
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
    
    print(f"\n📊 CLASSIFIER PERFORMANSI:")
    print(f"  Genel Accuracy: {cls_acc*100:.2f}%")
    print(f"  🔴 1.5 Altı: {below_acc*100:.2f}%")
    print(f"  🟢 1.5 Üstü: {above_acc*100:.2f}%")
    
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
Görev kesintiye uğramış. Devam ediyorum. CatBoost Standard dosyasını multi-scale sisteme dönüştüreceğim.

<write_to_file>
<path>notebooks/jetx_CATBOOST_TRAINING_MULTISCALE.py</path>
<content>#!/usr/bin/env python3
"""
🤖 JetX CATBOOST TRAINING - MULTI-SCALE WINDOW ENSEMBLE

YENİ YAKLAŞIM: Multi-Scale Window Ensemble
- Her pencere boyutu için ayrı model eğitimi
- Window boyutları: [500, 250, 100, 50, 20]
- Her model farklı zaman ölçeğinde desen öğrenir
- Final: Tüm modellerin ensemble'ı

AVANTAJLAR:
- Çok hızlı eğitim (~40-60 dakika toplam, 5 model)
- Feature importance analizi
- Class imbalance için native destek
- Overfitting'e dirençli

HEDEFLER:
- 1.5 ALTI Doğruluk: %70-80%+
- 1.5 ÜSTÜ Doğruluk: %70-80%+
- MAE: < 2.0

⚠️  VERİ BÜTÜNLİĞİ:
- Shuffle: YASAK
- Augmentation: YASAK
- Kronolojik sıra: KORUNUYOR

SÜRE: ~40-60 dakika (5 model × ~8-12 dakika)
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

print("="*80)
print("🤖 JetX CATBOOST TRAINING - MULTI-SCALE WINDOW ENSEMBLE")
print("="*80)
print(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()
print("🔧 YENİ SİSTEM: Her pencere boyutu için ayrı CatBoost model")
print("   Window boyutları: [500, 250, 100, 50, 20]")
print("   ⚠️  Veri sırası KORUNUYOR (shuffle=False)")
print("   ⚠️  Data augmentation KAPALI")
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix
from catboost import CatBoostRegressor, CatBoostClassifier
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

print(f"✅ CatBoost: Import edildi")

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
    train_ratio=0.8,
    val_ratio=0.1
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
    y_regression = []
    y_classification = []
    
    for i in tqdm(range(window_size, len(data)-1), desc=f'Window {window_size}'):
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

for window_size in window_sizes:
    print(f"\n🔧 Window {window_size} için feature extraction...")
    
    # Train data
    X_train, y_reg_train, y_cls_train = extract_features_for_window(train_data, window_size)
    
    # Val data
    X_val, y_reg_val, y_cls_val = extract_features_for_window(val_data, window_size)
    
    # Test data
    X_test, y_reg_test, y_cls_test = extract_features_for_window(test_data, window_size)
    
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
# HER PENCERE İÇİN MODEL EĞİTİMİ
# =============================================================================
print("\n" + "="*80)
print("🔥 MULTI-SCALE CATBOOST EĞİTİMİ BAŞLIYOR")
print("="*80)
print(f"Window boyutları: {window_sizes}")
print(f"Her window için ayrı Regressor + Classifier eğitilecek")
print("="*80 + "\n")

trained_models = {}
training_times = {}

for window_size in window_sizes:
    print("\n" + "="*80)
    print(f"🎯 WINDOW {window_size} - CATBOOST EĞİTİMİ")
    print("="*80)
    
    window_start_time = time.time()
    
    # Veriyi al
    data_dict = all_data_by_window[window_size]
    X_train, y_reg_train, y_cls_train = data_dict['train']
    X_val, y_reg_val, y_cls_val = data_dict['val']
    
    # =================================================================
    # REGRESSOR
    # =================================================================
    print(f"\n📊 Regressor (Window {window_size})...")
    
    regressor = CatBoostRegressor(
        iterations=1500,
        depth=10,
        learning_rate=0.03,
        l2_leaf_reg=5,
        bootstrap_type='Bernoulli',
        subsample=0.8,
        loss_function='MAE',
        eval_metric='MAE',
        task_type='CPU',
        verbose=False,
        random_state=42
    )
    
    regressor.fit(
        X_train, y_reg_train,
        eval_set=(X_val, y_reg_val),
        verbose=False
    )
    
    # =================================================================
    # CLASSIFIER
    # =================================================================
    print(f"📊 Classifier (Window {window_size})...")
    
    # Class weights - window boyutuna göre ayarla
    if window_size <= 50:
        class_weight = {0: 25.0, 1: 1.0}  # Çok agresif
    elif window_size <= 100:
        class_weight = {0: 20.0, 1: 1.0}  # Agresif
    elif window_size <= 250:
        class_weight = {0: 15.0, 1: 1.0}  # Orta
    else:
        class_weight = {0: 10.0, 1: 1.0}  # Dengeli
    
    print(f"  Class weights: {class_weight[0]:.0f}x (1.5 altı)")
    
    classifier = CatBoostClassifier(
        iterations=1500,
        depth=9,
        learning_rate=0.03,
        l2_leaf_reg=5,
        bootstrap_type='Bernoulli',
        subsample=0.8,
        loss_function='Logloss',
        eval_metric='Accuracy',
        task_type='CPU',
        class_weights=class_weight,
        verbose=False,
        random_state=42
    )
    
    classifier.fit(
        X_train, y_cls_train,
        eval_set=(X_val, y_cls_val),
        verbose=False
    )
    
    window_time = time.time() - window_start_time
    training_times[window_size] = window_time
    
    print(f"\n✅ Window {window_size} eğitimi tamamlandı!")
    print(f"⏱️  Süre: {window_time/60:.1f} dakika")
    
    # Test performansı
    X_test, y_reg_test, y_cls_test = data_dict['test']
    
    # Regressor metrics
    y_reg_pred = regressor.predict(X_test)
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    
    # Classifier metrics
    y_cls_pred = classifier.predict(X_test)
    cls_acc = accuracy_score(y_cls_test, y_cls_pred)
    
    below_mask = y_cls_test == 0
    above_mask = y_cls_test == 1
    below_acc = accuracy_score(y_cls_test[below_mask], y_cls_pred[below_mask]) if below_mask.sum() > 0 else 0
    above_acc = accuracy_score(y_cls_test[above_mask], y_cls_pred[above_mask]) if above_mask.sum() > 0 else 0
    
    print(f"\n📊 WINDOW {window_size} TEST PERFORMANSI:")
    print(f"  Regressor MAE: {mae:.4f}")
    print(f"  Classifier Accuracy: {cls_acc*100:.2f}%")
    print(f"  🔴 1.5 Altı: {below_acc*100:.2f}%")
    print(f"  🟢 1.5 Üstü: {above_acc*100:.2f}%")
    
    # Modelleri kaydet
    trained_models[window_size] = {
        'regressor': regressor,
        'classifier': classifier,
        'scaler': data_dict['scaler'],
        'mae': float(mae),
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

# Test data (500'lük window referans)
_, y_reg_test, y_cls_test = all_data_by_window[500]['test']

# Her modelden tahminleri al
ensemble_predictions_reg = []
ensemble_predictions_cls = []

for window_size in window_sizes:
    model_dict = trained_models[window_size]
    
    X_test, _, _ = all_data_by_window[window_size]['test']
    
    # Tahminler
    y_reg_pred = model_dict['regressor'].predict(X_test)
    y_cls_pred = model_dict['classifier'].predict(X_test)
    
    ensemble_predictions_reg.append(y_reg_pred)
    ensemble_predictions_cls.append(y_cls_pred)

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
print(f"  Regressor MAE: {mae_ensemble:.4f}")
print(f"  Regressor RMSE: {rmse_ensemble:.4f}")
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

print(f"\n{'Window':<10} {'Reg MAE':<10} {'Cls Acc':<12} {'Below':<12} {'Above':<12} {'Süre':<12}")
print("-"*70)
for window_size in window_sizes:
    model_dict = trained_models[window_size]
    print(
        f"{window_size:<10} "
        f"{model_dict['mae']:<10.4f} "
        f"{model_dict['cls_acc']*100:<12.2f}% "
        f"{model_dict['below_acc']*100:<12.2f}% "
        f"{model_dict['above_acc']*100:<12.2f}% "
        f"{model_dict['training_time']/60:<12.1f} dk"
    )
print("-"*70)
print(
    f"{'ENSEMBLE':<10} "
    f"{mae_ensemble:<10.4f} "
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

# Her window için modelleri kaydet
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
        'regressor_mae': float(mae_ensemble),
        'regressor_rmse': float(rmse_ensemble),
        'classifier_accuracy': float(cls_acc_ensemble),
        'below_15_accuracy': float(below_acc_ensemble),
        'above_15_accuracy': float(above_acc_ensemble),
        'money_loss_risk': float(fpr) if cm[0,0] + cm[0,1] > 0 else 0.0
    },
    'individual_models': {
        str(ws): {
            'regressor_mae': trained_models[ws]['mae'],
            'classifier_acc': trained_models[ws]['cls_acc'],
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
print("🎉 MULTI-SCALE CATBOOST TRAINING TAMAMLANDI!")
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
