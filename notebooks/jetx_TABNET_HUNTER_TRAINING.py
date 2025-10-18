#!/usr/bin/env python3
"""
🚀 JetX TABNET HUNTER TRAINING - Yüksek-X Avcısı

AMAÇ: Nadir görülen yüksek çarpanları (10x ve üzeri) tahmin etmeye odaklanmış 
özel bir regresyon modeli eğitmek.

STRATEJİ:
- TabNet: Attention mekanizması ile tabular veriler için özel tasarlanmış
- Hedef: Yüksek çarpan regresyonu (10x+, 20x+, 50x+)
- Focus: Weighted loss ile yüksek değerlere odaklanma
- Specialization: Yüksek çarpan pattern'lerini öğrenme

HEDEFLER:
- 10x+ Tahmin Doğruluğu: %60+ (zorlu ama ulaşılabilir)
- 5x+ Tahmin Doğruluğu: %70+ 
- MAE (yüksek değerler için): < 5.0
- Yüksek çarpan kaçırma oranı: <30%

SÜRE: ~30-45 dakika (GPU ile)
"""

import subprocess
import sys
import os
import time
from datetime import datetime
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import sqlite3
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🚀 JetX TABNET HUNTER TRAINING - Yüksek-X Avcısı")
print("="*80)
print(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Kütüphaneleri yükle
print("📦 Kütüphaneler yükleniyor...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                      "pytorch-tabnet", "torch", "scikit-learn", "pandas", "numpy", 
                      "scipy", "matplotlib", "seaborn", "tqdm",
                      "PyWavelets", "nolds"])

try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    print("✅ TabNet başarıyla yüklendi")
except ImportError as e:
    print(f"❌ TabNet yüklenemedi: {e}")
    sys.exit(1)

# GPU kontrolü
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Cihaz: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA: {torch.version.cuda}")
else:
    print("   GPU bulunamadı - CPU modunda çalışılacak")

# Google Drive mount (Colab için)
try:
    from google.colab import drive
    
    if not os.path.exists('/content/drive'):
        print("\n📦 Google Drive bağlanıyor...")
        drive.mount('/content/drive')
    
    # Model kayıt dizini
    DRIVE_MODEL_DIR = '/content/drive/MyDrive/JetX_Models/TabNet_Hunter/'
    os.makedirs(DRIVE_MODEL_DIR, exist_ok=True)
    print(f"✅ Google Drive bağlandı: {DRIVE_MODEL_DIR}")
    USE_DRIVE = True
except ImportError:
    print("⚠️ Google Colab dışında - lokal kayıt kullanılacak")
    DRIVE_MODEL_DIR = 'models/tabnet_hunter/'
    os.makedirs(DRIVE_MODEL_DIR, exist_ok=True)
    USE_DRIVE = False
except Exception as e:
    print(f"⚠️ Google Drive mount hatası: {e}")
    DRIVE_MODEL_DIR = 'models/tabnet_hunter/'
    os.makedirs(DRIVE_MODEL_DIR, exist_ok=True)
    USE_DRIVE = False

# Proje yükle
if not os.path.exists('jetxpredictor'):
    print("\n📥 Proje klonlanıyor...")
    subprocess.check_call(["git", "clone", "https://github.com/onndd/jetxpredictor.git"])

os.chdir('jetxpredictor')
sys.path.append(os.getcwd())

from category_definitions import CategoryDefinitions, FeatureEngineering
print(f"✅ Proje yüklendi - Kritik eşik: {CategoryDefinitions.CRITICAL_THRESHOLD}x\n")

# =============================================================================
# ÖZEL KAYIP FONKSİYONU - YÜKSEK DEĞERLERE ODAKLANMA
# =============================================================================
class HighXWeightedMSELoss(nn.Module):
    """
    Yüksek değerlere odaklanan özel MSE kayıp fonksiyonu
    
    Ağırlıklandırma:
    - 10x+ değerler: 20x ağırlık (çok önemli)
    - 5x-10x değerler: 5x ağırlık (önemli)
    - Diğer değerler: 1x ağırlık (normal)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred, y_true):
        # MSE hata
        mse_error = (y_pred - y_true) ** 2
        
        # Ağırlıklandırma
        weights = torch.ones_like(y_true)
        
        # 10x+ için 20x ağırlık
        weights[y_true >= 10.0] = 20.0
        
        # 5x-10x için 5x ağırlık
        weights[(y_true >= 5.0) & (y_true < 10.0)] = 5.0
        
        # Weighted MSE
        weighted_mse = (mse_error * weights).mean()
        
        return weighted_mse

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

# Yüksek değer dağılımı
high_5x = (all_values >= 5.0).sum()
high_10x = (all_values >= 10.0).sum()
high_20x = (all_values >= 20.0).sum()
high_50x = (all_values >= 50.0).sum()

print(f"\n📊 YÜKSEK DEĞER DAĞILIMI:")
print(f"  5x+: {high_5x:,} ({high_5x/len(all_values)*100:.1f}%)")
print(f"  10x+: {high_10x:,} ({high_10x/len(all_values)*100:.1f}%)")
print(f"  20x+: {high_20x:,} ({high_20x/len(all_values)*100:.1f}%)")
print(f"  50x+: {high_50x:,} ({high_50x/len(all_values)*100:.1f}%)")

# =============================================================================
# GELİŞMİŞ FEATURE ENGINEERING (YÜKSEK ÇARPAN ÖZELLİKLERİ)
# =============================================================================
print("\n🔧 Gelişmiş feature extraction (yüksek çarpan odaklı)...")

def extract_high_x_features(history):
    """
    Yüksek çarpan pattern'lerine odaklanan feature engineering
    """
    base_features = FeatureEngineering.extract_all_features(history)
    
    # Yüksek çarpan spesifik özellikler
    high_x_features = {}
    
    # Son yüksek çarpanlara olan mesafe
    last_5x_idx = -1
    last_10x_idx = -1
    last_20x_idx = -1
    
    for i in range(len(history)-1, -1, -1):
        if history[i] >= 5.0 and last_5x_idx == -1:
            last_5x_idx = i
        if history[i] >= 10.0 and last_10x_idx == -1:
            last_10x_idx = i
        if history[i] >= 20.0 and last_20x_idx == -1:
            last_20x_idx = i
    
    current_idx = len(history) - 1
    
    high_x_features['distance_from_last_5x'] = current_idx - last_5x_idx if last_5x_idx != -1 else 999
    high_x_features['distance_from_last_10x'] = current_idx - last_10x_idx if last_10x_idx != -1 else 999
    high_x_features['distance_from_last_20x'] = current_idx - last_20x_idx if last_20x_idx != -1 else 999
    
    # Son 500 ve 100 eldeki yüksek çarpan frekansı
    recent_500 = history[-500:] if len(history) >= 500 else history
    recent_100 = history[-100:] if len(history) >= 100 else history
    
    high_x_features['high_5x_freq_last_500'] = sum(1 for v in recent_500 if v >= 5.0) / len(recent_500)
    high_x_features['high_10x_freq_last_500'] = sum(1 for v in recent_500 if v >= 10.0) / len(recent_500)
    high_x_features['high_5x_freq_last_100'] = sum(1 for v in recent_100 if v >= 5.0) / len(recent_100)
    high_x_features['high_10x_freq_last_100'] = sum(1 for v in recent_100 if v >= 10.0) / len(recent_100)
    
    # Yüksek çarpan volatilitesi
    high_values = [v for v in recent_500 if v >= 5.0]
    if len(high_values) >= 2:
        high_x_features['high_x_volatility'] = np.std(high_values)
        high_x_features['high_x_trend'] = (high_values[-1] - high_values[0]) / len(high_values)
    else:
        high_x_features['high_x_volatility'] = 0.0
        high_x_features['high_x_trend'] = 0.0
    
    # Merge with base features
    base_features.update(high_x_features)
    
    return base_features

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================
window_size = 1000
X_features = []
y_regression = []

print("🔧 Yüksek çarpan odaklı özellikler çıkarılıyor...")

for i in tqdm(range(window_size, len(all_values)-1), desc='High-X Features'):
    hist = all_values[:i].tolist()
    target = all_values[i]
    
    # Yüksek çarpan odaklı feature extraction
    feats = extract_high_x_features(hist)
    X_features.append(list(feats.values()))
    
    # Regression target
    y_regression.append(target)

X = np.array(X_features)
y_reg = np.array(y_regression)

print(f"✅ {len(X):,} örnek hazırlandı")
print(f"✅ Feature sayısı: {X.shape[1]}")

# =============================================================================
# FEATURE İSİMLERİ
# =============================================================================
feature_names = list(extract_high_x_features(all_values[:window_size].tolist()).keys())
print(f"✅ Feature isimleri oluşturuldu: {len(feature_names)} adet")

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

# Test seti: Son 1000 kayıt
test_size = 1000
train_end = len(X) - test_size

# Train/Test split (kronolojik)
X_train = X[:train_end]
X_test = X[train_end:]
y_reg_train = y_reg[:train_end]
y_reg_test = y_reg[train_end:]

print(f"✅ Train: {len(X_train):,}")
print(f"✅ Test: {len(X_test):,} (tüm verinin son {test_size} kaydı)")

# Validation için train setini böl (kronolojik)
val_size = int(len(X_train) * 0.2)
val_start = len(X_train) - val_size

X_tr = X_train[:val_start]
X_val = X_train[val_start:]
y_reg_tr = y_reg_train[:val_start]
y_reg_val = y_reg_train[val_start:]

print(f"   ├─ Actual Train: {len(X_tr):,}")
print(f"   └─ Validation: {len(X_val):,} (train'in son %20'si)")

# =============================================================================
# TABNET MODEL EĞİTİMİ
# =============================================================================
print("\n" + "="*80)
print("🚀 TABNET HUNTER MODEL EĞİTİMİ")
print("="*80)

training_start = time.time()

# Özel kayıp fonksiyonu
criterion = HighXWeightedMSELoss()

# TabNet parametreleri
tabnet_params = {
    'n_d': 64,                    # Decision layer width
    'n_a': 64,                    # Attention layer width
    'n_steps': 5,                 # Decision steps (attention steps)
    'gamma': 1.5,                 # Feature selection regularization
    'lambda_sparse': 1e-3,        # Sparsity regularization
    'optimizer_fn': torch.optim.Adam,
    'optimizer_params': dict(lr=2e-2, weight_decay=1e-5),
    'mask_type': 'entmax',        # Attention mask type
    'scheduler_params': dict(step_size=50, gamma=0.9),
    'verbose': 10,
    'device_name': device,
    'seed': 42
}

print("📊 Model Konfigürasyonu:")
print(f"  • n_d (decision width): {tabnet_params['n_d']}")
print(f"  • n_a (attention width): {tabnet_params['n_a']}")
print(f"  • n_steps (attention steps): {tabnet_params['n_steps']}")
print(f"  • gamma (feature selection): {tabnet_params['gamma']}")
print(f"  • Loss: HighXWeightedMSE (yüksek değerlere odaklı)")
print(f"  • Device: {device}")

# Model oluştur
model = TabNetRegressor(**tabnet_params)

# Eğitim
print("\n🔥 TabNet eğitimi başlıyor...")
print("⏱️  Tahmini süre: 30-45 dakika")
print()

model.fit(
    X_train=X_tr,
    y_train=y_reg_tr,
    eval_set=[(X_val, y_reg_val)],
    eval_name=['validation'],
    eval_metric=['mae'],
    max_epochs=100,
    patience=20,
    batch_size=256,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False,
    loss_fn=criterion,
    pretraining_ratio=0.1  # %10 pretraining
)

training_time = time.time() - training_start
print(f"\n✅ TabNet eğitimi tamamlandı! Süre: {training_time/60:.1f} dakika")

# =============================================================================
# MODEL KAYDETME
# =============================================================================
print("\n💾 Model kaydediliyor...")

model_path = os.path.join(DRIVE_MODEL_DIR, 'tabnet_high_x_model')
model.save_model(model_path)

# Scaler'ı da kaydet
import joblib
scaler_path = os.path.join(DRIVE_MODEL_DIR, 'tabnet_scaler.pkl')
joblib.dump(scaler, scaler_path)

print(f"✅ Model kaydedildi: {model_path}.zip")
print(f"✅ Scaler kaydedildi: {scaler_path}")

# =============================================================================
# DETAYLI DEĞERLENDİRME
# =============================================================================
print("\n" + "="*80)
print("📊 DETAYLI PERFORMANS DEĞERLENDİRMESİ")
print("="*80)

# Test tahminleri
y_pred = model.predict(X_test)

# Genel metrikler
mae = mean_absolute_error(y_reg_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred))
r2 = r2_score(y_reg_test, y_pred)

print(f"📊 GENEL METRİKLER:")
print(f"  MAE: {mae:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²: {r2:.4f}")

# Yüksek değer bazında metrikler
def evaluate_high_performance(y_true, y_pred, threshold):
    """Belirli bir threshold üstü için performans değerlendirme"""
    mask = y_true >= threshold
    if mask.sum() == 0:
        return 0.0, 0.0, 0
    
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    mae_filtered = mean_absolute_error(y_true_filtered, y_pred_filtered)
    accuracy_within_50pct = np.mean(np.abs(y_pred_filtered - y_true_filtered) / y_true_filtered <= 0.5)
    
    return mae_filtered, accuracy_within_50pct, mask.sum()

# Farklı threshold'lar için değerlendirme
thresholds = [5.0, 10.0, 20.0, 50.0]

print(f"\n📊 YÜKSEK DEĞER PERFORMANSI:")
for threshold in thresholds:
    mae_thr, acc_thr, count_thr = evaluate_high_performance(y_reg_test, y_pred, threshold)
    print(f"  {threshold}x+: MAE={mae_thr:.3f}, %50 Accuracy={acc_thr*100:.1f}%, Count={count_thr}")

# Yüksek değer yakalama oranı
high_predictions = y_pred >= 5.0
high_actuals = y_reg_test >= 5.0

if high_actuals.sum() > 0:
    high_recall = (high_predictions & high_actuals).sum() / high_actuals.sum()
    high_precision = (high_predictions & high_actuals).sum() / high_predictions.sum() if high_predictions.sum() > 0 else 0
    
    print(f"\n📊 YÜKSEK DEĞER YAKALAMA:")
    print(f"  5x+ Recall: {high_recall*100:.1f}% (gerçek yüksek değerleri yakalama oranı)")
    print(f"  5x+ Precision: {high_precision*100:.1f}% (tahminlerin doğruluk oranı)")

# =============================================================================
# HUNTER SIMÜLASYONU (Yüksek Değer Avı Testi)
# =============================================================================
print("\n" + "="*80)
print("🚀 HUNTER SIMÜLASYONU - Yüksek Değer Avı Testi")
print("="*80)

print("Strateji: Hunter 8x+ tahmin ettiğinde yüksek değer stratejisi")
print("Hedef: Yüksek çarpanları yakalamak")

test_count = len(y_reg_test)
initial_bankroll = test_count * 10
bet_amount = 10.0

print(f"📊 Test Veri Sayısı: {test_count:,}")
print(f"💰 Başlangıç Kasası: {initial_bankroll:,.2f} TL")
print(f"💵 Bahis Tutarı: {bet_amount:.2f} TL")

# Hunter stratejisi
hunter_wallet = initial_bankroll
total_bets = 0
total_wins = 0
total_losses = 0
high_x_wins = 0

for i in range(len(y_reg_test)):
    hunter_pred = y_pred[i]
    actual_value = y_reg_test[i]
    
    # Sadece 8x+ tahmin ettiğinde oyna
    if hunter_pred >= 8.0:
        hunter_wallet -= bet_amount
        total_bets += 1
        
        # Çıkış stratejisi: Tahminin %80'inde çık
        exit_point = hunter_pred * 0.80
        
        if actual_value >= exit_point:
            hunter_wallet += exit_point * bet_amount
            total_wins += 1
            
            # Yüksek değer yakalama kontrolü
            if actual_value >= 10.0:
                high_x_wins += 1
        else:
            total_losses += 1

# Sonuçlar
profit_loss = hunter_wallet - initial_bankroll
roi = (profit_loss / initial_bankroll) * 100
win_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0
high_x_rate = (high_x_wins / total_bets * 100) if total_bets > 0 else 0

print(f"\n📊 HUNTER SONUÇLARI:")
print(f"{'='*70}")
print(f"Toplam Oyun: {total_bets:,}")
print(f"✅ Kazanan: {total_wins:,} ({win_rate:.1f}%)")
print(f"❌ Kaybeden: {total_losses:,}")
print(f"🚀 Yüksek Değer Kazançları: {high_x_wins:,} ({high_x_rate:.1f}%)")
print(f"")
print(f"💰 Başlangıç Kasası: {initial_bankroll:,.2f} TL")
print(f"💰 Final Kasa: {hunter_wallet:,.2f} TL")
print(f"📈 Net Kar/Zarar: {profit_loss:+,.2f} TL")
print(f"📊 ROI: {roi:+.2f}%")
print(f"{'='*70}")

# =============================================================================
# FEATURE IMPORTANCE (TabNet)
# =============================================================================
print("\n" + "="*80)
print("🔍 TABNET FEATURE IMPORTANCE")
print("="*80)

try:
    # Global feature importance
    feature_importance = model.feature_importances_
    
    # Top 15 önemli özellikler
    top_indices = np.argsort(feature_importance)[-15:][::-1]
    
    print("📊 TOP 15 ÖNEMLİ ÖZELLİKLER:")
    for i, idx in enumerate(top_indices, 1):
        feature_name = feature_names[idx]
        importance = feature_importance[idx]
        print(f"  {i:2d}. {feature_name:30s}: {importance:.4f}")
        
except Exception as e:
    print(f"⚠️ Feature importance alınamadı: {e}")

# =============================================================================
# MODEL KAYDETME
# =============================================================================
print("\n" + "="*80)
print("💾 MODEL BİLGİLERİ KAYDEDİLİYOR")
print("="*80)

# Model bilgileri
info = {
    'model': 'TabNet_Hunter',
    'version': '1.0',
    'date': datetime.now().strftime('%Y-%m-%d'),
    'purpose': 'Yüksek-X Avcısı - 10x+ çarpan tahmini',
    'training_time_minutes': round(training_time/60, 1),
    'feature_count': X.shape[1],
    'sample_count': len(X_train),
    'hyperparameters': {
        'n_d': tabnet_params['n_d'],
        'n_a': tabnet_params['n_a'],
        'n_steps': tabnet_params['n_steps'],
        'gamma': tabnet_params['gamma'],
        'max_epochs': 100,
        'patience': 20,
        'batch_size': 256,
        'loss_function': 'HighXWeightedMSE'
    },
    'metrics': {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2_score': float(r2),
        'high_value_performance': {
            f'{threshold}x_plus': {
                'mae': float(evaluate_high_performance(y_reg_test, y_pred, threshold)[0]),
                'accuracy_within_50pct': float(evaluate_high_performance(y_reg_test, y_pred, threshold)[1]),
                'count': int(evaluate_high_performance(y_reg_test, y_pred, threshold)[2])
            } for threshold in thresholds
        },
        'high_value_detection': {
            'recall_5x_plus': float(high_recall) if 'high_recall' in locals() else 0.0,
            'precision_5x_plus': float(high_precision) if 'high_precision' in locals() else 0.0
        }
    },
    'hunter_simulation': {
        'roi': float(roi),
        'win_rate': float(win_rate),
        'high_x_win_rate': float(high_x_rate),
        'total_bets': int(total_bets),
        'profit_loss': float(profit_loss)
    },
    'feature_names': feature_names
}

# Model bilgilerini kaydet
info_path = os.path.join(DRIVE_MODEL_DIR, 'hunter_model_info.json')
with open(info_path, 'w') as f:
    json.dump(info, f, indent=2)

print(f"✅ Model bilgileri kaydedildi: {info_path}")

# =============================================================================
# FINAL RAPOR
# =============================================================================
print("\n" + "="*80)
print("🎉 TABNET HUNTER TRAINING TAMAMLANDI!")
print("="*80)
print(f"Toplam Süre: {training_time/60:.1f} dakika")
print()

# Hedef kontrolü
high_5x_mae, high_5x_acc, _ = evaluate_high_performance(y_reg_test, y_pred, 5.0)

if high_5x_acc >= 0.60 and mae < 5.0:
    print("✅ ✅ ✅ TÜM HEDEFLER BAŞARIYLA AŞILDI!")
    print(f"  🚀 5x+ Accuracy: {high_5x_acc*100:.1f}% (Hedef: 60%+)")
    print(f"  📊 MAE: {mae:.3f} (Hedef: <5.0)")
    print(f"  🎯 Hunter ROI: {roi:+.2f}%")
    print("\n🚀 Hunter modeli kullanıma hazır!")
elif high_5x_acc >= 0.50:
    print("✅ ✅ İYİ PERFORMANS!")
    print(f"  🚀 5x+ Accuracy: {high_5x_acc*100:.1f}%")
    print(f"  📊 MAE: {mae:.3f}")
else:
    print("⚠️ Geliştirme gerekli")
    print(f"  🚀 5x+ Accuracy: {high_5x_acc*100:.1f}% (Hedef: 60%+)")

print(f"\n{'='*80}")
print(f"📁 Model kaydedildi: {DRIVE_MODEL_DIR}")
print(f"🔥 Özel Loss: HighXWeightedMSE (yüksek değerlere odaklı)")
print(f"{'='*80}")
print(f"Bitiş: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
