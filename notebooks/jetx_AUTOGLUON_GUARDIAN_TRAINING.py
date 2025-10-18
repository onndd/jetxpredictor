#!/usr/bin/env python3
"""
🛡️ JetX AUTOGLUON GUARDIAN TRAINING - Sermaye Koruma Uzmanı

AMAÇ: 1.5x altı tahminleri yüksek doğrulukla yapacak, sermayeyi korumaya odaklanmış 
en güçlü classification modeli AutoGluon ile otomatik olarak bulmak.

STRATEJİ:
- AutoML: Birden fazla model türünü test et (XGBoost, LightGBM, CatBoost, Neural Networks)
- Hedef: 1.5x binary classification (0 = 1.5 altı, 1 = 1.5 üstü)
- Focus: Para kaybını önlemek (False Positive minimizasyonu)
- Preset: high_quality (en iyi modeller, makul süre)

HEDEFLER:
- 1.5 ALTI Doğruluk: %80+ (para kaybını önleme)
- 1.5 ÜSTÜ Doğruluk: %70+ (fırsat kaçırma minimizasyonu)
- False Positive Rate: <15% (kritik!)
- Precision: %75+ (tahmin ettiğinde doğru olma oranı)

SÜRE: ~30-60 dakika (GPU ile)
"""

import subprocess
import sys
import os
import time
from datetime import datetime
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
import sqlite3
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🛡️ JetX AUTOGLUON GUARDIAN TRAINING - Sermaye Koruma Uzmanı")
print("="*80)
print(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Kütüphaneleri yükle
print("📦 Kütüphaneler yükleniyor...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                      "autogluon.tabular", "scikit-learn", "pandas", "numpy", 
                      "scipy", "matplotlib", "seaborn", "tqdm",
                      "PyWavelets", "nolds"])

try:
    from autogluon.tabular import TabularPredictor
    print("✅ AutoGluon başarıyla yüklendi")
except ImportError as e:
    print(f"❌ AutoGluon yüklenemedi: {e}")
    sys.exit(1)

# Google Drive mount (Colab için)
try:
    from google.colab import drive
    
    if not os.path.exists('/content/drive'):
        print("\n📦 Google Drive bağlanıyor...")
        drive.mount('/content/drive')
    
    # Model kayıt dizini
    DRIVE_MODEL_DIR = '/content/drive/MyDrive/JetX_Models/AutoGluon_Guardian/'
    os.makedirs(DRIVE_MODEL_DIR, exist_ok=True)
    print(f"✅ Google Drive bağlandı: {DRIVE_MODEL_DIR}")
    USE_DRIVE = True
except ImportError:
    print("⚠️ Google Colab dışında - lokal kayıt kullanılacak")
    DRIVE_MODEL_DIR = 'models/autogluon_guardian/'
    os.makedirs(DRIVE_MODEL_DIR, exist_ok=True)
    USE_DRIVE = False
except Exception as e:
    print(f"⚠️ Google Drive mount hatası: {e}")
    DRIVE_MODEL_DIR = 'models/autogluon_guardian/'
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
# FEATURE ENGINEERING (CatBoost ile aynı)
# =============================================================================
print("\n🔧 Feature extraction...")
window_size = 1000  # CatBoost ile aynı window size
X_features = []
y_classification = []

for i in tqdm(range(window_size, len(all_values)-1), desc='Features'):
    hist = all_values[:i].tolist()
    target = all_values[i]
    
    # Tüm özellikleri çıkar (CatBoost ile aynı)
    feats = FeatureEngineering.extract_all_features(hist)
    X_features.append(list(feats.values()))
    
    # Classification target (1.5 altı/üstü)
    y_classification.append(1 if target >= 1.5 else 0)

X = np.array(X_features)
y_cls = np.array(y_classification)

print(f"✅ {len(X):,} örnek hazırlandı")
print(f"✅ Feature sayısı: {X.shape[1]}")

# =============================================================================
# FEATURE İSİMLERİ
# =============================================================================
feature_names = list(FeatureEngineering.extract_all_features(all_values[:window_size].tolist()).keys())
print(f"✅ Feature isimleri oluşturuldu: {len(feature_names)} adet")

# =============================================================================
# DATAFRAME OLUŞTURMA (AutoGluon için)
# =============================================================================
print("\n📊 DataFrame oluşturuluyor...")

# Features DataFrame
X_df = pd.DataFrame(X, columns=feature_names)

# Target DataFrame
y_df = pd.DataFrame({'is_above_1_5': y_cls})

# Birleştir
train_data = pd.concat([X_df, y_df], axis=1)

print(f"✅ DataFrame oluşturuldu: {train_data.shape}")
print(f"  • Features: {X_df.shape[1]}")
print(f"  • Samples: {len(train_data)}")
print(f"  • Target: 'is_above_1_5'")

# =============================================================================
# TIME-SERIES SPLIT (KRONOLOJIK) - SHUFFLE YOK!
# =============================================================================
print("\n📊 TIME-SERIES SPLIT (Kronolojik Bölme)...")
print("⚠️  UYARI: Shuffle devre dışı - Zaman serisi yapısı korunuyor!")

# Test seti: Son 1000 kayıt
test_size = 1000
train_end = len(train_data) - test_size

# Train/Test split (kronolojik)
train_df = train_data[:train_end]
test_df = train_data[train_end:]

print(f"✅ Train: {len(train_df):,}")
print(f"✅ Test: {len(test_df):,} (tüm verinin son {test_size} kaydı)")

# =============================================================================
# SAMPLE WEIGHTS (Class Imbalance için)
# =============================================================================
print("\n⚖️ Sample weights hesaplanıyor...")

# Class weights
class_counts = train_df['is_above_1_5'].value_counts()
count_0 = class_counts.get(0, 0)  # 1.5 altı
count_1 = class_counts.get(1, 0)  # 1.5 üstü

# Weight: az olan sınıfa daha fazla ağırlık
weight_0 = 1.0   # 1.5 altı (çok olan)
weight_1 = 3.0   # 1.5 üstü (az olan)

print(f"📊 CLASS WEIGHTS:")
print(f"  1.5 altı (class 0): {weight_0:.1f}x (count: {count_0:,})")
print(f"  1.5 üstü (class 1): {weight_1:.1f}x (count: {count_1:,})")

# Sample weights oluştur
sample_weights = train_df['is_above_1_5'].map({0: weight_0, 1: weight_1}).values

print(f"✅ Sample weights oluşturuldu: {len(sample_weights)} adet")

# =============================================================================
# AUTOGLUON MODEL EĞİTİMİ
# =============================================================================
print("\n" + "="*80)
print("🤖 AUTOGLUON GUARDIAN MODEL EĞİTİMİ")
print("="*80)

training_start = time.time()

# AutoGluon predictor
predictor = TabularPredictor(
    label='is_above_1_5',
    path=DRIVE_MODEL_DIR,
    eval_metric='accuracy',  # Ana metrik
    problem_type='binary'
)

print("📊 Model Konfigürasyonu:")
print(f"  • Preset: high_quality (en iyi modeller)")
print(f"  • Time limit: 1800 saniye (30 dakika)")
print(f"  • Eval metric: accuracy")
print(f"  • Sample weights: Evet (class imbalance)")
print(f"  • Feature engineering: AutoGluon otomatik")
print(f"  • Hyperparameter tuning: Evet")

# Eğitim
print("\n🔥 AutoGluon eğitimi başlıyor...")
print("⏱️  Tahmini süre: 30-45 dakika")
print()

predictor.fit(
    train_data=train_df,
    sample_weights=sample_weights,
    presets='high_quality',
    time_limit=1800,  # 30 dakika
    num_bag_folds=5,
    num_bag_sets=1,
    num_stack_levels=1,
    holdout_frac=0.1,  # Son %10'u validation için ayır
    verbosity=2
)

training_time = time.time() - training_start
print(f"\n✅ AutoGluon eğitimi tamamlandı! Süre: {training_time/60:.1f} dakika")

# =============================================================================
# MODEL LİDERBOARD
# =============================================================================
print("\n" + "="*80)
print("🏆 MODEL LİDERBOARD")
print("="*80)

leaderboard = predictor.leaderboard(test_df, silent=True)
print(leaderboard)

# En iyi model
best_model_name = leaderboard.iloc[0]['model']
best_model_score = leaderboard.iloc[0]['score_test']
print(f"\n🥇 En iyi model: {best_model_name}")
print(f"📊 Test skoru: {best_model_score:.4f}")

# =============================================================================
# DETAYLI DEĞERLENDİRME
# =============================================================================
print("\n" + "="*80)
print("📊 DETAYLI PERFORMANS DEĞERLENDİRMESİ")
print("="*80)

# Test tahminleri
y_true = test_df['is_above_1_5'].values
y_pred = predictor.predict(test_df)
y_proba = predictor.predict_proba(test_df, as_multiclass=False)

# Genel metrikler
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print(f"📊 GENEL METRİKLER:")
print(f"  Accuracy: {accuracy*100:.2f}%")
print(f"  Precision: {precision*100:.2f}%")
print(f"  Recall: {recall*100:.2f}%")

# Sınıf bazında accuracy
below_mask = y_true == 0
above_mask = y_true == 1

below_acc = accuracy_score(y_true[below_mask], y_pred[below_mask]) if below_mask.sum() > 0 else 0
above_acc = accuracy_score(y_true[above_mask], y_pred[above_mask]) if above_mask.sum() > 0 else 0

print(f"\n📊 SINIF BAZINDA DOĞRULUK:")
print(f"  🔴 1.5 Altı Doğruluk: {below_acc*100:.2f}% (PARA KAYBI KORUMASI)")
print(f"  🟢 1.5 Üstü Doğruluk: {above_acc*100:.2f}% (FIRSAT YAKALAMA)")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print(f"\n📋 CONFUSION MATRIX:")
print(f"                Tahmin")
print(f"Gerçek   1.5 Altı | 1.5 Üstü")
print(f"1.5 Altı {cm[0,0]:6d}   | {cm[0,1]:6d}  ⚠️ PARA KAYBI")
print(f"1.5 Üstü {cm[1,0]:6d}   | {cm[1,1]:6d}")

# False Positive Rate (en kritik metrik)
if cm[0,0] + cm[0,1] > 0:
    fpr = cm[0,1] / (cm[0,0] + cm[0,1])
    print(f"\n💰 PARA KAYBI RİSKİ (False Positive Rate): {fpr*100:.1f}%")
    if fpr < 0.15:
        print("✅ HEDEF AŞILDI! (%15 altında)")
    else:
        print(f"⚠️ HEDEFİN ÜZÜNDE (Hedef: <15%)")

# Classification Report
print(f"\n📊 DETAYLI CLASSIFICATION REPORT:")
print(classification_report(y_true, y_pred, target_names=['1.5 Altı', '1.5 Üstü']))

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================
print("\n" + "="*80)
print("🔍 FEATURE IMPORTANCE")
print("="*80)

try:
    # En iyi modelin feature importance'ı
    feature_importance = predictor.feature_importance(test_df)
    top_features = feature_importance.head(15)
    
    print("📊 TOP 15 ÖNEMLİ ÖZELLİKLER:")
    for i, (feature, importance) in enumerate(top_features.items(), 1):
        print(f"  {i:2d}. {feature:30s}: {importance:.4f}")
        
except Exception as e:
    print(f"⚠️ Feature importance alınamadı: {e}")

# =============================================================================
# GUARDIAN SIMÜLASYONU (Para Kaybı Testi)
# =============================================================================
print("\n" + "="*80)
print("🛡️ GUARDIAN SIMÜLASYONU - Para Kaybı Testi")
print("="*80)

print("Strateji: Guardian modeli 'BEKLE' dediğinde oynama")
print("Hedef: Para kaybını en aza indirmek")

test_count = len(y_true)
initial_bankroll = test_count * 10
bet_amount = 10.0

print(f"📊 Test Veri Sayısı: {test_count:,}")
print(f"💰 Başlangıç Kasası: {initial_bankroll:,.2f} TL")
print(f"💵 Bahis Tutarı: {bet_amount:.2f} TL")

# Guardian stratejisi
guardian_wallet = initial_bankroll
total_bets = 0
total_wins = 0
total_losses = 0
money_saved = 0

for i in range(len(y_true)):
    actual_value = all_values[train_end + window_size + i]  # Gerçek değer
    
    # Guardian modeli tahmini
    guardian_pred = y_pred[i]
    
    # Sadece Guardian '1.5 üstü' (1) dediğinde oyna
    if guardian_pred == 1:
        guardian_wallet -= bet_amount
        total_bets += 1
        
        if actual_value >= 1.5:
            guardian_wallet += 1.5 * bet_amount
            total_wins += 1
        else:
            total_losses += 1
    else:
        # Guardian 'BEKLE' dedi → para kaybından kurtulduk
        if actual_value < 1.5:
            money_saved += bet_amount

# Sonuçlar
profit_loss = guardian_wallet - initial_bankroll
roi = (profit_loss / initial_bankroll) * 100
win_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0

print(f"\n📊 GUARDIAN SONUÇLARI:")
print(f"{'='*70}")
print(f"Toplam Oyun: {total_bets:,}")
print(f"✅ Kazanan: {total_wins:,} ({win_rate:.1f}%)")
print(f"❌ Kaybeden: {total_losses:,}")
print(f"💰 Para Kaybından Kurtarılan: {money_saved:,} TL")
print(f"")
print(f"💰 Başlangıç Kasası: {initial_bankroll:,.2f} TL")
print(f"💰 Final Kasa: {guardian_wallet:,.2f} TL")
print(f"📈 Net Kar/Zarar: {profit_loss:+,.2f} TL")
print(f"📊 ROI: {roi:+.2f}%")
print(f"{'='*70}")

# Karşılaştırma: Tüm oyunları oynasaydık?
all_games_wallet = initial_bankroll
all_wins = 0
for i in range(len(y_true)):
    actual_value = all_values[train_end + window_size + i]
    all_games_wallet -= bet_amount
    if actual_value >= 1.5:
        all_games_wallet += 1.5 * bet_amount
        all_wins += 1

all_games_profit = all_games_wallet - initial_bankroll
all_games_roi = (all_games_profit / initial_bankroll) * 100

print(f"\n📊 KARŞILAŞTIRMA:")
print(f"  Guardian ile ROI: {roi:+.2f}%")
print(f"  Tüm oyunlar ROI: {all_games_roi:+.2f}%")
print(f"  Guardian farkı: {roi - all_games_roi:+.2f}%")

if roi > all_games_roi:
    print("✅ Guardian modeli para kaybını azalttı!")
else:
    print("⚠️ Guardian modeli beklenen performansı göstermedi")

# =============================================================================
# MODEL KAYDETME
# =============================================================================
print("\n" + "="*80)
print("💾 MODEL BİLGİLERİ KAYDEDİLİYOR")
print("="*80)

# Model bilgileri
info = {
    'model': 'AutoGluon_Guardian',
    'version': '1.0',
    'date': datetime.now().strftime('%Y-%m-%d'),
    'purpose': 'Sermaye Koruma Uzmanı - 1.5x Altı Tahmini',
    'training_time_minutes': round(training_time/60, 1),
    'best_model': best_model_name,
    'feature_count': X.shape[1],
    'sample_count': len(train_df),
    'class_weights': {0: weight_0, 1: weight_1},
    'metrics': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'below_15_accuracy': float(below_acc),
        'above_15_accuracy': float(above_acc),
        'false_positive_rate': float(fpr) if cm[0,0] + cm[0,1] > 0 else 0.0,
        'confusion_matrix': cm.tolist()
    },
    'guardian_simulation': {
        'roi': float(roi),
        'win_rate': float(win_rate),
        'total_bets': int(total_bets),
        'money_saved': int(money_saved),
        'profit_loss': float(profit_loss)
    },
    'hyperparameters': {
        'preset': 'high_quality',
        'time_limit': 1800,
        'eval_metric': 'accuracy',
        'num_bag_folds': 5,
        'num_stack_levels': 1,
        'holdout_frac': 0.1
    }
}

# Model bilgilerini kaydet
info_path = os.path.join(DRIVE_MODEL_DIR, 'guardian_model_info.json')
with open(info_path, 'w') as f:
    json.dump(info, f, indent=2)

print(f"✅ Model bilgileri kaydedildi: {info_path}")

# =============================================================================
# FINAL RAPOR
# =============================================================================
print("\n" + "="*80)
print("🎉 AUTOGLUON GUARDIAN TRAINING TAMAMLANDI!")
print("="*80)
print(f"Toplam Süre: {training_time/60:.1f} dakika")
print()

# Hedef kontrolü
if below_acc >= 0.80 and fpr < 0.15:
    print("✅ ✅ ✅ TÜM HEDEFLER BAŞARIYLA AŞILDI!")
    print(f"  🔴 1.5 ALTI: {below_acc*100:.1f}% (Hedef: 80%+)")
    print(f"  💰 Para kaybı: {fpr*100:.1f}% (Hedef: <15%)")
    print(f"  🛡️ Guardian ROI: {roi:+.2f}%")
    print("\n🚀 Guardian modeli kullanıma hazır!")
elif below_acc >= 0.75:
    print("✅ ✅ İYİ PERFORMANS!")
    print(f"  🔴 1.5 ALTI: {below_acc*100:.1f}%")
    print(f"  💰 Para kaybı: {fpr*100:.1f}%")
else:
    print("⚠️ Geliştirme gerekli")
    print(f"  🔴 1.5 ALTI: {below_acc*100:.1f}% (Hedef: 80%+)")

print(f"\n{'='*80}")
print(f"📁 Model kaydedildi: {DRIVE_MODEL_DIR}")
print(f"🥇 En iyi model: {best_model_name}")
print(f"{'='*80}")
print(f"Bitiş: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
