#!/usr/bin/env python3
"""
🤖 JetX XGBOOST TRAINING - Feature Engineering Bazlı Model

AMAÇ: XGBoost ile hızlı ve etkili tahmin modeli eğitmek

AVANTAJLAR:
- Çok daha hızlı eğitim (~30-60 dakika vs 2-3 saat)
- Feature importance analizi yapılabilir
- Daha az bellek kullanımı
- Overfitting'e daha dirençli

STRATEJI:
- XGBRegressor: Değer tahmini için
- XGBClassifier: 1.5 eşik tahmini için (scale_pos_weight ile dengeleme)

HEDEFLER:
- 1.5 ALTI Doğruluk: %70-75%+
- 1.5 ÜSTÜ Doğruluk: %70-75%+
- MAE: < 5.0

SÜRE: ~30-60 dakika (GPU ile)
"""

import subprocess
import sys
import os
import time
from datetime import datetime

print("="*80)
print("🤖 JetX XGBOOST TRAINING")
print("="*80)
print(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Kütüphaneleri yükle
print("📦 Kütüphaneler yükleniyor...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                      "xgboost", "scikit-learn", "pandas", "numpy", 
                      "scipy", "joblib", "matplotlib", "seaborn", "tqdm",
                      "PyWavelets", "nolds"])

import numpy as np
import pandas as pd
import joblib
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

print(f"✅ XGBoost: {xgb.__version__}")

# Proje yükle
if not os.path.exists('jetxpredictor'):
    print("\n📥 Proje klonlanıyor...")
    subprocess.check_call(["git", "clone", "https://github.com/onndd/jetxpredictor.git"])

os.chdir('jetxpredictor')
sys.path.append(os.getcwd())

from category_definitions import CategoryDefinitions, FeatureEngineering
from utils.advanced_bankroll import AdvancedBankrollManager
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
window_size = 1000  # Neural network ile aynı
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

# Train/Test split - STRATIFIED SAMPLING EKLENDI
X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
    X, y_reg, y_cls, test_size=0.2, shuffle=True, stratify=y_cls, random_state=42
)

print(f"✅ Train: {len(X_train):,}, Test: {len(X_test):,}")

# =============================================================================
# XGBOOST REGRESSOR (Değer Tahmini)
# =============================================================================
print("\n" + "="*80)
print("🎯 XGBOOST REGRESSOR EĞİTİMİ (Değer Tahmini)")
print("="*80)

reg_start = time.time()

# XGBoost parametreleri
regressor = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    random_state=42,
    tree_method='hist',  # CPU için optimize
    objective='reg:squarederror',
    eval_metric='mae'
)

print("📊 Model Parametreleri:")
print(f"  n_estimators: 500")
print(f"  max_depth: 8")
print(f"  learning_rate: 0.05")
print(f"  subsample: 0.8")
print(f"  colsample_bytree: 0.8\n")

# Eğitim
print("🔥 Eğitim başlıyor...")
regressor.fit(
    X_train, y_reg_train,
    eval_set=[(X_test, y_reg_test)],
    verbose=50
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
# XGBOOST CLASSIFIER (Eşik Tahmini)
# =============================================================================
print("\n" + "="*80)
print("🎯 XGBOOST CLASSIFIER EĞİTİMİ (1.5 Eşik Tahmini)")
print("="*80)

cls_start = time.time()

# sample_weight ile class dengeleme (HATA DÜZELTİLDİ!)
# Eski hatalı kod: scale_pos_weight = above_count / below_count (YANLIŞ!)
# Yeni çözüm: sample_weight kullanarak doğrudan 1.5 altı örneklere ağırlık ver
below_count = (y_cls_train == 0).sum()
above_count = (y_cls_train == 1).sum()

# Her örnek için sample weight hesapla
# 1.5 altı için 5.0x, 1.5 üstü için 1.0x
WEIGHT_MULTIPLIER = 5.0  # 1.5 altı için ağırlık çarpanı
sample_weights_train = np.where(y_cls_train == 0, WEIGHT_MULTIPLIER, 1.0)

print(f"📊 SAMPLE WEIGHT (HATA DÜZELTİLDİ!):")
print(f"  1.5 altı: {below_count:,} örnek → Her biri {WEIGHT_MULTIPLIER}x ağırlık")
print(f"  1.5 üstü: {above_count:,} örnek → Her biri 1.0x ağırlık")
print(f"  ✅ scale_pos_weight HATASI düzeltildi → sample_weight kullanılıyor\n")

# XGBoost parametreleri - scale_pos_weight KALDIRILDI
classifier = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    # scale_pos_weight KALDIRILDI - sample_weight kullanılacak
    random_state=42,
    tree_method='hist',
    objective='binary:logistic',
    eval_metric='logloss'
)

print("📊 Model Parametreleri:")
print(f"  n_estimators: 500")
print(f"  max_depth: 7")
print(f"  learning_rate: 0.05")
print(f"  sample_weight kullanılıyor: {WEIGHT_MULTIPLIER}x (1.5 altı için)\n")

# Eğitim - sample_weight ile
print("🔥 Eğitim başlıyor (sample_weight ile)...")
classifier.fit(
    X_train, y_cls_train,
    sample_weight=sample_weights_train,  # SAMPLE WEIGHT EKLENDI
    eval_set=[(X_test, y_cls_test)],
    verbose=50
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
    # GELİŞMİŞ SANAL KASA SİMÜLASYONU
    # =============================================================================
    print("\n" + "="*70)
    print("💰 GELİŞMİŞ SANAL KASA SİMÜLASYONU (Kelly Criterion)")
    print("="*70)
    print("3 farklı risk tolerance stratejisi ile test ediliyor...")
    
    # 3 farklı risk tolerance ile test
    for risk_tolerance in ['conservative', 'moderate', 'aggressive']:
        print(f"\n{'='*70}")
        print(f"📊 {risk_tolerance.upper()} STRATEJİ")
        print(f"{'='*70}")
        
        # Advanced Bankroll Manager oluştur
        bankroll = AdvancedBankrollManager(
            initial_bankroll=1000.0,
            risk_tolerance=risk_tolerance,
            win_multiplier=1.5,
            min_bet=10.0
        )
        
        # Test seti üzerinde simülasyon
        for i in range(len(y_reg_test)):
            # Model tahmini
            model_pred_cls = y_cls_pred[i]  # 0 veya 1
            model_confidence = y_cls_proba[i]  # Confidence score
            actual_value = y_reg_test[i]
            
            # Model "1.5 üstü" (1) tahmin ediyorsa bahis yap
            if model_pred_cls == 1:
                # Optimal bahis miktarını hesapla (Kelly Criterion)
                bet_size = bankroll.calculate_bet_size(
                    confidence=model_confidence,
                    predicted_value=1.5
                )
                
                # Bet size > 0 ise bahis yap
                if bet_size > 0:
                    result = bankroll.place_bet(
                        bet_size=bet_size,
                        predicted_value=1.5,
                        actual_value=actual_value,
                        confidence=model_confidence
                    )
                
                # Stop-loss veya take-profit kontrolü
                should_stop, reason = bankroll.should_stop()
                if should_stop:
                    print(f"\n⚠️ {reason}")
                    print(f"Simülasyon durduruluyor (Test örneği {i+1}/{len(y_reg_test)})")
                    break
        
        # Detaylı rapor
        bankroll.print_report()
    
    print("\n" + "="*70)
    print("✅ Gelişmiş sanal kasa simülasyonu tamamlandı!")
    print("="*70)

# =============================================================================
# MODEL KAYDETME
# =============================================================================
print("\n" + "="*80)
print("💾 Modeller kaydediliyor...")
print("="*80)

# models/ klasörünü oluştur
os.makedirs('models', exist_ok=True)

# Modelleri kaydet
regressor.save_model('models/xgboost_regressor.json')
classifier.save_model('models/xgboost_classifier.json')
joblib.dump(scaler, 'models/xgboost_scaler.pkl')

print("✅ Dosyalar kaydedildi:")
print("  - models/xgboost_regressor.json")
print("  - models/xgboost_classifier.json")
print("  - models/xgboost_scaler.pkl")

# Model bilgilerini kaydet
import json
total_time = reg_time + cls_time
info = {
    'model': 'XGBOOST_DUAL_MODEL',
    'version': '1.0_XGBOOST',
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
            'n_estimators': 500,
            'max_depth': 8,
            'learning_rate': 0.05
        },
        'classifier': {
            'n_estimators': 500,
            'max_depth': 7,
            'learning_rate': 0.05,
            'scale_pos_weight': float(scale_pos_weight)
        }
    },
    'top_features': [{'name': feat, 'importance': float(imp)} for feat, imp in top_features]
}

with open('models/xgboost_model_info.json', 'w') as f:
    json.dump(info, f, indent=2)

print("  - models/xgboost_model_info.json")

print(f"\n📊 Model Bilgisi:")
print(json.dumps(info, indent=2))

# Google Colab'da indir - İyileştirilmiş kontrol
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    try:
        from google.colab import files
        print("\n📥 Dosyalar indiriliyor...")
        files.download('models/xgboost_regressor.json')
        print("✅ xgboost_regressor.json indirildi")
        files.download('models/xgboost_classifier.json')
        print("✅ xgboost_classifier.json indirildi")
        files.download('models/xgboost_scaler.pkl')
        print("✅ xgboost_scaler.pkl indirildi")
        files.download('models/xgboost_model_info.json')
        print("✅ xgboost_model_info.json indirildi")
        print("\n✅ Tüm dosyalar başarıyla indirildi!")
        print("📌 Bu dosyaları lokal projenizin models/ klasörüne kopyalayın")
    except Exception as e:
        print(f"\n⚠️ İndirme hatası: {e}")
        print("📁 Dosyalar models/ klasöründe kaydedildi.")
else:
    print("\n⚠️ Google Colab ortamı algılanamadı - dosyalar sadece kaydedildi")
    print("📁 Dosyalar models/ klasöründe mevcut:")
    print("   • xgboost_regressor.json")
    print("   • xgboost_classifier.json")
    print("   • xgboost_scaler.pkl")
    print("   • xgboost_model_info.json")
    print("\n💡 Not: Bu script Google Colab'da çalıştırıldığında dosyalar otomatik indirilir.")

# Final rapor
print("\n" + "="*80)
print("🎉 XGBOOST TRAINING TAMAMLANDI!")
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
    print("\nXGBoost Neural Network'e göre daha basit ama hızlı bir alternatiftir.")

print("\n📁 Sonraki adım:")
print("  1. XGBoost modellerini lokal projeye kopyalayın")
print("  2. Predictor'da model_type='xgboost' ile kullanın")
print("  3. Neural Network ile karşılaştırın")
print("="*80)
print(f"Bitiş: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
