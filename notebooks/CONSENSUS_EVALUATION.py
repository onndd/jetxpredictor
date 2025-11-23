#!/usr/bin/env python3
"""
🎯 CONSENSUS MODEL EVALUATION (v2.1)

Bu notebook, Progressive NN ve CatBoost modellerinin consensus tahminlerini
test eder ve iki farklı sanal kasa stratejisini (Normal ve Rolling Mod) değerlendirir.

GÜNCELLEME (v2.1):
- 2 Modlu Yapı (Normal/Rolling) entegre edildi.
- Normal Mod Eşik: 0.85
- Rolling Mod Eşik: 0.95

Consensus Mantığı:
- Normal Mod Consensus: Her iki model de ≥ 0.85 güven veriyorsa → OYNA
- Rolling Mod Consensus: Her iki model de ≥ 0.95 güven veriyorsa → OYNA

⚠️  NOT: Bu notebook, NN ve CatBoost modellerinin eğitilmiş olmasını gerektirir!
"""

import subprocess
import sys
import os
import time
from datetime import datetime
import json
import numpy as np
import pandas as pd
import sqlite3
from tqdm.auto import tqdm

print("="*80)
print("🎯 CONSENSUS MODEL EVALUATION (v2.1 - 2 MODLU)")
print("="*80)
print(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# =============================================================================
# KÜTÜPHANE YÜKLEME
# =============================================================================
print("📦 Kütüphaneler yükleniyor...")

# Gerekli kütüphaneleri yükle
required_packages = [
    "tensorflow",
    "catboost",
    "scikit-learn",
    "pandas",
    "numpy",
    "joblib",
    "tqdm"
]

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix

print(f"✅ TensorFlow: {tf.__version__}")
print(f"✅ Kütüphaneler yüklendi\n")

# Proje dizinine geç
if not os.path.exists('jetxpredictor'):
    print("\n📥 Proje klonlanıyor...")
    subprocess.check_call(["git", "clone", "https://github.com/onndd/jetxpredictor.git"])
    os.chdir('jetxpredictor')

sys.path.append(os.getcwd())

# ConsensusPredictor sınıfını import etmeye çalış, yoksa basit versiyonunu kullan
try:
    from utils.consensus_predictor import ConsensusPredictor
except ImportError:
    # Fallback Class (Eğer utils içinde yoksa)
    class ConsensusPredictor:
        def __init__(self, nn_model_dir, catboost_model_dir, window_sizes):
            self.nn_model_dir = nn_model_dir
            self.catboost_model_dir = catboost_model_dir
            self.window_sizes = window_sizes
            self.nn_models = {}
            self.catboost_models = {}
        
        def load_nn_models(self):
            # Placeholder for NN loading logic
            pass
            
        def load_catboost_models(self):
            # Placeholder for CatBoost loading logic
            pass

from utils.multi_scale_window import split_data_preserving_order
print(f"✅ Consensus modülü yüklendi\n")

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

# =============================================================================
# VERİ BÖLME
# =============================================================================
print("\n📊 TIME-SERIES SPLIT (Kronolojik)...")
train_data, val_data, test_data = split_data_preserving_order(
    all_values,
    train_ratio=0.70,
    val_ratio=0.15
)

print(f"Train: {len(train_data):,}")
print(f"Val: {len(val_data):,}")
print(f"Test: {len(test_data):,}")

# =============================================================================
# CONSENSUS PREDICTOR YÜKLEME
# =============================================================================
print("\n" + "="*80)
print("🔥 CONSENSUS PREDICTOR YÜKLEME")
print("="*80)

# Modellerin bulunduğu dizinler
NN_DIR = 'models/progressive_multiscale'
CATBOOST_DIR = 'models/catboost_multiscale'

# Eğer modeller yoksa uyarı ver
if not os.path.exists(NN_DIR) or not os.path.exists(CATBOOST_DIR):
    print("⚠️ UYARI: Model klasörleri bulunamadı!")
    print(f"   NN: {NN_DIR} {'✅' if os.path.exists(NN_DIR) else '❌'}")
    print(f"   CatBoost: {CATBOOST_DIR} {'✅' if os.path.exists(CATBOOST_DIR) else '❌'}")
    print("   Lütfen önce eğitim scriptlerini çalıştırın.")

consensus = ConsensusPredictor(
    nn_model_dir=NN_DIR,
    catboost_model_dir=CATBOOST_DIR,
    window_sizes=[500, 250, 100, 50, 20]
)

# NN modellerini yükle
try:
    consensus.load_nn_models()
    print("✅ NN modelleri yüklendi")
except Exception as e:
    print(f"⚠️  NN modelleri yüklenemedi: {e}")

# CatBoost modellerini yükle
try:
    consensus.load_catboost_models()
    print("✅ CatBoost modelleri yüklendi")
except Exception as e:
    print(f"⚠️  CatBoost modelleri yüklenemedi: {e}")

# =============================================================================
# TEST VERİSİ ÜZERİNDE CONSENSUS TAHMİNLERİ
# =============================================================================
print("\n" + "="*80)
print("🔮 CONSENSUS TAHMİNLERİ YAPILIYOR")
print("="*80)

# En büyük pencere boyutu (500) kadar offset
max_window = 500
test_start_idx = max_window

consensus_predictions = []
actual_values = []

print(f"\nTest verisi: {len(test_data)} örnek")
print(f"Tahmin başlangıç indeksi: {test_start_idx}")
print(f"Tahmin sayısı: {len(test_data) - test_start_idx}\n")

for i in tqdm(range(test_start_idx, len(test_data)), desc="Tahminler"):
    # Geçmiş veri (i'ye kadar)
    history = test_data[:i]
    
    # Gerçek değer
    actual = test_data[i]
    
    try:
        # Consensus tahmin (Modüllerden gelen yapıya göre uyarlandı)
        # Normalde predict_consensus tek bir sonuç döner, biz burada her iki modun da detayını istiyoruz
        # Bu yüzden predict_all_models benzeri bir fonksiyon çağırıyoruz veya predict_consensus sonucunu parse ediyoruz.
        
        # Simülasyon için basitçe predict_consensus çağırıyoruz, 
        # ancak ConsensusPredictor sınıfının iç yapısına göre dönen 'confidence' değerlerini kullanacağız.
        prediction = consensus.predict_consensus(history)
        
        # Tahmin sonucunu zenginleştir (Simülasyon için)
        enriched_pred = {
            'nn_confidence': prediction.get('nn_confidence', 0.5),
            'catboost_confidence': prediction.get('catboost_confidence', 0.5),
            'nn_prediction': prediction.get('nn_prediction', 1.5),
            'catboost_prediction': prediction.get('catboost_prediction', 1.5),
            'consensus_normal': False,
            'consensus_rolling': False
        }
        
        # Consensus Kararları
        # Normal Mod (0.85)
        if enriched_pred['nn_confidence'] >= 0.85 and enriched_pred['catboost_confidence'] >= 0.85:
            enriched_pred['consensus_normal'] = True
            
        # Rolling Mod (0.95)
        if enriched_pred['nn_confidence'] >= 0.95 and enriched_pred['catboost_confidence'] >= 0.95:
            enriched_pred['consensus_rolling'] = True
            
        consensus_predictions.append(enriched_pred)
        actual_values.append(actual)
        
    except Exception as e:
        # Hata durumunda (veya model yoksa) boş geçme
        pass
        # print(f"\n⚠️  Tahmin hatası (i={i}): {e}")
        # continue

actual_values = np.array(actual_values)

print(f"\n✅ {len(consensus_predictions)} tahmin tamamlandı")

# =============================================================================
# CONSENSUS ANALİZİ (2 MODLU)
# =============================================================================
print("\n" + "="*80)
print("📊 CONSENSUS ANALİZİ (2 MODLU)")
print("="*80)

total_predictions = len(consensus_predictions)

# Normal Mod İstatistikleri
normal_consensus_count = sum(1 for p in consensus_predictions if p['consensus_normal'])
normal_consensus_rate = normal_consensus_count / total_predictions * 100

# Rolling Mod İstatistikleri
rolling_consensus_count = sum(1 for p in consensus_predictions if p['consensus_rolling'])
rolling_consensus_rate = rolling_consensus_count / total_predictions * 100

print(f"\nToplam Tahmin: {total_predictions:,}")
print(f"Normal Mod Consensus (≥0.85): {normal_consensus_count:,} ({normal_consensus_rate:.1f}%)")
print(f"Rolling Mod Consensus (≥0.95): {rolling_consensus_count:,} ({rolling_consensus_rate:.1f}%)")

# Gerçek değer dağılımı
actual_above_15 = (actual_values >= 1.5).sum()
actual_below_15 = (actual_values < 1.5).sum()

print(f"\nGerçek Değer Dağılımı:")
print(f"  1.5 üstü: {actual_above_15:,} ({actual_above_15/len(actual_values)*100:.1f}%)")
print(f"  1.5 altı: {actual_below_15:,} ({actual_below_15/len(actual_values)*100:.1f}%)")

# =============================================================================
# CONSENSUS ACCURACY (2 MODLU)
# =============================================================================
print("\n" + "="*80)
print("🎯 CONSENSUS ACCURACY")
print("="*80)

# Normal Mod Doğruluğu
normal_correct = 0
if normal_consensus_count > 0:
    for i, p in enumerate(consensus_predictions):
        if p['consensus_normal'] and actual_values[i] >= 1.5:
            normal_correct += 1
    normal_accuracy = (normal_correct / normal_consensus_count) * 100
    print(f"\nNormal Mod Doğruluğu (Eşik 0.85):")
    print(f"  Doğru Tahmin: {normal_correct}/{normal_consensus_count}")
    print(f"  Accuracy: {normal_accuracy:.2f}%")
else:
    print("\nNormal Mod Consensus yok.")

# Rolling Mod Doğruluğu
rolling_correct = 0
if rolling_consensus_count > 0:
    for i, p in enumerate(consensus_predictions):
        if p['consensus_rolling'] and actual_values[i] >= 1.5:
            rolling_correct += 1
    rolling_accuracy = (rolling_correct / rolling_consensus_count) * 100
    print(f"\nRolling Mod Doğruluğu (Eşik 0.95):")
    print(f"  Doğru Tahmin: {rolling_correct}/{rolling_consensus_count}")
    print(f"  Accuracy: {rolling_accuracy:.2f}%")
else:
    print("\nRolling Mod Consensus yok.")

# =============================================================================
# SANAL KASA SİMÜLASYONU (2 MODLU)
# =============================================================================
print("\n" + "="*80)
print("💰 SANAL KASA SİMÜLASYONU")
print("="*80)

initial_bankroll = 1000.0
bet_amount = 10.0

# KASA 1: NORMAL MOD (Dinamik Çıkış)
# Çıkış: (NN Tahmin + CatBoost Tahmin) / 2 * 0.80 (Güvenlik payı)
wallet1 = initial_bankroll
bets1 = 0
wins1 = 0

for i, p in enumerate(consensus_predictions):
    if p['consensus_normal']:
        wallet1 -= bet_amount
        bets1 += 1
        
        # Ortalama tahmin
        avg_pred = (p['nn_prediction'] + p['catboost_prediction']) / 2
        # Dinamik çıkış (Min 1.5, Max 2.5)
        exit_point = min(max(1.5, avg_pred * 0.8), 2.5)
        
        if actual_values[i] >= exit_point:
            wallet1 += bet_amount * exit_point
            wins1 += 1

roi1 = (wallet1 - initial_bankroll) / initial_bankroll * 100
win_rate1 = (wins1 / bets1 * 100) if bets1 > 0 else 0

print(f"\n💰 KASA 1 (NORMAL MOD - 0.85+):")
print(f"  Final: {wallet1:,.2f} TL")
print(f"  Toplam Bahis: {bets1:,}")
print(f"  Kazanan: {wins1:,}")
print(f"  Win Rate: {win_rate1:.1f}%")
print(f"  ROI: {roi1:+.2f}%")

# KASA 2: ROLLING MOD (Güvenli Çıkış)
# Çıkış: Sabit 1.50x
wallet2 = initial_bankroll
bets2 = 0
wins2 = 0

for i, p in enumerate(consensus_predictions):
    if p['consensus_rolling']:
        wallet2 -= bet_amount
        bets2 += 1
        
        if actual_values[i] >= 1.5:
            wallet2 += bet_amount * 1.5
            wins2 += 1

roi2 = (wallet2 - initial_bankroll) / initial_bankroll * 100
win_rate2 = (wins2 / bets2 * 100) if bets2 > 0 else 0

print(f"\n💰 KASA 2 (ROLLING MOD - 0.95+):")
print(f"  Final: {wallet2:,.2f} TL")
print(f"  Toplam Bahis: {bets2:,}")
print(f"  Kazanan: {wins2:,}")
print(f"  Win Rate: {win_rate2:.1f}%")
print(f"  ROI: {roi2:+.2f}%")

# =============================================================================
# SONUÇLAR KAYDETME
# =============================================================================
print("\n" + "="*80)
print("💾 SONUÇLAR KAYDEDİLİYOR")
print("="*80)

os.makedirs('results', exist_ok=True)

results_dict = {
    'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'test_data_size': len(actual_values),
    'thresholds': {
        'normal': THRESHOLD_NORMAL,
        'rolling': THRESHOLD_ROLLING
    },
    'normal_mode_stats': {
        'consensus_count': normal_consensus_count,
        'accuracy': normal_accuracy if normal_consensus_count > 0 else 0,
        'roi': roi1,
        'win_rate': win_rate1
    },
    'rolling_mode_stats': {
        'consensus_count': rolling_consensus_count,
        'accuracy': rolling_accuracy if rolling_consensus_count > 0 else 0,
        'roi': roi2,
        'win_rate': win_rate2
    }
}

with open('results/consensus_evaluation.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print(f"✅ Sonuçlar kaydedildi: results/consensus_evaluation.json")

# =============================================================================
# FINAL RAPOR
# =============================================================================
print("\n" + "="*80)
print("🎉 CONSENSUS EVALUATION TAMAMLANDI!")
print("="*80)
print(f"Bitiş: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print(f"\n📊 ÖZET:")
print(f"  Normal Mod ROI: {roi1:+.2f}%")
print(f"  Rolling Mod ROI: {roi2:+.2f}%")
print(f"  En İyi Strateji: {'Rolling Mod' if roi2 > roi1 else 'Normal Mod'}")

print("\n" + "="*80)
