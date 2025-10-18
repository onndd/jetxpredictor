#!/usr/bin/env python3
"""
🎯 CONSENSUS MODEL EVALUATION

Bu notebook, Progressive NN ve CatBoost modellerinin consensus tahminlerini
test eder ve iki farklı sanal kasa stratejisini değerlendirir.

Consensus Mantığı:
- Her iki model de 1.5 üstü tahmin ediyorsa → OYNA
- Aksi durumda → OYNAMA

İki Sanal Kasa Stratejisi:
- Kasa 1: 1.5x eşikte çık
- Kasa 2: İki modelin regression tahminlerinin ortalamasının %70'inde çık

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
print("🎯 CONSENSUS MODEL EVALUATION")
print("="*80)
print(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# =============================================================================
# KÜTÜPHANE YÜ YÜKLEME
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

from utils.consensus_predictor import ConsensusPredictor, simulate_consensus_bankroll
from utils.multi_scale_window import split_data_preserving_order
print(f"✅ Consensus modülü yüklendi\n")

# Google Drive mount (Colab için)
try:
    from google.colab import drive
    
    if not os.path.exists('/content/drive'):
        print("\n📦 Google Drive bağlanıyor...")
        drive.mount('/content/drive')
    
    # Model kayıt dizini
    DRIVE_MODEL_DIR = '/content/drive/MyDrive/JetX_Models/Consensus/'
    os.makedirs(DRIVE_MODEL_DIR, exist_ok=True)
    print(f"✅ Google Drive bağlandı: {DRIVE_MODEL_DIR}")
    USE_DRIVE = True
except ImportError:
    print("⚠️ Google Colab dışında - lokal kayıt kullanılacak")
    DRIVE_MODEL_DIR = 'results/'
    USE_DRIVE = False
except Exception as e:
    print(f"⚠️ Google Drive mount hatası: {e}")
    DRIVE_MODEL_DIR = 'results/'
    USE_DRIVE = False

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

consensus = ConsensusPredictor(
    nn_model_dir='models/progressive_multiscale',
    catboost_model_dir='models/catboost_multiscale',
    window_sizes=[500, 250, 100, 50, 20]
)

# NN modellerini yükle
try:
    consensus.load_nn_models()
    print("✅ NN modelleri yüklendi")
except Exception as e:
    print(f"⚠️  NN modelleri yüklenemedi: {e}")
    print("NN modellerini eğitmek için jetx_PROGRESSIVE_TRAINING_MULTISCALE.py'yi çalıştırın")

# CatBoost modellerini yükle
try:
    consensus.load_catboost_models()
    print("✅ CatBoost modelleri yüklendi")
except Exception as e:
    print(f"⚠️  CatBoost modelleri yüklenemedi: {e}")
    print("CatBoost modellerini eğitmek için jetx_CATBOOST_TRAINING_MULTISCALE.py'yi çalıştırın")

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
        # Consensus tahmin
        prediction = consensus.predict_consensus(history)
        
        consensus_predictions.append(prediction)
        actual_values.append(actual)
    except Exception as e:
        print(f"\n⚠️  Tahmin hatası (i={i}): {e}")
        continue

actual_values = np.array(actual_values)

print(f"\n✅ {len(consensus_predictions)} tahmin tamamlandı")

# =============================================================================
# CONSENSUS ANALİZİ
# =============================================================================
print("\n" + "="*80)
print("📊 CONSENSUS ANALİZİ")
print("="*80)

# Consensus istatistikleri
total_predictions = len(consensus_predictions)
consensus_count = sum(1 for p in consensus_predictions if p['consensus'])
no_consensus_count = total_predictions - consensus_count

print(f"\nToplam Tahmin: {total_predictions:,}")
print(f"Consensus VAR: {consensus_count:,} ({consensus_count/total_predictions*100:.1f}%)")
print(f"Consensus YOK: {no_consensus_count:,} ({no_consensus_count/total_predictions*100:.1f}%)")

# Model anlaşma analizi
nn_says_play = sum(1 for p in consensus_predictions if p['nn_threshold'] == 1)
catboost_says_play = sum(1 for p in consensus_predictions if p['catboost_threshold'] == 1)

print(f"\nModel Tahminleri:")
print(f"  NN '1.5 üstü' dedi: {nn_says_play:,} ({nn_says_play/total_predictions*100:.1f}%)")
print(f"  CatBoost '1.5 üstü' dedi: {catboost_says_play:,} ({catboost_says_play/total_predictions*100:.1f}%)")

# Gerçek değer dağılımı
actual_above_15 = (actual_values >= 1.5).sum()
actual_below_15 = (actual_values < 1.5).sum()

print(f"\nGerçek Değer Dağılımı:")
print(f"  1.5 üstü: {actual_above_15:,} ({actual_above_15/len(actual_values)*100:.1f}%)")
print(f"  1.5 altı: {actual_below_15:,} ({actual_below_15/len(actual_values)*100:.1f}%)")

# =============================================================================
# CONSENSUS ACCURACY
# =============================================================================
print("\n" + "="*80)
print("🎯 CONSENSUS ACCURACY")
print("="*80)

# Sadece consensus olan tahminleri filtrele
consensus_only_predictions = [p for p in consensus_predictions if p['consensus']]
consensus_only_actuals = actual_values[[i for i, p in enumerate(consensus_predictions) if p['consensus']]]

if len(consensus_only_predictions) > 0:
    # İki model de 1.5 üstü dediğinde gerçekten 1.5 üstü mü?
    correct_consensus = sum(1 for p, a in zip(consensus_only_predictions, consensus_only_actuals) if a >= 1.5)
    consensus_accuracy = (correct_consensus / len(consensus_only_predictions)) * 100
    
    print(f"\nConsensus Doğruluğu:")
    print(f"  Consensus olduğunda doğru tahmin: {correct_consensus}/{len(consensus_only_predictions)}")
    print(f"  Consensus Accuracy: {consensus_accuracy:.2f}%")
else:
    print("\n⚠️  Hiç consensus tahmini yok!")

# =============================================================================
# SANAL KASA SİMÜLASYONU
# =============================================================================
print("\n" + "="*80)
print("💰 SANAL KASA SİMÜLASYONU")
print("="*80)

results = simulate_consensus_bankroll(
    predictions=consensus_predictions,
    actuals=actual_values,
    bet_amount=10.0
)

kasa1 = results['kasa_1']
kasa2 = results['kasa_2']

print(f"\n💰 KASA 1 (1.5x EŞİK):")
print(f"  Başlangıç: {kasa1['initial']:,.2f} TL")
print(f"  Final: {kasa1['final']:,.2f} TL")
print(f"  Toplam Bahis: {kasa1['total_bets']:,}")
print(f"  Kazanan: {kasa1['total_wins']:,}")
print(f"  Win Rate: {kasa1['win_rate']:.1f}%")
print(f"  Net Kar/Zarar: {kasa1['profit']:+,.2f} TL")
print(f"  ROI: {kasa1['roi']:+.2f}%")

print(f"\n💰 KASA 2 (%70 ÇIKIŞ):")
print(f"  Başlangıç: {kasa2['initial']:,.2f} TL")
print(f"  Final: {kasa2['final']:,.2f} TL")
print(f"  Toplam Bahis: {kasa2['total_bets']:,}")
print(f"  Kazanan: {kasa2['total_wins']:,}")
print(f"  Win Rate: {kasa2['win_rate']:.1f}%")
print(f"  Net Kar/Zarar: {kasa2['profit']:+,.2f} TL")
print(f"  ROI: {kasa2['roi']:+.2f}%")

# Karşılaştırma
roi_diff = kasa2['roi'] - kasa1['roi']
wr_diff = kasa2['win_rate'] - kasa1['win_rate']

print(f"\n📊 KARŞILAŞTIRMA:")
print(f"  Kasa 2 vs Kasa 1:")
print(f"    ROI Farkı: {roi_diff:+.2f}%")
print(f"    Win Rate Farkı: {wr_diff:+.1f}%")

if kasa2['roi'] > kasa1['roi']:
    print(f"  ✅ Kasa 2 (%70 çıkış) daha iyi performans gösterdi!")
else:
    print(f"  ✅ Kasa 1 (1.5x eşik) daha iyi performans gösterdi!")

# =============================================================================
# DETAYLI ANALİZ
# =============================================================================
print("\n" + "="*80)
print("🔬 DETAYLI ANALİZ")
print("="*80)

# Ortalama tahminler
if consensus_only_predictions:
    avg_nn_pred = np.mean([p['nn_prediction'] for p in consensus_only_predictions])
    avg_catboost_pred = np.mean([p['catboost_prediction'] for p in consensus_only_predictions])
    avg_consensus_pred = np.mean([p['average_prediction'] for p in consensus_only_predictions])
    avg_exit_70 = np.mean([p['exit_point_70'] for p in consensus_only_predictions])
    avg_actual = np.mean(consensus_only_actuals)
    
    print(f"\nOrtalama Tahminler (Consensus olduğunda):")
    print(f"  NN Tahmini: {avg_nn_pred:.3f}x")
    print(f"  CatBoost Tahmini: {avg_catboost_pred:.3f}x")
    print(f"  Consensus Tahmini: {avg_consensus_pred:.3f}x")
    print(f"  %70 Çıkış Noktası: {avg_exit_70:.3f}x")
    print(f"  Gerçek Değer: {avg_actual:.3f}x")

# Kaçırılan fırsatlar
missed_opportunities = []
for i, (p, a) in enumerate(zip(consensus_predictions, actual_values)):
    if not p['consensus'] and a >= 1.5:
        missed_opportunities.append((i, a))

print(f"\nKaçırılan Fırsatlar:")
print(f"  Consensus yoktu ama 1.5 üstü çıktı: {len(missed_opportunities)}")
if len(missed_opportunities) > 0:
    missed_avg = np.mean([a for _, a in missed_opportunities])
    print(f"  Ortalama değer: {missed_avg:.3f}x")

# Yanlış consensus
false_consensus = []
for i, (p, a) in enumerate(zip(consensus_predictions, actual_values)):
    if p['consensus'] and a < 1.5:
        false_consensus.append((i, a))

print(f"\nYanlış Consensus (Para Kaybı):")
print(f"  Consensus vardı ama 1.5 altı çıktı: {len(false_consensus)}")
if len(false_consensus) > 0:
    false_avg = np.mean([a for _, a in false_consensus])
    print(f"  Ortalama değer: {false_avg:.3f}x")

# =============================================================================
# SONUÇLAR KAYDETME
# =============================================================================
print("\n" + "="*80)
print("💾 SONUÇLAR KAYDEDİLİYOR")
print("="*80)

os.makedirs(DRIVE_MODEL_DIR, exist_ok=True)

results_dict = {
    'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'test_data_size': len(actual_values),
    'consensus_stats': {
        'total_predictions': total_predictions,
        'consensus_count': consensus_count,
        'consensus_rate': consensus_count / total_predictions,
        'consensus_accuracy': consensus_accuracy if len(consensus_only_predictions) > 0 else 0
    },
    'model_agreement': {
        'nn_says_play': nn_says_play,
        'catboost_says_play': catboost_says_play,
        'nn_play_rate': nn_says_play / total_predictions,
        'catboost_play_rate': catboost_says_play / total_predictions
    },
    'kasa_1_15x': {
        'roi': kasa1['roi'],
        'win_rate': kasa1['win_rate'],
        'total_bets': kasa1['total_bets'],
        'total_wins': kasa1['total_wins'],
        'profit': kasa1['profit']
    },
    'kasa_2_70pct': {
        'roi': kasa2['roi'],
        'win_rate': kasa2['win_rate'],
        'total_bets': kasa2['total_bets'],
        'total_wins': kasa2['total_wins'],
        'profit': kasa2['profit']
    },
    'missed_opportunities': len(missed_opportunities),
    'false_consensus': len(false_consensus)
}

with open(f'{DRIVE_MODEL_DIR}consensus_evaluation.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print(f"✅ Sonuçlar kaydedildi: {DRIVE_MODEL_DIR}consensus_evaluation.json")

# =============================================================================
# FINAL RAPOR
# =============================================================================
print("\n" + "="*80)
print("🎉 CONSENSUS EVALUATION TAMAMLANDI!")
print("="*80)
print(f"Bitiş: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print(f"\n📊 ÖZET:")
print(f"  Consensus Rate: {consensus_count/total_predictions*100:.1f}%")
print(f"  Consensus Accuracy: {consensus_accuracy:.1f}%" if len(consensus_only_predictions) > 0 else "  Consensus Accuracy: N/A")
print(f"  Kasa 1 ROI: {kasa1['roi']:+.2f}%")
print(f"  Kasa 2 ROI: {kasa2['roi']:+.2f}%")
print(f"  En İyi Strateji: {'Kasa 2 (%70 çıkış)' if kasa2['roi'] > kasa1['roi'] else 'Kasa 1 (1.5x eşik)'}")

print("\n" + "="*80)
