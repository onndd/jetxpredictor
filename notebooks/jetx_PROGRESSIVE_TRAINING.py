#!/usr/bin/env python3
"""
🎯 JetX PROGRESSIVE TRAINING - 3 Aşamalı Eğitim Stratejisi (v5.3 FIXED)

BU DOSYA TEK BAŞINA ÇALIŞIR (STANDALONE).
Tüm yardımcı sınıflar, loss fonksiyonları ve katmanlar içine gömülmüştür.

MİMARİ:
- Inputs: Features + 4 Sequence (50, 200, 500, 1000)
- Layers: N-Beats + TCN + Transformer Encoder + Fusion
- Outputs: Regression (Değer), Classification (3 Sınıf), Threshold (Binary)

GÜNCELLEME (v5.3):
- ✅ PATH FIX: Proje ana dizini otomatik tespit edilir (ModuleNotFoundError çözümü).
- ✅ DATA CLEANING: Veritabanı okuma sırasında sıkı temizlik (ValueError çözümü).
- ✅ Class Weight Düzeltmesi: 2.0x
- ✅ 2 MODLU YAPI: Normal (0.85) ve Rolling (0.95)
"""

import sys
import os
from pathlib import Path

# --- KRİTİK PATH DÜZELTMESİ ---
# Bu scriptin bulunduğu klasörün bir üstünü (proje kök dizini) sys.path'e ekle
# Böylece 'utils' ve 'category_definitions' modülleri sorunsuz bulunur.
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"🔧 Python Path Eklendi: {project_root}")

import time
from datetime import datetime
import json
import shutil
import pickle
import warnings
import math
import random
import subprocess

# Uyarıları kapat
warnings.filterwarnings('ignore')

print("="*80)
print("🎯 JetX PROGRESSIVE TRAINING - 3 Aşamalı Eğitim (v5.3 FIXED)")
print("="*80)
print(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# -----------------------------------------------------------------------------
# 1. KÜTÜPHANE KURULUMU VE İMPORTLAR
# -----------------------------------------------------------------------------
print("📦 Kütüphaneler kontrol ediliyor...")
required_packages = [
    "tensorflow", "scikit-learn", "pandas", "numpy", 
    "scipy", "joblib", "matplotlib", "seaborn", "tqdm"
]

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"   ⬇️ {package} kuruluyor...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

import numpy as np
import pandas as pd
import joblib
import sqlite3
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix, classification_report
from tqdm.auto import tqdm

# GPU Ayarları
print("\n🚀 GPU Ayarları Yapılandırılıyor...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        print(f"✅ GPU Aktif: {len(gpus)} adet (Mixed Precision)")
    except RuntimeError as e:
        print(f"⚠️ GPU Hatası: {e}")
else:
    print("⚠️ GPU Bulunamadı! CPU modunda devam ediliyor.")

# Kritik Eşikler
THRESHOLD_NORMAL = 0.85
THRESHOLD_ROLLING = 0.95

# =============================================================================
# 2. GÖMÜLÜ YARDIMCI MODÜLLER
# =============================================================================

# --- A. FEATURE ENGINEERING & DEFINITIONS ---
class CategoryDefinitions:
    """Kategori tanımları"""
    CRITICAL_THRESHOLD = 1.5
    
    @staticmethod
    def get_category_numeric(val): 
        if val < 1.5: return 0
        elif val < 2.0: return 1
        else: return 2

class FeatureEngineering:
    """Özellik çıkarma fonksiyonları"""
    
    @staticmethod
    def extract_all_features(history: list) -> dict:
        features = {}
        if not history:
            return features
            
        # Temel istatistikler
        features['mean_50'] = np.mean(history[-50:]) if len(history) >= 50 else np.mean(history)
        features['std_50'] = np.std(history[-50:]) if len(history) >= 50 else np.std(history)
        features['min_50'] = np.min(history[-50:]) if len(history) >= 50 else np.min(history)
        features['max_50'] = np.max(history[-50:]) if len(history) >= 50 else np.max(history)
        
        # Threshold özellikleri
        recent_10 = history[-10:] if len(history) >= 10 else history
        features['below_threshold_10'] = sum(1 for x in recent_10 if x < 1.5)
        features['above_threshold_10'] = sum(1 for x in recent_10 if x >= 1.5)
        
        # Volatilite
        if len(history) >= 20:
            recent_20 = history[-20:]
            features['volatility_20'] = np.std(recent_20) / (np.mean(recent_20) + 1e-8)
        else:
            features['volatility_20'] = 0.0
        
        # Son değerler
        features['last_val'] = history[-1]
        features['diff_1'] = history[-1] - history[-2] if len(history) > 1 else 0
        
        return features

# --- B. CUSTOM LOSS FUNCTIONS ---
def percentage_aware_regression_loss(y_true, y_pred):
    """Yüzde hataya dayalı regression loss"""
    epsilon = K.epsilon()
    percentage_error = K.abs(y_true - y_pred) / (K.abs(y_true) + epsilon)
    high_value_weight = tf.where(y_true >= 5.0, 1.2, 1.0)
    weighted_percentage_error = percentage_error * high_value_weight
    return K.mean(weighted_percentage_error)

def balanced_focal_loss(gamma=2.0, alpha=0.7):
    """Dengeli Focal Loss"""
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = alpha * K.pow(1 - pt, gamma)
        return -K.mean(focal_weight * K.log(pt))
    return loss

def create_weighted_binary_crossentropy(weight_0, weight_1):
    """Ağırlıklı Binary Crossentropy"""
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        bce = -(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        weights = y_true * weight_1 + (1 - y_true) * weight_0
        return K.mean(bce * weights)
    return loss

# --- C. ATTENTION & TRANSFORMER LAYERS ---
class PositionalEncoding(layers.Layer):
    """Transformer için Positional Encoding"""
    def __init__(self, max_seq_len=1000, d_model=256, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.pe = None
        
    def build(self, input_shape):
        position = tf.range(self.max_seq_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, self.d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / self.d_model))
        pe_sin = tf.sin(position * div_term)
        pe_cos = tf.cos(position * div_term)
        
        pe_list = []
        for i in range(self.d_model):
            if i % 2 == 0:
                pe_list.append(pe_sin[:, i // 2:i // 2 + 1])
            else:
                pe_list.append(pe_cos[:, i // 2:i // 2 + 1])
        
        pe = tf.concat(pe_list, axis=1)
        self.pe = tf.constant(pe, dtype=tf.float32)
        super().build(input_shape)
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:seq_len, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({'max_seq_len': self.max_seq_len, 'd_model': self.d_model})
        return config

class LightweightTransformerEncoder(layers.Layer):
    """Hafif Transformer Encoder Bloğu"""
    def __init__(self, d_model=256, num_layers=4, num_heads=8, dff=1024, dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout
        
        self.input_projection = layers.Dense(d_model)
        self.pos_encoding = PositionalEncoding(max_seq_len=1000, d_model=d_model)
        
        self.encoder_layers = []
        for _ in range(num_layers):
            self.encoder_layers.append({
                'mha': layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout),
                'ffn': tf.keras.Sequential([
                    layers.Dense(dff, activation='relu'),
                    layers.Dropout(dropout),
                    layers.Dense(d_model)
                ]),
                'layernorm1': layers.LayerNormalization(epsilon=1e-6),
                'layernorm2': layers.LayerNormalization(epsilon=1e-6),
                'dropout1': layers.Dropout(dropout),
                'dropout2': layers.Dropout(dropout)
            })
        
        self.global_pool = layers.GlobalAveragePooling1D()
        self.output_projection = layers.Dense(d_model)
        self.dropout_final = layers.Dropout(dropout)
    
    def call(self, inputs, training=None):
        x = self.input_projection(inputs)
        x = self.pos_encoding(x)
        
        for layer in self.encoder_layers:
            attn_output = layer['mha'](query=x, key=x, value=x, training=training)
            attn_output = layer['dropout1'](attn_output, training=training)
            x = layer['layernorm1'](x + attn_output)
            
            ffn_output = layer['ffn'](x)
            ffn_output = layer['dropout2'](ffn_output, training=training)
            x = layer['layernorm2'](x + ffn_output)
        
        x = self.global_pool(x)
        x = self.output_projection(x)
        x = self.dropout_final(x, training=training)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model, 'num_layers': self.num_layers,
            'num_heads': self.num_heads, 'dff': self.dff, 'dropout': self.dropout_rate
        })
        return config

# --- D. CUSTOM CALLBACKS ---
class AdaptiveLearningRateScheduler:
    """Model performansına göre learning rate'i adapte eden scheduler"""
    def __init__(self, initial_lr=0.001, max_lr=0.01, min_lr=0.0001, patience=5, factor=0.5):
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.patience = patience
        self.factor = factor
        self.best_score = float('-inf')
        self.patience_counter = 0
        self.current_lr = initial_lr

    def __call__(self, epoch, logs=None):
        # Loss azaldıkça score artar (-loss)
        current_score = -logs.get('val_loss', 0)
        
        if current_score > self.best_score:
            self.best_score = current_score
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.patience:
            self.current_lr = max(self.current_lr * self.factor, self.min_lr)
            self.patience_counter = 0
            print(f"\n📉 LR Azaltıldı: {self.current_lr:.6f}")
        
        return self.current_lr

class DynamicWeightCallback(callbacks.Callback):
    """Eğitim sırasında class weight'i otomatik ayarlayan callback"""
    def __init__(self, validation_data, initial_weight=2.0):
        super().__init__()
        self.validation_data = validation_data
        self.current_weight = initial_weight
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 != 0: return
        
        try:
            # Validation setini çöz
            X_val_data, y_val_data = self.validation_data
            y_thr_val = y_val_data.get('threshold') if isinstance(y_val_data, dict) else y_val_data
            
            # Tahmin
            preds = self.model.predict(X_val_data, verbose=0)
            p = preds[2].flatten() # Threshold output
            t = y_thr_val.flatten()
            
            p_cls = (p >= THRESHOLD_NORMAL).astype(int)
            t_cls = (t >= 1.5).astype(int)
            
            # 1.5 Altı doğruluğu
            mask_below = t_cls == 0
            if mask_below.sum() > 0:
                below_acc = accuracy_score(t_cls[mask_below], p_cls[mask_below])
            else:
                below_acc = 0
            
            # Ayarlama (Çok agresif artırmıyoruz)
            old_weight = self.current_weight
            if below_acc < 0.50: self.current_weight *= 1.1
            elif below_acc > 0.80: self.current_weight *= 0.95
            
            # Limitler (1.0 - 10.0 arası) - Lazy learning önlemek için max düşürüldü
            self.current_weight = max(1.0, min(10.0, self.current_weight))
            
            print(f"\n⚖️  Epoch {epoch}: Class Weight {old_weight:.2f} -> {self.current_weight:.2f} (1.5 Altı Acc: {below_acc:.2%})")
            
        except Exception as e:
            print(f"⚠️ DynamicWeightCallback hatası: {e}")

class ProgressiveMetricsCallback(callbacks.Callback):
    """2 Modlu (Normal/Rolling) Performans Raporu"""
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 != 0: return
        
        try:
            X_val_data, y_val_data = self.validation_data
            y_reg_val = y_val_data.get('regression') if isinstance(y_val_data, dict) else y_val_data
            
            preds = self.model.predict(X_val_data, verbose=0)
            p = preds[2].flatten()
            t = (y_reg_val >= 1.5).astype(int)
            
            # Normal Mod (0.85)
            p_norm = (p >= THRESHOLD_NORMAL).astype(int)
            acc_norm = accuracy_score(t, p_norm)
            
            # Rolling Mod (0.95)
            p_roll = (p >= THRESHOLD_ROLLING).astype(int)
            acc_roll = accuracy_score(t, p_roll)
            
            print(f"\n📊 Epoch {epoch+1} Metrics:")
            print(f"   🎯 Normal Mod ({THRESHOLD_NORMAL}): {acc_norm:.2%}")
            print(f"   🚀 Rolling Mod ({THRESHOLD_ROLLING}): {acc_roll:.2%}")
        except Exception as e:
            print(f"⚠️ Metrics Callback Hatası: {e}")

class VirtualBankrollCallback(callbacks.Callback):
    """Her epoch'ta ÇİFT KASA (Normal + Rolling) simülasyonu"""
    def __init__(self, stage_name, validation_data, starting_capital=1000.0, bet_amount=10.0):
        super().__init__()
        self.stage_name = stage_name
        self.validation_data = validation_data
        self.starting_capital = starting_capital
        self.bet_amount = bet_amount
        self.best_roi_normal = -float('inf')
        self.best_roi_rolling = -float('inf')
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 != 0: return
        
        try:
            X_val_data, y_val_data = self.validation_data
            y_reg_val = y_val_data.get('regression') if isinstance(y_val_data, dict) else y_val_data
            
            preds = self.model.predict(X_val_data, verbose=0)
            p_thr = preds[2].flatten()
            p_reg = preds[0].flatten()
            actuals = y_reg_val.flatten()
            
            # Kasa 1: Normal Mod (0.85) + Dinamik Çıkış
            wallet1 = self.starting_capital
            bets1 = 0
            
            # Kasa 2: Rolling Mod (0.95) + Sabit 1.5x Çıkış
            wallet2 = self.starting_capital
            bets2 = 0
            
            for i in range(len(p_thr)):
                # --- NORMAL MOD ---
                if p_thr[i] >= THRESHOLD_NORMAL:
                    wallet1 -= self.bet_amount
                    bets1 += 1
                    # Çıkış noktası: Dinamik (Model tahmininin %80'i, max 2.5x)
                    exit_pt = min(max(1.5, p_reg[i] * 0.8), 2.5)
                    
                    if actuals[i] >= exit_pt:
                        wallet1 += self.bet_amount * exit_pt
                
                # --- ROLLING MOD ---
                if p_thr[i] >= THRESHOLD_ROLLING:
                    wallet2 -= self.bet_amount
                    bets2 += 1
                    # Çıkış noktası: Sabit 1.50x (Güvenli Liman)
                    if actuals[i] >= 1.5:
                        wallet2 += self.bet_amount * 1.5
            
            roi1 = (wallet1 - self.starting_capital) / self.starting_capital * 100
            roi2 = (wallet2 - self.starting_capital) / self.starting_capital * 100
            
            if roi1 > self.best_roi_normal: self.best_roi_normal = roi1
            if roi2 > self.best_roi_rolling: self.best_roi_rolling = roi2
            
            # Ekrana iki kasayı da yaz
            print(f"\n💰 {self.stage_name} CANLI KASA DURUMU:")
            print(f"   🎯 Kasa 1 (Normal): ROI {roi1:+.2f}% (Best: {self.best_roi_normal:+.2f}%) | Bets: {bets1}")
            print(f"   🛡️ Kasa 2 (Rolling): ROI {roi2:+.2f}% (Best: {self.best_roi_rolling:+.2f}%) | Bets: {bets2}")
            
        except Exception as e:
            print(f"⚠️ Bankroll Callback Hatası: {e}")

class WeightedModelCheckpoint(callbacks.Callback):
    """
    AKILLI MODEL SEÇİMİ:
    - ROI (Normal): %40
    - Rolling Acc: %30
    - Precision: %20
    - Win Rate: %10
    """
    def __init__(self, filepath, validation_data):
        super().__init__()
        self.filepath = filepath
        self.validation_data = validation_data
        self.best_score = -float('inf')
    
    def normalize_roi(self, roi):
        if roi < 0:
            return max(0, 40 + roi * 0.4)
        else:
            return min(100, 50 + roi * 0.5)
    
    def on_epoch_end(self, epoch, logs=None):
        try:
            X_val_data, y_val_data = self.validation_data
            y_reg_val = y_val_data.get('regression') if isinstance(y_val_data, dict) else y_val_data
            
            preds = self.model.predict(X_val_data, verbose=0)
            # Threshold output genellikle 3. output (index 2)
            if isinstance(preds, list) and len(preds) >= 3:
                threshold_preds = preds[2].flatten()
            else:
                return 
            
            y_true = (y_reg_val.flatten() >= 1.5).astype(int)
            
            # --- Normal Mod Metrikleri ---
            y_pred_norm = (threshold_preds >= THRESHOLD_NORMAL).astype(int)
            FP = np.sum((y_true == 0) & (y_pred_norm == 1))
            TP = np.sum((y_true == 1) & (y_pred_norm == 1))
            precision = (TP / (TP + FP) * 100) if (TP + FP) > 0 else 0
            
            # ROI Hesapla (Normal Mod)
            initial = 10000
            wallet = initial
            total_bets = 0
            wins = 0
            
            for pred, actual in zip(threshold_preds, y_reg_val.flatten()):
                if pred >= THRESHOLD_NORMAL:
                    total_bets += 1
                    wallet -= 10
                    if actual >= 1.5:
                        wallet += 15
                        wins += 1
            
            roi = ((wallet - initial) / initial) * 100 if total_bets > 0 else 0
            win_rate = (wins / total_bets) * 100 if total_bets > 0 else 0
            normalized_roi = self.normalize_roi(roi)
            
            # --- Rolling Mod Metrikleri ---
            y_pred_roll = (threshold_preds >= THRESHOLD_ROLLING).astype(int)
            roll_mask = y_pred_roll == 1
            if roll_mask.sum() > 0:
                roll_acc = accuracy_score(y_true[roll_mask], y_pred_roll[roll_mask]) * 100
            else:
                roll_acc = 0
            
            # GÜNCELLENMİŞ SKORLAMA
            weighted_score = (
                0.40 * normalized_roi +
                0.30 * roll_acc +
                0.20 * precision +
                0.10 * win_rate
            )
            
            if weighted_score > self.best_score:
                self.best_score = weighted_score
                self.model.save(self.filepath)
                print(f"\n✨ YENİ EN İYİ MODEL! (Score: {weighted_score:.2f})")
                print(f"   ROI: {roi:.2f}% | Rolling Acc: {roll_acc:.1f}% | FP: {FP}")
        except Exception as e:
            print(f"⚠️ Checkpoint hatası: {e}")

# -----------------------------------------------------------------------------
# 3. VERİ YÜKLEME VE HAZIRLIK
# -----------------------------------------------------------------------------
print("\n📊 Veri yükleniyor...")
# Veritabanı yolunu bul (önce current dir, sonra project root)
db_names = ['jetx_data.db', os.path.join(str(project_root), 'jetx_data.db')]
db_path = None
for name in db_names:
    if os.path.exists(name):
        db_path = name
        break

if not db_path:
    print("⚠️ Veritabanı bulunamadı! Sentetik veri oluşturuluyor...")
    all_values = np.random.lognormal(0.5, 0.8, 5000)
    all_values = np.clip(all_values, 1.0, 100.0)
else:
    print(f"   DB bulundu: {db_path}")
    conn = sqlite3.connect(db_path)
    data = pd.read_sql_query("SELECT value FROM jetx_results ORDER BY id", conn)
    conn.close()
    
    # --- VERİ TEMİZLEME (FIX) ---
    # Bozuk karakterleri (Unicode) temizle ve float'a çevir
    raw_values = data['value'].values
    cleaned_values = []
    for val in raw_values:
        try:
            val_str = str(val).replace('\u2028', '').replace('\u2029', '').strip()
            if ' ' in val_str: val_str = val_str.split()[0]
            cleaned_values.append(float(val_str))
        except:
            continue
    all_values = np.array(cleaned_values)
    # ----------------------------

print(f"✅ {len(all_values):,} veri temizlendi ve yüklendi")
if len(all_values) < 100:
    print("⚠️ Çok az veri var, sentetik veri ekleniyor...")
    synth = np.random.lognormal(0.5, 0.8, 1000)
    synth = np.clip(synth, 1.0, 100.0)
    all_values = np.concatenate([all_values, synth])

print(f"   Kullanılacak Veri: {len(all_values)}")
print(f"   Aralık: {all_values.min():.2f}x - {all_values.max():.2f}x")

# Feature Extraction Loop
print("\n🔧 Feature extraction (Multi-Scale)...")
window_size = 1000 
X_f, X_50, X_200, X_500, X_1000 = [], [], [], [], []
y_reg, y_cls, y_thr = [], [], []

# Hız için limit koyalım (Çok büyük veride yavaşlamasın)
MAX_SAMPLES = 10000
if len(all_values) > MAX_SAMPLES + window_size:
    start_idx = len(all_values) - MAX_SAMPLES
else:
    start_idx = window_size

for i in tqdm(range(start_idx, len(all_values)-1), desc='Features'):
    hist = all_values[:i].tolist()
    target = all_values[i]
    
    # Features (Dahili class ile)
    feats = FeatureEngineering.extract_all_features(hist)
    X_f.append(list(feats.values()))
    
    # Sequences
    X_50.append(all_values[i-50:i])
    X_200.append(all_values[i-200:i])
    X_500.append(all_values[i-500:i])
    X_1000.append(all_values[i-1000:i])
    
    # Targets
    y_reg.append(target)
    
    # Classification (3 Class: <1.5, 1.5-10, >10)
    if target < 1.5: cat = 0
    elif target < 10: cat = 1
    else: cat = 2
    
    onehot = np.zeros(3)
    onehot[cat] = 1
    y_cls.append(onehot)
    
    # Threshold (Binary)
    y_thr.append(1.0 if target >= 1.5 else 0.0)

# Numpy dönüşümü
X_f = np.array(X_f)
X_50 = np.array(X_50).reshape(-1, 50, 1)
X_200 = np.array(X_200).reshape(-1, 200, 1)
X_500 = np.array(X_500).reshape(-1, 500, 1)
X_1000 = np.array(X_1000).reshape(-1, 1000, 1)
y_reg = np.array(y_reg)
y_cls = np.array(y_cls)
y_thr = np.array(y_thr).reshape(-1, 1)

print(f"✅ {len(X_f):,} örnek hazırlandı")

# Normalizasyon
print("\n📊 Normalizasyon...")
scaler = StandardScaler()
X_f = scaler.fit_transform(X_f)
# Sequence'leri log scale yap (stabilite için)
X_50 = np.log10(X_50 + 1e-8)
X_200 = np.log10(X_200 + 1e-8)
X_500 = np.log10(X_500 + 1e-8)
X_1000 = np.log10(X_1000 + 1e-8)

# Kronolojik Split
print("\n📊 TIME-SERIES SPLIT (Kronolojik)...")
test_size = int(len(X_f) * 0.15)
val_size = int(len(X_f) * 0.15)
train_size = len(X_f) - test_size - val_size

# Train
X_f_tr = X_f[:train_size]
X_50_tr = X_50[:train_size]
X_200_tr = X_200[:train_size]
X_500_tr = X_500[:train_size]
X_1000_tr = X_1000[:train_size]
y_reg_tr = y_reg[:train_size]
y_cls_tr = y_cls[:train_size]
y_thr_tr = y_thr[:train_size]

# Validation
X_f_val = X_f[train_size:train_size+val_size]
X_50_val = X_50[train_size:train_size+val_size]
X_200_val = X_200[train_size:train_size+val_size]
X_500_val = X_500[train_size:train_size+val_size]
X_1000_val = X_1000[train_size:train_size+val_size]
y_reg_val = y_reg[train_size:train_size+val_size]
y_cls_val = y_cls[train_size:train_size+val_size]
y_thr_val = y_thr[train_size:train_size+val_size]

# Test
X_f_te = X_f[train_size+val_size:]
X_50_te = X_50[train_size+val_size:]
X_200_te = X_200[train_size+val_size:]
X_500_te = X_500[train_size+val_size:]
X_1000_te = X_1000[train_size+val_size:]
y_reg_te = y_reg[train_size+val_size:]
y_cls_te = y_cls[train_size+val_size:]
y_thr_te = y_thr[train_size+val_size:]

print(f"   Train: {len(X_f_tr):,}")
print(f"   Val:   {len(X_f_val):,}")
print(f"   Test:  {len(X_f_te):,}")

# -----------------------------------------------------------------------------
# 4. MODEL MİMARİSİ OLUŞTURMA FONKSİYONU
# -----------------------------------------------------------------------------
def build_progressive_model(n_features):
    """
    N-Beats + TCN + Transformer Hybrid Mimari
    """
    # Inputs
    inp_f = layers.Input((n_features,), name='features')
    inp_50 = layers.Input((50, 1), name='seq50')
    inp_200 = layers.Input((200, 1), name='seq200')
    inp_500 = layers.Input((500, 1), name='seq500')
    inp_1000 = layers.Input((1000, 1), name='seq1000')
    
    # --- N-BEATS Blokları ---
    def nbeats_block(x, units, blocks):
        for _ in range(blocks):
            x = layers.Dense(units, activation='relu', kernel_regularizer='l2')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        return x

    nb_s = nbeats_block(layers.Flatten()(inp_50), 128, 5)
    nb_m = nbeats_block(layers.Flatten()(inp_200), 192, 6)
    nb_l = nbeats_block(layers.Flatten()(inp_500), 256, 7)
    nb_xl = nbeats_block(layers.Flatten()(inp_1000), 384, 9)
    
    nb_all = layers.Concatenate()([nb_s, nb_m, nb_l, nb_xl])
    
    # --- TCN Bloğu ---
    def tcn_block(x, filters, dilation):
        conv = layers.Conv1D(filters, 3, dilation_rate=dilation, padding='causal', activation='relu')(x)
        conv = layers.BatchNormalization()(conv)
        residual = layers.Conv1D(filters, 1, padding='same')(x) if x.shape[-1] != filters else x
        return layers.Add()([conv, residual])
    
    tcn = inp_500
    for i, dilation in enumerate([1, 2, 4, 8, 16, 32]):
        filters = 128 if i < 3 else 256
        tcn = tcn_block(tcn, filters, dilation)
    tcn = layers.GlobalAveragePooling1D()(tcn)
    
    # --- Transformer Bloğu ---
    transformer = LightweightTransformerEncoder(
        d_model=256, num_layers=4, num_heads=8, dff=1024, dropout=0.2
    )(inp_1000)
    
    # --- Fusion ---
    fus = layers.Concatenate()([inp_f, nb_all, tcn, transformer])
    fus = layers.Dense(512, activation='relu')(fus)
    fus = layers.BatchNormalization()(fus)
    fus = layers.Dropout(0.3)(fus)
    fus = layers.Dense(256, activation='relu')(fus)
    fus = layers.Dropout(0.2)(fus)
    
    # --- Outputs ---
    out_reg = layers.Dense(1, activation='linear', name='regression')(fus)
    out_cls = layers.Dense(3, activation='softmax', name='classification')(fus)
    out_thr = layers.Dense(1, activation='sigmoid', name='threshold')(fus)
    
    return models.Model([inp_f, inp_50, inp_200, inp_500, inp_1000], [out_reg, out_cls, out_thr])

# -----------------------------------------------------------------------------
# 5. YARDIMCI FONKSİYONLAR (CHECKPOINT)
# -----------------------------------------------------------------------------
def save_checkpoint(stage, epoch, model):
    """Checkpoint kaydet"""
    filename = f'checkpoint_stage{stage}.pkl'
    checkpoint = {
        'stage': stage, 'epoch': epoch, 'weights': model.get_weights(),
        'timestamp': datetime.now().isoformat()
    }
    with open(filename, 'wb') as f: pickle.dump(checkpoint, f)
    print(f"💾 Stage {stage} checkpoint kaydedildi.")

def load_checkpoint(stage):
    """Checkpoint yükle"""
    filename = f'checkpoint_stage{stage}.pkl'
    if os.path.exists(filename):
        with open(filename, 'rb') as f: return pickle.load(f)
    return None

# -----------------------------------------------------------------------------
# 6. EĞİTİM AŞAMALARI
# -----------------------------------------------------------------------------
model = build_progressive_model(X_f.shape[1])
print(f"\n🏗️ Model oluşturuldu: {model.count_params():,} parametre")

# Hazırlık: Callback'ler için validation data yapısı
val_data_dict = {
    'regression': y_reg_val,
    'classification': y_cls_val,
    'threshold': y_thr_val
}
val_inputs = [X_f_val, X_50_val, X_200_val, X_500_val, X_1000_val]

# Callback Wrapper for AdaptiveScheduler
class AdaptiveLRCallback(callbacks.Callback):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler
    def on_epoch_end(self, epoch, logs=None):
        if logs is None: logs = {}
        current_lr = self.scheduler(epoch, logs)
        K.set_value(self.model.optimizer.learning_rate, current_lr)

# --- AŞAMA 1: Foundation ---
print("\n" + "="*60)
print("🔥 AŞAMA 1: FOUNDATION TRAINING (100 Epoch)")
print("="*60)

chk1 = load_checkpoint(1)
if chk1: 
    model.set_weights(chk1['weights'])
    print("🔄 AŞAMA 1 Checkpoint yüklendi.")

# Adaptive Scheduler
adaptive_scheduler = AdaptiveLearningRateScheduler(initial_lr=0.001, patience=5)
lr_callback = AdaptiveLRCallback(adaptive_scheduler)

# DÜZELTME: Class Weight 2.0 (Lazy Learning'i önlemek için düşürüldü)
w0 = 2.0 
w1 = 1.0

model.compile(
    optimizer=Adam(0.0001),
    loss={'regression': percentage_aware_regression_loss, 'classification': 'categorical_crossentropy', 'threshold': create_weighted_binary_crossentropy(w0, w1)},
    loss_weights={'regression': 0.65, 'classification': 0.10, 'threshold': 0.25},
    metrics={'threshold': ['accuracy']}
)

hist1 = model.fit(
    [X_f_tr, X_50_tr, X_200_tr, X_500_tr, X_1000_tr],
    {'regression': y_reg_tr, 'classification': y_cls_tr, 'threshold': y_thr_tr},
    epochs=100, batch_size=64, shuffle=False,
    validation_data=(val_inputs, val_data_dict),
    callbacks=[
        DynamicWeightCallback(validation_data=(val_inputs, val_data_dict), initial_weight=w0),
        ProgressiveMetricsCallback(validation_data=(val_inputs, val_data_dict)),
        VirtualBankrollCallback("AŞAMA 1", validation_data=(val_inputs, val_data_dict), starting_capital=1000.0), 
        lr_callback,
        callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ],
    verbose=1
)
save_checkpoint(1, len(hist1.history['loss']), model)

# --- AŞAMA 2: Fine-Tuning ---
print("\n" + "="*60)
print("🔥 AŞAMA 2: THRESHOLD FINE-TUNING (80 Epoch)")
print("="*60)

chk2 = load_checkpoint(2)
if chk2: model.set_weights(chk2['weights'])

# DÜZELTME: İkinci aşamada da düşük weight (2.5x)
w0_stage2 = 2.5

model.compile(
    optimizer=Adam(0.00005), # Daha düşük LR
    loss={'regression': percentage_aware_regression_loss, 'classification': 'categorical_crossentropy', 'threshold': create_weighted_binary_crossentropy(w0_stage2, 1.0)},
    loss_weights={'regression': 0.55, 'classification': 0.10, 'threshold': 0.35},
    metrics={'threshold': ['accuracy']}
)

hist2 = model.fit(
    [X_f_tr, X_50_tr, X_200_tr, X_500_tr, X_1000_tr],
    {'regression': y_reg_tr, 'classification': y_cls_tr, 'threshold': y_thr_tr},
    epochs=80, batch_size=32, shuffle=False,
    validation_data=(val_inputs, val_data_dict),
    callbacks=[
        ProgressiveMetricsCallback(validation_data=(val_inputs, val_data_dict)),
        VirtualBankrollCallback("AŞAMA 2", validation_data=(val_inputs, val_data_dict)),
        callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ],
    verbose=1
)
save_checkpoint(2, len(hist2.history['loss']), model)

# --- AŞAMA 3: Final Polish ---
print("\n" + "="*60)
print("🔥 AŞAMA 3: FULL MODEL FINE-TUNING (80 Epoch)")
print("="*60)

chk3 = load_checkpoint(3)
if chk3: model.set_weights(chk3['weights'])

model.compile(
    optimizer=Adam(0.00001), # En düşük LR
    loss={'regression': percentage_aware_regression_loss, 'classification': 'categorical_crossentropy', 'threshold': balanced_focal_loss(gamma=2.0, alpha=0.7)},
    loss_weights={'regression': 0.50, 'classification': 0.15, 'threshold': 0.35},
    metrics={'threshold': ['accuracy']}
)

# Models dizini oluştur
os.makedirs('models', exist_ok=True)

# Weighted Checkpoint (YENİ: Rolling Acc DAHİL)
checkpoint_callback = WeightedModelCheckpoint(
    filepath='models/jetx_progressive_final.h5',
    validation_data=(val_inputs, val_data_dict)
)

hist3 = model.fit(
    [X_f_tr, X_50_tr, X_200_tr, X_500_tr, X_1000_tr],
    {'regression': y_reg_tr, 'classification': y_cls_tr, 'threshold': y_thr_tr},
    epochs=80, batch_size=16, shuffle=False,
    validation_data=(val_inputs, val_data_dict),
    callbacks=[
        ProgressiveMetricsCallback(validation_data=(val_inputs, val_data_dict)),
        VirtualBankrollCallback("AŞAMA 3", validation_data=(val_inputs, val_data_dict)),
        checkpoint_callback,
        callbacks.EarlyStopping(patience=8, restore_best_weights=True)
    ],
    verbose=1
)
save_checkpoint(3, len(hist3.history['loss']), model)

# -----------------------------------------------------------------------------
# 7. FİNAL DEĞERLENDİRME VE SİMÜLASYON
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("📊 FİNAL DEĞERLENDİRME & KASA SİMÜLASYONU")
print("="*60)

# Modeli yükle (en iyi hali)
if os.path.exists('models/jetx_progressive_final.h5'):
    try:
        model.load_weights('models/jetx_progressive_final.h5')
    except:
        print("⚠️ Ağırlıklar yüklenemedi, mevcut model kullanılıyor.")

# Test verisi üzerinde tahmin
pred = model.predict([X_f_te, X_50_te, X_200_te, X_500_te, X_1000_te], verbose=0)
p_reg = pred[0].flatten()
p_thr = pred[2].flatten()

# Metrikler
mae = mean_absolute_error(y_reg_te, p_reg)
y_true_cls = (y_reg_te >= 1.5).astype(int)
p_norm = (p_thr >= THRESHOLD_NORMAL).astype(int)
p_roll = (p_thr >= THRESHOLD_ROLLING).astype(int)

acc_norm = accuracy_score(y_true_cls, p_norm)
acc_roll = accuracy_score(y_true_cls, p_roll)

print(f"\n📈 Regression MAE: {mae:.4f}")
print(f"🎯 Normal Mod Accuracy: {acc_norm:.2%}")
print(f"🚀 Rolling Mod Accuracy: {acc_roll:.2%}")

# Simülasyon
initial_bankroll = 1000.0
bet_amount = 10.0

# Kasa 1: Normal (0.85+) -> Dinamik Çıkış
w1 = initial_bankroll
b1, w_cnt1 = 0, 0
for i in range(len(y_reg_te)):
    if p_thr[i] >= THRESHOLD_NORMAL:
        w1 -= bet_amount
        b1 += 1
        exit_pt = min(max(1.5, p_reg[i] * 0.8), 2.5)
        if y_reg_te[i] >= exit_pt:
            w1 += exit_pt * bet_amount
            w_cnt1 += 1

roi1 = (w1 - initial_bankroll) / initial_bankroll * 100
wr1 = (w_cnt1 / b1 * 100) if b1 > 0 else 0
print(f"\n💰 KASA 1 (NORMAL): ROI {roi1:+.2f}% | Win Rate {wr1:.1f}% | Bets {b1}")

# Kasa 2: Rolling (0.95+) -> Sabit 1.5x
w2 = initial_bankroll
b2, w_cnt2 = 0, 0
for i in range(len(y_reg_te)):
    if p_thr[i] >= THRESHOLD_ROLLING:
        w2 -= bet_amount
        b2 += 1
        if y_reg_te[i] >= 1.5:
            w2 += 1.5 * bet_amount
            w_cnt2 += 1

roi2 = (w2 - initial_bankroll) / initial_bankroll * 100
wr2 = (w_cnt2 / b2 * 100) if b2 > 0 else 0
print(f"💰 KASA 2 (ROLLING): ROI {roi2:+.2f}% | Win Rate {wr2:.1f}% | Bets {b2}")

# -----------------------------------------------------------------------------
# 8. KAYDET VE PAKETLE
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("📦 KAYIT VE PAKETLEME")
print("="*60)

joblib.dump(scaler, 'models/scaler_progressive.pkl')

# Info
info = {
    'model': 'Progressive_Transformer_Ultimate',
    'version': '5.3_FIXED',
    'thresholds': {'normal': THRESHOLD_NORMAL, 'rolling': THRESHOLD_ROLLING},
    'metrics': {'mae': float(mae), 'normal_acc': float(acc_norm), 'rolling_acc': float(acc_roll)},
    'simulation': {'normal_roi': float(roi1), 'rolling_roi': float(roi2)}
}
with open('models/model_info.json', 'w') as f: json.dump(info, f, indent=2)

# Zip (models klasörünü ziple)
shutil.make_archive('jetx_models_progressive_v5.2', 'zip', 'models')
print("✅ ZIP oluşturuldu.")

print("\n🎉 İŞLEM BAŞARIYLA TAMAMLANDI!")
print("="*80)
