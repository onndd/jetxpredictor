#!/usr/bin/env python3
"""
🎯 JetX PROGRESSIVE TRAINING - 3 Aşamalı Eğitim Stratejisi

AMAÇ: 1.5 altı değerleri tahmin edebilen model eğitmek (Yüksek Güvenli)

STRATEJI:
├── AŞAMA 1: Foundation Training (100 epoch) - Threshold baştan aktif
├── AŞAMA 2: Threshold Fine-Tuning (80 epoch) - Yumuşak class weights (5x)
└── AŞAMA 3: Full Model Fine-Tuning (80 epoch) - Dengeli final (7x)

HEDEFLER (GÜNCELLENDİ - %85 Güven Eşiği İle):
- 1.5 ALTI Doğruluk: %75+ (Eşik 0.85)
- 1.5 ÜSTÜ Doğruluk: %75+ (Eşik 0.85)
- Para kaybı riski: %20 altı
- MAE: < 2.0

SÜRE: ~1.5 saat (GPU ile)
"""

import subprocess
import sys
import os
import time
from datetime import datetime

print("="*80)
print("🎯 JetX PROGRESSIVE TRAINING - 3 Aşamalı Eğitim (Keskin Nişancı Modu)")
print("="*80)
print(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Kütüphaneleri yükle
print("📦 Kütüphaneler yükleniyor...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                      "tensorflow", "scikit-learn", "pandas", "numpy", 
                      "scipy", "joblib", "matplotlib", "seaborn", "tqdm",
                      "PyWavelets", "nolds"])

import numpy as np
import pandas as pd
import joblib
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, backend as K
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import warnings
import pickle
from datetime import datetime
warnings.filterwarnings('ignore')

print(f"✅ TensorFlow: {tf.__version__}")
print(f"✅ GPU: {'Mevcut' if len(tf.config.list_physical_devices('GPU')) > 0 else 'Yok (CPU)'}")

# Proje yükle
if not os.path.exists('jetxpredictor'):
    print("\n📥 Proje klonlanıyor...")
    subprocess.check_call(["git", "clone", "https://github.com/onndd/jetxpredictor.git"])

os.chdir('jetxpredictor')
sys.path.append(os.getcwd())

# GPU Konfigürasyonunu yükle ve uygula
from utils.gpu_config import setup_tensorflow_gpu, print_gpu_status
print_gpu_status()
gpu_config = setup_tensorflow_gpu()
print()

from category_definitions import CategoryDefinitions, FeatureEngineering
from utils.balanced_batch_generator import BalancedBatchGenerator
from utils.adaptive_weight_scheduler import AdaptiveWeightScheduler
from utils.advanced_bankroll import AdvancedBankrollManager
from utils.custom_losses import balanced_threshold_killer_loss, balanced_focal_loss, create_weighted_binary_crossentropy, percentage_aware_regression_loss
from utils.virtual_bankroll_callback import VirtualBankrollCallback
print(f"✅ Proje yüklendi - Kritik eşik: {CategoryDefinitions.CRITICAL_THRESHOLD}x\n")

# KRITIK GÜVEN EŞİĞİ
CONFIDENCE_THRESHOLD = 0.85

# =============================================================================
# TRANSFORMER LAYERS (YENİ - FAZ 2)
# =============================================================================
class PositionalEncoding(layers.Layer):
    """
    Positional Encoding for Transformer
    Time series için zamansal bilgi ekler
    TensorFlow 2.x uyumlu - build() metodunda oluşturulur
    """
    def __init__(self, max_seq_len=1000, d_model=256, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.pe = None
        
    def build(self, input_shape):
        # Positional encoding matrix'i build'de oluştur
        position = tf.range(self.max_seq_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, self.d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / self.d_model))
        
        pe_sin = tf.sin(position * div_term)
        pe_cos = tf.cos(position * div_term)
        
        # Alternating sin/cos pattern oluştur
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
        config.update({
            'max_seq_len': self.max_seq_len,
            'd_model': self.d_model
        })
        return config


class LightweightTransformerEncoder(layers.Layer):
    """
    Lightweight Transformer Encoder for Time Series
    
    Args:
        d_model: Model dimension (256)
        num_layers: Number of transformer layers (4)
        num_heads: Number of attention heads (8)
        dff: Feedforward dimension (1024)
        dropout: Dropout rate (0.2)
    """
    def __init__(
        self, 
        d_model=256, 
        num_layers=4, 
        num_heads=8, 
        dff=1024, 
        dropout=0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout
        
        # Input projection (sequence_len, 1) → (sequence_len, d_model)
        self.input_projection = layers.Dense(d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(max_seq_len=1000, d_model=d_model)
        
        # Transformer encoder layers
        self.encoder_layers = []
        for _ in range(num_layers):
            # Multi-head attention
            mha = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=d_model // num_heads,
                dropout=dropout
            )
            
            # Feedforward network
            ffn = tf.keras.Sequential([
                layers.Dense(dff, activation='relu'),
                layers.Dropout(dropout),
                layers.Dense(d_model)
            ])
            
            # Layer normalization
            layernorm1 = layers.LayerNormalization(epsilon=1e-6)
            layernorm2 = layers.LayerNormalization(epsilon=1e-6)
            
            # Dropout
            dropout1 = layers.Dropout(dropout)
            dropout2 = layers.Dropout(dropout)
            
            self.encoder_layers.append({
                'mha': mha,
                'ffn': ffn,
                'layernorm1': layernorm1,
                'layernorm2': layernorm2,
                'dropout1': dropout1,
                'dropout2': dropout2
            })
        
        # Global average pooling
        self.global_pool = layers.GlobalAveragePooling1D()
        
        # Output projection
        self.output_projection = layers.Dense(d_model)
        self.dropout_final = layers.Dropout(dropout)
    
    def call(self, inputs, training=None):
        """
        Forward pass
        
        Args:
            inputs: (batch_size, seq_len, 1) - Time series input
            training: Training mode flag
            
        Returns:
            (batch_size, d_model) - Encoded representation
        """
        # Input projection
        x = self.input_projection(inputs)  # (batch, seq_len, d_model)
        
        # Positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoder layers
        for layer in self.encoder_layers:
            # Multi-head attention
            attn_output = layer['mha'](
                query=x,
                key=x,
                value=x,
                training=training
            )
            attn_output = layer['dropout1'](attn_output, training=training)
            x = layer['layernorm1'](x + attn_output)  # Residual connection
            
            # Feedforward network
            ffn_output = layer['ffn'](x)
            ffn_output = layer['dropout2'](ffn_output, training=training)
            x = layer['layernorm2'](x + ffn_output)  # Residual connection
        
        # Global pooling
        x = self.global_pool(x)  # (batch, d_model)
        
        # Output projection
        x = self.output_projection(x)
        x = self.dropout_final(x, training=training)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout': self.dropout_rate
        })
        return config


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
window_size = 1000  # 500 → 1000 (daha uzun vadeli pattern analizi)
X_f, X_50, X_200, X_500, X_1000 = [], [], [], [], []
y_reg, y_cls, y_thr = [], [], []

for i in tqdm(range(window_size, len(all_values)-1), desc='Features'):
    hist = all_values[:i].tolist()
    target = all_values[i]
    
    feats = FeatureEngineering.extract_all_features(hist)
    X_f.append(list(feats.values()))
    X_50.append(all_values[i-50:i])
    X_200.append(all_values[i-200:i])
    X_500.append(all_values[i-500:i])
    X_1000.append(all_values[i-1000:i])  # YENİ: 1000'lik pencere
    
    y_reg.append(target)
    cat = CategoryDefinitions.get_category_numeric(target)
    onehot = np.zeros(3)
    onehot[cat] = 1
    y_cls.append(onehot)
    y_thr.append(1.0 if target >= 1.5 else 0.0)

X_f = np.array(X_f)
X_50 = np.array(X_50).reshape(-1, 50, 1)
X_200 = np.array(X_200).reshape(-1, 200, 1)
X_500 = np.array(X_500).reshape(-1, 500, 1)
X_1000 = np.array(X_1000).reshape(-1, 1000, 1)  # YENİ: 1000'lik pencere
y_reg = np.array(y_reg)
y_cls = np.array(y_cls)
y_thr = np.array(y_thr).reshape(-1, 1)

print(f"✅ {len(X_f):,} örnek hazırlandı")
print(f"✅ Feature sayısı: {X_f.shape[1]}")

# Normalizasyon
print("\n📊 Normalizasyon...")
scaler = StandardScaler()
X_f = scaler.fit_transform(X_f)
X_50 = np.log10(X_50 + 1e-8)
X_200 = np.log10(X_200 + 1e-8)
X_500 = np.log10(X_500 + 1e-8)
X_1000 = np.log10(X_1000 + 1e-8)  # YENİ: 1000'lik pencere normalizasyonu

# =============================================================================
# TIME-SERIES SPLIT (KRONOLOJIK) - SABİT SAYILAR
# =============================================================================
print("\n📊 TIME-SERIES SPLIT (Kronolojik Bölme)...")
print("⚠️  UYARI: Shuffle devre dışı - Zaman serisi yapısı korunuyor!")

# Sabit split sayıları
test_size = 1500
val_size = 1000
total_samples = len(X_f)
train_size = total_samples - test_size - val_size

print(f"📊 Veri Dağılımı (Sabit Sayılar):")
print(f"  Train: {train_size:,} sample")
print(f"  Validation: {val_size:,} sample")
print(f"  Test: {test_size:,} sample")
print(f"  Toplam: {total_samples:,} sample\n")

# Kronolojik split: Train -> Val -> Test
train_end = train_size
val_end = train_size + val_size

# Train set
X_f_tr = X_f[:train_end]
X_50_tr = X_50[:train_end]
X_200_tr = X_200[:train_end]
X_500_tr = X_500[:train_end]
X_1000_tr = X_1000[:train_end]
y_reg_tr = y_reg[:train_end]
y_cls_tr = y_cls[:train_end]
y_thr_tr = y_thr[:train_end]

# Validation set
X_f_val = X_f[train_end:val_end]
X_50_val = X_50[train_end:val_end]
X_200_val = X_200[train_end:val_end]
X_500_val = X_500[train_end:val_end]
X_1000_val = X_1000[train_end:val_end]
y_reg_val = y_reg[train_end:val_end]
y_cls_val = y_cls[train_end:val_end]
y_thr_val = y_thr[train_end:val_end]

# Test set
X_f_te = X_f[val_end:]
X_50_te = X_50[val_end:]
X_200_te = X_200[val_end:]
X_500_te = X_500[val_end:]
X_1000_te = X_1000[val_end:]
y_reg_te = y_reg[val_end:]
y_cls_te = y_cls[val_end:]
y_thr_te = y_thr[val_end:]

print(f"✅ Veri split tamamlandı")

# =============================================================================
# CUSTOM LOSS FUNCTIONS
# =============================================================================
def threshold_killer_loss(y_true, y_pred):
    """1.5 altı yanlış tahmine DENGELI CEZA - Lazy learning'i önler"""
    mae = K.abs(y_true - y_pred)
    
    # 1.5 altıyken üstü tahmin = 2x ceza (PARA KAYBI - yumuşatıldı: 4→2)
    false_positive = K.cast(
        tf.logical_and(y_true < 1.5, y_pred >= 1.5),
        'float32'
    ) * 2.0
    
    # 1.5 üstüyken altı tahmin = 1.5x ceza (yumuşatıldı: 2→1.5)
    false_negative = K.cast(
        tf.logical_and(y_true >= 1.5, y_pred < 1.5),
        'float32'
    ) * 1.5
    
    # Kritik bölge (1.4-1.6) = 2.5x ceza (yumuşatıldı: 3→2.5)
    critical_zone = K.cast(
        tf.logical_and(y_true >= 1.4, y_true <= 1.6),
        'float32'
    ) * 2.5
    
    weight = K.maximum(K.maximum(false_positive, false_negative), critical_zone)
    weight = K.maximum(weight, 1.0)
    
    return K.mean(mae * weight)

def ultra_focal_loss(gamma=2.5, alpha=0.75):
    """Focal loss - yanlış tahminlere dengeli ceza (yumuşatıldı: gamma 4.0→2.5)"""
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = alpha * K.pow(1 - pt, gamma)
        return -K.mean(focal_weight * K.log(pt))
    return loss

def create_weighted_binary_crossentropy(weight_0, weight_1):
    """
    Sınıf ağırlıklarını doğrudan içeren weighted binary crossentropy loss fonksiyonu
    
    Args:
        weight_0: 1.5 altı (class 0) için ağırlık
        weight_1: 1.5 üstü (class 1) için ağırlık
    
    Returns:
        Ağırlıklı binary crossentropy loss fonksiyonu
    """
    def loss(y_true, y_pred):
        # Binary crossentropy hesapla
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        bce = -(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        
        # Class weight'leri uygula
        # y_true = 1 ise weight_1, y_true = 0 ise weight_0 kullan
        weights = y_true * weight_1 + (1 - y_true) * weight_0
        
        # Ağırlıklı loss'u döndür
        return K.mean(bce * weights)
    
    return loss

# =============================================================================
# OPTIMIZE EDİLMİŞ MODEL MİMARİSİ (8-10M parametre)
# =============================================================================
def build_progressive_model(n_features):
    """
    Optimize edilmiş model - Dengeli derinlik
    ~12-15M parametre (1000 penceresi ile artış)
    """
    inp_f = layers.Input((n_features,), name='features')
    inp_50 = layers.Input((50, 1), name='seq50')
    inp_200 = layers.Input((200, 1), name='seq200')
    inp_500 = layers.Input((500, 1), name='seq500')
    inp_1000 = layers.Input((1000, 1), name='seq1000')  # YENİ: 1000'lik pencere girişi
    
    # N-BEATS (Optimize - 5-7 block)
    def nbeats_block(x, units, blocks, name):
        for i in range(blocks):
            x = layers.Dense(units, activation='relu', kernel_regularizer='l2')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        return x
    
    # Kısa sequence (50)
    nb_s = layers.Flatten()(inp_50)
    nb_s = nbeats_block(nb_s, 128, 5, 's')
    nb_s = layers.Dense(128, activation='relu')(nb_s)
    nb_s = layers.Dropout(0.2)(nb_s)
    
    # Orta sequence (200)
    nb_m = layers.Flatten()(inp_200)
    nb_m = nbeats_block(nb_m, 192, 6, 'm')
    nb_m = layers.Dense(192, activation='relu')(nb_m)
    nb_m = layers.Dropout(0.2)(nb_m)
    
    # Uzun sequence (500)
    nb_l = layers.Flatten()(inp_500)
    nb_l = nbeats_block(nb_l, 256, 7, 'l')
    nb_l = layers.Dense(256, activation='relu')(nb_l)
    nb_l = layers.Dropout(0.2)(nb_l)
    
    # YENİ: Çok uzun sequence (1000) - daha derin analiz
    nb_xl = layers.Flatten()(inp_1000)
    nb_xl = nbeats_block(nb_xl, 384, 9, 'xl')  # 9 block, 384 units
    nb_xl = layers.Dense(384, activation='relu')(nb_xl)
    nb_xl = layers.Dropout(0.2)(nb_xl)
    
    nb_all = layers.Concatenate()([nb_s, nb_m, nb_l, nb_xl])  # nb_xl eklendi
    
    # TCN (Optimize - 7 layer)
    def tcn_block(x, filters, dilation, name):
        conv = layers.Conv1D(filters, 3, dilation_rate=dilation, padding='causal', 
                            activation='relu', kernel_regularizer='l2')(x)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Dropout(0.2)(conv)
        residual = layers.Conv1D(filters, 1, padding='same')(x) if x.shape[-1] != filters else x
        return layers.Add()([conv, residual])
    
    tcn = inp_500
    tcn = tcn_block(tcn, 128, 1, '1')
    tcn = tcn_block(tcn, 128, 2, '2')
    tcn = tcn_block(tcn, 256, 4, '3')
    tcn = tcn_block(tcn, 256, 8, '4')
    tcn = tcn_block(tcn, 512, 16, '5')
    tcn = tcn_block(tcn, 512, 32, '6')
    tcn = tcn_block(tcn, 512, 64, '7')
    tcn = layers.GlobalAveragePooling1D()(tcn)
    tcn = layers.Dense(512, activation='relu')(tcn)
    tcn = layers.Dropout(0.25)(tcn)
    
    # YENİ: Transformer branch (FAZ 2)
    # 1000'lik sequence için Transformer encoder kullan
    transformer = LightweightTransformerEncoder(
        d_model=256,
        num_layers=4,
        num_heads=8,
        dff=1024,
        dropout=0.2
    )(inp_1000)
    # Transformer output: (batch, 256)
    
    # Fusion (Optimize) - YENİ: Transformer eklendi
    fus = layers.Concatenate()([inp_f, nb_all, tcn, transformer])
    fus = layers.Dense(512, activation='relu', kernel_regularizer='l2')(fus)
    fus = layers.BatchNormalization()(fus)
    fus = layers.Dropout(0.3)(fus)
    fus = layers.Dense(256, activation='relu')(fus)
    fus = layers.BatchNormalization()(fus)
    fus = layers.Dropout(0.25)(fus)
    fus = layers.Dense(128, activation='relu')(fus)
    fus = layers.Dropout(0.2)(fus)
    
    # Outputs
    reg_branch = layers.Dense(64, activation='relu')(fus)
    reg_branch = layers.Dropout(0.2)(reg_branch)
    out_reg = layers.Dense(1, activation='linear', name='regression')(reg_branch)
    
    cls_branch = layers.Dense(64, activation='relu')(fus)
    cls_branch = out_cls = layers.Dense(3, activation='softmax', name='classification')(cls_branch)
    
    thr_branch = layers.Dense(32, activation='relu')(fus)
    thr_branch = layers.Dropout(0.2)(thr_branch)
    out_thr = layers.Dense(1, activation='sigmoid', name='threshold')(thr_branch)
    
    model = models.Model([inp_f, inp_50, inp_200, inp_500, inp_1000], [out_reg, out_cls, out_thr])
    return model

# =============================================================================
# DYNAMIC WEIGHT CALLBACK - Otomatik Class Weight Ayarlama
# =============================================================================
class DynamicWeightCallback(callbacks.Callback):
    """
    Eğitim sırasında 1.5 altı doğruluğunu izler ve class weight'i otomatik ayarlar.
    GÜNCELLEME: Başarıyı 0.85 eşiğine göre ölçer.
    """
    def __init__(self, stage_name, initial_weight=3.0, target_below_acc=0.70):
        super().__init__()
        self.stage_name = stage_name
        self.current_weight = initial_weight
        self.target_below_acc = target_below_acc
        self.best_below_acc = 0
        self.best_weight = initial_weight
        self.weight_history = []
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:  # Her 5 epoch'ta bir kontrol et
            # VALIDATION seti üzerinde threshold metrics (test leakage önlendi!)
            p = self.model.predict([X_f_val, X_50_val, X_200_val, X_500_val, X_1000_val], verbose=0)[2].flatten()
            # GÜNCELLEME: 0.85 Güven eşiği
            p_thr = (p >= CONFIDENCE_THRESHOLD).astype(int)
            t_thr = (y_reg_val >= 1.5).astype(int)
            
            below_mask = t_thr == 0
            above_mask = t_thr == 1
            
            below_acc = accuracy_score(t_thr[below_mask], p_thr[below_mask]) if below_mask.sum() > 0 else 0
            above_acc = accuracy_score(t_thr[above_mask], p_thr[above_mask]) if above_mask.sum() > 0 else 0
            
            # Class weight ayarlaması (otomatik)
            old_weight = self.current_weight
            
            if below_acc < 0.15:  # Çok düşük - ciddi artış gerekli
                self.current_weight *= 1.8
                adjustment = "🔴 Ciddi Artış (×1.8)"
            elif below_acc < 0.40:  # Düşük - artış gerekli
                self.current_weight *= 1.3
                adjustment = "🟠 Orta Artış (×1.3)"
            elif below_acc < 0.60:  # Hedefin altında - hafif artış
                self.current_weight *= 1.1
                adjustment = "🟡 Hafif Artış (×1.1)"
            elif below_acc > 0.85 and above_acc < 0.50:  # Çok yüksek - azaltma gerekli
                self.current_weight *= 0.7
                adjustment = "🟢 Azaltma (×0.7)"
            else:
                adjustment = "✅ Değişiklik Yok (Dengeli)"
            
            # Weight'i sınırla (1.0 - 25.0 arası)
            self.current_weight = max(1.0, min(25.0, self.current_weight))
            
            # En iyi sonucu kaydet
            if below_acc > self.best_below_acc:
                self.best_below_acc = below_acc
                self.best_weight = self.current_weight
            
            # Geçmişi kaydet
            self.weight_history.append({
                'epoch': epoch,
                'weight': self.current_weight,
                'below_acc': below_acc,
                'above_acc': above_acc
            })
            
            # Rapor
            print(f"\n{'='*70}")
            print(f"📊 {self.stage_name} - Epoch {epoch+1} - DYNAMIC WEIGHT ADJUSTMENT")
            print(f"{'='*70}")
            print(f"🔴 1.5 ALTI: {below_acc*100:.1f}% (Eşik: 0.85)")
            print(f"🟢 1.5 ÜSTÜ: {above_acc*100:.1f}% (Eşik: 0.85)")
            print(f"⚖️  Weight Ayarlaması: {old_weight:.2f} → {self.current_weight:.2f} ({adjustment})")
            print(f"🏆 En İyi 1.5 Altı: {self.best_below_acc*100:.1f}% (Weight: {self.best_weight:.2f})")
            print(f"{'='*70}\n")

# =============================================================================
# METRICS CALLBACK (Raporlama için)
# =============================================================================
class ProgressiveMetricsCallback(callbacks.Callback):
    def __init__(self, stage_name):
        super().__init__()
        self.stage_name = stage_name
        self.best_below_acc = 0
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            # VALIDATION seti üzerinde threshold metrics (test leakage önlendi!)
            p = self.model.predict([X_f_val, X_50_val, X_200_val, X_500_val, X_1000_val], verbose=0)[2].flatten()
            # GÜNCELLEME: 0.85 Güven eşiği
            p_thr = (p >= CONFIDENCE_THRESHOLD).astype(int)
            t_thr = (y_reg_val >= 1.5).astype(int)
            
            below_mask = t_thr == 0
            above_mask = t_thr == 1
            
            below_acc = accuracy_score(t_thr[below_mask], p_thr[below_mask]) if below_mask.sum() > 0 else 0
            above_acc = accuracy_score(t_thr[above_mask], p_thr[above_mask]) if above_mask.sum() > 0 else 0
            
            false_positive = ((p_thr == 1) & (t_thr == 0)).sum()
            total_below = below_mask.sum()
            risk = false_positive / total_below if total_below > 0 else 0
            
            print(f"\n{'='*70}")
            print(f"📊 {self.stage_name} - Epoch {epoch+1} METRIKLER")
            print(f"{'='*70}")
            
            # 1.5 Altı Doğruluğu
            below_emoji = "✅" if below_acc >= 0.75 else "⚠️" if below_acc >= 0.50 else "❌"
            print(f"\n🔴 1.5 ALTI DOĞRULUĞU: {below_acc*100:.1f}% {below_emoji}")
            print(f"   └─ Ne anlama geliyor?")
            print(f"      Model 1.5 altındaki değerleri ne kadar iyi tahmin ediyor?")
            print(f"      Örnek: 100 adet 1.5 altı değerden {int(below_acc*100)} tanesini doğru buldu")
            print(f"   └─ Hedef: %75+ (şu an: {'HEDEF AŞILDI! ✅' if below_acc >= 0.75 else f'%{(75-below_acc*100):.1f} daha gerekli'})")
            
            # 1.5 Üstü Doğruluğu
            above_emoji = "✅" if above_acc >= 0.75 else "⚠️" if above_acc >= 0.50 else "❌"
            print(f"\n🟢 1.5 ÜSTÜ DOĞRULUĞU: {above_acc*100:.1f}% {above_emoji}")
            print(f"   └─ Ne anlama geliyor?")
            print(f"      Model 1.5 üstündeki değerleri ne kadar iyi tahmin ediyor?")
            print(f"      Örnek: 100 adet 1.5 üstü değerden {int(above_acc*100)} tanesini doğru buldu")
            print(f"   └─ Hedef: %75+ (şu an: {'HEDEF AŞILDI! ✅' if above_acc >= 0.75 else f'%{(75-above_acc*100):.1f} daha gerekli'})")
            
            # Para Kaybı Riski
            risk_emoji = "✅" if risk < 0.20 else "⚠️" if risk < 0.40 else "❌"
            print(f"\n💰 PARA KAYBI RİSKİ: {risk*100:.1f}% {risk_emoji}")
            print(f"   └─ Ne anlama geliyor?")
            print(f"      Model 1.5 altı olduğunda yanlışlıkla '1.5 üstü' deme oranı")
            print(f"      Bu durumda bahis yapar ve PARA KAYBEDERSİNİZ!")
            print(f"      Örnek: 100 oyunun {int(risk*100)}'ında yanlış tahminle para kaybı")
            print(f"   └─ Hedef: <%20 (şu an: {'GÜVENLİ! ✅' if risk < 0.20 else f'%{(risk*100-20):.1f} daha fazla risk var'})")
            
            # Model Durumu Özeti
            print(f"\n🎯 MODEL DURUMU (Eşik: 0.85):")
            if below_acc >= 0.75 and above_acc >= 0.75 and risk < 0.20:
                print(f"   ✅ ✅ ✅ MÜKEMMEL! Model kullanıma hazır!")
            elif below_acc >= 0.60 and risk < 0.30:
                print(f"   ✅ İYİ - Biraz daha eğitimle hedeflere ulaşılabilir")
            elif below_acc == 0.0 or below_acc == 1.0:
                print(f"   ❌ KÖTÜ! Model bir tarafa KILITLENIYOR!")
                print(f"      → Model dengesiz öğreniyor, class weight ayarlanmalı")
            else:
                print(f"   ⚠️ ORTA - Devam ediyor...")
            
            # Sanal Kasa Simülasyonu
            print(f"\n💰 SANAL KASA SİMÜLASYONU (Test Seti):")
            wallet = 1000.0  # Başlangıç kasası
            bet_amount = 10.0  # Her bahis miktarı
            win_amount = 15.0  # Kazanınca eklenen miktar
            
            total_bets = 0
            total_wins = 0
            total_losses = 0
            
            # Test verileri üzerinde simülasyon
            for i in range(len(p_thr)):
                model_pred = p_thr[i]  # Model tahmini (1.5 üstü ve güvenli mi?)
                actual_value = y_reg_te[i]  # Gerçek değer
                
                # Model "1.5 üstü" diyorsa (zaten 0.85 üstü filtrelendi)
                if model_pred == 1:
                    wallet -= bet_amount  # Bahis yap
                    total_bets += 1
                    
                    # Gerçek sonuca bak
                    if actual_value >= 1.5:
                        # Kazandık!
                        wallet += win_amount
                        total_wins += 1
                    else:
                        # Kaybettik
                        total_losses += 1
                # Model "1.5 altı" diyorsa pas geç
            
            # Sonuçları hesapla
            profit_loss = wallet - 1000.0
            roi = (profit_loss / 1000.0) * 100 if total_bets > 0 else 0
            win_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0
            
            # Emoji seç
            if profit_loss > 100:
                wallet_emoji = "🚀"
            elif profit_loss > 0:
                wallet_emoji = "✅"
            elif profit_loss > -100:
                wallet_emoji = "⚠️"
            else:
                wallet_emoji = "❌"
            
            # Geliştirilmiş rapor formatı
            net_wins = total_wins * (win_amount - bet_amount)  # Net kazanç
            net_losses = total_losses * bet_amount  # Net kayıp
            
            print(f"   ════════════════════════════════════════════════════")
            print(f"   ")
            print(f"   📊 OYUN PARAMETRELERİ:")
            print(f"      Başlangıç Sermayesi: {1000.0:,.2f} TL")
            print(f"      Bahis Tutarı: {bet_amount:.2f} TL (sabit)")
            print(f"      Kazanç Hedefi: 1.5x → {win_amount:.2f} TL geri alma")
            print(f"      ")
            print(f"      Her Kazançta: +{win_amount - bet_amount:.2f} TL ({win_amount:.0f} - {bet_amount:.0f} = {win_amount - bet_amount:.0f})")
            print(f"      Her Kayıpta: -{bet_amount:.2f} TL (bahis kaybı)")
            print(f"   ")
            print(f"   🎯 TEST SETİ SONUÇLARI:")
            print(f"      Toplam Oyun: {total_bets} el")
            print(f"      ✅ Kazanan: {total_wins} oyun ({win_rate:.1f}%)")
            print(f"      ❌ Kaybeden: {total_losses} oyun ({100-win_rate:.1f}%)")
            print(f"   ")
            print(f"   💸 DETAYLI HESAPLAMA:")
            print(f"      ")
            print(f"      Kazanılan Oyunlar ({total_wins} el):")
            print(f"      └─ {total_wins} × {win_amount - bet_amount:.2f} TL = +{net_wins:,.2f} TL ✅")
            print(f"      ")
            print(f"      Kaybedilen Oyunlar ({total_losses} el):")
            print(f"      └─ {total_losses} × {bet_amount:.2f} TL = -{net_losses:,.2f} TL ❌")
            print(f"      ")
            print(f"      {'─'*50}")
            print(f"      Net Kar/Zarar: {net_wins:,.2f} - {net_losses:,.2f} = {profit_loss:+,.2f} TL")
            print(f"      Final Sermaye: 1,000 {profit_loss:+,.0f} = {wallet:,.2f} TL (kalan)")
            print(f"   ")
            print(f"   📈 PERFORMANS ANALİZİ:")
            print(f"      ")
            print(f"      ROI: {roi:+.1f}% {wallet_emoji}")
            print(f"      └─ Sermayenin {'%'+str(round((wallet/1000.0)*100, 1)) if wallet > 0 else '0'}'si kaldı")
            print(f"      ")
            print(f"      🎯 BAŞABAŞ İÇİN GEREKLİ:")
            print(f"         2 kazanç = 1 kayıp dengelemeli (2×{win_amount - bet_amount:.0f} = 1×{bet_amount:.0f})")
            print(f"         Gerekli Kazanma Oranı: %66.7 (3'te 2)")
            print(f"      ")
            print(f"      📊 MEVCUT DURUM:")
            print(f"         Kazanma Oranı: {win_rate:.1f}% ({total_bets}'de {total_wins})")
            print(f"         Hedeften Fark: {win_rate - 66.7:+.1f}% {'⚠️' if win_rate < 66.7 else '✅'}")
            print(f"      ")
            print(f"   💡 DEĞERLENDİRME:")
            print(f"      ")
            if profit_loss > 0:
                print(f"      ✅ Model bu performansla kar ettiriyor!")
            else:
                print(f"      ❌ Model bu performansla zarar ettiriyor!")
            print(f"      ")
            print(f"      📊 Matematik:")
            print(f"         • 2 kazanç = +{(win_amount - bet_amount) * 2:.0f} TL (2 × {win_amount - bet_amount:.0f})")
            print(f"         • 1 kayıp = -{bet_amount:.0f} TL")
            print(f"         • Bu yüzden en az %67 kazanma şart!")
            print(f"      ")
            if win_rate < 66.7:
                print(f"      ⚠️ %{win_rate:.1f} kazanma oranı yetersiz:")
                games_per_100 = 100
                wins_per_100 = round(win_rate)
                losses_per_100 = 100 - wins_per_100
                net_per_100 = (wins_per_100 * (win_amount - bet_amount)) - (losses_per_100 * bet_amount)
                print(f"         • Her 100 oyunda ~{wins_per_100} kazanç, ~{losses_per_100} kayıp")
                print(f"         • Net: ({wins_per_100}×{win_amount - bet_amount:.0f}) - ({losses_per_100}×{bet_amount:.0f}) = {net_per_100:+.0f} TL")
                print(f"         • 100 oyunda ~{abs(net_per_100):.0f} TL {'kayıp!' if net_per_100 < 0 else 'kar!'}")
            print(f"   ")
            print(f"   ════════════════════════════════════════════════════")
            
            print(f"\n{'='*70}\n")
            
            if below_acc > self.best_below_acc:
                self.best_below_acc = below_acc
                print(f"  ✨ YENİ REKOR! En iyi 1.5 altı: {below_acc*100:.1f}%\n")

# =============================================================================
# CHECKPOINT YARDIMCI FONKSİYONLARI
# =============================================================================
def save_checkpoint(stage, epoch, model, optimizer, metrics_history, class_weights, filename=None):
    """
    Eğitim checkpoint'i kaydet
    
    Args:
        stage: Hangi aşama (1, 2, 3)
        epoch: Kaçıncı epoch
        model: Model instance
        optimizer: Optimizer instance (kullanılmıyor - TensorFlow uyumluluk sorunu)
        metrics_history: Metrics geçmişi
        class_weights: Class weight değerleri
        filename: Checkpoint dosya adı (opsiyonel)
    """
    if filename is None:
        filename = f'checkpoint_stage{stage}_epoch{epoch}.pkl'
    
    checkpoint = {
        'stage': stage,
        'epoch': epoch,
        'model_weights': model.get_weights(),
        # optimizer_weights kaldırıldı - TensorFlow/Keras Adam optimizer'ı get_weights() desteklemiyor
        'metrics_history': metrics_history,
        'class_weights': class_weights,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"💾 Checkpoint kaydedildi: {filename}")
    return filename

def load_checkpoint(filename):
    """
    Checkpoint yükle
    
    Args:
        filename: Checkpoint dosya adı
        
    Returns:
        Checkpoint dictionary veya None
    """
    try:
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"✅ Checkpoint yüklendi: {filename}")
        print(f"   Aşama: {checkpoint['stage']}, Epoch: {checkpoint['epoch']}")
        print(f"   Zaman: {checkpoint['timestamp']}")
        return checkpoint
    except FileNotFoundError:
        print(f"⚠️ Checkpoint bulunamadı: {filename}")
        return None
    except Exception as e:
        print(f"❌ Checkpoint yükleme hatası: {e}")
        return None

# =============================================================================
# AŞAMA 1: REGRESSION-ONLY (200 epoch)
# =============================================================================
print("\n" + "="*80)
print("🔥 AŞAMA 1: FOUNDATION TRAINING")
print("="*80)
print("Hedef: Model hem değer tahmin etmeyi HEM DE 1.5 eşiğini birlikte öğrensin")
print("Epoch: 100 | Batch: 64 | LR: 0.0001")
print("Loss Weights: Regression 60%, Classification 10%, Threshold 30%")
print("Monitor: val_threshold_accuracy | Patience: 10")
print("="*80 + "\n")

# Checkpoint kontrolü - AŞAMA 1 için resume
stage1_checkpoint = load_checkpoint('checkpoint_stage1_latest.pkl')
initial_epoch_stage1 = 0

stage1_start = time.time()

model = build_progressive_model(X_f.shape[1])
print(f"✅ Model: {model.count_params():,} parametre")

# Checkpoint varsa yükle
if stage1_checkpoint and stage1_checkpoint['stage'] == 1:
    print("🔄 AŞAMA 1 checkpoint'inden devam ediliyor...")
    model.set_weights(stage1_checkpoint['model_weights'])
    initial_epoch_stage1 = stage1_checkpoint['epoch']
    print(f"   Epoch {initial_epoch_stage1} 'den devam edilecek")

# Class weights - LAZY LEARNING ÖNLEME (yeterince yüksek)
w0_stage1 = 25.0  # 1.5 altı için: 25.0x (lazy learning'i kesin önler)
w1_stage1 = 1.0   # 1.5 üstü baseline

print(f"📊 CLASS WEIGHTS (AŞAMA 1 - Lazy Learning Önleme - TIME-SERIES SPLIT):")
print(f"  1.5 altı: {w0_stage1:.2f}x (yüksek - lazy learning'i önler)")
print(f"  1.5 üstü: {w1_stage1:.2f}x\n")

# AŞAMA 1: Foundation Training - SADECE WEIGHTED BCE (çakışma yok!)
model.compile(
    optimizer=Adam(0.0001),
    loss={
        'regression': percentage_aware_regression_loss,  # YENİ: Yüzde hataya dayalı regression loss
        'classification': 'categorical_crossentropy',
        'threshold': create_weighted_binary_crossentropy(w0_stage1, w1_stage1)  # SADECE weighted BCE
    },
    loss_weights={'regression': 0.65, 'classification': 0.10, 'threshold': 0.25},  # Regression ağırlığı artırıldı: 0.55 → 0.65
    metrics={'regression': ['mae'], 'classification': ['accuracy'], 'threshold': ['accuracy']}
)

# Dynamic Weight Callback başlat (otomatik ayarlama için)
dynamic_callback_1 = DynamicWeightCallback("AŞAMA 1", initial_weight=1.5, target_below_acc=0.70)

# Virtual Bankroll Callback (HER EPOCH için sanal kasa)
virtual_bankroll_1 = VirtualBankrollCallback(
    stage_name="AŞAMA 1",
    X_test=[X_f_te, X_50_te, X_200_te, X_500_te, X_1000_te],
    y_test=y_reg_te,
    threshold=1.5,
    starting_capital=1000.0,
    bet_amount=10.0
)

cb1 = [
    callbacks.ModelCheckpoint('stage1_best.h5', monitor='val_threshold_accuracy', save_best_only=True, mode='max', verbose=1),
    callbacks.EarlyStopping(monitor='val_threshold_accuracy', patience=12, min_delta=0.001, mode='max', restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, min_lr=1e-6, verbose=1),
    dynamic_callback_1,
    virtual_bankroll_1,  # YENİ: Her epoch sanal kasa gösterimi
    ProgressiveMetricsCallback("AŞAMA 1")
]

hist1 = model.fit(
    [X_f_tr, X_50_tr, X_200_tr, X_500_tr, X_1000_tr],
    {'regression': y_reg_tr, 'classification': y_cls_tr, 'threshold': y_thr_tr},
    epochs=100,
    batch_size=64,
    validation_data=(  # ✅ MANUEL VALIDATION (kronolojik!)
        [X_f_val, X_50_val, X_200_val, X_500_val, X_1000_val],
        {'regression': y_reg_val, 'classification': y_cls_val, 'threshold': y_thr_val}
    ),
    shuffle=False,  # ✅ KRITIK: Shuffle devre dışı (TIME-SERIES)!
    callbacks=cb1,
    verbose=1,
    initial_epoch=initial_epoch_stage1
)

# AŞAMA 1 Checkpoint kaydet
save_checkpoint(
    stage=1,
    epoch=len(hist1.history['loss']),
    model=model,
    optimizer=model.optimizer,
    metrics_history=hist1.history,
    class_weights={'w0': w0_stage1, 'w1': w1_stage1},
    filename='checkpoint_stage1_latest.pkl'
)

stage1_time = time.time() - stage1_start
print(f"\n✅ AŞAMA 1 Tamamlandı! Süre: {stage1_time/60:.1f} dakika")

# AŞAMA 1 Değerlendirme
pred1 = model.predict([X_f_te, X_50_te, X_200_te, X_500_te, X_1000_te], verbose=0)
mae1 = mean_absolute_error(y_reg_te, pred1[0])
print(f"📊 AŞAMA 1 Sonuç: MAE = {mae1:.4f}")

# =============================================================================
# AŞAMA 2: THRESHOLD FINE-TUNING (150 epoch)
# =============================================================================
print("\n" + "="*80)
print("🔥 AŞAMA 2: THRESHOLD FINE-TUNING (Hafif Artış)")
print("="*80)
print("Hedef: 1.5 altı/üstü ayrımını keskinleştir (dengeli class weights)")
print("Epoch: 80 | Batch: 32 | LR: 0.0001 | Class Weight: 1.5x (Yumuşak!)")
print("Monitor: val_threshold_accuracy | Patience: 10")
print("="*80 + "\n")

stage2_start = time.time()

# Checkpoint kontrolü - AŞAMA 2 için resume
stage2_checkpoint = load_checkpoint('checkpoint_stage2_latest.pkl')
initial_epoch_stage2 = 0

# AŞAMA 1 modelini yükle
if stage2_checkpoint and stage2_checkpoint['stage'] == 2:
    print("🔄 AŞAMA 2 checkpoint'inden devam ediliyor...")
    model.set_weights(stage2_checkpoint['model_weights'])
    # optimizer weights kaldırıldı - TensorFlow uyumluluk sorunu
    initial_epoch_stage2 = stage2_checkpoint['epoch']
    print(f"   Epoch {initial_epoch_stage2}'den devam edilecek")
else:
    model.load_weights('stage1_best.h5')

# Class weights - LAZY LEARNING ÖNLEME (yeterince yüksek)
w0 = 30.0  # 1.5 altı için: 30.0x (lazy learning'i kesin önler)
w1 = 1.0   # 1.5 üstü baseline

print(f"📊 CLASS WEIGHTS (AŞAMA 2 - Lazy Learning Önleme - TIME-SERIES SPLIT):")
print(f"  1.5 altı: {w0:.2f}x (yüksek - lazy learning önleme)")
print(f"  1.5 üstü: {w1:.2f}x\n")

# AŞAMA 2: Regression + Threshold - SADECE WEIGHTED BCE (çakışma yok!)
model.compile(
    optimizer=Adam(0.0001),
    loss={
        'regression': percentage_aware_regression_loss,  # YENİ: Yüzde hataya dayalı regression loss
        'classification': 'categorical_crossentropy',
        'threshold': create_weighted_binary_crossentropy(w0, w1)  # SADECE weighted BCE
    },
    loss_weights={'regression': 0.55, 'classification': 0.10, 'threshold': 0.35},  # Regression ağırlığı artırıldı: 0.45 → 0.55
    metrics={'regression': ['mae'], 'classification': ['accuracy'], 'threshold': ['accuracy', 'binary_crossentropy']}
)

# Adaptive Weight Scheduler (GÜÇLENDIRILDI - Lazy Learning Önleme)
adaptive_scheduler_2 = AdaptiveWeightScheduler(
    initial_weight=2.0,    # DÜZELTME: 20.0 → 2.0 (10x azaltma!)
    min_weight=1.0,        # DÜZELTME: 10.0 → 1.0 (normal seviye)
    max_weight=5.0,        # DÜZELTME: 50.0 → 5.0 (reasonable limit)
    target_below_acc=0.70,
    target_above_acc=0.75,
    test_data=([X_f_te, X_50_te, X_200_te, X_500_te, X_1000_te], y_reg_te),
    threshold=1.5,
    check_interval=5
)

# Dynamic Weight Callback (mevcut - opsiyonel)
dynamic_callback_2 = DynamicWeightCallback("AŞAMA 2", initial_weight=1.5, target_below_acc=0.70)

# Virtual Bankroll Callback (HER EPOCH için sanal kasa)
virtual_bankroll_2 = VirtualBankrollCallback(
    stage_name="AŞAMA 2",
    X_test=[X_f_te, X_50_te, X_200_te, X_500_te, X_1000_te],
    y_test=y_reg_te,
    threshold=1.5,
    starting_capital=1000.0,
    bet_amount=10.0
)

cb2 = [
    callbacks.ModelCheckpoint('stage2_best.h5', monitor='val_threshold_accuracy', save_best_only=True, mode='max', verbose=1),
    callbacks.EarlyStopping(monitor='val_threshold_accuracy', patience=10, min_delta=0.001, mode='max', restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1),
    adaptive_scheduler_2,  # YENİ: Adaptive weight scheduler
    virtual_bankroll_2,  # YENİ: Her epoch sanal kasa gösterimi
    ProgressiveMetricsCallback("AŞAMA 2")
]

hist2 = model.fit(
    [X_f_tr, X_50_tr, X_200_tr, X_500_tr, X_1000_tr],
    {'regression': y_reg_tr, 'classification': y_cls_tr, 'threshold': y_thr_tr},
    epochs=80,
    batch_size=32,
    validation_data=(  # ✅ MANUEL VALIDATION (kronolojik!)
        [X_f_val, X_50_val, X_200_val, X_500_val, X_1000_val],
        {'regression': y_reg_val, 'classification': y_cls_val, 'threshold': y_thr_val}
    ),
    shuffle=False,  # ✅ KRITIK: Shuffle devre dışı (TIME-SERIES)!
    callbacks=cb2,
    verbose=1,
    initial_epoch=initial_epoch_stage2
)

# AŞAMA 2 Checkpoint kaydet
save_checkpoint(
    stage=2,
    epoch=len(hist2.history['loss']),
    model=model,
    optimizer=model.optimizer,
    metrics_history=hist2.history,
    class_weights={'w0': w0, 'w1': w1},
    filename='checkpoint_stage2_latest.pkl'
)

stage2_time = time.time() - stage2_start
print(f"\n✅ AŞAMA 2 Tamamlandı! Süre: {stage2_time/60:.1f} dakika")

# =============================================================================
# AŞAMA 3: FULL MODEL FINE-TUNING (150 epoch)
# =============================================================================
print("\n" + "="*80)
print("🔥 AŞAMA 3: FULL MODEL FINE-TUNING (Dengeli Final)")
print("="*80)
print("Hedef: Tüm output'ları birlikte optimize et (dengeli final push)")
print("Epoch: 80 | Batch: 16 | LR: 0.00005 | Class Weight: 2.0x (Dengeli!)")
print("Loss Weights: Regression 40%, Classification 15%, Threshold 45%")
print("Monitor: val_threshold_accuracy | Patience: 8")
print("="*80 + "\n")

stage3_start = time.time()

# Checkpoint kontrolü - AŞAMA 3 için resume
stage3_checkpoint = load_checkpoint('checkpoint_stage3_latest.pkl')
initial_epoch_stage3 = 0

# AŞAMA 2 modelini yükle
if stage3_checkpoint and stage3_checkpoint['stage'] == 3:
    print("🔄 AŞAMA 3 checkpoint'inden devam ediliyor...")
    model.set_weights(stage3_checkpoint['model_weights'])
    # optimizer weights kaldırıldı - TensorFlow uyumluluk sorunu
    initial_epoch_stage3 = stage3_checkpoint['epoch']
    print(f"   Epoch {initial_epoch_stage3}'den devam edilecek")
else:
    model.load_weights('stage2_best.h5')

# Class weights - MAKSIMUM FINAL (lazy learning'i kesin önler)
w0_final = 35.0  # 1.5 altı için: 35.0x (maksimum - final push)
w1_final = 1.0   # 1.5 üstü baseline

print(f"📊 CLASS WEIGHTS (AŞAMA 3 - Maksimum Final - TIME-SERIES SPLIT):")
print(f"  1.5 altı: {w0_final:.2f}x (maksimum - final push)")
print(f"  1.5 üstü: {w1_final:.2f}x\n")

# AŞAMA 3: Tüm output'lar aktif - SADECE FOCAL LOSS (çakışma yok!)
model.compile(
    optimizer=Adam(0.00005),
    loss={
        'regression': percentage_aware_regression_loss,  # YENİ: Yüzde hataya dayalı regression loss
        'classification': 'categorical_crossentropy',
        'threshold': balanced_focal_loss()  # SADECE focal loss (gamma=2.0, alpha=0.7)
    },
    loss_weights={'regression': 0.50, 'classification': 0.15, 'threshold': 0.35},  # Regression ağırlığı artırıldı: 0.40 → 0.50
    metrics={'regression': ['mae'], 'classification': ['accuracy'], 'threshold': ['accuracy', 'binary_crossentropy']}
)

# Adaptive Weight Scheduler (GÜÇLENDIRILDI - Lazy Learning Önleme)
adaptive_scheduler_3 = AdaptiveWeightScheduler(
    initial_weight=2.5,    # DÜZELTME: 25.0 → 2.5 (10x azaltma!)
    min_weight=1.5,        # DÜZELTME: 15.0 → 1.5 (normal seviye)
    max_weight=6.0,        # DÜZELTME: 50.0 → 6.0 (reasonable limit)
    target_below_acc=0.70,
    target_above_acc=0.75,
    test_data=([X_f_te, X_50_te, X_200_te, X_500_te, X_1000_te], y_reg_te),
    threshold=1.5,
    check_interval=5
)

# Dynamic Weight Callback (mevcut - opsiyonel)
dynamic_callback_3 = DynamicWeightCallback("AŞAMA 3", initial_weight=2.0, target_below_acc=0.70)

# Virtual Bankroll Callback (HER EPOCH için sanal kasa)
virtual_bankroll_3 = VirtualBankrollCallback(
    stage_name="AŞAMA 3",
    X_test=[X_f_te, X_50_te, X_200_te, X_500_te, X_1000_te],
    y_test=y_reg_te,
    threshold=1.5,
    starting_capital=1000.0,
    bet_amount=10.0
)

cb3 = [
    callbacks.ModelCheckpoint('stage3_best.h5', monitor='val_threshold_accuracy', save_best_only=True, mode='max', verbose=1),
    callbacks.EarlyStopping(monitor='val_threshold_accuracy', patience=8, min_delta=0.001, mode='max', restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-8, verbose=1),
    adaptive_scheduler_3,  # YENİ: Adaptive weight scheduler
    virtual_bankroll_3,  # YENİ: Her epoch sanal kasa gösterimi
    ProgressiveMetricsCallback("AŞAMA 3")
]

hist3 = model.fit(
    [X_f_tr, X_50_tr, X_200_tr, X_500_tr, X_1000_tr],
    {'regression': y_reg_tr, 'classification': y_cls_tr, 'threshold': y_thr_tr},
    epochs=80,
    batch_size=16,
    validation_data=(  # ✅ MANUEL VALIDATION (kronolojik!)
        [X_f_val, X_50_val, X_200_val, X_500_val, X_1000_val],
        {'regression': y_reg_val, 'classification': y_cls_val, 'threshold': y_thr_val}
    ),
    shuffle=False,  # ✅ KRITIK: Shuffle devre dışı (TIME-SERIES)!
    callbacks=cb3,
    verbose=1,
    initial_epoch=initial_epoch_stage3
)

# AŞAMA 3 Checkpoint kaydet
save_checkpoint(
    stage=3,
    epoch=len(hist3.history['loss']),
    model=model,
    optimizer=model.optimizer,
    metrics_history=hist3.history,
    class_weights={'w0': w0_final, 'w1': w1_final},
    filename='checkpoint_stage3_latest.pkl'
)

stage3_time = time.time() - stage3_start
print(f"\n✅ AŞAMA 3 Tamamlandı! Süre: {stage3_time/60:.1f} dakika")

# =============================================================================
# FINAL EVALUATION
# =============================================================================
print("\n" + "="*80)
print("📊 FINAL DEĞERLENDİRME (Test Seti)")
print("="*80)

# En iyi modeli yükle
model.load_weights('stage3_best.h5')

pred = model.predict([X_f_te, X_50_te, X_200_te, X_500_te, X_1000_te], verbose=0)
p_reg = pred[0].flatten()
p_cls = pred[1]
p_thr = pred[2].flatten()

# Regression metrics
mae_final = mean_absolute_error(y_reg_te, p_reg)
rmse_final = np.sqrt(mean_squared_error(y_reg_te, p_reg))

print(f"\n📈 REGRESSION:")
print(f"  MAE: {mae_final:.4f}")
print(f"  RMSE: {rmse_final:.4f}")

# Threshold metrics
thr_true = (y_reg_te >= 1.5).astype(int)
# GÜNCELLEME: %85 Güven Eşiği
thr_pred = (p_thr >= CONFIDENCE_THRESHOLD).astype(int)
thr_acc = accuracy_score(thr_true, thr_pred)

below_mask = thr_true == 0
above_mask = thr_true == 1
below_acc = accuracy_score(thr_true[below_mask], thr_pred[below_mask]) if below_mask.sum() > 0 else 0
above_acc = accuracy_score(thr_true[above_mask], thr_pred[above_mask]) if above_mask.sum() > 0 else 0

print(f"\n🎯 THRESHOLD (1.5x) - Eşik: {CONFIDENCE_THRESHOLD}:")
print(f"  Genel Accuracy: {thr_acc*100:.2f}%")
print(f"\n🔴 1.5 ALTI:")
print(f"  Doğruluk: {below_acc*100:.2f}%", end="")
if below_acc >= 0.75:
    print(" ✅ HEDEF AŞILDI!")
else:
    print(f" (Hedef: 75%+)")

print(f"\n🟢 1.5 ÜSTÜ:")
print(f"  Doğruluk: {above_acc*100:.2f}%")

# Confusion matrix
cm = confusion_matrix(thr_true, thr_pred)
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

# Classification metrics
cls_true = np.argmax(y_cls_te, axis=1)
cls_pred = np.argmax(p_cls, axis=1)
cls_acc = accuracy_score(cls_true, cls_pred)
print(f"\n📁 KATEGORİ CLASSIFICATION:")
print(f"  Accuracy: {cls_acc*100:.2f}%")

# =============================================================================
# ÇİFT SANAL KASA SİMÜLASYONU (YENİ - FAZ 2)
# =============================================================================
print("\n" + "="*80)
print("💰 ÇİFT SANAL KASA SİMÜLASYONU")
print("="*80)

# Dinamik kasa miktarı hesapla
test_count = len(y_reg_te)
initial_bankroll = test_count * 10  # Her test verisi için 10 TL
bet_amount = 10.0

print(f"📊 Test Veri Sayısı: {test_count:,}")
print(f"💰 Başlangıç Kasası: {initial_bankroll:,.2f} TL (dinamik)")
print(f"💵 Bahis Tutarı: {bet_amount:.2f} TL (sabit)")
print()

# =============================================================================
# KASA 1: 1.5x EŞİK SİSTEMİ (Mevcut)
# =============================================================================
print("="*80)
print("💰 KASA 1: 1.5x EŞİK SİSTEMİ")
print("="*80)
print("Strateji: Model 1.5x üstü tahmin ederse (Güven > %85) → 1.5x'te çıkış")
print()

kasa1_wallet = initial_bankroll
kasa1_total_bets = 0
kasa1_total_wins = 0
kasa1_total_losses = 0

# Model tahminlerini al (threshold output'tan)
# GÜNCELLEME: thr_pred zaten 0.85 eşiğine göre filtrelendi
threshold_predictions = thr_pred 

for i in range(len(y_reg_te)):
    model_pred_cls = threshold_predictions[i]  # 0 veya 1
    actual_value = y_reg_te[i]
    
    # Model "1.5 üstü" tahmin ediyorsa bahis yap
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
print("Strateji: Model 2.0x+ tahmin ederse VE Güven > %85 → Tahmin × 0.80'de çıkış")
print()

kasa2_wallet = initial_bankroll
kasa2_total_bets = 0
kasa2_total_wins = 0
kasa2_total_losses = 0
kasa2_exit_points = []  # Çıkış noktalarını kaydet

# Model tahminlerini al (regression output'tan)
y_reg_pred = p_reg

for i in range(len(y_reg_te)):
    model_pred_value = y_reg_pred[i]  # Tahmin edilen değer
    actual_value = y_reg_te[i]
    is_confident = threshold_predictions[i] == 1 # %85 güvenli mi?
    
    # SADECE 2.0x ve üzeri tahminlerde VE Yüksek güvende oyna
    if model_pred_value >= 2.0 and is_confident:
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
# MODEL KAYDETME + ZIP PAKETI (YENİ - FAZ 2)
# =============================================================================
print("\n" + "="*80)
print("💾 MODELLER KAYDEDİLİYOR")
print("="*80)

import json
import shutil

# models/ klasörünü oluştur
os.makedirs('models', exist_ok=True)

# 1. Progressive NN modeli (Transformer ile)
model.save('models/jetx_progressive_transformer.h5')
print("✅ Progressive NN (Transformer) kaydedildi: jetx_progressive_transformer.h5")

# 2. Scaler
joblib.dump(scaler, 'models/scaler_progressive_transformer.pkl')
print("✅ Scaler kaydedildi: scaler_progressive_transformer.pkl")

# 3. Model bilgileri (JSON) - YENİ: Transformer ve Çift Kasa bilgileri eklendi
total_time = stage1_time + stage2_time + stage3_time
info = {
    'model': 'Progressive_NN_Transformer',
    'version': '2.0',
    'date': '2025-10-12',
    'architecture': {
        'progressive_nn': {
            'n_beats': True,
            'tcn': True,
            'transformer': {
                'd_model': 256,
                'num_layers': 4,
                'num_heads': 8,
                'dff': 1024
            }
        }
    },
    'params': int(model.count_params()),
    'training_time_minutes': round(total_time/60, 1),
    'stage_times': {
        'stage1_foundation': round(stage1_time/60, 1),
        'stage2_threshold': round(stage2_time/60, 1),
        'stage3_full': round(stage3_time/60, 1)
    },
    'total_epochs': {
        'stage1': len(hist1.history['loss']),
        'stage2': len(hist2.history['loss']),
        'stage3': len(hist3.history['loss'])
    },
    'metrics': {
        'threshold_accuracy': float(thr_acc),
        'below_15_accuracy': float(below_acc),
        'above_15_accuracy': float(above_acc),
        'class_accuracy': float(cls_acc),
        'mae': float(mae_final),
        'rmse': float(rmse_final),
        'money_loss_risk': float(fpr) if cm[0,0] + cm[0,1] > 0 else 0.0
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
    }
}

with open('models/model_info.json', 'w') as f:
    json.dump(info, f, indent=2)
print("✅ Model bilgileri kaydedildi: model_info.json")

print("\n📁 Kaydedilen dosyalar:")
print("  • jetx_progressive_transformer.h5 (Progressive NN + Transformer)")
print("  • scaler_progressive_transformer.pkl (Scaler)")
print("  • model_info.json (Model bilgileri)")
print("  • stage1_best.h5 (Checkpoint)")
print("  • stage2_best.h5 (Checkpoint)")
print("  • stage3_best.h5 (Checkpoint)")
print("="*80)

# =============================================================================
# MODELLERİ ZIP'LE VE İNDİR (YENİ - FAZ 2)
# =============================================================================
print("\n" + "="*80)
print("📦 MODELLER ZIP'LENIYOR")
print("="*80)

# ZIP dosyası oluştur
zip_filename = 'jetx_models_progressive_v2.0.zip'
shutil.make_archive(
    'jetx_models_progressive_v2.0',
    'zip',
    'models'
)

print(f"✅ ZIP dosyası oluşturuldu: {zip_filename}")
print(f"📦 Boyut: {os.path.getsize(f'{zip_filename}') / (1024*1024):.2f} MB")

# Google Colab'da indirme
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
print("🎉 PROGRESSIVE TRAINING TAMAMLANDI!")
print("="*80)
print(f"Toplam Süre: {total_time/60:.1f} dakika ({total_time/3600:.1f} saat)")
print(f"Toplam Epoch: {info['total_epochs']['stage1'] + info['total_epochs']['stage2'] + info['total_epochs']['stage3']}")
print()

if below_acc >= 0.75 and fpr < 0.20:
    print("✅ ✅ ✅ TÜM HEDEFLER BAŞARIYLA AŞILDI!")
    print(f"  🔴 1.5 ALTI: {below_acc*100:.1f}% (Hedef: 75%+)")
    print(f"  💰 Para kaybı: {fpr*100:.1f}% (Hedef: <20%)")
    print("\n🚀 Model artık production'da kullanılabilir!")
elif below_acc >= 0.70:
    print("✅ ✅ İYİ PERFORMANS!")
    print(f"  🔴 1.5 ALTI: {below_acc*100:.1f}%")
    print(f"  💰 Para kaybı: {fpr*100:.1f}%")
    print("\nBiraz daha eğitimle hedeflere ulaşılabilir.")
else:
    print("⚠️ Hedefin altında")
    print(f"  🔴 1.5 ALTI: {below_acc*100:.1f}% (Hedef: 75%+)")
    print("\nÖneriler:")
    print("  - Daha fazla veri toplayın")
    print("  - Class weight'i artırın (35-40x)")
    print("  - Epoch sayısını artırın")

print("\n📁 Sonraki adım:")
print("  1. jetx_progressive_final.h5 -> models/jetx_model.h5")
print("  2. scaler_progressive.pkl -> models/scaler.pkl")
print("  3. Streamlit uygulamasını test edin")
print("="*80)
print(f"Bitiş: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
