#!/usr/bin/env python3
"""
🎯 JetX PROGRESSIVE TRAINING - 3 Aşamalı Eğitim Stratejisi

AMAÇ: 1.5 altı değerleri tahmin edebilen model eğitmek

STRATEJI:
├── AŞAMA 1: Foundation Training (100 epoch) - Threshold baştan aktif
├── AŞAMA 2: Threshold Fine-Tuning (80 epoch) - Yumuşak class weights (5x)
└── AŞAMA 3: Full Model Fine-Tuning (80 epoch) - Dengeli final (7x)

HEDEFLER:
- 1.5 ALTI Doğruluk: %70-80%+
- 1.5 ÜSTÜ Doğruluk: %75-85%+
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
print("🎯 JetX PROGRESSIVE TRAINING - 3 Aşamalı Eğitim")
print("="*80)
print(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Kütüphaneleri yükle
print("📦 Kütüphaneler yükleniyor...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                      "tensorflow", "scikit-learn", "pandas", "numpy", 
                      "scipy", "joblib", "matplotlib", "seaborn", "tqdm"])

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
warnings.filterwarnings('ignore')

print(f"✅ TensorFlow: {tf.__version__}")
print(f"✅ GPU: {'Mevcut' if len(tf.config.list_physical_devices('GPU')) > 0 else 'Yok (CPU)'}")

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
# FEATURE ENGINEERING
# =============================================================================
print("\n🔧 Feature extraction...")
window_size = 500
X_f, X_50, X_200, X_500 = [], [], [], []
y_reg, y_cls, y_thr = [], [], []

for i in tqdm(range(window_size, len(all_values)-1), desc='Features'):
    hist = all_values[:i].tolist()
    target = all_values[i]
    
    feats = FeatureEngineering.extract_all_features(hist)
    X_f.append(list(feats.values()))
    X_50.append(all_values[i-50:i])
    X_200.append(all_values[i-200:i])
    X_500.append(all_values[i-500:i])
    
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

# Train/Test split
idx = np.arange(len(X_f))
tr_idx, te_idx = train_test_split(idx, test_size=0.2, shuffle=False)

X_f_tr, X_50_tr, X_200_tr, X_500_tr = X_f[tr_idx], X_50[tr_idx], X_200[tr_idx], X_500[tr_idx]
y_reg_tr, y_cls_tr, y_thr_tr = y_reg[tr_idx], y_cls[tr_idx], y_thr[tr_idx]

X_f_te, X_50_te, X_200_te, X_500_te = X_f[te_idx], X_50[te_idx], X_200[te_idx], X_500[te_idx]
y_reg_te, y_cls_te, y_thr_te = y_reg[te_idx], y_cls[te_idx], y_thr[te_idx]

print(f"✅ Train: {len(X_f_tr):,}, Test: {len(X_f_te):,}")

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
    ~8-10M parametre
    """
    inp_f = layers.Input((n_features,), name='features')
    inp_50 = layers.Input((50, 1), name='seq50')
    inp_200 = layers.Input((200, 1), name='seq200')
    inp_500 = layers.Input((500, 1), name='seq500')
    
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
    
    nb_all = layers.Concatenate()([nb_s, nb_m, nb_l])
    
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
    
    # Fusion (Optimize)
    fus = layers.Concatenate()([inp_f, nb_all, tcn])
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
    
    model = models.Model([inp_f, inp_50, inp_200, inp_500], [out_reg, out_cls, out_thr])
    return model

# =============================================================================
# METRICS CALLBACK
# =============================================================================
class ProgressiveMetricsCallback(callbacks.Callback):
    def __init__(self, stage_name):
        super().__init__()
        self.stage_name = stage_name
        self.best_below_acc = 0
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            # Test seti üzerinde threshold metrics
            p = self.model.predict([X_f_te, X_50_te, X_200_te, X_500_te], verbose=0)[2].flatten()
            p_thr = (p >= 0.5).astype(int)
            t_thr = (y_reg_te >= 1.5).astype(int)
            
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
            print(f"\n🎯 MODEL DURUMU:")
            if below_acc >= 0.75 and above_acc >= 0.75 and risk < 0.20:
                print(f"   ✅ ✅ ✅ MÜKEMMEL! Model kullanıma hazır!")
            elif below_acc >= 0.60 and risk < 0.30:
                print(f"   ✅ İYİ - Biraz daha eğitimle hedeflere ulaşılabilir")
            elif below_acc == 0.0 or below_acc == 1.0:
                print(f"   ❌ KÖTÜ! Model bir tarafa KILITLENIYOR!")
                print(f"      → Model dengesiz öğreniyor, class weight ayarlanmalı")
            else:
                print(f"   ⚠️ ORTA - Devam ediyor...")
            
            # Dengesizlik Uyarısı
            if below_acc == 0.0 and above_acc > 0.95:
                print(f"\n⚠️ UYARI: Model sadece '1.5 üstü' tahmin ediyor!")
                print(f"   → Class weight çok DÜŞÜK veya model 'lazy learning' yapıyor")
                print(f"   → Öneri: Class weight'i artırın (5x → 7x)")
            elif below_acc > 0.95 and above_acc == 0.0:
                print(f"\n⚠️ UYARI: Model sadece '1.5 altı' tahmin ediyor!")
                print(f"   → Class weight çok YÜKSEK!")
                print(f"   → Öneri: Class weight'i azaltın (örn: 25x → 5x)")
            elif abs(below_acc - above_acc) > 0.40:
                print(f"\n⚠️ UYARI: Model dengesiz! (Fark: %{abs(below_acc - above_acc)*100:.1f})")
                print(f"   → Bir sınıfa aşırı öğreniyor, diğerini ihmal ediyor")
            
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
                model_pred = p_thr[i]  # Model tahmini (1.5 üstü mü?)
                actual_value = y_reg_te[i]  # Gerçek değer
                
                # Model "1.5 üstü" diyorsa bahis yap
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
            
            # Rapor göster
            print(f"   Başlangıç: 1,000.00 TL")
            print(f"   Toplam Bahis: {total_bets} oyun × {bet_amount:.0f} TL = {total_bets * bet_amount:,.0f} TL")
            print(f"   Kazanılan: {total_wins} oyun × {win_amount:.0f} TL = {total_wins * win_amount:,.0f} TL")
            print(f"   Kaybedilen: {total_losses} oyun × {bet_amount:.0f} TL = {total_losses * bet_amount:,.0f} TL")
            print(f"   Kazanma Oranı: {win_rate:.1f}% ({total_wins}/{total_bets})")
            print(f"   {'─'*50}")
            print(f"   Final Kasa: {wallet:,.2f} TL ({profit_loss:+,.2f} TL) {wallet_emoji}")
            print(f"   ROI (Yatırım Getirisi): {roi:+.1f}%")
            
            print(f"\n{'='*70}\n")
            
            if below_acc > self.best_below_acc:
                self.best_below_acc = below_acc
                print(f"  ✨ YENİ REKOR! En iyi 1.5 altı: {below_acc*100:.1f}%\n")

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

stage1_start = time.time()

model = build_progressive_model(X_f.shape[1])
print(f"✅ Model: {model.count_params():,} parametre")

# Class weights - DENGELI BAŞLANGIÇ (lazy learning önleme)
w0_stage1 = 1.2  # 1.5 altı için: 1.2x (2.0 → 1.2, çok yumuşak)
w1_stage1 = 1.0  # 1.5 üstü baseline

print(f"📊 CLASS WEIGHTS (AŞAMA 1 - Çok Yumuşak Başlangıç):")
print(f"  1.5 altı: {w0_stage1:.2f}x (dengeli - lazy learning'i önler)")
print(f"  1.5 üstü: {w1_stage1:.2f}x\n")

# AŞAMA 1: Foundation Training - Threshold baştan weighted BCE ile aktif!
model.compile(
    optimizer=Adam(0.0001),
    loss={'regression': threshold_killer_loss, 'classification': 'categorical_crossentropy', 'threshold': create_weighted_binary_crossentropy(w0_stage1, w1_stage1)},
    loss_weights={'regression': 0.55, 'classification': 0.10, 'threshold': 0.35},  # Regression vurgusu
    metrics={'regression': ['mae'], 'classification': ['accuracy'], 'threshold': ['accuracy']}
)

cb1 = [
    callbacks.ModelCheckpoint('stage1_best.h5', monitor='val_threshold_accuracy', save_best_only=True, mode='max', verbose=1),
    callbacks.EarlyStopping(monitor='val_threshold_accuracy', patience=12, min_delta=0.001, mode='max', restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1),
    ProgressiveMetricsCallback("AŞAMA 1")
]

hist1 = model.fit(
    [X_f_tr, X_50_tr, X_200_tr, X_500_tr],
    {'regression': y_reg_tr, 'classification': y_cls_tr, 'threshold': y_thr_tr},
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=cb1,
    verbose=1
)

stage1_time = time.time() - stage1_start
print(f"\n✅ AŞAMA 1 Tamamlandı! Süre: {stage1_time/60:.1f} dakika")

# AŞAMA 1 Değerlendirme
pred1 = model.predict([X_f_te, X_50_te, X_200_te, X_500_te], verbose=0)
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

# AŞAMA 1 modelini yükle
model.load_weights('stage1_best.h5')

# Class weights - YUMUŞAK ARTIŞ
w0 = 1.5  # 1.5 altı için: 1.5x (3.5 → 1.5, hafif artış)
w1 = 1.0  # 1.5 üstü baseline

print(f"📊 CLASS WEIGHTS (AŞAMA 2 - Hafif Artış):")
print(f"  1.5 altı: {w0:.2f}x (yumuşak - model dengeyi korur)")
print(f"  1.5 üstü: {w1:.2f}x\n")

# AŞAMA 2: Regression + Threshold (weighted binary crossentropy ile)
model.compile(
    optimizer=Adam(0.0001),
    loss={'regression': threshold_killer_loss, 'classification': 'categorical_crossentropy', 'threshold': create_weighted_binary_crossentropy(w0, w1)},
    loss_weights={'regression': 0.45, 'classification': 0.10, 'threshold': 0.45},
    metrics={'regression': ['mae'], 'classification': ['accuracy'], 'threshold': ['accuracy', 'binary_crossentropy']}
)

cb2 = [
    callbacks.ModelCheckpoint('stage2_best.h5', monitor='val_threshold_accuracy', save_best_only=True, mode='max', verbose=1),
    callbacks.EarlyStopping(monitor='val_threshold_accuracy', patience=10, min_delta=0.001, mode='max', restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1),
    ProgressiveMetricsCallback("AŞAMA 2")
]

hist2 = model.fit(
    [X_f_tr, X_50_tr, X_200_tr, X_500_tr],
    {'regression': y_reg_tr, 'classification': y_cls_tr, 'threshold': y_thr_tr},
    epochs=80,
    batch_size=32,
    validation_split=0.2,
    callbacks=cb2,
    verbose=1
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

# AŞAMA 2 modelini yükle
model.load_weights('stage2_best.h5')

# Class weights - DENGELI FINAL
w0_final = 2.0  # 1.5 altı için: 2.0x (5.0 → 2.0, dengeli)
w1_final = 1.0  # 1.5 üstü baseline

print(f"📊 CLASS WEIGHTS (AŞAMA 3 - Dengeli Final):")
print(f"  1.5 altı: {w0_final:.2f}x (dengeli - fazla agresif değil)")
print(f"  1.5 üstü: {w1_final:.2f}x\n")

# AŞAMA 3: Tüm output'lar aktif (weighted binary crossentropy ile)
model.compile(
    optimizer=Adam(0.00005),
    loss={'regression': threshold_killer_loss, 'classification': 'categorical_crossentropy', 'threshold': create_weighted_binary_crossentropy(w0_final, w1_final)},
    loss_weights={'regression': 0.40, 'classification': 0.15, 'threshold': 0.45},
    metrics={'regression': ['mae'], 'classification': ['accuracy'], 'threshold': ['accuracy', 'binary_crossentropy']}
)

cb3 = [
    callbacks.ModelCheckpoint('stage3_best.h5', monitor='val_threshold_accuracy', save_best_only=True, mode='max', verbose=1),
    callbacks.EarlyStopping(monitor='val_threshold_accuracy', patience=8, min_delta=0.001, mode='max', restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-8, verbose=1),
    ProgressiveMetricsCallback("AŞAMA 3")
]

hist3 = model.fit(
    [X_f_tr, X_50_tr, X_200_tr, X_500_tr],
    {'regression': y_reg_tr, 'classification': y_cls_tr, 'threshold': y_thr_tr},
    epochs=80,
    batch_size=16,
    validation_split=0.2,
    callbacks=cb3,
    verbose=1
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

pred = model.predict([X_f_te, X_50_te, X_200_te, X_500_te], verbose=0)
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
thr_pred = (p_thr >= 0.5).astype(int)
thr_acc = accuracy_score(thr_true, thr_pred)

below_mask = thr_true == 0
above_mask = thr_true == 1
below_acc = accuracy_score(thr_true[below_mask], thr_pred[below_mask]) if below_mask.sum() > 0 else 0
above_acc = accuracy_score(thr_true[above_mask], thr_pred[above_mask]) if above_mask.sum() > 0 else 0

print(f"\n🎯 THRESHOLD (1.5x):")
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
# KAYDET & İNDİR
# =============================================================================
print("\n💾 Model ve dosyalar kaydediliyor...")

model.save('jetx_progressive_final.h5')
joblib.dump(scaler, 'scaler_progressive.pkl')

import json
total_time = stage1_time + stage2_time + stage3_time
info = {
    'model': 'PROGRESSIVE_TRAINING_3_STAGE',
    'version': '1.0_PROGRESSIVE',
    'params': int(model.count_params()),
    'training_time_minutes': round(total_time/60, 1),
    'stage_times': {
        'stage1_regression': round(stage1_time/60, 1),
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
    }
}

with open('progressive_model_info.json', 'w') as f:
    json.dump(info, f, indent=2)

print("✅ Dosyalar kaydedildi:")
print("  - jetx_progressive_final.h5")
print("  - scaler_progressive.pkl")
print("  - progressive_model_info.json")
print("  - stage1_best.h5 (checkpoint)")
print("  - stage2_best.h5 (checkpoint)")
print("  - stage3_best.h5 (checkpoint)")

print(f"\n📊 Model Bilgisi:")
print(json.dumps(info, indent=2))

# Google Colab'da indir
try:
    from google.colab import files
    files.download('jetx_progressive_final.h5')
    files.download('scaler_progressive.pkl')
    files.download('progressive_model_info.json')
    print("\n✅ Ana dosyalar indirildi!")
    print("Not: Checkpoint dosyaları (stage1-3_best.h5) çok büyük olabilir.")
except:
    print("\n⚠️ Colab dışında - dosyalar sadece kaydedildi")

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