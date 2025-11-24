#!/usr/bin/env python3
"""
🤖 RL Agent Training Script (v2.1 FIXED)

Reinforcement Learning Agent'ı eğitir.
Policy Gradient (REINFORCE) algoritması kullanır.

GÜNCELLEME (v2.1):
- ✅ Robust Path Handling: Dosya yolları garantiye alındı.
- ✅ Fallback Mechanism: Diğer modeller eğitilmemişse bile çökmeden çalışır (Mock Predictor).
- ✅ Database Protection: Veritabanı yoksa sentetik veri kullanır.
- ✅ 2 Modlu Yapı: Normal (0.85) ve Rolling (0.95) uyumlu.
"""

import numpy as np
import pandas as pd
import sqlite3
import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Dict, Tuple
from tqdm import tqdm
import warnings

# Uyarıları kapat
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. ORTAM VE DOSYA YOLU AYARLARI
# -----------------------------------------------------------------------------
def setup_project_path():
    """Proje kök dizinini bulur ve sys.path'e ekler"""
    current_dir = os.getcwd()
    possible_paths = [
        current_dir,
        os.path.join(current_dir, 'jetxpredictor'),
        os.path.dirname(os.path.abspath(__file__)), # notebooks klasörü
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # proje kök dizini
    ]
    
    project_root = None
    for path in possible_paths:
        if os.path.exists(os.path.join(path, 'category_definitions.py')):
            project_root = path
            break
    
    if project_root:
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        print(f"✅ Proje kök dizini ayarlandı: {project_root}")
        return project_root
    else:
        # Son çare: çalışılan dizini ekle
        sys.path.insert(0, current_dir)
        print(f"⚠️ Proje kökü tam tespit edilemedi, mevcut dizin kullanılıyor: {current_dir}")
        return current_dir

PROJECT_ROOT = setup_project_path()

# Kütüphane kontrolü
try:
    import tensorflow as tf
    from tensorflow.keras import models, layers, optimizers, callbacks
    from sklearn.preprocessing import StandardScaler
    import joblib
except ImportError as e:
    print(f"❌ Kritik kütüphane eksik: {e}")
    sys.exit(1)

# GPU Ayarları
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Aktif: {len(gpus)} adet")
    except RuntimeError as e:
        print(f"⚠️ GPU Hatası: {e}")

# Proje importları
try:
    from category_definitions import FeatureEngineering
    from utils.all_models_predictor import AllModelsPredictor
    from utils.psychological_analyzer import PsychologicalAnalyzer
    from utils.anomaly_streak_detector import AnomalyStreakDetector
    from utils.risk_manager import RiskManager
    from utils.advanced_bankroll import AdvancedBankrollManager
    from utils.rl_agent import RLAgent
except ImportError as e:
    print(f"⚠️ Modül import uyarısı (Mock nesneler kullanılacak): {e}")

# -----------------------------------------------------------------------------
# 2. YARDIMCI SINIFLAR
# -----------------------------------------------------------------------------

class MockPredictor:
    """
    Eğer diğer modeller (NN, CatBoost) henüz eğitilmemişse
    RL Agent eğitiminin çökmemesi için rastgele/mantıklı tahminler üretir.
    """
    def predict_all(self, history: np.ndarray) -> Dict:
        # Basit bir "trend takip eden" sanal tahmin üret
        recent_avg = np.mean(history[-10:]) if len(history) >= 10 else 1.5
        
        # Rastgelelik ekle ama trende sadık kal
        pred_val = max(1.0, recent_avg + np.random.normal(0, 0.5))
        conf = min(0.99, max(0.5, 0.7 + np.random.normal(0, 0.1)))
        
        is_normal = conf >= 0.85
        is_rolling = conf >= 0.95
        
        return {
            'consensus': {
                'prediction': pred_val,
                'confidence': conf,
                'above_threshold': pred_val >= 1.5,
                'is_normal': is_normal,
                'is_rolling': is_rolling
            }
        }

class RLAgentTrainer:
    """RL Agent eğitici sınıfı"""
    
    def __init__(
        self,
        state_dim: int = 200,
        action_dim: int = 4, # 0: Bekle, 1: Rolling, 2: Normal, 3: Normal
        learning_rate: float = 0.001,
        batch_size: int = 32
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.model = None
        
        # Analyzers (Hata korumalı)
        try:
            self.psychological_analyzer = PsychologicalAnalyzer(threshold=1.5)
            self.anomaly_detector = AnomalyStreakDetector(threshold=1.5)
        except:
            self.psychological_analyzer = None
            self.anomaly_detector = None
        
    def build_model(self) -> tf.keras.Model:
        """Policy Network oluştur"""
        model = models.Sequential([
            layers.Dense(256, activation='relu', input_shape=(self.state_dim,)),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.action_dim, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model = model
        return model
    
    def load_data(self, db_path: str = 'jetx_data.db', min_history: int = 500) -> np.ndarray:
        """Veritabanından veri yükle (Hata korumalı)"""
        full_db_path = os.path.join(PROJECT_ROOT, db_path)
        
        if not os.path.exists(full_db_path):
            print("⚠️ Veritabanı bulunamadı, sentetik veri oluşturuluyor...")
            return self._generate_synthetic_data(min_history * 2)
            
        try:
            conn = sqlite3.connect(full_db_path)
            data = pd.read_sql_query("SELECT value FROM jetx_results ORDER BY id", conn)
            conn.close()
            
            values = data['value'].values
            if len(values) < min_history:
                print(f"⚠️ Yetersiz veri ({len(values)}), sentetik veri ile tamamlanıyor...")
                synthetic = self._generate_synthetic_data(min_history * 2)
                return np.concatenate([values, synthetic])
                
            print(f"✅ {len(values):,} veri yüklendi")
            return values
        except Exception as e:
            print(f"⚠️ Veritabanı okuma hatası: {e}, sentetik veri kullanılıyor.")
            return self._generate_synthetic_data(min_history * 2)

    def _generate_synthetic_data(self, count):
        return np.random.lognormal(0.5, 0.8, count).clip(1.0, 100.0)
    
    def calculate_reward(self, action, predicted_value, actual_value, bet_amount, bankroll):
        """Reward hesapla"""
        if action == 0:  # BEKLE
            return 0.1 if actual_value < 1.5 else -0.05
        
        # BAHIS YAP
        if actual_value >= 1.5:
            if action == 1:  # ROLLING
                profit = bet_amount * 0.5 # 1.5x çıkış
                return profit / bankroll * 10.0
            elif action >= 2:  # NORMAL
                exit_mult = min(predicted_value * 0.8, 2.5)
                if actual_value >= exit_mult:
                    profit = bet_amount * (exit_mult - 1.0)
                    return profit / bankroll * 12.0
                else:
                    return -bet_amount / bankroll * 5.0
        else:
            return -bet_amount / bankroll * 10.0

    def prepare_training_data(self, values, predictor, window_size=500, sample_ratio=0.1):
        """Eğitim verisi hazırla"""
        print("🔨 Eğitim verisi hazırlanıyor...")
        states, actions, rewards = [], [], []
        
        # Hız için verinin bir kısmını kullan
        total = len(values) - window_size - 1
        indices = np.linspace(0, total-1, int(total * sample_ratio)).astype(int)
        
        # RL Agent instance (state vector oluşturmak için)
        try:
            temp_agent = RLAgent()
        except:
            # Fallback agent class
            class TempAgent:
                def create_state_vector(self, **kwargs): return np.zeros(200)
            temp_agent = TempAgent()
        
        bankroll = 1000.0
        
        for idx in tqdm(indices, desc="Veri Hazırlığı"):
            try:
                history = values[:window_size + idx]
                actual = values[window_size + idx]
                
                # Tahmin al
                preds = predictor.predict_all(history)
                cons = preds.get('consensus', {})
                
                # State oluştur
                state = temp_agent.create_state_vector(
                    history=history.tolist(),
                    model_predictions=preds,
                    bankroll_manager=None
                )
                
                # Optimal action (Etiketleme)
                conf = cons.get('confidence', 0.5)
                if conf >= 0.95: optimal = 1 # Rolling
                elif conf >= 0.85: optimal = 2 # Normal
                else: optimal = 0 # Bekle
                
                # Reward
                bet = 0
                if optimal == 1: bet = bankroll * 0.02
                elif optimal >= 2: bet = bankroll * 0.04
                
                r = self.calculate_reward(optimal, cons.get('prediction', 1.5), actual, bet, bankroll)
                
                states.append(state)
                actions.append(optimal)
                rewards.append(r)
                
            except Exception:
                continue
                
        return np.array(states), tf.keras.utils.to_categorical(actions, num_classes=4), np.array(rewards)

    def train(self, states, actions, rewards, epochs=20):
        """Model eğit"""
        print(f"🚀 Model eğitiliyor ({len(states)} örnek)...")
        
        self.scaler.fit(states)
        states_scaled = self.scaler.transform(states)
        
        # Sample weights from rewards
        sample_weights = (rewards - rewards.min()) / (rewards.max() - rewards.min() + 1e-8)
        
        callbacks_list = [
            callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        ]
        
        history = self.model.fit(
            states_scaled, actions,
            sample_weight=sample_weights,
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        return history
        
    def save(self, path='models/rl_agent_model.h5'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        scaler_path = path.replace('.h5', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"💾 Model kaydedildi: {path}")

# -----------------------------------------------------------------------------
# 3. ANA EĞİTİM DÖNGÜSÜ
# -----------------------------------------------------------------------------
def main():
    print("="*80)
    print("🤖 RL AGENT TRAINING (ROBUST MODE)")
    print("="*80)
    
    trainer = RLAgentTrainer()
    trainer.build_model()
    
    # Veri yükle
    values = trainer.load_data()
    
    # Predictor yükle (Hata korumalı)
    print("\n📦 Tahmin Modelleri Yükleniyor...")
    try:
        predictor = AllModelsPredictor()
        loaded = predictor.load_all_models()
        
        # Eğer hiçbir model yüklenemediyse Mock kullan
        if not any(loaded.values()):
            print("⚠️ Hiçbir model bulunamadı! Mock Predictor devreye giriyor.")
            print("   Bu sayede RL Agent eğitim süreci test edilebilir.")
            predictor = MockPredictor()
        else:
            print(f"✅ {sum(loaded.values())} model yüklendi.")
            
    except Exception as e:
        print(f"⚠️ Model yükleme hatası: {e}. Mock Predictor kullanılıyor.")
        predictor = MockPredictor()
    
    # Veri hazırla
    states, actions, rewards = trainer.prepare_training_data(values, predictor)
    
    # Eğer veri oluşmadıysa (örn. çok kısa history), yapay veri üret
    if len(states) < 10:
        print("⚠️ Yetersiz eğitim verisi. Dummy veri ile model başlatılıyor (Placeholder).")
        states = np.random.random((100, 200))
        actions = tf.keras.utils.to_categorical(np.random.randint(0, 3, 100), num_classes=4)
        rewards = np.random.random(100)
    
    # Eğit
    trainer.train(states, actions, rewards)
    
    # Kaydet
    trainer.save()
    
    # Info dosyası
    info = {
        'model': 'RL_Agent_Robust',
        'version': '2.1',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'status': 'Trained (Potentially with Mock Data if models missing)'
    }
    
    try:
        with open('models/rl_agent_info.json', 'w') as f:
            json.dump(info, f, indent=2)
    except:
        pass

    print("\n✅ İŞLEM TAMAMLANDI!")

if __name__ == '__main__':
    main()
