"""
RL Agent Training Script - SIMPLE VERSION

TensorFlow ve protobuf sorunlarından kaçınmak için 
basitleştirilmiş sürüm. Sadece temel numpy/pandas kullanır.
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

# Simple imports (TensorFlow olmadan)
from sklearn.preprocessing import StandardScaler
import joblib

# Project imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from category_definitions import FeatureEngineering
    from utils.all_models_predictor import AllModelsPredictor
    from utils.psychological_analyzer import PsychologicalAnalyzer
    from utils.anomaly_streak_detector import AnomalyStreakDetector
    from utils.risk_manager import RiskManager
    from utils.advanced_bankroll import AdvancedBankrollManager
    from utils.rl_agent import RLAgent
except ImportError as e:
    print(f"⚠️ Project import warning: {e}")
    print("Basit fallback kullanılacak")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


class SimpleRLAgent:
    """Basit RL Agent - TensorFlow olmadan"""
    
    def __init__(self, state_dim: int = 200, action_dim: int = 4, learning_rate: float = 0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # Simple neural network (numpy ile)
        self.weights1 = np.random.randn(state_dim, 128) * 0.1
        self.bias1 = np.zeros(128)
        self.weights2 = np.random.randn(128, 64) * 0.1
        self.bias2 = np.zeros(64)
        self.weights3 = np.random.randn(64, action_dim) * 0.1
        self.bias3 = np.zeros(action_dim)
        
        # Scaler
        self.scaler = StandardScaler()
        
        logger.info("✅ Simple RL Agent initialized")
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def forward(self, state):
        """Forward propagation"""
        # State scaling
        state_scaled = self.scaler.transform(state.reshape(1, -1))[0]
        
        # Layer 1
        z1 = np.dot(state_scaled, self.weights1) + self.bias1
        a1 = self.relu(z1)
        
        # Layer 2
        z2 = np.dot(a1, self.weights2) + self.bias2
        a2 = self.relu(z2)
        
        # Layer 3 (output)
        z3 = np.dot(a2, self.weights3) + self.bias3
        output = self.softmax(z3)
        
        return output, (state_scaled, a1, a2, z3)
    
    def predict(self, state):
        """Tahmin yap"""
        output, _ = self.forward(state)
        return np.argmax(output), output
    
    def train_step(self, state, action, reward):
        """Basit eğitim adımı"""
        output, cache = self.forward(state)
        state_scaled, a1, a2, z3 = cache
        
        # Simple policy gradient update
        target = output.copy()
        target[action] += self.learning_rate * reward
        target = self.softmax(target)
        
        # Basit weight update (gradient descent approximation)
        error = target - output
        
        # Update output layer
        self.weights3 += self.learning_rate * np.outer(a2, error)
        self.bias3 += self.learning_rate * error
        
        return output
    
    def fit_scaler(self, states):
        """Scaler'ı eğit"""
        self.scaler.fit(states)
    
    def save_model(self, model_path: str = 'models/simple_rl_agent_model.pkl'):
        """Modeli kaydet"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'weights1': self.weights1,
            'bias1': self.bias1,
            'weights2': self.weights2,
            'bias2': self.bias2,
            'weights3': self.weights3,
            'bias3': self.bias3,
            'scaler': self.scaler,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"✅ Model kaydedildi: {model_path}")
    
    def load_model(self, model_path: str):
        """Model yükle"""
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            self.weights1 = model_data['weights1']
            self.bias1 = model_data['bias1']
            self.weights2 = model_data['weights2']
            self.bias2 = model_data['bias2']
            self.weights3 = model_data['weights3']
            self.bias3 = model_data['bias3']
            self.scaler = model_data['scaler']
            self.state_dim = model_data['state_dim']
            self.action_dim = model_data['action_dim']
            logger.info(f"✅ Model yüklendi: {model_path}")
            return True
        return False


class SimpleRLAgentTrainer:
    """Basit RL Agent eğitici"""
    
    def __init__(self, state_dim: int = 200, action_dim: int = 4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent = SimpleRLAgent(state_dim, action_dim)
        logger.info("✅ SimpleRLAgentTrainer başlatıldı")
    
    def load_data(self, db_path: str = 'data/jetx_data.db', min_history: int = 100) -> np.ndarray:
        """Veritabanından veri yükle"""
        try:
            logger.info(f"Veri yükleniyor: {db_path}")
            
            # Database path kontrolü
            if not os.path.exists(db_path):
                alt_paths = [
                    'jetx_data.db',
                    '../data/jetx_data.db',
                    os.path.join(os.getcwd(), 'data', 'jetx_data.db')
                ]
                
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        db_path = alt_path
                        break
                else:
                    raise FileNotFoundError(f"Database not found: {db_path}")
            
            conn = sqlite3.connect(db_path)
            
            # Tablo kontrolü
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='jetx_results'")
            if not cursor.fetchone():
                # Tablo yoksa oluştur ve örnek veri ekle
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS jetx_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        value REAL NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Örnek veri ekle
                sample_data = [1.2, 1.8, 1.5, 2.1, 1.3, 1.6, 3.2, 1.1, 1.4, 1.9]
                for val in sample_data:
                    cursor.execute("INSERT INTO jetx_results (value) VALUES (?)", (val,))
                conn.commit()
                logger.info("⚠️ Sample data added to empty database")
            
            data = pd.read_sql_query("SELECT value FROM jetx_results ORDER BY id", conn)
            conn.close()
            
            values = data['value'].values
            logger.info(f"✅ {len(values):,} veri yüklendi")
            
            return values
            
        except Exception as e:
            logger.error(f"❌ Veri yükleme başarısız: {e}")
            raise
    
    def create_simple_state(self, history: List[float], window_size: int = 50) -> np.ndarray:
        """Basit state vector oluştur"""
        if len(history) < window_size:
            # Pad with mean value
            mean_val = np.mean(history) if history else 1.5
            history = [mean_val] * (window_size - len(history)) + history
        
        # Son window_size kadar al
        recent_history = history[-window_size:]
        
        # Basit özellikler
        features = []
        
        # 1. Raw values (normalized)
        mean_val = np.mean(recent_history)
        normalized_values = [(x - mean_val) / mean_val for x in recent_history]
        features.extend(normalized_values[:50])  # Max 50 değer
        
        # 2. İstatistiksel özellikler
        features.extend([
            np.mean(recent_history),
            np.std(recent_history),
            np.min(recent_history),
            np.max(recent_history),
            np.median(recent_history),
            len([x for x in recent_history if x < 1.5]) / len(recent_history),
            len([x for x in recent_history if x >= 2.0]) / len(recent_history),
            len([x for x in recent_history if x >= 3.0]) / len(recent_history)
        ])
        
        # 3. Hareketli ortalamalar
        if len(recent_history) >= 10:
            features.extend([
                np.mean(recent_history[-5:]),
                np.mean(recent_history[-10:]),
                np.mean(recent_history[-20:]) if len(recent_history) >= 20 else np.mean(recent_history)
            ])
        else:
            features.extend([mean_val] * 3)
        
        # 4. Trend özellikleri
        if len(recent_history) >= 5:
            recent_changes = np.diff(recent_history[-5:])
            features.extend([
                np.mean(recent_changes),
                np.std(recent_changes)
            ])
        else:
            features.extend([0.0, 0.0])
        
        # 5. Sabit değerlerle 200'e tamamla
        while len(features) < self.state_dim:
            features.append(0.0)
        
        return np.array(features[:self.state_dim])
    
    def calculate_reward(self, action: int, actual_value: float) -> float:
        """Basit reward hesapla"""
        if action == 0:  # BEKLE
            return 0.1 if actual_value < 1.5 else -0.05
        else:  # BAHIS YAP
            if actual_value >= 1.5:
                if action == 1:  # Konservatif
                    return 0.2
                elif action == 2:  # Normal
                    return 0.4 if actual_value >= 2.0 else 0.1
                else:  # Agresif
                    return 0.6 if actual_value >= 3.0 else 0.2
            else:
                return -1.0  # Kayıp cezası
    
    def prepare_training_data(self, values: np.ndarray, window_size: int = 50) -> Tuple[List, List, List]:
        """Eğitim verisi hazırla"""
        logger.info("Eğitim verisi hazırlanıyor...")
        
        states = []
        actions = []
        rewards = []
        
        # Basit örnekler oluştur
        for i in range(window_size, len(values)):
            try:
                # State
                history = values[:i].tolist()
                state = self.create_simple_state(history, window_size)
                
                # Basit optimal action (rule-based)
                recent_avg = np.mean(values[i-10:i]) if i >= 10 else np.mean(values[:i])
                
                if recent_avg < 1.3:
                    optimal_action = 0  # BEKLE
                elif recent_avg < 1.8:
                    optimal_action = 1  # Konservatif
                elif recent_avg < 2.5:
                    optimal_action = 2  # Normal
                else:
                    optimal_action = 3  # Agresif
                
                # Reward
                reward = self.calculate_reward(optimal_action, values[i])
                
                states.append(state)
                actions.append(optimal_action)
                rewards.append(reward)
                
            except Exception as e:
                logger.warning(f"Örnek {i} atlandı: {e}")
                continue
        
        logger.info(f"✅ {len(states)} örnek hazırlandı")
        return states, actions, rewards
    
    def train(self, states: List, actions: List, rewards: List, epochs: int = 10):
        """Model eğit"""
        logger.info(f"Model eğitiliyor ({epochs} epochs)...")
        
        try:
            # Scaler eğit
            states_array = np.array(states)
            self.agent.fit_scaler(states_array)
            
            # Training loop
            for epoch in range(epochs):
                total_loss = 0
                correct_predictions = 0
                
                # Shuffle data
                indices = np.random.permutation(len(states))
                
                for idx in indices:
                    state = states[idx]
                    action = actions[idx]
                    reward = rewards[idx]
                    
                    # Forward pass
                    output = self.agent.train_step(state, action, reward)
                    
                    # Calculate accuracy
                    predicted_action = np.argmax(output)
                    if predicted_action == action:
                        correct_predictions += 1
                    
                    total_loss += -np.log(output[action] + 1e-8)
                
                avg_loss = total_loss / len(states)
                accuracy = correct_predictions / len(states)
                
                logger.info(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2%}")
            
            logger.info("✅ Model eğitimi tamamlandı")
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            raise
    
    def evaluate(self, states: List, actions: List) -> Dict:
        """Model değerlendir"""
        try:
            correct_predictions = 0
            action_dist = [0] * self.action_dim
            
            for state, true_action in zip(states, actions):
                predicted_action, _ = self.agent.predict(state)
                action_dist[predicted_action] += 1
                
                if predicted_action == true_action:
                    correct_predictions += 1
            
            accuracy = correct_predictions / len(states)
            
            return {
                'accuracy': float(accuracy),
                'action_distribution': action_dist,
                'total_samples': len(states)
            }
            
        except Exception as e:
            logger.error(f"❌ Model evaluation failed: {e}")
            return {'error': str(e)}


def main():
    """Ana eğitim fonksiyonu"""
    print("="*80)
    print("🤖 SIMPLE RL AGENT TRAINING")
    print("="*80)
    
    try:
        # Trainer
        trainer = SimpleRLAgentTrainer(
            state_dim=200,
            action_dim=4
        )
        
        # Veri yükle
        print("\n📊 Veri yükleniyor...")
        values = trainer.load_data(db_path='data/jetx_data.db', min_history=50)
        
        # Eğitim verisi hazırla
        print("\n🔧 Eğitim verisi hazırlanıyor...")
        states, actions, rewards = trainer.prepare_training_data(
            values=values,
            window_size=min(50, len(values)-1)
        )
        
        if len(states) < 10:
            raise ValueError(f"Çok az eğitim verisi: {len(states)}")
        
        # Eğit
        print("\n🚀 Model eğitimi başlatılıyor...")
        trainer.train(
            states=states,
            actions=actions,
            rewards=rewards,
            epochs=20
        )
        
        # Değerlendir
        print("\n📈 Model değerlendiriliyor...")
        eval_results = trainer.evaluate(states, actions)
        
        if 'error' not in eval_results:
            print(f"✅ Doğruluk: {eval_results['accuracy']:.2%}")
            print(f"   Action dağılımı: {eval_results['action_distribution']}")
            print(f"   Toplam örnek: {eval_results['total_samples']}")
        else:
            print(f"❌ Değerlendirme başarısız: {eval_results['error']}")
        
        # Kaydet
        print("\n💾 Model kaydediliyor...")
        trainer.agent.save_model('models/simple_rl_agent_model.pkl')
        
        # Model info
        model_info = {
            'model': 'Simple_RL_Agent',
            'version': '1.0',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'state_dim': trainer.state_dim,
            'action_dim': trainer.action_dim,
            'training_samples': len(states),
            'epochs': 20,
            'final_accuracy': eval_results.get('accuracy', 0.0),
            'action_distribution': eval_results.get('action_distribution', [0, 0, 0, 0])
        }
        
        with open('models/simple_rl_agent_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print("\n" + "="*80)
        print("✅ SIMPLE RL AGENT EĞİTİMİ BAŞARIYLA TAMAMLANDI!")
        print("="*80)
        print(f"📊 Model: {model_info['model']} v{model_info['version']}")
        print(f"🎯 Doğruluk: {model_info['final_accuracy']:.2%}")
        print(f"📝 Eğitim örnekleri: {model_info['training_samples']:,}")
        print(f"💾 Model dosyası: models/simple_rl_agent_model.pkl")
        
    except Exception as e:
        print(f"\n❌ KRİTİK HATA: {e}")
        logger.error(f"Training failed: {e}")
        
        # Debug bilgisi
        print("\n🔍 Debug bilgileri:")
        print(f"   Python version: {sys.version}")
        print(f"   Working directory: {os.getcwd()}")
        
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
