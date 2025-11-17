"""
Imitation Learning Agent Training Script

Bu script, uzman modellerin (Consensus, TabNet vb.) kararlarını taklit eden bir 
Imitation Learning (Taklit Öğrenme) ajanı eğitir. Gerçek Reinforcement Learning 
değildir, uzman kurallarını sinir ağına öğretir.

Not: Bu bir Imitation Learning / Supervised Learning yaklaşımıdır, gerçek RL değildir.
Ajan, kendi politikasını keşfetmez, tanımlanan kuralları öğrenir.
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

# TensorFlow
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, callbacks
from sklearn.preprocessing import StandardScaler
import joblib

# Project imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from category_definitions import FeatureEngineering
from utils.all_models_predictor import AllModelsPredictor
from utils.psychological_analyzer import PsychologicalAnalyzer
from utils.anomaly_streak_detector import AnomalyStreakDetector
from utils.risk_manager import RiskManager
from utils.advanced_bankroll import AdvancedBankrollManager
from utils.rl_agent import RLAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImitationLearningTrainer:
    """Imitation Learning Agent eğitici sınıfı"""
    
    def __init__(
        self,
        state_dim: int = 200,
        action_dim: int = 4,
        learning_rate: float = 0.001,
        batch_size: int = 32
    ):
        """
        Args:
            state_dim: State vector boyutu
            action_dim: Action space boyutu (4)
            learning_rate: Öğrenme oranı
            batch_size: Batch boyutu
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Scaler
        self.scaler = StandardScaler()
        
        # Model
        self.model = None
        
        # Analyzers
        self.psychological_analyzer = PsychologicalAnalyzer(threshold=1.5)
        self.anomaly_detector = AnomalyStreakDetector(threshold=1.5)
        
        logger.info("ImitationLearningTrainer başlatıldı")
    
    def build_model(self) -> tf.keras.Model:
        """Policy Network oluştur - Uzman kararlarını taklit etmek için"""
        model = models.Sequential([
            layers.Dense(256, activation='relu', input_shape=(self.state_dim,)),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.action_dim, activation='softmax')  # Action probabilities
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info("Policy Network oluşturuldu")
        return model
    
    def load_data(self, db_path: str = 'jetx_data.db', min_history: int = 500) -> np.ndarray:
        """Veritabanından veri yükle"""
        logger.info(f"Veri yükleniyor: {db_path}")
        
        conn = sqlite3.connect(db_path)
        data = pd.read_sql_query("SELECT value FROM jetx_results ORDER BY id", conn)
        conn.close()
        
        values = data['value'].values
        logger.info(f"✅ {len(values):,} veri yüklendi")
        
        # Minimum history kontrolü
        if len(values) < min_history:
            raise ValueError(f"Yeterli veri yok! En az {min_history} veri gerekli.")
        
        return values
    
    def calculate_reward(
        self,
        action: int,
        predicted_value: float,
        actual_value: float,
        bet_amount: float,
        bankroll: float
    ) -> float:
        """
        Reward hesapla - Training sırasında sample weight için kullanılır
        
        Args:
            action: Seçilen action (0-3)
            predicted_value: Tahmin edilen değer
            actual_value: Gerçekleşen değer
            bet_amount: Bahis miktarı
            bankroll: Mevcut bankroll
            
        Returns:
            Reward değeri
        """
        if action == 0:  # BEKLE
            # Beklemek için küçük reward (risk yok)
            if actual_value < 1.5:
                return 0.1  # Doğru karar (kayıp olurdu)
            else:
                return -0.05  # Yanlış karar (kazanç kaçtı)
        
        # BAHIS YAP
        if actual_value >= 1.5:
            # Kazandık
            if action == 1:  # Konservatif
                exit_multiplier = 1.5
                profit = bet_amount * (exit_multiplier - 1.0)
                reward = profit / bankroll * 10.0  # Normalize
            elif action == 2:  # Normal
                exit_multiplier = min(predicted_value * 0.8, 2.5)
                if actual_value >= exit_multiplier:
                    profit = bet_amount * (exit_multiplier - 1.0)
                    reward = profit / bankroll * 10.0
                else:
                    # Çıkış noktasına ulaşamadık, kayıp
                    reward = -bet_amount / bankroll * 5.0
            else:  # Agresif
                exit_multiplier = min(predicted_value * 0.85, 5.0)
                if actual_value >= exit_multiplier:
                    profit = bet_amount * (exit_multiplier - 1.0)
                    reward = profit / bankroll * 15.0  # Daha yüksek reward
                else:
                    reward = -bet_amount / bankroll * 8.0  # Daha yüksek ceza
        else:
            # Kaybettik
            reward = -bet_amount / bankroll * 10.0  # Kayıp cezası
        
        return reward
    
    def prepare_training_data(
        self,
        values: np.ndarray,
        all_models_predictor: AllModelsPredictor,
        window_size: int = 500,
        sample_ratio: float = 0.1  # Verinin %10'unu kullan (hız için)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Eğitim verisi hazırla - Uzman modellerden optimal action'ları öğren
        
        Returns:
            (states, actions, rewards) tuple
        """
        logger.info("Eğitim verisi hazırlanıyor...")
        
        states = []
        actions = []
        rewards = []
        
        # Sample indices
        total_samples = len(values) - window_size - 1
        sample_count = int(total_samples * sample_ratio)
        sample_indices = np.random.choice(total_samples, sample_count, replace=False)
        sample_indices = np.sort(sample_indices)
        
        logger.info(f"Toplam {total_samples} örnek var, {sample_count} örnek kullanılacak")
        
        # Virtual bankroll
        virtual_bankroll = 1000.0
        bankroll_manager = AdvancedBankrollManager(
            initial_bankroll=virtual_bankroll,
            risk_tolerance='moderate'
        )
        
        # RL Agent (state vector oluşturmak için)
        rl_agent = RLAgent()
        
        # Progress bar
        pbar = tqdm(sample_indices, desc="Training data preparation")
        
        for idx in pbar:
            try:
                # History
                history = values[:window_size + idx].tolist()
                actual_value = values[window_size + idx]
                
                # Model predictions
                history_array = np.array(history)
                model_predictions = all_models_predictor.predict_all(history_array)
                
                # State vector
                state_vector = rl_agent.create_state_vector(
                    history=history,
                    model_predictions=model_predictions,
                    bankroll_manager=bankroll_manager
                )
                
                # UZMAN KURALLARI - Optimal action hesapla
                consensus_pred = model_predictions.get('consensus')
                if consensus_pred and consensus_pred.get('above_threshold', False):
                    confidence = consensus_pred.get('confidence', 0.5)
                    prediction = consensus_pred.get('prediction', 1.5)
                    
                    if confidence >= 0.8:
                        if prediction >= 2.5:
                            optimal_action = 3  # Agresif
                        else:
                            optimal_action = 2  # Normal
                    elif confidence >= 0.65:
                        optimal_action = 1  # Konservatif
                    else:
                        optimal_action = 0  # BEKLE
                else:
                    optimal_action = 0  # BEKLE
                
                # Reward hesapla (sample weight için)
                if optimal_action > 0:
                    if optimal_action == 1:
                        bet_amount = virtual_bankroll * 0.02
                    elif optimal_action == 2:
                        bet_amount = virtual_bankroll * 0.04
                    else:
                        bet_amount = virtual_bankroll * 0.06
                else:
                    bet_amount = 0.0
                
                reward = self.calculate_reward(
                    action=optimal_action,
                    predicted_value=consensus_pred.get('prediction', 1.5) if consensus_pred else 1.5,
                    actual_value=actual_value,
                    bet_amount=bet_amount,
                    bankroll=virtual_bankroll
                )
                
                # Bankroll güncelle (simülasyon için)
                if optimal_action > 0:
                    bankroll_manager.place_bet(
                        bet_size=bet_amount,
                        predicted_value=consensus_pred.get('prediction', 1.5) if consensus_pred else 1.5,
                        actual_value=actual_value,
                        confidence=consensus_pred.get('confidence', 0.5) if consensus_pred else 0.5
                    )
                    virtual_bankroll = bankroll_manager.current_bankroll
                
                # Store
                states.append(state_vector)
                actions.append(optimal_action)
                rewards.append(reward)
                
            except Exception as e:
                logger.warning(f"Örnek {idx} atlandı: {e}")
                continue
        
        pbar.close()
        
        # Convert to numpy
        states = np.array(states)
        actions_onehot = tf.keras.utils.to_categorical(actions, num_classes=self.action_dim)
        rewards = np.array(rewards)
        
        logger.info(f"✅ {len(states)} örnek hazırlandı")
        logger.info(f"   States shape: {states.shape}")
        logger.info(f"   Actions distribution: {np.bincount(actions)}")
        logger.info(f"   Reward stats: mean={rewards.mean():.4f}, std={rewards.std():.4f}")
        
        return states, actions_onehot, rewards
    
    def train(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        epochs: int = 50,
        validation_split: float = 0.2
    ):
        """Model eğit - Uzman kararlarını taklit et"""
        logger.info("Model eğitiliyor...")
        
        # Scaler fit
        self.scaler.fit(states)
        states_scaled = self.scaler.transform(states)
        
        # Reward normalization (sample weight için)
        rewards_normalized = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Weighted loss (reward'a göre)
        # Yüksek reward'lu örnekler daha önemli
        sample_weights = (rewards_normalized + 1.0) / 2.0  # 0-1 arası
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ModelCheckpoint(
                'models/imitation_learning_model_best.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train
        history = self.model.fit(
            states_scaled,
            actions,
            epochs=epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            sample_weight=sample_weights,
            callbacks=callbacks_list,
            verbose=1
        )
        
        logger.info("✅ Model eğitimi tamamlandı")
        
        return history
    
    def save_model(self, model_path: str = 'models/imitation_learning_model.h5'):
        """Modeli kaydet"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Best model'i kopyala
        if os.path.exists('models/imitation_learning_model_best.h5'):
            import shutil
            shutil.copy('models/imitation_learning_model_best.h5', model_path)
            logger.info(f"✅ Model kaydedildi: {model_path}")
        else:
            self.model.save(model_path)
            logger.info(f"✅ Model kaydedildi: {model_path}")
        
        # Scaler kaydet
        scaler_path = model_path.replace('.h5', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"✅ Scaler kaydedildi: {scaler_path}")
    
    def evaluate(self, states: np.ndarray, actions: np.ndarray) -> Dict:
        """Model değerlendir - Uzman kararlara ne kadar benziyor?"""
        states_scaled = self.scaler.transform(states)
        predictions = self.model.predict(states_scaled, verbose=0)
        predicted_actions = np.argmax(predictions, axis=1)
        true_actions = np.argmax(actions, axis=1)
        
        accuracy = np.mean(predicted_actions == true_actions)
        
        # Action distribution
        action_dist = np.bincount(predicted_actions, minlength=self.action_dim)
        
        return {
            'accuracy': float(accuracy),
            'action_distribution': action_dist.tolist()
        }


def main():
    """Ana eğitim fonksiyonu"""
    print("="*80)
    print("🤖 IMITATION LEARNING AGENT TRAINING")
    print("="*80)
    
    # Trainer
    trainer = ImitationLearningTrainer(
        state_dim=200,
        action_dim=4,
        learning_rate=0.001,
        batch_size=32
    )
    
    # Model oluştur
    trainer.build_model()
    trainer.model.summary()
    
    # Veri yükle
    values = trainer.load_data(db_path='jetx_data.db', min_history=500)
    
    # AllModelsPredictor yükle
    print("\n📦 Modeller yükleniyor...")
    all_models_predictor = AllModelsPredictor()
    load_results = all_models_predictor.load_all_models()
    
    if not any(load_results.values()):
        raise ValueError("Hiçbir model yüklenemedi! Önce modelleri eğitin.")
    
    print(f"✅ {sum(load_results.values())} model yüklendi")
    
    # Eğitim verisi hazırla
    states, actions, rewards = trainer.prepare_training_data(
        values=values,
        all_models_predictor=all_models_predictor,
        window_size=500,
        sample_ratio=0.1  # Hız için %10 kullan
    )
    
    # Eğit
    history = trainer.train(
        states=states,
        actions=actions,
        rewards=rewards,
        epochs=50,
        validation_split=0.2
    )
    
    # Değerlendir
    eval_results = trainer.evaluate(states, actions)
    print(f"\n✅ Doğruluk: {eval_results['accuracy']:.2%}")
    print(f"   Action dağılımı: {eval_results['action_distribution']}")
    
    # Kaydet
    trainer.save_model('models/imitation_learning_model.h5')
    
    # Model info
    model_info = {
        'model': 'Imitation_Learning_Agent',
        'version': '1.0',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'state_dim': trainer.state_dim,
        'action_dim': trainer.action_dim,
        'training_samples': len(states),
        'accuracy': eval_results['accuracy'],
        'action_distribution': eval_results['action_distribution']
    }
    
    with open('models/imitation_learning_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ IMITATION LEARNING AGENT EĞİTİMİ TAMAMLANDI!")
    print("📝 Not: Bu model uzman kararlarını taklit eder, kendi politikasını keşfetmez.")
    print("="*80)


if __name__ == '__main__':
    main()
