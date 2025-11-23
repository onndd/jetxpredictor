"""
JetX Predictor - Reinforcement Learning Agent

RL Ajanı, tüm model çıktılarını birleştirerek en kârlı aksiyonu seçer.
State vector: Model tahminleri + Risk analizi + Psikolojik analiz + Anomali tespit + Finansal metrikler
Action space: BEKLE, ROLLING (%95+), NORMAL (%85+)
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import os

try:
    import tensorflow as tf
    from tensorflow.keras import models, layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow yüklü değil, RL Agent kullanılamayacak")

from category_definitions import FeatureEngineering
from utils.psychological_analyzer import PsychologicalAnalyzer
from utils.anomaly_streak_detector import AnomalyStreakDetector
from utils.risk_manager import RiskManager
from utils.advanced_bankroll import AdvancedBankrollManager

logger = logging.getLogger(__name__)


class RLAgent:
    """
    Reinforcement Learning Agent
    
    Tüm model çıktılarını birleştirerek en kârlı aksiyonu seçer.
    """
    
    # Action space - SADECE 2 MOD (Rolling ve Normal)
    # Modelin 4 çıktısı olduğu için 3. çıktıyı da Normal'e bağlıyoruz.
    ACTIONS = {
        0: {'name': 'BEKLE', 'description': 'Bahis yapma', 'risk': 'NONE'},
        1: {'name': 'BAHIS_YAP_ROLLING', 'description': 'Rolling Mod (%95+ Güven, 1.5x Sabit)', 'risk': 'MINIMAL'},
        2: {'name': 'BAHIS_YAP_NORMAL', 'description': 'Normal Mod (%85+ Güven, Dinamik)', 'risk': 'LOW'},
        3: {'name': 'BAHIS_YAP_NORMAL', 'description': 'Normal Mod (%85+ Güven, Dinamik)', 'risk': 'LOW'} # Eski agresif iptal, Normal'e bağlandı
    }
    
    def __init__(self, model_path: str = 'models/rl_agent_model.h5', threshold: float = 1.5):
        """
        Args:
            model_path: RL model dosya yolu
            threshold: Kritik eşik değeri (default: 1.5)
        """
        self.model_path = model_path
        self.threshold = threshold
        self.model = None
        self.scaler = None
        
        # Analyzers
        self.psychological_analyzer = PsychologicalAnalyzer(threshold=threshold)
        self.anomaly_detector = AnomalyStreakDetector(threshold=threshold)
        # Risk manager default normal, mod seçimi dinamik yapılacak
        self.risk_manager = RiskManager(mode='normal')
        
        logger.info("RLAgent başlatıldı")
    
    def load_model(self) -> bool:
        """RL modelini yükle"""
        if not TF_AVAILABLE:
            logger.error("TensorFlow yüklü değil, model yüklenemiyor")
            return False
        
        if not os.path.exists(self.model_path):
            logger.warning(f"RL model bulunamadı: {self.model_path}")
            return False
        
        try:
            self.model = models.load_model(self.model_path, compile=False)
            logger.info(f"✅ RL model yüklendi: {self.model_path}")
            
            # Scaler yükle (varsa)
            scaler_path = self.model_path.replace('.h5', '_scaler.pkl')
            if os.path.exists(scaler_path):
                import joblib
                self.scaler = joblib.load(scaler_path)
                logger.info("✅ RL model scaler yüklendi")
            
            return True
        except Exception as e:
            logger.error(f"RL model yükleme hatası: {e}")
            return False
    
    def create_state_vector(
        self,
        history: List[float],
        model_predictions: Dict,
        risk_analysis: Optional[Dict] = None,
        bankroll_manager: Optional[AdvancedBankrollManager] = None
    ) -> np.ndarray:
        """State vector oluştur"""
        state_parts = []
        
        # 1. Model çıktıları (20 boyut)
        model_features = self._extract_model_features(model_predictions)
        state_parts.append(model_features)
        
        # 2. Risk skorları (10 boyut)
        risk_features = self._extract_risk_features(risk_analysis, history)
        state_parts.append(risk_features)
        
        # 3. Psikolojik analiz (30 boyut)
        psych_features = self._extract_psychological_features(history)
        state_parts.append(psych_features)
        
        # 4. Anomali tespit (15 boyut)
        anomaly_features = self._extract_anomaly_features(history)
        state_parts.append(anomaly_features)
        
        # 5. Finansal metrikler (10 boyut)
        financial_features = self._extract_financial_features(bankroll_manager, model_predictions)
        state_parts.append(financial_features)
        
        # 6. Feature özetleri (115 boyut)
        feature_summary = self._extract_feature_summary(history)
        state_parts.append(feature_summary)
        
        # Birleştir
        state_vector = np.concatenate(state_parts)
        
        # Normalize et (scaler varsa)
        if self.scaler is not None:
            state_vector = self.scaler.transform(state_vector.reshape(1, -1))[0]
        
        return state_vector
    
    def _extract_model_features(self, model_predictions: Dict) -> np.ndarray:
        """Model çıktılarını özellik vektörüne çevir"""
        features = []
        
        # Progressive NN
        if 'progressive_nn' in model_predictions and model_predictions['progressive_nn']:
            pn = model_predictions['progressive_nn']
            features.extend([
                pn.get('prediction', 1.5),
                pn.get('threshold_prob', 0.5),
                pn.get('confidence', 0.5),
                1.0 if pn.get('above_threshold', False) else 0.0
            ])
        else:
            features.extend([1.5, 0.5, 0.5, 0.0])
        
        # CatBoost
        if 'catboost' in model_predictions and model_predictions['catboost']:
            cb = model_predictions['catboost']
            features.extend([
                cb.get('prediction', 1.5),
                cb.get('threshold_prob', 0.5),
                cb.get('confidence', 0.5),
                1.0 if cb.get('above_threshold', False) else 0.0
            ])
        else:
            features.extend([1.5, 0.5, 0.5, 0.0])
        
        # AutoGluon
        if 'autogluon' in model_predictions and model_predictions['autogluon']:
            ag = model_predictions['autogluon']
            features.extend([
                ag.get('prediction', 1.5),
                ag.get('threshold_prob', 0.5),
                ag.get('confidence', 0.5),
                1.0 if ag.get('above_threshold', False) else 0.0
            ])
        else:
            features.extend([1.5, 0.5, 0.5, 0.0])
        
        # TabNet
        if 'tabnet' in model_predictions and model_predictions['tabnet']:
            tn = model_predictions['tabnet']
            features.extend([
                tn.get('prediction', 1.5),
                tn.get('threshold_prob', 0.5),
                tn.get('confidence', 0.5),
                1.0 if tn.get('above_threshold', False) else 0.0
            ])
        else:
            features.extend([1.5, 0.5, 0.5, 0.0])
        
        # Consensus
        if 'consensus' in model_predictions and model_predictions['consensus']:
            cs = model_predictions['consensus']
            features.extend([
                cs.get('prediction', 1.5),
                cs.get('agreement', 0.5),
                cs.get('confidence', 0.5),
                cs.get('models_agreed', 0) / max(cs.get('total_models', 1), 1)
            ])
        else:
            features.extend([1.5, 0.5, 0.5, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    def _extract_risk_features(self, risk_analysis: Optional[Dict], history: List[float]) -> np.ndarray:
        """Risk skorlarını özellik vektörüne çevir"""
        features = []
        
        if risk_analysis:
            features.extend([
                1.0 if risk_analysis.get('should_play', False) else 0.0,
                self._risk_level_to_float(risk_analysis.get('risk_level', 'HIGH')),
                risk_analysis.get('confidence_threshold', 0.65),
                len(risk_analysis.get('reasons', []))
            ])
        else:
            # Fallback
            features.extend([0.0, 0.5, 0.65, 0])
        
        # Ardışık kayıp/kazanç
        if len(history) >= 10:
            recent_10 = history[-10:]
            consecutive_above = 0
            consecutive_below = 0
            
            for i in range(len(recent_10) - 1, -1, -1):
                if recent_10[i] >= self.threshold:
                    consecutive_above += 1
                    if consecutive_below > 0: break
                else:
                    consecutive_below += 1
                    if consecutive_above > 0: break
            
            features.extend([
                consecutive_above / 10.0,
                consecutive_below / 10.0,
                sum(1 for v in recent_10 if v >= self.threshold) / len(recent_10)
            ])
        else:
            features.extend([0.0, 0.0, 0.5])
        
        # Volatilite
        if len(history) >= 20:
            recent_20 = history[-20:]
            volatility = np.std(recent_20) / (np.mean(recent_20) + 1e-8)
            features.append(min(volatility, 2.0))
        else:
            features.append(0.0)
        
        # Son değer
        if len(history) > 0:
            last_value = history[-1]
            features.append(last_value)
            features.append(1.0 if last_value >= self.threshold else 0.0)
        else:
            features.extend([1.5, 0.0])
        
        # Padding
        while len(features) < 10:
            features.append(0.0)
        
        return np.array(features[:10], dtype=np.float32)
    
    def _extract_psychological_features(self, history: List[float]) -> np.ndarray:
        """Psikolojik analiz özelliklerini çıkar"""
        if len(history) < 20:
            return np.zeros(30, dtype=np.float32)
        
        try:
            psych_features = self.psychological_analyzer.analyze_psychological_patterns(history)
            
            # Önemli feature'ları seç
            important_features = [
                'bait_switch_score', 'trap_risk', 'false_confidence_score',
                'heating_score', 'cooling_score', 'volatility_shift',
                'desperation_level', 'gambler_fallacy_risk',
                'momentum_reversal_score', 'sudden_spike_up', 'sudden_spike_down',
                'pattern_repetition_score', 'mean_reversion_score',
                'extreme_high_cluster', 'extreme_low_cluster',
                'short_term_trend', 'medium_term_trend', 'long_term_trend',
                'timeframe_divergence', 'z_score_current', 'macd_signal', 'rsi_momentum',
                'manipulation_score'
            ]
            
            features = []
            for feat in important_features:
                features.append(psych_features.get(feat, 0.0))
            
            # Padding
            while len(features) < 30:
                features.append(0.0)
            
            return np.array(features[:30], dtype=np.float32)
        except Exception:
            return np.zeros(30, dtype=np.float32)
    
    def _extract_anomaly_features(self, history: List[float]) -> np.ndarray:
        """Anomali tespit özelliklerini çıkar"""
        if len(history) < 5:
            return np.zeros(15, dtype=np.float32)
        
        try:
            anomaly_features = self.anomaly_detector.extract_streak_features(history)
            
            important_features = [
                'current_above_streak', 'current_below_streak', 'current_streak_length',
                'max_above_streak_10', 'max_below_streak_10',
                'max_above_streak_20', 'max_below_streak_20',
                'has_extreme_above_streak', 'has_extreme_below_streak',
                'extreme_streak_risk', 'streak_break_probability',
                'alternating_pattern_score', 'is_alternating'
            ]
            
            features = []
            for feat in important_features:
                features.append(anomaly_features.get(feat, 0.0))
            
            while len(features) < 15:
                features.append(0.0)
            
            return np.array(features[:15], dtype=np.float32)
        except Exception:
            return np.zeros(15, dtype=np.float32)
    
    def _extract_financial_features(self, bankroll_manager: Optional[AdvancedBankrollManager], model_predictions: Dict) -> np.ndarray:
        """Finansal metrikleri özellik vektörüne çevir"""
        features = []
        
        if bankroll_manager:
            report = bankroll_manager.get_report()
            features.extend([
                report.get('roi', 0.0) / 100.0,
                report.get('win_rate', 0.0) / 100.0,
                report.get('current_bankroll', 1000.0) / 10000.0,
                report.get('total_bets', 0) / 1000.0,
                report.get('current_streak', 0) / 10.0
            ])
        else:
            features.extend([0.0, 0.5, 0.1, 0.0, 0.0])
        
        if 'consensus' in model_predictions and model_predictions['consensus']:
            cs = model_predictions['consensus']
            confidence = cs.get('confidence', 0.5)
            prediction = cs.get('prediction', 1.5)
            
            win_prob = confidence
            win_multiplier = prediction - 1.0
            if win_multiplier > 0:
                kelly_frac = (win_prob * win_multiplier - (1 - win_prob)) / win_multiplier
                kelly_frac = max(0, min(kelly_frac, 0.25))
            else:
                kelly_frac = 0.0
            
            features.append(kelly_frac)
            features.append(confidence)
            features.append(prediction / 10.0)
        else:
            features.extend([0.0, 0.5, 0.15])
        
        while len(features) < 10:
            features.append(0.0)
        
        return np.array(features[:10], dtype=np.float32)
    
    def _extract_feature_summary(self, history: List[float]) -> np.ndarray:
        """Feature engineering özeti"""
        if len(history) < 50:
            return np.zeros(115, dtype=np.float32)
        
        try:
            all_features = FeatureEngineering.extract_all_features(history)
            feature_values = list(all_features.values())
            
            if len(feature_values) >= 115:
                return np.array(feature_values[:115], dtype=np.float32)
            else:
                padded = feature_values + [0.0] * (115 - len(feature_values))
                return np.array(padded, dtype=np.float32)
        except Exception:
            return np.zeros(115, dtype=np.float32)
    
    def _risk_level_to_float(self, risk_level: str) -> float:
        """Risk seviyesini float'a çevir"""
        risk_map = {'LOW': 0.2, 'MEDIUM': 0.5, 'HIGH': 0.8, 'CRITICAL': 1.0}
        return risk_map.get(risk_level, 0.5)
    
    def predict_action(self, state_vector: np.ndarray, use_greedy: bool = True) -> Tuple[int, np.ndarray]:
        """State vector'den action tahmin et"""
        if self.model is None:
            logger.error("Model yüklenmemiş!")
            return 0, np.array([1.0, 0.0, 0.0, 0.0])
        
        try:
            state_input = state_vector.reshape(1, -1)
            probabilities = self.model.predict(state_input, verbose=0)[0]
            
            if use_greedy:
                action = int(np.argmax(probabilities))
            else:
                action = int(np.random.choice(4, p=probabilities))
            
            return action, probabilities
        except Exception as e:
            logger.error(f"Action prediction hatası: {e}")
            return 0, np.array([1.0, 0.0, 0.0, 0.0])
    
    def interpret_action(
        self,
        action: int,
        probabilities: np.ndarray,
        model_predictions: Dict,
        bankroll: Optional[float] = None
    ) -> Dict:
        """
        Action'ı senin sistemindeki 2 moda (Rolling/Normal) uygun şekilde yorumlar.
        Aksiyon 3 (Eski Agresif/Yüksek Risk) artık Normal Mod olarak işlem görür.
        """
        action_info = self.ACTIONS[action].copy()
        
        # Consensus prediction'dan bilgi al
        consensus_pred = None
        consensus_conf = 0.0
        if 'consensus' in model_predictions and model_predictions['consensus']:
            consensus_pred = model_predictions['consensus']
            consensus_conf = consensus_pred.get('confidence', 0.0)
        
        result = {
            'action': action,
            'action_name': action_info['name'],
            'action_description': action_info['description'],
            'risk_level': action_info['risk'],
            'confidence': float(probabilities[action]),
            'probabilities': probabilities.tolist(),
            'should_bet': action > 0,
            'bet_amount': None,
            'exit_multiplier': None,
            'reasoning': []
        }
        
        # Action 0: BEKLE
        if action == 0:
            result['reasoning'].append("Ajan bekleme kararı aldı")
            if consensus_conf > 0 and consensus_conf < 0.85:
                 result['reasoning'].append(f"Güven %85'in altında (%{consensus_conf*100:.0f})")

        # Action 1: ROLLING MOD (Eski Konservatif)
        # Hedef: %95 Güven, 1.50x Çıkış
        elif action == 1:
            result['exit_multiplier'] = 1.50
            result['bet_percentage'] = 2.0 # Rolling'de kasa koruma ön planda
            
            default_bankroll = bankroll if bankroll else 1000.0
            result['bet_amount'] = default_bankroll * (result['bet_percentage'] / 100)
            
            result['reasoning'].append("🛡️ ROLLING STRATEJİSİ: Güvenli Liman")
            result['reasoning'].append("Hedef: 1.50x garantilemek")
            
            if consensus_conf < 0.95:
                 result['reasoning'].append(f"⚠️ DİKKAT: Ajan Rolling seçti ama Consensus güveni (%{consensus_conf*100:.0f}) %95 sınırında değil.")

        # Action 2 ve 3: NORMAL MOD (Eski Normal ve Agresif Birleştirildi)
        # Hedef: %85 Güven, Dinamik Çıkış
        elif action in [2, 3]:
            # Çıkış noktasını tahmine göre ayarla ama uçmama izin verme
            if consensus_pred:
                pred_value = consensus_pred.get('prediction', 1.5)
                # Normal modda bile max 2.5x ile sınırla, güvenli olsun
                result['exit_multiplier'] = min(pred_value * 0.8, 2.5) 
            else:
                result['exit_multiplier'] = 1.8
            
            result['bet_percentage'] = 4.0
            default_bankroll = bankroll if bankroll else 1000.0
            result['bet_amount'] = default_bankroll * (result['bet_percentage'] / 100)
            
            result['reasoning'].append("🎯 NORMAL STRATEJİ: Dengeli Oyun")
            result['reasoning'].append(f"Hedef: {result['exit_multiplier']:.2f}x (Tahmine göre ayarlandı)")
            
            if consensus_conf < 0.85:
                result['reasoning'].append(f"⚠️ UYARI: Güven %85'in altında (%{consensus_conf*100:.0f}) - Riskli olabilir")
                
        return result


# Factory function
def create_rl_agent(model_path: str = 'models/rl_agent_model.h5', threshold: float = 1.5) -> RLAgent:
    """RL Agent oluştur"""
    agent = RLAgent(model_path=model_path, threshold=threshold)
    agent.load_model()
    return agent
