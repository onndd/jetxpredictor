"""
JetX Predictor - Reinforcement Learning Agent

RL Ajanı, tüm model çıktılarını birleştirerek en kârlı aksiyonu seçer.
State vector: Model tahminleri + Risk analizi + Psikolojik analiz + Anomali tespit + Finansal metrikler
Action space: BEKLE, BAHIS YAP (Konservatif, Normal, Yüksek Risk)
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
    
    # Action space
    ACTIONS = {
        0: {'name': 'BEKLE', 'description': 'Bahis yapma', 'risk': 'LOW'},
        1: {'name': 'BAHIS_YAP_KONSERVATIF', 'description': '1.5x çıkış, %2 bahis', 'risk': 'LOW'},
        2: {'name': 'BAHIS_YAP_NORMAL', 'description': 'Dinamik çıkış, %4 bahis', 'risk': 'MEDIUM'},
        3: {'name': 'BAHIS_YAP_YUKSEK_RISK', 'description': 'Yüksek çıkış, %6 bahis', 'risk': 'HIGH'}
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
        self.risk_manager = RiskManager(mode='normal')
        
        logger.info("RLAgent başlatıldı")
    
    def load_model(self) -> bool:
        """
        RL modelini yükle
        
        Returns:
            Başarılı ise True
        """
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
        """
        State vector oluştur
        
        Args:
            history: Geçmiş değerler
            model_predictions: Tüm model tahminleri (AllModelsPredictor çıktısı)
            risk_analysis: Risk analizi sonuçları (opsiyonel)
            bankroll_manager: Bankroll manager (opsiyonel)
            
        Returns:
            State vector (1D numpy array)
        """
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
        
        # 6. Feature özetleri (115 boyut - önemli feature'lar)
        feature_summary = self._extract_feature_summary(history)
        state_parts.append(feature_summary)
        
        # Birleştir
        state_vector = np.concatenate(state_parts)
        
        # Normalize et (scaler varsa)
        if self.scaler is not None:
            state_vector = self.scaler.transform(state_vector.reshape(1, -1))[0]
        
        return state_vector
    
    def _extract_model_features(self, model_predictions: Dict) -> np.ndarray:
        """Model çıktılarını özellik vektörüne çevir (20 boyut)"""
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
        """Risk skorlarını özellik vektörüne çevir (10 boyut)"""
        features = []
        
        if risk_analysis:
            features.extend([
                1.0 if risk_analysis.get('should_play', False) else 0.0,
                self._risk_level_to_float(risk_analysis.get('risk_level', 'HIGH')),
                risk_analysis.get('confidence_threshold', 0.65),
                len(risk_analysis.get('reasons', []))
            ])
        else:
            # Risk manager'dan al
            if len(history) >= 50:
                recent = history[-50:]
                # Basit risk hesaplama
                above_count = sum(1 for v in recent if v >= self.threshold)
                risk_score = above_count / len(recent)
                features.extend([0.5, 0.5, 0.65, 0])
            else:
                features.extend([0.0, 0.5, 0.65, 0])
        
        # Ardışık kayıp/kazanç
        if len(history) >= 10:
            recent_10 = history[-10:]
            consecutive_above = 0
            consecutive_below = 0
            
            for i in range(len(recent_10) - 1, -1, -1):
                if recent_10[i] >= self.threshold:
                    consecutive_above += 1
                    if consecutive_below > 0:
                        break
                else:
                    consecutive_below += 1
                    if consecutive_above > 0:
                        break
            
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
            features.append(min(volatility, 2.0))  # Cap at 2.0
        else:
            features.append(0.0)
        
        # Son değer
        if len(history) > 0:
            last_value = history[-1]
            features.append(last_value)
            features.append(1.0 if last_value >= self.threshold else 0.0)
        else:
            features.extend([1.5, 0.0])
        
        # Padding if needed
        while len(features) < 10:
            features.append(0.0)
        
        return np.array(features[:10], dtype=np.float32)
    
    def _extract_psychological_features(self, history: List[float]) -> np.ndarray:
        """Psikolojik analiz özelliklerini çıkar (30 boyut)"""
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
        except Exception as e:
            logger.warning(f"Psikolojik analiz hatası: {e}")
            return np.zeros(30, dtype=np.float32)
    
    def _extract_anomaly_features(self, history: List[float]) -> np.ndarray:
        """Anomali tespit özelliklerini çıkar (15 boyut)"""
        if len(history) < 5:
            return np.zeros(15, dtype=np.float32)
        
        try:
            anomaly_features = self.anomaly_detector.extract_streak_features(history)
            
            # Önemli feature'ları seç
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
            
            # Padding
            while len(features) < 15:
                features.append(0.0)
            
            return np.array(features[:15], dtype=np.float32)
        except Exception as e:
            logger.warning(f"Anomali tespit hatası: {e}")
            return np.zeros(15, dtype=np.float32)
    
    def _extract_financial_features(
        self,
        bankroll_manager: Optional[AdvancedBankrollManager],
        model_predictions: Dict
    ) -> np.ndarray:
        """Finansal metrikleri özellik vektörüne çevir (10 boyut)"""
        features = []
        
        if bankroll_manager:
            report = bankroll_manager.get_report()
            features.extend([
                report.get('roi', 0.0) / 100.0,  # Normalize
                report.get('win_rate', 0.0) / 100.0,  # Normalize
                report.get('current_bankroll', 1000.0) / 10000.0,  # Normalize
                report.get('total_bets', 0) / 1000.0,  # Normalize
                report.get('current_streak', 0) / 10.0  # Normalize
            ])
        else:
            features.extend([0.0, 0.5, 0.1, 0.0, 0.0])
        
        # Consensus prediction'dan Kelly Criterion benzeri hesaplama
        if 'consensus' in model_predictions and model_predictions['consensus']:
            cs = model_predictions['consensus']
            confidence = cs.get('confidence', 0.5)
            prediction = cs.get('prediction', 1.5)
            
            # Basit Kelly-like calculation
            win_prob = confidence
            win_multiplier = prediction - 1.0
            if win_multiplier > 0:
                kelly_frac = (win_prob * win_multiplier - (1 - win_prob)) / win_multiplier
                kelly_frac = max(0, min(kelly_frac, 0.25))  # Cap at 25%
            else:
                kelly_frac = 0.0
            
            features.append(kelly_frac)
            features.append(confidence)
            features.append(prediction / 10.0)  # Normalize
        else:
            features.extend([0.0, 0.5, 0.15])
        
        # Padding
        while len(features) < 10:
            features.append(0.0)
        
        return np.array(features[:10], dtype=np.float32)
    
    def _extract_feature_summary(self, history: List[float]) -> np.ndarray:
        """Feature engineering özeti (115 boyut - önemli feature'lar)"""
        if len(history) < 50:
            return np.zeros(115, dtype=np.float32)
        
        try:
            # Tüm feature'ları çıkar
            all_features = FeatureEngineering.extract_all_features(history)
            feature_values = list(all_features.values())
            
            # İlk 115 feature'ı al (veya padding)
            if len(feature_values) >= 115:
                return np.array(feature_values[:115], dtype=np.float32)
            else:
                # Padding
                padded = feature_values + [0.0] * (115 - len(feature_values))
                return np.array(padded, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Feature extraction hatası: {e}")
            return np.zeros(115, dtype=np.float32)
    
    def _risk_level_to_float(self, risk_level: str) -> float:
        """Risk seviyesini float'a çevir"""
        risk_map = {
            'LOW': 0.2,
            'MEDIUM': 0.5,
            'HIGH': 0.8,
            'CRITICAL': 1.0,
            'SAFE': 0.1,
            'CAUTION': 0.4,
            'WARNING': 0.7,
            'DANGER': 0.9
        }
        return risk_map.get(risk_level, 0.5)
    
    def predict_action(
        self,
        state_vector: np.ndarray,
        use_greedy: bool = True
    ) -> Tuple[int, np.ndarray]:
        """
        State vector'den action tahmin et
        
        Args:
            state_vector: State vector
            use_greedy: Greedy action seçimi (True) veya probability sampling (False)
            
        Returns:
            (action, probabilities) tuple
        """
        if self.model is None:
            logger.error("Model yüklenmemiş!")
            return 0, np.array([1.0, 0.0, 0.0, 0.0])
        
        try:
            # Reshape for model input
            state_input = state_vector.reshape(1, -1)
            
            # Predict
            probabilities = self.model.predict(state_input, verbose=0)[0]
            
            # Action seç
            if use_greedy:
                action = int(np.argmax(probabilities))
            else:
                # Probability sampling
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
        Action'ı insan okunabilir formata çevir
        
        Args:
            action: Action ID (0-3)
            probabilities: Action probabilities
            model_predictions: Model tahminleri
            bankroll: Mevcut bankroll (opsiyonel)
            
        Returns:
            Action interpretation dictionary
        """
        action_info = self.ACTIONS[action].copy()
        
        # Consensus prediction'dan bilgi al
        consensus_pred = None
        if 'consensus' in model_predictions and model_predictions['consensus']:
            consensus_pred = model_predictions['consensus']
        
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
        
        if action == 0:  # BEKLE
            result['reasoning'].append("RL Ajanı bahis yapmamayı öneriyor")
            if consensus_pred:
                if not consensus_pred.get('above_threshold', False):
                    result['reasoning'].append(f"Modeller 1.5x altı tahmin ediyor ({consensus_pred.get('prediction', 0):.2f}x)")
                if consensus_pred.get('confidence', 0) < 0.6:
                    result['reasoning'].append(f"Düşük güven skoru (%{consensus_pred.get('confidence', 0)*100:.0f})")
        
        elif action == 1:  # KONSERVATIF
            result['exit_multiplier'] = 1.5
            result['bet_percentage'] = 2.0
            # Bankroll yoksa varsayılan değer kullan
            default_bankroll = bankroll if bankroll else 1000.0
            result['bet_amount'] = default_bankroll * 0.02
            result['reasoning'].append("Konservatif strateji: 1.5x'te çık, sermaye koruma öncelikli")
            if consensus_pred:
                result['reasoning'].append(f"Consensus tahmin: {consensus_pred.get('prediction', 0):.2f}x")
        
        elif action == 2:  # NORMAL
            if consensus_pred:
                pred_value = consensus_pred.get('prediction', 1.5)
                result['exit_multiplier'] = min(pred_value * 0.8, 2.5)
            else:
                result['exit_multiplier'] = 1.8
            result['bet_percentage'] = 4.0
            # Bankroll yoksa varsayılan değer kullan
            default_bankroll = bankroll if bankroll else 1000.0
            result['bet_amount'] = default_bankroll * 0.04
            result['reasoning'].append("Normal strateji: Dinamik çıkış noktası")
            if consensus_pred:
                result['reasoning'].append(f"Consensus tahmin: {consensus_pred.get('prediction', 0):.2f}x")
        
        elif action == 3:  # YUKSEK RISK
            if consensus_pred:
                pred_value = consensus_pred.get('prediction', 1.5)
                result['exit_multiplier'] = min(pred_value * 0.85, 5.0)
            else:
                result['exit_multiplier'] = 2.5
            result['bet_percentage'] = 6.0
            # Bankroll yoksa varsayılan değer kullan
            default_bankroll = bankroll if bankroll else 1000.0
            result['bet_amount'] = default_bankroll * 0.06
            result['reasoning'].append("Yüksek Risk stratejisi: Yüksek risk, yüksek getiri")
            if consensus_pred:
                result['reasoning'].append(f"Consensus tahmin: {consensus_pred.get('prediction', 0):.2f}x")
        
        return result


# Factory function
def create_rl_agent(model_path: str = 'models/rl_agent_model.h5', threshold: float = 1.5) -> RLAgent:
    """
    RL Agent oluştur
    
    Args:
        model_path: RL model dosya yolu
        threshold: Kritik eşik değeri
        
    Returns:
        RLAgent instance
    """
    agent = RLAgent(model_path=model_path, threshold=threshold)
    agent.load_model()
    return agent
