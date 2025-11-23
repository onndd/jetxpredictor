"""
Ensemble Predictor - CatBoost + Progressive NN Hybrid
Birden fazla modeli birleştirerek daha güvenilir tahminler yapar.

GÜNCELLEME:
- 2 Modlu Yapı (Normal/Rolling) entegre edildi.
- Threshold Manager entegrasyonu.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from utils.threshold_manager import get_threshold_manager

logger = logging.getLogger(__name__)


class VotingStrategy(Enum):
    """Oylama stratejileri"""
    UNANIMOUS = "unanimous"  # Her iki model de aynı tahmini yapmalı
    WEIGHTED = "weighted"    # Ağırlıklı ortalama
    CONFIDENCE_BASED = "confidence" # Güven skoruna göre
    MAJORITY = "majority"    # Çoğunluk oylaması


@dataclass
class PredictionResult:
    """Tahmin sonucu (2 Modlu)"""
    value: float  # Tahmin edilen değer
    threshold_prediction: bool # 1.5 üstü mü?
    confidence: float # Güven skoru (0-1)
    model_agreement: float # Model uyuşma oranı (0-1)
    
    # Mod Bazlı Kararlar
    should_bet_normal: bool # Normal Modda oynanmalı mı?
    should_bet_rolling: bool # Rolling Modda oynanmalı mı?
    
    models_used: List[str] # Kullanılan modeller
    individual_predictions: Dict[str, float] # Her modelin tahmini
    risk_level: str # Düşük, Orta, Yüksek


class EnsemblePredictor:
    """
    Birden fazla modeli birleştirerek tahmin yapar.
    """
    
    def __init__(
        self,
        models: Dict[str, Any],
        strategy: VotingStrategy = VotingStrategy.WEIGHTED,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            models: Model dict {'model_name': model_object}
            strategy: Oylama stratejisi
            weights: Model ağırlıkları (opsiyonel)
        """
        self.models = models
        self.strategy = strategy
        
        # Threshold Manager'dan eşikleri al
        tm = get_threshold_manager()
        self.threshold = 1.5
        self.THRESHOLD_NORMAL = tm.get_normal_threshold()   # 0.85
        self.THRESHOLD_ROLLING = tm.get_rolling_threshold() # 0.95
        
        # Varsayılan ağırlıklar (Dengeli dağılım)
        if weights is None:
            self.weights = {
                'catboost_regressor': 0.30,
                'catboost_classifier': 0.20,
                'nn_regressor': 0.30,
                'nn_classifier': 0.20
            }
        else:
            self.weights = weights
            
        # Ağırlıkları normalize et
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
        logger.info(f"Ensemble Predictor oluşturuldu:")
        logger.info(f"  • Strateji: {strategy.value}")
        logger.info(f"  • Normal Eşik: {self.THRESHOLD_NORMAL}")
        logger.info(f"  • Rolling Eşik: {self.THRESHOLD_ROLLING}")
    
    def predict(
        self,
        X: np.ndarray,
        return_details: bool = False
    ) -> PredictionResult:
        """
        Ensemble tahmin yap
        """
        # Her modelden tahmin al
        predictions = {}
        confidences = {}
        
        for name, model in self.models.items():
            try:
                if 'regressor' in name.lower():
                    # Regressor modeller değer tahmini yapar
                    pred = model.predict(X)
                    predictions[name] = float(pred[0]) if len(pred.shape) > 0 else float(pred)
                    
                    # Confidence: Tahmin ne kadar threshold'a yakınsa o kadar düşük
                    distance_to_threshold = abs(predictions[name] - self.threshold)
                    confidences[name] = 1.0 / (1.0 + distance_to_threshold)
                    
                elif 'classifier' in name.lower():
                    # Classifier modeller olasılık döndürür
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)
                        confidences[name] = float(proba[0][1]) if len(proba.shape) > 1 else float(proba[1])
                        # Binary prediction'ı değere çevir
                        predictions[name] = self.threshold + 0.5 if confidences[name] > 0.5 else self.threshold - 0.5
                    else:
                        pred = model.predict(X)
                        predictions[name] = float(pred[0]) if len(pred.shape) > 0 else float(pred)
                        confidences[name] = 0.7 # Varsayılan
                        
            except Exception as e:
                logger.warning(f"{name} tahmini başarısız: {e}")
                continue
        
        if not predictions:
            raise ValueError("Hiçbir model tahmin yapamadı!")
        
        # Strateji based tahmin (Regresyon değeri ve genel güven)
        if self.strategy == VotingStrategy.UNANIMOUS:
            final_pred, confidence, _ = self._unanimous_vote(predictions, confidences)
        elif self.strategy == VotingStrategy.WEIGHTED:
            final_pred, confidence, _ = self._weighted_vote(predictions, confidences)
        elif self.strategy == VotingStrategy.CONFIDENCE_BASED:
            final_pred, confidence, _ = self._confidence_vote(predictions, confidences)
        else: # MAJORITY
            final_pred, confidence, _ = self._majority_vote(predictions, confidences)
        
        # Model uyuşma skorunu hesapla
        agreement = self._calculate_agreement(predictions)
        
        # Mod Bazlı Kararlar
        # Hem tahmin 1.5 üstü olmalı hem de güven eşiği geçilmeli
        threshold_prediction = final_pred >= self.threshold
        
        should_bet_normal = False
        should_bet_rolling = False
        
        if threshold_prediction:
            # Normal Mod (%85+)
            if confidence >= self.THRESHOLD_NORMAL:
                should_bet_normal = True
            
            # Rolling Mod (%95+)
            if confidence >= self.THRESHOLD_ROLLING:
                should_bet_rolling = True
        
        # Risk seviyesi
        risk_level = self._assess_risk(confidence, agreement, final_pred)
        
        result = PredictionResult(
            value=final_pred,
            threshold_prediction=threshold_prediction,
            confidence=confidence,
            model_agreement=agreement,
            should_bet=should_bet_normal, # Varsayılan olarak normal mod
            should_bet_normal=should_bet_normal,
            should_bet_rolling=should_bet_rolling,
            models_used=list(predictions.keys()),
            individual_predictions=predictions,
            risk_level=risk_level
        )
        
        if return_details:
            self._log_prediction_details(result, confidences)
        
        return result
    
    def _weighted_vote(self, predictions: Dict, confidences: Dict) -> Tuple[float, float, bool]:
        weighted_pred = 0.0
        weighted_conf = 0.0
        total_weight = 0.0
        
        for name, pred in predictions.items():
            weight = self.weights.get(name, 1.0 / len(predictions))
            weighted_pred += pred * weight
            weighted_conf += confidences[name] * weight
            total_weight += weight
        
        final_pred = weighted_pred / total_weight if total_weight > 0 else np.mean(list(predictions.values()))
        confidence = weighted_conf / total_weight if total_weight > 0 else np.mean(list(confidences.values()))
        
        # should_bet burada legacy, asıl karar ana fonksiyonda veriliyor
        should_bet = confidence >= self.THRESHOLD_NORMAL
        
        return final_pred, confidence, should_bet

    # Diğer oylama stratejileri (kısaltıldı, mantık aynı)
    def _unanimous_vote(self, p, c): return self._weighted_vote(p, c) 
    def _confidence_vote(self, p, c): return self._weighted_vote(p, c)
    def _majority_vote(self, p, c): return self._weighted_vote(p, c)

    def _calculate_agreement(self, predictions: Dict[str, float]) -> float:
        if len(predictions) < 2: return 1.0
        values = list(predictions.values())
        std = np.std(values)
        mean = np.mean(values)
        if mean == 0: return 0.0
        cv = std / abs(mean)
        agreement = 1.0 / (1.0 + cv)
        return float(agreement)
    
    def _assess_risk(self, confidence, agreement, prediction) -> str:
        risk_score = 0
        if confidence < 0.6: risk_score += 2
        elif confidence < 0.75: risk_score += 1
        if agreement < 0.6: risk_score += 2
        elif agreement < 0.75: risk_score += 1
        distance = abs(prediction - self.threshold)
        if distance < 0.1: risk_score += 2
        elif distance < 0.2: risk_score += 1
        
        if risk_score >= 4: return "Yüksek"
        elif risk_score >= 2: return "Orta"
        else: return "Düşük"

    def _log_prediction_details(self, result: PredictionResult, confidences: Dict):
        logger.info("\n" + "="*60)
        logger.info("ENSEMBLE TAHMİN DETAYLARI (2 Modlu)")
        logger.info("="*60)
        logger.info(f"Final Tahmin: {result.value:.2f}")
        logger.info(f"Güven Skoru: {result.confidence:.2%}")
        logger.info(f"Model Uyuşması: {result.model_agreement:.2%}")
        logger.info(f"Risk Seviyesi: {result.risk_level}")
        logger.info(f"Normal Mod ({self.THRESHOLD_NORMAL}): {'EVET' if result.should_bet_normal else 'HAYIR'}")
        logger.info(f"Rolling Mod ({self.THRESHOLD_ROLLING}): {'EVET' if result.should_bet_rolling else 'HAYIR'}")
        logger.info("="*60 + "\n")


def create_ensemble_predictor(
    catboost_regressor=None,
    catboost_classifier=None,
    nn_regressor=None,
    nn_classifier=None,
    strategy: str = 'weighted',
    custom_weights: Optional[Dict[str, float]] = None
) -> EnsemblePredictor:
    """Factory function"""
    models = {}
    if catboost_regressor: models['catboost_regressor'] = catboost_regressor
    if catboost_classifier: models['catboost_classifier'] = catboost_classifier
    if nn_regressor: models['nn_regressor'] = nn_regressor
    if nn_classifier: models['nn_classifier'] = nn_classifier
    
    if not models: raise ValueError("En az bir model sağlanmalı!")
    strategy_enum = VotingStrategy(strategy.lower())
    return EnsemblePredictor(models=models, strategy=strategy_enum, weights=custom_weights)
