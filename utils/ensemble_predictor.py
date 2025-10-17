"""
Ensemble Predictor - CatBoost + Progressive NN Hybrid
Birden fazla modeli birleştirerek daha güvenilir tahminler yapar.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class VotingStrategy(Enum):
    """Oylama stratejileri"""
    UNANIMOUS = "unanimous"  # Her iki model de aynı tahmini yapmalı
    WEIGHTED = "weighted"  # Ağırlıklı ortalama
    CONFIDENCE_BASED = "confidence"  # Güven skoruna göre
    MAJORITY = "majority"  # Çoğunluk oylaması


@dataclass
class PredictionResult:
    """Tahmin sonucu"""
    value: float  # Tahmin edilen değer
    threshold_prediction: bool  # 1.5 üstü mü?
    confidence: float  # Güven skoru (0-1)
    model_agreement: float  # Model uyuşma oranı (0-1)
    should_bet: bool  # Bahse girilmeli mi?
    models_used: List[str]  # Kullanılan modeller
    individual_predictions: Dict[str, float]  # Her modelin tahmini
    risk_level: str  # Düşük, Orta, Yüksek


class EnsemblePredictor:
    """
    Birden fazla modeli birleştirerek tahmin yapar.
    
    Stratejiler:
    1. UNANIMOUS: Her iki model de aynı yönde tahmin yapmalı
    2. WEIGHTED: Model güvenilirliklerine göre ağırlıklı ortalama
    3. CONFIDENCE_BASED: Yüksek güvenli modele daha fazla ağırlık
    4. MAJORITY: Basit çoğunluk oylaması
    """
    
    def __init__(
        self,
        models: Dict[str, Any],
        strategy: VotingStrategy = VotingStrategy.WEIGHTED,
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 1.5,
        min_confidence: float = 0.5
    ):
        """
        Args:
            models: Model dict {'model_name': model_object}
            strategy: Oylama stratejisi
            weights: Model ağırlıkları (opsiyonel)
            threshold: Eşik değeri (varsayılan: 1.5)
            min_confidence: Minimum güven skoru (varsayılan: 0.5)
        """
        self.models = models
        self.strategy = strategy
        self.threshold = threshold
        self.min_confidence = min_confidence
        
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
        logger.info(f"  • Model sayısı: {len(models)}")
        logger.info(f"  • Ağırlıklar: {self.weights}")
        logger.info(f"  • Threshold: {threshold}")
        logger.info(f"  • Min güven: {min_confidence}")
    
    def predict(
        self,
        X: np.ndarray,
        return_details: bool = False
    ) -> PredictionResult:
        """
        Ensemble tahmin yap
        
        Args:
            X: Input features
            return_details: Detaylı sonuç döndür
            
        Returns:
            PredictionResult object
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
                        # 1.5 üstü olma olasılığı
                        confidences[name] = float(proba[0][1]) if len(proba.shape) > 1 else float(proba[1])
                        # Binary prediction'ı değere çevir
                        predictions[name] = self.threshold + 0.5 if confidences[name] > 0.5 else self.threshold - 0.5
                    else:
                        pred = model.predict(X)
                        predictions[name] = float(pred[0]) if len(pred.shape) > 0 else float(pred)
                        confidences[name] = 0.7  # Varsayılan
                        
            except Exception as e:
                logger.warning(f"{name} tahmini başarısız: {e}")
                continue
        
        if not predictions:
            raise ValueError("Hiçbir model tahmin yapamadı!")
        
        # Strateji based tahmin
        if self.strategy == VotingStrategy.UNANIMOUS:
            final_pred, confidence, should_bet = self._unanimous_vote(predictions, confidences)
        elif self.strategy == VotingStrategy.WEIGHTED:
            final_pred, confidence, should_bet = self._weighted_vote(predictions, confidences)
        elif self.strategy == VotingStrategy.CONFIDENCE_BASED:
            final_pred, confidence, should_bet = self._confidence_vote(predictions, confidences)
        else:  # MAJORITY
            final_pred, confidence, should_bet = self._majority_vote(predictions, confidences)
        
        # Model uyuşma skorunu hesapla
        agreement = self._calculate_agreement(predictions)
        
        # Risk seviyesi
        risk_level = self._assess_risk(confidence, agreement, final_pred)
        
        result = PredictionResult(
            value=final_pred,
            threshold_prediction=final_pred >= self.threshold,
            confidence=confidence,
            model_agreement=agreement,
            should_bet=should_bet,
            models_used=list(predictions.keys()),
            individual_predictions=predictions,
            risk_level=risk_level
        )
        
        if return_details:
            self._log_prediction_details(result, confidences)
        
        return result
    
    def _unanimous_vote(
        self,
        predictions: Dict[str, float],
        confidences: Dict[str, float]
    ) -> Tuple[float, float, bool]:
        """
        Oybirliği stratejisi: Tüm modeller aynı yönde tahmin yapmalı
        
        Returns:
            (final_prediction, confidence, should_bet)
        """
        # Tüm tahminler threshold'un aynı tarafında mı?
        above_threshold = [p >= self.threshold for p in predictions.values()]
        
        if all(above_threshold) or not any(above_threshold):
            # Tüm modeller hemfikir
            final_pred = np.mean(list(predictions.values()))
            confidence = np.mean(list(confidences.values()))
            should_bet = confidence >= self.min_confidence
        else:
            # Modeller anlaşamadı - gradient confidence hesapla
            final_pred = np.mean(list(predictions.values()))
            pred_values = list(predictions.values())
            std_dev = np.std(pred_values)
            mean_pred = np.mean(pred_values)
            
            # Uyuşmazlık skoruna göre confidence (düşük std = yüksek confidence)
            if mean_pred > 0:
                agreement_score = 1.0 - min(1.0, std_dev / mean_pred)
                confidence = max(0.3, min(0.7, agreement_score))
            else:
                confidence = 0.3
            
            should_bet = False  # Anlaşmazlık durumunda bahse girme
        
        return final_pred, confidence, should_bet
    
    def _weighted_vote(
        self,
        predictions: Dict[str, float],
        confidences: Dict[str, float]
    ) -> Tuple[float, float, bool]:
        """
        Ağırlıklı ortalama stratejisi
        
        Returns:
            (final_prediction, confidence, should_bet)
        """
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
        should_bet = confidence >= self.min_confidence
        
        return final_pred, confidence, should_bet
    
    def _confidence_vote(
        self,
        predictions: Dict[str, float],
        confidences: Dict[str, float]
    ) -> Tuple[float, float, bool]:
        """
        Güven bazlı strateji: En güvenli modele en fazla ağırlık
        
        Returns:
            (final_prediction, confidence, should_bet)
        """
        # Güven skorlarını normalize et
        total_conf = sum(confidences.values())
        if total_conf == 0:
            return self._weighted_vote(predictions, confidences)
        
        dynamic_weights = {name: conf / total_conf for name, conf in confidences.items()}
        
        weighted_pred = sum(predictions[name] * dynamic_weights[name] for name in predictions.keys())
        overall_confidence = max(confidences.values())  # En yüksek güven
        should_bet = overall_confidence >= self.min_confidence
        
        return weighted_pred, overall_confidence, should_bet
    
    def _majority_vote(
        self,
        predictions: Dict[str, float],
        confidences: Dict[str, float]
    ) -> Tuple[float, float, bool]:
        """
        Çoğunluk oylaması: Basit oy sayımı
        
        Returns:
            (final_prediction, confidence, should_bet)
        """
        above_count = sum(1 for p in predictions.values() if p >= self.threshold)
        below_count = len(predictions) - above_count
        
        # Çoğunluk kararı
        if above_count > below_count:
            # Çoğunluk 1.5 üstü diyor
            final_pred = np.mean([p for p in predictions.values() if p >= self.threshold])
            confidence = above_count / len(predictions)
        else:
            # Çoğunluk 1.5 altı diyor
            final_pred = np.mean([p for p in predictions.values() if p < self.threshold])
            confidence = below_count / len(predictions)
        
        should_bet = confidence >= self.min_confidence
        
        return final_pred, confidence, should_bet
    
    def _calculate_agreement(self, predictions: Dict[str, float]) -> float:
        """
        Model uyuşma skorunu hesapla (0-1)
        
        1.0 = Tüm modeller çok yakın tahminler yapıyor
        0.0 = Modeller çok farklı tahminler yapıyor
        """
        if len(predictions) < 2:
            return 1.0
        
        values = list(predictions.values())
        std = np.std(values)
        mean = np.mean(values)
        
        # Coefficient of variation (normalize edilmiş standart sapma)
        if mean == 0:
            return 0.0
        
        cv = std / abs(mean)
        
        # CV'yi 0-1 aralığına çevir (düşük CV = yüksek uyuşma)
        agreement = 1.0 / (1.0 + cv)
        
        return float(agreement)
    
    def _assess_risk(
        self,
        confidence: float,
        agreement: float,
        prediction: float
    ) -> str:
        """
        Risk seviyesini değerlendir
        
        Returns:
            'Düşük', 'Orta', 'Yüksek'
        """
        # Risk faktörleri
        risk_score = 0
        
        # Düşük güven = risk
        if confidence < 0.6:
            risk_score += 2
        elif confidence < 0.75:
            risk_score += 1
        
        # Düşük uyuşma = risk
        if agreement < 0.6:
            risk_score += 2
        elif agreement < 0.75:
            risk_score += 1
        
        # Threshold'a çok yakın = risk
        distance = abs(prediction - self.threshold)
        if distance < 0.1:
            risk_score += 2
        elif distance < 0.2:
            risk_score += 1
        
        # Risk seviyesi
        if risk_score >= 4:
            return "Yüksek"
        elif risk_score >= 2:
            return "Orta"
        else:
            return "Düşük"
    
    def _log_prediction_details(
        self,
        result: PredictionResult,
        confidences: Dict[str, float]
    ):
        """Tahmin detaylarını logla"""
        logger.info("\n" + "="*60)
        logger.info("ENSEMBLE TAHMİN DETAYLARI")
        logger.info("="*60)
        logger.info(f"Strateji: {self.strategy.value}")
        logger.info(f"\nModel Tahminleri:")
        for name, pred in result.individual_predictions.items():
            conf = confidences.get(name, 0.0)
            logger.info(f"  • {name}: {pred:.2f} (güven: {conf:.2%})")
        logger.info(f"\nFinal Tahmin: {result.value:.2f}")
        logger.info(f"Threshold Tahmini: {'1.5 ÜSTÜ ✅' if result.threshold_prediction else '1.5 ALTI ❌'}")
        logger.info(f"Güven Skoru: {result.confidence:.2%}")
        logger.info(f"Model Uyuşması: {result.model_agreement:.2%}")
        logger.info(f"Risk Seviyesi: {result.risk_level}")
        logger.info(f"Bahse Girilmeli mi: {'EVET' if result.should_bet else 'HAYIR'}")
        logger.info("="*60 + "\n")
    
    def batch_predict(
        self,
        X_batch: np.ndarray,
        return_details: bool = False
    ) -> List[PredictionResult]:
        """
        Batch tahmin
        
        Args:
            X_batch: Batch input features
            return_details: Detaylı sonuçlar
            
        Returns:
            List of PredictionResult
        """
        results = []
        for X in X_batch:
            X_reshaped = X.reshape(1, -1) if len(X.shape) == 1 else X
            result = self.predict(X_reshaped, return_details=return_details)
            results.append(result)
        return results
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """
        Model performans özetini döndür
        
        Returns:
            Performance summary dict
        """
        return {
            'strategy': self.strategy.value,
            'num_models': len(self.models),
            'model_names': list(self.models.keys()),
            'weights': self.weights,
            'threshold': self.threshold,
            'min_confidence': self.min_confidence
        }


def create_ensemble_predictor(
    catboost_regressor=None,
    catboost_classifier=None,
    nn_regressor=None,
    nn_classifier=None,
    strategy: str = 'weighted',
    custom_weights: Optional[Dict[str, float]] = None
) -> EnsemblePredictor:
    """
    Ensemble predictor factory function
    
    Args:
        catboost_regressor: CatBoost regressor model
        catboost_classifier: CatBoost classifier model
        nn_regressor: Neural network regressor model
        nn_classifier: Neural network classifier model
        strategy: Voting strategy ('unanimous', 'weighted', 'confidence', 'majority')
        custom_weights: Custom model weights (opsiyonel)
        
    Returns:
        EnsemblePredictor instance
    """
    models = {}
    
    if catboost_regressor is not None:
        models['catboost_regressor'] = catboost_regressor
    if catboost_classifier is not None:
        models['catboost_classifier'] = catboost_classifier
    if nn_regressor is not None:
        models['nn_regressor'] = nn_regressor
    if nn_classifier is not None:
        models['nn_classifier'] = nn_classifier
    
    if not models:
        raise ValueError("En az bir model sağlanmalı!")
    
    strategy_enum = VotingStrategy(strategy.lower())
    
    return EnsemblePredictor(
        models=models,
        strategy=strategy_enum,
        weights=custom_weights
    )
