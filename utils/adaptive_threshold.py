"""
Adaptive Threshold Manager - Dinamik Eşik Yönetimi
Güven skoruna ve model performansına göre threshold'u dinamik olarak ayarlar.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ThresholdDecision:
    """Threshold kararı"""
    threshold: Optional[float]  # None = bahse girme
    confidence: float
    should_bet: bool
    risk_level: str
    reasoning: str
    fallback_threshold: float = 1.7  # Yedek threshold


class AdaptiveThresholdManager:
    """
    Dinamik threshold yönetimi
    
    Güven skoruna, model uyuşmasına ve geçmiş performansa göre
    threshold'u otomatik olarak ayarlar.
    
    Stratejiler:
    1. Confidence-based: Güven skoruna göre
    2. Performance-based: Geçmiş performansa göre
    3. Hybrid: Her ikisinin kombinasyonu
    """
    
    def __init__(
        self,
        base_threshold: float = 1.5,
        min_confidence: float = 0.5,
        max_threshold: float = 2.0,
        history_window: int = 100,
        config_path: Optional[str] = None
    ):
        """
        Args:
            base_threshold: Temel threshold değeri
            min_confidence: Minimum güven skoru
            max_threshold: Maximum threshold değeri
            history_window: Geçmiş performans penceresi
            config_path: Konfigürasyon dosyası yolu
        """
        self.base_threshold = base_threshold
        self.min_confidence = min_confidence
        self.max_threshold = max_threshold
        self.history_window = history_window
        
        # Threshold haritası (confidence range -> threshold)
        self.threshold_map = {
            (0.90, 1.00): 1.50,  # Çok yüksek güven
            (0.80, 0.90): 1.55,  # Yüksek güven
            (0.70, 0.80): 1.60,  # Orta-yüksek güven
            (0.60, 0.70): 1.65,  # Orta güven
            (0.50, 0.60): 1.70,  # Düşük güven
            (0.00, 0.50): None   # Çok düşük → bahse girme
        }
        
        # Performans geçmişi
        self.prediction_history = deque(maxlen=history_window)
        self.performance_metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'win_rate': 0.0,
            'avg_confidence': 0.0,
            'false_positives': 0,  # 1.5 altı ama tahmin 1.5 üstü
            'false_negatives': 0   # 1.5 üstü ama tahmin 1.5 altı
        }
        
        # Konfigürasyon yükle
        if config_path:
            self.load_config(config_path)
        
        logger.info(f"Adaptive Threshold Manager oluşturuldu:")
        logger.info(f"  • Base threshold: {base_threshold}")
        logger.info(f"  • Min confidence: {min_confidence}")
        logger.info(f"  • Max threshold: {max_threshold}")
        logger.info(f"  • History window: {history_window}")
    
    def get_threshold(
        self,
        confidence: float,
        model_agreement: float,
        prediction: float,
        strategy: str = 'hybrid'
    ) -> ThresholdDecision:
        """
        Dinamik threshold hesapla
        
        Args:
            confidence: Model güven skoru (0-1)
            model_agreement: Model uyuşma skoru (0-1)
            prediction: Tahmin edilen değer
            strategy: 'confidence', 'performance', 'hybrid'
            
        Returns:
            ThresholdDecision object
        """
        if strategy == 'confidence':
            decision = self._confidence_based_threshold(confidence, model_agreement, prediction)
        elif strategy == 'performance':
            decision = self._performance_based_threshold(confidence, model_agreement, prediction)
        else:  # hybrid
            decision = self._hybrid_threshold(confidence, model_agreement, prediction)
        
        return decision
    
    def _confidence_based_threshold(
        self,
        confidence: float,
        model_agreement: float,
        prediction: float
    ) -> ThresholdDecision:
        """
        Sadece güven skoruna göre threshold belirle
        
        Returns:
            ThresholdDecision
        """
        # Adjusted confidence (model uyuşmasını da hesaba kat)
        adjusted_confidence = confidence * 0.7 + model_agreement * 0.3
        
        # Threshold haritasından threshold bul
        threshold = None
        for (min_conf, max_conf), thresh in self.threshold_map.items():
            if min_conf <= adjusted_confidence < max_conf:
                threshold = thresh
                break
        
        # Risk değerlendirmesi
        if threshold is None:
            risk_level = "Çok Yüksek"
            should_bet = False
            reasoning = f"Güven çok düşük ({adjusted_confidence:.2%}), bahse girilmemeli"
        else:
            # Tahmin ile threshold arası mesafe
            distance = abs(prediction - threshold)
            
            if adjusted_confidence >= 0.80 and distance > 0.2:
                risk_level = "Düşük"
            elif adjusted_confidence >= 0.60 and distance > 0.1:
                risk_level = "Orta"
            else:
                risk_level = "Yüksek"
            
            should_bet = adjusted_confidence >= self.min_confidence
            reasoning = f"Güven: {adjusted_confidence:.2%}, Threshold: {threshold}x"
        
        return ThresholdDecision(
            threshold=threshold,
            confidence=adjusted_confidence,
            should_bet=should_bet,
            risk_level=risk_level,
            reasoning=reasoning
        )
    
    def _performance_based_threshold(
        self,
        confidence: float,
        model_agreement: float,
        prediction: float
    ) -> ThresholdDecision:
        """
        Geçmiş performansa göre threshold ayarla
        
        Returns:
            ThresholdDecision
        """
        # Eğer yeterli geçmiş yoksa, confidence-based kullan
        if len(self.prediction_history) < 20:
            return self._confidence_based_threshold(confidence, model_agreement, prediction)
        
        # Geçmiş performansı hesapla
        recent_win_rate = self.performance_metrics['win_rate']
        
        # Performansa göre threshold ayarla
        if recent_win_rate >= 0.75:
            # Çok iyi performans → agresif (düşük threshold)
            threshold = self.base_threshold
            risk_level = "Düşük"
            reasoning = f"Yüksek kazanma oranı ({recent_win_rate:.1%}), agresif threshold"
        elif recent_win_rate >= 0.67:
            # İyi performans → normal
            threshold = self.base_threshold + 0.1
            risk_level = "Orta"
            reasoning = f"Normal kazanma oranı ({recent_win_rate:.1%}), standart threshold"
        elif recent_win_rate >= 0.60:
            # Orta performans → temkinli
            threshold = self.base_threshold + 0.2
            risk_level = "Orta-Yüksek"
            reasoning = f"Düşük kazanma oranı ({recent_win_rate:.1%}), temkinli threshold"
        else:
            # Kötü performans → çok temkinli veya dur
            if recent_win_rate < 0.50:
                threshold = None
                risk_level = "Çok Yüksek"
                reasoning = f"Çok düşük kazanma oranı ({recent_win_rate:.1%}), bahse girilmemeli"
            else:
                threshold = self.base_threshold + 0.3
                risk_level = "Yüksek"
                reasoning = f"Düşük kazanma oranı ({recent_win_rate:.1%}), çok temkinli"
        
        should_bet = threshold is not None and confidence >= self.min_confidence
        
        return ThresholdDecision(
            threshold=threshold,
            confidence=confidence,
            should_bet=should_bet,
            risk_level=risk_level,
            reasoning=reasoning
        )
    
    def _hybrid_threshold(
        self,
        confidence: float,
        model_agreement: float,
        prediction: float
    ) -> ThresholdDecision:
        """
        Hem güven hem de performans bazlı (hybrid)
        
        Returns:
            ThresholdDecision
        """
        # Her iki metodu çalıştır
        conf_decision = self._confidence_based_threshold(confidence, model_agreement, prediction)
        perf_decision = self._performance_based_threshold(confidence, model_agreement, prediction)
        
        # Eğer yeterli geçmiş yoksa, sadece confidence kullan
        if len(self.prediction_history) < 20:
            return conf_decision
        
        # İkisinden de threshold varsa, daha yüksek olanı kullan (daha güvenli)
        if conf_decision.threshold is not None and perf_decision.threshold is not None:
            threshold = max(conf_decision.threshold, perf_decision.threshold)
        elif conf_decision.threshold is not None:
            threshold = conf_decision.threshold
        elif perf_decision.threshold is not None:
            threshold = perf_decision.threshold
        else:
            threshold = None
        
        # Risk seviyesi (daha yüksek olanı al)
        risk_levels = ["Düşük", "Orta", "Orta-Yüksek", "Yüksek", "Çok Yüksek"]
        conf_risk_idx = risk_levels.index(conf_decision.risk_level)
        perf_risk_idx = risk_levels.index(perf_decision.risk_level)
        risk_level = risk_levels[max(conf_risk_idx, perf_risk_idx)]
        
        # Adjusted confidence
        adjusted_confidence = confidence * 0.7 + model_agreement * 0.3
        
        should_bet = (
            threshold is not None and
            adjusted_confidence >= self.min_confidence and
            conf_decision.should_bet and
            perf_decision.should_bet
        )
        
        reasoning = f"Hybrid: {conf_decision.reasoning} + {perf_decision.reasoning}"
        
        return ThresholdDecision(
            threshold=threshold,
            confidence=adjusted_confidence,
            should_bet=should_bet,
            risk_level=risk_level,
            reasoning=reasoning
        )
    
    def update_history(
        self,
        prediction: float,
        actual: float,
        threshold_used: Optional[float],
        confidence: float,
        bet_placed: bool
    ):
        """
        Geçmiş performansı güncelle
        
        Args:
            prediction: Tahmin edilen değer
            actual: Gerçek değer
            threshold_used: Kullanılan threshold
            confidence: Güven skoru
            bet_placed: Bahse girildi mi?
        """
        # Tahmin doğru mu?
        if threshold_used is not None and bet_placed:
            # Bahse girildiyse ve threshold doğru tahmin edildiyse
            correct = (actual >= threshold_used) if (prediction >= threshold_used) else (actual < threshold_used)
        else:
            # Bahse girilmediyse, değer tahmini doğruluğuna bak
            error = abs(prediction - actual)
            correct = error < 0.5  # 0.5 tolerans
        
        # Geçmişe ekle
        self.prediction_history.append({
            'prediction': prediction,
            'actual': actual,
            'threshold_used': threshold_used,
            'confidence': confidence,
            'correct': correct,
            'bet_placed': bet_placed
        })
        
        # Metrikleri güncelle
        self._update_metrics()
    
    def _update_metrics(self):
        """Performans metriklerini yeniden hesapla"""
        if not self.prediction_history:
            return
        
        total = len(self.prediction_history)
        correct = sum(1 for h in self.prediction_history if h['correct'])
        
        # False positives/negatives
        fp = sum(1 for h in self.prediction_history 
                if h['threshold_used'] is not None and
                   h['prediction'] >= h['threshold_used'] and
                   h['actual'] < h['threshold_used'])
        
        fn = sum(1 for h in self.prediction_history 
                if h['threshold_used'] is not None and
                   h['prediction'] < h['threshold_used'] and
                   h['actual'] >= h['threshold_used'])
        
        # Metrikleri güncelle
        self.performance_metrics.update({
            'total_predictions': total,
            'correct_predictions': correct,
            'win_rate': correct / total if total > 0 else 0.0,
            'avg_confidence': np.mean([h['confidence'] for h in self.prediction_history]),
            'false_positives': fp,
            'false_negatives': fn
        })
        
        logger.debug(f"Metrics updated: Win rate={self.performance_metrics['win_rate']:.2%}")
    
    def get_performance_summary(self) -> Dict:
        """
        Performans özetini döndür
        
        Returns:
            Performance summary dict
        """
        if not self.prediction_history:
            return {
                'status': 'Henüz veri yok',
                'total_predictions': 0
            }
        
        recent_10 = list(self.prediction_history)[-10:] if len(self.prediction_history) >= 10 else list(self.prediction_history)
        recent_win_rate = sum(1 for h in recent_10 if h['correct']) / len(recent_10)
        
        return {
            'total_predictions': self.performance_metrics['total_predictions'],
            'overall_win_rate': self.performance_metrics['win_rate'],
            'recent_win_rate': recent_win_rate,
            'avg_confidence': self.performance_metrics['avg_confidence'],
            'false_positives': self.performance_metrics['false_positives'],
            'false_negatives': self.performance_metrics['false_negatives'],
            'false_positive_rate': self.performance_metrics['false_positives'] / max(1, self.performance_metrics['total_predictions']),
            'recommendation': self._get_recommendation()
        }
    
    def _get_recommendation(self) -> str:
        """Performansa göre öneri"""
        win_rate = self.performance_metrics['win_rate']
        fp_rate = self.performance_metrics['false_positives'] / max(1, self.performance_metrics['total_predictions'])
        
        if win_rate >= 0.75:
            return "✅ Mükemmel performans! Devam edin."
        elif win_rate >= 0.67:
            return "✅ İyi performans. Başabaş noktasında veya üstünde."
        elif win_rate >= 0.60:
            return "⚠️ Orta performans. Dikkatli olun."
        elif fp_rate > 0.30:
            return "❌ Yüksek false positive! Para kaybı riski yüksek."
        else:
            return "❌ Düşük performans. Bahisleri azaltın veya durdurun."
    
    def save_config(self, filepath: str):
        """Konfigürasyonu kaydet"""
        config = {
            'base_threshold': self.base_threshold,
            'min_confidence': self.min_confidence,
            'max_threshold': self.max_threshold,
            'threshold_map': {f"{k[0]}-{k[1]}": v for k, v in self.threshold_map.items()},
            'performance_metrics': self.performance_metrics
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Konfigürasyon kaydedildi: {filepath}")
    
    def load_config(self, filepath: str):
        """Konfigürasyonu yükle"""
        if not Path(filepath).exists():
            logger.warning(f"Konfigürasyon dosyası bulunamadı: {filepath}")
            return
        
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        self.base_threshold = config.get('base_threshold', self.base_threshold)
        self.min_confidence = config.get('min_confidence', self.min_confidence)
        self.max_threshold = config.get('max_threshold', self.max_threshold)
        
        if 'performance_metrics' in config:
            self.performance_metrics.update(config['performance_metrics'])
        
        logger.info(f"Konfigürasyon yüklendi: {filepath}")
    
    def reset_history(self):
        """Geçmişi sıfırla"""
        self.prediction_history.clear()
        self.performance_metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'win_rate': 0.0,
            'avg_confidence': 0.0,
            'false_positives': 0,
            'false_negatives': 0
        }
        logger.info("Performans geçmişi sıfırlandı")


def create_threshold_manager(
    base_threshold: float = 1.5,
    strategy: str = 'hybrid',
    config_path: Optional[str] = None
) -> AdaptiveThresholdManager:
    """
    Threshold manager factory function
    
    Args:
        base_threshold: Temel threshold değeri
        strategy: Varsayılan strateji
        config_path: Konfigürasyon dosyası
        
    Returns:
        AdaptiveThresholdManager instance
    """
    return AdaptiveThresholdManager(
        base_threshold=base_threshold,
        config_path=config_path
    )