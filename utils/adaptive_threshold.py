"""
Adaptive Threshold Manager - Dinamik Eşik Yönetimi

GÜNCELLEME:
- Eşikler 0.85 (Normal) ve 0.95 (Rolling) olarak sabitlendi.
- Dinamik yapı, "eşik değiştirme" yerine "mod seçimi" yapacak şekilde evrildi.
- %85 altı güven skorlarında bahis yapılmaz (None döner).
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
    mode: str = 'normal'  # 'normal' veya 'rolling'


class AdaptiveThresholdManager:
    """
    Dinamik threshold yönetimi (Adaptif Mod Seçimi)
    
    Güven skoruna ve model uyuşmasına göre Normal (%85) veya Rolling (%95)
    modunu seçer.
    """
    
    def __init__(
        self,
        base_threshold: float = 1.5,
        min_confidence: float = 0.85, # Güncellendi: 0.85
        max_threshold: float = 2.0,
        history_window: int = 100,
        config_path: Optional[str] = None
    ):
        """
        Args:
            base_threshold: Temel threshold değeri (1.5x)
            min_confidence: Minimum güven skoru (0.85)
            max_threshold: Maximum threshold değeri
            history_window: Geçmiş performans penceresi
            config_path: Konfigürasyon dosyası yolu
        """
        self.base_threshold = base_threshold
        self.min_confidence = min_confidence
        self.max_threshold = max_threshold
        self.history_window = history_window
        
        # Sabit Eşikler
        self.THRESHOLD_NORMAL = 0.85
        self.THRESHOLD_ROLLING = 0.95
        
        # Threshold Haritası (Güven -> Hedef Çarpan)
        self.threshold_map = {
            (0.95, 1.01): 1.50,  # Rolling Mod (%95+) -> 1.50x
            (0.85, 0.95): 1.65,  # Normal Mod (%85-95) -> 1.65x (Daha esnek)
            (0.00, 0.85): None   # Düşük Güven -> Oynama
        }
        
        # Performans geçmişi
        self.prediction_history = deque(maxlen=history_window)
        self.performance_metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'win_rate': 0.0,
            'avg_confidence': 0.0,
            'false_positives': 0, 
            'false_negatives': 0
        }
        
        if config_path:
            self.load_config(config_path)
            
        logger.info(f"Adaptive Threshold Manager oluşturuldu:")
        logger.info(f"  • Normal Threshold: {self.THRESHOLD_NORMAL}")
        logger.info(f"  • Rolling Threshold: {self.THRESHOLD_ROLLING}")
    
    def get_threshold(
        self,
        confidence: float,
        model_agreement: float,
        prediction: float,
        strategy: str = 'hybrid' # Legacy support
    ) -> ThresholdDecision:
        """
        Güven skoruna göre mod ve threshold belirle
        """
        # Adjusted confidence
        adjusted_confidence = confidence * 0.7 + model_agreement * 0.3
        
        # Karar mantığı
        threshold = None
        mode = 'normal'
        risk_level = 'Yüksek'
        should_bet = False
        reasoning = ""
        
        if adjusted_confidence >= self.THRESHOLD_ROLLING:
            # ROLLING MOD
            threshold = 1.50
            mode = 'rolling'
            risk_level = 'Düşük'
            should_bet = True
            reasoning = f"Güven çok yüksek ({adjusted_confidence:.2%}), Rolling Mod aktif"
            
        elif adjusted_confidence >= self.THRESHOLD_NORMAL:
            # NORMAL MOD
            threshold = 1.65 # Biraz daha yüksek hedef
            mode = 'normal'
            risk_level = 'Orta'
            should_bet = True
            reasoning = f"Güven yeterli ({adjusted_confidence:.2%}), Normal Mod aktif"
            
        else:
            # BAHİS YOK
            threshold = None
            risk_level = 'Çok Yüksek'
            should_bet = False
            reasoning = f"Güven yetersiz ({adjusted_confidence:.2%} < 85%)"
        
        return ThresholdDecision(
            threshold=threshold,
            confidence=adjusted_confidence,
            should_bet=should_bet,
            risk_level=risk_level,
            reasoning=reasoning,
            mode=mode
        )

    def update_history(
        self,
        prediction: float,
        actual: float,
        threshold_used: Optional[float],
        confidence: float,
        bet_placed: bool
    ):
        """Geçmiş performansı güncelle"""
        correct = False
        if threshold_used is not None and bet_placed:
            correct = (actual >= threshold_used)
        
        self.prediction_history.append({
            'prediction': prediction,
            'actual': actual,
            'threshold_used': threshold_used,
            'confidence': confidence,
            'correct': correct,
            'bet_placed': bet_placed
        })
        
        self._update_metrics()
    
    def _update_metrics(self):
        """Performans metriklerini yeniden hesapla"""
        if not self.prediction_history: return
        
        total = len(self.prediction_history)
        bets = sum(1 for h in self.prediction_history if h['bet_placed'])
        correct = sum(1 for h in self.prediction_history if h['correct'] and h['bet_placed'])
        
        fp = sum(1 for h in self.prediction_history 
                 if h['bet_placed'] and not h['correct'])
                 
        self.performance_metrics.update({
            'total_predictions': total,
            'correct_predictions': correct,
            'win_rate': correct / bets if bets > 0 else 0.0,
            'avg_confidence': np.mean([h['confidence'] for h in self.prediction_history]),
            'false_positives': fp
        })

    def get_performance_summary(self) -> Dict:
        """Performans özetini döndür"""
        return self.performance_metrics.copy()
    
    def save_config(self, filepath: str):
        """Konfigürasyonu kaydet"""
        config = {
            'base_threshold': self.base_threshold,
            'min_confidence': self.min_confidence,
            'max_threshold': self.max_threshold,
            'performance_metrics': self.performance_metrics
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
            
    def load_config(self, filepath: str):
        """Konfigürasyonu yükle"""
        if not Path(filepath).exists(): return
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        self.base_threshold = config.get('base_threshold', self.base_threshold)
        self.min_confidence = config.get('min_confidence', self.min_confidence)
        
        if 'performance_metrics' in config:
            self.performance_metrics.update(config['performance_metrics'])

    def reset_history(self):
        """Geçmişi sıfırla"""
        self.prediction_history.clear()
        self.performance_metrics = {
            'total_predictions': 0, 'correct_predictions': 0,
            'win_rate': 0.0, 'avg_confidence': 0.0,
            'false_positives': 0, 'false_negatives': 0
        }

def create_threshold_manager(
    base_threshold: float = 1.5,
    strategy: str = 'hybrid',
    config_path: Optional[str] = None
) -> AdaptiveThresholdManager:
    return AdaptiveThresholdManager(
        base_threshold=base_threshold,
        config_path=config_path
    )
