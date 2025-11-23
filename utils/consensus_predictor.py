"""
Consensus Predictor - NN ve CatBoost Ensemble Modellerinin Consensus Tahminleri

GÜNCELLEME:
- Consensus kararı için %85 güven eşiği şart koşuldu.
- "Keskin Nişancı" modu aktif.
"""

import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, Tuple, List, Optional
import logging

# ... (Import kısımları aynı kalacak) ...
# (Burayı kısaltıyorum, dosyanın tamamını aşağıda vereceğim)

class ConsensusPredictor:
    # ... (__init__ ve load fonksiyonları aynı) ...

    # ... (extract_features fonksiyonu aynı) ...

    def predict_nn_ensemble(self, data: np.ndarray) -> Dict:
        # ... (başlangıç aynı) ...
        
        # Ensemble ortalama
        avg_reg = np.mean(ensemble_reg)
        avg_thr = np.mean(ensemble_thr)
        
        # GÜNCELLEME: %85 Eşik
        # Threshold: 1.5 üstü mü? (Sadece %85 üzeri güvenle)
        threshold_prediction = 1 if avg_thr >= 0.85 else 0
        
        return {
            'regression': float(avg_reg),
            'threshold': threshold_prediction,
            'threshold_prob': float(avg_thr),
            # ...
        }
    
    def predict_catboost_ensemble(self, data: np.ndarray) -> Dict:
        # ... (başlangıç aynı) ...
        
        # Ensemble ortalama
        avg_reg = np.mean(ensemble_reg)
        # GÜNCELLEME: Sınıf tahminlerini değil, olasılıkları kullanmalıyız
        # Ancak bu yapıda direkt sınıf dönüyorsa, ortalaması 0.85 üstü olmalı
        # (Örn: 10 modelin 9'u 'üst' demeli)
        avg_cls_score = np.mean(ensemble_cls) 
        
        threshold_prediction = 1 if avg_cls_score >= 0.85 else 0
        
        return {
            'regression': float(avg_reg),
            'threshold': threshold_prediction,
            # ...
        }
    
    def predict_consensus(self, data: np.ndarray) -> Dict:
        # ...
        # Consensus kontrolü: İki model de 1.5 üstü (ve %85 güvenli) mü?
        nn_threshold = nn_result['threshold']
        catboost_threshold = catboost_result['threshold']
        
        # İkisi de 1 ("Üst") dediyse, ikisi de %85 barajını geçmiş demektir.
        consensus = (nn_threshold == 1 and catboost_threshold == 1)
        
        # ...
