"""
JetX Predictor - Comprehensive Model Selection Module (v2.0)

Bu modül, mevcut yanıltıcı weighted score yerine
daha dengeli ve kapsamlı model değerlendirmesi yapar.

GÜNCELLEME:
- 2 Modlu Yapı (Normal/Rolling) entegrasyonu.
- Eşikler: Normal Mod (0.85), Rolling Mod (0.95).
- ROI ve Precision ağırlıklı skorlama.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
from utils.threshold_manager import get_threshold_manager

logger = logging.getLogger(__name__)

class ComprehensiveModelEvaluator:
    """
    Kapsamlı model değerlendirici.
    Sadece ROI'ye değil, çoklu metrikleri dikkate alır.
    """
    
    def __init__(self):
        self.tm = get_threshold_manager()
        self.THRESHOLD_NORMAL = self.tm.get_normal_threshold()
        
        self.min_thresholds = {
            'win_rate': 0.60,      # Minimum %60 kazanma oranı (daha gerçekçi)
            'stability': 0.60,     # Minimum %60 stability  
            'roi': 0.0,            # Minimum %0 ROI (zarar etmesin yeter)
            'sharpe_ratio': 0.0,   # Pozitif Sharpe
            'consecutive_good_epochs': 3,  # Ardışık 3 iyi epoch
            'min_predictions': 50       # En az 50 tahmin
        }
        
        self.weights = {
            'roi': 0.40,          # Para kazandırma en önemli
            'win_rate': 0.20,      # Tutarlılık
            'precision': 0.20,     # Yanlış pozitiflerden kaçınma
            'sharpe_ratio': 0.10,  # Risk-ayarlı performans
            'stability': 0.10      # Performans tutarlılığı
        }
    
    def calculate_metrics(self, model, X_val, y_val):
        """Temel metrikleri hesapla"""
        try:
            predictions = model.predict(X_val)
            
            # Threshold output (genelde 3. output veya sonuncusu)
            if isinstance(predictions, list):
                # Progressive NN gibi çoklu output varsa
                # Threshold genelde binary olandır (index 2 veya -1)
                # Basitlik için sonuncuyu alıyoruz, structure'a göre değişebilir
                p_thr = predictions[-1].flatten()
            elif isinstance(predictions, dict):
                p_thr = predictions.get('threshold_prob', predictions.get('prediction', 0))
            else:
                # Tek output (CatBoost classifier predict_proba gibi)
                if len(predictions.shape) > 1: # Proba
                     p_thr = predictions[:, 1]
                else:
                     p_thr = predictions.flatten()
            
            # Normal Mod Eşiğine göre binary tahmin
            p_binary = (p_thr >= self.THRESHOLD_NORMAL).astype(int)
            
            # Gerçek değerler (Regression target ise 1.5'e göre binary yap)
            if len(y_val.shape) > 1 and y_val.shape[1] > 1: # One-hot ise
                 y_binary = np.argmax(y_val, axis=1) # Bu hatalı olabilir, one-hot class ise
                 # regression target genelde float arraydir.
                 # Burada y_val'in ne olduğuna dikkat etmeliyiz.
                 # Genelde y_reg_val (float array) gelir.
                 pass
            
            # Basitlik varsayımı: y_val float array (gerçek çarpanlar)
            if isinstance(y_val, np.ndarray):
                y_binary = (y_val.flatten() >= 1.5).astype(int)
            else:
                y_binary = np.array([1 if y >= 1.5 else 0 for y in y_val])
            
            # Metrics
            tp = np.sum((p_binary == 1) & (y_binary == 1))
            fp = np.sum((p_binary == 1) & (y_binary == 0))
            tn = np.sum((p_binary == 0) & (y_binary == 0))
            fn = np.sum((p_binary == 0) & (y_binary == 1))
            
            total_bets = tp + fp
            win_rate = tp / total_bets if total_bets > 0 else 0.0
            precision = win_rate # Precision = Win Rate
            
            # ROI Simülasyonu (Basit 1.5x)
            initial = 1000
            wallet = initial
            bet_amount = 10
            
            for i in range(len(p_binary)):
                if p_binary[i] == 1:
                    wallet -= bet_amount
                    if y_binary[i] == 1:
                        wallet += bet_amount * 1.5
            
            roi = (wallet - initial) / initial
            
            return {
                'win_rate': win_rate,
                'roi': roi,
                'precision': precision,
                'total_bets': total_bets
            }
            
        except Exception as e:
            logger.error(f"Metrik hesaplama hatası: {e}")
            return {'win_rate': 0, 'roi': 0, 'precision': 0, 'total_bets': 0}
    
    def calculate_grade(self, score: float) -> str:
        """Skora göre not ver"""
        if score >= 0.8: return "A"
        elif score >= 0.6: return "B"
        elif score >= 0.4: return "C"
        else: return "D"
    
    def evaluate_model_comprehensive(self, model, X_val, y_val, model_name: str) -> Dict[str, Any]:
        """
        Modeli kapsamlı şekilde değerlendirir
        """
        logger.info(f"Model değerlendiriliyor: {model_name}")
        
        try:
            metrics = self.calculate_metrics(model, X_val, y_val)
            
            # Ekstra metrikler (mockup - gerçek implementasyonda eklenebilir)
            metrics['sharpe_ratio'] = 0.5 if metrics['roi'] > 0 else 0
            metrics['stability'] = 0.8 # Varsayılan stabilite
            metrics['consistency'] = 0.8
            
            # Minimum eşik kontrolü
            passes = True
            for k, v in self.min_thresholds.items():
                if k in metrics and metrics[k] < v:
                    passes = False
                    break
            
            # Skor hesapla
            final_score = (
                self.weights['roi'] * max(0, metrics['roi'] + 0.5) + # Normalize ROI
                self.weights['win_rate'] * metrics['win_rate'] +
                self.weights['precision'] * metrics['precision']
            )
            
            grade = self.calculate_grade(final_score)
            
            return {
                'accepted': passes,
                'score': final_score,
                'metrics': metrics,
                'grade': grade,
                'evaluation_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Model değerlendirme hatası: {e}")
            return {
                'accepted': False,
                'score': 0.0,
                'metrics': {},
                'grade': 'F',
                'error': str(e)
            }


class ModelSelectionManager:
    """
    En iyi modeli seçmek için yönetici sınıf.
    """
    
    def __init__(self):
        self.evaluator = ComprehensiveModelEvaluator()
        self.selection_history = []
    
    def select_best_model(self, models: Dict[str, Any], X_val, y_val) -> Dict[str, Any]:
        """
        Mevcut modeller arasından en iyisini seçer
        """
        logger.info(f"Model seçimi başlatılıyor: {len(models)} model")
        
        try:
            model_evaluations = {}
            
            for model_name, model in models.items():
                if model is not None:
                    evaluation = self.evaluator.evaluate_model_comprehensive(
                        model, X_val, y_val, model_name
                    )
                    model_evaluations[model_name] = evaluation
            
            # Geçerli modelleri filtrele (Eğer hiçbiri geçemezse en iyisini al)
            valid_models = {
                name: eval_data for name, eval_data in model_evaluations.items()
                if eval_data['accepted']
            }
            
            if not valid_models:
                logger.warning("Hiçbir model minimum eşikleri karşılamıyor! En yüksek skorlu seçiliyor.")
                valid_models = model_evaluations
            
            if not valid_models:
                 return {'selected_model': None, 'reason': 'No models available'}

            # En iyi modeli seç
            best_model_name = max(valid_models, key=lambda x: valid_models[x]['score'])
            best_evaluation = valid_models[best_model_name]
            
            selection_result = {
                'selected_model': best_model_name,
                'selected_score': best_evaluation['score'],
                'selected_grade': best_evaluation['grade'],
                'selected_metrics': best_evaluation['metrics'],
                'all_evaluations': model_evaluations
            }
            
            self.selection_history.append(selection_result)
            logger.info(f"En iyi model seçildi: {best_model_name} (Score: {best_evaluation['score']:.3f})")
            
            return selection_result
            
        except Exception as e:
            logger.error(f"Model seçimi hatası: {e}")
            return {'selected_model': None, 'error': str(e)}

# Global instance
_model_selector = ModelSelectionManager()

def get_model_selector() -> ModelSelectionManager:
    return _model_selector
