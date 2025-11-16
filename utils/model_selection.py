"""
JetX Predictor - Comprehensive Model Selection Module

Bu modül, mevcut yanıltıcı weighted score yerine
daha dengeli ve kapsamlı model değerlendirmesi yapar.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ComprehensiveModelEvaluator:
    """
    Kapsamlı model değerlendirici.
    Sadece ROI'ye değil, çoklu metrikleri dikkate alır.
    """
    
    def __init__(self):
        self.min_thresholds = {
            'win_rate': 0.65,      # Minimum %65 kazanma oranı
            'stability': 0.70,      # Minimum %70 stability  
            'roi': 0.10,            # Minimum %10 ROI
            'sharpe_ratio': 0.5,    # Minimum 0.5 Sharpe
            'consecutive_good_epochs': 3,  # Ardışık 3 iyi epoch
            'min_predictions': 50       # En az 50 tahmin
        }
        
        self.weights = {
            'roi': 0.25,           # Para kazandırma en önemli
            'win_rate': 0.25,      # Tutarlılık çok önemli
            'sharpe_ratio': 0.25,   # Risk-ayarlı performans
            'stability': 0.15,      # Performans tutarlılığı
            'consistency': 0.10      # Zaman içinde tutarlılık
        }
    
    def calculate_win_rate(self, model, X_val, y_val):
        """Win rate hesapla"""
        try:
            predictions = model.predict(X_val)
            if isinstance(predictions, dict):
                predictions = predictions.get('predicted_value', predictions)
            
            # Binary classification için (1.5x üstü/altı)
            if len(y_val.shape) == 1:
                # Regression
                y_binary = (y_val >= 1.5).astype(int)
                pred_binary = (predictions >= 1.5).astype(int)
            else:
                # Multi-output classification - DÜZELTME: Shape kontrolü eklendi
                try:
                    y_binary = np.argmax(y_val, axis=1)
                    pred_binary = np.argmax(predictions, axis=1)
                except (ValueError, IndexError) as shape_error:
                    logger.error(f"Shape hatası: {shape_error}")
                    # Fallback: Basit binary conversion
                    if len(predictions.shape) == 1:
                        pred_binary = (predictions.flatten() >= 1.5).astype(int)
                    else:
                        pred_binary = np.argmax(predictions, axis=1) if len(predictions.shape) > 1 else (predictions.flatten() >= 1.5).astype(int)
                    
                    if len(y_val.shape) == 1:
                        y_binary = (y_val >= 1.5).astype(int)
                    else:
                        y_binary = np.argmax(y_val, axis=1) if len(y_val.shape) > 1 else (y_val >= 1.5).astype(int)
            
            accuracy = np.mean(y_binary == pred_binary)
            return accuracy
            
        except Exception as e:
            logger.error(f"Win rate hesaplama hatası: {e}")
            return 0.0
    
    def calculate_roi(self, model, X_val, y_val):
        """ROI hesapla (sanal bahis simülasyonu)"""
        try:
            predictions = model.predict(X_val)
            if isinstance(predictions, dict):
                predictions = predictions.get('predicted_value', predictions)
            
            # Basit ROI hesapla (1.5x eşik varsayımı)
            wins = 0
            losses = 0
            total_bet = 10.0  # Sabit bahis
            
            for i, (pred, actual) in enumerate(zip(predictions, y_val)):
                if pred >= 1.5 and actual >= 1.5:
                    # Kazanç
                    wins += 1
                    profit = actual - 1.5
                    losses += total_bet  # Bahis miktarı
                elif pred >= 1.5 and actual < 1.5:
                    # Kayıp (yanlış tahmin)
                    losses += total_bet
                elif pred < 1.5 and actual >= 1.5:
                    # Kaçırılan kazanç (beklemediği)
                    # Bu durumu ROI'yi düşürür (fırsat kaçırma)
                    pass
            
            total_invested = (wins + losses) * total_bet
            total_returned = wins * total_bet * 1.5  # 1.5x varsayımı
            
            roi = (total_returned - total_invested) / total_invested if total_invested > 0 else 0
            return roi
            
        except Exception as e:
            logger.error(f"ROI hesaplama hatası: {e}")
            return 0.0
    
    def calculate_sharpe_ratio(self, model, X_val, y_val):
        """Sharpe ratio hesapla"""
        try:
            predictions = model.predict(X_val)
            if isinstance(predictions, dict):
                predictions = predictions.get('predicted_value', predictions)
            
            # Günlük getiris (yıllık %15 varsayımı)
            risk_free_rate = 0.15 / 252  # Günlük risk-free oran
            
            # Returns hesapla
            returns = []
            for pred, actual in zip(predictions, y_val):
                if pred >= 1.5 and actual >= 1.5:
                    returns.append((actual - 1.5) / 1.5)
                else:
                    returns.append(-0.1)  # Bahis kaybı
            
            if len(returns) < 2:
                return 0.0
            
            excess_returns = [r - risk_free_rate for r in returns]
            avg_excess_return = np.mean(excess_returns)
            std_excess_return = np.std(excess_returns)
            
            sharpe = avg_excess_return / std_excess_return if std_excess_return > 0 else 0.0
            return sharpe
            
        except Exception as e:
            logger.error(f"Sharpe ratio hesaplama hatası: {e}")
            return 0.0
    
    def calculate_stability(self, model):
        """Model stability'sini hesapla"""
        try:
            if hasattr(model, 'training_history') and model.training_history:
                history = model.training_history
                
                if 'val_accuracy' in history:
                    accuracies = [epoch['val_accuracy'] for epoch in history if 'val_accuracy' in epoch]
                    if len(accuracies) >= 10:
                        # Son 10 epoch'un standart sapması
                        recent_accuracies = accuracies[-10:]
                        stability = 1.0 - (np.std(recent_accuracies) / np.mean(recent_accuracies))
                        return stability
                
            return 0.5  # Varsayılan stability
            
        except Exception as e:
            logger.error(f"Stability hesaplama hatası: {e}")
            return 0.5
    
    def calculate_consistency(self, model, X_val, y_val):
        """Model tutarlılığını hesapla"""
        try:
            predictions = model.predict(X_val)
            if isinstance(predictions, dict):
                predictions = predictions.get('predicted_value', predictions)
            
            # Accuracy volatility (düşük standart sapma = daha tutarlı)
            if hasattr(model, 'training_history') and model.training_history:
                history = model.training_history
                if 'val_accuracy' in history:
                    accuracies = [epoch['val_accuracy'] for epoch in history if 'val_accuracy' in epoch]
                    if len(accuracies) >= 5:
                        consistency = 1.0 - (np.std(accuracies[-5:]) / np.mean(accuracies[-5:]))
                        return max(0.0, consistency)
                
            return 0.5  # Varsayılan consistency
            
        except Exception as e:
            logger.error(f"Consistency hesaplama hatası: {e}")
            return 0.5
    
    def passes_minimum_thresholds(self, metrics: Dict[str, float]) -> bool:
        """Modelin minimum eşikleri geçip geçmediğini kontrol et"""
        for metric, threshold in self.min_thresholds.items():
            if metrics.get(metric, 0) < threshold:
                logger.warning(f"Model minimum eşik geçemedi: {metric}={metrics.get(metric):.3f} < {threshold}")
                return False
        return True
    
    def calculate_grade(self, score: float) -> str:
        """Skora göre not ver"""
        if score >= 0.8:
            return "A"
        elif score >= 0.6:
            return "B"
        elif score >= 0.4:
            return "C"
        else:
            return "D"
    
    def evaluate_model_comprehensive(self, model, X_val, y_val, model_name: str) -> Dict[str, Any]:
        """
        Modeli kapsamlı şekilde değerlendirir
        
        Args:
            model: Değerlendirilecek model
            X_val: Validation features
            y_val: Validation targets
            model_name: Model adı
            
        Returns:
            Dict with comprehensive evaluation results
        """
        logger.info(f"Model değerlendiriliyor: {model_name}")
        
        try:
            # Temel metrikler
            win_rate = self.calculate_win_rate(model, X_val, y_val)
            roi = self.calculate_roi(model, X_val, y_val)
            sharpe_ratio = self.calculate_sharpe_ratio(model, X_val, y_val)
            stability = self.calculate_stability(model)
            consistency = self.calculate_consistency(model, X_val, y_val)
            
            metrics = {
                'win_rate': win_rate,
                'roi': roi,
                'sharpe_ratio': sharpe_ratio,
                'stability': stability,
                'consistency': consistency
            }
            
            # Minimum eşik kontrolü
            passes_thresholds = self.passes_minimum_thresholds(metrics)
            
            # Dengeli skor hesapla
            if passes_thresholds:
                final_score = (
                    self.weights['roi'] * roi +
                    self.weights['win_rate'] * win_rate +
                    self.weights['sharpe_ratio'] * sharpe_ratio +
                    self.weights['stability'] * stability +
                    self.weights['consistency'] * consistency
                )
            else:
                final_score = 0.0
            
            grade = self.calculate_grade(final_score)
            
            return {
                'accepted': passes_thresholds,
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
                'error': str(e),
                'evaluation_time': datetime.now().isoformat()
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
        
        Args:
            models: Model dict (name: model)
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dict with selection results
        """
        logger.info(f"Model seçimi başlatılıyor: {len(models)} model")
        
        try:
            # Tüm modelleri değerlendir
            model_evaluations = {}
            
            for model_name, model in models.items():
                if model is not None:
                    evaluation = self.evaluator.evaluate_model_comprehensive(
                        model, X_val, y_val, model_name
                    )
                    model_evaluations[model_name] = evaluation
            
            # Geçerli modelleri filtrele
            valid_models = {
                name: eval_data for name, eval_data in model_evaluations.items()
                if eval_data['accepted']
            }
            
            if not valid_models:
                logger.warning("Hiçbir model minimum eşikleri karşılamıyor!")
                return {
                    'selected_model': None,
                    'reason': 'Hiçbir model kabul edilebilir',
                    'all_evaluations': model_evaluations
                }
            
            # En iyi modeli seç
            best_model_name = max(valid_models, key=lambda x: valid_models[x]['score'])
            best_evaluation = valid_models[best_model_name]
            
            # Sonuçları kaydet
            selection_result = {
                'selected_model': best_model_name,
                'selected_score': best_evaluation['score'],
                'selected_grade': best_evaluation['grade'],
                'selected_metrics': best_evaluation['metrics'],
                'all_evaluations': model_evaluations,
                'selection_time': datetime.now().isoformat(),
                'total_models_evaluated': len(models),
                'valid_models_count': len(valid_models)
            }
            
            self.selection_history.append(selection_result)
            
            logger.info(f"En iyi model seçildi: {best_model_name}")
            logger.info(f"Skor: {best_evaluation['score']:.3f} ({best_evaluation['grade']})")
            
            return selection_result
            
        except Exception as e:
            logger.error(f"Model seçimi hatası: {e}")
            return {
                'selected_model': None,
                'reason': f'Seçim hatası: {str(e)}',
                'error': str(e)
            }
    
    def get_selection_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Seçim geçmişini getir"""
        return self.selection_history[-limit:]


# Global instance
_model_selector = ModelSelectionManager()


def get_model_selector() -> ModelSelectionManager:
    """Global model selector instance'ını getir"""
    return _model_selector


def select_best_available_model(models: Dict[str, Any], X_val, y_val) -> Dict[str, Any]:
    """
    En iyi modeli seçmek için kolay function
    
    Args:
        models: Mevcut modeller
        X_val: Validation verisi
        y_val: Validation hedefleri
        
    Returns:
        Selection results
    """
    return _model_selector.select_best_model(models, X_val, y_val)


def evaluate_single_model(model, X_val, y_val, model_name: str) -> Dict[str, Any]:
    """
    Tek model değerlendirmesi için kolay function
    
    Args:
        model: Değerlendirilecek model
        X_val: Validation verisi
        y_val: Validation hedefleri
        model_name: Model adı
        
    Returns:
        Evaluation results
    """
    evaluator = ComprehensiveModelEvaluator()
    return evaluator.evaluate_model_comprehensive(model, X_val, y_val, model_name)


if __name__ == "__main__":
    # Test için
    print("Model Selection Module Test")
    print("Kullanım: select_best_available_model(models, X_val, y_val)")
