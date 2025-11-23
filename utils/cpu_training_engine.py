"""
CPU Training Engine - Advanced Training Features

Gelişmiş eğitim özellikleri için CPU optimized training engine.
Hyperparameter search, cross-validation, progress tracking sağlar.

GÜNCELLEME:
- Threshold Manager entegrasyonu.
- 2 Modlu (Normal/Rolling) performans takibi.
- Sanal kasa simülasyonunda 0.85 eşiği.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime
import warnings
from utils.threshold_manager import get_threshold_manager

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Optuna için lazy import
try:
    import optuna
    from optuna.visualization import plot_optimization_history, plot_param_importances
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna yüklü değil. 'pip install optuna' ile yükleyin.")

# Model imports
from .lightweight_model_manager import LightweightModelManager


class CPUTrainingEngine:
    """
    CPU optimized training engine
    
    Özellikler:
    - Multi-model training pipeline
    - Hyperparameter search (Optuna)
    - Cross-validation
    - Early stopping
    - Learning curves
    - Training progress tracking
    - Virtual bankroll simulation during training
    """
    
    def __init__(self, model_manager: LightweightModelManager):
        """
        Args:
            model_manager: LightweightModelManager instance
        """
        self.model_manager = model_manager
        self.training_history = {}
        self.hyperparameter_studies = {}
        
        # Threshold Manager
        tm = get_threshold_manager()
        self.THRESHOLD_NORMAL = tm.get_normal_threshold()   # 0.85
        self.THRESHOLD_ROLLING = tm.get_rolling_threshold() # 0.95
        
    def train_single_model(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        mode: str = 'classification',
        config: Optional[Dict] = None,
        training_config: Optional[Dict] = None,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[str, Dict]:
        """Tek model eğitimi"""
        logger.info(f"Tek model eğitimi başlıyor: {model_type} ({mode})")
        
        # Model oluştur
        model, model_id = self.model_manager.create_model(
            model_type=model_type,
            mode=mode,
            config=config
        )
        
        if progress_callback:
            progress_callback(0, f"Model oluşturuldu: {model_id}")
        
        if training_config is None:
            training_config = {}
        
        try:
            metrics = self.model_manager.train_model(
                model_id=model_id,
                X=X,
                y=y,
                training_config=training_config
            )
            
            self.training_history[model_id] = {
                'model_type': model_type,
                'mode': mode,
                'config': config,
                'training_config': training_config,
                'metrics': metrics,
                'trained_at': datetime.now().isoformat()
            }
            
            if progress_callback:
                progress_callback(100, f"Model eğitimi tamamlandı: {model_id}")
            
            logger.info(f"Tek model eğitimi tamamlandı: {model_id}")
            return model_id, metrics
            
        except Exception as e:
            logger.error(f"Tek model eğitimi başarısız: {model_id} - {str(e)}")
            if progress_callback:
                progress_callback(-1, f"Eğitim hatası: {str(e)}")
            raise
    
    def hyperparameter_search(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        mode: str = 'classification',
        n_trials: int = 50,
        timeout: int = 3600,
        cv_folds: int = 5,
        optimization_metric: str = 'accuracy',
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """Hyperparameter search (Optuna)"""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna yüklü değil. 'pip install optuna' ile yükleyin.")
        
        logger.info(f"Hyperparameter search başlıyor: {model_type} ({mode})")
        
        study_name = f"{model_type}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            direction='maximize' if optimization_metric in ['accuracy', 'f1_score', 'r2'] else 'minimize',
            study_name=study_name
        )
        
        def objective(trial):
            config = self._get_hyperparameter_space(model_type, trial)
            cv_scores = []
            
            from sklearn.model_selection import StratifiedKFold, KFold
            
            if mode in ['classification', 'multiclass']:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            else:
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model, model_id = self.model_manager.create_model(
                    model_type=model_type,
                    mode=mode,
                    config=config
                )
                
                try:
                    # Eğitim
                    model.train(X_train, y_train, validation_split=0.0)
                    
                    # Validation metrics
                    val_pred = model.predict(X_val)
                    
                    # Olasılık tabanlı metrikler için (Classifier)
                    val_pred_proba = None
                    if hasattr(model, 'predict_proba'):
                        try:
                             val_pred_proba = model.predict_proba(X_val)
                        except: pass

                    val_metrics = self._calculate_metric(
                        y_val, val_pred, optimization_metric, mode, val_pred_proba
                    )
                    
                    cv_scores.append(val_metrics)
                    
                except Exception as e:
                    logger.warning(f"Trial hatası: {str(e)}")
                    cv_scores.append(0.0 if optimization_metric in ['accuracy', 'f1_score', 'r2'] else 1000.0)
            
            return np.mean(cv_scores)
        
        try:
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=False
            )
            
            best_params = study.best_params
            best_value = study.best_value
            
            self.hyperparameter_studies[study_name] = {
                'study_name': study_name,
                'model_type': model_type,
                'mode': mode,
                'best_params': best_params,
                'best_value': best_value,
                'n_trials': len(study.trials),
                'completed_at': datetime.now().isoformat()
            }
            
            if progress_callback:
                progress_callback(100, f"Hyperparameter search tamamlandı: {study_name}")
            
            logger.info(f"Hyperparameter search tamamlandı: {study_name}")
            
            return {
                'study_name': study_name,
                'best_params': best_params,
                'best_value': best_value,
                'n_trials': len(study.trials),
                'study': study
            }
            
        except Exception as e:
            logger.error(f"Hyperparameter search başarısız: {str(e)}")
            if progress_callback:
                progress_callback(-1, f"Search hatası: {str(e)}")
            raise

    def _calculate_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric: str,
        mode: str,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> float:
        """Tek metrik hesaplama (Güncellenmiş)"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            mean_absolute_error, mean_squared_error, r2_score
        )
        
        # Eğer olasılık varsa ve metrik accuracy ise, threshold'a göre karar ver
        if metric == 'accuracy' and y_pred_proba is not None and mode == 'classification':
            # Binary classification (Class 1 olasılığı)
            if len(y_pred_proba.shape) > 1:
                probs = y_pred_proba[:, 1]
            else:
                probs = y_pred_proba
            
            # Normal Mod Eşiği (0.85) ile Accuracy hesapla
            # Bu, modelin "Emin olduğu" durumlardaki başarısını ölçer
            y_pred_thresholded = (probs >= self.THRESHOLD_NORMAL).astype(int)
            return accuracy_score(y_true, y_pred_thresholded)

        if metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif metric == 'precision':
            return precision_score(y_true, y_pred, average='weighted')
        elif metric == 'recall':
            return recall_score(y_true, y_pred, average='weighted')
        elif metric == 'f1_score':
            return f1_score(y_true, y_pred, average='weighted')
        elif metric == 'mae':
            return mean_absolute_error(y_true, y_pred)
        elif metric == 'mse':
            return mean_squared_error(y_true, y_pred)
        elif metric == 'rmse':
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric == 'r2':
            return r2_score(y_true, y_pred)
        else:
            return 0.0
            
    def train_multiple_models(
        self,
        model_types: List[str],
        X: np.ndarray,
        y: np.ndarray,
        mode: str = 'classification',
        configs: Optional[Dict[str, Dict]] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Tuple[str, Dict]]:
        """Çoklu model eğitimi"""
        logger.info(f"Çoklu model eğitimi başlıyor: {model_types}")
        
        results = {}
        total_models = len(model_types)
        
        for i, model_type in enumerate(model_types):
            try:
                if progress_callback:
                    progress = int((i / total_models) * 100)
                    progress_callback(progress, f"Eğitiliyor: {model_type}")
                
                config = configs.get(model_type) if configs else None
                
                model_id, metrics = self.train_single_model(
                    model_type=model_type,
                    X=X,
                    y=y,
                    mode=mode,
                    config=config
                )
                
                results[model_type] = (model_id, metrics)
                
            except Exception as e:
                logger.error(f"Model eğitimi başarısız: {model_type} - {str(e)}")
                results[model_type] = (None, {'error': str(e)})
        
        if progress_callback:
            progress_callback(100, "Çoklu model eğitimi tamamlandı")
        
        logger.info(f"Çoklu model eğitimi tamamlandı: {len(results)} model")
        return results

    def evaluate_model(
        self,
        model_id: str,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """Model değerlendirme"""
        return self.model_manager.evaluate_model_comprehensive(
            model_id, X_test, y_test
        ) # LightweightModelManager içinde güncellenmiş fonksiyonu çağırır

    def _get_hyperparameter_space(self, model_type: str, trial) -> Dict:
        """Model tipine göre hyperparameter space tanımla"""
        if model_type == 'lightgbm':
            return {
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),
                'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 1.0),
                'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 1.0)
            }
        
        elif model_type == 'catboost':
            return {
                'iterations': trial.suggest_int('iterations', 500, 2000),
                'depth': trial.suggest_int('depth', 4, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'random_strength': trial.suggest_float('random_strength', 0.5, 2.0)
            }
        
        elif model_type == 'tabnet':
            return {
                'n_d': trial.suggest_int('n_d', 4, 16),
                'n_a': trial.suggest_int('n_a', 4, 16),
                'n_steps': trial.suggest_int('n_steps', 2, 6),
                'gamma': trial.suggest_float('gamma', 1.0, 2.0),
                'n_independent': trial.suggest_int('n_independent', 1, 4),
                'n_shared': trial.suggest_int('n_shared', 1, 4)
            }
        
        else:
            return {}
