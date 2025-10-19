"""
CPU Training Engine - Advanced Training Features

Gelişmiş eğitim özellikleri için CPU optimized training engine.
Hyperparameter search, cross-validation, progress tracking sağlar.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime
import warnings
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
        """
        Tek model eğitimi
        
        Args:
            model_type: Model tipi
            X: Feature matrix
            y: Target values
            mode: Model modu
            config: Model konfigürasyonu
            training_config: Eğitim konfigürasyonu
            progress_callback: Progress callback function
            
        Returns:
            (model_id, metrics)
        """
        logger.info(f"Tek model eğitimi başlıyor: {model_type} ({mode})")
        
        # Model oluştur
        model, model_id = self.model_manager.create_model(
            model_type=model_type,
            mode=mode,
            config=config
        )
        
        # Progress callback
        if progress_callback:
            progress_callback(0, f"Model oluşturuldu: {model_id}")
        
        # Eğitim konfigürasyonu
        if training_config is None:
            training_config = {}
        
        # Model eğitimi
        try:
            metrics = self.model_manager.train_model(
                model_id=model_id,
                X=X,
                y=y,
                training_config=training_config
            )
            
            # Training history'ye kaydet
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
        """
        Hyperparameter search (Optuna)
        
        Args:
            model_type: Model tipi
            X: Feature matrix
            y: Target values
            mode: Model modu
            n_trials: Trial sayısı
            timeout: Timeout (saniye)
            cv_folds: Cross-validation fold sayısı
            optimization_metric: Optimize edilecek metrik
            progress_callback: Progress callback function
            
        Returns:
            Optimization results
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna yüklü değil. 'pip install optuna' ile yükleyin.")
        
        logger.info(f"Hyperparameter search başlıyor: {model_type} ({mode})")
        
        # Study oluştur
        study_name = f"{model_type}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            direction='maximize' if optimization_metric in ['accuracy', 'f1_score', 'r2'] else 'minimize',
            study_name=study_name
        )
        
        def objective(trial):
            # Hyperparameter space tanımla
            config = self._get_hyperparameter_space(model_type, trial)
            
            # Cross-validation
            cv_scores = []
            
            from sklearn.model_selection import StratifiedKFold, KFold
            
            if mode in ['classification', 'multiclass']:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            else:
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Model oluştur ve eğit
                model, model_id = self.model_manager.create_model(
                    model_type=model_type,
                    mode=mode,
                    config=config
                )
                
                try:
                    # Eğitim
                    metrics = model.train(X_train, y_train, validation_split=0.0)
                    
                    # Validation metrics
                    val_pred = model.predict(X_val)
                    val_metrics = self._calculate_metric(y_val, val_pred, optimization_metric, mode)
                    
                    cv_scores.append(val_metrics)
                    
                except Exception as e:
                    logger.warning(f"Trial hatası: {str(e)}")
                    cv_scores.append(0.0 if optimization_metric in ['accuracy', 'f1_score', 'r2'] else 1000.0)
            
            return np.mean(cv_scores)
        
        # Optimization çalıştır
        try:
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=False
            )
            
            # Results
            best_params = study.best_params
            best_value = study.best_value
            
            # Study kaydet
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
    
    def train_multiple_models(
        self,
        model_types: List[str],
        X: np.ndarray,
        y: np.ndarray,
        mode: str = 'classification',
        configs: Optional[Dict[str, Dict]] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Tuple[str, Dict]]:
        """
        Çoklu model eğitimi
        
        Args:
            model_types: Model tipleri listesi
            X: Feature matrix
            y: Target values
            mode: Model modu
            configs: Her model için konfigürasyon
            progress_callback: Progress callback function
            
        Returns:
            {model_type: (model_id, metrics)}
        """
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
        """
        Model değerlendirme
        
        Args:
            model_id: Model ID
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Evaluation metrics
        """
        logger.info(f"Model değerlendirme: {model_id}")
        
        try:
            model = self.model_manager.load_trained_model(model_id)
            model_info = self.model_manager.model_registry[model_id]
            
            # Tahminler
            y_pred = model.predict(X_test)
            
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
            else:
                y_pred_proba = None
            
            # Metrics hesapla
            metrics = self._calculate_comprehensive_metrics(
                y_test, y_pred, y_pred_proba, model_info['mode']
            )
            
            # Feature importance (varsa)
            if hasattr(model, 'get_feature_importance'):
                try:
                    feature_importance = model.get_feature_importance()
                    metrics['feature_importance'] = feature_importance.to_dict()
                except Exception as e:
                    logger.warning(f"Feature importance alınamadı: {str(e)}")
            
            logger.info(f"Model değerlendirme tamamlandı: {model_id}")
            return metrics
            
        except Exception as e:
            logger.error(f"Model değerlendirme başarısız: {model_id} - {str(e)}")
            raise
    
    def plot_learning_curves(self, model_id: str) -> Dict:
        """
        Learning curves çizimi
        
        Args:
            model_id: Model ID
            
        Returns:
            Learning curves data
        """
        if model_id not in self.training_history:
            raise ValueError(f"Training history bulunamadı: {model_id}")
        
        training_info = self.training_history[model_id]
        
        # Model'e göre learning curves çıkar
        # Bu basit bir implementasyon, gerçek uygulamada model'e özel olmalı
        curves = {
            'model_id': model_id,
            'model_type': training_info['model_type'],
            'mode': training_info['mode'],
            'metrics': training_info['metrics'],
            'trained_at': training_info['trained_at']
        }
        
        return curves
    
    def get_training_summary(self) -> pd.DataFrame:
        """Training summary döndür"""
        summary = []
        
        for model_id, info in self.training_history.items():
            summary.append({
                'model_id': model_id,
                'model_type': info['model_type'],
                'mode': info['mode'],
                'trained_at': info['trained_at'],
                'metrics': info['metrics']
            })
        
        return pd.DataFrame(summary)
    
    def get_hyperparameter_summary(self) -> pd.DataFrame:
        """Hyperparameter search summary döndür"""
        summary = []
        
        for study_name, info in self.hyperparameter_studies.items():
            summary.append({
                'study_name': study_name,
                'model_type': info['model_type'],
                'mode': info['mode'],
                'best_value': info['best_value'],
                'n_trials': info['n_trials'],
                'completed_at': info['completed_at']
            })
        
        return pd.DataFrame(summary)
    
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
    
    def _calculate_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric: str,
        mode: str
    ) -> float:
        """Tek metrik hesaplama"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            mean_absolute_error, mean_squared_error, r2_score
        )
        
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
    
    def _calculate_comprehensive_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray],
        mode: str
    ) -> Dict:
        """Kapsamlı metrikler hesaplama"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            mean_absolute_error, mean_squared_error, r2_score,
            classification_report, confusion_matrix
        )
        
        metrics = {}
        
        if mode in ['classification', 'multiclass']:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
            
            # Classification report
            metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
            
            # Confusion matrix
            metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
            
            # Virtual bankroll simulation
            if mode == 'classification':
                metrics['virtual_bankroll_roi'] = self._simulate_virtual_bankroll(y_true, y_pred)
        
        elif mode == 'regression':
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_true, y_pred)
        
        return metrics
    
    def _simulate_virtual_bankroll(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Virtual bankroll simulation"""
        initial = 10000
        wallet = initial
        
        for pred, actual in zip(y_pred, y_true):
            if pred == 1:  # Model 1.5 üstü dedi
                wallet -= 10
                if actual >= 1.5:
                    wallet += 15
        
        return ((wallet - initial) / initial) * 100
