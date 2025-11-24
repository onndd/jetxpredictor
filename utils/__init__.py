"""
JetX Predictor - Utils Package (v2.0)

Bu package projenin tüm yardımcı modüllerini, model yöneticilerini,
analiz araçlarını ve özel katmanlarını içerir.

YENİLİKLER:
- Threshold Manager (Merkezi Eşik Yönetimi)
- Model Loader & Versioning
- A/B Testing & Monitoring
- Gelişmiş Model Wrapper'ları (TabNet, AutoGluon, CatBoost Ensemble)
- Psikolojik ve Anomali Analizörleri
"""

# 1. Temel Yapıtaşları (Database, Config, Threshold)
from .database import DatabaseManager
from .config_loader import config, ConfigLoader
from .threshold_manager import (
    ThresholdManager, 
    get_threshold_manager, 
    get_threshold
)

# 2. Tahmin Motorları
from .predictor import JetXPredictor
from .ensemble_predictor import EnsemblePredictor, create_ensemble_predictor
from .consensus_predictor import ConsensusPredictor, simulate_consensus_bankroll
from .all_models_predictor import AllModelsPredictor

# 3. Risk ve Kasa Yönetimi
from .risk_manager import RiskManager
from .advanced_bankroll import AdvancedBankrollManager, BetResult
from .dual_bankroll_system import DualBankrollSystem, simulate_dual_bankroll

# 4. Model Yöneticileri ve Loader'lar
from .model_loader import ModelLoader, get_model_loader
from .model_versioning import ModelVersionManager, get_version_manager
from .lightweight_model_manager import LightweightModelManager
from .model_selection import ModelSelectionManager, get_model_selector

# 5. Analiz ve İzleme Araçları
from .psychological_analyzer import PsychologicalAnalyzer, create_psychological_analyzer
from .anomaly_streak_detector import AnomalyStreakDetector, create_anomaly_streak_detector
from .feature_monitor import FeatureMonitor, get_feature_monitor
from .ensemble_monitor import EnsembleMonitor
from .feature_validator import FeatureValidator, get_feature_validator, validate_model_compatibility
from .ab_testing import ABTestManager, get_ab_test_manager
from .backtesting import BacktestEngine, create_backtest_engine

# 6. Özel Model Wrapper'ları
from .catboost_ensemble import CatBoostEnsemble
from .tabnet_predictor import TabNetHighXPredictor, create_tabnet_high_x_predictor
from .autogluon_predictor import AutoGluonPredictor
from .lightgbm_predictor import LightGBMPredictor

# 7. Deep Learning Bileşenleri (Loss, Layers, Callbacks)
from .custom_losses import (
    percentage_aware_regression_loss,
    balanced_threshold_killer_loss,
    balanced_focal_loss,
    create_weighted_binary_crossentropy,
    CUSTOM_OBJECTS
)
from .ultra_custom_losses import (
    ultra_threshold_killer_loss,
    ultra_focal_loss,
    ultra_weighted_binary_crossentropy,
    ULTRA_CUSTOM_OBJECTS
)
from .attention_layers import (
    PositionalEncoding,
    LightweightTransformerEncoder,
    MultiHeadAttention,
    SelfAttention,
    TemporalAttention
)
from .balanced_batch_generator import BalancedBatchGenerator
from .adaptive_weight_scheduler import AdaptiveWeightScheduler
from .virtual_bankroll_callback import VirtualBankrollCallback, CatBoostBankrollCallback
from .lr_schedulers import (
    CosineAnnealingWarmup,
    OneCyclePolicy,
    ExponentialDecayWarmup
)
from .cpu_training_engine import CPUTrainingEngine

# 8. RL Agent
from .rl_agent import RLAgent, create_rl_agent

__all__ = [
    # Core
    'DatabaseManager',
    'config', 'ConfigLoader',
    'ThresholdManager', 'get_threshold_manager', 'get_threshold',
    
    # Predictors
    'JetXPredictor',
    'EnsemblePredictor', 'create_ensemble_predictor',
    'ConsensusPredictor', 'simulate_consensus_bankroll',
    'AllModelsPredictor',
    
    # Risk & Bankroll
    'RiskManager',
    'AdvancedBankrollManager', 'BetResult',
    'DualBankrollSystem', 'simulate_dual_bankroll',
    
    # Managers & Loaders
    'ModelLoader', 'get_model_loader',
    'ModelVersionManager', 'get_version_manager',
    'LightweightModelManager',
    'ModelSelectionManager', 'get_model_selector',
    
    # Analysis & Monitoring
    'PsychologicalAnalyzer', 'create_psychological_analyzer',
    'AnomalyStreakDetector', 'create_anomaly_streak_detector',
    'FeatureMonitor', 'get_feature_monitor',
    'EnsembleMonitor',
    'FeatureValidator', 'get_feature_validator', 'validate_model_compatibility',
    'ABTestManager', 'get_ab_test_manager',
    'BacktestEngine', 'create_backtest_engine',
    
    # Model Wrappers
    'CatBoostEnsemble',
    'TabNetHighXPredictor', 'create_tabnet_high_x_predictor',
    'AutoGluonPredictor',
    'LightGBMPredictor',
    
    # DL Components
    'percentage_aware_regression_loss',
    'balanced_threshold_killer_loss',
    'balanced_focal_loss',
    'create_weighted_binary_crossentropy',
    'CUSTOM_OBJECTS',
    'ultra_threshold_killer_loss',
    'ultra_focal_loss',
    'ultra_weighted_binary_crossentropy',
    'ULTRA_CUSTOM_OBJECTS',
    
    'PositionalEncoding',
    'LightweightTransformerEncoder',
    'MultiHeadAttention',
    'SelfAttention',
    'TemporalAttention',
    
    'BalancedBatchGenerator',
    'AdaptiveWeightScheduler',
    'VirtualBankrollCallback', 'CatBoostBankrollCallback',
    'CosineAnnealingWarmup', 'OneCyclePolicy', 'ExponentialDecayWarmup',
    'CPUTrainingEngine',
    
    # RL
    'RLAgent', 'create_rl_agent'
]
