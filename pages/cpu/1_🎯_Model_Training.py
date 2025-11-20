"""
Model Training Sayfası - Gelişmiş Eğitim Arayüzü

CPU modelleri için gelişmiş eğitim arayüzü.
Model seçimi, hyperparameter ayarları, real-time progress tracking.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import yaml
import json

# Utils imports
from utils.database import DatabaseManager
from utils.lightweight_model_manager import LightweightModelManager
from utils.cpu_training_engine import CPUTrainingEngine
from utils.predictor import JetXPredictor
from category_definitions import CategoryDefinitions

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="🎯 Model Training - CPU Lightweight Models",
    page_icon="🎯",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .training-card {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 10px 0;
    }
    .config-card {
        padding: 15px;
        border-radius: 8px;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        margin: 10px 0;
    }
    .progress-card {
        padding: 15px;
        border-radius: 8px;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Session state kontrolü
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = LightweightModelManager()
    st.session_state.training_engine = CPUTrainingEngine(st.session_state.model_manager)
    st.session_state.training_in_progress = False
    st.session_state.training_progress = 0
    st.session_state.training_status = ""

# Ana başlık
st.markdown("# 🎯 Model Training")
st.markdown("CPU modelleri için gelişmiş eğitim arayüzü")

# Sidebar - Model seçimi
with st.sidebar:
    st.markdown("## 🔧 Model Konfigürasyonu")
    
    # Model tipi seçimi
    available_models = st.session_state.model_manager.get_available_models()
    model_type = st.selectbox(
        "Model Tipi",
        list(available_models.keys()),
        help="Eğitilecek model tipini seçin"
    )
    
    # Model modu seçimi
    model_modes = available_models[model_type]['modes']
    mode = st.selectbox(
        "Model Modu",
        model_modes,
        help="Model modunu seçin"
    )
    
    # Model ID
    model_id = st.text_input(
        "Model ID (Opsiyonel)",
        value=f"{model_type}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Benzersiz model ID'si"
    )

# Ana içerik
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 📊 Veri Durumu")
    
    # Database bağlantısı
    try:
        db_manager = DatabaseManager()
        if db_manager.is_connected():
            st.success("✅ Database bağlantısı başarılı")
            
            # Veri sayısı
            data_count = db_manager.get_data_count()
            st.info(f"📈 Toplam veri sayısı: {data_count:,}")
            
            # Son veri
            latest_data = db_manager.get_latest_data(5)
            if not latest_data.empty:
                st.markdown("### 📋 Son Veriler")
                st.dataframe(latest_data, use_container_width=True)
        else:
            st.error("❌ Database bağlantısı başarısız")
    except Exception as e:
        st.error(f"❌ Database hatası: {str(e)}")

with col2:
    st.markdown("## ⚙️ Eğitim Ayarları")
    
    # Veri split ayarları
    st.markdown("### 📊 Veri Split")
    train_split = st.slider("Train Split", 0.5, 0.9, 0.7, 0.05)
    val_split = st.slider("Validation Split", 0.05, 0.3, 0.15, 0.05)
    test_split = 1.0 - train_split - val_split
    
    st.write(f"**Test Split:** {test_split:.2f}")
    
    # Cross-validation
    cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
    
    # Early stopping
    early_stopping = st.checkbox("Early Stopping", value=True)
    if early_stopping:
        patience = st.slider("Patience", 10, 100, 50)

# Model-specific hyperparameters
st.markdown("## 🔧 Model Hyperparameters")

# Model tipine göre hyperparameter formu
if model_type == 'lightgbm':
    st.markdown("### LightGBM Parametreleri")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_leaves = st.number_input("Num Leaves", 10, 100, 31)
        max_depth = st.number_input("Max Depth", 3, 15, 8)
        learning_rate = st.number_input("Learning Rate", 0.01, 0.3, 0.03, 0.01)
    
    with col2:
        feature_fraction = st.number_input("Feature Fraction", 0.5, 1.0, 0.8, 0.1)
        bagging_fraction = st.number_input("Bagging Fraction", 0.5, 1.0, 0.8, 0.1)
        bagging_freq = st.number_input("Bagging Freq", 1, 10, 5)
    
    with col3:
        min_data_in_leaf = st.number_input("Min Data in Leaf", 10, 50, 20)
        lambda_l1 = st.number_input("Lambda L1", 0.0, 1.0, 0.1, 0.1)
        lambda_l2 = st.number_input("Lambda L2", 0.0, 1.0, 0.1, 0.1)
    
    model_config = {
        'num_leaves': num_leaves,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'bagging_freq': bagging_freq,
        'min_data_in_leaf': min_data_in_leaf,
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'device': 'cpu',
        'verbosity': -1,
        'random_state': 42
    }

elif model_type == 'catboost':
    st.markdown("### CatBoost Parametreleri")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        iterations = st.number_input("Iterations", 500, 2000, 1000)
        depth = st.number_input("Depth", 4, 12, 8)
        learning_rate = st.number_input("Learning Rate", 0.01, 0.3, 0.03, 0.01)
    
    with col2:
        l2_leaf_reg = st.number_input("L2 Leaf Reg", 1, 10, 3)
        random_strength = st.number_input("Random Strength", 0.5, 2.0, 1.0, 0.1)
        border_count = st.number_input("Border Count", 32, 255, 128)
    
    with col3:
        leaf_estimation_iterations = st.number_input("Leaf Estimation Iterations", 1, 20, 10)
        auto_class_weights = st.selectbox("Auto Class Weights", ['None', 'Balanced', 'SqrtBalanced'])
        verbose = st.checkbox("Verbose", value=False)
    
    model_config = {
        'task_type': 'CPU',
        'iterations': iterations,
        'depth': depth,
        'learning_rate': learning_rate,
        'l2_leaf_reg': l2_leaf_reg,
        'random_strength': random_strength,
        'border_count': border_count,
        'leaf_estimation_iterations': leaf_estimation_iterations,
        'auto_class_weights': auto_class_weights if auto_class_weights != 'None' else None,
        'verbose': verbose
    }

elif model_type == 'tabnet':
    st.markdown("### TabNet Parametreleri")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_d = st.number_input("N_d", 4, 16, 8)
        n_a = st.number_input("N_a", 4, 16, 8)
        n_steps = st.number_input("N_steps", 2, 6, 3)
    
    with col2:
        gamma = st.number_input("Gamma", 1.0, 2.0, 1.3, 0.1)
        n_independent = st.number_input("N_independent", 1, 4, 2)
        n_shared = st.number_input("N_shared", 1, 4, 2)
    
    with col3:
        lambda_sparse = st.number_input("Lambda Sparse", 0.0, 0.01, 0.001, 0.001)
        optimizer_lr = st.number_input("Optimizer LR", 0.001, 0.1, 0.02, 0.001)
        scheduler_step_size = st.number_input("Scheduler Step Size", 5, 20, 10)
    
    model_config = {
        'n_d': n_d,
        'n_a': n_a,
        'n_steps': n_steps,
        'gamma': gamma,
        'n_independent': n_independent,
        'n_shared': n_shared,
        'lambda_sparse': lambda_sparse,
        'optimizer_fn': 'Adam',
        'optimizer_params': {'lr': optimizer_lr},
        'scheduler_params': {'step_size': scheduler_step_size, 'gamma': 0.9},
        'scheduler_fn': 'StepLR',
        'mask_type': 'entmax'
    }

elif model_type == 'autogluon':
    st.markdown("### AutoGluon Parametreleri")
    
    col1, col2 = st.columns(2)
    
    with col1:
        time_limit = st.number_input("Time Limit (seconds)", 300, 3600, 600)
        presets = st.selectbox("Presets", ['best_quality', 'high_quality', 'good_quality', 'medium_quality', 'optimize_for_deployment'])
        eval_metric = st.selectbox("Eval Metric", ['accuracy', 'f1', 'roc_auc', 'log_loss'])
    
    with col2:
        num_trials = st.number_input("Num Trials", 5, 50, 10)
        hyperparameter_tune = st.checkbox("Hyperparameter Tune", value=True)
        if hyperparameter_tune:
            scheduler = st.selectbox("Scheduler", ['local', 'remote'])
            searcher = st.selectbox("Searcher", ['auto', 'random', 'bayes'])
    
    model_config = {
        'time_limit': time_limit,
        'presets': presets,
        'eval_metric': eval_metric,
        'num_trials': num_trials,
        'hyperparameter_tune_kwargs': {
            'num_trials': num_trials,
            'scheduler': scheduler if hyperparameter_tune else 'local',
            'searcher': searcher if hyperparameter_tune else 'auto'
        } if hyperparameter_tune else None
    }

# Class weights ayarları
st.markdown("## ⚖️ Class Weights")

col1, col2 = st.columns(2)

with col1:
    use_class_weights = st.checkbox("Class Weights Kullan", value=True)
    
    if use_class_weights:
        weight_method = st.selectbox(
            "Weight Method",
            ['balanced', 'custom', 'auto']
        )
        
        if weight_method == 'custom':
            below_weight = st.number_input("Below 1.5x Weight", 1.0, 50.0, 20.0, 1.0)
            above_weight = st.number_input("Above 1.5x Weight", 1.0, 10.0, 1.0, 1.0)
            class_weights = [below_weight, above_weight]
        else:
            class_weights = weight_method

with col2:
    if use_class_weights:
        st.info(f"**Class Weights:** {class_weights if isinstance(class_weights, list) else weight_method}")

# Eğitim butonu
st.markdown("## 🚀 Eğitimi Başlat")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🎯 Model Eğitimini Başlat", use_container_width=True, type="primary"):
        if not st.session_state.training_in_progress:
            # Eğitim konfigürasyonu
            training_config = {
                'cv_folds': cv_folds,
                'early_stopping_rounds': patience if early_stopping else None,
                'validation_split': val_split
            }
            
            if use_class_weights:
                training_config['class_weights'] = class_weights
            
            # Progress callback
            def progress_callback(progress, status):
                st.session_state.training_progress = progress
                st.session_state.training_status = status
                if progress >= 0:
                    st.progress(progress / 100)
                st.write(status)
            
            # Eğitimi başlat
            try:
                st.session_state.training_in_progress = True
                st.session_state.training_progress = 0
                st.session_state.training_status = "Eğitim başlatılıyor..."
                
                # Veri yükleme
                progress_callback(10, "Veri yükleniyor...")
                
                # Database'den veri al
                db_manager = DatabaseManager()
                data = db_manager.get_all_data()
                
                if data.empty:
                    st.error("Veri bulunamadı!")
                    st.session_state.training_in_progress = False
                    st.stop()
                
                # Feature extraction
                progress_callback(20, "Feature extraction yapılıyor...")
                
                # Basit feature extraction (gerçek uygulamada utils'den import edilmeli)
                window_size = 1000
                X, y = [], []
                
                for i in range(window_size, len(data)):
                    hist = data['value'].iloc[:i].values
                    target = data['value'].iloc[i]
                    
                    # Basit features
                    features = [
                        np.mean(hist[-50:]),
                        np.std(hist[-50:]),
                        np.mean(hist[-200:]),
                        np.std(hist[-200:]),
                        np.mean(hist[-500:]),
                        np.std(hist[-500:]),
                        len(hist[hist >= 1.5]) / len(hist),
                        len(hist[hist >= 2.0]) / len(hist),
                        len(hist[hist >= 5.0]) / len(hist),
                        len(hist[hist >= 10.0]) / len(hist)
                    ]
                    
                    X.append(features)
                    
                    if mode == 'classification':
                        y.append(1 if target >= 1.5 else 0)
                    elif mode == 'multiclass':
                        if target < 1.5:
                            y.append(0)
                        elif target < 10:
                            y.append(1)
                        elif target < 50:
                            y.append(2)
                        else:
                            y.append(3)
                    else:  # regression
                        y.append(target)
                
                X = np.array(X)
                y = np.array(y)
                
                progress_callback(30, f"Veri hazırlandı: {len(X)} örnek")
                
                # Model eğitimi
                progress_callback(40, "Model eğitimi başlıyor...")
                
                model_id, metrics = st.session_state.training_engine.train_single_model(
                    model_type=model_type,
                    X=X,
                    y=y,
                    mode=mode,
                    config=model_config,
                    training_config=training_config,
                    progress_callback=progress_callback
                )
                
                progress_callback(100, f"Eğitim tamamlandı! Model ID: {model_id}")
                
                # Sonuçları göster
                st.success(f"✅ Model başarıyla eğitildi: {model_id}")
                
                # Metrics göster
                st.markdown("### 📊 Eğitim Metrikleri")
                
                col1, col2, col3, col4 = st.columns(4)
                
                if 'accuracy' in metrics:
                    col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                if 'mae' in metrics:
                    col1.metric("MAE", f"{metrics['mae']:.4f}")
                if 'precision' in metrics:
                    col2.metric("Precision", f"{metrics['precision']:.4f}")
                if 'recall' in metrics:
                    col3.metric("Recall", f"{metrics['recall']:.4f}")
                if 'f1_score' in metrics:
                    col4.metric("F1 Score", f"{metrics['f1_score']:.4f}")
                
                # Virtual bankroll simulation
                if 'virtual_bankroll_roi' in metrics:
                    st.metric("Virtual Bankroll ROI", f"{metrics['virtual_bankroll_roi']:.2f}%")
                
                # Model kaydetme
                st.session_state.current_model_id = model_id
                
                st.session_state.training_in_progress = False
                
                # Sayfayı yenile
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Eğitim hatası: {str(e)}")
                st.session_state.training_in_progress = False
        else:
            st.warning("⚠️ Eğitim zaten devam ediyor...")

# Eğitim durumu
if st.session_state.training_in_progress:
    st.markdown("## 🔄 Eğitim Durumu")
    
    progress_bar = st.progress(st.session_state.training_progress / 100)
    st.write(f"**Durum:** {st.session_state.training_status}")
    
    # Cancel butonu
    if st.button("❌ Eğitimi İptal Et"):
        st.session_state.training_in_progress = False
        st.session_state.training_progress = 0
        st.session_state.training_status = ""
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎯 Model Training - CPU Lightweight Models</p>
</div>
""", unsafe_allow_html=True)











