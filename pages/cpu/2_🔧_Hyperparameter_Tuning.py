"""
Hyperparameter Tuning Sayfası - Optuna Entegrasyonu

CPU modelleri için hyperparameter optimization.
Optuna ile automated hyperparameter search ve visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import json

# Utils imports
from utils.database import DatabaseManager
from utils.lightweight_model_manager import LightweightModelManager
from utils.cpu_training_engine import CPUTrainingEngine

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="🔧 Hyperparameter Tuning - CPU Lightweight Models",
    page_icon="🔧",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .tuning-card {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 10px 0;
    }
    .search-card {
        padding: 15px;
        border-radius: 8px;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        margin: 10px 0;
    }
    .result-card {
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
    st.session_state.tuning_in_progress = False
    st.session_state.tuning_progress = 0
    st.session_state.tuning_status = ""
    st.session_state.current_study = None

# Ana başlık
st.markdown("# 🔧 Hyperparameter Tuning")
st.markdown("Optuna ile automated hyperparameter optimization")

# Sidebar - Tuning konfigürasyonu
with st.sidebar:
    st.markdown("## ⚙️ Tuning Konfigürasyonu")
    
    # Model tipi seçimi
    available_models = st.session_state.model_manager.get_available_models()
    model_type = st.selectbox(
        "Model Tipi",
        list(available_models.keys()),
        help="Optimize edilecek model tipini seçin"
    )
    
    # Model modu seçimi
    model_modes = available_models[model_type]['modes']
    mode = st.selectbox(
        "Model Modu",
        model_modes,
        help="Model modunu seçin"
    )
    
    # Optimization parametreleri
    st.markdown("### 🎯 Optimization Parametreleri")
    
    n_trials = st.slider("Number of Trials", 10, 100, 50)
    timeout = st.slider("Timeout (minutes)", 10, 120, 60)
    cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
    
    # Optimization metric
    optimization_metrics = {
        'classification': ['accuracy', 'precision', 'recall', 'f1_score'],
        'multiclass': ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
        'regression': ['mae', 'mse', 'rmse', 'r2']
    }
    
    optimization_metric = st.selectbox(
        "Optimization Metric",
        optimization_metrics.get(mode, ['accuracy']),
        help="Optimize edilecek metrik"
    )

# Ana içerik
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 📊 Search Space Konfigürasyonu")
    
    # Model tipine göre search space
    if model_type == 'lightgbm':
        st.markdown("### LightGBM Search Space")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Num Leaves**")
            num_leaves_min = st.number_input("Min", 10, 50, 10, key="num_leaves_min")
            num_leaves_max = st.number_input("Max", 50, 200, 100, key="num_leaves_max")
            
            st.markdown("**Max Depth**")
            max_depth_min = st.number_input("Min", 3, 10, 3, key="max_depth_min")
            max_depth_max = st.number_input("Max", 10, 20, 15, key="max_depth_max")
        
        with col2:
            st.markdown("**Learning Rate**")
            lr_min = st.number_input("Min", 0.01, 0.1, 0.01, key="lr_min")
            lr_max = st.number_input("Max", 0.1, 0.5, 0.3, key="lr_max")
            
            st.markdown("**Feature Fraction**")
            ff_min = st.number_input("Min", 0.5, 0.8, 0.5, key="ff_min")
            ff_max = st.number_input("Max", 0.8, 1.0, 1.0, key="ff_max")
        
        with col3:
            st.markdown("**Bagging Fraction**")
            bf_min = st.number_input("Min", 0.5, 0.8, 0.5, key="bf_min")
            bf_max = st.number_input("Max", 0.8, 1.0, 1.0, key="bf_max")
            
            st.markdown("**Min Data in Leaf**")
            mdil_min = st.number_input("Min", 10, 30, 10, key="mdil_min")
            mdil_max = st.number_input("Max", 30, 100, 50, key="mdil_max")

    elif model_type == 'catboost':
        st.markdown("### CatBoost Search Space")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Iterations**")
            iter_min = st.number_input("Min", 500, 1000, 500, key="iter_min")
            iter_max = st.number_input("Max", 1000, 3000, 2000, key="iter_max")
            
            st.markdown("**Depth**")
            depth_min = st.number_input("Min", 4, 8, 4, key="depth_min")
            depth_max = st.number_input("Max", 8, 15, 12, key="depth_max")
        
        with col2:
            st.markdown("**Learning Rate**")
            lr_min = st.number_input("Min", 0.01, 0.1, 0.01, key="lr_min")
            lr_max = st.number_input("Max", 0.1, 0.5, 0.3, key="lr_max")
            
            st.markdown("**L2 Leaf Reg**")
            l2_min = st.number_input("Min", 1, 5, 1, key="l2_min")
            l2_max = st.number_input("Max", 5, 20, 10, key="l2_max")
        
        with col3:
            st.markdown("**Random Strength**")
            rs_min = st.number_input("Min", 0.5, 1.0, 0.5, key="rs_min")
            rs_max = st.number_input("Max", 1.0, 3.0, 2.0, key="rs_max")

    elif model_type == 'tabnet':
        st.markdown("### TabNet Search Space")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**N_d**")
            nd_min = st.number_input("Min", 4, 8, 4, key="nd_min")
            nd_max = st.number_input("Max", 8, 20, 16, key="nd_max")
            
            st.markdown("**N_a**")
            na_min = st.number_input("Min", 4, 8, 4, key="na_min")
            na_max = st.number_input("Max", 8, 20, 16, key="na_max")
        
        with col2:
            st.markdown("**N_steps**")
            ns_min = st.number_input("Min", 2, 4, 2, key="ns_min")
            ns_max = st.number_input("Max", 4, 8, 6, key="ns_max")
            
            st.markdown("**Gamma**")
            gamma_min = st.number_input("Min", 1.0, 1.5, 1.0, key="gamma_min")
            gamma_max = st.number_input("Max", 1.5, 2.5, 2.0, key="gamma_max")
        
        with col3:
            st.markdown("**N_independent**")
            ni_min = st.number_input("Min", 1, 2, 1, key="ni_min")
            ni_max = st.number_input("Max", 2, 6, 4, key="ni_max")
            
            st.markdown("**N_shared**")
            ns_min = st.number_input("Min", 1, 2, 1, key="ns_min")
            ns_max = st.number_input("Max", 2, 6, 4, key="ns_max")

with col2:
    st.markdown("## 🎯 Optimization Ayarları")
    
    # Study name
    study_name = st.text_input(
        "Study Name",
        value=f"{model_type}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Benzersiz study adı"
    )
    
    # Direction
    direction = st.selectbox(
        "Optimization Direction",
        ['maximize', 'minimize'],
        help="Metrik maksimize mi minimize mi edilecek"
    )
    
    # Pruning
    enable_pruning = st.checkbox("Enable Pruning", value=True)
    
    # Storage
    storage_type = st.selectbox(
        "Storage Type",
        ['memory', 'sqlite'],
        help="Study storage tipi"
    )
    
    if storage_type == 'sqlite':
        storage_path = st.text_input("Storage Path", "studies.db")

# Tuning başlatma
st.markdown("## 🚀 Hyperparameter Tuning Başlat")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🔧 Tuning Başlat", use_container_width=True, type="primary"):
        if not st.session_state.tuning_in_progress:
            try:
                st.session_state.tuning_in_progress = True
                st.session_state.tuning_progress = 0
                st.session_state.tuning_status = "Tuning başlatılıyor..."
                
                # Progress callback
                def progress_callback(progress, status):
                    st.session_state.tuning_progress = progress
                    st.session_state.tuning_status = status
                    if progress >= 0:
                        st.progress(progress / 100)
                    st.write(status)
                
                # Veri yükleme
                progress_callback(10, "Veri yükleniyor...")
                
                db_manager = DatabaseManager()
                data = db_manager.get_all_data()
                
                if data.empty:
                    st.error("Veri bulunamadı!")
                    st.session_state.tuning_in_progress = False
                    st.stop()
                
                # Feature extraction
                progress_callback(20, "Feature extraction yapılıyor...")
                
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
                
                # Hyperparameter search
                progress_callback(40, "Hyperparameter search başlıyor...")
                
                results = st.session_state.training_engine.hyperparameter_search(
                    model_type=model_type,
                    X=X,
                    y=y,
                    mode=mode,
                    n_trials=n_trials,
                    timeout=timeout * 60,  # Convert to seconds
                    cv_folds=cv_folds,
                    optimization_metric=optimization_metric,
                    progress_callback=progress_callback
                )
                
                progress_callback(100, f"Tuning tamamlandı! Study: {results['study_name']}")
                
                # Sonuçları kaydet
                st.session_state.current_study = results
                
                st.success(f"✅ Hyperparameter tuning tamamlandı!")
                
                # Best parameters göster
                st.markdown("### 🏆 En İyi Parametreler")
                
                best_params = results['best_params']
                best_value = results['best_value']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**En İyi Parametreler:**")
                    for param, value in best_params.items():
                        st.write(f"  {param}: {value}")
                
                with col2:
                    st.metric("En İyi Değer", f"{best_value:.4f}")
                    st.metric("Trial Sayısı", results['n_trials'])
                
                st.session_state.tuning_in_progress = False
                
                # Sayfayı yenile
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Tuning hatası: {str(e)}")
                st.session_state.tuning_in_progress = False
        else:
            st.warning("⚠️ Tuning zaten devam ediyor...")

# Tuning durumu
if st.session_state.tuning_in_progress:
    st.markdown("## 🔄 Tuning Durumu")
    
    progress_bar = st.progress(st.session_state.tuning_progress / 100)
    st.write(f"**Durum:** {st.session_state.tuning_status}")
    
    # Cancel butonu
    if st.button("❌ Tuning'i İptal Et"):
        st.session_state.tuning_in_progress = False
        st.session_state.tuning_progress = 0
        st.session_state.tuning_status = ""
        st.rerun()

# Sonuçlar ve görselleştirme
if st.session_state.current_study:
    st.markdown("## 📊 Tuning Sonuçları")
    
    study = st.session_state.current_study['study']
    
    # Optimization history
    st.markdown("### 📈 Optimization History")
    
    trials = study.trials
    trial_numbers = [trial.number for trial in trials]
    trial_values = [trial.value for trial in trials if trial.value is not None]
    
    if trial_values:
        fig_history = go.Figure()
        fig_history.add_trace(go.Scatter(
            x=trial_numbers[:len(trial_values)],
            y=trial_values,
            mode='lines+markers',
            name='Trial Values',
            line=dict(color='blue')
        ))
        
        fig_history.update_layout(
            title="Optimization History",
            xaxis_title="Trial Number",
            yaxis_title=f"{optimization_metric.title()}",
            height=400
        )
        
        st.plotly_chart(fig_history, use_container_width=True)
    
    # Parameter importances
    st.markdown("### 🎯 Parameter Importances")
    
    try:
        import optuna.visualization as vis
        
        # Parameter importance plot
        importance_fig = vis.plot_param_importances(study)
        st.plotly_chart(importance_fig, use_container_width=True)
        
        # Optimization history plot
        history_fig = vis.plot_optimization_history(study)
        st.plotly_chart(history_fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Visualization hatası: {str(e)}")
    
    # Best trial detayları
    st.markdown("### 🏆 En İyi Trial Detayları")
    
    best_trial = study.best_trial
    st.write(f"**Trial Number:** {best_trial.number}")
    st.write(f"**Value:** {best_trial.value:.4f}")
    st.write(f"**Duration:** {best_trial.duration.total_seconds():.2f} seconds")
    
    # Parametreler tablosu
    params_df = pd.DataFrame([best_trial.params])
    st.dataframe(params_df, use_container_width=True)
    
    # Bu parametrelerle eğitim butonu
    st.markdown("### 🎯 Bu Parametrelerle Eğitim")
    
    if st.button("🚀 En İyi Parametrelerle Model Eğit", type="primary"):
        # En iyi parametrelerle model eğitimi
        best_params = best_trial.params
        
        # Model oluştur ve eğit
        model, model_id = st.session_state.model_manager.create_model(
            model_type=model_type,
            mode=mode,
            config=best_params
        )
        
        # Eğitim konfigürasyonu
        training_config = {
            'cv_folds': cv_folds,
            'validation_split': 0.15
        }
        
        # Veri hazırla (yukarıdaki ile aynı)
        db_manager = DatabaseManager()
        data = db_manager.get_all_data()
        
        window_size = 1000
        X, y = [], []
        
        for i in range(window_size, len(data)):
            hist = data['value'].iloc[:i].values
            target = data['value'].iloc[i]
            
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
        
        # Model eğitimi
        try:
            metrics = st.session_state.training_engine.train_single_model(
                model_type=model_type,
                X=X,
                y=y,
                mode=mode,
                config=best_params,
                training_config=training_config
            )
            
            st.success(f"✅ Model başarıyla eğitildi: {model_id}")
            
            # Metrics göster
            st.markdown("#### 📊 Eğitim Metrikleri")
            
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
            
        except Exception as e:
            st.error(f"❌ Model eğitimi hatası: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔧 Hyperparameter Tuning - CPU Lightweight Models</p>
</div>
""", unsafe_allow_html=True)


