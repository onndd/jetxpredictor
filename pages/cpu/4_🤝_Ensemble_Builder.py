"""
Ensemble Builder Sayfası - Voting ve Stacking

Eğitilmiş modellerden ensemble oluşturma.
Voting strategy, weighted ensemble, stacking ensemble.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# Utils imports
from utils.database import DatabaseManager
from utils.lightweight_model_manager import LightweightModelManager
from utils.cpu_training_engine import CPUTrainingEngine

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="🤝 Ensemble Builder - CPU Lightweight Models",
    page_icon="🤝",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .ensemble-card {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 10px 0;
    }
    .strategy-card {
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
    st.session_state.ensemble_results = None

# Ana başlık
st.markdown("# 🤝 Ensemble Builder")
st.markdown("Eğitilmiş modellerden ensemble oluşturma")

# Model listesi
model_df = st.session_state.model_manager.get_model_list()
trained_models = model_df[model_df['status'] == 'trained']

if trained_models.empty:
    st.warning("⚠️ Henüz eğitilmiş model yok. Önce model eğitin.")
    st.stop()

# Model seçimi
st.markdown("## 🎯 Ensemble Modelleri Seçin")

col1, col2 = st.columns(2)

with col1:
    selected_models = st.multiselect(
        "Modeller",
        trained_models['model_id'].tolist(),
        default=trained_models['model_id'].head(3).tolist(),
        help="Ensemble'e dahil edilecek modelleri seçin (en az 2 model)"
    )

with col2:
    if len(selected_models) >= 2:
        st.success(f"✅ {len(selected_models)} model seçildi")
    else:
        st.warning("⚠️ En az 2 model seçmelisiniz.")

if len(selected_models) < 2:
    st.stop()

# Ensemble stratejisi
st.markdown("## ⚙️ Ensemble Stratejisi")

strategy = st.selectbox(
    "Ensemble Stratejisi",
    ['voting', 'stacking'],
    help="Ensemble stratejisini seçin"
)

if strategy == 'voting':
    st.markdown("### 🗳️ Voting Strategy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        voting_type = st.selectbox(
            "Voting Type",
            ['hard', 'soft'],
            help="Hard voting: majority vote, Soft voting: average probabilities"
        )
    
    with col2:
        use_weights = st.checkbox("Weighted Voting", value=True)
        
        if use_weights:
            st.markdown("**Model Ağırlıkları:**")
            weights = {}
            total_weight = 0
            
            for i, model_id in enumerate(selected_models):
                weight = st.number_input(
                    f"{model_id}",
                    min_value=0.0,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                    key=f"weight_{i}"
                )
                weights[model_id] = weight
                total_weight += weight
            
            # Normalize weights
            if total_weight > 0:
                normalized_weights = {k: v/total_weight for k, v in weights.items()}
                st.markdown("**Normalize Edilmiş Ağırlıklar:**")
                for model_id, weight in normalized_weights.items():
                    st.write(f"  {model_id}: {weight:.3f}")

elif strategy == 'stacking':
    st.markdown("### 📚 Stacking Strategy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        meta_model_type = st.selectbox(
            "Meta Model Tipi",
            ['lightgbm', 'catboost'],
            help="Stacking için meta model tipi"
        )
        
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
    
    with col2:
        use_probabilities = st.checkbox("Use Probabilities", value=True)
        
        if use_probabilities:
            st.info("Meta model probability tahminlerini kullanacak")
        else:
            st.info("Meta model class tahminlerini kullanacak")

# Test verisi
st.markdown("## 📊 Test Verisi")

col1, col2 = st.columns(2)

with col1:
    test_size = st.slider("Test Set Size", 100, 1000, 500)
    random_seed = st.number_input("Random Seed", 0, 1000, 42)

with col2:
    # Test verisi yükleme
    try:
        db_manager = DatabaseManager()
        data = db_manager.get_all_data()
        
        if data.empty:
            st.error("Veri bulunamadı!")
            st.stop()
        
        st.success(f"✅ {len(data):,} veri yüklendi")
        
        # Test verisi hazırlama
        window_size = 1000
        X_test, y_test = [], []
        
        # Son test_size kadar veriyi al
        start_idx = max(window_size, len(data) - test_size)
        
        for i in range(start_idx, len(data)):
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
            
            X_test.append(features)
            
            # Target encoding
            model_info = st.session_state.model_manager.model_registry[selected_models[0]]
            mode = model_info['mode']
            
            if mode == 'classification':
                y_test.append(1 if target >= 1.5 else 0)
            elif mode == 'multiclass':
                if target < 1.5:
                    y_test.append(0)
                elif target < 10:
                    y_test.append(1)
                elif target < 50:
                    y_test.append(2)
                else:
                    y_test.append(3)
            else:  # regression
                y_test.append(target)
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        st.info(f"📈 Test verisi hazırlandı: {len(X_test)} örnek")
        
    except Exception as e:
        st.error(f"❌ Test verisi hazırlama hatası: {str(e)}")
        st.stop()

# Ensemble oluştur
if st.button("🤝 Ensemble Oluştur", type="primary"):
    try:
        # Ensemble konfigürasyonu
        ensemble_config = {
            'strategy': strategy,
            'model_ids': selected_models,
            'created_at': datetime.now().isoformat()
        }
        
        if strategy == 'voting':
            ensemble_config['voting_type'] = voting_type
            if use_weights:
                ensemble_config['weights'] = normalized_weights if 'normalized_weights' in locals() else None
        
        elif strategy == 'stacking':
            ensemble_config['meta_model_type'] = meta_model_type
            ensemble_config['cv_folds'] = cv_folds
            ensemble_config['use_probabilities'] = use_probabilities
        
        # Ensemble oluştur
        ensemble_config = st.session_state.model_manager.create_ensemble(
            model_ids=selected_models,
            strategy=strategy,
            weights=normalized_weights if strategy == 'voting' and use_weights and 'normalized_weights' in locals() else None
        )
        
        st.success(f"✅ Ensemble oluşturuldu: {ensemble_config['ensemble_id']}")
        
        # Ensemble test et
        st.markdown("## 🧪 Ensemble Test")
        
        # Individual model predictions
        individual_predictions = {}
        individual_probabilities = {}
        
        for model_id in selected_models:
            try:
                model = st.session_state.model_manager.load_trained_model(model_id)
                
                # Predictions
                pred = model.predict(X_test)
                individual_predictions[model_id] = pred
                
                # Probabilities (if available)
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_test)
                    individual_probabilities[model_id] = proba
                
            except Exception as e:
                st.warning(f"Model {model_id} yüklenemedi: {str(e)}")
        
        # Ensemble prediction
        if strategy == 'voting':
            if voting_type == 'hard':
                # Hard voting: majority vote
                pred_array = np.array(list(individual_predictions.values()))
                ensemble_pred = np.round(np.mean(pred_array, axis=0)).astype(int)
            else:
                # Soft voting: average probabilities
                if individual_probabilities:
                    prob_array = np.array(list(individual_probabilities.values()))
                    ensemble_pred = np.round(np.mean(prob_array, axis=0)).astype(int)
                else:
                    # Fallback to hard voting
                    pred_array = np.array(list(individual_predictions.values()))
                    ensemble_pred = np.round(np.mean(pred_array, axis=0)).astype(int)
        
        elif strategy == 'stacking':
            # Basit stacking implementation
            # Gerçek uygulamada meta model eğitimi gerekir
            pred_array = np.array(list(individual_predictions.values()))
            ensemble_pred = np.round(np.mean(pred_array, axis=0)).astype(int)
        
        # Metrics hesapla
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        ensemble_precision = precision_score(y_test, ensemble_pred, average='weighted')
        ensemble_recall = recall_score(y_test, ensemble_pred, average='weighted')
        ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted')
        
        # Virtual bankroll simulation
        def simulate_virtual_bankroll(y_true, y_pred):
            initial = 10000
            wallet = initial
            
            for pred, actual in zip(y_pred, y_true):
                if pred == 1:  # Model 1.5 üstü dedi
                    wallet -= 10
                    if actual >= 1.5:
                        wallet += 15
            
            return ((wallet - initial) / initial) * 100
        
        ensemble_roi = simulate_virtual_bankroll(y_test, ensemble_pred)
        
        # Sonuçları kaydet
        ensemble_results = {
            'ensemble_id': ensemble_config['ensemble_id'],
            'strategy': strategy,
            'model_ids': selected_models,
            'metrics': {
                'accuracy': ensemble_accuracy,
                'precision': ensemble_precision,
                'recall': ensemble_recall,
                'f1_score': ensemble_f1,
                'virtual_bankroll_roi': ensemble_roi
            },
            'individual_predictions': individual_predictions,
            'ensemble_prediction': ensemble_pred
        }
        
        st.session_state.ensemble_results = ensemble_results
        
        # Sonuçları göster
        st.markdown("## 📊 Ensemble Sonuçları")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{ensemble_accuracy:.4f}")
        with col2:
            st.metric("Precision", f"{ensemble_precision:.4f}")
        with col3:
            st.metric("Recall", f"{ensemble_recall:.4f}")
        with col4:
            st.metric("F1 Score", f"{ensemble_f1:.4f}")
        
        st.metric("Virtual Bankroll ROI", f"{ensemble_roi:.2f}%")
        
    except Exception as e:
        st.error(f"❌ Ensemble oluşturma hatası: {str(e)}")

# Ensemble sonuçları göster
if st.session_state.ensemble_results is not None:
    results = st.session_state.ensemble_results
    
    st.markdown("## 📈 Ensemble vs Individual Models")
    
    # Individual model metrics
    individual_metrics = []
    
    for model_id in selected_models:
        if model_id in results['individual_predictions']:
            pred = results['individual_predictions'][model_id]
            
            accuracy = accuracy_score(y_test, pred)
            precision = precision_score(y_test, pred, average='weighted')
            recall = recall_score(y_test, pred, average='weighted')
            f1 = f1_score(y_test, pred, average='weighted')
            roi = simulate_virtual_bankroll(y_test, pred)
            
            individual_metrics.append({
                'model_id': model_id,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'virtual_bankroll_roi': roi
            })
    
    # Ensemble metrics ekle
    individual_metrics.append({
        'model_id': f"Ensemble ({results['strategy']})",
        'accuracy': results['metrics']['accuracy'],
        'precision': results['metrics']['precision'],
        'recall': results['metrics']['recall'],
        'f1_score': results['metrics']['f1_score'],
        'virtual_bankroll_roi': results['metrics']['virtual_bankroll_roi']
    })
    
    comparison_df = pd.DataFrame(individual_metrics)
    
    # Metrics tablosu
    st.markdown("### 📊 Karşılaştırma Tablosu")
    st.dataframe(comparison_df, use_container_width=True)
    
    # Görselleştirmeler
    st.markdown("### 📈 Accuracy Karşılaştırması")
    
    fig_acc = px.bar(
        comparison_df,
        x='model_id',
        y='accuracy',
        title='Ensemble vs Individual Models Accuracy',
        color='accuracy',
        color_continuous_scale='viridis'
    )
    fig_acc.update_layout(height=400)
    st.plotly_chart(fig_acc, use_container_width=True)
    
    # ROI karşılaştırması
    st.markdown("### 💰 Virtual Bankroll ROI Karşılaştırması")
    
    fig_roi = px.bar(
        comparison_df,
        x='model_id',
        y='virtual_bankroll_roi',
        title='Ensemble vs Individual Models ROI',
        color='virtual_bankroll_roi',
        color_continuous_scale='RdYlGn'
    )
    fig_roi.update_layout(height=400)
    st.plotly_chart(fig_roi, use_container_width=True)
    
    # Ensemble kaydetme
    st.markdown("## 💾 Ensemble Kaydet")
    
    if st.button("💾 Ensemble'i Kaydet", type="primary"):
        try:
            # Ensemble config'i kaydet
            ensemble_id = results['ensemble_id']
            
            # Registry'ye kaydet
            st.session_state.model_manager.model_registry[ensemble_id].update({
                'status': 'trained',
                'trained_at': datetime.now().isoformat(),
                'metrics': results['metrics'],
                'strategy': strategy,
                'model_ids': selected_models
            })
            
            # Registry'yi kaydet
            st.session_state.model_manager.save_model_registry()
            
            st.success(f"✅ Ensemble kaydedildi: {ensemble_id}")
            
        except Exception as e:
            st.error(f"❌ Kaydetme hatası: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🤝 Ensemble Builder - CPU Lightweight Models</p>
</div>
""", unsafe_allow_html=True)









