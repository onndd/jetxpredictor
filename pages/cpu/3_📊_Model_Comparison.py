"""
Model Comparison Sayfası - Metrikler ve Görselleştirme

Eğitilmiş modelleri karşılaştırma ve görselleştirme.
Side-by-side metric comparison, ROC curves, confusion matrices.
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
    page_title="📊 Model Comparison - CPU Lightweight Models",
    page_icon="📊",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .comparison-card {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 10px 0;
    }
    .metric-card {
        padding: 15px;
        border-radius: 8px;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        margin: 10px 0;
    }
    .best-model-card {
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
    st.session_state.comparison_results = None

# Ana başlık
st.markdown("# 📊 Model Comparison")
st.markdown("Eğitilmiş modelleri karşılaştırma ve görselleştirme")

# Model listesi
model_df = st.session_state.model_manager.get_model_list()
trained_models = model_df[model_df['status'] == 'trained']

if trained_models.empty:
    st.warning("⚠️ Henüz eğitilmiş model yok. Önce model eğitin.")
    st.stop()

# Model seçimi
st.markdown("## 🎯 Karşılaştırılacak Modelleri Seçin")

selected_models = st.multiselect(
    "Modeller",
    trained_models['model_id'].tolist(),
    default=trained_models['model_id'].head(3).tolist(),
    help="Karşılaştırılacak modelleri seçin (en az 2 model)"
)

if len(selected_models) < 2:
    st.warning("⚠️ En az 2 model seçmelisiniz.")
    st.stop()

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

# Karşılaştırma başlat
if st.button("📊 Modelleri Karşılaştır", type="primary"):
    try:
        # Model karşılaştırması
        comparison_results = st.session_state.model_manager.compare_models(
            model_ids=selected_models,
            X_test=X_test,
            y_test=y_test
        )
        
        st.session_state.comparison_results = comparison_results
        st.success("✅ Model karşılaştırması tamamlandı!")
        
    except Exception as e:
        st.error(f"❌ Karşılaştırma hatası: {str(e)}")

# Sonuçları göster
if st.session_state.comparison_results is not None:
    results_df = st.session_state.comparison_results
    
    st.markdown("## 📈 Karşılaştırma Sonuçları")
    
    # Metrics tablosu
    st.markdown("### 📊 Metrics Tablosu")
    
    # Numeric columns only
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    display_df = results_df[['model_id', 'model_type', 'mode'] + list(numeric_cols)]
    
    st.dataframe(display_df, use_container_width=True)
    
    # En iyi model
    st.markdown("### 🏆 En İyi Model")
    
    if 'accuracy' in results_df.columns:
        best_model_idx = results_df['accuracy'].idxmax()
        best_model = results_df.iloc[best_model_idx]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**Model:** {best_model['model_id']}")
            st.markdown(f"**Tip:** {best_model['model_type']}")
        
        with col2:
            st.markdown(f"**Accuracy:** {best_model['accuracy']:.4f}")
            if 'precision' in best_model:
                st.markdown(f"**Precision:** {best_model['precision']:.4f}")
        
        with col3:
            if 'recall' in best_model:
                st.markdown(f"**Recall:** {best_model['recall']:.4f}")
            if 'f1_score' in best_model:
                st.markdown(f"**F1 Score:** {best_model['f1_score']:.4f}")
    
    # Görselleştirmeler
    st.markdown("## 📊 Görselleştirmeler")
    
    # Accuracy comparison
    if 'accuracy' in results_df.columns:
        st.markdown("### 📈 Accuracy Karşılaştırması")
        
        fig_acc = px.bar(
            results_df,
            x='model_id',
            y='accuracy',
            title='Model Accuracy Karşılaştırması',
            color='accuracy',
            color_continuous_scale='viridis'
        )
        fig_acc.update_layout(height=400)
        st.plotly_chart(fig_acc, use_container_width=True)
    
    # Precision, Recall, F1 comparison
    if all(col in results_df.columns for col in ['precision', 'recall', 'f1_score']):
        st.markdown("### 📊 Precision, Recall, F1 Karşılaştırması")
        
        metrics_to_plot = ['precision', 'recall', 'f1_score']
        
        fig_metrics = go.Figure()
        
        for metric in metrics_to_plot:
            fig_metrics.add_trace(go.Bar(
                name=metric.title(),
                x=results_df['model_id'],
                y=results_df[metric]
            ))
        
        fig_metrics.update_layout(
            title='Precision, Recall, F1 Karşılaştırması',
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    # Virtual bankroll ROI comparison
    if 'virtual_bankroll_roi' in results_df.columns:
        st.markdown("### 💰 Virtual Bankroll ROI Karşılaştırması")
        
        fig_roi = px.bar(
            results_df,
            x='model_id',
            y='virtual_bankroll_roi',
            title='Virtual Bankroll ROI Karşılaştırması',
            color='virtual_bankroll_roi',
            color_continuous_scale='RdYlGn'
        )
        fig_roi.update_layout(height=400)
        st.plotly_chart(fig_roi, use_container_width=True)
    
    # Model tipi karşılaştırması
    st.markdown("### 🔍 Model Tipi Karşılaştırması")
    
    model_type_stats = results_df.groupby('model_type').agg({
        'accuracy': 'mean',
        'precision': 'mean',
        'recall': 'mean',
        'f1_score': 'mean'
    }).round(4)
    
    st.dataframe(model_type_stats, use_container_width=True)
    
    # Model tipi görselleştirmesi
    fig_type = px.bar(
        model_type_stats.reset_index(),
        x='model_type',
        y='accuracy',
        title='Model Tipi Ortalama Accuracy',
        color='accuracy',
        color_continuous_scale='viridis'
    )
    fig_type.update_layout(height=400)
    st.plotly_chart(fig_type, use_container_width=True)
    
    # Detaylı model analizi
    st.markdown("## 🔍 Detaylı Model Analizi")
    
    selected_model = st.selectbox(
        "Detaylı analiz için model seçin",
        results_df['model_id'].tolist()
    )
    
    if selected_model:
        model_row = results_df[results_df['model_id'] == selected_model].iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### 📋 {selected_model} Detayları")
            st.write(f"**Model Tipi:** {model_row['model_type']}")
            st.write(f"**Mod:** {model_row['mode']}")
            
            # Metrics
            st.markdown("**Metrikler:**")
            for col in numeric_cols:
                if col not in ['model_id', 'model_type', 'mode']:
                    st.write(f"  {col}: {model_row[col]:.4f}")
        
        with col2:
            # Model bilgileri
            model_info = st.session_state.model_manager.model_registry[selected_model]
            
            st.markdown("### ⚙️ Model Konfigürasyonu")
            if 'config' in model_info:
                config = model_info['config']
                for key, value in config.items():
                    st.write(f"**{key}:** {value}")
            
            # Eğitim tarihi
            if 'trained_at' in model_info:
                st.write(f"**Eğitildi:** {model_info['trained_at'][:16]}")
    
    # Export sonuçları
    st.markdown("## 💾 Sonuçları Export Et")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV export
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📊 CSV olarak indir",
            data=csv,
            file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # JSON export
        json_data = results_df.to_json(orient='records', indent=2)
        st.download_button(
            label="📄 JSON olarak indir",
            data=json_data,
            file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📊 Model Comparison - CPU Lightweight Models</p>
</div>
""", unsafe_allow_html=True)











