"""
JetX Predictor - 5 Model Ensemble Arayüzü

Bu uygulama tüm 5 modelin tahminlerini gösterir:
1. Progressive NN
2. CatBoost  
3. AutoGluon
4. TabNet
5. Consensus
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import logging

# Utils modüllerini import et
from utils.database import DatabaseManager
from utils.all_models_predictor import AllModelsPredictor
from utils.config_loader import config
from category_definitions import CategoryDefinitions

# Logging ayarla
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="🚀 JetX 5 Model Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS
st.markdown("""
<style>
    /* Model Kartları */
    .model-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        border-left: 4px solid #667eea;
        transition: transform 0.2s;
    }
    {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .model-card.best {
        border-left: 4px solid #10b981;
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    }
    
    .model-card.unavailable {
        opacity: 0.5;
        border-left: 4px solid #9ca3af;
    }
    
    .model-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }
    
    .model-name {
        font-size: 18px;
        font-weight: 600;
        color: #1f2937;
    }
    
    .model-badge {
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
    }
    
    .badge-loaded {
        background: #d1fae5;
        color: #065f46;
    }
    
    .badge-unavailable {
        background: #fee2e2;
        color: #991b1b;
    }
    
    .prediction-value {
        font-size: 36px;
        font-weight: 700;
        color: #667eea;
        text-align: center;
        margin: 15px 0;
    }
    
    .confidence-bar {
        width: 100%;
        height: 8px;
        background: #e5e7eb;
        border-radius: 4px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #10b981 0%, #3b82f6 50%, #8b5cf6 100%);
        transition: width 0.3s;
    }
    
    .confidence-text {
        font-size: 14px;
        color: #6b7280;
        text-align: center;
    }
    
    .consensus-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
        margin: 20px 0;
    }
    
    .consensus-title {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .consensus-value {
        font-size: 48px;
        font-weight: 800;
        text-align: center;
        margin: 20px 0;
    }
    
    .agreement-indicator {
        display: flex;
        justify-content: center;
        gap: 8px;
        margin: 15px 0;
    }
    
    .agreement-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
    }
    
    .agreement-dot.active {
        background: #10b981;
    }
    
    .category-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 500;
        margin: 5px 0;
    }
    
    .category-low {
        background: #fee2e2;
        color: #991b1b;
    }
    
    .category-medium {
        background: #fef3c7;
        color: #92400e;
    }
    
    .category-high {
        background: #d1fae5;
        color: #065f46;
    }
    
    .category-mega {
        background: #ddd6fe;
        color: #5b21b6;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'db_manager' not in st.session_state:
    db_path = config.get('database.path', 'jetx_data.db')
    st.session_state.db_manager = DatabaseManager(db_path)
    logger.info(f"Database manager başlatıldı: {db_path}")

if 'all_models_predictor' not in st.session_state:
    st.session_state.all_models_predictor = AllModelsPredictor()
    # Modelleri yükle
    with st.spinner("Modeller yükleniyor..."):
        load_results = st.session_state.all_models_predictor.load_all_models()
    logger.info(f"Modeller yüklendi: {load_results}")

if 'last_predictions' not in st.session_state:
    st.session_state.last_predictions = None

# Sidebar
with st.sidebar:
    st.title("🎮 Model Kontrol Paneli")
    
    # Model durumları
    st.subheader("📊 Model Durumları")
    predictor = st.session_state.all_models_predictor
    
    model_names = {
        'progressive_nn': '🧠 Progressive NN',
        'catboost': '🚀 CatBoost',
        'autogluon': '🤖 AutoGluon',
        'tabnet': '🎯 TabNet'
    }
    
    for model_key, model_name in model_names.items():
        if model_key in predictor.available_models:
            st.success(f"✅ {model_name}")
        else:
            st.error(f"❌ {model_name}")
    
    if len(predictor.available_models) >= 2:
        st.info(f"🎉 Consensus aktif ({len(predictor.available_models)} model)")
    else:
        st.warning("⚠️ Consensus için en az 2 model gerekli")
    
    st.divider()
    
    # İstatistikler
    st.subheader("📈 Veritabanı")
    db_stats = st.session_state.db_manager.get_database_stats()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Toplam Veri", f"{db_stats['total_results']:,}")
        st.metric("Ortalama", f"{db_stats['average_value']:.2f}x")
    with col2:
        st.metric("1.5x Üstü", f"{db_stats['above_threshold_ratio']:.0%}")
        st.metric("Maksimum", f"{db_stats['max_value']:.2f}x")

# Ana başlık
st.title("🚀 JetX 5 Model Ensemble Tahmin Sistemi")
st.markdown("**Tüm modellerin gücünü birleştiren AI destekli tahmin platformu**")

# Model yükleme durumu banner
if len(predictor.available_models) == 4:
    st.success("✨ Tüm 5 model (NN, CatBoost, AutoGluon, TabNet + Consensus) yüklendi ve hazır!")
elif len(predictor.available_models) >= 2:
    st.info(f"📊 {len(predictor.available_models)} model yüklendi. Consensus sistemi aktif.")
elif len(predictor.available_models) == 1:
    st.warning("⚠️ Sadece 1 model yüklü. Consensus için daha fazla model eğitin.")
else:
    st.error("❌ Hiç model yüklenmedi! Lütfen modelleri Google Colab'da eğitin.")

st.divider()

# Tahmin butonu
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔮 TÜM MODELLERDEN TAHMİN AL", type="primary", use_container_width=True):
        if len(predictor.available_models) == 0:
            st.error("❌ Hiç model yüklenmedi! Önce modelleri eğitin.")
        else:
            with st.spinner("Tüm modellerden tahminler alınıyor..."):
                # Son verileri al
                history = st.session_state.db_manager.get_recent_results(500)
                history = np.array([float(v) for v in history])
                
                if len(history) < 50:
                    st.warning("⚠️ Tahmin için en az 50 veri gerekli!")
                else:
                    # Tüm modellerden tahmin al
                    predictions = predictor.predict_all(history)
                    st.session_state.last_predictions = predictions
                    st.success("✅ Tahminler tamamlandı!")

st.divider()

# Tahminleri göster
if st.session_state.last_predictions:
    predictions = st.session_state.last_predictions
    
    # Consensus kartı (varsa)
    if predictions.get('consensus'):
        consensus = predictions['consensus']
        
        # Renk belirleme
        if consensus['confidence'] >= 0.8:
            gradient = "linear-gradient(135deg, #10b981 0%, #059669 100%)"
        elif consensus['confidence'] >= 0.6:
            gradient = "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)"
        else:
            gradient = "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)"
        
        st.markdown(f"""
        <div class="consensus-card" style="background: {gradient};">
            <div class="consensus-title">🏆 CONSENSUS TAHMİNİ</div>
            <div class="consensus-value">{consensus['prediction']:.2f}x</div>
            <div style="text-align: center; margin: 15px 0;">
                <span style="font-size: 18px; font-weight: 500;">
                    {'✅ OYNA' if consensus['above_threshold'] else '⛔ BEKLE'}
                </span>
            </div>
            <div style="text-align: center; margin: 10px 0;">
                <span style="font-size: 14px; opacity: 0.9;">
                    Güven: {consensus['confidence']*100:.0f}% | 
                    Uzlaşma: {consensus['agreement']*100:.0f}% 
                    ({consensus['models_agreed']}/{consensus['total_models']} model hemfikir)
                </span>
            </div>
            <div style="text-align: center;">
                <span class="category-badge category-{consensus['category'].lower()}">
                    {consensus['category']}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Uzlaşma göstergesi
        st.markdown("### 🤝 Model Uzlaşması")
        agreement_cols = st.columns(4)
        for idx, (model_key, model_name) in enumerate([
            ('progressive_nn', 'Progressive NN'),
            ('catboost', 'CatBoost'),
            ('autogluon', 'AutoGluon'),
            ('tabnet', 'TabNet')
        ]):
            with agreement_cols[idx]:
                if model_key in predictions and predictions[model_key]:
                    pred = predictions[model_key]
                    if pred['above_threshold']:
                        st.success(f"✅ {model_name}")
                    else:
                        st.error(f"❌ {model_name}")
                else:
                    st.info(f"⚪ {model_name}")
    
    st.divider()
    
    # Bireysel model tahminleri
    st.markdown("### 📊 Bireysel Model Tahminleri")
    
    # Model kartları için grid
    model_cols = st.columns(2)
    
    model_configs = [
        {
            'key': 'progressive_nn',
            'name': '🧠 Progressive NN',
            'description': 'Multi-Scale Ensemble'
        },
        {
            'key': 'catboost',
            'name': '🚀 CatBoost',
            'description': 'Gradient Boosting'
        },
        {
            'key': 'autogluon',
            'name': '🤖 AutoGluon',
            'description': 'AutoML Champion'
        },
        {
            'key': 'tabnet',
            'name': '🎯 TabNet',
            'description': 'High-X Specialist'
        }
    ]
    
    for idx, model_config in enumerate(model_configs):
        with model_cols[idx % 2]:
            model_key = model_config['key']
            model_name = model_config['name']
            model_desc = model_config['description']
            
            if model_key in predictions and predictions[model_key]:
                pred = predictions[model_key]
                
                # En iyi model mi?
                is_best = False
                if predictions.get('consensus'):
                    # Consensus tahminine en yakın olan
                    consensus_val = predictions['consensus']['prediction']
                    if abs(pred['prediction'] - consensus_val) < 0.2:
                        is_best = True
                
                card_class = "best" if is_best else ""
                
                # Kategori rengi
                category_lower = pred['category'].lower()
                if category_lower in ['düşük', 'low']:
                    cat_class = 'category-low'
                elif category_lower in ['orta', 'medium', 'orta']:
                    cat_class = 'category-medium'
                elif category_lower in ['yüksek', 'high']:
                    cat_class = 'category-high'
                else:
                    cat_class = 'category-mega'
                
                st.markdown(f"""
                <div class="model-card {card_class}">
                    <div class="model-header">
                        <div>
                            <div class="model-name">{model_name}</div>
                            <div style="font-size: 12px; color: #6b7280;">{model_desc}</div>
                        </div>
                        <span class="model-badge badge-loaded">Aktif</span>
                    </div>
                    <div class="prediction-value">{pred['prediction']:.2f}x</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {pred['confidence']*100}%"></div>
                    </div>
                    <div class="confidence-text">Güven: {pred['confidence']*100:.0f}%</div>
                    <div style="text-align: center; margin-top: 10px;">
                        <span class="category-badge {cat_class}">{pred['category']}</span>
                    </div>
                    <div style="text-align: center; margin-top: 10px; font-size: 16px;">
                        {'🟢 1.5x Üstü' if pred['above_threshold'] else '🔴 1.5x Altı'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Model yüklenmemiş
                st.markdown(f"""
                <div class="model-card unavailable">
                    <div class="model-header">
                        <div>
                            <div class="model-name">{model_name}</div>
                            <div style="font-size: 12px; color: #6b7280;">{model_desc}</div>
                        </div>
                        <span class="model-badge badge-unavailable">Yüklenmedi</span>
                    </div>
                    <div style="text-align: center; padding: 40px 0; color: #9ca3af;">
                        <div style="font-size: 48px;">-</div>
                        <div style="margin-top: 10px;">Model eğitilmemiş</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

st.divider()

# Son değerler grafiği
st.subheader("📈 Son Değerler")
window_size = st.slider("Gösterilecek el sayısı:", 50, 200, 100, key="graph_window")
recent_data = st.session_state.db_manager.get_recent_results(window_size)
recent_data = [float(v) for v in recent_data]

if len(recent_data) > 0:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=recent_data,
        mode='lines+markers',
        name='Değerler',
        line=dict(color='#667eea', width=2),
        marker=dict(size=4)
    ))
    fig.add_hline(y=1.5, line_dash="dash", line_color="red", 
                  annotation_text="1.5x Kritik Eşik")
    fig.update_layout(
        title=f"Son {len(recent_data)} El",
        xaxis_title="El",
        yaxis_title="Çarpan",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Veri girişi
st.subheader("➕ Yeni Veri Ekle")
col1, col2 = st.columns([3, 1])
with col1:
    new_value = st.number_input(
        "Gerçekleşen değeri girin:",
        min_value=1.0,
        max_value=1000.0,
        value=1.5,
        step=0.01,
        format="%.2f"
    )
with col2:
    st.write("")
    st.write("")
    if st.button("💾 Kaydet", use_container_width=True):
        try:
            result_id = st.session_state.db_manager.add_result(new_value)
            if result_id > 0:
                st.success(f"✅ {new_value:.2f}x kaydedildi!")
                st.rerun()
        except Exception as e:
            st.error(f"❌ Hata: {e}")

st.divider()

# Footer
st.caption(f"""
🚀 JetX 5 Model Ensemble v1.0 | 
Yüklü Modeller: {len(predictor.available_models)}/4 | 
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
