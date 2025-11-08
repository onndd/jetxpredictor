"""
JetX CPU Lightweight Models - Ana Streamlit Uygulaması

CPU ile eğitilebilen hafif modeller için özelleştirilmiş Streamlit uygulaması.
TabNet, AutoGluon, LightGBM, CatBoost gibi hafif modelleri destekler.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import logging
import yaml
import json
from pathlib import Path

# Utils modüllerini import et
from utils.database import DatabaseManager
from utils.lightweight_model_manager import LightweightModelManager
from utils.cpu_training_engine import CPUTrainingEngine
from utils.config_loader import config
from category_definitions import CategoryDefinitions

# Logging ayarla
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/cpu_models.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# CPU modelleri konfigürasyonunu yükle
def load_cpu_config():
    """CPU modelleri konfigürasyonunu yükle"""
    try:
        with open('config/cpu_models_config.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("CPU modelleri konfigürasyonu bulunamadı, varsayılan ayarlar kullanılıyor")
        return {}

# Konfigürasyonu yükle
CPU_CONFIG = load_cpu_config()

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="🚀 JetX CPU Lightweight Models",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS ile özelleştirme
st.markdown("""
<style>
    .big-font {
        font-size: 50px !important;
        font-weight: bold;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .metric-card {
        padding: 15px;
        border-radius: 8px;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        text-align: center;
        margin: 5px 0;
    }
    .model-card {
        padding: 15px;
        border-radius: 8px;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        margin: 10px 0;
    }
    .success-card {
        padding: 15px;
        border-radius: 8px;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        margin: 10px 0;
    }
    .warning-card {
        padding: 15px;
        border-radius: 8px;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin: 10px 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Session state başlat
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = LightweightModelManager()
    st.session_state.training_engine = CPUTrainingEngine(st.session_state.model_manager)
    st.session_state.training_in_progress = False
    st.session_state.current_model_id = None

# Ana başlık
st.markdown('<h1 class="big-font">🚀 JetX CPU Lightweight Models</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Model Yönetimi")
    
    # Model listesi
    model_df = st.session_state.model_manager.get_model_list()
    
    if not model_df.empty:
        st.markdown("### 📋 Eğitilmiş Modeller")
        for _, row in model_df.iterrows():
            status_color = "🟢" if row['status'] == 'trained' else "🟡" if row['status'] == 'created' else "🔴"
            st.write(f"{status_color} **{row['model_id']}**")
            st.write(f"   Tip: {row['type']} | Mod: {row['mode']}")
            st.write(f"   Durum: {row['status']}")
            if row['trained_at']:
                st.write(f"   Eğitildi: {row['trained_at'][:16]}")
            st.write("---")
    else:
        st.markdown("### 📋 Eğitilmiş Modeller")
        st.write("Henüz eğitilmiş model yok")
    
    # Sistem durumu
    st.markdown("### 🔧 Sistem Durumu")
    
    # Model availability
    available_models = st.session_state.model_manager.get_available_models()
    for model_name, info in available_models.items():
        status = "✅" if info else "❌"
        st.write(f"{status} {model_name.title()}")
    
    # Database durumu
    try:
        db_manager = DatabaseManager()
        db_status = "✅ Bağlı" if db_manager.is_connected() else "❌ Bağlantı Yok"
        st.write(f"Database: {db_status}")
    except:
        st.write("Database: ❌ Hata")
    
    # Memory usage (basit)
    import psutil
    memory_percent = psutil.virtual_memory().percent
    st.write(f"Memory: {memory_percent:.1f}%")
    
    # CPU usage (basit)
    cpu_percent = psutil.cpu_percent()
    st.write(f"CPU: {cpu_percent:.1f}%")

# Ana içerik
st.markdown("## 🎯 CPU Lightweight Models Dashboard")

# Model istatistikleri
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_models = len(model_df)
    st.metric("Toplam Model", total_models)

with col2:
    trained_models = len(model_df[model_df['status'] == 'trained'])
    st.metric("Eğitilmiş Model", trained_models)

with col3:
    model_types = model_df['type'].nunique() if not model_df.empty else 0
    st.metric("Model Tipi", model_types)

with col4:
    available_count = len(available_models)
    st.metric("Kullanılabilir Model", available_count)

# Hızlı eylemler
st.markdown("## ⚡ Hızlı Eylemler")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🎯 Model Eğit", use_container_width=True):
        st.switch_page("pages/cpu/1_🎯_Model_Training.py")

with col2:
    if st.button("🔧 Hyperparameter Tuning", use_container_width=True):
        st.switch_page("pages/cpu/2_🔧_Hyperparameter_Tuning.py")

with col3:
    if st.button("📊 Model Karşılaştır", use_container_width=True):
        st.switch_page("pages/cpu/3_📊_Model_Comparison.py")

with col4:
    if st.button("🤝 Ensemble Oluştur", use_container_width=True):
        st.switch_page("pages/cpu/4_🤝_Ensemble_Builder.py")

# Son eğitimler
if not model_df.empty:
    st.markdown("## 📈 Son Eğitimler")
    
    # Son 5 eğitimi göster
    recent_models = model_df.head(5)
    
    for _, row in recent_models.iterrows():
        with st.expander(f"🔍 {row['model_id']} ({row['type']}, {row['mode']})"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Durum:** {row['status']}")
                st.write(f"**Oluşturuldu:** {row['created_at'][:16]}")
            
            with col2:
                if row['trained_at']:
                    st.write(f"**Eğitildi:** {row['trained_at'][:16]}")
                else:
                    st.write("**Eğitildi:** Henüz eğitilmedi")
            
            with col3:
                if row['metrics'] and isinstance(row['metrics'], dict):
                    st.write("**Metrikler:**")
                    for metric, value in row['metrics'].items():
                        if isinstance(value, (int, float)):
                            st.write(f"  {metric}: {value:.4f}")
                else:
                    st.write("**Metrikler:** Yok")

# Model performansı özeti
if trained_models > 0:
    st.markdown("## 📊 Model Performansı Özeti")
    
    # En iyi modelleri göster
    trained_df = model_df[model_df['status'] == 'trained'].copy()
    
    if not trained_df.empty:
        # Accuracy'ye göre sırala
        accuracy_scores = []
        for _, row in trained_df.iterrows():
            if row['metrics'] and isinstance(row['metrics'], dict):
                accuracy = row['metrics'].get('accuracy', 0)
                accuracy_scores.append(accuracy)
            else:
                accuracy_scores.append(0)
        
        trained_df['accuracy'] = accuracy_scores
        best_models = trained_df.nlargest(3, 'accuracy')
        
        for i, (_, row) in enumerate(best_models.iterrows()):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            st.write(f"{rank} **{row['model_id']}** - Accuracy: {row['accuracy']:.4f}")

# Sistem bilgileri
st.markdown("## 🔧 Sistem Bilgileri")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📋 Kullanılabilir Modeller")
    for model_name, info in available_models.items():
        st.write(f"**{model_name.title()}:** {info['description']}")
        st.write(f"  Modlar: {', '.join(info['modes'])}")

with col2:
    st.markdown("### ⚙️ Konfigürasyon")
    st.write(f"**Window Size:** {CPU_CONFIG.get('training', {}).get('window_size', 1000)}")
    st.write(f"**Train Split:** {CPU_CONFIG.get('training', {}).get('train_split', 0.7)}")
    st.write(f"**Validation Split:** {CPU_CONFIG.get('training', {}).get('val_split', 0.15)}")
    st.write(f"**Test Split:** {CPU_CONFIG.get('training', {}).get('test_split', 0.15)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🚀 JetX CPU Lightweight Models v1.0 | CPU Optimized Training & Prediction</p>
    <p>Built with Streamlit, LightGBM, CatBoost, TabNet & AutoGluon</p>
</div>
""", unsafe_allow_html=True)

# Session state'i kaydet
if st.session_state.model_manager:
    st.session_state.model_manager.save_model_registry()






