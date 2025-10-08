"""
JetX Predictor - Ana Streamlit Uygulaması

Bu uygulama JetX tahmin sistemi için kullanıcı arayüzüdür.
Model Google Colab'da eğitilir, burada tahmin yapılır.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Utils modüllerini import et
from utils.database import DatabaseManager
from utils.predictor import JetXPredictor
from utils.risk_manager import RiskManager
from category_definitions import CategoryDefinitions

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="🚀 JetX Predictor",
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
    .safe-zone {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
    }
    .danger-zone {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
    .warning-box {
        padding: 10px;
        border-left: 5px solid #ff9800;
        background-color: #fff3e0;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = DatabaseManager("data/jetx_data.db")

if 'predictor' not in st.session_state:
    st.session_state.predictor = JetXPredictor()

if 'risk_manager' not in st.session_state:
    st.session_state.risk_manager = RiskManager()

if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

# Sidebar
with st.sidebar:
    st.title("🎮 Kontrol Paneli")
    
    # Mod seçimi
    st.subheader("📊 Tahmin Modu")
    mode = st.selectbox(
        "Mod seçin:",
        options=['normal', 'rolling', 'aggressive'],
        format_func=lambda x: {
            'normal': '🎯 Normal (Dengeli)',
            'rolling': '🛡️ Rolling (Konservatif)',
            'aggressive': '⚡ Agresif (Riskli)'
        }[x],
        help="Rolling: %80+ güven, Normal: %65+ güven, Agresif: %50+ güven"
    )
    
    st.session_state.risk_manager.set_mode(mode)
    
    st.divider()
    
    # İstatistikler
    st.subheader("📈 Genel İstatistikler")
    db_stats = st.session_state.db_manager.get_database_stats()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Toplam Veri", f"{db_stats['total_results']:,}")
        st.metric("1.5x Üstü", f"{db_stats['above_threshold_ratio']:.1%}")
    with col2:
        st.metric("Ortalama", f"{db_stats['average_value']:.2f}x")
        st.metric("Maksimum", f"{db_stats['max_value']:.2f}x")
    
    st.divider()
    
    # Risk seviyesi
    warning_level = st.session_state.risk_manager.get_warning_level()
    risk_colors = {
        'SAFE': '🟢',
        'CAUTION': '🟡',
        'WARNING': '🟠',
        'DANGER': '🔴'
    }
    st.subheader(f"{risk_colors.get(warning_level, '⚪')} Risk Seviyesi")
    st.write(warning_level)
    
    stats = st.session_state.risk_manager.get_statistics()
    if stats['total_predictions'] > 0:
        st.metric("Son 10 Doğruluk", f"{stats['accuracy']:.0%}")
        if stats['consecutive_wins'] > 0:
            st.success(f"🔥 {stats['consecutive_wins']} ardışık doğru!")
        if stats['consecutive_losses'] > 0:
            st.error(f"⚠️ {stats['consecutive_losses']} ardışık yanlış!")

# Ana içerik
st.title("🚀 JetX Tahmin Sistemi")
st.markdown("**AI destekli tahmin sistemi - Para kazandırmak için tasarlandı**")

# Model kontrolü
if st.session_state.predictor.model is None:
    st.error("⚠️ **Model yüklenmedi!** Önce Google Colab'da modeli eğitmeniz gerekiyor.")
    st.info("""
    **Yapılacaklar:**
    1. `notebooks/` klasöründeki Colab notebook'larını açın
    2. Modeli eğitin
    3. Eğitilmiş modeli `models/` klasörüne kaydedin
    4. Bu sayfayı yenileyin
    """)
else:
    st.success("✅ Model yüklendi ve hazır!")

st.divider()

# Ana iki kolon: Tahmin ve Grafik
main_col1, main_col2 = st.columns([1, 1])

with main_col1:
    st.subheader("🎯 Tahmin Yap")
    
    # Tahmin butonu
    if st.button("🔮 YENİ TAHMİN YAP", type="primary", use_container_width=True):
        with st.spinner("Tahmin yapılıyor..."):
            # Son verileri al
            history = st.session_state.db_manager.get_recent_results(500)
            
            if len(history) < 50:
                st.warning("⚠️ Tahmin için en az 50 veri gerekli!")
            else:
                # Tahmin yap
                prediction = st.session_state.predictor.predict(history, mode=mode)
                st.session_state.last_prediction = prediction
                
                # Risk analizi
                risk_decision = st.session_state.risk_manager.should_play(prediction)
                
                # Tahmini göster
                if 'error' in prediction:
                    st.error(f"❌ Hata: {prediction['error']}")
                else:
                    # Tahmin kartı
                    card_class = "safe-zone" if prediction['above_threshold'] else "danger-zone"
                    
                    st.markdown(f"""
                    <div class="prediction-card {card_class}">
                        <h2>Tahmin Edilen Değer</h2>
                        <p class="big-font">{prediction['predicted_value']:.2f}x</p>
                        <p>Güven: {prediction['confidence']:.0%}</p>
                        <p>{prediction['detailed_category']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Karar
                    st.subheader("🎲 Öneri")
                    if risk_decision['should_play']:
                        st.success(f"✅ **OYNA** - Risk: {risk_decision['risk_level']}")
                        
                        # Bahis önerisi
                        betting = st.session_state.risk_manager.get_betting_suggestion(prediction)
                        st.info(f"💡 Önerilen çıkış noktası: **{betting['suggested_multiplier']:.2f}x**")
                    else:
                        st.error("❌ **BEKLE** - Şu an oynamayın!")
                    
                    # Gerekçeler
                    with st.expander("📋 Detaylı Analiz"):
                        for reason in risk_decision['reasons']:
                            st.write(f"• {reason}")
                    
                    # Uyarılar
                    if prediction.get('warnings'):
                        st.subheader("⚠️ Uyarılar")
                        for warning in prediction['warnings']:
                            st.warning(warning)
    
    st.divider()
    
    # Son tahmin bilgisi
    if st.session_state.last_prediction:
        with st.expander("📊 Son Tahmin Detayları", expanded=False):
            pred = st.session_state.last_prediction
            if 'error' not in pred:
                st.json({
                    'Tahmin': f"{pred['predicted_value']:.2f}x",
                    'Güven': f"{pred['confidence']:.0%}",
                    '1.5x Üstü': 'Evet' if pred['above_threshold'] else 'Hayır',
                    'Kategori': pred['category'],
                    'Mod': pred['mode'].upper()
                })

with main_col2:
    st.subheader("📈 Son Değerler Grafiği")
    
    # Grafik boyutu seçici
    window_size = st.slider("Gösterilecek el sayısı:", 50, 200, 100)
    
    # Verileri al
    recent_data = st.session_state.db_manager.get_recent_results(window_size)
    
    if len(recent_data) > 0:
        # Plotly grafiği
        fig = go.Figure()
        
        # Ana çizgi
        fig.add_trace(go.Scatter(
            y=recent_data,
            mode='lines+markers',
            name='Değerler',
            line=dict(color='#667eea', width=2),
            marker=dict(size=4)
        ))
        
        # 1.5x eşik çizgisi
        fig.add_hline(
            y=1.5,
            line_dash="dash",
            line_color="red",
            annotation_text="1.5x Kritik Eşik",
            annotation_position="right"
        )
        
        # 3.0x çizgisi
        fig.add_hline(
            y=3.0,
            line_dash="dot",
            line_color="green",
            annotation_text="3.0x",
            annotation_position="right"
        )
        
        fig.update_layout(
            title=f"Son {len(recent_data)} El",
            xaxis_title="El",
            yaxis_title="Çarpan",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # İstatistikler
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Ortalama", f"{np.mean(recent_data):.2f}x")
        with col2:
            st.metric("Minimum", f"{np.min(recent_data):.2f}x")
        with col3:
            st.metric("Maksimum", f"{np.max(recent_data):.2f}x")
        with col4:
            above_15 = sum(1 for v in recent_data if v >= 1.5)
            st.metric("1.5x Üstü", f"{above_15}/{len(recent_data)}")
    else:
        st.info("📊 Henüz veri bulunmuyor.")

st.divider()

# Veri girişi bölümü
st.subheader("➕ Yeni Veri Ekle")

col1, col2 = st.columns([3, 1])
with col1:
    new_value = st.number_input(
        "Gerçekleşen değeri girin:",
        min_value=1.0,
        max_value=10000.0,
        value=1.5,
        step=0.01,
        format="%.2f"
    )
with col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    if st.button("💾 Kaydet", use_container_width=True):
        # Veritabanına ekle
        st.session_state.db_manager.add_result(new_value)
        st.success(f"✅ {new_value:.2f}x kaydedildi!")
        
        # Eğer bekleyen tahmin varsa, değerlendir
        if st.session_state.last_prediction and 'error' not in st.session_state.last_prediction:
            evaluation = st.session_state.risk_manager.evaluate_prediction(
                st.session_state.last_prediction,
                new_value
            )
            
            if evaluation['threshold_correct']:
                st.success(f"🎉 Tahmin doğru! (1.5x eşik tahmini)")
            else:
                st.error(f"❌ Tahmin yanlış!")
            
            st.info(f"Ardışık: {evaluation['consecutive_wins']} doğru, {evaluation['consecutive_losses']} yanlış")
        
        st.rerun()

st.divider()

# Alt bilgi
st.markdown("""
---
### ⚠️ ÖNEMLİ UYARILAR

- 🚨 **Bu sistem %100 doğru değildir**
- 💰 **Para kaybedebilirsiniz**
- 🎯 **1.5x kritik eşiktir**: Altı kayıp, üstü kazanç
- 🛡️ **Rolling modu** en güvenlidir (%80+ güven)
- ⚡ **Agresif mod** çok risklidir

**Sorumlu oynayın!**
""")

# Footer
st.caption(f"JetX Predictor v1.0 | Mod: {mode.upper()} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
