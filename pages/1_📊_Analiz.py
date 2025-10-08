"""
JetX Predictor - Veri Analiz Sayfası

Bu sayfa veritabanındaki verilerin detaylı analizini gösterir.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sys
import os

# Ana dizini path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import DatabaseManager
from category_definitions import CategoryDefinitions

st.set_page_config(
    page_title="Veri Analizi - JetX Predictor",
    page_icon="📊",
    layout="wide"
)

# Database manager
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = DatabaseManager("data/jetx_data.db")

st.title("📊 Veri Analizi")
st.markdown("Veritabanındaki tüm verilerin detaylı analizi")

# Genel İstatistikler
st.header("📈 Genel İstatistikler")

db_stats = st.session_state.db_manager.get_database_stats()
all_data = st.session_state.db_manager.get_all_results()

if len(all_data) > 0:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Toplam Kayıt", f"{db_stats['total_results']:,}")
        st.metric("Ortalama Değer", f"{db_stats['average_value']:.2f}x")
    
    with col2:
        st.metric("Minimum", f"{db_stats['min_value']:.2f}x")
        st.metric("Maksimum", f"{db_stats['max_value']:.2f}x")
    
    with col3:
        st.metric("1.5x Üstü Oran", f"{db_stats['above_threshold_ratio']:.1%}")
        median = np.median(all_data)
        st.metric("Medyan", f"{median:.2f}x")
    
    with col4:
        std = np.std(all_data)
        st.metric("Standart Sapma", f"{std:.2f}")
        q75 = np.percentile(all_data, 75)
        st.metric("75. Persentil", f"{q75:.2f}x")
    
    st.divider()
    
    # Kategori Dağılımı
    st.header("📋 Kategori Dağılımı")
    
    # Kategorileri hesapla
    categories = {}
    for value in all_data:
        cat = CategoryDefinitions.get_detailed_category(value)
        categories[cat] = categories.get(cat, 0) + 1
    
    # DataFrame oluştur
    df_categories = pd.DataFrame([
        {'Kategori': k, 'Adet': v, 'Yüzde': (v/len(all_data))*100}
        for k, v in sorted(categories.items(), key=lambda x: x[1], reverse=True)
    ])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Pasta grafiği
        fig = px.pie(
            df_categories,
            values='Adet',
            names='Kategori',
            title='Kategori Dağılımı'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(
            df_categories.style.format({
                'Adet': '{:,}',
                'Yüzde': '{:.2f}%'
            }),
            use_container_width=True,
            height=400
        )
    
    st.divider()
    
    # Histogram
    st.header("📊 Değer Dağılımı Histogramı")
    
    # Histogramı oluştur
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=all_data,
        nbinsx=50,
        name='Frekans',
        marker_color='#667eea'
    ))
    
    # 1.5x çizgisi
    fig.add_vline(
        x=1.5,
        line_dash="dash",
        line_color="red",
        annotation_text="1.5x Kritik Eşik"
    )
    
    # 3.0x çizgisi
    fig.add_vline(
        x=3.0,
        line_dash="dash",
        line_color="green",
        annotation_text="3.0x"
    )
    
    fig.update_layout(
        title="Değer Dağılımı",
        xaxis_title="Çarpan Değeri",
        yaxis_title="Frekans",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Zaman Serisi Analizi
    st.header("📈 Zaman Serisi Analizi")
    
    window = st.slider("Hareketli Ortalama Penceresi:", 10, 100, 50)
    
    # Hareketli ortalama hesapla
    moving_avg = pd.Series(all_data).rolling(window=window).mean()
    
    fig = go.Figure()
    
    # Ham veri
    fig.add_trace(go.Scatter(
        y=all_data,
        mode='lines',
        name='Ham Veri',
        line=dict(color='lightgray', width=1),
        opacity=0.5
    ))
    
    # Hareketli ortalama
    fig.add_trace(go.Scatter(
        y=moving_avg,
        mode='lines',
        name=f'{window} Periyot Hareketli Ortalama',
        line=dict(color='#667eea', width=2)
    ))
    
    # Eşikler
    fig.add_hline(y=1.5, line_dash="dash", line_color="red", annotation_text="1.5x")
    fig.add_hline(y=3.0, line_dash="dash", line_color="green", annotation_text="3.0x")
    
    fig.update_layout(
        title="Tüm Veri Trendi",
        xaxis_title="Oyun #",
        yaxis_title="Çarpan",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Kritik Eşik Analizi
    st.header("🎯 1.5x Kritik Eşik Detaylı Analizi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 1.4x - 1.6x arası detaylı bakış
        critical_zone = [v for v in all_data if 1.4 <= v <= 1.6]
        
        st.subheader("Kritik Bölge (1.4x - 1.6x)")
        st.metric("Kritik Bölgedeki Değer Sayısı", len(critical_zone))
        
        if len(critical_zone) > 0:
            below_15 = len([v for v in critical_zone if v < 1.5])
            above_15 = len([v for v in critical_zone if v >= 1.5])
            
            st.write(f"**1.5x Altı:** {below_15} ({below_15/len(critical_zone)*100:.1f}%)")
            st.write(f"**1.5x Üstü:** {above_15} ({above_15/len(critical_zone)*100:.1f}%)")
    
    with col2:
        # Ardışık 1.5x altı/üstü analizi
        st.subheader("Ardışık Pattern Analizi")
        
        consecutive_below = 0
        consecutive_above = 0
        max_consecutive_below = 0
        max_consecutive_above = 0
        
        for value in all_data:
            if value < 1.5:
                consecutive_below += 1
                consecutive_above = 0
                max_consecutive_below = max(max_consecutive_below, consecutive_below)
            else:
                consecutive_above += 1
                consecutive_below = 0
                max_consecutive_above = max(max_consecutive_above, consecutive_above)
        
        st.write(f"**Max Ardışık Kayıp (<1.5x):** {max_consecutive_below}")
        st.write(f"**Max Ardışık Kazanç (≥1.5x):** {max_consecutive_above}")
    
    st.divider()
    
    # Büyük Çarpanlar
    st.header("🚀 Büyük Çarpan Analizi")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        big_10x = [v for v in all_data if v >= 10.0]
        st.metric("10x+ Çarpanlar", len(big_10x))
        if big_10x:
            st.write(f"En yüksek: {max(big_10x):.2f}x")
    
    with col2:
        big_50x = [v for v in all_data if v >= 50.0]
        st.metric("50x+ Çarpanlar", len(big_50x))
        if big_50x:
            st.write(f"Ortalama: {np.mean(big_50x):.2f}x")
    
    with col3:
        big_100x = [v for v in all_data if v >= 100.0]
        st.metric("100x+ Çarpanlar", len(big_100x))
        if big_100x:
            st.write(f"Toplam: {len(big_100x)}")

else:
    st.info("📊 Henüz analiz için yeterli veri yok. Lütfen ana sayfadan veri ekleyin.")

st.divider()
st.caption("JetX Predictor - Veri Analiz Sayfası")
