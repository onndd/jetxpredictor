"""
JetX Predictor - Model Karşılaştırma Dashboard

Ensemble ve individual modellerin performanslarını karşılaştırmalı olarak gösterir.
Real-time tracking, confusion matrix, trend analizi.

GÜNCELLEME:
- 2 Modlu Yapı (Normal/Rolling) entegrasyonu.
- Normal Mod (0.85) ve Rolling Mod (0.95) performansları ayrı ayrı raporlanır.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Utils modüllerini import et
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.ensemble_monitor import EnsembleMonitor
from utils.database import DatabaseManager
from utils.threshold_manager import get_threshold_manager

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Model Karşılaştırma",
    page_icon="🔬",
    layout="wide"
)

# Threshold Manager
tm = get_threshold_manager()
THRESHOLD_NORMAL = tm.get_normal_threshold()
THRESHOLD_ROLLING = tm.get_rolling_threshold()

# CSS
st.markdown("""
<style>
    .metric-card {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .best-model {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%) !important;
    }
    .worst-model {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%) !important;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'ensemble_monitor' not in st.session_state:
    st.session_state.ensemble_monitor = EnsembleMonitor()

monitor = st.session_state.ensemble_monitor

# Başlık
st.title("🔬 Model Karşılaştırma & Ensemble Monitoring")
st.markdown(f"**Modellerin Normal ({THRESHOLD_NORMAL}) ve Rolling ({THRESHOLD_ROLLING}) mod performansları**")

# İstatistikler özeti
summary = monitor.get_statistics_summary()

# Üst bilgi
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Toplam Tahmin", f"{summary['total_predictions']:,}")
with col2:
    st.metric("En İyi Model (Normal)", summary.get('best_model', 'N/A').title() if summary.get('best_model') else 'Henüz yok')
with col3:
    if summary.get('best_accuracy'):
        st.metric("En İyi Accuracy", f"{summary['best_accuracy']*100:.1f}%")
with col4:
    if summary.get('last_prediction'):
        st.metric("Son Tahmin", summary['last_prediction'][:19] if summary['last_prediction'] else 'Henüz yok')

st.divider()

# Tahmin sayısı kontrolü
if summary['total_predictions'] == 0:
    st.info("""
    📊 **Henüz tahmin verisi yok!**
    
    Ensemble sistemi ile tahmin yapmaya başladıktan sonra buradan performans karşılaştırması görebilirsiniz.
    """)
    st.stop()

# Window size seçici
window_size = st.select_slider(
    "Analiz Penceresi (Son N Tahmin)",
    options=[50, 100, 200, 500, min(1000, summary['total_predictions'])],
    value=min(100, summary['total_predictions'])
)

# Comparison report
st.subheader("📊 Model Performans Karşılaştırması")

report = monitor.generate_comparison_report(window=window_size)

# Metrics tablosu
col1, col2, col3, col4 = st.columns(4)

models = ['progressive', 'ultra', 'xgboost', 'ensemble']
model_names = {
    'progressive': '🎯 Progressive',
    'ultra': '⚡ Ultra Aggressive',
    'xgboost': '🤖 XGBoost',
    'ensemble': '⭐ Ensemble'
}

# En iyi modeli bul (Normal Mod Accuracy bazlı)
best_model = max(models, key=lambda m: report.get(m, {}).get('accuracy_normal', 0))
worst_model = min(models, key=lambda m: report.get(m, {}).get('accuracy_normal', 1))

for col, model in zip([col1, col2, col3, col4], models):
    with col:
        stats = report.get(model, {})
        
        # Card sınıfı
        card_class = ""
        if model == best_model:
            card_class = "best-model"
        elif model == worst_model:
            card_class = "worst-model"
        
        st.markdown(f"""
        <div class="metric-card {card_class}">
            <h3>{model_names[model]}</h3>
            <h2>{stats.get('accuracy_normal', 0)*100:.1f}%</h2>
            <p>Normal Mod</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detaylı metrikler
        with st.expander("📋 Detaylar"):
            st.metric("Normal Mod Acc", f"{stats.get('accuracy_normal', 0)*100:.1f}%")
            st.metric("Rolling Mod Acc", f"{stats.get('accuracy_rolling', 0)*100:.1f}%")
            st.metric("MAE", f"{stats.get('mae', 0):.3f}")
            st.metric("Trend", stats.get('trend', '➡️'))

st.divider()

# Performance History Chart
st.subheader("📈 Performans Geçmişi (Rolling Average)")

chart_window = min(500, summary['total_predictions'])
chart_data = monitor.get_chart_data(window=chart_window)

if len(chart_data) > 0:
    # Plotly line chart
    fig = go.Figure()
    
    # Sadece Ensemble Normal ve Rolling modlarını göster (kalabalık olmasın)
    if 'ensemble_correct_normal' in chart_data.columns:
         # Rolling mean uygula
         y_normal = chart_data['ensemble_correct_normal'].rolling(window=20).mean() * 100
         fig.add_trace(go.Scatter(
            x=chart_data.index, y=y_normal, mode='lines', name='Ensemble Normal',
            line=dict(color='#f39c12', width=2)
        ))
         
    if 'ensemble_correct_rolling' in chart_data.columns:
         y_rolling = chart_data['ensemble_correct_rolling'].rolling(window=20).mean() * 100
         fig.add_trace(go.Scatter(
            x=chart_data.index, y=y_rolling, mode='lines', name='Ensemble Rolling',
            line=dict(color='#2ecc71', width=2)
        ))
    
    fig.update_layout(
        title=f"Ensemble Rolling Accuracy Trend (Son {chart_window} Tahmin)",
        xaxis_title="Tahmin Sırası",
        yaxis_title="Accuracy (%)",
        hovermode='x unified',
        height=400,
        yaxis=dict(range=[0, 100])
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Grafik için yeterli veri yok.")

st.divider()

# Confusion Matrix Karşılaştırması
st.subheader("🎯 Confusion Matrix (Normal Mod - 0.85)")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

cols = [col1, col2, col3, col4]

for col, model in zip(cols, models):
    with col:
        st.markdown(f"**{model_names[model]}**")
        
        cm = np.array(report.get(model, {}).get('confusion_matrix', [[0,0],[0,0]]))
        
        if cm.sum() > 0:
            # Heatmap oluştur
            fig = go.Figure(data=go.Heatmap(
                z=[[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]],
                x=['1.5 Altı (Pred)', '1.5 Üstü (Pred)'],
                y=['1.5 Altı (True)', '1.5 Üstü (True)'],
                colorscale='RdYlGn',
                text=[[f'{cm[0][0]}', f'{cm[0][1]}'], [f'{cm[1][0]}', f'{cm[1][1]}']],
                texttemplate='%{text}',
                textfont={"size": 16},
                showscale=False
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            tn, fp = cm[0]
            fn, tp = cm[1]
            
            if tn + fp > 0:
                st.caption(f"⚠️ Para Kaybı: {fp} / {tn + fp} ({fp/(tn+fp)*100:.1f}%)")
        else:
            st.info("Veri yok")

st.divider()

# Detaylı Metrik Tabloları
st.subheader("📋 Detaylı Metrik Tablosu")

# DataFrame oluştur
metrics_data = []
for model in models:
    stats = report.get(model, {})
    metrics_data.append({
        'Model': model_names[model],
        'Normal Acc': f"{stats.get('accuracy_normal', 0)*100:.1f}%",
        'Rolling Acc': f"{stats.get('accuracy_rolling', 0)*100:.1f}%",
        'MAE': f"{stats.get('mae', 0):.3f}",
        'Trend': stats.get('trend', '➡️'),
        'Para Kaybı': f"{stats.get('money_loss_risk', 0)*100:.1f}%"
    })

df = pd.DataFrame(metrics_data)

# Renklendirme
def highlight_best(s):
    if s.name in ['Normal Acc', 'Rolling Acc']:
        values = [float(x.strip('%')) for x in s]
        max_val = max(values)
        return ['background-color: #a8e063; color: black' if v == max_val else '' for v in values]
    elif s.name in ['MAE', 'Para Kaybı']:
        values = [float(x.strip('%')) for x in s]
        min_val = min(values)
        return ['background-color: #a8e063; color: black' if v == min_val else '' for v in values]
    return ['' for _ in s]

st.dataframe(
    df.style.apply(highlight_best, axis=0),
    use_container_width=True,
    hide_index=True
)

st.divider()

# Export Section
st.subheader("💾 Veri Export")

col1, col2 = st.columns(2)

with col1:
    if st.button("📥 CSV Olarak İndir (Son 1000 Tahmin)", use_container_width=True):
        with st.spinner("CSV oluşturuluyor..."):
            filename = monitor.export_to_csv(
                filename="data/ensemble_performance_export.csv",
                window=min(1000, summary['total_predictions'])
            )
            st.success(f"✅ CSV kaydedildi: {filename}")

with col2:
    if st.button("📥 Tüm Verileri İndir", use_container_width=True):
        with st.spinner("CSV oluşturuluyor..."):
            filename = monitor.export_to_csv(
                filename="data/ensemble_performance_full.csv",
                window=None
            )
            st.success(f"✅ CSV kaydedildi: {filename}")

st.caption(f"Son güncelleme: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
