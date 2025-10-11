"""
JetX Predictor - Model Karşılaştırma Dashboard

Ensemble ve individual modellerin performanslarını karşılaştırmalı olarak gösterir.
Real-time tracking, confusion matrix, trend analizi.
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

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="🔬 Model Karşılaştırma",
    page_icon="🔬",
    layout="wide"
)

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
st.markdown("**3 base model + ensemble performans karşılaştırması**")

# İstatistikler özeti
summary = monitor.get_statistics_summary()

# Üst bilgi
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Toplam Tahmin", f"{summary['total_predictions']:,}")
with col2:
    st.metric("En İyi Model", summary.get('best_model', 'N/A').title() if summary.get('best_model') else 'Henüz yok')
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
    
    **Nasıl başlayabilirim?**
    1. Ana sayfaya gidin
    2. Ensemble mode'u seçin
    3. Tahmin yapın ve gerçek değeri girin
    4. Performans verileri otomatik olarak buraya yansıyacak
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

# En iyi modeli bul
best_model = max(models, key=lambda m: report.get(m, {}).get('accuracy', 0))
worst_model = min(models, key=lambda m: report.get(m, {}).get('accuracy', 1))

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
            <h2>{stats.get('accuracy', 0)*100:.1f}%</h2>
            <p>{stats.get('trend', '➡️ stable')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detaylı metrikler
        with st.expander("📋 Detaylar"):
            st.metric("Genel Accuracy", f"{stats.get('accuracy', 0)*100:.1f}%")
            st.metric("1.5 Altı Accuracy", f"{stats.get('below_1.5_accuracy', 0)*100:.1f}%")
            st.metric("1.5 Üstü Accuracy", f"{stats.get('above_1.5_accuracy', 0)*100:.1f}%")
            st.metric("MAE", f"{stats.get('mae', 0):.3f}")
            st.metric("Para Kaybı Riski", f"{stats.get('money_loss_risk', 0)*100:.1f}%")
            st.metric("Toplam Tahmin", stats.get('total_predictions', 0))

st.divider()

# Performance History Chart
st.subheader("📈 Performans Geçmişi (Rolling Average)")

chart_window = min(500, summary['total_predictions'])
chart_data = monitor.get_chart_data(window=chart_window)

if len(chart_data) > 0:
    # Plotly line chart
    fig = go.Figure()
    
    colors = {
        'progressive': '#3498db',
        'ultra': '#e74c3c',
        'xgboost': '#2ecc71',
        'ensemble': '#f39c12'
    }
    
    for model in models:
        col_name = f'{model}_correct_rolling'
        if col_name in chart_data.columns:
            fig.add_trace(go.Scatter(
                x=chart_data.index,
                y=chart_data[col_name] * 100,  # Yüzdeye çevir
                mode='lines',
                name=model_names[model],
                line=dict(color=colors[model], width=2),
                hovertemplate='%{y:.1f}%<extra></extra>'
            ))
    
    fig.update_layout(
        title=f"Rolling Accuracy Trend (Son {chart_window} Tahmin, 20-tahmin ortalaması)",
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
st.subheader("🎯 Confusion Matrix Karşılaştırması")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

cols = [col1, col2, col3, col4]

for col, model in zip(cols, models):
    with col:
        st.markdown(f"**{model_names[model]}**")
        
        cm = monitor.get_confusion_matrix(model, window=window_size)
        
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
        'Accuracy': f"{stats.get('accuracy', 0)*100:.1f}%",
        '1.5 Altı': f"{stats.get('below_1.5_accuracy', 0)*100:.1f}%",
        '1.5 Üstü': f"{stats.get('above_1.5_accuracy', 0)*100:.1f}%",
        'MAE': f"{stats.get('mae', 0):.3f}",
        'Para Kaybı': f"{stats.get('money_loss_risk', 0)*100:.1f}%",
        'Trend': stats.get('trend', '➡️'),
        'Tahmin #': stats.get('total_predictions', 0)
    })

df = pd.DataFrame(metrics_data)

# Renklendirme için stil
def highlight_best(s):
    """En iyi değeri yeşil, en kötüyü kırmızı yap"""
    if s.name in ['Accuracy', '1.5 Altı', '1.5 Üstü']:
        # Yüzdeleri sayıya çevir
        values = [float(x.strip('%')) for x in s]
        max_val = max(values)
        min_val = min(values)
        
        colors = []
        for v in values:
            if v == max_val:
                colors.append('background-color: #a8e063; color: black')
            elif v == min_val:
                colors.append('background-color: #f45c43; color: white')
            else:
                colors.append('')
        return colors
    elif s.name in ['MAE', 'Para Kaybı']:
        # Düşük iyi
        values = [float(x.strip('%')) for x in s]
        max_val = max(values)
        min_val = min(values)
        
        colors = []
        for v in values:
            if v == min_val:
                colors.append('background-color: #a8e063; color: black')
            elif v == max_val:
                colors.append('background-color: #f45c43; color: white')
            else:
                colors.append('')
        return colors
    else:
        return ['' for _ in s]

st.dataframe(
    df.style.apply(highlight_best, axis=0),
    use_container_width=True,
    hide_index=True
)

st.divider()

# Karşılaştırmalı Bar Charts
st.subheader("📊 Karşılaştırmalı Grafikler")

col1, col2 = st.columns(2)

with col1:
    # Accuracy karşılaştırması
    fig = go.Figure()
    
    accuracies = [report.get(m, {}).get('accuracy', 0) * 100 for m in models]
    
    fig.add_trace(go.Bar(
        x=[model_names[m] for m in models],
        y=accuracies,
        marker_color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'],
        text=[f'{a:.1f}%' for a in accuracies],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Genel Accuracy Karşılaştırması",
        yaxis_title="Accuracy (%)",
        height=400,
        yaxis=dict(range=[0, 100])
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Para kaybı riski karşılaştırması
    fig = go.Figure()
    
    risks = [report.get(m, {}).get('money_loss_risk', 0) * 100 for m in models]
    
    # Renk: Düşük=yeşil, Yüksek=kırmızı
    colors = []
    for r in risks:
        if r < 15:
            colors.append('#2ecc71')
        elif r < 25:
            colors.append('#f39c12')
        else:
            colors.append('#e74c3c')
    
    fig.add_trace(go.Bar(
        x=[model_names[m] for m in models],
        y=risks,
        marker_color=colors,
        text=[f'{r:.1f}%' for r in risks],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Para Kaybı Riski Karşılaştırması",
        yaxis_title="Risk (%)",
        height=400
    )
    
    # Hedef çizgisi (20%)
    fig.add_hline(
        y=20,
        line_dash="dash",
        line_color="red",
        annotation_text="Hedef: <20%",
        annotation_position="right"
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# MAE Karşılaştırması
st.subheader("📏 MAE (Mean Absolute Error) Karşılaştırması")

fig = go.Figure()

maes = [report.get(m, {}).get('mae', 0) for m in models]

fig.add_trace(go.Bar(
    x=[model_names[m] for m in models],
    y=maes,
    marker_color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'],
    text=[f'{mae:.3f}' for mae in maes],
    textposition='auto'
))

fig.update_layout(
    title="Value Prediction Error (MAE)",
    yaxis_title="MAE",
    height=400
)

st.plotly_chart(fig, use_container_width=True)

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

st.divider()

# Insights & Recommendations
st.subheader("💡 Öneriler ve İçgörüler")

# En iyi ve en kötü modeli analiz et
best_stats = report.get(best_model, {})
worst_stats = report.get(worst_model, {})

col1, col2 = st.columns(2)

with col1:
    st.success(f"**🏆 En İyi Model: {model_names[best_model]}**")
    st.write(f"- Accuracy: {best_stats.get('accuracy', 0)*100:.1f}%")
    st.write(f"- 1.5 Altı: {best_stats.get('below_1.5_accuracy', 0)*100:.1f}%")
    st.write(f"- Para Kaybı: {best_stats.get('money_loss_risk', 0)*100:.1f}%")
    st.write(f"- Trend: {best_stats.get('trend', '➡️')}")

with col2:
    st.error(f"**📉 En Düşük Performans: {model_names[worst_model]}**")
    st.write(f"- Accuracy: {worst_stats.get('accuracy', 0)*100:.1f}%")
    st.write(f"- 1.5 Altı: {worst_stats.get('below_1.5_accuracy', 0)*100:.1f}%")
    st.write(f"- Para Kaybı: {worst_stats.get('money_loss_risk', 0)*100:.1f}%")
    st.write(f"- Trend: {worst_stats.get('trend', '➡️')}")

# Genel öneriler
st.info("""
**🎯 Genel Değerlendirme:**

Ensemble sisteminin amacı, bireysel modellerin güçlü yönlerini birleştirerek daha iyi performans elde etmektir.

**İdeal Senaryo:**
- Ensemble accuracy > Tüm bireysel modellerin accuracy'si
- Para kaybı riski < %15
- 1.5 altı accuracy > %75

**Eğer Ensemble beklenenden kötüyse:**
1. Meta-model eğitime ihtiyaç duyuyor olabilir
2. Bazı base modeller çok kötü performans gösteriyor olabilir
3. Model ağırlıkları optimize edilmeli

**Sonraki Adımlar:**
- Düşük performanslı modelleri yeniden eğitin
- Meta-model training scriptini çalıştırın
- Hyperparameter optimization yapın
""")

# Footer
st.caption(f"Son güncelleme: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
