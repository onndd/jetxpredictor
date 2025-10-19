"""
Prediction & Backtesting Sayfası

Eğitilmiş modellerle tahmin yapma ve backtesting.
Real-time prediction, historical backtesting, performance analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json

# Utils imports
from utils.database import DatabaseManager
from utils.lightweight_model_manager import LightweightModelManager
from utils.cpu_training_engine import CPUTrainingEngine

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="🎲 Prediction & Backtesting - CPU Lightweight Models",
    page_icon="🎲",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 10px 0;
    }
    .backtest-card {
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
    st.session_state.prediction_results = None
    st.session_state.backtest_results = None

# Ana başlık
st.markdown("# 🎲 Prediction & Backtesting")
st.markdown("Eğitilmiş modellerle tahmin yapma ve backtesting")

# Model listesi
model_df = st.session_state.model_manager.get_model_list()
trained_models = model_df[model_df['status'] == 'trained']

if trained_models.empty:
    st.warning("⚠️ Henüz eğitilmiş model yok. Önce model eğitin.")
    st.stop()

# Model seçimi
st.markdown("## 🎯 Model Seçimi")

selected_model = st.selectbox(
    "Model Seçin",
    trained_models['model_id'].tolist(),
    help="Tahmin yapmak için model seçin"
)

if not selected_model:
    st.stop()

# Model bilgileri
model_info = st.session_state.model_manager.model_registry[selected_model]
st.info(f"**Model:** {selected_model} | **Tip:** {model_info['type']} | **Mod:** {model_info['mode']}")

# Tahmin modu seçimi
st.markdown("## 🎯 Tahmin Modu")

prediction_mode = st.selectbox(
    "Tahmin Modu",
    ['Real-time Prediction', 'Historical Backtesting'],
    help="Tahmin modunu seçin"
)

if prediction_mode == 'Real-time Prediction':
    st.markdown("## 🔮 Real-time Prediction")
    
    # Son verileri göster
    try:
        db_manager = DatabaseManager()
        latest_data = db_manager.get_latest_data(10)
        
        if not latest_data.empty:
            st.markdown("### 📊 Son Veriler")
            st.dataframe(latest_data, use_container_width=True)
            
            # Tahmin yap
            if st.button("🔮 Tahmin Yap", type="primary"):
                try:
                    # Model yükle
                    model = st.session_state.model_manager.load_trained_model(selected_model)
                    
                    # Son verileri al
                    all_data = db_manager.get_all_data()
                    
                    if len(all_data) < 1000:
                        st.error("Yeterli veri yok (en az 1000 veri gerekli)")
                        st.stop()
                    
                    # Feature extraction
                    hist = all_data['value'].iloc[-1000:].values
                    
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
                    
                    X = np.array(features).reshape(1, -1)
                    
                    # Tahmin yap
                    prediction = model.predict(X)[0]
                    
                    if hasattr(model, 'predict_proba'):
                        probability = model.predict_proba(X)[0]
                    else:
                        probability = None
                    
                    # Sonuçları göster
                    st.markdown("### 🎯 Tahmin Sonuçları")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if model_info['mode'] == 'classification':
                            result = "1.5x Üstü" if prediction == 1 else "1.5x Altı"
                            st.metric("Tahmin", result)
                        elif model_info['mode'] == 'multiclass':
                            categories = ['Düşük (< 1.5x)', 'Orta (1.5x - 10x)', 'Yüksek (10x - 50x)', 'Mega (50x+)']
                            result = categories[prediction] if prediction < len(categories) else "Bilinmeyen"
                            st.metric("Tahmin", result)
                        else:  # regression
                            st.metric("Tahmin", f"{prediction:.2f}x")
                    
                    with col2:
                        if probability is not None:
                            if model_info['mode'] == 'classification':
                                confidence = max(probability, 1 - probability)
                                st.metric("Güven", f"{confidence:.2%}")
                            elif model_info['mode'] == 'multiclass':
                                confidence = np.max(probability)
                                st.metric("Güven", f"{confidence:.2%}")
                    
                    with col3:
                        # Son gerçek değer
                        last_value = all_data['value'].iloc[-1]
                        st.metric("Son Değer", f"{last_value:.2f}x")
                    
                    # Probability distribution (if available)
                    if probability is not None and model_info['mode'] == 'classification':
                        st.markdown("### 📊 Probability Distribution")
                        
                        fig_prob = go.Figure(data=[
                            go.Bar(x=['1.5x Altı', '1.5x Üstü'], y=probability, 
                                   marker_color=['red', 'green'])
                        ])
                        fig_prob.update_layout(
                            title="Prediction Probabilities",
                            yaxis_title="Probability",
                            height=300
                        )
                        st.plotly_chart(fig_prob, use_container_width=True)
                    
                    # Tahmin geçmişi kaydet
                    prediction_history = {
                        'timestamp': datetime.now().isoformat(),
                        'model_id': selected_model,
                        'prediction': prediction,
                        'probability': probability.tolist() if probability is not None else None,
                        'last_value': last_value,
                        'features': features
                    }
                    
                    st.session_state.prediction_results = prediction_history
                    
                except Exception as e:
                    st.error(f"❌ Tahmin hatası: {str(e)}")
        else:
            st.error("Veri bulunamadı!")
            
    except Exception as e:
        st.error(f"❌ Veri yükleme hatası: {str(e)}")

elif prediction_mode == 'Historical Backtesting':
    st.markdown("## 📈 Historical Backtesting")
    
    # Backtesting parametreleri
    col1, col2 = st.columns(2)
    
    with col1:
        backtest_period = st.selectbox(
            "Backtest Period",
            ['Last 100', 'Last 500', 'Last 1000', 'Last 2000', 'Custom'],
            help="Backtest periyodu"
        )
        
        if backtest_period == 'Custom':
            custom_period = st.number_input("Custom Period", 100, 5000, 1000)
        else:
            custom_period = int(backtest_period.split()[-1])
    
    with col2:
        prediction_threshold = st.number_input(
            "Prediction Threshold",
            0.0, 1.0, 0.5, 0.1,
            help="Classification için threshold"
        )
        
        bet_amount = st.number_input(
            "Bet Amount",
            1, 100, 10,
            help="Sanal bahis miktarı"
        )
    
    # Backtesting başlat
    if st.button("📈 Backtesting Başlat", type="primary"):
        try:
            # Model yükle
            model = st.session_state.model_manager.load_trained_model(selected_model)
            
            # Veri yükle
            db_manager = DatabaseManager()
            all_data = db_manager.get_all_data()
            
            if len(all_data) < custom_period + 1000:
                st.error(f"Yeterli veri yok (en az {custom_period + 1000} veri gerekli)")
                st.stop()
            
            # Backtest verisi hazırla
            window_size = 1000
            start_idx = len(all_data) - custom_period
            
            predictions = []
            actuals = []
            probabilities = []
            timestamps = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(start_idx, len(all_data)):
                progress = (i - start_idx) / (len(all_data) - start_idx)
                progress_bar.progress(progress)
                status_text.text(f"Backtesting: {i - start_idx + 1}/{len(all_data) - start_idx}")
                
                # Historical data
                hist = all_data['value'].iloc[:i].values
                actual = all_data['value'].iloc[i]
                
                # Feature extraction
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
                
                X = np.array(features).reshape(1, -1)
                
                # Prediction
                pred = model.predict(X)[0]
                
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X)[0]
                else:
                    prob = None
                
                predictions.append(pred)
                actuals.append(actual)
                probabilities.append(prob)
                timestamps.append(i)
            
            progress_bar.progress(1.0)
            status_text.text("Backtesting tamamlandı!")
            
            # Backtest sonuçları
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            # Metrics hesapla
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            if model_info['mode'] == 'classification':
                # Binary classification metrics
                accuracy = accuracy_score(actuals >= 1.5, predictions)
                precision = precision_score(actuals >= 1.5, predictions)
                recall = recall_score(actuals >= 1.5, predictions)
                f1 = f1_score(actuals >= 1.5, predictions)
                
                # Virtual bankroll simulation
                initial_wallet = 10000
                wallet = initial_wallet
                bets_made = 0
                wins = 0
                
                for pred, actual in zip(predictions, actuals):
                    if pred == 1:  # Model 1.5 üstü dedi
                        bets_made += 1
                        wallet -= bet_amount
                        if actual >= 1.5:
                            wallet += bet_amount * 1.5
                            wins += 1
                
                roi = ((wallet - initial_wallet) / initial_wallet) * 100
                win_rate = (wins / bets_made * 100) if bets_made > 0 else 0
                
            else:
                # Regression metrics
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                
                accuracy = None
                precision = None
                recall = None
                f1 = None
                
                mae = mean_absolute_error(actuals, predictions)
                mse = mean_squared_error(actuals, predictions)
                rmse = np.sqrt(mse)
                r2 = r2_score(actuals, predictions)
                
                roi = None
                win_rate = None
            
            # Sonuçları kaydet
            backtest_results = {
                'model_id': selected_model,
                'period': custom_period,
                'predictions': predictions.tolist(),
                'actuals': actuals.tolist(),
                'probabilities': [p.tolist() if p is not None else None for p in probabilities],
                'timestamps': timestamps,
                'metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'mae': mae if 'mae' in locals() else None,
                    'mse': mse if 'mse' in locals() else None,
                    'rmse': rmse if 'rmse' in locals() else None,
                    'r2': r2 if 'r2' in locals() else None,
                    'virtual_bankroll_roi': roi,
                    'win_rate': win_rate,
                    'bets_made': bets_made if 'bets_made' in locals() else 0,
                    'wins': wins if 'wins' in locals() else 0
                }
            }
            
            st.session_state.backtest_results = backtest_results
            
            # Sonuçları göster
            st.markdown("## 📊 Backtest Sonuçları")
            
            col1, col2, col3, col4 = st.columns(4)
            
            if accuracy is not None:
                with col1:
                    st.metric("Accuracy", f"{accuracy:.4f}")
                with col2:
                    st.metric("Precision", f"{precision:.4f}")
                with col3:
                    st.metric("Recall", f"{recall:.4f}")
                with col4:
                    st.metric("F1 Score", f"{f1:.4f}")
            else:
                with col1:
                    st.metric("MAE", f"{mae:.4f}")
                with col2:
                    st.metric("MSE", f"{mse:.4f}")
                with col3:
                    st.metric("RMSE", f"{rmse:.4f}")
                with col4:
                    st.metric("R²", f"{r2:.4f}")
            
            if roi is not None:
                st.metric("Virtual Bankroll ROI", f"{roi:.2f}%")
                st.metric("Win Rate", f"{win_rate:.2f}%")
                st.metric("Bets Made", bets_made)
            
        except Exception as e:
            st.error(f"❌ Backtesting hatası: {str(e)}")

# Backtest sonuçları göster
if st.session_state.backtest_results is not None:
    results = st.session_state.backtest_results
    
    st.markdown("## 📈 Backtest Görselleştirmeleri")
    
    # Prediction vs Actual
    st.markdown("### 🎯 Prediction vs Actual")
    
    if model_info['mode'] == 'classification':
        # Classification: scatter plot
        fig_scatter = go.Figure()
        
        fig_scatter.add_trace(go.Scatter(
            x=results['actuals'],
            y=results['predictions'],
            mode='markers',
            name='Predictions',
            marker=dict(color='blue', size=6)
        ))
        
        # Add threshold line
        fig_scatter.add_hline(y=0.5, line_dash="dash", line_color="red", 
                             annotation_text="Threshold")
        fig_scatter.add_vline(x=1.5, line_dash="dash", line_color="red", 
                             annotation_text="1.5x")
        
        fig_scatter.update_layout(
            title="Prediction vs Actual Values",
            xaxis_title="Actual Values",
            yaxis_title="Predictions",
            height=400
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    else:
        # Regression: line plot
        fig_line = go.Figure()
        
        fig_line.add_trace(go.Scatter(
            x=list(range(len(results['actuals']))),
            y=results['actuals'],
            mode='lines',
            name='Actual',
            line=dict(color='blue')
        ))
        
        fig_line.add_trace(go.Scatter(
            x=list(range(len(results['predictions']))),
            y=results['predictions'],
            mode='lines',
            name='Predicted',
            line=dict(color='red')
        ))
        
        fig_line.update_layout(
            title="Actual vs Predicted Values Over Time",
            xaxis_title="Time Steps",
            yaxis_title="Values",
            height=400
        )
        
        st.plotly_chart(fig_line, use_container_width=True)
    
    # Performance over time
    st.markdown("### 📊 Performance Over Time")
    
    if model_info['mode'] == 'classification':
        # Rolling accuracy
        window_size = 50
        rolling_accuracy = []
        
        for i in range(window_size, len(results['predictions'])):
            window_preds = results['predictions'][i-window_size:i]
            window_actuals = results['actuals'][i-window_size:i]
            
            acc = accuracy_score(window_actuals >= 1.5, window_preds)
            rolling_accuracy.append(acc)
        
        fig_rolling = go.Figure()
        fig_rolling.add_trace(go.Scatter(
            x=list(range(window_size, len(results['predictions']))),
            y=rolling_accuracy,
            mode='lines',
            name='Rolling Accuracy',
            line=dict(color='green')
        ))
        
        fig_rolling.update_layout(
            title=f"Rolling Accuracy (Window Size: {window_size})",
            xaxis_title="Time Steps",
            yaxis_title="Accuracy",
            height=400
        )
        
        st.plotly_chart(fig_rolling, use_container_width=True)
    
    # Confusion Matrix (for classification)
    if model_info['mode'] == 'classification':
        st.markdown("### 🔍 Confusion Matrix")
        
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(results['actuals'] >= 1.5, results['predictions'])
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted < 1.5x', 'Predicted >= 1.5x'],
            y=['Actual < 1.5x', 'Actual >= 1.5x'],
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16}
        ))
        
        fig_cm.update_layout(
            title="Confusion Matrix",
            height=400
        )
        
        st.plotly_chart(fig_cm, use_container_width=True)
    
    # Export sonuçları
    st.markdown("## 💾 Sonuçları Export Et")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV export
        export_df = pd.DataFrame({
            'timestamp': results['timestamps'],
            'actual': results['actuals'],
            'prediction': results['predictions']
        })
        
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📊 CSV olarak indir",
            data=csv,
            file_name=f"backtest_results_{selected_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # JSON export
        json_data = json.dumps(results, indent=2)
        st.download_button(
            label="📄 JSON olarak indir",
            data=json_data,
            file_name=f"backtest_results_{selected_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎲 Prediction & Backtesting - CPU Lightweight Models</p>
</div>
""", unsafe_allow_html=True)
