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
import logging
import re
import sqlite3

# Utils modüllerini import et
from utils.database import DatabaseManager
from utils.predictor import JetXPredictor
from utils.risk_manager import RiskManager
from utils.config_loader import config
from category_definitions import CategoryDefinitions

# Yeni sistemleri import et
try:
    from utils.ensemble_predictor import create_ensemble_predictor, VotingStrategy
    from utils.adaptive_threshold import create_threshold_manager
    from utils.backtesting import create_backtest_engine
    from utils.all_models_predictor import AllModelsPredictor
    from utils.model_versioning import get_version_manager
    from utils.ab_testing import get_ab_test_manager
    from utils.model_loader import get_model_loader
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Gelişmiş özellikler yüklenemedi: {e}")
    ADVANCED_FEATURES_AVAILABLE = False

# Logging ayarla (Model kontrolünden ÖNCE)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.get('logging.file', 'data/app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Model varlık kontrolü fonksiyonu
def check_model_files():
    """Gerekli model dosyalarının varlığını kontrol eder"""
    model_files = {
        'Neural Network Model': config.get('model.path', 'models/jetx_model.h5'),
        'Scaler': config.get('model.scaler_path', 'models/scaler.pkl'),
        'CatBoost Regressor': 'models/catboost_regressor.cbm',
        'CatBoost Classifier': 'models/catboost_classifier.cbm',
        'CatBoost Scaler': 'models/catboost_scaler.pkl'
    }
    
    missing_files = []
    for name, path in model_files.items():
        if not os.path.exists(path):
            missing_files.append((name, path))
    
    return missing_files

# Modelleri kontrol et (logger tanımlandıktan SONRA)
MISSING_MODEL_FILES = check_model_files()
if MISSING_MODEL_FILES:
    logger.warning("=" * 70)
    logger.warning("EKSIK MODEL DOSYALARI TESPİT EDİLDİ!")
    logger.warning("=" * 70)
    for name, path in MISSING_MODEL_FILES:
        logger.warning(f"  ❌ {name}: {path}")
    logger.warning("")
    logger.warning("Bazı özellikler kullanılamayabilir.")
    logger.warning("Modelleri eğitmek için notebooks/ klasöründeki Colab notebook'larını kullanın.")
    logger.warning("=" * 70)

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
    .warning-zone {
        background: linear-gradient(135deg, #ff9800 0%, #ffc107 100%);
    }
    .warning-box {
        padding: 10px;
        border-left: 5px solid #ff9800;
        background-color: #fff3e0;
        margin: 10px 0;
    }
    .info-box {
        padding: 10px;
        border-left: 5px solid #2196F3;
        background-color: #e3f2fd;
        margin: 10px 0;
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'db_manager' not in st.session_state:
    db_path = config.get('database.path', 'data/jetx_data.db')
    st.session_state.db_manager = DatabaseManager(db_path)
    logger.info(f"Database manager başlatıldı: {db_path}")

if 'predictor' not in st.session_state:
    model_path = config.get('model.path', 'models/jetx_model.h5')
    scaler_path = config.get('model.scaler_path', 'models/scaler.pkl')
    st.session_state.predictor = JetXPredictor(model_path, scaler_path)
    logger.info(f"Predictor başlatıldı: {model_path}")

if 'risk_manager' not in st.session_state:
    default_mode = config.get('prediction.default_mode', 'normal')
    st.session_state.risk_manager = RiskManager(mode=default_mode)
    logger.info(f"Risk manager başlatıldı: {default_mode} mod")

if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

# Yeni sistemleri session state'e ekle
if 'use_ensemble' not in st.session_state:
    st.session_state.use_ensemble = False

if 'use_dynamic_threshold' not in st.session_state:
    st.session_state.use_dynamic_threshold = False

if 'voting_strategy' not in st.session_state:
    st.session_state.voting_strategy = 'weighted'

if 'threshold_strategy' not in st.session_state:
    st.session_state.threshold_strategy = 'hybrid'

# All Models Predictor
if 'all_models_predictor' not in st.session_state and ADVANCED_FEATURES_AVAILABLE:
    try:
        st.session_state.all_models_predictor = AllModelsPredictor()
        st.session_state.all_models_predictor.load_all_models()
        logger.info("All Models Predictor başlatıldı")
    except Exception as e:
        logger.warning(f"All Models Predictor yüklenemedi: {e}")
        st.session_state.all_models_predictor = None

# RL Agent
if 'rl_agent' not in st.session_state and ADVANCED_FEATURES_AVAILABLE:
    try:
        from utils.rl_agent import create_rl_agent
        st.session_state.rl_agent = create_rl_agent(model_path='models/rl_agent_model.h5')
        if st.session_state.rl_agent.model is None:
            logger.warning("RL Agent model yüklenemedi, RL özellikleri kullanılamayacak")
        else:
            logger.info("RL Agent başlatıldı")
    except Exception as e:
        logger.warning(f"RL Agent yüklenemedi: {e}")
        st.session_state.rl_agent = None

# Model Version Manager
if 'version_manager' not in st.session_state and ADVANCED_FEATURES_AVAILABLE:
    try:
        st.session_state.version_manager = get_version_manager()
    except Exception as e:
        logger.warning(f"Version Manager yüklenemedi: {e}")
        st.session_state.version_manager = None

# AB Test Manager
if 'ab_test_manager' not in st.session_state and ADVANCED_FEATURES_AVAILABLE:
    try:
        st.session_state.ab_test_manager = get_ab_test_manager()
    except Exception as e:
        logger.warning(f"AB Test Manager yüklenemedi: {e}")
        st.session_state.ab_test_manager = None

# Model Loader
if 'model_loader' not in st.session_state and ADVANCED_FEATURES_AVAILABLE:
    try:
        st.session_state.model_loader = get_model_loader()
    except Exception as e:
        logger.warning(f"Model Loader yüklenemedi: {e}")
        st.session_state.model_loader = None

# Sidebar
with st.sidebar:
    st.title("🎮 Kontrol Paneli")
    
    # Model durumu kontrolü (gelişmiş)
    if st.session_state.model_loader:
        model_summary = st.session_state.model_loader.get_model_summary()
        
        if model_summary['incomplete_models'] > 0:
            st.warning(f"⚠️ {model_summary['incomplete_models']} model eksik veya tamamlanmamış")
            with st.expander("📋 Model Durumu", expanded=False):
                for model_name, model_info in model_summary['models'].items():
                    status_icon = "✅" if model_info['complete'] else "❌"
                    st.write(f"{status_icon} **{model_name}**")
                    if not model_info['complete']:
                        st.caption(f"Eksik: {model_info['files_missing']} dosya")
                        if model_info['missing_files']:
                            st.code("\n".join(model_info['missing_files'][:3]), language="text")
        else:
            st.success(f"✅ Tüm modeller yüklü ({model_summary['complete_models']})")
        
        # Model kurulum rehberi
        if st.button("📖 Kurulum Rehberi"):
            guide = st.session_state.model_loader.get_installation_guide()
            st.code(guide, language="text")
        
        st.divider()
    else:
        # Eski sistem (geriye dönük uyumluluk)
        if MISSING_MODEL_FILES:
            st.error(f"⚠️ {len(MISSING_MODEL_FILES)} model dosyası eksik!")
            with st.expander("📋 Eksik Dosyalar"):
                for name, path in MISSING_MODEL_FILES:
                    st.write(f"❌ **{name}**")
                    st.code(path, language="text")
            st.divider()
    
    # Gelişmiş Özellikler
    if ADVANCED_FEATURES_AVAILABLE:
        st.subheader("🚀 Gelişmiş Özellikler")
        
        use_ensemble = st.checkbox(
            "🔗 Ensemble Predictor",
            value=st.session_state.use_ensemble,
            help="Birden fazla modeli birleştirerek daha güvenilir tahminler"
        )
        st.session_state.use_ensemble = use_ensemble
        
        if use_ensemble:
            voting_strategy = st.selectbox(
                "Oylama Stratejisi:",
                options=['weighted', 'unanimous', 'confidence', 'majority'],
                index=['weighted', 'unanimous', 'confidence', 'majority'].index(st.session_state.voting_strategy),
                format_func=lambda x: {
                    'weighted': '⚖️ Ağırlıklı (Önerilen)',
                    'unanimous': '🤝 Oybirliği',
                    'confidence': '🎯 Güven Bazlı',
                    'majority': '📊 Çoğunluk'
                }[x],
                help="Weighted: CatBoost %60, NN %40\nUnanimous: Her iki model de aynı tahminde\nConfidence: En güvenli modele öncelik\nMajority: Basit çoğunluk"
            )
            st.session_state.voting_strategy = voting_strategy
        
        use_dynamic_threshold = st.checkbox(
            "🎚️ Dinamik Threshold",
            value=st.session_state.use_dynamic_threshold,
            help="Güven skoruna göre threshold otomatik ayarlama"
        )
        st.session_state.use_dynamic_threshold = use_dynamic_threshold
        
        if use_dynamic_threshold:
            threshold_strategy = st.selectbox(
                "Threshold Stratejisi:",
                options=['hybrid', 'confidence', 'performance'],
                index=['hybrid', 'confidence', 'performance'].index(st.session_state.threshold_strategy),
                format_func=lambda x: {
                    'hybrid': '🔄 Hibrit (Önerilen)',
                    'confidence': '🎯 Güven Bazlı',
                    'performance': '📈 Performans Bazlı'
                }[x],
                help="Hybrid: Güven + Performans\nConfidence: Sadece güven skoru\nPerformance: Geçmiş performans"
            )
            st.session_state.threshold_strategy = threshold_strategy
        
        st.divider()
        
        # Model Versiyonlama
        if st.session_state.version_manager:
            st.subheader("📦 Model Versiyonlama")
            
            model_names = st.session_state.version_manager.list_all_models()
            if model_names:
                selected_model = st.selectbox("Model Seç:", model_names)
                
                versions = st.session_state.version_manager.get_all_versions(selected_model)
                if versions:
                    version_info = st.selectbox(
                        "Versiyon Seç:",
                        options=[v['version'] for v in versions],
                        format_func=lambda v: f"v{v} {'(Production)' if versions[0].get('is_production', False) else ''}"
                    )
                    
                    if st.button("📊 Versiyon Detayları"):
                        info = st.session_state.version_manager.get_model_info(selected_model, version_info)
                        if info:
                            with st.expander("📋 Versiyon Bilgileri", expanded=True):
                                st.json({
                                    'Model ID': info['model_id'],
                                    'Versiyon': info['version'],
                                    'Tip': info['model_type'],
                                    'Production': info['is_production'],
                                    'Oluşturulma': info['created_at'],
                                    'Metrikler': info.get('metrics', {}),
                                    'Metadata': info.get('metadata', {})
                                })
            else:
                st.info("Henüz kayıtlı model versiyonu yok")
        
        st.divider()
        
        # A/B Testing
        if st.session_state.ab_test_manager:
            st.subheader("🧪 A/B Testing")
            
            active_tests = st.session_state.ab_test_manager.get_active_tests()
            if active_tests:
                st.write(f"**Aktif Testler:** {len(active_tests)}")
                for test in active_tests[:3]:  # İlk 3'ü göster
                    with st.expander(f"📊 {test['test_name']} ({test['test_id'][:8]}...)"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Model A", test['model_a'])
                            st.metric("Accuracy A", f"{test['accuracy_a']:.1f}%")
                            st.metric("ROI A", f"{test['roi_a']:.2f}%")
                        with col2:
                            st.metric("Model B", test['model_b'])
                            st.metric("Accuracy B", f"{test['accuracy_b']:.1f}%")
                            st.metric("ROI B", f"{test['roi_b']:.2f}%")
                        
                        if test['winner']:
                            st.success(f"🏆 Kazanan: Model {test['winner']}")
                        else:
                            st.info("⚖️ Henüz kazanan belli değil")
                        
                        st.caption(f"Güven Seviyesi: {test['confidence_level']:.1f}%")
            else:
                st.info("Aktif A/B testi yok")
            
            if st.button("➕ Yeni A/B Testi Oluştur"):
                st.info("A/B testi oluşturmak için sayfayı yenileyin ve test ayarlarını yapın")
        
        st.divider()
        
        # Model Durumu ve Sanal Kasa Bilgisi
        st.subheader("💰 Sanal Kasa Sistemi")
        
        with st.expander("📊 Sanal Kasa Sistemleri Hakkında", expanded=False):
            st.markdown("""
            **3 Farklı Sanal Kasa Sistemi:**
            
            1. **VirtualBankrollCallback** (Eğitim Sırasında)
               - Her epoch'ta performans ölçümü
               - Kasa 1: 1.5x eşik sistemi
               - Kasa 2: %70 çıkış sistemi
            
            2. **DualBankrollSystem** (Test/Değerlendirme)
               - Güven skoru filtresi ile
               - Dinamik kasa miktarı
               - Detaylı raporlama
            
            3. **AdvancedBankrollManager** (Production)
               - Kelly Criterion (optimal bahis)
               - Risk stratejileri (normal, rolling)
               - Stop-loss & Take-profit
               - Streak tracking
            
            **Detaylı bilgi için:** `docs/WORKFLOW_AND_SYSTEMS.md`
            """)
        
        st.divider()
    
    # Mod seçimi
    st.subheader("📊 Tahmin Modu")
    mode = st.selectbox(
        "Mod seçin:",
        options=['normal', 'rolling'],  # Sadece 2 seçenek kaldı
        format_func=lambda x: {
            'normal': '🎯 Normal (%85+ Güven)',
            'rolling': '🛡️ Rolling / Kasa Katlama (%95+ Güven)'
        }[x],
        help="Normal: %85 üzeri güven, Rolling: %95 üzeri güven (Çok güvenli)"
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

# Sistem durumu banner
if ADVANCED_FEATURES_AVAILABLE:
    features_active = []
    if st.session_state.use_ensemble:
        features_active.append(f"Ensemble ({st.session_state.voting_strategy})")
    if st.session_state.use_dynamic_threshold:
        features_active.append(f"Dynamic Threshold ({st.session_state.threshold_strategy})")
    
    if features_active:
        st.success(f"✨ Aktif Özellikler: {', '.join(features_active)}")
    else:
        st.info("💡 Gelişmiş özellikler mevcut ama aktif değil. Sol menüden aktifleştirebilirsiniz.")
else:
    st.warning("⚠️ Gelişmiş özellikler henüz yüklenmedi. Modeller eksik olabilir.")

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
                # Tüm modellerden tahmin al (RL Agent için gerekli)
                all_predictions = None
                if st.session_state.all_models_predictor:
                    try:
                        history_array = np.array(history)
                        all_predictions = st.session_state.all_models_predictor.predict_all(history_array)
                    except Exception as e:
                        logger.error(f"All models prediction hatası: {e}")
                        all_predictions = None
                
                # Ana tahmin yap (geriye dönük uyumluluk için)
                prediction = st.session_state.predictor.predict(history, mode=mode)
                st.session_state.last_prediction = prediction
                
                # Risk analizi
                risk_decision = st.session_state.risk_manager.should_play(prediction)
                
                # RL Agent kullan (eğer yüklüyse)
                rl_action = None
                rl_interpretation = None
                
                if st.session_state.rl_agent and st.session_state.rl_agent.model is not None and all_predictions:
                    try:
                        # State vector oluştur
                        state_vector = st.session_state.rl_agent.create_state_vector(
                            history=history,
                            model_predictions=all_predictions,
                            risk_analysis=risk_decision
                        )
                        
                        # Action tahmin et
                        action, probabilities = st.session_state.rl_agent.predict_action(state_vector)
                        
                        # Action'ı interpret et
                        rl_interpretation = st.session_state.rl_agent.interpret_action(
                            action=action,
                            probabilities=probabilities,
                            model_predictions=all_predictions,
                            bankroll=None  # Bankroll manager yoksa None
                        )
                        
                        rl_action = rl_interpretation
                        logger.info(f"RL Agent action: {action} ({rl_interpretation['action_name']})")
                    except Exception as e:
                        logger.error(f"RL Agent hatası: {e}")
                        rl_action = None
                
                # Tahmini göster
                if 'error' in prediction:
                    st.error(f"❌ Hata: {prediction['error']}")
                else:
                    # RL Agent varsa, Ana Eylem Kartı göster
                    if rl_action:
                        # Ana Eylem Kartı (RL Agent kararı)
                        action_name = rl_action['action_name']
                        should_bet = rl_action['should_bet']
                        
                        if should_bet:
                            card_class = "safe-zone"
                            action_emoji = "📈"
                            action_text = "BAHİS YAP"
                        else:
                            card_class = "danger-zone"
                            action_emoji = "⛔"
                            action_text = "BEKLE (BAHİS YAPMA)"
                        
                        # Kart içeriği
                        card_content = f"""
                        <div class="prediction-card {card_class}">
                            <h2>🤖 AJAN AKSİYONU: {action_emoji} {action_text}</h2>
                        """
                        
                        if should_bet:
                            if rl_action.get('bet_amount'):
                                card_content += f'<p><strong>Bahis Miktarı:</strong> {rl_action["bet_amount"]:.2f} TL'
                                if rl_action.get('bet_percentage'):
                                    card_content += f' (Kasanızın %{rl_action["bet_percentage"]:.1f}\'ü)'
                                card_content += '</p>'
                            
                            if rl_action.get('exit_multiplier'):
                                card_content += f'<p><strong>Çıkış Noktası (Cash-out):</strong> {rl_action["exit_multiplier"]:.2f}x</p>'
                        else:
                            card_content += f'<p><strong>Risk Seviyesi:</strong> {rl_action.get("risk_level", "Yüksek")}</p>'
                            if rl_action.get('reasoning'):
                                card_content += f'<p><strong>Gerekçe:</strong> {rl_action["reasoning"][0] if rl_action["reasoning"] else "Modeller arası fikir ayrılığı ve yüksek tuzak riski."}</p>'
                        
                        card_content += f'<p><strong>Güven Skoru:</strong> {rl_action["confidence"]:.0%}</p>'
                        card_content += '</div>'
                        
                        st.markdown(card_content, unsafe_allow_html=True)
                        
                        # Detaylı Bilgiler (Expander)
                        with st.expander("▼ Ajanın Değerlendirmesi: Detaylar için Tıkla", expanded=False):
                            # 1. Tahmin Modeli Çıktıları
                            st.subheader("1. Tahmin Modeli Çıktıları")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if all_predictions:
                                    if 'progressive_nn' in all_predictions and all_predictions['progressive_nn']:
                                        pn = all_predictions['progressive_nn']
                                        st.metric("Progressive NN Tahmini", f"{pn.get('prediction', 0):.2f}x")
                                        st.metric("Progressive NN Güven", f"{pn.get('confidence', 0):.0%}")
                                    
                                    if 'catboost' in all_predictions and all_predictions['catboost']:
                                        cb = all_predictions['catboost']
                                        st.metric("CatBoost Tahmini", f"{cb.get('prediction', 0):.2f}x")
                                        st.metric("CatBoost Güven", f"{cb.get('confidence', 0):.0%}")
                            
                            with col2:
                                if all_predictions:
                                    if 'tabnet' in all_predictions and all_predictions['tabnet']:
                                        tn = all_predictions['tabnet']
                                        st.metric("TabNet (Yüksek X) Riski", f"{100 - tn.get('confidence', 0)*100:.0f}%")
                                    
                                    if 'consensus' in all_predictions and all_predictions['consensus']:
                                        cs = all_predictions['consensus']
                                        st.metric("Modeller Arası Güven", f"{cs.get('agreement', 0):.0%}")
                                        st.metric("Toplam Model Sayısı", cs.get('total_models', 0))
                            
                            # 2. Risk ve Desen Analizi
                            st.subheader("2. Risk ve Desen Analizi")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Psikolojik analiz
                                try:
                                    from utils.psychological_analyzer import PsychologicalAnalyzer
                                    psych_analyzer = PsychologicalAnalyzer(threshold=1.5)
                                    psych_features = psych_analyzer.analyze_psychological_patterns(history)
                                    manipulation_score = psych_features.get('manipulation_score', 0)
                                    st.metric("Psikolojik Tuzak Riski", f"{manipulation_score*100:.0f}% {'(Düşük)' if manipulation_score < 0.3 else '(Orta)' if manipulation_score < 0.7 else '(Yüksek)'}")
                                except:
                                    st.metric("Psikolojik Tuzak Riski", "N/A")
                                
                                # Anomali tespit
                                try:
                                    from utils.anomaly_streak_detector import AnomalyStreakDetector
                                    anomaly_detector = AnomalyStreakDetector(threshold=1.5)
                                    anomaly_features = anomaly_detector.extract_streak_features(history)
                                    extreme_risk = anomaly_features.get('extreme_streak_risk', 0)
                                    st.metric("Anormal Seri Riski", f"{extreme_risk*100:.0f}% {'(Normal)' if extreme_risk < 0.5 else '(Yüksek)'}")
                                except:
                                    st.metric("Anormal Seri Riski", "N/A")
                            
                            with col2:
                                # Mevcut seri
                                if len(history) >= 10:
                                    recent_10 = history[-10:]
                                    below_count = sum(1 for v in recent_10 if v < 1.5)
                                    st.metric("Mevcut Seri (1.5 Altı)", below_count)
                                
                                # Volatilite
                                if len(history) >= 20:
                                    recent_20 = history[-20:]
                                    volatility = np.std(recent_20) / (np.mean(recent_20) + 1e-8)
                                    vol_level = "Düşük" if volatility < 0.3 else "Orta" if volatility < 0.6 else "Yüksek"
                                    st.metric("Volatilite", vol_level)
                            
                            # 3. Finansal Strateji Analizi
                            st.subheader("3. Finansal Strateji Analizi")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Kelly Criterion
                                if all_predictions and 'consensus' in all_predictions and all_predictions['consensus']:
                                    cs = all_predictions['consensus']
                                    confidence = cs.get('confidence', 0.5)
                                    prediction = cs.get('prediction', 1.5)
                                    
                                    # Basit Kelly calculation
                                    win_prob = confidence
                                    win_multiplier = prediction - 1.0
                                    if win_multiplier > 0:
                                        kelly_frac = (win_prob * win_multiplier - (1 - win_prob)) / win_multiplier
                                        kelly_frac = max(0, min(kelly_frac, 0.25)) * 100
                                    else:
                                        kelly_frac = 0.0
                                    
                                    st.metric("Kelly Criterion Oranı (Optimal Bahis)", f"%{kelly_frac:.1f}")
                            
                            with col2:
                                st.metric("Risk Modu (RiskManager)", mode.upper())
                            
                            # RL Action probabilities
                            if rl_action:
                                st.subheader("4. RL Agent Action Probabilities")
                                prob_df = pd.DataFrame({
                                    'Action': ['BEKLE', 'Konservatif', 'Normal', 'Yüksek Risk'],
                                    'Probability': rl_action.get('probabilities', [0, 0, 0, 0])
                                })
                                st.bar_chart(prob_df.set_index('Action'))
                    
                    else:
                        # RL Agent yoksa, eski UI'ı göster (geriye dönük uyumluluk)
                        # Güven seviyesine göre kart rengi
                        confidence = prediction['confidence']
                        if confidence >= 0.8:
                            card_class = "safe-zone"
                        elif confidence >= 0.6:
                            card_class = "warning-zone"
                        else:
                            card_class = "danger-zone"
                        
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
                # Volatilite riskini hesapla ve göster
                manipulation_score = 0.0
                volatility_risk_level = "Düşük"
                
                try:
                    from utils.psychological_analyzer import PsychologicalAnalyzer
                    analyzer = PsychologicalAnalyzer(threshold=1.5)
                    history = st.session_state.db_manager.get_recent_results(100)
                    
                    if len(history) >= 20:
                        psych_features = analyzer.analyze_psychological_patterns(history)
                        manipulation_score = psych_features.get('manipulation_score', 0.0)
                        
                        if manipulation_score > 0.7:
                            volatility_risk_level = "Yüksek"
                        elif manipulation_score > 0.5:
                            volatility_risk_level = "Orta"
                        else:
                            volatility_risk_level = "Düşük"
                except Exception as e:
                    logger.error(f"Volatilite riski hesaplama hatası: {e}")
                    manipulation_score = 0.0
                    volatility_risk_level = "Hesaplanamadı"
                
                # Tahmin detaylarını göster
                col1, col2 = st.columns(2)
                
                with col1:
                    st.json({
                        'Tahmin': f"{pred['predicted_value']:.2f}x",
                        'Güven': f"{pred['confidence']:.0%}",
                        '1.5x Üstü': 'Evet' if pred['above_threshold'] else 'Hayır',
                        'Kategori': pred['category'],
                        'Mod': pred['mode'].upper()
                    })
                
                with col2:
                    # Volatilite riski bilgisi
                    st.subheader("🛡️ Volatilite Riski")
                    risk_color = "🔴" if manipulation_score > 0.7 else "🟡" if manipulation_score > 0.5 else "🟢"
                    st.metric(
                        "Risk Seviyesi", 
                        f"{volatility_risk_level}",
                        delta=f"{manipulation_score*100:.0f}%",
                        delta_color="inverse" if manipulation_score > 0.5 else "normal"
                    )
                    
                    # Volatilite riskine göre bahis önerisi
                    if manipulation_score > 0.7:
                        st.warning("⚠️ Yüksek volatilite riski! Bahis miktarı %80 küçültüldü.")
                    elif manipulation_score > 0.5:
                        st.info("⚠️ Orta volatilite riski! Bahis miktarı %50 küçültüldü.")
                    else:
                        st.success("✅ Düşük risk! Normal bahis miktarı uygulanıyor.")
                        
                    # Advanced Bankroll Manager ile örnek bahis hesaplama
                    try:
                        from utils.advanced_bankroll import AdvancedBankrollManager
                        bankroll_manager = AdvancedBankrollManager(
                            initial_bankroll=1000.0,
                            risk_tolerance=mode
                        )
                        
                        original_bet = bankroll_manager.calculate_bet_size(
                            confidence=pred['confidence'],
                            predicted_value=pred['predicted_value']
                        )
                        
                        adjusted_bet = bankroll_manager.calculate_bet_size(
                            confidence=pred['confidence'],
                            predicted_value=pred['predicted_value'],
                            volatility_risk=manipulation_score
                        )
                        
                        if original_bet > 0 and adjusted_bet > 0:
                            reduction = ((original_bet - adjusted_bet) / original_bet) * 100
                            st.write(f"**Örnek Bahis:** {adjusted_bet:.2f} TL (Orijinal: {original_bet:.2f} TL)")
                            if reduction > 0:
                                st.write(f"**Küçültme:** %{reduction:.0f}")
                    except Exception as e:
                        logger.error(f"Bahis hesaplama hatası: {e}")
                        st.write("Bahis hesaplama için bankroll manager gerekli.")

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

# Backtesting Bölümü (varsa)
if ADVANCED_FEATURES_AVAILABLE:
    with st.expander("🔬 Backtesting - Model Performans Testi", expanded=False):
        st.markdown("""
        Geçmiş veriler üzerinde model performansını test edin.
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            backtest_window = st.number_input("Test Veri Sayısı:", 50, 500, 200)
        with col2:
            starting_capital = st.number_input("Başlangıç Sermayesi:", 100, 10000, 1000)
        with col3:
            bet_size = st.number_input("Bahis Tutarı:", 1, 100, 10)
        
        if st.button("🧪 Backtest Çalıştır"):
            with st.spinner("Backtest çalıştırılıyor..."):
                from utils.backtesting import create_backtest_engine
                
                # Veriyi al
                test_data = st.session_state.db_manager.get_recent_results(backtest_window)
                
                if len(test_data) < 50:
                    st.error("En az 50 veri gerekli!")
                else:
                    # Tahminler yap
                    predictions = []
                    for i in range(50, len(test_data)):
                        history = test_data[max(0, i-500):i]
                        pred = st.session_state.predictor.predict(history, mode=mode)
                        if 'error' not in pred:
                            predictions.append(pred['predicted_value'])
                        else:
                            predictions.append(1.0)
                    
                    actuals = test_data[50:]
                    predictions = np.array(predictions[:len(actuals)])
                    
                    # Backtest engine
                    engine = create_backtest_engine(
                        starting_capital=starting_capital,
                        bet_size=bet_size,
                        strategy='fixed'
                    )
                    
                    result = engine.run(predictions, actuals)
                    
                    # Sonuçları göster
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        roi_color = "normal" if result.roi >= 0 else "inverse"
                        st.metric("ROI", f"{result.roi:.2f}%", delta_color=roi_color)
                    
                    with col2:
                        st.metric("Kazanma Oranı", f"{result.win_rate:.1%}")
                    
                    with col3:
                        st.metric("Net Kar/Zarar", f"{result.net_profit:+.2f} TL")
                    
                    with col4:
                        st.metric("Max Drawdown", f"{result.max_drawdown_pct:.1f}%")
                    
                    # Equity curve
                    if result.equity_curve:
                        fig_equity = go.Figure()
                        fig_equity.add_trace(go.Scatter(
                            y=result.equity_curve,
                            mode='lines',
                            name='Sermaye',
                            line=dict(color='#2196F3', width=2)
                        ))
                        fig_equity.add_hline(
                            y=starting_capital,
                            line_dash="dash",
                            line_color="gray",
                            annotation_text="Başlangıç"
                        )
                        fig_equity.update_layout(
                            title="Sermaye Değişimi",
                            xaxis_title="İşlem",
                            yaxis_title="Sermaye (TL)",
                            height=300
                        )
                        st.plotly_chart(fig_equity, use_container_width=True)
                    
                    # Detaylar
                    st.subheader("📊 Detaylı Sonuçlar")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Toplam Oyun:** {result.total_games}")
                        st.write(f"**Kazanan:** {result.wins}")
                        st.write(f"**Kaybeden:** {result.losses}")
                        st.write(f"**Atlanan:** {result.skipped}")
                    with col2:
                        st.write(f"**Sharpe Ratio:** {result.sharpe_ratio:.3f}")
                        st.write(f"**En Uzun Kazanma:** {result.max_win_streak}")
                        st.write(f"**En Uzun Kaybetme:** {result.max_loss_streak}")
                        st.write(f"**Ortalama Güven:** {result.avg_confidence:.1%}")

st.divider()

# Veri girişi bölümü
st.subheader("➕ Yeni Veri Ekle")

def validate_input_value(value: float) -> tuple[bool, str]:
    """
    Input değerini validate eder
    
    Args:
        value: Kontrol edilecek değer
        
    Returns:
        (is_valid, error_message) tuple'ı
    """
    # Değer aralığı kontrolü
    if value < 1.0:
        return False, "❌ Değer 1.0x'den küçük olamaz!"
    
    if value > 10000.0:
        return False, "❌ Değer 10000x'den büyük olamaz! Lütfen gerçekçi bir değer girin."
    
    # Ondalık basamak kontrolü (en fazla 2 basamak)
    if not re.match(r'^\d+(\.\d{1,2})?$', str(value)):
        return False, "❌ Değer en fazla 2 ondalık basamak içerebilir!"
    
    # Anomali kontrolü - aşırı yüksek değerler
    if value > 1000.0:
        logger.warning(f"Aşırı yüksek değer girildi: {value}x")
        return False, f"⚠️ {value:.2f}x çok yüksek bir değer! Gerçekten bu değeri girmek istiyor musunuz? Lütfen kontrol edin."
    
    # NaN veya Infinity kontrolü
    if not np.isfinite(value):
        return False, "❌ Geçersiz sayı! Lütfen geçerli bir değer girin."
    
    return True, ""

col1, col2 = st.columns([3, 1])
with col1:
    new_value = st.number_input(
        "Gerçekleşen değeri girin:",
        min_value=1.0,
        max_value=10000.0,
        value=1.5,
        step=0.01,
        format="%.2f",
        help="1.0x ile 1000x arası bir değer girin (en fazla 2 ondalık basamak)"
    )
with col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    if st.button("💾 Kaydet", use_container_width=True):
        # Input validation
        is_valid, error_message = validate_input_value(new_value)
        
        if is_valid:
            try:
                # Veritabanına ekle - Güçlendirilmiş error handling
                result_id = st.session_state.db_manager.add_result(new_value)
                
                if result_id > 0:
                    logger.info(f"Yeni değer kaydedildi: {new_value:.2f}x (ID: {result_id})")
                    st.success(f"✅ {new_value:.2f}x kaydedildi!")
                    
                    # Eğer bekleyen tahmin varsa, değerlendir
                    if st.session_state.last_prediction and 'error' not in st.session_state.last_prediction:
                        try:
                            evaluation = st.session_state.risk_manager.evaluate_prediction(
                                st.session_state.last_prediction,
                                new_value
                            )
                            
                            if evaluation['threshold_correct']:
                                st.success(f"🎉 Tahmin doğru! (1.5x eşik tahmini)")
                                logger.info(f"Doğru tahmin: {evaluation['consecutive_wins']} ardışık")
                            else:
                                st.error(f"❌ Tahmin yanlış!")
                                logger.warning(f"Yanlış tahmin: {evaluation['consecutive_losses']} ardışık")
                            
                            st.info(f"Ardışık: {evaluation['consecutive_wins']} doğru, {evaluation['consecutive_losses']} yanlış")
                        except Exception as e:
                            logger.error(f"Tahmin değerlendirme hatası: {e}", exc_info=True)
                            st.warning(f"⚠️ Tahmin değerlendirme hatası: {str(e)}")
                    
                    st.rerun()
                else:
                    logger.error(f"Veri kaydedilemedi: Veri kaydedilemedi! Lütfen tekrar deneyin.")
            except sqlite3.IntegrityError as e:
                logger.error(f"Veritabanı bütünlük hatası: {e}", exc_info=True)
                st.error(f"❌ Veritabanı bütünlük hatası: Aynı veri zaten mevcut olabilir.")
            except sqlite3.OperationalError as e:
                logger.error(f"Veritabanı işlem hatası: {e}", exc_info=True)
                st.error(f"❌ Veritabanı kilitli veya erişilemiyor. Lütfen tekrar deneyin.")
            except Exception as e:
                logger.error(f"Beklenmeyen veritabanı hatası: {e}", exc_info=True)
                st.error(f"❌ Beklenmeyen hata: {str(e)}")
        else:
            st.error(error_message)

st.divider()

# Alt bilgi
st.markdown("""
---
### ⚠️ ÖNEMLİ UYARILAR

- 🚨 **Bu sistem %100 doğru değildir**
- 💰 **Para kaybedebilirsiniz**
- 🎯 **1.5x kritik eşiktir**: Altı kayıp, üstü kazanç
- 🛡️ **Rolling modu** en güvenlidir (%95+ güven)

**Sorumlu oynayın!**
""")

# Footer
footer_text = f"JetX Predictor v2.0 | Mod: {mode.upper()}"
if ADVANCED_FEATURES_AVAILABLE:
    if st.session_state.use_ensemble:
        footer_text += f" | Ensemble: {st.session_state.voting_strategy}"
    if st.session_state.use_dynamic_threshold:
        footer_text += f" | Dynamic Threshold: {st.session_state.threshold_strategy}"
footer_text += f" | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

st.caption(footer_text)
