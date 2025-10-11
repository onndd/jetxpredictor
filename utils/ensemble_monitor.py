"""
JetX Predictor - Ensemble Performance Monitoring

Real-time performans takibi ve model karşılaştırması için monitoring sistemi.
Her model tahmininin doğruluğunu izler ve detaylı raporlar sunar.
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import os

# Logging ayarla
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleMonitor:
    """
    Ensemble ve individual model performanslarını izler
    
    Features:
    - Real-time accuracy tracking
    - Rolling window metrics (son N tahmin)
    - Confusion matrix tracking
    - Trend analysis
    - CSV export
    - Performance comparison
    """
    
    def __init__(self, db_path: str = "data/ensemble_performance.db"):
        """
        Args:
            db_path: Performance database dosya yolu
        """
        self.db_path = db_path
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Database ve tabloları oluştur"""
        # Dizini oluştur
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Predictions tablosu
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                
                -- Base model tahminleri
                progressive_pred REAL,
                progressive_threshold_prob REAL,
                progressive_above_threshold INTEGER,
                
                ultra_pred REAL,
                ultra_threshold_prob REAL,
                ultra_above_threshold INTEGER,
                
                xgboost_pred REAL,
                xgboost_threshold_prob REAL,
                xgboost_above_threshold INTEGER,
                
                -- Ensemble tahmini
                ensemble_pred REAL,
                ensemble_threshold_prob REAL,
                ensemble_above_threshold INTEGER,
                ensemble_method TEXT,
                
                -- Gerçek değer
                actual_value REAL,
                actual_above_threshold INTEGER,
                
                -- Doğruluk bilgileri (1.5 eşik bazında)
                progressive_correct INTEGER,
                ultra_correct INTEGER,
                xgboost_correct INTEGER,
                ensemble_correct INTEGER,
                
                -- Value prediction hatası (MAE)
                progressive_error REAL,
                ultra_error REAL,
                xgboost_error REAL,
                ensemble_error REAL
            )
        """)
        
        # Index oluştur (performans için)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON predictions(timestamp DESC)
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Performance database hazır: {self.db_path}")
    
    def log_prediction(
        self,
        individual_predictions: Dict,
        ensemble_prediction: Dict,
        actual_value: float
    ):
        """
        Tahmin ve gerçek değeri kaydet
        
        Args:
            individual_predictions: Her modelin tahmini {'progressive': {...}, 'ultra': {...}, 'xgboost': {...}}
            ensemble_prediction: Ensemble tahmini
            actual_value: Gerçekleşen değer
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Gerçek eşik değeri
        actual_above_threshold = 1 if actual_value >= 1.5 else 0
        
        # Her model için bilgileri çıkar
        prog = individual_predictions.get('progressive') or {}
        ultra = individual_predictions.get('ultra') or {}
        xgb = individual_predictions.get('xgboost') or {}
        
        # Doğruluk hesapla
        prog_correct = None
        if prog.get('above_threshold') is not None:
            prog_correct = 1 if (prog['above_threshold'] == (actual_value >= 1.5)) else 0
        
        ultra_correct = None
        if ultra.get('above_threshold') is not None:
            ultra_correct = 1 if (ultra['above_threshold'] == (actual_value >= 1.5)) else 0
        
        xgb_correct = None
        if xgb.get('above_threshold') is not None:
            xgb_correct = 1 if (xgb['above_threshold'] == (actual_value >= 1.5)) else 0
        
        ens_correct = None
        if ensemble_prediction.get('above_threshold') is not None:
            ens_correct = 1 if (ensemble_prediction['above_threshold'] == (actual_value >= 1.5)) else 0
        
        # Hata hesapla (MAE)
        prog_error = abs(prog.get('predicted_value', 0) - actual_value) if prog.get('predicted_value') else None
        ultra_error = abs(ultra.get('predicted_value', 0) - actual_value) if ultra.get('predicted_value') else None
        xgb_error = abs(xgb.get('predicted_value', 0) - actual_value) if xgb.get('predicted_value') else None
        ens_error = abs(ensemble_prediction.get('predicted_value', 0) - actual_value) if ensemble_prediction.get('predicted_value') else None
        
        # Kaydet
        cursor.execute("""
            INSERT INTO predictions (
                progressive_pred, progressive_threshold_prob, progressive_above_threshold,
                ultra_pred, ultra_threshold_prob, ultra_above_threshold,
                xgboost_pred, xgboost_threshold_prob, xgboost_above_threshold,
                ensemble_pred, ensemble_threshold_prob, ensemble_above_threshold, ensemble_method,
                actual_value, actual_above_threshold,
                progressive_correct, ultra_correct, xgboost_correct, ensemble_correct,
                progressive_error, ultra_error, xgboost_error, ensemble_error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prog.get('predicted_value'), prog.get('threshold_probability'), 
            1 if prog.get('above_threshold') else 0,
            
            ultra.get('predicted_value'), ultra.get('threshold_probability'),
            1 if ultra.get('above_threshold') else 0,
            
            xgb.get('predicted_value'), xgb.get('threshold_probability'),
            1 if xgb.get('above_threshold') else 0,
            
            ensemble_prediction.get('predicted_value'), ensemble_prediction.get('threshold_probability'),
            1 if ensemble_prediction.get('above_threshold') else 0,
            ensemble_prediction.get('ensemble_method', 'unknown'),
            
            actual_value, actual_above_threshold,
            
            prog_correct, ultra_correct, xgb_correct, ens_correct,
            
            prog_error, ultra_error, xgb_error, ens_error
        ))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Tahmin kaydedildi: Actual={actual_value:.2f}x, Ensemble={ensemble_prediction.get('predicted_value'):.2f}x")
    
    def get_rolling_accuracy(
        self, 
        model_name: str, 
        window: int = 100
    ) -> float:
        """
        Son N tahmindeki accuracy (1.5 eşik bazında)
        
        Args:
            model_name: 'progressive', 'ultra', 'xgboost', veya 'ensemble'
            window: Pencere boyutu
            
        Returns:
            Accuracy (0-1 arası)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = f"""
            SELECT AVG({model_name}_correct) 
            FROM (
                SELECT {model_name}_correct 
                FROM predictions 
                WHERE {model_name}_correct IS NOT NULL
                ORDER BY timestamp DESC 
                LIMIT ?
            )
        """
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result[0] is not None else 0.0
    
    def get_rolling_mae(
        self,
        model_name: str,
        window: int = 100
    ) -> float:
        """
        Son N tahmindeki MAE
        
        Args:
            model_name: 'progressive', 'ultra', 'xgboost', veya 'ensemble'
            window: Pencere boyutu
            
        Returns:
            MAE
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = f"""
            SELECT AVG({model_name}_error)
            FROM (
                SELECT {model_name}_error
                FROM predictions
                WHERE {model_name}_error IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT ?
            )
        """
        
        cursor.execute(query, (window,))
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result[0] is not None else 0.0
    
    def calculate_trend(
        self,
        model_name: str,
        window1: int = 50,
        window2: int = 100
    ) -> str:
        """
        Model performans trendi
        
        Args:
            model_name: Model adı
            window1: Yakın pencere (son 50)
            window2: Uzak pencere (son 100)
            
        Returns:
            Trend string: '↗️ improving', '↘️ declining', '➡️ stable'
        """
        recent = self.get_rolling_accuracy(model_name, window1)
        older = self.get_rolling_accuracy(model_name, window2)
        
        diff = recent - older
        
        if diff > 0.05:
            return '↗️ improving'
        elif diff < -0.05:
            return '↘️ declining'
        else:
            return '➡️ stable'
    
    def get_confusion_matrix(
        self,
        model_name: str,
        window: Optional[int] = None
    ) -> np.ndarray:
        """
        Confusion matrix (1.5 eşik bazında)
        
        Args:
            model_name: Model adı
            window: Pencere boyutu (None ise tümü)
            
        Returns:
            2x2 confusion matrix [[TN, FP], [FN, TP]]
        """
        conn = sqlite3.connect(self.db_path)
        
        if window:
            query = f"""
                SELECT actual_above_threshold, {model_name}_above_threshold
                FROM (
                    SELECT actual_above_threshold, {model_name}_above_threshold
                    FROM predictions
                    WHERE {model_name}_above_threshold IS NOT NULL
                    ORDER BY timestamp DESC
                    LIMIT ?
                )
            """
            df = pd.read_sql_query(query, conn, params=(window,))
        else:
            query = f"""
                SELECT actual_above_threshold, {model_name}_above_threshold
                FROM predictions
                WHERE {model_name}_above_threshold IS NOT NULL
            """
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        
        if len(df) == 0:
            return np.array([[0, 0], [0, 0]])
        
        # Confusion matrix hesapla
        tn = ((df['actual_above_threshold'] == 0) & (df[f'{model_name}_above_threshold'] == 0)).sum()
        fp = ((df['actual_above_threshold'] == 0) & (df[f'{model_name}_above_threshold'] == 1)).sum()
        fn = ((df['actual_above_threshold'] == 1) & (df[f'{model_name}_above_threshold'] == 0)).sum()
        tp = ((df['actual_above_threshold'] == 1) & (df[f'{model_name}_above_threshold'] == 1)).sum()
        
        return np.array([[tn, fp], [fn, tp]])
    
    def get_money_loss_risk(
        self,
        model_name: str,
        window: Optional[int] = None
    ) -> float:
        """
        Para kaybı riski (1.5 altıyken yanlışlıkla üstü deme oranı)
        
        Args:
            model_name: Model adı
            window: Pencere boyutu
            
        Returns:
            Risk oranı (0-1 arası)
        """
        cm = self.get_confusion_matrix(model_name, window)
        tn, fp = cm[0]
        
        total_below = tn + fp
        if total_below == 0:
            return 0.0
        
        return fp / total_below
    
    def generate_comparison_report(
        self,
        window: int = 100
    ) -> Dict:
        """
        Tüm modellerin karşılaştırmalı raporu
        
        Args:
            window: Pencere boyutu
            
        Returns:
            Model comparison dictionary
        """
        models = ['progressive', 'ultra', 'xgboost', 'ensemble']
        report = {}
        
        for model_name in models:
            accuracy = self.get_rolling_accuracy(model_name, window)
            mae = self.get_rolling_mae(model_name, window)
            trend = self.calculate_trend(model_name)
            money_risk = self.get_money_loss_risk(model_name, window)
            cm = self.get_confusion_matrix(model_name, window)
            
            # Below/Above threshold accuracy
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            below_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
            above_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            report[model_name] = {
                'accuracy': accuracy,
                'mae': mae,
                'below_1.5_accuracy': below_acc,
                'above_1.5_accuracy': above_acc,
                'money_loss_risk': money_risk,
                'trend': trend,
                'confusion_matrix': cm.tolist(),
                'total_predictions': int(cm.sum())
            }
        
        return report
    
    def get_chart_data(
        self,
        window: int = 500
    ) -> pd.DataFrame:
        """
        Grafik için time series data
        
        Args:
            window: Kaç tahmin
            
        Returns:
            DataFrame with columns: timestamp, progressive_correct, ultra_correct, xgboost_correct, ensemble_correct
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                timestamp,
                progressive_correct,
                ultra_correct,
                xgboost_correct,
                ensemble_correct
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(window,))
        conn.close()
        
        # Reverse (eskiden yeniye)
        df = df.iloc[::-1].reset_index(drop=True)
        
        # Rolling average hesapla (smoothing için)
        for col in ['progressive_correct', 'ultra_correct', 'xgboost_correct', 'ensemble_correct']:
            if col in df.columns:
                df[f'{col}_rolling'] = df[col].rolling(window=20, min_periods=1).mean()
        
        return df
    
    def export_to_csv(
        self,
        filename: str = "data/ensemble_performance.csv",
        window: Optional[int] = None
    ):
        """
        Performans verilerini CSV'ye export et
        
        Args:
            filename: Dosya adı
            window: Kaç tahmin (None ise tümü)
        """
        conn = sqlite3.connect(self.db_path)
        
        if window:
            query = "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?"
            df = pd.read_sql_query(query, conn, params=(window,))
        else:
            query = "SELECT * FROM predictions ORDER BY timestamp DESC"
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        
        # CSV'ye kaydet
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        
        logger.info(f"✅ {len(df)} tahmin CSV'ye export edildi: {filename}")
        return filename
    
    def get_statistics_summary(self) -> Dict:
        """
        Genel istatistikler özeti
        
        Returns:
            Summary dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Toplam tahmin sayısı
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total = cursor.fetchone()[0]
        
        # En iyi performans gösteren model
        if total > 0:
            models = ['progressive', 'ultra', 'xgboost', 'ensemble']
            best_model = max(
                models,
                key=lambda m: self.get_rolling_accuracy(m, min(100, total))
            )
            best_acc = self.get_rolling_accuracy(best_model, min(100, total))
        else:
            best_model = None
            best_acc = 0.0
        
        # İlk ve son tahmin zamanı
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM predictions")
        first_pred, last_pred = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_predictions': total,
            'best_model': best_model,
            'best_accuracy': best_acc,
            'first_prediction': first_pred,
            'last_prediction': last_pred,
            'database_path': self.db_path
        }
