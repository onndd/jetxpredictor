"""
JetX Predictor - Ensemble Performance Monitoring

Real-time performans takibi ve model karşılaştırması için monitoring sistemi.
Her model tahmininin doğruluğunu izler ve detaylı raporlar sunar.

GÜNCELLEME:
- 2 Modlu Yapı (Normal/Rolling) takibi eklendi.
- Veritabanı şeması Normal ve Rolling modlar için ayrıştırıldı.
- Threshold Manager entegrasyonu.
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import os
from utils.threshold_manager import get_threshold_manager

# Logging ayarla
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleMonitor:
    """
    Ensemble ve individual model performanslarını izler
    """
    
    def __init__(self, db_path: str = "data/ensemble_performance.db"):
        """
        Args:
            db_path: Performance database dosya yolu
        """
        self.db_path = db_path
        self.tm = get_threshold_manager()
        self.THRESHOLD_NORMAL = self.tm.get_normal_threshold()   # 0.85
        self.THRESHOLD_ROLLING = self.tm.get_rolling_threshold() # 0.95
        
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Database ve tabloları oluştur"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Predictions tablosu (GÜNCELLENDİ - 2 Modlu)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                
                -- Base model tahminleri
                progressive_pred REAL,
                progressive_threshold_prob REAL,
                
                ultra_pred REAL,
                ultra_threshold_prob REAL,
                
                xgboost_pred REAL,
                xgboost_threshold_prob REAL,
                
                -- Ensemble tahmini
                ensemble_pred REAL,
                ensemble_threshold_prob REAL,
                ensemble_method TEXT,
                
                -- Gerçek değer
                actual_value REAL,
                actual_above_threshold INTEGER,
                
                -- Doğruluk bilgileri (Normal Mod - 0.85)
                progressive_correct_normal INTEGER,
                ultra_correct_normal INTEGER,
                xgboost_correct_normal INTEGER,
                ensemble_correct_normal INTEGER,

                -- Doğruluk bilgileri (Rolling Mod - 0.95)
                progressive_correct_rolling INTEGER,
                ultra_correct_rolling INTEGER,
                xgboost_correct_rolling INTEGER,
                ensemble_correct_rolling INTEGER,
                
                -- Value prediction hatası (MAE)
                progressive_error REAL,
                ultra_error REAL,
                xgboost_error REAL,
                ensemble_error REAL
            )
        """)
        
        # Index oluştur
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON predictions(timestamp DESC)
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Performance database hazır: {self.db_path}")
    
    def _calculate_correct(self, prob, actual_above, threshold):
        """Olasılığa göre tahminin doğruluğunu hesapla"""
        if prob is None: return None
        pred_above = 1 if prob >= threshold else 0
        # Sadece pozitif tahminlerde (1.5 üstü dediğinde) başarıyı ölçmek istersek:
        # if pred_above == 0: return None # Pas geçtiği için nötr
        # Ancak genel doğruluk için her iki durumu da sayıyoruz:
        return 1 if pred_above == actual_above else 0

    def log_prediction(
        self,
        individual_predictions: Dict,
        ensemble_prediction: Dict,
        actual_value: float
    ):
        """
        Tahmin ve gerçek değeri kaydet
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Gerçek eşik değeri
        actual_above_threshold = 1 if actual_value >= 1.5 else 0
        
        # Modelleri al
        prog = individual_predictions.get('progressive') or {}
        ultra = individual_predictions.get('ultra') or {}
        xgb = individual_predictions.get('xgboost') or {}
        
        # Olasılıklar
        prog_prob = prog.get('threshold_probability')
        ultra_prob = ultra.get('threshold_probability')
        xgb_prob = xgb.get('threshold_probability')
        ens_prob = ensemble_prediction.get('threshold_probability')
        
        # Doğruluk Hesaplama - Normal Mod (0.85)
        prog_corr_norm = self._calculate_correct(prog_prob, actual_above_threshold, self.THRESHOLD_NORMAL)
        ultra_corr_norm = self._calculate_correct(ultra_prob, actual_above_threshold, self.THRESHOLD_NORMAL)
        xgb_corr_norm = self._calculate_correct(xgb_prob, actual_above_threshold, self.THRESHOLD_NORMAL)
        ens_corr_norm = self._calculate_correct(ens_prob, actual_above_threshold, self.THRESHOLD_NORMAL)

        # Doğruluk Hesaplama - Rolling Mod (0.95)
        prog_corr_roll = self._calculate_correct(prog_prob, actual_above_threshold, self.THRESHOLD_ROLLING)
        ultra_corr_roll = self._calculate_correct(ultra_prob, actual_above_threshold, self.THRESHOLD_ROLLING)
        xgb_corr_roll = self._calculate_correct(xgb_prob, actual_above_threshold, self.THRESHOLD_ROLLING)
        ens_corr_roll = self._calculate_correct(ens_prob, actual_above_threshold, self.THRESHOLD_ROLLING)
        
        # Hata hesapla (MAE)
        prog_error = abs(prog.get('predicted_value', 0) - actual_value) if prog.get('predicted_value') else None
        ultra_error = abs(ultra.get('predicted_value', 0) - actual_value) if ultra.get('predicted_value') else None
        xgb_error = abs(xgb.get('predicted_value', 0) - actual_value) if xgb.get('predicted_value') else None
        ens_error = abs(ensemble_prediction.get('predicted_value', 0) - actual_value) if ensemble_prediction.get('predicted_value') else None
        
        # Kaydet
        cursor.execute("""
            INSERT INTO predictions (
                progressive_pred, progressive_threshold_prob,
                ultra_pred, ultra_threshold_prob,
                xgboost_pred, xgboost_threshold_prob,
                ensemble_pred, ensemble_threshold_prob, ensemble_method,
                actual_value, actual_above_threshold,
                progressive_correct_normal, ultra_correct_normal, xgboost_correct_normal, ensemble_correct_normal,
                progressive_correct_rolling, ultra_correct_rolling, xgboost_correct_rolling, ensemble_correct_rolling,
                progressive_error, ultra_error, xgboost_error, ensemble_error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prog.get('predicted_value'), prog_prob,
            ultra.get('predicted_value'), ultra_prob,
            xgb.get('predicted_value'), xgb_prob,
            ensemble_prediction.get('predicted_value'), ens_prob,
            ensemble_prediction.get('ensemble_method', 'unknown'),
            
            actual_value, actual_above_threshold,
            
            prog_corr_norm, ultra_corr_norm, xgb_corr_norm, ens_corr_norm,
            prog_corr_roll, ultra_corr_roll, xgb_corr_roll, ens_corr_roll,
            
            prog_error, ultra_error, xgb_error, ens_error
        ))
        
        conn.commit()
        conn.close()
        logger.debug(f"Tahmin kaydedildi: Actual={actual_value:.2f}x")
    
    def get_rolling_accuracy(self, model_name: str, window: int = 100, mode: str = 'normal') -> float:
        """Son N tahmindeki accuracy"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        col_name = f"{model_name}_correct_{mode}"
        
        query = f"""
            SELECT AVG({col_name}) 
            FROM (
                SELECT {col_name} 
                FROM predictions 
                WHERE {col_name} IS NOT NULL
                ORDER BY timestamp DESC 
                LIMIT ?
            )
        """
        
        cursor.execute(query, (window,))
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result[0] is not None else 0.0
    
    def get_rolling_mae(self, model_name: str, window: int = 100) -> float:
        """Son N tahmindeki MAE"""
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
    
    def calculate_trend(self, model_name: str, mode: str = 'normal') -> str:
        """Model performans trendi"""
        recent = self.get_rolling_accuracy(model_name, 50, mode)
        older = self.get_rolling_accuracy(model_name, 100, mode)
        
        diff = recent - older
        if diff > 0.05: return '↗️ improving'
        elif diff < -0.05: return '↘️ declining'
        else: return '➡️ stable'

    def generate_comparison_report(self, window: int = 100) -> Dict:
        """Tüm modellerin karşılaştırmalı raporu"""
        models = ['progressive', 'ultra', 'xgboost', 'ensemble']
        report = {}
        
        for model_name in models:
            acc_norm = self.get_rolling_accuracy(model_name, window, 'normal')
            acc_roll = self.get_rolling_accuracy(model_name, window, 'rolling')
            mae = self.get_rolling_mae(model_name, window)
            trend = self.calculate_trend(model_name, 'normal')
            
            report[model_name] = {
                'accuracy_normal': acc_norm,
                'accuracy_rolling': acc_roll,
                'mae': mae,
                'trend': trend
            }
        return report

    def get_chart_data(self, window: int = 500) -> pd.DataFrame:
        """Grafik için time series data"""
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT 
                timestamp,
                ensemble_correct_normal,
                ensemble_correct_rolling
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(window,))
        conn.close()
        
        df = df.iloc[::-1].reset_index(drop=True)
        # Rolling average
        df['normal_rolling_acc'] = df['ensemble_correct_normal'].rolling(window=20).mean()
        df['rolling_rolling_acc'] = df['ensemble_correct_rolling'].rolling(window=20).mean()
        
        return df

    def export_to_csv(self, filename: str = "data/ensemble_performance.csv", window: Optional[int] = None):
        conn = sqlite3.connect(self.db_path)
        if window:
            query = "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?"
            df = pd.read_sql_query(query, conn, params=(window,))
        else:
            query = "SELECT * FROM predictions ORDER BY timestamp DESC"
            df = pd.read_sql_query(query, conn)
        conn.close()
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        logger.info(f"✅ CSV export: {filename}")
        return filename

    def get_statistics_summary(self) -> Dict:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total = cursor.fetchone()[0]
        
        # En iyi modeli bul (Normal Mod)
        best_model = None
        best_acc = 0.0
        if total > 0:
            models = ['progressive', 'ultra', 'xgboost', 'ensemble']
            best_model = max(models, key=lambda m: self.get_rolling_accuracy(m, min(100, total), 'normal'))
            best_acc = self.get_rolling_accuracy(best_model, min(100, total), 'normal')
            
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM predictions")
        first_pred, last_pred = cursor.fetchone()
        conn.close()
        
        return {
            'total_predictions': total,
            'best_model': best_model,
            'best_accuracy': best_acc,
            'first_prediction': first_pred,
            'last_prediction': last_pred
        }
