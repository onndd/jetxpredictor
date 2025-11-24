"""
JetX Predictor - Veritabanı Yönetim Modülü

Bu modül SQLite veritabanı işlemlerini yönetir.

GÜNCELLEME:
- 2 Modlu (Normal/Rolling) yapıya uygun veri kaydı ve okuma.
- Genişletilmiş 'predictions' tablosu desteği.
"""

import sqlite3
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import pandas as pd
import logging
import time

# Logging ayarla
logger = logging.getLogger(__name__)


class DatabaseManager:
    """SQLite veritabanı işlemlerini yöneten sınıf"""
    
    # Connection ayarları
    CONNECTION_TIMEOUT = 30.0  # 30 saniye timeout
    MAX_RETRIES = 3  # Maksimum deneme sayısı
    RETRY_DELAY = 0.5  # Denemeler arası bekleme (saniye)
    
    def __init__(self, db_path: str = "data/jetx_data.db"):
        """
        Args:
            db_path: Veritabanı dosyasının yolu
        """
        self.db_path = db_path
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Veritabanı ve tabloların varlığını kontrol eder, yoksa oluşturur"""
        # Dizini oluştur
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # jetx_results tablosu
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jetx_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                value REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # predictions tablosu (GÜNCELLENMİŞ ŞEMA)
        # Not: Eğer tablo zaten varsa ve eski şemadaysa, ALTER TABLE gerekebilir.
        # Basitlik için burada IF NOT EXISTS kullanıyoruz.
        # Üretim ortamında migration scripti gerekir.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                
                -- Temel Alanlar (Legacy & New)
                predicted_value REAL,
                confidence_score REAL,
                above_threshold INTEGER,
                actual_value REAL,
                was_correct INTEGER,
                mode TEXT DEFAULT 'normal',
                
                -- Yeni Alanlar (2 Modlu Yapı)
                progressive_pred REAL,
                progressive_threshold_prob REAL,
                ultra_pred REAL,
                ultra_threshold_prob REAL,
                xgboost_pred REAL,
                xgboost_threshold_prob REAL,
                
                ensemble_pred REAL,
                ensemble_threshold_prob REAL,
                ensemble_method TEXT,
                
                actual_above_threshold INTEGER,
                
                progressive_correct_normal INTEGER,
                ultra_correct_normal INTEGER,
                xgboost_correct_normal INTEGER,
                ensemble_correct_normal INTEGER,

                progressive_correct_rolling INTEGER,
                ultra_correct_rolling INTEGER,
                xgboost_correct_rolling INTEGER,
                ensemble_correct_rolling INTEGER,
                
                progressive_error REAL,
                ultra_error REAL,
                xgboost_error REAL,
                ensemble_error REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def get_connection(self) -> sqlite3.Connection:
        """Veritabanı bağlantısı döndürür"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=self.CONNECTION_TIMEOUT)
            conn.execute("PRAGMA journal_mode=WAL")
            return conn
        except sqlite3.Error as e:
            logger.error(f"Veritabanı bağlantı hatası: {e}", exc_info=True)
            raise
    
    def get_all_results(self, limit: Optional[int] = None) -> List[float]:
        """Tüm JetX sonuçlarını getirir"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            if limit:
                cursor.execute("SELECT value FROM jetx_results ORDER BY id DESC LIMIT ?", (limit,))
            else:
                cursor.execute("SELECT value FROM jetx_results ORDER BY id")
            
            results = [row[0] for row in cursor.fetchall()]
            
            if limit:
                results.reverse()
            
            return results
        except Exception as e:
            logger.error(f"Hata (get_all_results): {e}")
            return []
        finally:
            if conn: conn.close()
    
    def get_recent_results(self, n: int = 100) -> List[float]:
        """Son N sonucu getirir"""
        return self.get_all_results(limit=n)
    
    def add_result(self, value: float) -> int:
        """Yeni bir JetX sonucu ekler"""
        for attempt in range(self.MAX_RETRIES):
            conn = None
            try:
                conn = self.get_connection()
                cursor = conn.cursor()
                
                cursor.execute("INSERT INTO jetx_results (value) VALUES (?)", (value,))
                result_id = cursor.lastrowid
                
                conn.commit()
                logger.info(f"Veri eklendi: ID={result_id}, value={value:.2f}x")
                return result_id
                
            except sqlite3.OperationalError as e:
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                    continue
                else:
                    logger.error(f"DB Kilitli: {e}")
                    return -1
            except Exception as e:
                logger.error(f"Veri ekleme hatası: {e}")
                return -1
            finally:
                if conn: conn.close()
        return -1
    
    def add_prediction(
        self,
        predicted_value: float,
        confidence_score: float,
        above_threshold: bool,
        actual_value: Optional[float] = None,
        was_correct: Optional[bool] = None,
        mode: str = 'normal',
        extra_data: Optional[Dict] = None # Yeni alanlar için
    ) -> int:
        """Yeni bir tahmin kaydı ekler (Genişletilmiş)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Temel alanlar
        columns = ['predicted_value', 'confidence_score', 'above_threshold', 'actual_value', 'was_correct', 'mode']
        values = [predicted_value, confidence_score, 1 if above_threshold else 0, actual_value, 1 if was_correct else 0 if was_correct is not None else None, mode]
        
        # Ekstra alanlar (varsa)
        if extra_data:
            for key, value in extra_data.items():
                # Sütun adının güvenliği kontrol edilmeli ama şimdilik güveniyoruz
                columns.append(key)
                values.append(value)
        
        placeholders = ', '.join(['?'] * len(values))
        columns_str = ', '.join(columns)
        
        try:
            cursor.execute(f"INSERT INTO predictions ({columns_str}) VALUES ({placeholders})", values)
            prediction_id = cursor.lastrowid
            conn.commit()
            return prediction_id
        except Exception as e:
            logger.error(f"Tahmin ekleme hatası: {e}")
            return -1
        finally:
            conn.close()
    
    def update_prediction_result(
        self,
        prediction_id: int,
        actual_value: float,
        was_correct: bool
    ):
        """Tahmin sonucunu günceller"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE predictions
                SET actual_value = ?, was_correct = ?
                WHERE id = ?
            """, (actual_value, 1 if was_correct else 0, prediction_id))
            conn.commit()
        except Exception as e:
            logger.error(f"Sonuç güncelleme hatası: {e}")
        finally:
            conn.close()
    
    def get_predictions(
        self,
        limit: Optional[int] = None,
        mode: Optional[str] = None,
        only_evaluated: bool = False
    ) -> pd.DataFrame:
        """Tahminleri getirir"""
        conn = None
        try:
            conn = self.get_connection()
            query = "SELECT * FROM predictions WHERE 1=1"
            params = []
            
            if mode:
                query += " AND mode = ?"
                params.append(mode)
            
            if only_evaluated:
                query += " AND actual_value IS NOT NULL"
            
            query += " ORDER BY id DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            return pd.read_sql_query(query, conn, params=params or None)
        except Exception as e:
            logger.error(f"Tahminleri getirme hatası: {e}")
            return pd.DataFrame()
        finally:
            if conn: conn.close()
    
    def get_database_stats(self) -> Dict:
        """Genel veritabanı istatistikleri"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM jetx_results")
            total_results = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM jetx_results WHERE value >= 1.5")
            above_threshold = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(value), MIN(value), MAX(value) FROM jetx_results")
            avg, min_val, max_val = cursor.fetchone()
            
            return {
                'total_results': total_results,
                'above_threshold_count': above_threshold,
                'above_threshold_ratio': above_threshold / total_results if total_results > 0 else 0,
                'average_value': avg or 0,
                'min_value': min_val or 0,
                'max_value': max_val or 0
            }
        except Exception as e:
            logger.error(f"DB stats hatası: {e}")
            return {'total_results': 0, 'above_threshold_ratio': 0, 'average_value': 0, 'min_value': 0, 'max_value': 0}
        finally:
            if conn: conn.close()
    
    def backup_database(self, backup_path: Optional[str] = None) -> str:
        """Veritabanını yedekler"""
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"data/backups/jetx_backup_{timestamp}.db"
        
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        import shutil
        shutil.copy2(self.db_path, backup_path)
        
        return backup_path
