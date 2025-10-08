"""
JetX Predictor - Veritabanı Yönetim Modülü

Bu modül SQLite veritabanı işlemlerini yönetir.
"""

import sqlite3
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import pandas as pd


class DatabaseManager:
    """SQLite veritabanı işlemlerini yöneten sınıf"""
    
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
        
        # predictions tablosu
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                predicted_value REAL,
                confidence_score REAL,
                above_threshold INTEGER,
                actual_value REAL,
                was_correct INTEGER,
                mode TEXT DEFAULT 'normal',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def get_connection(self) -> sqlite3.Connection:
        """Veritabanı bağlantısı döndürür"""
        return sqlite3.connect(self.db_path)
    
    def get_all_results(self, limit: Optional[int] = None) -> List[float]:
        """
        Tüm JetX sonuçlarını getirir
        
        Args:
            limit: Maksimum kayıt sayısı (None ise tümü)
            
        Returns:
            Değerler listesi
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if limit:
            cursor.execute("SELECT value FROM jetx_results ORDER BY id DESC LIMIT ?", (limit,))
        else:
            cursor.execute("SELECT value FROM jetx_results ORDER BY id")
        
        results = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        # Ters çevir (en eskiden en yeniye)
        if limit:
            results.reverse()
        
        return results
    
    def get_recent_results(self, n: int = 100) -> List[float]:
        """
        Son N sonucu getirir
        
        Args:
            n: Kaç sonuç getirileceği
            
        Returns:
            Son N değer
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT value FROM jetx_results ORDER BY id DESC LIMIT ?", (n,))
        results = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        # Ters çevir (eskiden yeniye)
        results.reverse()
        
        return results
    
    def add_result(self, value: float) -> int:
        """
        Yeni bir JetX sonucu ekler
        
        Args:
            value: Çarpan değeri
            
        Returns:
            Eklenen kaydın ID'si
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("INSERT INTO jetx_results (value) VALUES (?)", (value,))
        result_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return result_id
    
    def add_prediction(
        self,
        predicted_value: float,
        confidence_score: float,
        above_threshold: bool,
        actual_value: Optional[float] = None,
        was_correct: Optional[bool] = None,
        mode: str = 'normal'
    ) -> int:
        """
        Yeni bir tahmin kaydı ekler
        
        Args:
            predicted_value: Tahmin edilen değer
            confidence_score: Güven skoru (0-1 arası)
            above_threshold: 1.5x üstü mü tahmin edildi
            actual_value: Gerçekleşen değer (varsa)
            was_correct: Tahmin doğru muydu (varsa)
            mode: Tahmin modu (normal, rolling, aggressive)
            
        Returns:
            Eklenen tahmin ID'si
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO predictions 
            (predicted_value, confidence_score, above_threshold, actual_value, was_correct, mode)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            predicted_value,
            confidence_score,
            1 if above_threshold else 0,
            actual_value,
            1 if was_correct else 0 if was_correct is not None else None,
            mode
        ))
        
        prediction_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return prediction_id
    
    def update_prediction_result(
        self,
        prediction_id: int,
        actual_value: float,
        was_correct: bool
    ):
        """
        Tahmin sonucunu günceller
        
        Args:
            prediction_id: Tahmin ID'si
            actual_value: Gerçekleşen değer
            was_correct: Tahmin doğru muydu
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE predictions
            SET actual_value = ?, was_correct = ?
            WHERE id = ?
        """, (actual_value, 1 if was_correct else 0, prediction_id))
        
        conn.commit()
        conn.close()
    
    def get_predictions(
        self,
        limit: Optional[int] = None,
        mode: Optional[str] = None,
        only_evaluated: bool = False
    ) -> pd.DataFrame:
        """
        Tahminleri getirir
        
        Args:
            limit: Maksimum kayıt sayısı
            mode: Belirli bir mod filtreleme
            only_evaluated: Sadece değerlendirilen tahminler
            
        Returns:
            DataFrame
        """
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
        
        df = pd.read_sql_query(query, conn, params=params if params else None)
        conn.close()
        
        return df
    
    def get_prediction_stats(self, mode: Optional[str] = None) -> Dict:
        """
        Tahmin istatistiklerini getirir
        
        Args:
            mode: Belirli bir mod için (None ise tümü)
            
        Returns:
            İstatistik sözlüğü
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        query = "SELECT COUNT(*), SUM(was_correct), AVG(confidence_score) FROM predictions WHERE actual_value IS NOT NULL"
        params = []
        
        if mode:
            query += " AND mode = ?"
            params.append(mode)
        
        cursor.execute(query, params if params else None)
        total, correct, avg_confidence = cursor.fetchone()
        
        conn.close()
        
        if total and total > 0:
            return {
                'total_predictions': total,
                'correct_predictions': correct or 0,
                'accuracy': (correct or 0) / total if total > 0 else 0,
                'average_confidence': avg_confidence or 0
            }
        else:
            return {
                'total_predictions': 0,
                'correct_predictions': 0,
                'accuracy': 0,
                'average_confidence': 0
            }
    
    def get_database_stats(self) -> Dict:
        """
        Genel veritabanı istatistikleri
        
        Returns:
            İstatistik sözlüğü
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Toplam sonuç sayısı
        cursor.execute("SELECT COUNT(*) FROM jetx_results")
        total_results = cursor.fetchone()[0]
        
        # 1.5x istatistikleri
        cursor.execute("SELECT COUNT(*) FROM jetx_results WHERE value >= 1.5")
        above_threshold = cursor.fetchone()[0]
        
        # Ortalama, min, max
        cursor.execute("SELECT AVG(value), MIN(value), MAX(value) FROM jetx_results")
        avg, min_val, max_val = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_results': total_results,
            'above_threshold_count': above_threshold,
            'above_threshold_ratio': above_threshold / total_results if total_results > 0 else 0,
            'average_value': avg or 0,
            'min_value': min_val or 0,
            'max_value': max_val or 0
        }
    
    def backup_database(self, backup_path: Optional[str] = None):
        """
        Veritabanını yedekler
        
        Args:
            backup_path: Yedek dosya yolu (None ise otomatik oluşturulur)
        """
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"data/backups/jetx_backup_{timestamp}.db"
        
        # Backup dizinini oluştur
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        
        # Kopyala
        import shutil
        shutil.copy2(self.db_path, backup_path)
        
        return backup_path
