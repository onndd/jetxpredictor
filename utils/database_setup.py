"""
JetX Predictor - Database Setup ve İndexleme

Bu modül veritabanı yapısını oluşturur ve optimize eder.
"""

import sqlite3
import os


def setup_database(db_path: str = "data/jetx_data.db"):
    """
    Veritabanını oluşturur ve optimize eder
    
    Args:
        db_path: Veritabanı dosyası yolu
    """
    # Dizini oluştur
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("📊 Veritabanı tabloları oluşturuluyor...")
    
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
    
    print("✅ Tablolar oluşturuldu")
    
    # Index'leri ekle
    print("🔧 Performans index'leri ekleniyor...")
    
    try:
        # jetx_results için index'ler
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_jetx_timestamp 
            ON jetx_results(timestamp DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_jetx_value 
            ON jetx_results(value)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_jetx_id_desc 
            ON jetx_results(id DESC)
        """)
        
        # predictions için index'ler
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_timestamp 
            ON predictions(timestamp DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_mode 
            ON predictions(mode)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_evaluated 
            ON predictions(actual_value) 
            WHERE actual_value IS NOT NULL
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_mode_evaluated 
            ON predictions(mode, actual_value)
        """)
        
        print("✅ Index'ler başarıyla eklendi")
        
    except sqlite3.Error as e:
        print(f"⚠️ Index ekleme hatası (zaten var olabilir): {e}")
    
    # Veritabanı optimizasyonları
    print("⚡ Veritabanı optimize ediliyor...")
    
    try:
        # VACUUM - veritabanını sıkıştır ve optimize et
        cursor.execute("VACUUM")
        
        # ANALYZE - query planner için istatistikleri güncelle
        cursor.execute("ANALYZE")
        
        print("✅ Optimizasyon tamamlandı")
        
    except sqlite3.Error as e:
        print(f"⚠️ Optimizasyon hatası: {e}")
    
    conn.commit()
    conn.close()
    
    print("\n🎉 Veritabanı kurulumu tamamlandı!")
    return True


def get_database_info(db_path: str = "data/jetx_data.db"):
    """
    Veritabanı hakkında bilgi verir
    
    Args:
        db_path: Veritabanı dosyası yolu
    """
    if not os.path.exists(db_path):
        print(f"❌ Veritabanı bulunamadı: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print(f"\n📊 Veritabanı Bilgileri: {db_path}")
    print("=" * 60)
    
    # Tablo sayısı
    cursor.execute("""
        SELECT COUNT(*) FROM sqlite_master 
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
    """)
    table_count = cursor.fetchone()[0]
    print(f"Tablolar: {table_count}")
    
    # Index sayısı
    cursor.execute("""
        SELECT COUNT(*) FROM sqlite_master 
        WHERE type='index' AND name NOT LIKE 'sqlite_%'
    """)
    index_count = cursor.fetchone()[0]
    print(f"Index'ler: {index_count}")
    
    # jetx_results bilgileri
    cursor.execute("SELECT COUNT(*) FROM jetx_results")
    results_count = cursor.fetchone()[0]
    print(f"\nJetX Results: {results_count:,} kayıt")
    
    # predictions bilgileri
    cursor.execute("SELECT COUNT(*) FROM predictions")
    predictions_count = cursor.fetchone()[0]
    print(f"Predictions: {predictions_count:,} kayıt")
    
    # Dosya boyutu
    file_size = os.path.getsize(db_path)
    size_mb = file_size / (1024 * 1024)
    print(f"\nDosya Boyutu: {size_mb:.2f} MB")
    
    # Index listesi
    print("\n📋 Index Listesi:")
    cursor.execute("""
        SELECT name, tbl_name FROM sqlite_master 
        WHERE type='index' AND name NOT LIKE 'sqlite_%'
        ORDER BY tbl_name, name
    """)
    for idx_name, tbl_name in cursor.fetchall():
        print(f"  - {idx_name} (tablo: {tbl_name})")
    
    conn.close()
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    db_path = "data/jetx_data.db"
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "setup":
            setup_database(db_path)
        elif command == "info":
            get_database_info(db_path)
        else:
            print(f"❌ Bilinmeyen komut: {command}")
            print("Kullanım: python database_setup.py [setup|info]")
    else:
        # Varsayılan: setup
        setup_database(db_path)
        get_database_info(db_path)
