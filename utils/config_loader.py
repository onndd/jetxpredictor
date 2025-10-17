"""
JetX Predictor - Config Loader

Bu modül config.yaml dosyasını yükler ve uygulamanın her yerinde kullanılabilir hale getirir.
"""

import yaml
import os
from typing import Dict, Any


class ConfigLoader:
    """Config dosyasını yükleyen ve yöneten sınıf"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        """Singleton pattern - sadece bir instance olsun"""
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Config dosyasını yükle"""
        if self._config is None:
            self.load_config()
    
    def load_config(self, config_path: str = "config/config.yaml") -> None:
        """
        Config dosyasını yükle
        
        Args:
            config_path: Config dosyasının yolu
        """
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f)
                print(f"✅ Config yüklendi: {config_path}")
            else:
                # Config dosyası yoksa default değerler
                print("=" * 70)
                print("⚠️ UYARI: Config dosyası bulunamadı!")
                print("=" * 70)
                print(f"  Aranan yol: {config_path}")
                print(f"  Default değerler kullanılacak.")
                print("  Config dosyası oluşturmak için:")
                print(f"  1. '{os.path.dirname(config_path)}' klasörünü oluşturun")
                print(f"  2. '{config_path}' dosyasını oluşturun")
                print("=" * 70)
                self._config = self._get_default_config()
        except yaml.YAMLError as e:
            print("=" * 70)
            print("❌ YAML Parse Hatası!")
            print("=" * 70)
            print(f"  Config dosyası geçersiz YAML formatında: {e}")
            print(f"  Dosya: {config_path}")
            print("  Default değerler kullanılacak.")
            print("=" * 70)
            self._config = self._get_default_config()
        except Exception as e:
            print("=" * 70)
            print("❌ Config Yükleme Hatası!")
            print("=" * 70)
            print(f"  Hata: {e}")
            print(f"  Dosya: {config_path}")
            print("  Default değerler kullanılacak.")
            print("=" * 70)
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Default config değerleri
        
        Returns:
            Default config dictionary
        """
        return {
            'database': {
                'path': 'data/jetx_data.db',
                'backup_path': 'data/backups/'
            },
            'model': {
                'path': 'models/jetx_model.h5',
                'scaler_path': 'models/scaler.pkl',
                'sequence_length': 50
            },
            'prediction': {
                'critical_threshold': 1.5,
                'high_multiplier_threshold': 3.0,
                'default_mode': 'normal'
            },
            'ui': {
                'theme': 'dark',
                'page_title': 'JetX Predictor',
                'page_icon': '🚀'
            }
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Config değeri al (nested keys desteklenir)
        
        Args:
            key_path: Config key'i (örn: 'database.path' veya 'model.path')
            default: Bulunamazsa döndürülecek default değer
            
        Returns:
            Config değeri veya default
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    @property
    def config(self) -> Dict[str, Any]:
        """Tüm config dictionary'yi döndür"""
        return self._config


# Global config instance
config = ConfigLoader()
