"""
JetX Predictor - Config Loader

Bu modül config.yaml dosyasını yükler ve uygulamanın her yerinde kullanılabilir hale getirir.

GÜNCELLEME:
- Varsayılan config değerleri yeni sisteme (2 Modlu, Threshold Manager) uyarlandı.
- Logging entegrasyonu.
"""

import yaml
import os
import logging
from typing import Dict, Any
from pathlib import Path

# Logging ayarla
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
                logger.info(f"✅ Config yüklendi: {config_path}")
            else:
                # Config dosyası yoksa default değerler
                logger.warning("⚠️ Config dosyası bulunamadı! Default değerler kullanılacak.")
                self._config = self._get_default_config()
                
                # Opsiyonel: Default config'i kaydetme denemesi
                try:
                    os.makedirs(os.path.dirname(config_path), exist_ok=True)
                    with open(config_path, 'w', encoding='utf-8') as f:
                        yaml.dump(self._config, f, default_flow_style=False)
                    logger.info(f"📝 Default config dosyası oluşturuldu: {config_path}")
                except:
                    pass

        except yaml.YAMLError as e:
            logger.error(f"❌ YAML Parse Hatası: {e}")
            self._config = self._get_default_config()
        except Exception as e:
            logger.error(f"❌ Config Yükleme Hatası: {e}")
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Default config değerleri (Güncel Sistem Uyumlu)
        """
        return {
            'database': {
                'path': 'data/jetx_data.db',
                'backup_path': 'data/backups/'
            },
            'model': {
                'path': 'models/jetx_model.h5',
                'scaler_path': 'models/scaler.pkl',
                'sequence_length': 1000  # Güncellendi: 50 -> 1000 (Multi-scale)
            },
            'prediction': {
                'critical_threshold': 1.5,
                'default_mode': 'normal'
            },
            'training_thresholds': {
                'normal': 0.85,
                'rolling': 0.95,
                'detailed_metrics': 0.85,
                'production_default': 0.95,
                'model_checkpoint': 0.85
            },
            'loss_penalties': {
                'false_positive_penalty': 2.5, # Güncellendi: 5.0 -> 2.5 (Dengeli)
                'false_negative_penalty': 1.5,
                'critical_zone_penalty': 3.0
            },
            'adaptive_weights': {
                'initial_false_positive_weight': 2.0
            },
            'logging': {
                'file': 'data/app.log',
                'level': 'INFO'
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
            key_path: Config key'i (örn: 'database.path')
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
