"""
Model Versiyonlama Sistemi (v2.0)

Model versiyonlarını yönetir, kaydeder ve yükler.
Her model versiyonu için metadata, performans metrikleri (ROI, Accuracy) ve dosya yolları saklanır.

GÜNCELLEME:
- 2 Modlu yapıya uygun metriklerin saklanması desteklendi.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)


class ModelVersionManager:
    """Model versiyonlarını yönetir"""
    
    def __init__(self, registry_path: str = "models/model_registry.json"):
        """
        Args:
            registry_path: Model registry dosyası yolu
        """
        self.registry_path = registry_path
        self.registry = self._load_registry()
        self.models_dir = "models"
        
        # Registry dosyasının dizinini oluştur
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    
    def _load_registry(self) -> Dict:
        """Registry dosyasını yükle"""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Registry yükleme hatası: {e}")
                return {}
        return {}
    
    def _save_registry(self):
        """Registry dosyasını kaydet"""
        try:
            with open(self.registry_path, 'w', encoding='utf-8') as f:
                json.dump(self.registry, f, indent=2, ensure_ascii=False)
            logger.info("Model registry kaydedildi")
        except Exception as e:
            logger.error(f"Registry kaydetme hatası: {e}")
    
    def register_model(
        self,
        model_name: str,
        model_type: str,
        version: str,
        model_files: Dict[str, str],
        metadata: Optional[Dict] = None,
        metrics: Optional[Dict] = None,
        is_production: bool = False
    ) -> str:
        """
        Yeni model versiyonu kaydet
        """
        model_id = f"{model_name}_v{version}"
        timestamp = datetime.now().isoformat()
        
        # Eğer production modeli ise, eski production modelini kaldır
        if is_production:
            self._unset_production(model_name)
        
        model_entry = {
            'model_id': model_id,
            'model_name': model_name,
            'model_type': model_type,
            'version': version,
            'model_files': model_files,
            'metadata': metadata or {},
            'metrics': metrics or {},
            'is_production': is_production,
            'created_at': timestamp,
            'updated_at': timestamp
        }
        
        # Registry'ye ekle
        if model_name not in self.registry:
            self.registry[model_name] = {}
        
        self.registry[model_name][version] = model_entry
        self._save_registry()
        
        logger.info(f"Model kaydedildi: {model_id} (Production: {is_production})")
        return model_id
    
    def get_model_info(self, model_name: str, version: Optional[str] = None) -> Optional[Dict]:
        """
        Model bilgilerini al
        """
        if model_name not in self.registry:
            return None
        
        if version is None:
            # Production modelini bul
            for v, info in self.registry[model_name].items():
                if info.get('is_production', False):
                    return info
            # Production yoksa en son versiyonu döndür
            if self.registry[model_name]:
                latest = max(self.registry[model_name].keys(), key=lambda x: self._parse_version(x))
                return self.registry[model_name][latest]
            return None
        
        if version in self.registry[model_name]:
            return self.registry[model_name][version]
        
        return None
    
    def get_all_versions(self, model_name: str) -> List[Dict]:
        """Modelin tüm versiyonlarını listele"""
        if model_name not in self.registry:
            return []
        
        versions = list(self.registry[model_name].values())
        # Versiyona göre sırala (en yeni önce)
        versions.sort(key=lambda x: self._parse_version(x['version']), reverse=True)
        return versions
    
    def set_production(self, model_name: str, version: str) -> bool:
        """Belirli bir versiyonu production yap"""
        if model_name not in self.registry:
            logger.error(f"Model bulunamadı: {model_name}")
            return False
        
        if version not in self.registry[model_name]:
            logger.error(f"Versiyon bulunamadı: {model_name} v{version}")
            return False
        
        # Eski production'ı kaldır
        self._unset_production(model_name)
        
        # Yeni production'ı ayarla
        self.registry[model_name][version]['is_production'] = True
        self.registry[model_name][version]['updated_at'] = datetime.now().isoformat()
        self._save_registry()
        
        logger.info(f"Production modeli ayarlandı: {model_name} v{version}")
        return True
    
    def _unset_production(self, model_name: str):
        """Modelin production flag'ini kaldır"""
        if model_name in self.registry:
            for version in self.registry[model_name]:
                self.registry[model_name][version]['is_production'] = False
    
    def update_metrics(self, model_name: str, version: str, metrics: Dict):
        """Model metriklerini güncelle"""
        if model_name not in self.registry:
            logger.error(f"Model bulunamadı: {model_name}")
            return False
        
        if version not in self.registry[model_name]:
            logger.error(f"Versiyon bulunamadı: {model_name} v{version}")
            return False
        
        self.registry[model_name][version]['metrics'].update(metrics)
        self.registry[model_name][version]['updated_at'] = datetime.now().isoformat()
        self._save_registry()
        
        logger.info(f"Metrikler güncellendi: {model_name} v{version}")
        return True
    
    def get_production_model(self, model_name: str) -> Optional[Dict]:
        """Production modelini al"""
        return self.get_model_info(model_name, version=None)
    
    def list_all_models(self) -> List[str]:
        """Tüm model isimlerini listele"""
        return list(self.registry.keys())
    
    def _parse_version(self, version: str) -> tuple:
        """Versiyon string'ini tuple'a çevir (sıralama için)"""
        try:
            parts = version.split('.')
            return tuple(int(p) for p in parts)
        except:
            return (0, 0, 0)
    
    def compare_versions(self, model_name: str, version1: str, version2: str) -> Dict:
        """İki versiyonu karşılaştır"""
        info1 = self.get_model_info(model_name, version1)
        info2 = self.get_model_info(model_name, version2)
        
        if not info1 or not info2:
            return {'error': 'Versiyon bulunamadı'}
        
        comparison = {
            'model_name': model_name,
            'version1': {
                'version': version1,
                'metrics': info1.get('metrics', {}),
                'created_at': info1.get('created_at'),
                'is_production': info1.get('is_production', False)
            },
            'version2': {
                'version': version2,
                'metrics': info2.get('metrics', {}),
                'created_at': info2.get('created_at'),
                'is_production': info2.get('is_production', False)
            },
            'differences': {}
        }
        
        # Metrikleri karşılaştır
        metrics1 = info1.get('metrics', {})
        metrics2 = info2.get('metrics', {})
        
        all_metrics = set(metrics1.keys()) | set(metrics2.keys())
        for metric in all_metrics:
            val1 = metrics1.get(metric)
            val2 = metrics2.get(metric)
            if val1 != val2:
                comparison['differences'][metric] = {
                    'version1': val1,
                    'version2': val2,
                    'improvement': val2 - val1 if isinstance(val1, (int, float)) and isinstance(val2, (int, float)) else None
                }
        
        return comparison


# Global instance
_version_manager = None

def get_version_manager() -> ModelVersionManager:
    """Global version manager instance'ı al"""
    global _version_manager
    if _version_manager is None:
        _version_manager = ModelVersionManager()
    return _version_manager
