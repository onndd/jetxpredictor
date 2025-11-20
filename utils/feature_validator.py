"""
Feature Hash Validation System - Scaler Compatibility Checker

Bu modül, FeatureEngineering özellik sayısını doğrular ve 
scaler'ların model ile uyumlu olup olmadığını kontrol eder.

Özellikle:
- Eğitim zamanı kaydedilen scaler'ların uyumluluğunu kontrol eder
- Shape mismatch hatasını önler
- Feature hash validation sağlar
- Version control sistemi sunar
"""

import hashlib
import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)


class FeatureValidator:
    """
    Feature hash ve scaler validator sınıfı
    
    - Feature sayısı tutarlılığını kontrol eder
    - Scaler compatibility check sağlar
    - Version control sistemi sunar
    """
    
    def __init__(self, hash_file_path: str = "models/feature_hashes.json"):
        """
        Feature validator'ı başlat
        
        Args:
            hash_file_path: Feature hash'lerini saklayacak dosya yolu
        """
        self.hash_file_path = hash_file_path
        self.feature_hashes = {}
        self.scaler_hashes = {}
        
        # Hash dosyasını yükle
        self._load_hashes()
    
    def _load_hashes(self):
        """Hash dosyasını yükle"""
        try:
            if os.path.exists(self.hash_file_path):
                with open(self.hash_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.feature_hashes = data.get('feature_hashes', {})
                    self.scaler_hashes = data.get('scaler_hashes', {})
                logger.info(f"✅ Feature hashes loaded from {self.hash_file_path}")
            else:
                logger.info(f"📝 Hash file not found, creating new one: {self.hash_file_path}")
                self.feature_hashes = {}
                self.scaler_hashes = {}
        except Exception as e:
            logger.error(f"❌ Error loading hashes: {e}")
            self.feature_hashes = {}
            self.scaler_hashes = {}
    
    def _save_hashes(self):
        """Hash dosyasını kaydet"""
        try:
            # Dizin oluştur
            os.makedirs(os.path.dirname(self.hash_file_path), exist_ok=True)
            
            data = {
                'feature_hashes': self.feature_hashes,
                'scaler_hashes': self.scaler_hashes,
                'last_updated': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            with open(self.hash_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Hashes saved to {self.hash_file_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving hashes: {e}")
    
    def generate_feature_hash(self, features: Dict[str, float]) -> str:
        """
        Feature sözlüğünden hash oluştur
        
        Args:
            features: Feature sözlüğü
            
        Returns:
            SHA256 hash string
        """
        try:
            # Feature key'lerini sırala ve string'e çevir
            sorted_keys = sorted(features.keys())
            feature_string = json.dumps({k: features.get(k, 0.0) for k in sorted_keys}, sort_keys=True)
            
            # SHA256 hash oluştur
            hash_object = hashlib.sha256(feature_string.encode('utf-8'))
            return hash_object.hexdigest()
            
        except Exception as e:
            logger.error(f"❌ Error generating feature hash: {e}")
            return ""
    
    def validate_feature_count(self, features: Dict[str, float], 
                              expected_hash: Optional[str] = None) -> Tuple[bool, str]:
        """
        Feature sayısını ve hash'ini doğrula
        
        Args:
            features: Feature sözlüğü
            expected_hash: Beklenen hash (varsa)
            
        Returns:
            (is_valid, message) tuple
        """
        try:
            # Feature sayısı kontrolü
            feature_count = len(features)
            if feature_count < 50:  # Minimum feature count
                return False, f"❌ Too few features: {feature_count} (minimum: 50)"
            
            if feature_count > 200:  # Maximum feature count  
                return False, f"❌ Too many features: {feature_count} (maximum: 200)"
            
            # Feature hash kontrolü
            current_hash = self.generate_feature_hash(features)
            if not current_hash:
                return False, "❌ Failed to generate feature hash"
            
            # Expected hash varsa karşılaştır
            if expected_hash and current_hash != expected_hash:
                return False, f"❌ Feature mismatch - Expected: {expected_hash[:8]}..., Got: {current_hash[:8]}..."
            
            return True, f"✅ Features valid - Count: {feature_count}, Hash: {current_hash[:8]}..."
            
        except Exception as e:
            return False, f"❌ Validation error: {e}"
    
    def register_features(self, features: Dict[str, float], 
                          model_name: str = "default", version: str = "1.0"):
        """
        Feature'ları kaydet
        
        Args:
            features: Feature sözlüğü
            model_name: Model adı
            version: Versiyon
        """
        try:
            feature_hash = self.generate_feature_hash(features)
            if not feature_hash:
                raise ValueError("Failed to generate feature hash")
            
            # Feature bilgilerini kaydet
            self.feature_hashes[model_name] = {
                'hash': feature_hash,
                'count': len(features),
                'version': version,
                'created_at': datetime.now().isoformat(),
                'features': list(features.keys())[:10],  # İlk 10 feature'ı sakla
                'sample_values': {k: features.get(k, 0.0) for k in list(features.keys())[:5]}
            }
            
            # Kaydet
            self._save_hashes()
            
            logger.info(f"✅ Features registered for {model_name} v{version}")
            logger.info(f"   Hash: {feature_hash[:8]}..., Count: {len(features)}")
            
        except Exception as e:
            logger.error(f"❌ Error registering features: {e}")
            raise
    
    def register_scaler(self, scaler, model_name: str = "default", 
                      version: str = "1.0", feature_count: int = 0):
        """
        Scaler'ı kaydet
        
        Args:
            scaler: Scaler objesi
            model_name: Model adı
            version: Versiyon
            feature_count: Feature sayısı
        """
        try:
            # Scaler'dan feature sayısı al
            if hasattr(scaler, 'n_features_in_'):
                scaler_features = scaler.n_features_in_
            elif hasattr(scaler, 'scale_'):
                scaler_features = len(scaler.scale_) if scaler.scale_ is not None else 0
            else:
                scaler_features = feature_count
            
            # Scaler hash'i oluştur
            scaler_info = {
                'type': type(scaler).__name__,
                'features': scaler_features,
                'version': version,
                'created_at': datetime.now().isoformat()
            }
            
            # Scaler string'inden hash oluştur
            scaler_string = json.dumps(scaler_info, sort_keys=True)
            scaler_hash = hashlib.sha256(scaler_string.encode('utf-8')).hexdigest()
            
            # Kaydet
            self.scaler_hashes[model_name] = {
                'hash': scaler_hash,
                'info': scaler_info
            }
            
            self._save_hashes()
            
            logger.info(f"✅ Scaler registered for {model_name} v{version}")
            logger.info(f"   Hash: {scaler_hash[:8]}..., Features: {scaler_features}")
            
        except Exception as e:
            logger.error(f"❌ Error registering scaler: {e}")
            raise
    
    def validate_compatibility(self, features: Dict[str, float], 
                            scaler, model_name: str = "default") -> Tuple[bool, str]:
        """
        Features ve scaler uyumluluğunu kontrol et
        
        Args:
            features: Feature sözlüğü
            scaler: Scaler objesi
            model_name: Model adı
            
        Returns:
            (is_compatible, message) tuple
        """
        try:
            # Feature validation
            feature_valid, feature_msg = self.validate_feature_count(features)
            if not feature_valid:
                return False, f"Feature validation failed: {feature_msg}"
            
            # Scaler feature count kontrolü
            if hasattr(scaler, 'n_features_in_'):
                expected_features = scaler.n_features_in_
                actual_features = len(features)
                
                if expected_features != actual_features:
                    return False, f"❌ Feature count mismatch: Scaler expects {expected_features}, got {actual_features}"
            
            # Hash kontrolü
            if model_name in self.feature_hashes:
                expected_hash = self.feature_hashes[model_name]['hash']
                current_hash = self.generate_feature_hash(features)
                
                if current_hash != expected_hash:
                    return False, f"❌ Feature hash mismatch: Model trained with different features"
            
            # Scaler hash kontrolü
            if model_name in self.scaler_hashes:
                current_scaler_hash = hashlib.sha256(
                    json.dumps({
                        'type': type(scaler).__name__,
                        'features': len(features)
                    }, sort_keys=True).encode('utf-8')
                ).hexdigest()
                
                expected_scaler_hash = self.scaler_hashes[model_name]['hash']
                
                if current_scaler_hash != expected_scaler_hash:
                    return False, f"❌ Scaler type mismatch: Expected different scaler"
            
            return True, "✅ Features and scaler are compatible"
            
        except Exception as e:
            return False, f"❌ Compatibility check failed: {e}"
    
    def get_model_info(self, model_name: str = "default") -> Optional[Dict]:
        """
        Model bilgilerini al
        
        Args:
            model_name: Model adı
            
        Returns:
            Model bilgileri veya None
        """
        try:
            info = {}
            
            if model_name in self.feature_hashes:
                info['features'] = self.feature_hashes[model_name]
            
            if model_name in self.scaler_hashes:
                info['scaler'] = self.scaler_hashes[model_name]
            
            return info if info else None
            
        except Exception as e:
            logger.error(f"❌ Error getting model info: {e}")
            return None
    
    def list_models(self) -> List[str]:
        """
        Kayıtlı modelleri listele
        
        Returns:
            Model adları listesi
        """
        try:
            models = set()
            models.update(self.feature_hashes.keys())
            models.update(self.scaler_hashes.keys())
            return sorted(list(models))
        except Exception as e:
            logger.error(f"❌ Error listing models: {e}")
            return []
    
    def cleanup_old_hashes(self, days_old: int = 30):
        """
        Eski hash'leri temizle
        
        Args:
            days_old: Kaç günden eski hash'ler silinecek
        """
        try:
            cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
            
            # Feature hash'leri temizle
            to_remove = []
            for model_name, data in self.feature_hashes.items():
                try:
                    created_at = datetime.fromisoformat(data['created_at']).timestamp()
                    if created_at < cutoff_date:
                        to_remove.append(model_name)
                except:
                    to_remove.append(model_name)
            
            for model_name in to_remove:
                del self.feature_hashes[model_name]
                if model_name in self.scaler_hashes:
                    del self.scaler_hashes[model_name]
            
            if to_remove:
                self._save_hashes()
                logger.info(f"🧹 Cleaned up {len(to_remove)} old model hashes")
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up old hashes: {e}")


# Global instance
_validator = None

def get_feature_validator() -> FeatureValidator:
    """
    Global feature validator instance'ı al
    
    Returns:
        FeatureValidator instance
    """
    global _validator
    if _validator is None:
        _validator = FeatureValidator()
    return _validator


def validate_model_compatibility(features: Dict[str, float], 
                               scaler, model_name: str = "default") -> Tuple[bool, str]:
    """
    Model uyumluluğunu kontrol et (convenience function)
    
    Args:
        features: Feature sözlüğü
        scaler: Scaler objesi
        model_name: Model adı
        
    Returns:
        (is_compatible, message) tuple
    """
    validator = get_feature_validator()
    return validator.validate_compatibility(features, scaler, model_name)


def register_model_features(features: Dict[str, float], 
                          scaler, model_name: str = "default", 
                          version: str = "1.0"):
    """
    Model features ve scaler'ı kaydet (convenience function)
    
    Args:
        features: Feature sözlüğü
        scaler: Scaler objesi
        model_name: Model adı
        version: Versiyon
    """
    validator = get_feature_validator()
    validator.register_features(features, model_name, version)
    validator.register_scaler(scaler, model_name, version, len(features))


def check_feature_hash_consistency(features: Dict[str, float], 
                                 expected_hash: Optional[str] = None) -> Tuple[bool, str]:
    """
    Feature hash tutarlılığını kontrol et (convenience function)
    
    Args:
        features: Feature sözlüğü
        expected_hash: Beklenen hash
        
    Returns:
        (is_valid, message) tuple
    """
    validator = get_feature_validator()
    return validator.validate_feature_count(features, expected_hash)


# Test fonksiyonu
if __name__ == "__main__":
    # Test
    validator = FeatureValidator()
    
    # Test features
    test_features = {
        'mean_50': 1.5,
        'std_50': 0.5,
        'min_50': 0.8,
        'max_50': 2.5,
        'median_50': 1.4,
        'below_threshold_10': 3,
        'above_threshold_10': 7,
        'threshold_ratio_10': 0.7,
        'threshold_ratio_50': 0.6,
        'in_critical_zone_10': 2
    }
    
    print("🧪 Feature Validator Test")
    print("="*50)
    
    # Feature validation
    valid, msg = validator.validate_feature_count(test_features)
    print(f"Feature validation: {valid}")
    print(f"Message: {msg}")
    
    # Register features
    validator.register_features(test_features, "test_model", "1.0")
    print("\n✅ Test features registered")
    
    # List models
    models = validator.list_models()
    print(f"\n📝 Registered models: {models}")
    
    # Get model info
    info = validator.get_model_info("test_model")
    if info:
        print(f"\n📊 Test model info:")
        print(json.dumps(info, indent=2))
    
    print("\n✅ Test completed successfully!")
