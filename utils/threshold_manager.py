#!/usr/bin/env python3
"""
JetX Predictor - Threshold Manager (v2.0)
Merkezi threshold yönetimi sistemi

GÜNCELLEME:
- 2 Modlu Yapı (Normal/Rolling) entegrasyonu.
- Varsayılan Değerler: Normal=0.85, Rolling=0.95
- Config dosyası olmasa bile bu değerlerin kullanılmasını garanti eder.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

# Varsayılan Sabitler (Config dosyası bulunamazsa devreye girer)
DEFAULT_THRESHOLDS = {
    'normal': 0.85,          # Normal Mod Eşiği
    'rolling': 0.95,         # Rolling Mod Eşiği
    'detailed_metrics': 0.85, 
    'production_default': 0.95,
    'model_checkpoint': 0.85
}

class ThresholdManager:
    """
    Merkezi Threshold Yönetimi Sistemi
    
    Bu sınıf tüm training ve evaluation threshold'larını tek yerden yönetir.
    
    Özellikler:
    - Config dosyasından okuma (varsa)
    - Varsayılan değerlere fallback (yoksa)
    - Normal ve Rolling modlar için özel getter metodları
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Threshold Manager'ı başlat
        
        Args:
            config_path: Config dosyasının yolu (varsayılan: config/config.yaml)
        """
        if config_path is None:
            # Proje kök dizinini bul (utils'in bir üst klasörü)
            current_dir = Path(__file__).parent
            project_root = current_dir.parent
            config_path = project_root / 'config' / 'config.yaml'
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Config dosyasını yükle veya varsayılanları oluştur"""
        config = {}
        
        # 1. Dosyadan yüklemeyi dene
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
                    if loaded_config:
                        config = loaded_config
            else:
                print(f"⚠️ ThresholdManager: Config dosyası bulunamadı ({self.config_path}). Varsayılanlar kullanılıyor.")
        except Exception as e:
            print(f"❌ ThresholdManager: Config okuma hatası: {e}. Varsayılanlar kullanılıyor.")

        # 2. Eksikleri tamamla (Fallback mekanizması)
        if 'training_thresholds' not in config:
            config['training_thresholds'] = {}
            
        if 'loss_penalties' not in config:
            config['loss_penalties'] = {
                'false_positive_penalty': 5.0,
                'false_negative_penalty': 2.0
            }
            
        if 'adaptive_weights' not in config:
            config['adaptive_weights'] = {
                'initial_false_positive_weight': 2.0
            }

        # Kritik thresholdları garantile
        for key, val in DEFAULT_THRESHOLDS.items():
            if key not in config['training_thresholds']:
                config['training_thresholds'][key] = val
        
        return config
    
    def get_threshold(self, context: str) -> float:
        """
        Belirli bir context için threshold değeri al
        """
        # Alias desteği
        if context == 'normal_mode': context = 'normal'
        if context == 'rolling_mode': context = 'rolling'

        if context in self.config['training_thresholds']:
            return float(self.config['training_thresholds'][context])
        
        print(f"⚠️ Bilinmeyen context '{context}', varsayılan 0.85 döndürülüyor.")
        return 0.85
    
    def get_normal_threshold(self) -> float:
        """Normal Mod eşiğini döndür (0.85)"""
        return self.get_threshold('normal')

    def get_rolling_threshold(self) -> float:
        """Rolling Mod eşiğini döndür (0.95)"""
        return self.get_threshold('rolling')

    def get_loss_penalty(self, penalty_type: str) -> float:
        """Belirli bir penalty türü için değer al"""
        return float(self.config.get('loss_penalties', {}).get(penalty_type, 1.0))
    
    def get_adaptive_weight(self, weight_type: str) -> float:
        """Adaptive weight parametresi al"""
        return float(self.config.get('adaptive_weights', {}).get(weight_type, 1.0))
    
    def validate_consistency(self) -> Dict[str, Any]:
        """Threshold tutarlılığını doğrula"""
        report = {
            'status': 'success',
            'warnings': [],
            'errors': [],
            'thresholds': self.config['training_thresholds']
        }
        
        normal = self.get_normal_threshold()
        rolling = self.get_rolling_threshold()
        
        if normal > rolling:
            report['errors'].append(f"Mantık Hatası: Normal mod ({normal}) > Rolling mod ({rolling}) olamaz!")
            report['status'] = 'error'
            
        if normal < 0.5:
            report['warnings'].append(f"Uyarı: Normal mod eşiği ({normal}) çok düşük riskli olabilir.")

        return report
    
    def get_all_thresholds(self) -> Dict[str, float]:
        """Tüm threshold'ları döndür"""
        return self.config['training_thresholds'].copy()
    
    def get_all_loss_penalties(self) -> Dict[str, float]:
        """Tüm loss penalty'leri döndür"""
        return self.config.get('loss_penalties', {}).copy()
    
    def get_all_adaptive_weights(self) -> Dict[str, float]:
        """Tüm adaptive weight'leri döndür"""
        return self.config.get('adaptive_weights', {}).copy()
    
    def reload_config(self) -> None:
        """Config dosyasını yeniden yükle"""
        self.config = self._load_config()
    
    def print_summary(self) -> None:
        """Threshold özetini yazdır"""
        print("="*80)
        print("🎯 JETX THRESHOLD MANAGER - ÖZET (v2.0)")
        print("="*80)
        
        print(f"\n📋 MODLAR:")
        print(f"  Normal Mod:       {self.get_normal_threshold():.2f}")
        print(f"  Rolling Mod:      {self.get_rolling_threshold():.2f}")
        
        print(f"\n📋 DİĞER AYARLAR:")
        for context, value in self.config['training_thresholds'].items():
            if context not in ['normal', 'rolling']:
                print(f"  {context:25}: {value:.2f}")
        
        validation = self.validate_consistency()
        if validation['status'] != 'success':
             print(f"\n⚠️ DURUM: {validation['status'].upper()}")
             for err in validation['errors']: print(f"  ❌ {err}")
             for warn in validation['warnings']: print(f"  ⚠️ {warn}")
        
        print("="*80)


# Global instance (Singleton)
_threshold_manager = None

def get_threshold_manager(config_path: Optional[str] = None) -> ThresholdManager:
    """Global Threshold Manager instance'ı al"""
    global _threshold_manager
    if _threshold_manager is None:
        _threshold_manager = ThresholdManager(config_path)
    return _threshold_manager

# Yardımcı Fonksiyonlar (Convenience functions)
def get_threshold(context: str) -> float:
    return get_threshold_manager().get_threshold(context)

def get_loss_penalty(penalty_type: str) -> float:
    return get_threshold_manager().get_loss_penalty(penalty_type)

def get_adaptive_weight(weight_type: str) -> float:
    return get_threshold_manager().get_adaptive_weight(weight_type)
