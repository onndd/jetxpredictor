"""
JetX Predictor - Threshold Manager
Merkezi threshold yönetimi sistemi
"Raporlama vs. Eylem" tutarsızlıklarını önlemek için tasarlandı
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ThresholdManager:
    """
    Merkezi Threshold Yönetimi Sistemi
    
    Bu sınıf tüm training ve evaluation threshold'larını tek yerden yönetir.
    Config dosyasından threshold değerlerini okur ve tutarlılık sağlar.
    
    Kullanım:
        tm = ThresholdManager()
        threshold = tm.get_threshold('detailed_metrics')  # 0.70
        penalty = tm.get_loss_penalty('false_positive_penalty')  # 5.0
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Threshold Manager'ı başlat
        
        Args:
            config_path: Config dosyasının yolu (varsayılan: config/config.yaml)
        """
        if config_path is None:
            # Proje kök dizinini bul
            current_dir = Path(__file__).parent
            project_root = current_dir.parent
            config_path = project_root / 'config' / 'config.yaml'
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Config dosyasını yükle"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Config validation
            required_sections = ['training_thresholds', 'loss_penalties', 'adaptive_weights']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Config dosyasında '{section}' bölümü bulunamadı!")
            
            return config
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Config dosyası bulunamadı: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Config dosyası parse hatası: {e}")
    
    def get_threshold(self, context: str) -> float:
        """
        Belirli bir context için threshold değeri al
        
        Args:
            context: Context adı (örn: 'detailed_metrics', 'virtual_bankroll')
            
        Returns:
            Threshold değeri (0.0-1.0 arası)
            
        Raises:
            KeyError: Bilinmeyen context
        """
        if 'training_thresholds' not in self.config:
            raise KeyError("Config dosyasında 'training_thresholds' bölümü bulunamadı!")
        
        if context not in self.config['training_thresholds']:
            available_contexts = list(self.config['training_thresholds'].keys())
            raise KeyError(f"'{context}' context'i bulunamadı! Mevcut context'ler: {available_contexts}")
        
        return float(self.config['training_thresholds'][context])
    
    def get_loss_penalty(self, penalty_type: str) -> float:
        """
        Belirli bir penalty türü için değer al
        
        Args:
            penalty_type: Penalty türü (örn: 'false_positive_penalty')
            
        Returns:
            Penalty değeri
        """
        if 'loss_penalties' not in self.config:
            raise KeyError("Config dosyasında 'loss_penalties' bölümü bulunamadı!")
        
        if penalty_type not in self.config['loss_penalties']:
            available_penalties = list(self.config['loss_penalties'].keys())
            raise KeyError(f"'{penalty_type}' penalty'si bulunamadı! Mevcut penalty'ler: {available_penalties}")
        
        return float(self.config['loss_penalties'][penalty_type])
    
    def get_adaptive_weight(self, weight_type: str) -> float:
        """
        Adaptive weight parametresi al
        
        Args:
            weight_type: Weight türü (örn: 'initial_false_positive_weight')
            
        Returns:
            Weight değeri
        """
        if 'adaptive_weights' not in self.config:
            raise KeyError("Config dosyasında 'adaptive_weights' bölümü bulunamadı!")
        
        if weight_type not in self.config['adaptive_weights']:
            available_weights = list(self.config['adaptive_weights'].keys())
            raise KeyError(f"'{weight_type}' weight'i bulunamadı! Mevcut weight'ler: {available_weights}")
        
        return float(self.config['adaptive_weights'][weight_type])
    
    def validate_consistency(self) -> Dict[str, Any]:
        """
        Threshold tutarlılığını doğrula
        
        Returns:
            Validation raporu
        """
        report = {
            'status': 'success',
            'warnings': [],
            'errors': [],
            'thresholds': {}
        }
        
        try:
            # Tüm threshold'ları kontrol et
            for context, value in self.config['training_thresholds'].items():
                if not 0.0 <= value <= 1.0:
                    report['errors'].append(f"Invalid threshold for '{context}': {value} (0.0-1.0 arası olmalı)")
                else:
                    report['thresholds'][context] = value
            
            # Mantıksal kontroller
            prod_default = self.get_threshold('production_default')
            model_checkpoint = self.get_threshold('model_checkpoint')
            
            if prod_default < model_checkpoint:
                report['warnings'].append(
                    f"Production threshold ({prod_default}) < model_checkpoint threshold ({model_checkpoint})"
                )
            
            # Loss penalty kontrolleri
            for penalty_type, value in self.config['loss_penalties'].items():
                if value <= 0:
                    report['errors'].append(f"Invalid penalty for '{penalty_type}': {value} (pozitif olmalı)")
            
            if report['errors']:
                report['status'] = 'error'
            elif report['warnings']:
                report['status'] = 'warning'
                
        except Exception as e:
            report['status'] = 'error'
            report['errors'].append(f"Validation hatası: {str(e)}")
        
        return report
    
    def get_all_thresholds(self) -> Dict[str, float]:
        """
        Tüm threshold'ları döndür
        
        Returns:
            Dictionary of all thresholds
        """
        return self.config['training_thresholds'].copy()
    
    def get_all_loss_penalties(self) -> Dict[str, float]:
        """
        Tüm loss penalty'leri döndür
        
        Returns:
            Dictionary of all loss penalties
        """
        return self.config['loss_penalties'].copy()
    
    def get_all_adaptive_weights(self) -> Dict[str, float]:
        """
        Tüm adaptive weight'leri döndür
        
        Returns:
            Dictionary of all adaptive weights
        """
        return self.config['adaptive_weights'].copy()
    
    def reload_config(self) -> None:
        """Config dosyasını yeniden yükle"""
        self.config = self._load_config()
    
    def print_summary(self) -> None:
        """Threshold özetini yazdır"""
        print("="*80)
        print("🎯 JETX THRESHOLD MANAGER - ÖZET")
        print("="*80)
        
        print(f"\n📋 TRAINING_THRESHOLDS:")
        for context, value in self.config['training_thresholds'].items():
            print(f"  {context:25}: {value:.2f}")
        
        print(f"\n💰 LOSS PENALTIES:")
        for penalty, value in self.config['loss_penalties'].items():
            if 'penalty' in penalty:
                print(f"  {penalty:25}: {value:.1f}x")
            else:
                print(f"  {penalty:25}: {value:.2f}")
        
        print(f"\n⚖️  ADAPTIVE WEIGHTS:")
        for weight, value in self.config['adaptive_weights'].items():
            print(f"  {weight:25}: {value:.2f}")
        
        # Validation raporu
        validation = self.validate_consistency()
        print(f"\n✅ VALIDATION STATUS: {validation['status'].upper()}")
        
        if validation['warnings']:
            print(f"⚠️  WARNINGS ({len(validation['warnings'])}):")
            for warning in validation['warnings']:
                print(f"    - {warning}")
        
        if validation['errors']:
            print(f"❌ ERRORS ({len(validation['errors'])}):")
            for error in validation['errors']:
                print(f"    - {error}")
        
        print("="*80)


# Global instance (singleton pattern)
_threshold_manager = None


def get_threshold_manager(config_path: Optional[str] = None) -> ThresholdManager:
    """
    Global Threshold Manager instance'ı al (singleton pattern)
    
    Args:
        config_path: Config dosyasının yolu
        
    Returns:
        ThresholdManager instance
    """
    global _threshold_manager
    
    if _threshold_manager is None:
        _threshold_manager = ThresholdManager(config_path)
    
    return _threshold_manager


# Convenience functions
def get_threshold(context: str) -> float:
    """
    Belirli bir context için threshold al (convenience function)
    
    Args:
        context: Context adı
        
    Returns:
        Threshold değeri
    """
    return get_threshold_manager().get_threshold(context)


def get_loss_penalty(penalty_type: str) -> float:
    """
    Belirli bir loss penalty al (convenience function)
    
    Args:
        penalty_type: Penalty türü
        
    Returns:
        Penalty değeri
    """
    return get_threshold_manager().get_loss_penalty(penalty_type)


def get_adaptive_weight(weight_type: str) -> float:
    """
    Belirli bir adaptive weight al (convenience function)
    
    Args:
        weight_type: Weight türü
        
    Returns:
        Weight değeri
    """
    return get_threshold_manager().get_adaptive_weight(weight_type)
