"""
🔍 FEATURE MONITOR & DRIFT DETECTION SYSTEM

Bu modül, JetX Predictor'da feature drift tespiti ve monitoring yapar:
1. Feature dağılımı drift detection
2. Model performance monitoring
3. Anomaly detection
4. Alert sistemi

KULLANIM:
- Production ortamında real-time monitoring
- Model degradation tespiti
- Feature engineering hatalarını erken uyarı
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import json
import os
from collections import deque
import warnings

logger = logging.getLogger(__name__)


class FeatureDriftDetector:
    """
    Feature drift detection sınıfı
    
    - Feature dağılımlarındaki değişimleri tespit eder
    - Model performance degradation'i izler
    - Anomalileri belirler
    """
    
    def __init__(self, 
                 window_size: int = 1000,
                 drift_threshold: float = 0.3,
                 performance_threshold: float = 0.1):
        """
        Feature drift detector'ı başlat
        
        Args:
            window_size: Monitoring pencere boyutu
            drift_threshold: Drift tespit eşiği (0-1 arası)
            performance_threshold: Performance degradation eşiği
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        
        # Feature statistics
        self.baseline_stats = {}
        self.current_window = deque(maxlen=window_size)
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        self.baseline_performance = None
        
        # Alert tracking
        self.alerts = []
        self.last_alert_time = None
        
        # Monitoring state
        self.is_monitoring = False
        self.start_time = None
        
        logger.info(f"Feature drift detector başlatıldı - window_size: {window_size}")
    
    def set_baseline(self, features: Dict[str, float], performance: Optional[float] = None):
        """
        Baseline feature statistics'ini ayarla
        
        Args:
            features: Baseline feature sözlüğü
            performance: Baseline performance (opsiyonel)
        """
        try:
            feature_array = np.array(list(features.values()))
            
            self.baseline_stats = {
                'means': np.mean(feature_array),
                'stds': np.std(feature_array),
                'feature_names': list(features.keys()),
                'feature_values': {k: v for k, v in features.items()},
                'timestamp': datetime.now().isoformat(),
                'count': len(features)
            }
            
            if performance is not None:
                self.baseline_performance = performance
            
            logger.info(f"Baseline ayarlandı - {len(features)} feature")
            
        except Exception as e:
            logger.error(f"Baseline ayarlama hatası: {e}")
            raise
    
    def add_observation(self, features: Dict[str, float], performance: Optional[float] = None):
        """
        Yeni gözlem ekle ve drift kontrolü yap
        
        Args:
            features: Feature sözlüğü
            performance: Performance değeri (opsiyonel)
        """
        try:
            # Window'a ekle
            self.current_window.append(features)
            
            if performance is not None:
                self.performance_history.append({
                    'performance': performance,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Monitoring başlat
            if not self.is_monitoring and len(self.current_window) >= self.window_size:
                self.is_monitoring = True
                self.start_time = datetime.now()
                logger.info("Monitoring başlatıldı")
            
            # Drift kontrolü
            if self.is_monitoring:
                drift_score = self._calculate_drift_score()
                perf_drift = self._check_performance_drift()
                
                # Alert kontrolü
                self._check_alerts(drift_score, perf_drift)
                
        except Exception as e:
            logger.error(f"Gözlem ekleme hatası: {e}")
    
    def _calculate_drift_score(self) -> float:
        """
        Drift score hesapla (0-1 arası)
        
        Returns:
            Drift score (0 = no drift, 1 = severe drift)
        """
        try:
            if not self.baseline_stats or len(self.current_window) < 100:
                return 0.0
            
            # Window'daki feature'ları al
            window_features = list(self.current_window)
            # Sadece baseline'da olan feature'ları al (uyumluluk için)
            baseline_keys = self.baseline_stats['feature_names']
            
            # Dict listesinden array oluşturma (eksik anahtarları 0 ile doldurarak)
            window_array = []
            for f in window_features:
                row = [f.get(k, 0.0) for k in baseline_keys]
                window_array.append(row)
            window_array = np.array(window_array)
            
            # Baseline ile karşılaştır
            current_means = np.mean(window_array, axis=0)
            # baseline_stats['feature_values'] dict'inden array sırasına göre değerleri çek
            baseline_means_array = np.array([self.baseline_stats['feature_values'].get(k, 0.0) for k in baseline_keys])
            
            # Basit Z-Score mantığı ile sapma hesapla (Std yoksa 1 kabul et)
            # Not: baseline_stats['stds'] tek bir float değerdi, feature bazlı std saklanmalıydı.
            # Burada basitleştirilmiş global std kullanıyoruz.
            global_std = self.baseline_stats.get('stds', 1.0)
            if global_std == 0: global_std = 1.0
            
            drift_scores = np.abs(current_means - baseline_means_array) / global_std
            mean_drift = np.mean(np.minimum(drift_scores, 1.0)) # 1.0 ile caple
            
            return float(mean_drift)
                
        except Exception as e:
            logger.error(f"Drift score hesaplama hatası: {e}")
            return 0.0
    
    def _check_performance_drift(self) -> Optional[Dict[str, Any]]:
        """
        Performance drift kontrolü
        
        Returns:
            Performance drift bilgisi veya None
        """
        try:
            if not self.baseline_performance or len(self.performance_history) < 10:
                return None
            
            # Son 10 performance değerini al
            recent_performances = [p['performance'] for p in list(self.performance_history)[-10:]]
            avg_recent = np.mean(recent_performances)
            
            # Drift hesapla
            performance_drop = self.baseline_performance - avg_recent
            relative_drop = performance_drop / self.baseline_performance if self.baseline_performance > 0 else 0
            
            if relative_drop > self.performance_threshold:
                return {
                    'type': 'performance_drift',
                    'baseline': self.baseline_performance,
                    'current': avg_recent,
                    'drop': performance_drop,
                    'relative_drop': relative_drop,
                    'severity': 'high' if relative_drop > 0.2 else 'medium'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Performance drift kontrolü hatası: {e}")
            return None
    
    def _check_alerts(self, drift_score: float, performance_drift: Optional[Dict[str, Any]]):
        """
        Alert kontrolü ve yönetimi
        """
        try:
            current_time = datetime.now()
            
            # Drift alert'i
            if drift_score > self.drift_threshold:
                alert = {
                    'type': 'feature_drift',
                    'severity': 'high' if drift_score > 0.6 else 'medium',
                    'drift_score': drift_score,
                    'timestamp': current_time.isoformat(),
                    'message': f"Feature drift detected: {drift_score:.3f}"
                }
                self._add_alert(alert)
            
            # Performance drift alert'i
            if performance_drift:
                alert = {
                    'type': performance_drift['type'],
                    'severity': performance_drift['severity'],
                    'performance_drop': performance_drift['relative_drop'],
                    'timestamp': current_time.isoformat(),
                    'message': f"Performance drift: {performance_drift['relative_drop']:.1%}"
                }
                self._add_alert(alert)
            
        except Exception as e:
            logger.error(f"Alert kontrolü hatası: {e}")
    
    def _add_alert(self, alert: Dict[str, Any]):
        """Alert ekle"""
        try:
            self.alerts.append(alert)
            self.last_alert_time = datetime.now()
            logger.warning(f"ALERT: {alert['message']}")
            if len(self.alerts) > 50:
                self.alerts = self.alerts[-50:]
        except Exception as e:
            logger.error(f"Alert ekleme hatası: {e}")
    
    def get_feature_health_status(self) -> Dict[str, Any]:
        """
        Feature health status'u al
        """
        try:
            if not self.is_monitoring:
                return {'status': 'not_monitoring'}
            
            drift_score = self._calculate_drift_score()
            
            if drift_score < 0.1: health_status = 'healthy'
            elif drift_score < 0.3: health_status = 'warning'
            else: health_status = 'critical'
            
            return {
                'status': health_status,
                'drift_score': drift_score,
                'window_size': len(self.current_window),
                'last_alert': self.last_alert_time.isoformat() if self.last_alert_time else None,
                'monitoring_duration': str(datetime.now() - self.start_time) if self.start_time else None
            }
            
        except Exception as e:
            logger.error(f"Feature health status hatası: {e}")
            return {'error': str(e), 'status': 'error'}


class FeatureMonitor:
    """
    Global feature monitoring sistemi
    """
    
    def __init__(self):
        """Global feature monitor başlat"""
        self.detectors = {}
        self.global_alerts = []
        self.start_time = datetime.now()
        logger.info("Global feature monitor başlatıldı")
    
    def register_detector(self, name: str, detector: FeatureDriftDetector):
        """Detector kaydet"""
        self.detectors[name] = detector
        logger.info(f"Detector kaydedildi: {name}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Sistem health durumunu al"""
        try:
            health_summary = {
                'system_status': 'healthy',
                'total_detectors': len(self.detectors),
                'monitoring_detectors': 0,
                'total_alerts': 0,
                'critical_alerts': 0,
                'uptime': str(datetime.now() - self.start_time),
                'detectors': {}
            }
            
            for name, detector in self.detectors.items():
                detector_health = detector.get_feature_health_status()
                health_summary['detectors'][name] = detector_health
                
                if detector_health.get('status') == 'critical':
                    health_summary['system_status'] = 'critical'
                elif detector_health.get('status') == 'warning' and health_summary['system_status'] == 'healthy':
                    health_summary['system_status'] = 'warning'
                
                if detector.is_monitoring:
                    health_summary['monitoring_detectors'] += 1
                
                health_summary['total_alerts'] += len(detector.alerts)
                critical_alerts = len([a for a in detector.alerts if a.get('severity') == 'high'])
                health_summary['critical_alerts'] += critical_alerts
            
            return health_summary
            
        except Exception as e:
            logger.error(f"Sistem health kontrolü hatası: {e}")
            return {'error': str(e), 'system_status': 'error'}


# Global instance
_global_monitor = None

def get_feature_monitor() -> FeatureMonitor:
    """Global feature monitor instance'ı al"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = FeatureMonitor()
    return _global_monitor


def create_drift_detector(name: str, **kwargs) -> FeatureDriftDetector:
    """Yeni drift detector oluştur"""
    detector = FeatureDriftDetector(**kwargs)
    monitor = get_feature_monitor()
    monitor.register_detector(name, detector)
    logger.info(f"Drift detector oluşturuldu: {name}")
    return detector


def monitor_features(name: str, features: Dict[str, float], performance: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Feature monitoring yap (convenience function)"""
    try:
        monitor = get_feature_monitor()
        if name in monitor.detectors:
            detector = monitor.detectors[name]
            detector.add_observation(features, performance)
            return detector.get_feature_health_status()
        else:
            detector = create_drift_detector(name)
            detector.add_observation(features, performance)
            return detector.get_feature_health_status()
    except Exception as e:
        logger.error(f"Feature monitoring hatası: {e}")
        return None


def get_system_health() -> Dict[str, Any]:
    """Sistem health durumunu al (convenience function)"""
    try:
        monitor = get_feature_monitor()
        return monitor.get_system_health()
    except Exception as e:
        logger.error(f"Sistem health kontrolü hatası: {e}")
        return {'error': str(e), 'system_status': 'error'}

if __name__ == "__main__":
    # Test
    detector = create_drift_detector("test_detector")
    baseline_features = {'mean_50': 1.5, 'std_50': 0.8}
    detector.set_baseline(baseline_features, performance=0.75)
    
    for i in range(10):
        test_features = {'mean_50': 1.5 + np.random.normal(0, 0.1), 'std_50': 0.8 + np.random.normal(0, 0.05)}
        detector.add_observation(test_features, performance=0.75 - i * 0.02)
        
    print(f"Health Status: {detector.get_feature_health_status()}")
