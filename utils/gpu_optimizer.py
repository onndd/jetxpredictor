#!/usr/bin/env python3
"""
🚀 GPU Optimizer - JetX Project

GPU kullanımını optimize eden, monitoring ve configuration sağlayan modül.
TensorFlow, PyTorch ve CatBoost için GPU optimizasyonları içerir.
"""

import os
import subprocess
import sys
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# TensorFlow
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# CatBoost
try:
    import catboost
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPUOptimizer:
    """GPU optimizasyonları için ana sınıf"""
    
    def __init__(self):
        self.gpu_info = {}
        self.optimization_settings = {}
        self.performance_history = []
        
        # Başlangıçta GPU bilgisini al
        self.detect_gpus()
        
    def detect_gpus(self) -> Dict[str, Any]:
        """Mevcut GPU'ları tespit et"""
        gpu_info = {
            'tensorflow': self._detect_tensorflow_gpus(),
            'pytorch': self._detect_pytorch_gpus(),
            'catboost': self._detect_catboost_gpu_support()
        }
        
        self.gpu_info = gpu_info
        logger.info("GPU tespiti tamamlandı")
        self._print_gpu_info()
        
        return gpu_info
    
    def _detect_tensorflow_gpus(self) -> Dict[str, Any]:
        """TensorFlow GPU'larını tespit et"""
        if not TF_AVAILABLE:
            return {'available': False, 'error': 'TensorFlow not installed'}
        
        try:
            # GPU listesi
            gpus = tf.config.list_physical_devices('GPU')
            
            if not gpus:
                return {'available': False, 'gpus': []}
            
            gpu_details = []
            for i, gpu in enumerate(gpus):
                gpu_details.append({
                    'id': i,
                    'name': gpu.name,
                    'device_type': gpu.device_type
                })
            
            return {
                'available': True,
                'count': len(gpus),
                'gpus': gpu_details,
                'cuda_version': tf.sysconfig.get_build_info().get('cuda_version', 'Unknown'),
                'cudnn_version': tf.sysconfig.get_build_info().get('cudnn_version', 'Unknown')
            }
            
        except Exception as e:
            return {'available': False, 'error': str(e)}
    
    def _detect_pytorch_gpus(self) -> Dict[str, Any]:
        """PyTorch GPU'larını tespit et"""
        if not TORCH_AVAILABLE:
            return {'available': False, 'error': 'PyTorch not installed'}
        
        try:
            if not torch.cuda.is_available():
                return {'available': False, 'reason': 'CUDA not available'}
            
            gpu_count = torch.cuda.device_count()
            gpu_details = []
            
            for i in range(gpu_count):
                gpu_details.append({
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_total': torch.cuda.get_device_properties(i).total_memory,
                    'memory_total_gb': torch.cuda.get_device_properties(i).total_memory / (1024**3)
                })
            
            return {
                'available': True,
                'count': gpu_count,
                'gpus': gpu_details,
                'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version()
            }
            
        except Exception as e:
            return {'available': False, 'error': str(e)}
    
    def _detect_catboost_gpu_support(self) -> Dict[str, Any]:
        """CatBoost GPU desteğini tespit et"""
        if not CATBOOST_AVAILABLE:
            return {'available': False, 'error': 'CatBoost not installed'}
        
        try:
            # Test için geçici model
            from catboost import CatBoostClassifier
            temp_model = CatBoostClassifier(iterations=1, task_type='GPU', devices='0')
            temp_model.fit([[1]], [0], verbose=False)
            
            return {
                'available': True,
                'test_passed': True
            }
            
        except Exception as e:
            return {
                'available': False,
                'test_passed': False,
                'error': str(e)
            }
    
    def _print_gpu_info(self):
        """GPU bilgilerini yazdır"""
        print("🚀 GPU BİLGİLERİ")
        print("="*50)
        
        # TensorFlow
        tf_info = self.gpu_info.get('tensorflow', {})
        if tf_info.get('available', False):
            print(f"✅ TensorFlow GPU: {tf_info['count']} GPU bulundu")
            print(f"   CUDA Version: {tf_info.get('cuda_version', 'Unknown')}")
            print(f"   cuDNN Version: {tf_info.get('cudnn_version', 'Unknown')}")
        else:
            print(f"❌ TensorFlow GPU: Mevcut değil")
            if 'error' in tf_info:
                print(f"   Hata: {tf_info['error']}")
        
        # PyTorch
        pt_info = self.gpu_info.get('pytorch', {})
        if pt_info.get('available', False):
            print(f"✅ PyTorch GPU: {pt_info['count']} GPU bulundu")
            print(f"   CUDA Version: {pt_info.get('cuda_version', 'Unknown')}")
            for gpu in pt_info.get('gpus', []):
                print(f"   GPU {gpu['id']}: {gpu['name']} ({gpu['memory_total_gb']:.1f}GB)")
        else:
            print(f"❌ PyTorch GPU: Mevcut değil")
            if 'error' in pt_info:
                print(f"   Hata: {pt_info['error']}")
        
        # CatBoost
        cb_info = self.gpu_info.get('catboost', {})
        if cb_info.get('available', False):
            print(f"✅ CatBoost GPU: Destekleniyor")
        else:
            print(f"❌ CatBoost GPU: Desteklenmiyor")
            if 'error' in cb_info:
                print(f"   Hata: {cb_info['error']}")
        
        print("="*50)
    
    def optimize_tensorflow(self) -> Dict[str, Any]:
        """TensorFlow GPU optimizasyonları"""
        if not TF_AVAILABLE:
            return {'success': False, 'error': 'TensorFlow not available'}
        
        tf_info = self.gpu_info.get('tensorflow', {})
        if not tf_info.get('available', False):
            return {'success': False, 'error': 'No GPUs available'}
        
        try:
            # GPU memory growth ayarla
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Mixed precision training
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            
            # XLA JIT compilation (Colab'da bazen sorun yaratabilir)
            # os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
            
            optimizations = {
                'memory_growth': True,
                'mixed_precision': True,
                'xla_compilation': False  # Colab uyumluluğu için kapalı
            }
            
            self.optimization_settings['tensorflow'] = optimizations
            
            print("✅ TensorFlow GPU optimizasyonları uygulandı:")
            print(f"   - Memory Growth: Aktif")
            print(f"   - Mixed Precision: Aktif (float16)")
            print(f"   - XLA Compilation: Pasif (Colab uyumluluğu)")
            
            return {'success': True, 'optimizations': optimizations}
            
        except Exception as e:
            print(f"❌ TensorFlow optimizasyon hatası: {e}")
            return {'success': False, 'error': str(e)}
    
    def optimize_pytorch(self) -> Dict[str, Any]:
        """PyTorch GPU optimizasyonları"""
        if not TORCH_AVAILABLE:
            return {'success': False, 'error': 'PyTorch not available'}
        
        pt_info = self.gpu_info.get('pytorch', {})
        if not pt_info.get('available', False):
            return {'success': False, 'error': 'No GPUs available'}
        
        try:
            # GPU device ayarla
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
            # Memory management
            torch.cuda.empty_cache()
            
            # Benchmark modu (deterministic değil, daha hızlı)
            torch.backends.cudnn.benchmark = True
            
            # Automatic mixed precision
            scaler = torch.cuda.amp.GradScaler()
            
            optimizations = {
                'device': str(device),
                'benchmark': True,
                'amp_enabled': True,
                'memory_cache_cleared': True
            }
            
            self.optimization_settings['pytorch'] = optimizations
            
            print("✅ PyTorch GPU optimizasyonları uygulandı:")
            print(f"   - Device: {device}")
            print(f"   - cuDNN Benchmark: Aktif")
            print(f"   - Auto Mixed Precision: Aktif")
            print(f"   - Memory Cache: Temizlendi")
            
            return {'success': True, 'optimizations': optimizations, 'scaler': scaler}
            
        except Exception as e:
            print(f"❌ PyTorch optimizasyon hatası: {e}")
            return {'success': False, 'error': str(e)}
    
    def optimize_catboost(self) -> Dict[str, Any]:
        """CatBoost GPU optimizasyonları"""
        if not CATBOOST_AVAILABLE:
            return {'success': False, 'error': 'CatBoost not available'}
        
        cb_info = self.gpu_info.get('catboost', {})
        if not cb_info.get('available', False):
            return {'success': False, 'error': 'GPU not supported'}
        
        try:
            # GPU task type ve devices
            task_type = 'GPU'
            devices = '0'  # İlk GPU
            
            # Optimize CatBoost parameters for GPU
            gpu_params = {
                'task_type': task_type,
                'devices': devices,
                'bootstrap_type': 'Bernoulli',  # GPU için daha iyi
                'subsample': 0.8,
                'colsample_bylevel': 0.8,  # GPU için feature sampling
                'logging_level': 'Silent'  # GPU'da daha az log
            }
            
            self.optimization_settings['catboost'] = gpu_params
            
            print("✅ CatBoost GPU optimizasyonları uygulandı:")
            print(f"   - Task Type: {task_type}")
            print(f"   - Devices: {devices}")
            print(f"   - Bootstrap Type: Bernoulli")
            print(f"   - Subsample: 0.8")
            print(f"   - Feature Sampling: 0.8")
            
            return {'success': True, 'params': gpu_params}
            
        except Exception as e:
            print(f"❌ CatBoost optimizasyon hatası: {e}")
            return {'success': False, 'error': str(e)}
    
    def optimize_all(self) -> Dict[str, Any]:
        """Tüm framework'ler için GPU optimizasyonları"""
        print("🚀 TÜM FRAMEWORK'LER İÇİN GPU OPTİMİZASYONU")
        print("="*50)
        
        results = {}
        
        # TensorFlow
        print("📊 TensorFlow optimizasyonları...")
        results['tensorflow'] = self.optimize_tensorflow()
        
        # PyTorch
        print("🔥 PyTorch optimizasyonları...")
        results['pytorch'] = self.optimize_pytorch()
        
        # CatBoost
        print("🤖 CatBoost optimizasyonları...")
        results['catboost'] = self.optimize_catboost()
        
        # Özet
        successful_optimizations = sum(1 for r in results.values() if r.get('success', False))
        print(f"\n✅ {successful_optimizations}/{len(results)} framework optimize edildi")
        
        return results
    
    def get_gpu_memory_usage(self) -> Dict[str, Any]:
        """GPU memory kullanımını kontrol et"""
        memory_info = {}
        
        # PyTorch memory
        if TORCH_AVAILABLE and torch.cuda.is_available():
            memory_info['pytorch'] = {
                'allocated': torch.cuda.memory_allocated() / (1024**3),  # GB
                'cached': torch.cuda.memory_reserved() / (1024**3),  # GB
                'total': torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            }
        
        # TensorFlow memory (daha zor)
        if TF_AVAILABLE:
            try:
                # GPU memory info için workaround
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if lines:
                        used, total = map(int, lines[0].split(','))
                        memory_info['system'] = {
                            'used_gb': used / 1024,
                            'total_gb': total / 1024,
                            'free_gb': (total - used) / 1024
                        }
            except:
                pass
        
        return memory_info
    
    def monitor_gpu_usage(self, duration_seconds: int = 10) -> Dict[str, Any]:
        """GPU kullanımını belirli süre boyunca izle"""
        print(f"📊 GPU kullanımı {duration_seconds} saniye izleniyor...")
        
        measurements = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            memory_info = self.get_gpu_memory_usage()
            measurements.append({
                'timestamp': time.time() - start_time,
                'memory': memory_info
            })
            time.sleep(1)
        
        # İstatistikler
        if measurements:
            pytorch_memory = [m['memory'].get('pytorch', {}).get('allocated', 0) for m in measurements]
            if pytorch_memory:
                pytorch_stats = {
                    'avg': np.mean(pytorch_memory),
                    'max': np.max(pytorch_memory),
                    'min': np.min(pytorch_memory)
                }
            else:
                pytorch_stats = {}
            
            result = {
                'duration': duration_seconds,
                'measurement_count': len(measurements),
                'pytorch_memory_stats': pytorch_stats,
                'measurements': measurements[-5:]  # Son 5 ölçüm
            }
            
            print(f"✅ GPU izleme tamamlandı:")
            if pytorch_stats:
                print(f"   PyTorch Memory: Avg={pytorch_stats['avg']:.2f}GB, Max={pytorch_stats['max']:.2f}GB")
            
            return result
        
        return {'error': 'No measurements collected'}
    
    def save_optimization_report(self, filepath: str = 'gpu_optimization_report.json'):
        """Optimizasyon raporunu kaydet"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'gpu_info': self.gpu_info,
            'optimization_settings': self.optimization_settings,
            'current_memory_usage': self.get_gpu_memory_usage()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"✅ Optimizasyon raporu kaydedildi: {filepath}")
        except Exception as e:
            print(f"❌ Rapor kaydetme hatası: {e}")
    
    def print_performance_summary(self):
        """Performans özeti yazdır"""
        print("\n🚀 GPU PERFORMANS ÖZETİ")
        print("="*50)
        
        # GPU bilgisi
        print(f"📊 GPU Durumu:")
        for framework, info in self.gpu_info.items():
            status = "✅ Aktif" if info.get('available', False) else "❌ Pasif"
            print(f"   {framework.title()}: {status}")
        
        # Optimizasyonlar
        print(f"\n🔧 Optimizasyonlar:")
        for framework, settings in self.optimization_settings.items():
            print(f"   {framework.title()}: {len(settings)} ayar")
        
        # Memory usage
        memory_info = self.get_gpu_memory_usage()
        if memory_info:
            print(f"\n💾 Memory Kullanımı:")
            for system, info in memory_info.items():
                if isinstance(info, dict) and 'allocated' in info:
                    print(f"   {system.title()}: {info['allocated']:.2f}GB / {info.get('total', 0):.2f}GB")
        
        print("="*50)


# Global GPU optimizer instance
_gpu_optimizer = None

def get_gpu_optimizer() -> GPUOptimizer:
    """Global GPU optimizer instance al"""
    global _gpu_optimizer
    if _gpu_optimizer is None:
        _gpu_optimizer = GPUOptimizer()
    return _gpu_optimizer

def setup_gpu_optimization():
    """GPU optimizasyonlarını hızlıca kur"""
    optimizer = get_gpu_optimizer()
    return optimizer.optimize_all()

def print_gpu_status():
    """GPU durumunu yazdır"""
    optimizer = get_gpu_optimizer()
    optimizer.print_performance_summary()

# Colab için özel fonksiyonlar
def setup_colab_gpu_optimization():
    """Colab için özel GPU optimizasyonları"""
    print("🔥 GOOGLE COLAB GPU OPTİMİZASYONU")
    print("="*50)
    
    # Colab için environment variables
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow log'larını azalt
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # CUDA işlemlerini senkronize et (debug için)
    
    # GPU optimizer
    optimizer = get_gpu_optimizer()
    results = optimizer.optimize_all()
    
    # Colab için özel ayarlar
    if results.get('tensorflow', {}).get('success', False):
        print("✅ TensorFlow Colab GPU optimizasyonları aktif")
    
    if results.get('pytorch', {}).get('success', False):
        print("✅ PyTorch Colab GPU optimizasyonları aktif")
    
    if results.get('catboost', {}).get('success', False):
        print("✅ CatBoost Colab GPU optimizasyonları aktif")
    
    # Memory optimization
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✅ GPU memory cache temizlendi")
    
    print("="*50)
    print("🚀 Colab GPU optimizasyonları tamamlandı!")
    
    return results

if __name__ == "__main__":
    # Test
    optimizer = GPUOptimizer()
    optimizer.optimize_all()
    optimizer.print_performance_summary()
    optimizer.monitor_gpu_usage(5)
    optimizer.save_optimization_report()
