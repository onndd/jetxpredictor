#!/usr/bin/env python3
"""
🚀 GPU Konfigürasyon Yardımcı Modülü

TensorFlow ve CatBoost için GPU optimizasyonu yapılandırması.
Tüm eğitim scriptlerinde kullanılmalı.
"""

import os
import warnings

def setup_tensorflow_gpu():
    """
    TensorFlow için GPU'yu optimize şekilde yapılandırır.
    
    Optimizasyonlar:
    - Memory growth: GPU belleğini dinamik kullanım
    - Mixed precision: float16 kullanarak 2x hız artışı
    - XLA: Derleyici optimizasyonları (güvenli mod)
    
    Returns:
        dict: GPU konfigürasyon bilgileri
    """
    import tensorflow as tf
    
    config_info = {
        'gpu_available': False,
        'gpu_count': 0,
        'memory_growth': False,
        'mixed_precision': False,
        'xla_enabled': True,
        'gpu_names': []
    }
    
    # XLA optimizasyonu - güvenli mod
    # Register overflow sorunlarını önlemek için auto_jit kullan
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow uyarılarını azalt
    
    # GPU'ları bul
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            config_info['gpu_available'] = True
            config_info['gpu_count'] = len(gpus)
            config_info['gpu_names'] = [gpu.name for gpu in gpus]
            
            # Memory growth ayarla - GPU belleğini dinamik kullan
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            config_info['memory_growth'] = True
            
            # Mixed precision training - 2x hız artışı
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            config_info['mixed_precision'] = True
            
            print(f"✅ TensorFlow GPU Konfigürasyonu:")
            print(f"   - GPU Sayısı: {len(gpus)}")
            print(f"   - GPU'lar: {[gpu.name for gpu in gpus]}")
            print(f"   - Memory Growth: Aktif")
            print(f"   - Mixed Precision: Aktif (float16)")
            print(f"   - XLA Optimizasyon: Aktif (auto_jit)")
            
        except RuntimeError as e:
            warnings.warn(f"GPU konfigürasyon hatası: {e}")
            print(f"⚠️ GPU mevcut ama konfigürasyon hatası: {e}")
            print(f"   CPU modunda devam ediliyor...")
    else:
        print(f"⚠️ GPU bulunamadı - CPU modunda çalışacak")
        print(f"   Eğitim süresi ~5-10x daha uzun olabilir")
    
    return config_info


def setup_catboost_gpu():
    """
    CatBoost için GPU parametrelerini döndürür.
    
    Returns:
        dict: CatBoost GPU parametreleri
    """
    import subprocess
    
    # GPU varlığını kontrol et
    gpu_available = False
    try:
        # nvidia-smi ile GPU kontrolü
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, 
                              text=True, 
                              timeout=2)
        gpu_available = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # nvidia-smi yoksa veya timeout olursa
        pass
    
    if gpu_available:
        print(f"✅ CatBoost GPU Konfigürasyonu:")
        print(f"   - GPU: Aktif")
        print(f"   - Task Type: GPU")
        return {
            'task_type': 'GPU',
            'devices': '0',  # İlk GPU'yu kullan
            'gpu_ram_part': 0.8  # GPU belleğinin %80'ini kullan
        }
    else:
        print(f"⚠️ CatBoost: GPU bulunamadı - CPU modunda çalışacak")
        return {
            'task_type': 'CPU'
        }


def print_gpu_status():
    """
    GPU durumunu detaylı şekilde yazdırır.
    """
    import subprocess
    
    print("\n" + "="*80)
    print("🔍 GPU DURUM KONTROLÜ")
    print("="*80)
    
    # nvidia-smi ile detaylı bilgi
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu', 
                               '--format=csv,noheader'],
                              capture_output=True,
                              text=True,
                              timeout=3)
        
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split('\n')
            for i, info in enumerate(gpu_info, 1):
                parts = info.split(', ')
                if len(parts) >= 5:
                    name, driver, total_mem, free_mem, temp = parts[:5]
                    print(f"\n📊 GPU {i}:")
                    print(f"   Model: {name}")
                    print(f"   Driver: {driver}")
                    print(f"   Toplam Bellek: {total_mem}")
                    print(f"   Boş Bellek: {free_mem}")
                    print(f"   Sıcaklık: {temp}°C")
        else:
            print("⚠️ nvidia-smi çalıştırılamadı")
            print("   GPU mevcut olmayabilir veya driver kurulu değil")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("⚠️ nvidia-smi bulunamadı")
        print("   GPU kontrolü yapılamıyor")
    
    print("="*80 + "\n")


if __name__ == '__main__':
    # Test
    print("Testing GPU configuration...")
    print_gpu_status()
    
    print("\n--- TensorFlow GPU Setup ---")
    tf_config = setup_tensorflow_gpu()
    print(f"TensorFlow Config: {tf_config}")
    
    print("\n--- CatBoost GPU Setup ---")
    cb_config = setup_catboost_gpu()
    print(f"CatBoost Config: {cb_config}")
