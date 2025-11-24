"""
Model Loader - Colab → Lokal Döngüsü için Optimize Edilmiş Model Yükleme

Google Colab'da eğitilen modelleri lokal projede otomatik tespit eder ve yükler.
Model versiyonlama ve doğrulama özellikleri içerir.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import zipfile
import shutil

logger = logging.getLogger(__name__)


class ModelLoader:
    """Model yükleme ve doğrulama sistemi"""
    
    def __init__(self, models_dir: str = "models"):
        """
        Args:
            models_dir: Model klasörü yolu
        """
        self.models_dir = models_dir
        self.models_base = Path(models_dir)
        
        # Model yapısı tanımları
        self.model_structures = {
            'progressive_nn': {
                'base_path': 'progressive_multiscale',
                'files': {
                    'models': ['model_window_{size}.h5' for size in [500, 250, 100, 50, 20]],
                    'scalers': ['scaler_window_{size}.pkl' for size in [500, 250, 100, 50, 20]],
                    'info': ['model_info.json']
                },
                'required': True
            },
            'progressive_transformer': { # YENİ: Monolitik script çıktısı
                'base_path': '',
                'files': {
                     'model': ['jetx_progressive_transformer.h5'],
                     'scaler': ['scaler_progressive_transformer.pkl'],
                     'info': ['model_info.json']
                },
                'required': False
            },
            'catboost': {
                'base_path': 'catboost_multiscale',
                'files': {
                    'regressors': ['regressor_window_{size}.cbm' for size in [500, 250, 100, 50, 20]],
                    'classifiers': ['classifier_window_{size}.cbm' for size in [500, 250, 100, 50, 20]],
                    'scalers': ['scaler_window_{size}.pkl' for size in [500, 250, 100, 50, 20]],
                    'info': ['model_info.json']
                },
                'required': False
            },
            'ultra': {
                'base_path': '',
                'files': {
                    'model': ['jetx_ultra_model.h5'],
                    'scaler': ['scaler_ultra.pkl'],
                    'info': ['ultra_model_info.json']
                },
                'required': False
            },
            'meta_model': {
                 'base_path': '',
                 'files': {
                     'model': ['meta_model.json'],
                     'info': ['meta_model_info.json']
                 },
                 'required': False
            }
        }
    
    def check_models(self) -> Dict[str, Dict]:
        """Tüm modellerin durumunu kontrol et"""
        status = {}
        
        for model_name, structure in self.model_structures.items():
            base_path = self.models_base / structure['base_path'] if structure['base_path'] else self.models_base
            
            model_status = {
                'name': model_name,
                'base_path': str(base_path),
                'exists': base_path.exists() if structure['base_path'] else True,
                'files': {},
                'complete': True,
                'missing_files': []
            }
            
            # Dosyaları kontrol et
            for file_type, file_patterns in structure['files'].items():
                found_files = []
                missing_files = []
                
                for pattern in file_patterns:
                    if '{size}' in pattern:
                        for size in [500, 250, 100, 50, 20]:
                            file_path = base_path / pattern.format(size=size)
                            if file_path.exists():
                                found_files.append(str(file_path))
                            else:
                                missing_files.append(str(file_path))
                    else:
                        file_path = base_path / pattern if structure['base_path'] else self.models_base / pattern
                        if file_path.exists():
                            found_files.append(str(file_path))
                        else:
                            missing_files.append(str(file_path))
                
                model_status['files'][file_type] = {
                    'found': found_files,
                    'missing': missing_files,
                    'count': len(found_files),
                    'total': len(file_patterns) * (5 if '{size}' in str(file_patterns) else 1)
                }
                
                if missing_files:
                    model_status['complete'] = False
                    model_status['missing_files'].extend(missing_files)
            
            status[model_name] = model_status
        
        return status
    
    def get_model_summary(self) -> Dict:
        """Model durum özeti"""
        status = self.check_models()
        
        summary = {
            'total_models': len(status),
            'complete_models': sum(1 for s in status.values() if s['complete']),
            'incomplete_models': sum(1 for s in status.values() if not s['complete']),
            'models': {}
        }
        
        for model_name, model_status in status.items():
            summary['models'][model_name] = {
                'complete': model_status['complete'],
                'files_found': sum(f['count'] for f in model_status['files'].values()),
                'files_missing': len(model_status['missing_files']),
                'missing_files': model_status['missing_files'][:5]
            }
        
        return summary

    def get_installation_guide(self) -> str:
        """Model kurulum rehberi"""
        summary = self.get_model_summary()
        
        guide = []
        guide.append("=" * 70)
        guide.append("📦 MODEL KURULUM REHBERİ")
        guide.append("=" * 70)
        guide.append("")
        guide.append(f"✅ Tamamlanmış Modeller: {summary['complete_models']}/{summary['total_models']}")
        guide.append(f"⚠️ Eksik Modeller: {summary['incomplete_models']}/{summary['total_models']}")
        guide.append("")
        guide.append("1. Google Colab'da 'jetx_PROGRESSIVE_TRAINING.py' dosyasını çalıştırın.")
        guide.append("2. Oluşan ZIP dosyasını indirin ve 'models/' klasörüne çıkarın.")
        guide.append("3. Diğer modeller (CatBoost, Ultra) için ilgili notebookları çalıştırın.")
        
        return "\n".join(guide)

# Global instance
_model_loader = None

def get_model_loader() -> ModelLoader:
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader
