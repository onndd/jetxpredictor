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
            'catboost_single': {
                'base_path': '',
                'files': {
                    'regressor': ['catboost_regressor.cbm'],
                    'classifier': ['catboost_classifier.cbm'],
                    'scaler': ['catboost_scaler.pkl']
                },
                'required': False
            },
            'autogluon': {
                'base_path': 'autogluon_model',
                'files': {
                    'model': ['autogluon_model/'],
                    'scaler': ['autogluon_scaler.pkl']
                },
                'required': False
            },
            'tabnet': {
                'base_path': '',
                'files': {
                    'model': ['tabnet_high_x.pkl'],
                    'scaler': ['tabnet_scaler.pkl']
                },
                'required': False
            }
        }
    
    def check_models(self) -> Dict[str, Dict]:
        """
        Tüm modellerin durumunu kontrol et
        
        Returns:
            Model durumları dictionary
        """
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
                    # Window size placeholder'ları değiştir
                    if '{size}' in pattern:
                        for size in [500, 250, 100, 50, 20]:
                            file_path = base_path / pattern.format(size=size)
                            if file_path.exists():
                                found_files.append(str(file_path))
                            else:
                                missing_files.append(str(file_path))
                    else:
                        file_path = base_path / pattern if structure['base_path'] else self.models_base / pattern
                        if file_path.exists() or (file_path.is_dir() and file_path.exists()):
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
                'missing_files': model_status['missing_files'][:5]  # İlk 5'i göster
            }
        
        return summary
    
    def validate_model_files(self, model_name: str) -> Tuple[bool, List[str]]:
        """
        Model dosyalarını doğrula
        
        Args:
            model_name: Model adı
            
        Returns:
            (is_valid, errors) tuple
        """
        if model_name not in self.model_structures:
            return False, [f"Bilinmeyen model: {model_name}"]
        
        status = self.check_models()
        model_status = status.get(model_name, {})
        
        if not model_status:
            return False, ["Model durumu bulunamadı"]
        
        errors = []
        
        # Dosya varlık kontrolü
        if not model_status['complete']:
            errors.append(f"Eksik dosyalar: {len(model_status['missing_files'])}")
            errors.extend(model_status['missing_files'][:10])  # İlk 10'u göster
        
        # Model info dosyası kontrolü
        if 'info' in model_status['files']:
            info_files = model_status['files']['info']['found']
            if info_files:
                try:
                    info_path = Path(info_files[0])
                    if info_path.exists():
                        with open(info_path, 'r') as f:
                            info = json.load(f)
                        # Temel alanları kontrol et
                        required_fields = ['model', 'version', 'date']
                        for field in required_fields:
                            if field not in info:
                                errors.append(f"Model info'da eksik alan: {field}")
                except Exception as e:
                    errors.append(f"Model info okunamadı: {e}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def extract_zip_to_models(self, zip_path: str, overwrite: bool = False) -> Dict:
        """
        ZIP dosyasını models/ klasörüne çıkar
        
        Args:
            zip_path: ZIP dosyası yolu
            overwrite: Mevcut dosyaları üzerine yaz
            
        Returns:
            Çıkarma sonuçları
        """
        results = {
            'success': False,
            'extracted_files': [],
            'skipped_files': [],
            'errors': []
        }
        
        try:
            zip_file = Path(zip_path)
            if not zip_file.exists():
                results['errors'].append(f"ZIP dosyası bulunamadı: {zip_path}")
                return results
            
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # ZIP içeriğini listele
                file_list = zip_ref.namelist()
                
                for file_name in file_list:
                    # ZIP içindeki dosya yolu
                    source_path = file_name
                    
                    # Hedef yol (models/ klasörüne)
                    if 'progressive_multiscale' in source_path:
                        target_path = self.models_base / 'progressive_multiscale' / Path(source_path).name
                    elif 'catboost_multiscale' in source_path:
                        target_path = self.models_base / 'catboost_multiscale' / Path(source_path).name
                    else:
                        target_path = self.models_base / Path(source_path).name
                    
                    # Klasör oluştur
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Dosya zaten varsa
                    if target_path.exists() and not overwrite:
                        results['skipped_files'].append(str(target_path))
                        continue
                    
                    # Dosyayı çıkar
                    try:
                        zip_ref.extract(source_path, target_path.parent)
                        # Dosya adını düzelt (ZIP içindeki klasör yapısı)
                        extracted_file = target_path.parent / Path(source_path).name
                        if extracted_file.exists() and extracted_file != target_path:
                            extracted_file.rename(target_path)
                        
                        results['extracted_files'].append(str(target_path))
                    except Exception as e:
                        results['errors'].append(f"Dosya çıkarılamadı {source_path}: {e}")
                
                results['success'] = len(results['errors']) == 0
                
        except Exception as e:
            results['errors'].append(f"ZIP çıkarma hatası: {e}")
        
        return results
    
    def get_installation_guide(self) -> str:
        """Model kurulum rehberi"""
        status = self.check_models()
        summary = self.get_model_summary()
        
        guide = []
        guide.append("=" * 70)
        guide.append("📦 MODEL KURULUM REHBERİ")
        guide.append("=" * 70)
        guide.append("")
        
        # Genel durum
        guide.append(f"✅ Tamamlanmış Modeller: {summary['complete_models']}/{summary['total_models']}")
        guide.append(f"⚠️ Eksik Modeller: {summary['incomplete_models']}/{summary['total_models']}")
        guide.append("")
        
        # Her model için durum
        for model_name, model_status in status.items():
            if model_status['complete']:
                guide.append(f"✅ {model_name.upper()}: Tamamlanmış")
            else:
                guide.append(f"⚠️ {model_name.upper()}: Eksik")
                guide.append(f"   Eksik dosyalar: {len(model_status['missing_files'])}")
                if model_status['missing_files']:
                    guide.append(f"   Örnek: {model_status['missing_files'][0]}")
            guide.append("")
        
        # Kurulum adımları
        guide.append("=" * 70)
        guide.append("📋 KURULUM ADIMLARI")
        guide.append("=" * 70)
        guide.append("")
        guide.append("1. Google Colab'da model eğitimi yapın")
        guide.append("   - notebooks/jetx_PROGRESSIVE_TRAINING_MULTISCALE.py")
        guide.append("   - notebooks/jetx_CATBOOST_TRAINING_MULTISCALE.py")
        guide.append("")
        guide.append("2. ZIP dosyasını indirin")
        guide.append("   - Colab otomatik olarak ZIP oluşturur ve indirir")
        guide.append("")
        guide.append("3. ZIP'i lokal projeye çıkarın")
        guide.append("   - ZIP'i açın")
        guide.append("   - İçeriği models/ klasörüne kopyalayın")
        guide.append("")
        guide.append("4. Model doğrulaması yapın")
        guide.append("   - Uygulamayı başlatın")
        guide.append("   - Sidebar'da model durumunu kontrol edin")
        guide.append("")
        
        return "\n".join(guide)


# Global instance
_model_loader = None

def get_model_loader() -> ModelLoader:
    """Global model loader instance'ı al"""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader

