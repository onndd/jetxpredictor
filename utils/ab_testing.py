"""
A/B Testing Sistemi

Farklı modelleri veya versiyonları karşılaştırmak için A/B testi yapar.
Test sonuçlarını kaydeder ve istatistiksel analiz yapar.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ABTestResult:
    """A/B test sonucu"""
    test_id: str
    model_a: str
    model_b: str
    start_date: str
    end_date: Optional[str]
    total_predictions: int
    predictions_a: int
    predictions_b: int
    wins_a: int
    wins_b: int
    losses_a: int
    losses_b: int
    accuracy_a: float
    accuracy_b: float
    roi_a: float
    roi_b: float
    is_active: bool
    winner: Optional[str]  # 'A', 'B', veya None (berabere)
    confidence_level: float
    p_value: Optional[float]


class ABTestManager:
    """A/B test yönetimi"""
    
    def __init__(self, results_path: str = "data/ab_test_results.json"):
        """
        Args:
            results_path: Test sonuçları dosyası yolu
        """
        self.results_path = results_path
        self.tests = self._load_tests()
        self.active_tests = {}
        
        # Results dizinini oluştur
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    def _load_tests(self) -> Dict:
        """Test sonuçlarını yükle"""
        if os.path.exists(self.results_path):
            try:
                with open(self.results_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Test sonuçları yükleme hatası: {e}")
                return {}
        return {}
    
    def _save_tests(self):
        """Test sonuçlarını kaydet"""
        try:
            with open(self.results_path, 'w', encoding='utf-8') as f:
                json.dump(self.tests, f, indent=2, ensure_ascii=False)
            logger.info("A/B test sonuçları kaydedildi")
        except Exception as e:
            logger.error(f"Test sonuçları kaydetme hatası: {e}")
    
    def create_test(
        self,
        test_name: str,
        model_a: str,
        model_b: str,
        split_ratio: float = 0.5,
        min_samples: int = 100
    ) -> str:
        """
        Yeni A/B testi oluştur
        
        Args:
            test_name: Test adı
            model_a: A modeli (kontrol grubu)
            model_b: B modeli (test grubu)
            split_ratio: B grubuna giden oran (0.0-1.0)
            min_samples: Minimum örnek sayısı
            
        Returns:
            Test ID
        """
        test_id = f"{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        test_config = {
            'test_id': test_id,
            'test_name': test_name,
            'model_a': model_a,
            'model_b': model_b,
            'split_ratio': split_ratio,
            'min_samples': min_samples,
            'created_at': datetime.now().isoformat(),
            'is_active': True,
            'results': {
                'total_predictions': 0,
                'predictions_a': 0,
                'predictions_b': 0,
                'wins_a': 0,
                'wins_b': 0,
                'losses_a': 0,
                'losses_b': 0,
                'correct_a': 0,
                'correct_b': 0,
                'roi_a': 0.0,
                'roi_b': 0.0
            }
        }
        
        self.tests[test_id] = test_config
        self.active_tests[test_id] = test_config
        self._save_tests()
        
        logger.info(f"A/B testi oluşturuldu: {test_id} ({model_a} vs {model_b})")
        return test_id
    
    def record_prediction(
        self,
        test_id: str,
        model_used: str,
        predicted_value: float,
        actual_value: float,
        confidence: float,
        was_correct: bool
    ):
        """
        Tahmin sonucunu kaydet
        
        Args:
            test_id: Test ID
            model_used: Kullanılan model ('A' veya 'B')
            predicted_value: Tahmin edilen değer
            actual_value: Gerçek değer
            confidence: Güven skoru
            was_correct: Doğru mu?
        """
        if test_id not in self.tests:
            logger.error(f"Test bulunamadı: {test_id}")
            return
        
        test = self.tests[test_id]
        if not test['is_active']:
            logger.warning(f"Test aktif değil: {test_id}")
            return
        
        results = test['results']
        results['total_predictions'] += 1
        
        if model_used == 'A':
            results['predictions_a'] += 1
            if was_correct:
                results['wins_a'] += 1
                results['correct_a'] += 1
            else:
                results['losses_a'] += 1
            
            # ROI hesapla (basit)
            if predicted_value >= 1.5 and actual_value >= 1.5:
                results['roi_a'] += 0.5  # Kazanç
            elif predicted_value >= 1.5 and actual_value < 1.5:
                results['roi_a'] -= 1.0  # Kayıp
        
        elif model_used == 'B':
            results['predictions_b'] += 1
            if was_correct:
                results['wins_b'] += 1
                results['correct_b'] += 1
            else:
                results['losses_b'] += 1
            
            # ROI hesapla
            if predicted_value >= 1.5 and actual_value >= 1.5:
                results['roi_b'] += 0.5
            elif predicted_value >= 1.5 and actual_value < 1.5:
                results['roi_b'] -= 1.0
        
        self._save_tests()
    
    def get_test_results(self, test_id: str) -> Optional[Dict]:
        """Test sonuçlarını al"""
        if test_id not in self.tests:
            return None
        
        test = self.tests[test_id]
        results = test['results']
        
        # Accuracy hesapla
        accuracy_a = (results['correct_a'] / results['predictions_a'] * 100) if results['predictions_a'] > 0 else 0.0
        accuracy_b = (results['correct_b'] / results['predictions_b'] * 100) if results['predictions_b'] > 0 else 0.0
        
        # ROI normalize et
        roi_a = (results['roi_a'] / results['predictions_a'] * 100) if results['predictions_a'] > 0 else 0.0
        roi_b = (results['roi_b'] / results['predictions_b'] * 100) if results['predictions_b'] > 0 else 0.0
        
        # İstatistiksel test (chi-square)
        p_value = None
        confidence_level = 0.0
        
        if results['predictions_a'] >= 30 and results['predictions_b'] >= 30:
            # Chi-square test
            contingency = np.array([
                [results['wins_a'], results['losses_a']],
                [results['wins_b'], results['losses_b']]
            ])
            
            try:
                chi2, p_value = stats.chi2_contingency(contingency)[:2]
                confidence_level = (1 - p_value) * 100 if p_value else 0.0
            except:
                pass
        
        # Kazanan belirle
        winner = None
        if accuracy_b > accuracy_a + 2.0 and roi_b > roi_a:  # %2+ fark ve ROI üstünlüğü
            winner = 'B'
        elif accuracy_a > accuracy_b + 2.0 and roi_a > roi_b:
            winner = 'A'
        
        return {
            'test_id': test_id,
            'test_name': test['test_name'],
            'model_a': test['model_a'],
            'model_b': test['model_b'],
            'total_predictions': results['total_predictions'],
            'predictions_a': results['predictions_a'],
            'predictions_b': results['predictions_b'],
            'wins_a': results['wins_a'],
            'wins_b': results['wins_b'],
            'losses_a': results['losses_a'],
            'losses_b': results['losses_b'],
            'accuracy_a': round(accuracy_a, 2),
            'accuracy_b': round(accuracy_b, 2),
            'roi_a': round(roi_a, 2),
            'roi_b': round(roi_b, 2),
            'is_active': test['is_active'],
            'winner': winner,
            'confidence_level': round(confidence_level, 2),
            'p_value': round(p_value, 4) if p_value else None,
            'created_at': test['created_at']
        }
    
    def stop_test(self, test_id: str) -> bool:
        """Testi durdur"""
        if test_id not in self.tests:
            logger.error(f"Test bulunamadı: {test_id}")
            return False
        
        self.tests[test_id]['is_active'] = False
        if test_id in self.active_tests:
            del self.active_tests[test_id]
        
        self._save_tests()
        logger.info(f"Test durduruldu: {test_id}")
        return True
    
    def get_active_tests(self) -> List[Dict]:
        """Aktif testleri listele"""
        active = []
        for test_id, test in self.tests.items():
            if test.get('is_active', False):
                results = self.get_test_results(test_id)
                if results:
                    active.append(results)
        return active
    
    def get_all_tests(self) -> List[Dict]:
        """Tüm testleri listele"""
        all_tests = []
        for test_id in self.tests:
            results = self.get_test_results(test_id)
            if results:
                all_tests.append(results)
        return all_tests
    
    def select_model_for_prediction(self, test_id: str) -> str:
        """
        Tahmin için model seç (A/B split)
        
        Args:
            test_id: Test ID
            
        Returns:
            'A' veya 'B'
        """
        if test_id not in self.tests:
            return 'A'  # Default
        
        test = self.tests[test_id]
        if not test.get('is_active', False):
            return 'A'
        
        # Random split
        split_ratio = test.get('split_ratio', 0.5)
        return 'B' if np.random.random() < split_ratio else 'A'


# Global instance
_ab_test_manager = None

def get_ab_test_manager() -> ABTestManager:
    """Global AB test manager instance'ı al"""
    global _ab_test_manager
    if _ab_test_manager is None:
        _ab_test_manager = ABTestManager()
    return _ab_test_manager

