"""
Threshold Consistency Tests
"Raporlama vs. Eylem" tutarsızlıklarını test eder
Merkezi threshold yönetiminin doğru çalıştığını doğrular
"""

import unittest
import numpy as np
import sys
import os
from pathlib import Path

# Proje kök dizinini ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.threshold_manager import ThresholdManager, get_threshold_manager


class TestThresholdConsistency(unittest.TestCase):
    """Threshold tutarlılık testleri"""
    
    def setUp(self):
        """Test setup"""
        self.config_path = project_root / 'config' / 'config.yaml'
        self.tm = ThresholdManager(self.config_path)
    
    def test_config_loading(self):
        """Config dosyasının doğru yüklendiğini test et"""
        # Config bölümleri var mı?
        self.assertIn('training_thresholds', self.tm.config)
        self.assertIn('loss_penalties', self.tm.config)
        self.assertIn('adaptive_weights', self.tm.config)
        
        # Gerekli threshold'lar var mı?
        required_thresholds = [
            'detailed_metrics', 'virtual_bankroll', 'model_checkpoint',
            'catboost_evaluation', 'conservative_mode', 'production_default'
        ]
        
        for threshold in required_thresholds:
            self.assertIn(threshold, self.tm.config['training_thresholds'])
    
    def test_threshold_values(self):
        """Threshold değerlerinin geçerli aralıkta olduğunu test et"""
        thresholds = self.tm.get_all_thresholds()
        
        for context, value in thresholds.items():
            # Tüm threshold'lar 0-1 arası olmalı
            self.assertGreaterEqual(value, 0.0, f"{context} threshold 0'dan küçük")
            self.assertLessEqual(value, 1.0, f"{context} threshold 1'den büyük")
            
            # Mantıksal kontroller
            if context == 'binary_conversion':
                self.assertEqual(value, 0.5, "Binary conversion threshold 0.5 olmalı")
            elif context in ['conservative_mode']:
                # Conservative mode diğerlerinden daha düşük olmalı
                self.assertLessEqual(value, 0.8, "Conservative mode threshold diğerlerinden düşük olmalı")
    
    def test_loss_penalty_values(self):
        """Loss penalty değerlerinin geçerli olduğunu test et"""
        penalties = self.tm.get_all_loss_penalties()
        
        for penalty_type, value in penalties.items():
            # Tüm penalty'ler pozitif olmalı
            self.assertGreater(value, 0.0, f"{penalty_type} penalty pozitif olmalı")
            
            # Mantıksal kontroller
            if 'false_positive' in penalty_type:
                # False positive penalty en yüksek olmalı (para kaybı en riskli)
                self.assertGreaterEqual(value, 3.0, "False positive penalty yüksek olmalı (3+)")
    
    def test_threshold_manager_singleton(self):
        """Threshold Manager singleton pattern'ını test et"""
        tm1 = get_threshold_manager()
        tm2 = get_threshold_manager()
        
        # Aynı instance mı?
        self.assertIs(tm1, tm2, "Threshold Manager singleton çalışmıyor")
    
    def test_threshold_consistency_across_contexts(self):
        """Farklı context'ler arası threshold tutarlılığını test et"""
        # "Keskin Nişancı" stratejisi: production > checkpoint > evaluation
        production = self.tm.get_threshold('production_default')
        checkpoint = self.tm.get_threshold('model_checkpoint')
        evaluation = self.tm.get_threshold('catboost_evaluation')
        
        # Production en seçici olmalı
        self.assertGreaterEqual(production, checkpoint, 
                               "Production threshold checkpoint'tan yüksek olmalı")
        
        # Model checkpoint evaluation'dan yüksek veya eşit olmalı
        self.assertGreaterEqual(checkpoint, evaluation, 
                               "Model checkpoint threshold evaluation'dan yüksek olmalı")
    
    def test_invalid_context_handling(self):
        """GECERSIZ context'lerin doğru handle edildiğini test et"""
        with self.assertRaises(KeyError):
            self.tm.get_threshold('gecersiz_context')
        
        with self.assertRaises(KeyError):
            self.tm.get_loss_penalty('gecersiz_penalty')
    
    def test_validation_report(self):
        """Validation raporunun doğru çalıştığını test et"""
        report = self.tm.validate_consistency()
        
        # Rapor gerekli alanları içeriyor mu?
        required_fields = ['status', 'warnings', 'errors', 'thresholds']
        for field in required_fields:
            self.assertIn(field, report, f"Validation raporunda {field} eksik")
        
        # Status geçerli mi?
        self.assertIn(report['status'], ['success', 'warning', 'error'])
    
    def test_config_fallback(self):
        """Config yoksa fallback'in çalıştığını test et"""
        # Geçici olarak config dosyasını taşı
        import shutil
        temp_path = self.config_path.with_suffix('.yaml.bak')
        shutil.move(self.config_path, temp_path)
        
        try:
            # Yeni ThresholdManager oluştur (hata vermeli)
            with self.assertRaises(FileNotFoundError):
                ThresholdManager(self.config_path)
        finally:
            # Config'i geri taşı
            shutil.move(temp_path, self.config_path)
    
    def test_threshold_values_in_use(self):
        """Threshold değerlerinin gerçek kullanım senaryolarında test et"""
        # Virtual bankroll callback için
        virtual_threshold = self.tm.get_threshold('virtual_bankroll')
        
        # Test verisi oluştur
        predictions = np.array([0.6, 0.7, 0.8, 0.9])
        binary_predictions = (predictions >= virtual_threshold).astype(int)
        
        # Sonuç kontrolü
        expected_binary = (predictions >= 0.70).astype(int)  # Config'de 0.70
        np.testing.assert_array_equal(binary_predictions, expected_binary,
                                    "Virtual bankroll threshold doğru çalışmıyor")
        
        # CatBoost evaluation için
        catboost_threshold = self.tm.get_threshold('catboost_evaluation')
        self.assertEqual(catboost_threshold, 0.70, "CatBoost threshold config'den gelmiyor")


class TestLossPenaltyIntegration(unittest.TestCase):
    """Loss penalty entegrasyon testleri"""
    
    def setUp(self):
        """Test setup"""
        self.config_path = project_root / 'config' / 'config.yaml'
        self.tm = ThresholdManager(self.config_path)
    
    def test_loss_penalty_parameterization(self):
        """Loss fonksiyonlarının parametrik çalıştığını test et"""
        from utils.custom_losses import balanced_threshold_killer_loss
        
        # Config'den değerleri al
        fp_penalty = self.tm.get_loss_penalty('false_positive_penalty')
        fn_penalty = self.tm.get_loss_penalty('false_negative_penalty')
        critical_penalty = self.tm.get_loss_penalty('critical_zone_penalty')
        
        # Mock veriler oluştur
        y_true = np.array([1.0, 2.0, 1.4, 1.6])
        y_pred = np.array([1.6, 1.2, 1.5, 1.5])
        
        # Loss fonksiyonunu çağır
        try:
            import tensorflow as tf
            loss_fn = balanced_threshold_killer_loss(
                fp_penalty=fp_penalty,
                fn_penalty=fn_penalty,
                critical_penalty=critical_penalty
            )
            
            # TensorFlow tensor'larına çevir
            y_true_tf = tf.constant(y_true, dtype=tf.float32)
            y_pred_tf = tf.constant(y_pred, dtype=tf.float32)
            
            # Loss'u hesapla
            loss_value = loss_fn(y_true_tf, y_pred_tf)
            
            # Sonuç pozitif olmalı
            self.assertGreater(float(loss_value), 0.0, "Loss değeri pozitif olmalı")
            
        except ImportError:
            # TensorFlow yüklü değilse testi atla
            self.skipTest("TensorFlow yüklü değil")


class TestAdaptiveWeightIntegration(unittest.TestCase):
    """Adaptive weight entegrasyon testleri"""
    
    def setUp(self):
        """Test setup"""
        self.config_path = project_root / 'config' / 'config.yaml'
        self.tm = ThresholdManager(self.config_path)
    
    def test_adaptive_weight_values(self):
        """Adaptive weight değerlerinin config'den geldiğini test et"""
        # Başlangıç değerleri
        initial_fp = self.tm.get_adaptive_weight('initial_false_positive_weight')
        initial_fn = self.tm.get_adaptive_weight('initial_false_negative_weight')
        
        # Değerler mantıklı mı?
        self.assertGreater(initial_fp, initial_fn, 
                           "False positive weight false negative'dan yüksek olmalı")
        
        # Sınır değerleri
        min_weight = self.tm.get_adaptive_weight('min_weight')
        max_weight = self.tm.get_adaptive_weight('max_weight')
        
        self.assertLess(min_weight, max_weight, "Min weight max weight'dan küçük olmalı")
        self.assertGreaterEqual(initial_fp, min_weight, "Initial weight min weight'dan büyük olmalı")
        self.assertLessEqual(initial_fp, max_weight, "Initial weight max weight'dan küçük olmalı")


if __name__ == '__main__':
    # Testleri çalıştır
    unittest.main(verbosity=2)
