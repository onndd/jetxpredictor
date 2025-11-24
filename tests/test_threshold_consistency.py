"""
Threshold Consistency Tests
"Raporlama vs. Eylem" tutarsızlıklarını test eder
Merkezi threshold yönetiminin doğru çalıştığını doğrular

GÜNCELLEME:
- 2 Modlu (Normal/Rolling) yapı testleri eklendi.
- Eşikler 0.85 ve 0.95 olarak güncellendi.
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
            'normal', 'rolling', 'detailed_metrics', 
            'virtual_bankroll', 'model_checkpoint',
            'production_default'
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
            
            # Mantıksal kontroller (2 Modlu)
            if context == 'normal':
                self.assertEqual(value, 0.85, "Normal mod threshold 0.85 olmalı")
            elif context == 'rolling':
                self.assertEqual(value, 0.95, "Rolling mod threshold 0.95 olmalı")
    
    def test_loss_penalty_values(self):
        """Loss penalty değerlerinin geçerli olduğunu test et"""
        penalties = self.tm.get_all_loss_penalties()
        
        for penalty_type, value in penalties.items():
            # Tüm penalty'ler pozitif olmalı
            self.assertGreater(value, 0.0, f"{penalty_type} penalty pozitif olmalı")
            
            # Mantıksal kontroller
            if 'false_positive' in penalty_type:
                # False positive penalty yüksek olmalı (para kaybı en riskli)
                self.assertGreaterEqual(value, 2.0, "False positive penalty yeterince yüksek olmalı")
    
    def test_threshold_manager_singleton(self):
        """Threshold Manager singleton pattern'ını test et"""
        tm1 = get_threshold_manager()
        tm2 = get_threshold_manager()
        
        # Aynı instance mı?
        self.assertIs(tm1, tm2, "Threshold Manager singleton çalışmıyor")
    
    def test_threshold_consistency_across_contexts(self):
        """Farklı context'ler arası threshold tutarlılığını test et"""
        # Rolling > Normal olmalı
        rolling = self.tm.get_rolling_threshold()
        normal = self.tm.get_normal_threshold()
        
        self.assertGreater(rolling, normal, "Rolling threshold normalden büyük olmalı")
        
        # Production default en az Normal mod kadar güvenli olmalı
        production = self.tm.get_threshold('production_default')
        self.assertGreaterEqual(production, 0.85, "Production threshold en az 0.85 olmalı")
    
    def test_invalid_context_handling(self):
        """GECERSIZ context'lerin doğru handle edildiğini test et"""
        # Varsayılan değer dönmeli (0.85)
        val = self.tm.get_threshold('gecersiz_context')
        self.assertEqual(val, 0.85, "Geçersiz context varsayılan değeri döndürmüyor")
    
    def test_validation_report(self):
        """Validation raporunun doğru çalıştığını test et"""
        report = self.tm.validate_consistency()
        
        # Rapor gerekli alanları içeriyor mu?
        required_fields = ['status', 'warnings', 'errors', 'thresholds']
        for field in required_fields:
            self.assertIn(field, report, f"Validation raporunda {field} eksik")
        
        # Status geçerli mi?
        self.assertIn(report['status'], ['success', 'warning', 'error'])
    
    def test_threshold_values_in_use(self):
        """Threshold değerlerinin gerçek kullanım senaryolarında test et"""
        # Virtual bankroll callback için (genelde Normal mod kullanılır)
        virtual_threshold = self.tm.get_threshold('virtual_bankroll')
        
        # Test verisi oluştur
        predictions = np.array([0.80, 0.84, 0.86, 0.96])
        
        # Normal Mod (0.85) testi
        binary_predictions = (predictions >= 0.85).astype(int)
        expected_binary = np.array([0, 0, 1, 1])
        np.testing.assert_array_equal(binary_predictions, expected_binary, "Normal mod threshold hatası")


class TestLossPenaltyIntegration(unittest.TestCase):
    """Loss penalty entegrasyon testleri"""
    
    def setUp(self):
        """Test setup"""
        self.config_path = project_root / 'config' / 'config.yaml'
        self.tm = ThresholdManager(self.config_path)
    
    def test_loss_penalty_parameterization(self):
        """Loss fonksiyonlarının parametrik çalıştığını test et"""
        # Bu test TensorFlow gerektirir, yoksa atla
        try:
            import tensorflow as tf
            from utils.custom_losses import balanced_threshold_killer_loss
        except ImportError:
            self.skipTest("TensorFlow yüklü değil")
            return

        # Mock veriler oluştur
        y_true = np.array([1.0, 2.0, 1.4, 1.6])
        y_pred = np.array([1.6, 1.2, 1.5, 1.5])
        
        # TensorFlow tensor'larına çevir
        y_true_tf = tf.constant(y_true, dtype=tf.float32)
        y_pred_tf = tf.constant(y_pred, dtype=tf.float32)
        
        # Loss'u hesapla (fonksiyon artık parametre almaz, config'den okur)
        # Ancak test ortamında config mocklanmadığı için varsayılanları kullanır
        loss_value = balanced_threshold_killer_loss(y_true_tf, y_pred_tf)
        
        # Sonuç pozitif olmalı
        self.assertGreater(float(loss_value), 0.0, "Loss değeri pozitif olmalı")


if __name__ == '__main__':
    # Testleri çalıştır
    unittest.main(verbosity=2)
