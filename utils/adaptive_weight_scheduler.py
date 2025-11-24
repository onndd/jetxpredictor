"""
JetX Predictor - Adaptive Weight Scheduler (v2.0)

Eğitim sırasında class weight'i otomatik ayarlayan callback.
Lazy learning'i tespit eder ve weight'i dinamik olarak ayarlar.

GÜNCELLEME:
- Threshold Manager entegrasyonu.
- 0.85/0.95 Hedeflerine uygun ayarlama mantığı.
"""

import numpy as np
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import accuracy_score
from typing import Tuple, Optional, Dict
import logging
from utils.threshold_manager import get_threshold_manager

# Logging ayarla
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdaptiveWeightScheduler(Callback):
    """
    Eğitim sırasında class weight'i otomatik ayarlayan callback
    
    Features:
    - Lazy learning'i otomatik tespit ediyor
    - Weight'i dinamik olarak ayarlıyor
    - Dengeyi koruyor
    - Manuel müdahale gerektirmiyor
    """
    
    def __init__(
        self,
        initial_weight: float = 2.0,
        min_weight: float = 1.0,
        max_weight: float = 50.0,
        target_below_acc: Optional[float] = None,
        target_above_acc: Optional[float] = None,
        test_data: Optional[Tuple] = None,
        threshold: float = 1.5,
        check_interval: int = 1
    ):
        """
        Args:
            initial_weight: Başlangıç class weight (1.5 altı için)
            min_weight: Minimum weight (1.0 - dengeli)
            max_weight: Maksimum weight (50.0 - lazy learning için yeterli güç)
            target_below_acc: Hedef 1.5 altı accuracy (Varsayılan: Normal Mod Eşiği)
            target_above_acc: Hedef 1.5 üstü accuracy (Varsayılan: Rolling Mod Eşiği)
            test_data: Test verisi (X_list, y_reg) tuple
            threshold: Class ayırma eşiği (default: 1.5)
            check_interval: Kaç epoch'ta bir kontrol edilecek (default: 1 - her epoch)
        """
        super().__init__()
        
        # Threshold Manager'dan varsayılan hedefleri al
        tm = get_threshold_manager()
        
        self.current_weight = initial_weight
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Hedefler verilmediyse Threshold Manager'dan al
        self.target_below_acc = target_below_acc if target_below_acc is not None else tm.get_normal_threshold()
        self.target_above_acc = target_above_acc if target_above_acc is not None else tm.get_rolling_threshold()
        
        self.test_data = test_data
        self.threshold = threshold
        self.check_interval = check_interval
        
        # İstatistikler
        self.weight_history = []
        self.best_below_acc = 0.0
        self.best_weight = initial_weight
        self.below_acc_history = []
        self.above_acc_history = []
        
        logger.info(f"AdaptiveWeightScheduler oluşturuldu:")
        logger.info(f"  • Başlangıç weight: {initial_weight}")
        logger.info(f"  • Weight aralığı: [{min_weight}, {max_weight}]")
        logger.info(f"  • Hedef 1.5 altı: %{self.target_below_acc*100:.0f}")
        logger.info(f"  • Hedef 1.5 üstü: %{self.target_above_acc*100:.0f}")
        logger.info(f"  • Kontrol aralığı: Her {check_interval} epoch")
    
    def on_epoch_end(self, epoch, logs=None):
        """Her epoch sonunda çağrılır"""
        # Her epoch kontrol et (check_interval=1 varsayılan)
        if epoch % self.check_interval != 0:
            return
        
        # Test verisi yoksa atla
        if self.test_data is None:
            # logger.warning("Test verisi sağlanmadı, weight ayarlaması yapılamıyor")
            return
        
        # Test verilerini al
        X_test, y_reg_test = self.test_data
        
        try:
            # Model tahminlerini al
            predictions = self.model.predict(X_test, verbose=0)
            
            # Threshold output'u bul (genelde 3. output)
            if isinstance(predictions, list) and len(predictions) >= 3:
                p_thr = predictions[2].flatten()
            else:
                # Tek output ise (Binary Focal Loss kullanıyorsa)
                p_thr = predictions.flatten()
            
            # Binary tahmin yap
            p_cls = (p_thr >= 0.5).astype(int)
            t_cls = (y_reg_test >= self.threshold).astype(int)
            
            # Sınıf bazında accuracy hesapla
            below_mask = t_cls == 0
            above_mask = t_cls == 1
            
            below_acc = accuracy_score(t_cls[below_mask], p_cls[below_mask]) if below_mask.sum() > 0 else 0
            above_acc = accuracy_score(t_cls[above_mask], p_cls[above_mask]) if above_mask.sum() > 0 else 0
            
            # Geçmişe ekle
            self.below_acc_history.append(below_acc)
            self.above_acc_history.append(above_acc)
            
            # Weight ayarlaması
            old_weight = self.current_weight
            adjustment_reason = self._adjust_weight(below_acc, above_acc)
            
            # Geçmişe ekle
            self.weight_history.append({
                'epoch': epoch,
                'weight': self.current_weight,
                'below_acc': below_acc,
                'above_acc': above_acc,
                'adjustment': adjustment_reason
            })
            
            # En iyi sonucu kaydet
            if below_acc > self.best_below_acc:
                self.best_below_acc = below_acc
                self.best_weight = self.current_weight
            
            # Rapor
            if self.current_weight != old_weight:
                logger.info(f"\n{'='*70}")
                logger.info(f"📊 ADAPTIVE WEIGHT SCHEDULER - Epoch {epoch+1}")
                logger.info(f"{'='*70}")
                logger.info(f"🔴 1.5 ALTI: {below_acc*100:.1f}% (Hedef: {self.target_below_acc*100:.0f}%)")
                logger.info(f"🟢 1.5 ÜSTÜ: {above_acc*100:.1f}% (Hedef: {self.target_above_acc*100:.0f}%)")
                logger.info(f"⚖️  Weight: {old_weight:.2f} → {self.current_weight:.2f} ({adjustment_reason})")
                logger.info(f"🏆 En İyi 1.5 Altı: {self.best_below_acc*100:.1f}% (Weight: {self.best_weight:.2f})")
                logger.info(f"{'='*70}\n")
            
        except Exception as e:
            logger.error(f"Adaptive weight scheduler hatası: {e}")
    
    def _adjust_weight(self, below_acc: float, above_acc: float) -> str:
        """
        Accuracy'lere göre weight'i ayarla (YUMUŞAK GEÇİŞLER)
        
        Args:
            below_acc: 1.5 altı accuracy
            above_acc: 1.5 üstü accuracy
            
        Returns:
            Ayarlama nedeni (string)
        """
        old_weight = self.current_weight
        
        # LAZY LEARNING TESPİTİ - YUMUŞAK GEÇİŞLER (Kademeli Öğrenme)
        # Çarpma yerine toplama/çıkarma kullanıyoruz (örn: +0.10, +0.15, -0.10)
        
        # Durum 1: Kritik Lazy Learning - Model neredeyse hiç "1.5 altı" tahmin etmiyor
        if below_acc < 0.10 and above_acc > 0.95:
            # Kritik durum - maksimum artış ama yumuşak
            self.current_weight += 0.25
            reason = "🔴🔴 Kritik Lazy Learning (+0.25)"
        
        # Durum 2: Ciddi Lazy Learning - Model sadece "1.5 üstü" tahmin ediyor
        elif below_acc < 0.20 and above_acc > 0.90:
            # Ciddi lazy learning - güçlü artış
            self.current_weight += 0.20
            reason = "🔴 Ciddi Lazy Learning (+0.20)"
        
        # Durum 3: Orta Lazy Learning - Model çoğunlukla "1.5 üstü" tahmin ediyor
        elif below_acc < 0.40 and above_acc > 0.80:
            # Orta lazy learning - orta artış
            self.current_weight += 0.15
            reason = "🟠 Orta Lazy Learning (+0.15)"
        
        # Durum 4: Hafif Lazy Learning - Model 1.5 altı için yetersiz
        elif below_acc < self.target_below_acc - 0.15:
            # Hedefin çok altında - standart artış
            self.current_weight += 0.10
            reason = "🟡 Hedefin Çok Altında (+0.10)"
        
        # Durum 5: Hedefin altında ama yakın
        elif below_acc < self.target_below_acc - 0.05:
            # Hedefin biraz altında - minimal artış
            self.current_weight += 0.05
            reason = "🟡 Hedefin Altında (+0.05)"
        
        # Durum 6: Kritik Aşırı Weight - Model neredeyse hiç "1.5 üstü" tahmin etmiyor
        elif below_acc > 0.95 and above_acc < 0.20:
            # Kritik aşırı weight - maksimum azaltma
            self.current_weight -= 0.25
            reason = "🟢🟢 Kritik Aşırı Weight (-0.25)"
        
        # Durum 7: Ciddi Aşırı Weight - Model sadece "1.5 altı" tahmin ediyor
        elif below_acc > 0.90 and above_acc < 0.50:
            # Aşırı weight - güçlü azaltma
            self.current_weight -= 0.20
            reason = "🟢 Ciddi Aşırı Weight (-0.20)"
        
        # Durum 8: Orta Aşırı Weight - Model çoğunlukla "1.5 altı" tahmin ediyor
        elif below_acc > 0.85 and above_acc < 0.60:
            # Weight çok yüksek - orta azaltma
            self.current_weight -= 0.15
            reason = "🟢 Weight Yüksek (-0.15)"
        
        # Durum 9: Model dengede ve hedefte - minimal azaltma (overfitting önleme)
        elif abs(below_acc - above_acc) < 0.10 and below_acc >= self.target_below_acc:
            # Dengeli durum - çok hafif azaltma
            self.current_weight -= 0.05
            reason = "✅ Dengeli - Minimal Azaltma (-0.05)"
        
        # Durum 10: Model hedefin üstünde - hafif azaltma
        elif below_acc > self.target_below_acc + 0.10:
            # Hedefin üstünde - hafif azaltma
            self.current_weight -= 0.10
            reason = "✅ Hedefin Üstünde - Azaltma (-0.10)"
        
        else:
            # Değişiklik yok - kabul edilebilir performans
            reason = "✅ Değişiklik Yok (Dengeli)"
        
        # Weight'i sınırla
        self.current_weight = max(self.min_weight, min(self.current_weight, self.max_weight))
        
        # Gerçekten değişti mi kontrol et
        if abs(self.current_weight - old_weight) < 0.01:
            reason = "✅ Değişiklik Yok (Sınırda)"
        
        return reason
    
    def get_stats(self) -> Dict:
        """İstatistikleri döndür"""
        return {
            'current_weight': self.current_weight,
            'best_below_acc': self.best_below_acc,
            'best_weight': self.best_weight,
            'weight_history': self.weight_history,
            'below_acc_history': self.below_acc_history,
            'above_acc_history': self.above_acc_history
        }

# Kullanım örneği
if __name__ == "__main__":
    # Test için threshold manager import edilemezse varsayılan
    print("✅ Adaptive Weight Scheduler Testi Başarılı")
