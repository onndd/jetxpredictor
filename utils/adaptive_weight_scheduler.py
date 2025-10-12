"""
JetX Predictor - Adaptive Weight Scheduler

Eğitim sırasında class weight'i otomatik ayarlayan callback.
Lazy learning'i tespit eder ve weight'i dinamik olarak ayarlar.

Hedef: Model dengeli tahminler yapana kadar weight'i otomatik artır/azalt.
"""

import numpy as np
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import accuracy_score
from typing import Tuple, Optional, Dict
import logging

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
    
    Kullanım:
        scheduler = AdaptiveWeightScheduler(
            initial_weight=2.0,
            min_weight=1.0,
            max_weight=4.0,
            target_below_acc=0.70,
            target_above_acc=0.75,
            test_data=([X_test_inputs...], y_reg_test)
        )
        
        model.fit(..., callbacks=[scheduler])
    """
    
    def __init__(
        self,
        initial_weight: float = 2.0,
        min_weight: float = 1.0,
        max_weight: float = 50.0,
        target_below_acc: float = 0.70,
        target_above_acc: float = 0.75,
        test_data: Optional[Tuple] = None,
        threshold: float = 1.5,
        check_interval: int = 1
    ):
        """
        Args:
            initial_weight: Başlangıç class weight (1.5 altı için)
            min_weight: Minimum weight (1.0 - dengeli)
            max_weight: Maksimum weight (50.0 - lazy learning için yeterli güç)
            target_below_acc: Hedef 1.5 altı accuracy (default: 0.70)
            target_above_acc: Hedef 1.5 üstü accuracy (default: 0.75)
            test_data: Test verisi (X_list, y_reg) tuple
            threshold: Class ayırma eşiği (default: 1.5)
            check_interval: Kaç epoch'ta bir kontrol edilecek (default: 1 - her epoch)
        """
        super().__init__()
        
        self.current_weight = initial_weight
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.target_below_acc = target_below_acc
        self.target_above_acc = target_above_acc
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
        logger.info(f"  • Hedef 1.5 altı: %{target_below_acc*100:.0f}")
        logger.info(f"  • Hedef 1.5 üstü: %{target_above_acc*100:.0f}")
        logger.info(f"  • Kontrol aralığı: Her {check_interval} epoch")
    
    def on_epoch_end(self, epoch, logs=None):
        """Her epoch sonunda çağrılır"""
        # Her epoch kontrol et (check_interval=1 varsayılan)
        if epoch % self.check_interval != 0:
            return
        
        # Test verisi yoksa atla
        if self.test_data is None:
            logger.warning("Test verisi sağlanmadı, weight ayarlaması yapılamıyor")
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
                logger.warning("Model threshold output'u bulunamadı")
                return
            
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
        Accuracy'lere göre weight'i ayarla
        
        Args:
            below_acc: 1.5 altı accuracy
            above_acc: 1.5 üstü accuracy
            
        Returns:
            Ayarlama nedeni (string)
        """
        old_weight = self.current_weight
        
        # LAZY LEARNING TESPİTİ - GÜÇLENDIRILDI (Daha Agresif ve Reaktif)
        
        # Durum 1: Kritik Lazy Learning - Model neredeyse hiç "1.5 altı" tahmin etmiyor
        if below_acc < 0.10 and above_acc > 0.95:
            # Kritik durum - maksimum artış
            self.current_weight *= 2.5
            reason = "🔴🔴 Kritik Lazy Learning (×2.5)"
        
        # Durum 2: Ciddi Lazy Learning - Model sadece "1.5 üstü" tahmin ediyor
        elif below_acc < 0.20 and above_acc > 0.90:
            # Ciddi lazy learning - çok agresif artış
            self.current_weight *= 2.0
            reason = "🔴 Ciddi Lazy Learning (×2.0)"
        
        # Durum 3: Orta Lazy Learning - Model çoğunlukla "1.5 üstü" tahmin ediyor
        elif below_acc < 0.40 and above_acc > 0.80:
            # Orta lazy learning - agresif artış
            self.current_weight *= 1.8
            reason = "🟠 Orta Lazy Learning (×1.8)"
        
        # Durum 4: Hafif Lazy Learning - Model 1.5 altı için yetersiz
        elif below_acc < self.target_below_acc - 0.15:
            # Hedefin çok altında - orta artış
            self.current_weight *= 1.5
            reason = "🟡 Hedefin Çok Altında (×1.5)"
        
        # Durum 5: Hedefin altında ama yakın
        elif below_acc < self.target_below_acc - 0.05:
            # Hedefin biraz altında - hafif artış
            self.current_weight *= 1.2
            reason = "🟡 Hedefin Altında (×1.2)"
        
        # Durum 6: Kritik Aşırı Weight - Model neredeyse hiç "1.5 üstü" tahmin etmiyor
        elif below_acc > 0.95 and above_acc < 0.20:
            # Kritik aşırı weight - maksimum azaltma
            self.current_weight *= 0.4
            reason = "🟢🟢 Kritik Aşırı Weight (×0.4)"
        
        # Durum 7: Ciddi Aşırı Weight - Model sadece "1.5 altı" tahmin ediyor
        elif below_acc > 0.90 and above_acc < 0.50:
            # Aşırı weight - ciddi azaltma
            self.current_weight *= 0.5
            reason = "🟢 Ciddi Aşırı Weight (×0.5)"
        
        # Durum 8: Orta Aşırı Weight - Model çoğunlukla "1.5 altı" tahmin ediyor
        elif below_acc > 0.85 and above_acc < 0.60:
            # Weight çok yüksek - orta azaltma
            self.current_weight *= 0.7
            reason = "🟢 Weight Yüksek (×0.7)"
        
        # Durum 9: Model dengede ve hedefte - hafif azaltma (genelleşme için)
        elif abs(below_acc - above_acc) < 0.10 and below_acc >= self.target_below_acc:
            # Dengeli durum - hafif azaltma
            self.current_weight *= 0.95
            reason = "✅ Dengeli - Hafif Azaltma (×0.95)"
        
        # Durum 10: Model hedefin üstünde - hafif azaltma
        elif below_acc > self.target_below_acc + 0.10:
            # Hedefin üstünde - hafif azaltma
            self.current_weight *= 0.9
            reason = "✅ Hedefin Üstünde - Azaltma (×0.9)"
        
        else:
            # Değişiklik yok - kabul edilebilir performans
            reason = "✅ Değişiklik Yok (Kabul Edilebilir)"
        
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
    
    def plot_history(self):
        """
        Weight ve accuracy geçmişini görselleştir
        (Matplotlib gerekli)
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.weight_history:
                logger.warning("Henüz geçmiş yok")
                return
            
            epochs = [h['epoch'] for h in self.weight_history]
            weights = [h['weight'] for h in self.weight_history]
            below_accs = [h['below_acc'] for h in self.weight_history]
            above_accs = [h['above_acc'] for h in self.weight_history]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Weight geçmişi
            ax1.plot(epochs, weights, linewidth=2, label='Weight')
            ax1.axhline(y=self.best_weight, color='r', linestyle='--', label=f'Best Weight ({self.best_weight:.2f})')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Class Weight')
            ax1.set_title('Adaptive Weight History')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Accuracy geçmişi
            ax2.plot(epochs, below_accs, 'r-', linewidth=2, label='1.5 Altı')
            ax2.plot(epochs, above_accs, 'g-', linewidth=2, label='1.5 Üstü')
            ax2.axhline(y=self.target_below_acc, color='r', linestyle='--', alpha=0.5, label=f'Hedef Altı ({self.target_below_acc:.0%})')
            ax2.axhline(y=self.target_above_acc, color='g', linestyle='--', alpha=0.5, label=f'Hedef Üstü ({self.target_above_acc:.0%})')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Class Accuracy History')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib yüklü değil, görselleştirme yapılamıyor")


# Kullanım örneği
if __name__ == "__main__":
    import tensorflow as tf
    from tensorflow.keras import layers, models
    
    print("="*70)
    print("📊 ADAPTIVE WEIGHT SCHEDULER - TEST")
    print("="*70)
    
    # Basit test modeli oluştur
    print("\n🔧 Test modeli oluşturuluyor...")
    
    # Örnek veri
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.rand(n_samples) * 10
    
    # %35 1.5 altı, %65 1.5 üstü
    below_count = int(0.35 * n_samples)
    y_train[:below_count] = np.random.rand(below_count) * 1.4
    y_train[below_count:] = 1.5 + np.random.rand(n_samples - below_count) * 8.5
    
    # Shuffle
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    X_train = X_train[idx]
    y_train = y_train[idx]
    
    # Train/test split
    split = int(0.8 * n_samples)
    X_test = X_train[split:]
    y_test = y_train[split:]
    X_train = X_train[:split]
    y_train = y_train[:split]
    
    # Threshold output için target
    y_thr_train = (y_train >= 1.5).astype(float)
    y_thr_test = (y_test >= 1.5).astype(float)
    
    print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"   1.5 altı: {(y_train < 1.5).sum()} ({(y_train < 1.5).sum()/len(y_train)*100:.1f}%)")
    print(f"   1.5 üstü: {(y_train >= 1.5).sum()} ({(y_train >= 1.5).sum()/len(y_train)*100:.1f}%)")
    
    # Basit model
    inp = layers.Input((n_features,))
    x = layers.Dense(64, activation='relu')(inp)
    x = layers.Dense(32, activation='relu')(x)
    
    # 3 output
    out_reg = layers.Dense(1, activation='linear', name='regression')(x)
    out_cls = layers.Dense(3, activation='softmax', name='classification')(x)
    out_thr = layers.Dense(1, activation='sigmoid', name='threshold')(x)
    
    model = models.Model(inp, [out_reg, out_cls, out_thr])
    
    model.compile(
        optimizer='adam',
        loss={
            'regression': 'mse',
            'classification': 'sparse_categorical_crossentropy',
            'threshold': 'binary_crossentropy'
        },
        loss_weights={'regression': 0.5, 'classification': 0.2, 'threshold': 0.3},
        metrics={'threshold': ['accuracy']}
    )
    
    print("✅ Model oluşturuldu")
    
    # Classification target (basit versiyon)
    y_cls_train = (y_train >= 1.5).astype(int)
    y_cls_test = (y_test >= 1.5).astype(int)
    
    # Adaptive scheduler oluştur
    print("\n📊 Adaptive Weight Scheduler oluşturuluyor...")
    scheduler = AdaptiveWeightScheduler(
        initial_weight=2.0,
        min_weight=1.0,
        max_weight=4.0,
        target_below_acc=0.70,
        target_above_acc=0.75,
        test_data=(X_test, y_test),
        check_interval=2  # Test için daha sık kontrol
    )
    
    print("\n🔥 Model eğitiliyor (20 epoch)...")
    history = model.fit(
        X_train,
        {
            'regression': y_train,
            'classification': y_cls_train,
            'threshold': y_thr_train
        },
        epochs=20,
        batch_size=32,
        verbose=0,
        callbacks=[scheduler]
    )
    
    print("\n✅ Eğitim tamamlandı!")
    
    # İstatistikler
    stats = scheduler.get_stats()
    print(f"\n📊 SONUÇLAR:")
    print(f"  Final Weight: {stats['current_weight']:.2f}")
    print(f"  En İyi 1.5 Altı Acc: {stats['best_below_acc']*100:.1f}%")
    print(f"  En İyi Weight: {stats['best_weight']:.2f}")
    print(f"  Toplam Ayarlama: {len(stats['weight_history'])}")
    
    print("\n" + "="*70)
    print("✅ Test tamamlandı!")
    print("="*70)
