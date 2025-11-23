"""
Virtual Bankroll Callback - Her Epoch İçin Sanal Kasa Gösterimi
Progressive NN ve CatBoost eğitimleri için

GÜNCELLEME:
- 2 Modlu Yapı (Normal/Rolling) entegre edildi.
- Kasa 1: Normal Mod (0.85+ Güven, Dinamik Çıkış)
- Kasa 2: Rolling Mod (0.95+ Güven, Sabit 1.5x Çıkış)
- Threshold Manager entegrasyonu
"""

import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import callbacks

# Threshold Manager import
from .threshold_manager import get_threshold_manager


class VirtualBankrollCallback(callbacks.Callback):
    """
    TensorFlow/Keras için Virtual Bankroll Callback
    Her epoch sonunda 2 sanal kasa simülasyonu gösterir.
    """
    
    def __init__(self, stage_name, X_test, y_test, starting_capital=1000.0, bet_amount=10.0):
        """
        Args:
            stage_name: Aşama adı (örn: "AŞAMA 1")
            X_test: Test verileri (list of arrays)
            y_test: Test hedef değerleri
            starting_capital: Başlangıç sermayesi
            bet_amount: Bahis tutarı
        """
        super().__init__()
        self.stage_name = stage_name
        self.X_test = X_test
        self.y_test = y_test
        self.starting_capital = starting_capital
        self.bet_amount = bet_amount
        
        # Threshold Manager'dan eşikleri al
        tm = get_threshold_manager()
        self.threshold_normal = tm.get_normal_threshold()   # 0.85
        self.threshold_rolling = tm.get_rolling_threshold() # 0.95
        
        # Kasa takibi
        self.best_roi_normal = -float('inf')
        self.best_roi_rolling = -float('inf')
        
    def on_epoch_end(self, epoch, logs=None):
        """Her epoch sonunda çağrılır - 2 kasa simülasyonu yapar"""
        # Model tahminlerini al
        predictions = self.model.predict(self.X_test, verbose=0)
        
        # Regression output (birinci output)
        p_reg = predictions[0].flatten() if len(predictions) > 0 else None
        
        # Threshold output'u al (üçüncü output, genelde binary prob)
        # Eğer model tek çıktılıysa (sadece reg veya sadece thr) ona göre davran
        p_thr = predictions[2].flatten() if len(predictions) > 2 else predictions[0].flatten()
        
        # Gerçek değerler (Regression target)
        # Eğer y_test sadece binary ise (0/1), simülasyon için gerçek değerlere ihtiyacımız var.
        # Ancak burada y_test genellikle regression target olarak geliyor.
        actual_values = self.y_test
        
        # ========================================================================
        # KASA 1: NORMAL MOD (0.85+ Güven, Dinamik Çıkış)
        # ========================================================================
        wallet1 = self.starting_capital
        bets1 = 0
        wins1 = 0
        
        for i in range(len(p_thr)):
            # Güven kontrolü
            if p_thr[i] >= self.threshold_normal:
                wallet1 -= self.bet_amount
                bets1 += 1
                
                # Dinamik çıkış: Tahminin %80'i, min 1.5, max 2.5
                # Eğer p_reg yoksa (sadece classifier ise) 1.5 sabit
                if p_reg is not None:
                    exit_pt = min(max(1.5, p_reg[i] * 0.8), 2.5)
                else:
                    exit_pt = 1.5
                
                if actual_values[i] >= exit_pt:
                    wallet1 += self.bet_amount * exit_pt
                    wins1 += 1
        
        roi1 = (wallet1 - self.starting_capital) / self.starting_capital * 100 if bets1 > 0 else 0
        if roi1 > self.best_roi_normal: self.best_roi_normal = roi1
        
        # ========================================================================
        # KASA 2: ROLLING MOD (0.95+ Güven, Sabit 1.5x Çıkış)
        # ========================================================================
        wallet2 = self.starting_capital
        bets2 = 0
        wins2 = 0
        
        for i in range(len(p_thr)):
            # Güven kontrolü
            if p_thr[i] >= self.threshold_rolling:
                wallet2 -= self.bet_amount
                bets2 += 1
                
                # Sabit güvenli çıkış
                if actual_values[i] >= 1.5:
                    wallet2 += self.bet_amount * 1.5
                    wins2 += 1
        
        roi2 = (wallet2 - self.starting_capital) / self.starting_capital * 100 if bets2 > 0 else 0
        if roi2 > self.best_roi_rolling: self.best_roi_rolling = roi2
        
        # RAPORLAMA
        print(f"\n💰 {self.stage_name} - Epoch {epoch+1} BANKROLL:")
        
        # Kasa 1 Raporu
        emoji1 = "🚀" if roi1 > 0 else "❌"
        print(f"   🎯 Normal ({self.threshold_normal}): ROI {roi1:+.2f}% | Win {wins1}/{bets1} ({0 if bets1==0 else wins1/bets1*100:.1f}%) {emoji1}")
        
        # Kasa 2 Raporu
        emoji2 = "🚀" if roi2 > 0 else "❌"
        print(f"   🛡️ Rolling ({self.threshold_rolling}): ROI {roi2:+.2f}% | Win {wins2}/{bets2} ({0 if bets2==0 else wins2/bets2*100:.1f}%) {emoji2}")
        print("-" * 60)


class CatBoostBankrollCallback:
    """
    CatBoost için Virtual Bankroll Callback
    Her N iteration'da bir sanal kasa gösterir
    """
    
    def __init__(self, X_test, y_test, starting_capital=1000.0, bet_amount=10.0, interval=50):
        self.X_test = X_test
        self.y_test = y_test
        self.starting_capital = starting_capital
        self.bet_amount = bet_amount
        self.interval = interval
        
        tm = get_threshold_manager()
        self.threshold_normal = tm.get_normal_threshold()
        self.threshold_rolling = tm.get_rolling_threshold()
        
    def after_iteration(self, info):
        if info.iteration % self.interval != 0: return True
        
        # CatBoost classifier tahmini (Olasılık)
        # Not: CatBoost callback içinde model erişimi bazen kısıtlıdır,
        # bu yüzden basit predict kullanıyoruz. Eğer raw prediction ise sigmoid gerekebilir.
        # Burada standart predict (class) varsayıyoruz, ancak olasılık erişimi varsa daha iyi.
        try:
            probs = info.model.predict_proba(self.X_test)[:, 1]
        except:
            # Eğer proba yoksa (regressor ise)
            probs = info.model.predict(self.X_test)
            # Regressor çıktısını proba gibi 0-1 arasına sıkıştırmak veya direkt kullanmak?
            # Regressor ise bu değer 'tahmin edilen çarpan'dır.
            # Bu durumda threshold mantığı değişir. 
            # Basitlik için: Regressor 1.5 üstü tahmin ediyorsa prob=1, yoksa 0 varsayalım.
            probs = (probs >= 1.5).astype(float)

        actuals = self.y_test
        
        # Kasa 1 (Normal - Basitleştirilmiş: 1.5 üstü tahmin edilirse gir, 1.5'te çık)
        # CatBoost eğitiminde genellikle ya regressor ya classifier tek tek eğitilir.
        # Bu yüzden karmaşık "regressor'dan al, classifier'dan onayla" yapısını burada kurmak zor.
        # Basit bir simülasyon yapıyoruz:
        
        wallet = self.starting_capital
        bets = 0
        wins = 0
        
        for i in range(len(probs)):
            # Eğer model 1.5 üstü olacağına %85+ ihtimal veriyorsa (veya regressor 1.5+ dediyse)
            if probs[i] >= self.threshold_normal:
                wallet -= self.bet_amount
                bets += 1
                if actuals[i] >= 1.5:
                    wallet += self.bet_amount * 1.5
                    wins += 1
        
        roi = (wallet - self.starting_capital) / self.starting_capital * 100 if bets > 0 else 0
        
        print(f"\n💰 CatBoost Iter {info.iteration}: ROI {roi:+.2f}% | Bets {bets}")
        return True
