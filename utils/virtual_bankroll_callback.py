"""
Virtual Bankroll Callback - Her Epoch İçin Sanal Kasa Gösterimi
Progressive NN ve CatBoost eğitimleri için

GÜNCELLEME: Threshold değerleri artık config'den alınıyor
"Raporlama vs. Eylem" tutarsızlıkları önleniyor
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
    Her epoch sonunda 2 sanal kasa simülasyonu gösterir:
    - Kasa 1: 1.5x eşik sistemi
    - Kasa 2: %70 çıkış sistemi (yüksek tahminler için)
    """
    
    def __init__(self, stage_name, X_test, y_test, threshold=1.5, 
                 starting_capital=1000.0, bet_amount=10.0, exit_multiplier=0.70):
        """
        Args:
            stage_name: Aşama adı (örn: "AŞAMA 1")
            X_test: Test verileri (list of arrays)
            y_test: Test hedef değerleri
            threshold: Eşik değeri (varsayılan: 1.5)
            starting_capital: Başlangıç sermayesi
            bet_amount: Bahis tutarı
            exit_multiplier: Kasa 2 için çıkış çarpanı (varsayılan: 0.70)
        """
        super().__init__()
        self.stage_name = stage_name
        self.X_test = X_test
        self.y_test = y_test
        self.threshold = threshold
        self.starting_capital = starting_capital
        self.bet_amount = bet_amount
        self.win_amount = threshold * bet_amount
        self.exit_multiplier = exit_multiplier
        
        # Kasa 1 için tracking
        self.best_roi_kasa1 = -float('inf')
        self.best_epoch_kasa1 = 0
        
        # Kasa 2 için tracking
        self.best_roi_kasa2 = -float('inf')
        self.best_epoch_kasa2 = 0
        
    def on_epoch_end(self, epoch, logs=None):
        """Her epoch sonunda çağrılır - 2 kasa simülasyonu yapar"""
        # Model tahminlerini al
        predictions = self.model.predict(self.X_test, verbose=0)
        
        # Regression output (birinci output)
        p_reg = predictions[0].flatten() if len(predictions) > 0 else None
        
        # Threshold output'u al (üçüncü output)
        p_thr = predictions[2].flatten() if len(predictions) > 2 else predictions[0].flatten()
        
        # GÜNCELLEME: Threshold değerini config'den al
        try:
            threshold_manager = get_threshold_manager()
            virtual_bankroll_threshold = threshold_manager.get_threshold('virtual_bankroll')
        except Exception as e:
            # Fallback: Config'den alınamazsa eski değeri kullan
            print(f"⚠️ Threshold manager hatası: {e}, varsayılan %70 kullanılıyor")
            virtual_bankroll_threshold = 0.70
        
        # Binary predictions - ARTIK CONFIG'DEN GELEN DEĞER KULLANILIYOR
        p_thr_binary = (p_thr >= virtual_bankroll_threshold).astype(int)
        t_thr = (self.y_test >= self.threshold).astype(int)
        
        # Debug info
        if epoch == 0:  # Sadece ilk epoch'ta göster
            print(f"🎯 VirtualBankroll Threshold: {virtual_bankroll_threshold:.2f} (Config'den alındı)")
        
        # ========================================================================
        # KASA 1: 1.5x EŞİK SİSTEMİ
        # ========================================================================
        wallet1 = self.starting_capital
        total_bets1 = 0
        total_wins1 = 0
        total_losses1 = 0
        
        for i in range(len(p_thr_binary)):
            model_pred = p_thr_binary[i]
            actual_value = self.y_test[i]
            
            # Model "1.5 üstü" diyorsa bahse gir
            if model_pred == 1:
                wallet1 -= self.bet_amount
                total_bets1 += 1
                
                if actual_value >= self.threshold:
                    # Kazandık!
                    wallet1 += self.win_amount
                    total_wins1 += 1
                else:
                    # Kaybettik
                    total_losses1 += 1
        
        # Kasa 1 sonuçları
        profit_loss1 = wallet1 - self.starting_capital
        roi1 = (profit_loss1 / self.starting_capital) * 100 if total_bets1 > 0 else 0
        win_rate1 = (total_wins1 / total_bets1 * 100) if total_bets1 > 0 else 0
        
        # En iyi ROI'yi takip et
        if roi1 > self.best_roi_kasa1:
            self.best_roi_kasa1 = roi1
            self.best_epoch_kasa1 = epoch + 1
        
        # Emoji seçimi
        if profit_loss1 > 100:
            wallet_emoji1 = "🚀"
        elif profit_loss1 > 0:
            wallet_emoji1 = "✅"
        elif profit_loss1 > -100:
            wallet_emoji1 = "⚠️"
        else:
            wallet_emoji1 = "❌"
        
        # ========================================================================
        # KASA 2: %70 ÇIKIŞ SİSTEMİ
        # ========================================================================
        wallet2 = self.starting_capital
        total_bets2 = 0
        total_wins2 = 0
        total_losses2 = 0
        exit_points = []
        
        if p_reg is not None:
            for i in range(len(p_reg)):
                model_pred_value = p_reg[i]
                actual_value = self.y_test[i]
                
                # Model 2.0+ tahmin ediyorsa bahse gir
                if model_pred_value >= 2.0:
                    wallet2 -= self.bet_amount
                    total_bets2 += 1
                    
                    # Çıkış noktası: tahmin × 0.70
                    exit_point = model_pred_value * self.exit_multiplier
                    exit_points.append(exit_point)
                    
                    if actual_value >= exit_point:
                        # Kazandık!
                        wallet2 += exit_point * self.bet_amount
                        total_wins2 += 1
                    else:
                        # Kaybettik
                        total_losses2 += 1
        
        # Kasa 2 sonuçları
        profit_loss2 = wallet2 - self.starting_capital
        roi2 = (profit_loss2 / self.starting_capital) * 100 if total_bets2 > 0 else 0
        win_rate2 = (total_wins2 / total_bets2 * 100) if total_bets2 > 0 else 0
        avg_exit = np.mean(exit_points) if exit_points else 0
        
        # En iyi ROI'yi takip et
        if roi2 > self.best_roi_kasa2:
            self.best_roi_kasa2 = roi2
            self.best_epoch_kasa2 = epoch + 1
        
        # Emoji seçimi
        if profit_loss2 > 100:
            wallet_emoji2 = "🚀"
        elif profit_loss2 > 0:
            wallet_emoji2 = "✅"
        elif profit_loss2 > -100:
            wallet_emoji2 = "⚠️"
        else:
            wallet_emoji2 = "❌"
        
        # ========================================================================
        # DETAYLI RAPOR - KASA 1
        # ========================================================================
        print(f"\n{'='*80}")
        print(f"💰 {self.stage_name} - Epoch {epoch+1} - KASA 1 (1.5x EŞİK)")
        print(f"{'='*80}")
        print(f"   📊 Test Seti: {len(self.y_test):,} örnek")
        print(f"   🎯 Model Tahmini: {total_bets1} oyunda '1.5 üstü' dedi")
        print(f"   ")
        print(f"   📈 SONUÇLAR:")
        print(f"      ✅ Kazanan: {total_wins1} oyun ({win_rate1:.1f}%)")
        print(f"      ❌ Kaybeden: {total_losses1} oyun")
        print(f"   ")
        print(f"   💰 KASA DURUMU:")
        print(f"      Başlangıç: {self.starting_capital:,.0f} TL")
        print(f"      Final: {wallet1:,.0f} TL")
        print(f"      Net: {profit_loss1:+,.0f} TL | ROI: {roi1:+.2f}% {wallet_emoji1}")
        print(f"   ")
        print(f"   🎯 DEĞERLENDİRME:")
        if total_bets1 == 0:
            print(f"      ⚠️ Model hiç '1.5 üstü' tahmin etmedi!")
        elif win_rate1 >= 66.7:
            print(f"      ✅ Kazanma oranı başabaş noktasının ÜSTÜNDE (%66.7)")
        else:
            print(f"      ❌ Kazanma oranı başabaş noktasının ALTINDA (Hedef: %66.7)")
        print(f"   ")
        print(f"   🏆 En İyi: Epoch {self.best_epoch_kasa1} (ROI: {self.best_roi_kasa1:+.2f}%)")
        print(f"{'='*80}\n")
        
        # ========================================================================
        # DETAYLI RAPOR - KASA 2
        # ========================================================================
        print(f"{'='*80}")
        print(f"💰 {self.stage_name} - Epoch {epoch+1} - KASA 2 (%{int(self.exit_multiplier*100)} ÇIKIŞ)")
        print(f"{'='*80}")
        print(f"   📊 Test Seti: {len(self.y_test):,} örnek")
        print(f"   🎯 Model Tahmini: {total_bets2} oyunda '2.0+ değer' dedi")
        print(f"   ")
        print(f"   📈 SONUÇLAR:")
        print(f"      ✅ Kazanan: {total_wins2} oyun ({win_rate2:.1f}%)")
        print(f"      ❌ Kaybeden: {total_losses2} oyun")
        print(f"      📊 Ortalama Çıkış: {avg_exit:.2f}x")
        print(f"   ")
        print(f"   💰 KASA DURUMU:")
        print(f"      Başlangıç: {self.starting_capital:,.0f} TL")
        print(f"      Final: {wallet2:,.0f} TL")
        print(f"      Net: {profit_loss2:+,.0f} TL | ROI: {roi2:+.2f}% {wallet_emoji2}")
        print(f"   ")
        print(f"   🎯 DEĞERLENDİRME:")
        if total_bets2 == 0:
            print(f"      ⚠️ Model hiç '2.0+' tahmin etmedi!")
        elif win_rate2 >= 66.7:
            print(f"      ✅ Kazanma oranı başabaş noktasının ÜSTÜNDE (%66.7)")
        else:
            print(f"      ❌ Kazanma oranı başabaş noktasının ALTINDA (Hedef: %66.7)")
        print(f"   ")
        print(f"   🏆 En İyi: Epoch {self.best_epoch_kasa2} (ROI: {self.best_roi_kasa2:+.2f}%)")
        print(f"{'='*80}\n")


class CatBoostBankrollCallback:
    """
    CatBoost için Virtual Bankroll Callback
    Her 10 iteration'da bir sanal kasa gösterir
    """
    
    def __init__(self, X_test, y_test, threshold=1.5, 
                 starting_capital=1000.0, bet_amount=10.0, 
                 model_type='regressor', interval=10):
        """
        Args:
            X_test: Test verileri
            y_test: Test hedef değerleri
            threshold: Eşik değeri (varsayılan: 1.5)
            starting_capital: Başlangıç sermayesi
            bet_amount: Bahis tutarı
            model_type: 'regressor' veya 'classifier'
            interval: Kaç iteration'da bir göster (varsayılan: 10)
        """
        self.X_test = X_test
        self.y_test = y_test
        self.threshold = threshold
        self.starting_capital = starting_capital
        self.bet_amount = bet_amount
        self.win_amount = threshold * bet_amount
        self.model_type = model_type
        self.interval = interval
        self.best_roi = -float('inf')
        self.best_iteration = 0
        
    def after_iteration(self, info):
        """
        CatBoost iteration callback
        info.iteration: 0-based iteration number
        """
        iteration = info.iteration
        
        # Sadece belirli aralıklarda rapor et
        if iteration % self.interval != 0:
            return True  # Devam et
        
        # Model tahminlerini al
        if self.model_type == 'regressor':
            # Regressor: direkt değer tahmini
            predictions = info.model.predict(self.X_test)
            p_binary = (predictions >= self.threshold).astype(int)
        else:
            # Classifier: sınıf tahmini
            predictions = info.model.predict(self.X_test)
            p_binary = predictions  # Zaten 0 veya 1
        
        t_binary = (self.y_test >= self.threshold).astype(int)
        
        # Sanal kasa simülasyonu
        wallet = self.starting_capital
        total_bets = 0
        total_wins = 0
        total_losses = 0
        
        for i in range(len(p_binary)):
            model_pred = p_binary[i]
            actual_value = self.y_test[i]
            
            # Model "1.5 üstü" diyorsa bahse gir
            if model_pred == 1:
                wallet -= self.bet_amount
                total_bets += 1
                
                if actual_value >= self.threshold:
                    # Kazandık!
                    wallet += self.win_amount
                    total_wins += 1
                else:
                    # Kaybettik
                    total_losses += 1
        
        # Sonuçları hesapla
        profit_loss = wallet - self.starting_capital
        roi = (profit_loss / self.starting_capital) * 100 if total_bets > 0 else 0
        win_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0
        
        # En iyi ROI'yi takip et
        if roi > self.best_roi:
            self.best_roi = roi
            self.best_iteration = iteration + 1
        
        # Emoji seçimi
        if profit_loss > 100:
            wallet_emoji = "🚀"
        elif profit_loss > 0:
            wallet_emoji = "✅"
        elif profit_loss > -100:
            wallet_emoji = "⚠️"
        else:
            wallet_emoji = "❌"
        
        # Rapor
        model_name = "REGRESSOR" if self.model_type == 'regressor' else "CLASSIFIER"
        print(f"\n{'='*80}")
        print(f"💰 CATBOOST {model_name} - Iteration {iteration+1} - SANAL KASA")
        print(f"{'='*80}")
        print(f"   🎲 Oyun: {total_bets} el ({total_wins} kazanç, {total_losses} kayıp)")
        print(f"   📊 Kazanma Oranı: {win_rate:.1f}%")
        print(f"   💰 Başlangıç: {self.starting_capital:,.0f} TL → Final: {wallet:,.0f} TL")
        print(f"   📈 Net: {profit_loss:+,.0f} TL | ROI: {roi:+.2f}% {wallet_emoji}")
        
        if win_rate >= 66.7:
            print(f"   ✅ Kazanma oranı başabaş noktasının ÜSTÜNDE (%66.7)")
        else:
            print(f"   ⚠️ Kazanma oranı başabaş noktasının ALTINDA (Hedef: %66.7)")
        
        print(f"   🏆 En İyi: Iteration {self.best_iteration} (ROI: {self.best_roi:+.2f}%)")
        print(f"{'='*80}\n")
        
        return True  # Devam et
