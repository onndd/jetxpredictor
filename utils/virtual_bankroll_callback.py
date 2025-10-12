"""
Virtual Bankroll Callback - Her Epoch İçin Sanal Kasa Gösterimi
Progressive NN ve CatBoost eğitimleri için
"""

import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import callbacks


class VirtualBankrollCallback(callbacks.Callback):
    """
    TensorFlow/Keras için Virtual Bankroll Callback
    Her epoch sonunda sanal kasa simülasyonu gösterir
    """
    
    def __init__(self, stage_name, X_test, y_test, threshold=1.5, 
                 starting_capital=1000.0, bet_amount=10.0):
        """
        Args:
            stage_name: Aşama adı (örn: "AŞAMA 1")
            X_test: Test verileri (list of arrays)
            y_test: Test hedef değerleri
            threshold: Eşik değeri (varsayılan: 1.5)
            starting_capital: Başlangıç sermayesi
            bet_amount: Bahis tutarı
        """
        super().__init__()
        self.stage_name = stage_name
        self.X_test = X_test
        self.y_test = y_test
        self.threshold = threshold
        self.starting_capital = starting_capital
        self.bet_amount = bet_amount
        self.win_amount = threshold * bet_amount
        self.best_roi = -float('inf')
        self.best_epoch = 0
        
    def on_epoch_end(self, epoch, logs=None):
        """Her epoch sonunda çağrılır"""
        # Model tahminlerini al
        predictions = self.model.predict(self.X_test, verbose=0)
        
        # Threshold output'u al (üçüncü output)
        p_thr = predictions[2].flatten() if len(predictions) > 2 else predictions[0].flatten()
        
        # Binary predictions
        p_thr_binary = (p_thr >= 0.5).astype(int)
        t_thr = (self.y_test >= self.threshold).astype(int)
        
        # Sanal kasa simülasyonu
        wallet = self.starting_capital
        total_bets = 0
        total_wins = 0
        total_losses = 0
        
        for i in range(len(p_thr_binary)):
            model_pred = p_thr_binary[i]
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
            self.best_epoch = epoch + 1
        
        # Emoji seçimi
        if profit_loss > 100:
            wallet_emoji = "🚀"
        elif profit_loss > 0:
            wallet_emoji = "✅"
        elif profit_loss > -100:
            wallet_emoji = "⚠️"
        else:
            wallet_emoji = "❌"
        
        # Her epoch için kısa rapor
        print(f"\n{'='*80}")
        print(f"💰 {self.stage_name} - Epoch {epoch+1} - SANAL KASA SİMÜLASYONU")
        print(f"{'='*80}")
        print(f"   🎲 Oyun: {total_bets} el ({total_wins} kazanç, {total_losses} kayıp)")
        print(f"   📊 Kazanma Oranı: {win_rate:.1f}%")
        print(f"   💰 Başlangıç: {self.starting_capital:,.0f} TL → Final: {wallet:,.0f} TL")
        print(f"   📈 Net: {profit_loss:+,.0f} TL | ROI: {roi:+.2f}% {wallet_emoji}")
        
        if win_rate >= 66.7:
            print(f"   ✅ Kazanma oranı başabaş noktasının ÜSTÜNDE (%66.7)")
        else:
            print(f"   ⚠️ Kazanma oranı başabaş noktasının ALTINDA (Hedef: %66.7)")
        
        print(f"   🏆 En İyi: Epoch {self.best_epoch} (ROI: {self.best_roi:+.2f}%)")
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