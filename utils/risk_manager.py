"""
JetX Predictor - Risk Yönetim Modülü

Bu modül tahmin sonuçlarına göre risk analizi yapar ve
mod bazlı öneriler sunar (Normal, Rolling, Agresif).
"""

from typing import Dict, List, Optional
import sys
import os

# Kategori tanımlarını import et
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from category_definitions import (
    CategoryDefinitions,
    CONFIDENCE_THRESHOLDS
)


class RiskManager:
    """Risk yönetimi ve karar verme sınıfı"""
    
    def __init__(self, mode: str = 'normal'):
        """
        Args:
            mode: Tahmin modu ('normal', 'rolling', 'aggressive')
        """
        self.mode = mode
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.last_predictions = []  # Son tahminlerin listesi
        
    def set_mode(self, mode: str):
        """
        Modu değiştirir
        
        Args:
            mode: Yeni mod ('normal', 'rolling', 'aggressive')
        """
        if mode in ['normal', 'rolling', 'aggressive']:
            self.mode = mode
        else:
            raise ValueError(f"Geçersiz mod: {mode}. 'normal', 'rolling' veya 'aggressive' olmalı.")
    
    def evaluate_prediction(
        self,
        prediction_result: Dict,
        actual_value: float
    ) -> Dict:
        """
        Tahmin sonucunu değerlendirir ve kayıt tutar
        
        Args:
            prediction_result: Predictor'dan gelen tahmin sonucu
            actual_value: Gerçekleşen değer
            
        Returns:
            Değerlendirme sonucu
        """
        predicted_value = prediction_result.get('predicted_value')
        above_threshold = prediction_result.get('above_threshold')
        
        # 1.5x eşik kontrolü
        actual_above_threshold = actual_value >= CategoryDefinitions.CRITICAL_THRESHOLD
        
        # Tahmin doğru muydu?
        threshold_prediction_correct = above_threshold == actual_above_threshold
        
        # Değer tahmini ne kadar yakın?
        if predicted_value:
            value_error = abs(predicted_value - actual_value)
            value_error_percentage = (value_error / actual_value) * 100
        else:
            value_error = None
            value_error_percentage = None
        
        # Ardışık kazanç/kayıp güncelle
        if threshold_prediction_correct:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        
        # Son tahminleri kaydet
        self.last_predictions.append({
            'predicted': predicted_value,
            'actual': actual_value,
            'correct': threshold_prediction_correct
        })
        
        # Son 10 tahmini tut
        if len(self.last_predictions) > 10:
            self.last_predictions.pop(0)
        
        return {
            'threshold_correct': threshold_prediction_correct,
            'value_error': value_error,
            'value_error_percentage': value_error_percentage,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses
        }
    
    def should_play(
        self,
        prediction_result: Dict,
        history: Optional[List[float]] = None
    ) -> Dict:
        """
        Mod bazlı oyun önerisi verir
        
        Args:
            prediction_result: Predictor'dan gelen tahmin
            history: Geçmiş değerler (opsiyonel)
            
        Returns:
            Öneri ve gerekçe
        """
        confidence = prediction_result.get('confidence', 0)
        above_threshold = prediction_result.get('above_threshold', False)
        predicted_value = prediction_result.get('predicted_value', 0)
        
        # Mod bazlı eşik
        confidence_threshold = CONFIDENCE_THRESHOLDS.get(self.mode, 0.65)
        
        reasons = []
        should_play = False
        risk_level = 'HIGH'
        
        # 1. Güven kontrolü
        if confidence < confidence_threshold:
            reasons.append(f"Güven seviyesi çok düşük ({confidence:.0%} < {confidence_threshold:.0%})")
        else:
            reasons.append(f"Güven seviyesi yeterli ({confidence:.0%})")
            
        # 2. Eşik kontrolü
        if not above_threshold:
            reasons.append(f"Tahmin 1.5x altında ({predicted_value:.2f}x) - RİSKLİ")
        else:
            reasons.append(f"Tahmin 1.5x üstünde ({predicted_value:.2f}x)")
            
        # 3. Ardışık kayıp kontrolü
        if self.consecutive_losses >= 3:
            reasons.append(f"⚠️ {self.consecutive_losses} ardışık yanlış tahmin - DUR!")
        
        # 4. Mod bazlı karar
        if self.mode == 'rolling':
            # Rolling: Çok konservatif
            if confidence >= 0.80 and above_threshold and self.consecutive_losses < 2:
                should_play = True
                risk_level = 'LOW'
                reasons.append("✅ Rolling mod: Yüksek güven ve düşük risk")
            else:
                reasons.append("❌ Rolling mod: Koşullar uygun değil - BEKLE")
                
        elif self.mode == 'normal':
            # Normal: Dengeli
            if confidence >= 0.65 and above_threshold and self.consecutive_losses < 3:
                should_play = True
                risk_level = 'MEDIUM'
                reasons.append("✅ Normal mod: Koşullar uygun")
            else:
                reasons.append("❌ Normal mod: Koşullar uygun değil - BEKLE")
                
        elif self.mode == 'aggressive':
            # Aggressive: Risk alır
            if confidence >= 0.50 and above_threshold:
                should_play = True
                risk_level = 'HIGH'
                reasons.append("⚠️ Agresif mod: Riskli ama oynanabilir")
            else:
                reasons.append("❌ Agresif mod bile oynamayı önermez")
        
        # 5. Kritik bölge uyarısı
        if 1.45 <= predicted_value <= 1.55:
            should_play = False
            risk_level = 'CRITICAL'
            reasons.append("🚨 KRİTİK BÖLGE! Kesinlikle oynama!")
        
        return {
            'should_play': should_play,
            'risk_level': risk_level,
            'reasons': reasons,
            'mode': self.mode,
            'confidence_threshold': confidence_threshold
        }
    
    def get_betting_suggestion(
        self,
        prediction_result: Dict,
        bankroll: Optional[float] = None
    ) -> Dict:
        """
        Bahis önerisi verir (varsa bankroll ile)
        
        Args:
            prediction_result: Tahmin sonucu
            bankroll: Mevcut sermaye (opsiyonel)
            
        Returns:
            Bahis önerisi
        """
        confidence = prediction_result.get('confidence', 0)
        predicted_value = prediction_result.get('predicted_value', 0)
        
        # Temel öneri
        suggestion = {
            'should_bet': False,
            'suggested_multiplier': 1.5,  # Varsayılan çıkış noktası
            'bet_percentage': 0,
            'reasons': []
        }
        
        # Oyun kararı
        play_decision = self.should_play(prediction_result)
        
        if not play_decision['should_play']:
            suggestion['reasons'].append("Oynamayı önermiyoruz")
            return suggestion
        
        suggestion['should_bet'] = True
        
        # Mod bazlı bahis stratejisi
        if self.mode == 'rolling':
            # Rolling: Düşük çarpanda çık, sermayeyi koru
            suggestion['suggested_multiplier'] = 1.5
            suggestion['bet_percentage'] = 2 if bankroll else 0
            suggestion['reasons'].append("Rolling: 1.5x'te çık, sermaye koruma öncelikli")
            
        elif self.mode == 'normal':
            # Normal: Tahmine göre karar ver
            if predicted_value >= 2.0:
                suggestion['suggested_multiplier'] = min(predicted_value * 0.8, 2.5)
                suggestion['bet_percentage'] = 3 if bankroll else 0
                suggestion['reasons'].append(f"Normal: Tahmin yüksek, {suggestion['suggested_multiplier']:.1f}x'te çık")
            else:
                suggestion['suggested_multiplier'] = 1.5
                suggestion['bet_percentage'] = 3 if bankroll else 0
                suggestion['reasons'].append("Normal: 1.5x'te güvenli çık")
                
        elif self.mode == 'aggressive':
            # Aggressive: Yüksek risk, yüksek getiri
            if predicted_value >= 3.0:
                suggestion['suggested_multiplier'] = min(predicted_value * 0.85, 5.0)
                suggestion['bet_percentage'] = 5 if bankroll else 0
                suggestion['reasons'].append(f"Agresif: Yüksek hedef {suggestion['suggested_multiplier']:.1f}x")
            else:
                suggestion['suggested_multiplier'] = 2.0
                suggestion['bet_percentage'] = 4 if bankroll else 0
                suggestion['reasons'].append("Agresif: 2.0x hedef")
        
        # Bankroll varsa miktar hesapla
        if bankroll:
            suggestion['suggested_amount'] = (bankroll * suggestion['bet_percentage']) / 100
            suggestion['reasons'].append(
                f"Önerilen bahis: {suggestion['suggested_amount']:.2f} " +
                f"(Sermayenin %{suggestion['bet_percentage']})"
            )
        
        return suggestion
    
    def get_statistics(self) -> Dict:
        """
        Risk yönetimi istatistikleri
        
        Returns:
            İstatistikler
        """
        if not self.last_predictions:
            return {
                'total_predictions': 0,
                'correct_predictions': 0,
                'accuracy': 0,
                'consecutive_wins': self.consecutive_wins,
                'consecutive_losses': self.consecutive_losses
            }
        
        correct = sum(1 for p in self.last_predictions if p['correct'])
        
        return {
            'total_predictions': len(self.last_predictions),
            'correct_predictions': correct,
            'accuracy': correct / len(self.last_predictions),
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'mode': self.mode
        }
    
    def reset_streak(self):
        """Ardışık kazanç/kayıp sayacını sıfırlar"""
        self.consecutive_wins = 0
        self.consecutive_losses = 0
    
    def get_warning_level(self) -> str:
        """
        Mevcut duruma göre uyarı seviyesi döndürür
        
        Returns:
            Uyarı seviyesi ('SAFE', 'CAUTION', 'WARNING', 'DANGER')
        """
        if self.consecutive_losses >= 5:
            return 'DANGER'
        elif self.consecutive_losses >= 3:
            return 'WARNING'
        elif self.consecutive_losses >= 2:
            return 'CAUTION'
        else:
            return 'SAFE'
