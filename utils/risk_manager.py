"""
JetX Predictor - Risk Yönetim Modülü

Bu modül tahmin sonuçlarına göre risk analizi yapar ve
mod bazlı öneriler sunar (Normal, Rolling).

GÜNCELLEME:
- 2 Modlu Yapı (Normal/Rolling) entegre edildi.
- Normal Mod Eşik: 0.85
- Rolling Mod Eşik: 0.95
"""

from typing import Dict, List, Optional
import sys
import os

# Kategori tanımlarını import et
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from category_definitions import CategoryDefinitions
from utils.threshold_manager import get_threshold_manager


class RiskManager:
    """Risk yönetimi ve karar verme sınıfı"""
    
    def __init__(self, mode: str = 'normal'):
        """
        Args:
            mode: Tahmin modu ('normal', 'rolling')
        """
        self.tm = get_threshold_manager()
        self.set_mode(mode)
        
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.last_predictions = []  # Son tahminlerin listesi
        
        # Eşik değerlerini al
        self.THRESHOLD_NORMAL = self.tm.get_normal_threshold()   # 0.85
        self.THRESHOLD_ROLLING = self.tm.get_rolling_threshold() # 0.95
        
    def set_mode(self, mode: str):
        """
        Modu değiştirir
        
        Args:
            mode: Yeni mod ('normal', 'rolling')
        """
        if mode in ['normal', 'rolling']:
            self.mode = mode
        else:
            # Hatalı mod durumunda varsayılan 'normal'
            print(f"⚠️ Geçersiz mod: {mode}. 'normal' moduna geçiliyor.")
            self.mode = 'normal'
    
    def evaluate_prediction(
        self,
        prediction_result: Dict,
        actual_value: float
    ) -> Dict:
        """
        Tahmin sonucunu değerlendirir ve kayıt tutar
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
        
        # Mod bazlı eşik belirleme
        if self.mode == 'rolling':
            confidence_threshold = self.THRESHOLD_ROLLING # 0.95
        else:
            confidence_threshold = self.THRESHOLD_NORMAL  # 0.85
        
        reasons = []
        should_play = False
        risk_level = 'HIGH'
        
        # 1. Güven kontrolü
        if confidence < confidence_threshold:
            reasons.append(f"Güven seviyesi çok düşük ({confidence:.0%} < {confidence_threshold:.0%})")
        else:
            reasons.append(f"Güven seviyesi yeterli ({confidence:.0%})")
            
        # 2. Eşik kontrolü - Sadece 1.5 üstü tahminlerde oyna
        if not above_threshold:
            reasons.append(f"⚠️ TAHMİN 1.5x ALTINDA ({predicted_value:.2f}x)")
            reasons.append(f"💰 PARA KAYBI RİSKİ YÜKSEK - OYNAMA!")
            should_play = False
            risk_level = 'CRITICAL'
        else:
            reasons.append(f"✅ Tahmin 1.5x üstünde ({predicted_value:.2f}x)")
            
        # 3. Ardışık kayıp kontrolü
        if self.consecutive_losses >= 3:
            reasons.append(f"⚠️ {self.consecutive_losses} ardışık yanlış tahmin - RİSKLİ!")
            # Rolling modda ardışık kayıp varsa durdur
            if self.mode == 'rolling':
                should_play = False
                reasons.append("⛔ Rolling Mod: Seri kayıpta durduruldu.")
        
        # 4. Mod bazlı nihai karar
        if above_threshold and confidence >= confidence_threshold:
            if self.mode == 'rolling':
                should_play = True
                risk_level = 'LOW'
                reasons.append("✅ ROLLING MOD: %95+ Güven sağlandı.")
            else:
                should_play = True
                risk_level = 'MEDIUM'
                reasons.append("✅ NORMAL MOD: %85+ Güven sağlandı.")
        else:
            should_play = False
        
        # 5. Kritik bölge uyarısı (1.45 - 1.55 arası belirsizlik)
        if 1.45 <= predicted_value <= 1.55:
            should_play = False
            risk_level = 'CRITICAL'
            reasons.append("🚨 KRİTİK BÖLGE (1.50 Sınırı)! Risk alma.")
        
        return {
            'should_play': should_play,
            'risk_level': risk_level,
            'reasons': reasons,
            'mode': self.mode,
            'confidence_threshold': confidence_threshold,
            'below_threshold_warning': not above_threshold
        }
    
    def get_betting_suggestion(
        self,
        prediction_result: Dict,
        bankroll: Optional[float] = None
    ) -> Dict:
        """
        Bahis önerisi verir (varsa bankroll ile)
        """
        confidence = prediction_result.get('confidence', 0)
        predicted_value = prediction_result.get('predicted_value', 0)
        
        suggestion = {
            'should_bet': False,
            'suggested_multiplier': 1.5,
            'bet_percentage': 0,
            'reasons': []
        }
        
        play_decision = self.should_play(prediction_result)
        
        if not play_decision['should_play']:
            suggestion['reasons'].append("Oynamayı önermiyoruz")
            return suggestion
        
        suggestion['should_bet'] = True
        
        # Mod bazlı strateji
        if self.mode == 'rolling':
            # Rolling: Sabit 1.50x çıkış, %2 kasa (Daha güvenli)
            suggestion['suggested_multiplier'] = 1.50
            suggestion['bet_percentage'] = 2
            suggestion['reasons'].append("Rolling: 1.50x Sabit Çıkış (Güvenli Liman)")
            
        elif self.mode == 'normal':
            # Normal: Dinamik çıkış (Max 2.5x), %4 kasa
            if predicted_value >= 2.0:
                suggestion['suggested_multiplier'] = min(predicted_value * 0.8, 2.5)
            else:
                suggestion['suggested_multiplier'] = 1.5
            
            suggestion['bet_percentage'] = 4
            suggestion['reasons'].append(f"Normal: {suggestion['suggested_multiplier']:.2f}x Dinamik Çıkış")
        
        # Bankroll varsa miktar hesapla
        if bankroll:
            suggestion['suggested_amount'] = (bankroll * suggestion['bet_percentage']) / 100
            suggestion['reasons'].append(
                f"Önerilen bahis: {suggestion['suggested_amount']:.2f} TL " +
                f"(%{suggestion['bet_percentage']})"
            )
        
        return suggestion
    
    def get_statistics(self) -> Dict:
        """Risk yönetimi istatistikleri"""
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
        """Mevcut duruma göre uyarı seviyesi"""
        if self.consecutive_losses >= 5: return 'DANGER'
        elif self.consecutive_losses >= 3: return 'WARNING'
        elif self.consecutive_losses >= 2: return 'CAUTION'
        else: return 'SAFE'
