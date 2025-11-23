"""
JetX Predictor - Advanced Bankroll Manager

Gelişmiş sanal kasa yönetimi (2 Modlu):
- Kelly Criterion (optimal bahis hesaplama)
- Risk tolerance seviyeleri:
  1. Normal Mod (0.85 Güven, Dengeli)
  2. Rolling Mod (0.95 Güven, Agresif Büyüme)
- Stop-loss & Take-profit otomasyonu
- Streak tracking
- Detaylı performans raporları
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from utils.threshold_manager import get_threshold

@dataclass
class BetResult:
    """Tek bir bahis sonucu"""
    bet_size: float
    predicted_value: float
    actual_value: float
    confidence: float
    result: str  # 'WIN' or 'LOSS'
    profit: float
    new_bankroll: float
    timestamp: str


class AdvancedBankrollManager:
    """
    Gelişmiş bankroll yönetimi sınıfı
    """
    
    # Strateji tanımları (2 MODLU)
    STRATEGIES = {
        'normal': {
            'name': 'Normal Mod',
            'max_bet_ratio': 0.10,      # Kasasının maks %10'u
            'kelly_fraction': 0.50,     # Kelly'nin yarısı (Dengeli)
            'stop_loss': 0.30,          # %30 kayıpta dur
            'take_profit': 1.00,        # Kasayı 2'ye katlayınca dur/kar al
            'min_confidence': 0.85      # Eşik: 0.85
        },
        'rolling': {
            'name': 'Rolling / Kasa Katlama',
            'max_bet_ratio': 0.05,      # Kasasının maks %5'i (Güvenlik için daha düşük)
            'kelly_fraction': 0.25,     # Kelly'nin çeyreği (Daha defansif başla)
            'stop_loss': 0.20,          # %20 kayıpta hemen dur
            'take_profit': 0.50,        # %50 kar hedefi (kısa vadeli)
            'min_confidence': 0.95      # Eşik: 0.95
        }
    }
    
    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        risk_tolerance: str = 'normal',
        win_multiplier: float = 1.5,
        min_bet: float = 10.0
    ):
        """
        Args:
            initial_bankroll: Başlangıç kasası (TL)
            risk_tolerance: Risk toleransı ('normal', 'rolling')
            win_multiplier: Kazanç çarpanı (default: 1.5x)
            min_bet: Minimum bahis miktarı (TL)
        """
        # Eğer geçersiz bir mod gelirse varsayılan olarak 'normal' seç
        if risk_tolerance not in self.STRATEGIES:
            print(f"⚠️ Geçersiz risk tolerance: {risk_tolerance}. 'normal' moda geçiliyor.")
            risk_tolerance = 'normal'
        
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.risk_tolerance = risk_tolerance
        self.win_multiplier = win_multiplier
        self.min_bet = min_bet
        
        # Strateji ayarları
        self.strategy = self.STRATEGIES[risk_tolerance]
        
        # İstatistikler
        self.stats = {
            'total_bets': 0,
            'wins': 0,
            'losses': 0,
            'current_streak': 0,      # Pozitif = kazanma serisi, negatif = kaybetme serisi
            'best_streak': 0,         # En iyi kazanma serisi
            'worst_streak': 0,        # En kötü kaybetme serisi
            'total_profit': 0.0,
            'total_wagered': 0.0,
            'roi': 0.0,
            'highest_bankroll': initial_bankroll,
            'lowest_bankroll': initial_bankroll
        }
        
        # Geçmiş
        self.bet_history = []
    
    def kelly_criterion(
        self,
        win_prob: float,
        win_multiplier: Optional[float] = None,
        loss: float = 1.0
    ) -> float:
        """
        Kelly Criterion ile optimal bahis oranını hesapla
        Formül: f = (p * b - q) / b
        """
        if win_multiplier is None:
            win_multiplier = self.win_multiplier
        
        # Kazanç oranı (net profit / bet)
        b = win_multiplier - loss
        
        # Kaybetme olasılığı
        q = 1 - win_prob
        
        # Kelly fraction
        if b > 0:
            kelly_fraction = (win_prob * b - q) / b
        else:
            kelly_fraction = 0
        
        # Güvenlik: Negatif veya çok yüksek fraksiyonları sınırla
        kelly_fraction = max(0, min(kelly_fraction, 0.25))
        
        # Risk tolerance'a göre ayarla
        adjusted_fraction = kelly_fraction * self.strategy['kelly_fraction']
        
        # Maksimum bahis oranını aşma
        adjusted_fraction = min(adjusted_fraction, self.strategy['max_bet_ratio'])
        
        return adjusted_fraction
    
    def calculate_bet_size(
        self,
        confidence: float,
        predicted_value: Optional[float] = None,
        volatility_risk: float = 0.0
    ) -> float:
        """
        Güven ve tahmine göre optimal bahis miktarını hesapla
        """
        # Minimum güven kontrolü
        if confidence < self.strategy['min_confidence']:
            return 0.0  # Bahis yapma
        
        # Kelly Criterion ile optimal oran hesapla
        kelly_frac = self.kelly_criterion(
            win_prob=confidence,
            win_multiplier=predicted_value if predicted_value else self.win_multiplier
        )
        
        # Bahis miktarı
        bet_size = self.current_bankroll * kelly_frac
        
        # Minimum ve maksimum bahis sınırları
        max_bet = self.current_bankroll * self.strategy['max_bet_ratio']
        
        bet_size = max(self.min_bet, min(bet_size, max_bet))
        
        # Bankroll yetersizse 0 döndür
        if bet_size > self.current_bankroll:
            return 0.0
        
        # 🛡️ VOLATİLİTE BAZLI POZİSYON KÜÇÜLTME - GÜVENLİK KATMANI
        if volatility_risk > 0.7:
            # YÜKSEK RİSK: Bahis miktarını %80 azalt
            bet_size = bet_size * 0.20
        elif volatility_risk > 0.5:
            # ORTA RİSK: Bahis miktarını %50 azalt
            bet_size = bet_size * 0.50
        
        return bet_size
    
    def should_stop(self) -> Tuple[bool, Optional[str]]:
        """Stop-loss veya take-profit kontrolü"""
        profit_ratio = (self.current_bankroll - self.initial_bankroll) / self.initial_bankroll
        
        # Stop-loss: Çok kaybettik mi?
        if profit_ratio <= -self.strategy['stop_loss']:
            return True, f"STOP-LOSS: %{self.strategy['stop_loss']*100:.0f} kayıp ({self.current_bankroll:.2f} TL kaldı)"
        
        # Take-profit: Yeterince kazandık mı?
        if profit_ratio >= self.strategy['take_profit']:
            return True, f"TAKE-PROFIT: %{self.strategy['take_profit']*100:.0f} kar ({self.current_bankroll:.2f} TL oldu)"
        
        # Bankroll çok düştü mü?
        if self.current_bankroll < self.min_bet:
            return True, f"BANKROLL BITTI: {self.current_bankroll:.2f} TL kaldı (min: {self.min_bet} TL)"
        
        return False, None
    
    def place_bet(
        self,
        bet_size: float,
        predicted_value: float,
        actual_value: float,
        confidence: float
    ) -> BetResult:
        """Bahis yap ve sonuçları kaydet"""
        # Bahis yap
        self.current_bankroll -= bet_size
        self.stats['total_bets'] += 1
        self.stats['total_wagered'] += bet_size
        
        # Kazandık mı? (1.5x sabit çıkış varsayımı, stratejiye göre değişebilir)
        # Rolling modda 1.5x sabit, Normal modda predicted_value'ya göre dinamik olabilir
        # Burada basitlik için 1.5x varsayıyoruz, çağıran kod exit_point belirlemeli
        # Şimdilik varsayılan 1.5x kontrolü:
        target_multiplier = 1.5
        
        if actual_value >= target_multiplier:
            # Kazanç
            winnings = bet_size * target_multiplier
            self.current_bankroll += winnings
            profit = winnings - bet_size
            result = 'WIN'
            
            # İstatistikleri güncelle
            self.stats['wins'] += 1
            self.stats['current_streak'] = max(0, self.stats['current_streak']) + 1
            self.stats['best_streak'] = max(self.stats['best_streak'], self.stats['current_streak'])
        else:
            # Kaybettik
            profit = -bet_size
            result = 'LOSS'
            
            # İstatistikleri güncelle
            self.stats['losses'] += 1
            self.stats['current_streak'] = min(0, self.stats['current_streak']) - 1
            self.stats['worst_streak'] = min(self.stats['worst_streak'], self.stats['current_streak'])
        
        # Genel istatistikler
        self.stats['total_profit'] += profit
        self.stats['roi'] = (self.current_bankroll - self.initial_bankroll) / self.initial_bankroll
        self.stats['highest_bankroll'] = max(self.stats['highest_bankroll'], self.current_bankroll)
        self.stats['lowest_bankroll'] = min(self.stats['lowest_bankroll'], self.current_bankroll)
        
        # Sonuç objesi
        bet_result = BetResult(
            bet_size=bet_size,
            predicted_value=predicted_value,
            actual_value=actual_value,
            confidence=confidence,
            result=result,
            profit=profit,
            new_bankroll=self.current_bankroll,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        self.bet_history.append(bet_result)
        return bet_result
    
    def get_report(self) -> Dict:
        """Detaylı performans raporu"""
        win_rate = self.stats['wins'] / self.stats['total_bets'] if self.stats['total_bets'] > 0 else 0
        avg_bet = self.stats['total_wagered'] / self.stats['total_bets'] if self.stats['total_bets'] > 0 else 0
        
        return {
            'strategy': self.strategy['name'],
            'risk_tolerance': self.risk_tolerance,
            'initial_bankroll': self.initial_bankroll,
            'current_bankroll': self.current_bankroll,
            'highest_bankroll': self.stats['highest_bankroll'],
            'lowest_bankroll': self.stats['lowest_bankroll'],
            'total_profit': self.stats['total_profit'],
            'roi': self.stats['roi'] * 100,
            'total_bets': self.stats['total_bets'],
            'total_wagered': self.stats['total_wagered'],
            'average_bet': avg_bet,
            'wins': self.stats['wins'],
            'losses': self.stats['losses'],
            'win_rate': win_rate * 100,
            'current_streak': self.stats['current_streak'],
            'best_streak': self.stats['best_streak'],
            'worst_streak': self.stats['worst_streak']
        }
    
    def print_report(self):
        """Detaylı raporu konsola yazdır"""
        report = self.get_report()
        print("\n" + "="*70)
        print(f"📊 {report['strategy'].upper()} STRATEJİ RAPORU")
        print("="*70)
        print(f"\n💰 BANKROLL: {report['current_bankroll']:,.2f} TL (Başlangıç: {report['initial_bankroll']:,.2f} TL)")
        print(f"📈 PERFORMANS: Kar {report['total_profit']:+,.2f} TL | ROI {report['roi']:+.1f}%")
        print(f"🎯 BAHİSLER: Toplam {report['total_bets']} | Kazanan {report['wins']} | Kaybeden {report['losses']}")
        print(f"🎲 WIN RATE: %{report['win_rate']:.1f}")
        print(f"📊 SERİLER: En İyi +{report['best_streak']} | En Kötü {report['worst_streak']}")
        print("="*70 + "\n")


if __name__ == "__main__":
    # Test
    manager = AdvancedBankrollManager(risk_tolerance='normal')
    print(f"Manager initialized with strategy: {manager.strategy['name']}")
    print(f"Min confidence: {manager.strategy['min_confidence']}")
