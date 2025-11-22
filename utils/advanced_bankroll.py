"""
JetX Predictor - Advanced Bankroll Manager

Gelişmiş sanal kasa yönetimi:
- Kelly Criterion (optimal bahis hesaplama)
- Risk tolerance seviyeleri (conservative, moderate, aggressive)
- Stop-loss & Take-profit otomasyonu
- Streak tracking
- Detaylı performans raporları
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


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
    
    Features:
    - Kelly Criterion ile optimal bahis hesaplama
    - Risk tolerance stratejileri (conservative, moderate, aggressive)
    - Stop-loss & Take-profit kuralları
    - Streak tracking (en iyi/en kötü seriler)
    - Detaylı performans raporları
    """
    
    # Strateji tanımları
    # Strateji tanımları
    STRATEGIES = {
        'rolling': {  # Eskiden conservative idi, rolling ile eşleştirdik
            'name': 'Rolling / Kasa Katlama',
            'max_bet_ratio': 0.05,
            'kelly_fraction': 0.25,
            'stop_loss': 0.20,
            'take_profit': 0.50,
            'min_confidence': 0.95       # %95 olarak güncellendi
        },
        'normal': {   # Eskiden moderate idi
            'name': 'Normal',
            'max_bet_ratio': 0.10,
            'kelly_fraction': 0.50,
            'stop_loss': 0.30,
            'take_profit': 1.00,
            'min_confidence': 0.85       # %85 olarak güncellendi
        }
        # Aggressive tamamen silindi
    }
    
    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        risk_tolerance: str = 'moderate',
        win_multiplier: float = 1.5,
        min_bet: float = 10.0
    ):
        """
        Args:
            initial_bankroll: Başlangıç kasası (TL)
            risk_tolerance: Risk toleransı ('conservative', 'moderate', 'aggressive')
            win_multiplier: Kazanç çarpanı (default: 1.5x)
            min_bet: Minimum bahis miktarı (TL)
        """
        if risk_tolerance not in self.STRATEGIES:
            raise ValueError(f"Geçersiz risk tolerance: {risk_tolerance}. Geçerli değerler: {list(self.STRATEGIES.keys())}")
        
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
            'best_streak': 0,          # En iyi kazanma serisi
            'worst_streak': 0,         # En kötü kaybetme serisi
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
        - f: Bahis oranı (bankroll'un kaçta kaçı)
        - p: Kazanma olasılığı
        - b: Kazanç oranı (1.5x için 0.5)
        - q: Kaybetme olasılığı (1 - p)
        
        Args:
            win_prob: Kazanma olasılığı (0-1 arası)
            win_multiplier: Kazanç çarpanı (None ise self.win_multiplier kullanılır)
            loss: Kayıp miktarı (genelde 1.0 - bahsin tamamı)
            
        Returns:
            Optimal bahis oranı (0-1 arası)
        """
        if win_multiplier is None:
            win_multiplier = self.win_multiplier
        
        # Kazanç oranı (net profit / bet)
        b = win_multiplier - loss
        
        # Kaybetme olasılığı
        q = 1 - win_prob
        
        # Kelly fraction
        kelly_fraction = (win_prob * b - q) / b
        
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
        
        Args:
            confidence: Model güven skoru (0-1 arası)
            predicted_value: Tahmin edilen değer (None ise self.win_multiplier kullanılır)
            volatility_risk: Volatilite risk skoru (0-1 arası, default: 0.0)
            
        Returns:
            Optimal bahis miktarı (TL)
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
        # Kelly Criterion hesaplandıktan SONRA volatilite riskine göre pozisyon küçült
        if volatility_risk > 0.7:
            # YÜKSEK RİSK: Bahis miktarını %80 azalt (%20'sini al)
            bet_size = bet_size * 0.20
        elif volatility_risk > 0.5:
            # ORTA RİSK: Bahis miktarını %50 azalt (%50'sini al)
            bet_size = bet_size * 0.50
        # Diğer durumlarda hesaplanan miktar korunur
        
        return bet_size
    
    def should_stop(self) -> Tuple[bool, Optional[str]]:
        """
        Stop-loss veya take-profit kontrolü
        
        Returns:
            (should_stop, reason) tuple
            - should_stop: Durmalı mı?
            - reason: Durma nedeni (None ise durmamalı)
        """
        profit_ratio = (self.current_bankroll - self.initial_bankroll) / self.initial_bankroll
        
        # Stop-loss: Çok kaybettik mi?
        if profit_ratio <= -self.strategy['stop_loss']:
            return True, f"STOP-LOSS: %{self.strategy['stop_loss']*100:.0f} kayıp ({self.current_bankroll:.2f} TL kaldı)"
        
        # Take-profit: Yeterince kazandık mı?
        if profit_ratio >= self.strategy['take_profit']:
            return True, f"TAKE-PROFIT: %{self.strategy['take_profit']*100:.0f} kar ({self.current_bankroll:.2f} TL oldu)"
        
        # Bankroll çok düştü mü? (minimum bahis bile yapamıyoruz)
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
        """
        Bahis yap ve sonuçları kaydet
        
        Args:
            bet_size: Bahis miktarı (TL)
            predicted_value: Tahmin edilen değer
            actual_value: Gerçekleşen değer
            confidence: Model güven skoru
            
        Returns:
            BetResult objesi
        """
        # Bahis yap
        self.current_bankroll -= bet_size
        self.stats['total_bets'] += 1
        self.stats['total_wagered'] += bet_size
        
        # Kazandık mı?
        if actual_value >= 1.5:
            # Kazanç: bet_size * win_multiplier
            winnings = bet_size * self.win_multiplier
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
        
        # Geçmişe ekle
        self.bet_history.append(bet_result)
        
        return bet_result
    
    def get_report(self) -> Dict:
        """
        Detaylı performans raporu
        
        Returns:
            Performans metrikleri dictionary
        """
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
        
        print(f"\n💰 BANKROLL:")
        print(f"  Başlangıç: {report['initial_bankroll']:,.2f} TL")
        print(f"  Güncel: {report['current_bankroll']:,.2f} TL")
        print(f"  En Yüksek: {report['highest_bankroll']:,.2f} TL")
        print(f"  En Düşük: {report['lowest_bankroll']:,.2f} TL")
        
        print(f"\n📈 PERFORMANS:")
        print(f"  Kar/Zarar: {report['total_profit']:+,.2f} TL")
        
        # ROI emoji
        roi_emoji = "🚀" if report['roi'] > 50 else "✅" if report['roi'] > 0 else "⚠️" if report['roi'] > -20 else "❌"
        print(f"  ROI: {report['roi']:+.1f}% {roi_emoji}")
        
        print(f"\n🎯 BAHİSLER:")
        print(f"  Toplam Bahis: {report['total_bets']}")
        print(f"  Toplam Yatırılan: {report['total_wagered']:,.2f} TL")
        print(f"  Ortalama Bahis: {report['average_bet']:,.2f} TL")
        
        print(f"\n🎲 SONUÇLAR:")
        print(f"  Kazanan: {report['wins']} ({report['win_rate']:.1f}%)")
        print(f"  Kaybeden: {report['losses']} ({100-report['win_rate']:.1f}%)")
        
        print(f"\n📊 SERİLER:")
        print(f"  Şu Anki: {report['current_streak']:+d}")
        print(f"  En İyi: +{report['best_streak']}")
        print(f"  En Kötü: {report['worst_streak']}")
        
        # Değerlendirme
        print(f"\n💡 DEĞERLENDİRME:")
        if report['roi'] > 50:
            print(f"  🚀 Mükemmel! Çok iyi performans!")
        elif report['roi'] > 20:
            print(f"  ✅ Harika! Karlı strateji!")
        elif report['roi'] > 0:
            print(f"  ✅ İyi! Karda!")
        elif report['roi'] > -10:
            print(f"  ⚠️ Nötr. Başa baş yakın.")
        elif report['roi'] > -20:
            print(f"  ⚠️ Dikkat! Hafif kayıpta.")
        else:
            print(f"  ❌ Tehlike! Ciddi kayıpta!")
        
        # Kazanma oranı değerlendirmesi
        if report['win_rate'] >= 67:
            print(f"  ✅ Kazanma oranı mükemmel! (%{report['win_rate']:.1f})")
        elif report['win_rate'] >= 60:
            print(f"  ✅ Kazanma oranı iyi! (%{report['win_rate']:.1f})")
        else:
            print(f"  ⚠️ Kazanma oranı düşük. (%{report['win_rate']:.1f}) Hedef: %67+")
        
        print("="*70 + "\n")
    
    def get_last_n_bets(self, n: int = 10) -> list:
        """Son N bahisi döndür"""
        return self.bet_history[-n:] if len(self.bet_history) >= n else self.bet_history


# Kullanım örneği
if __name__ == "__main__":
    print("="*70)
    print("💰 ADVANCED BANKROLL MANAGER - TEST")
    print("="*70)
    
    # 3 farklı risk tolerance ile test
    for risk_tolerance in ['conservative', 'moderate', 'aggressive']:
        print(f"\n{'='*70}")
        print(f"🎯 {risk_tolerance.upper()} STRATEJİ TESTİ")
        print(f"{'='*70}")
        
        # Manager oluştur
        manager = AdvancedBankrollManager(
            initial_bankroll=1000.0,
            risk_tolerance=risk_tolerance
        )
        
        # Simülasyon: 100 bahis
        np.random.seed(42)
        
        for i in range(100):
            # Rastgele confidence ve actual value
            confidence = np.random.uniform(0.5, 0.95)
            actual_value = np.random.choice([
                np.random.uniform(1.0, 1.49),  # %35 1.5 altı
                np.random.uniform(1.5, 10.0)   # %65 1.5 üstü
            ], p=[0.35, 0.65])
            
            # Optimal bahis hesapla
            bet_size = manager.calculate_bet_size(
                confidence=confidence,
                predicted_value=1.5
            )
            
            # Bahis yap (model confidence yüksekse)
            if bet_size > 0:
                result = manager.place_bet(
                    bet_size=bet_size,
                    predicted_value=1.5,
                    actual_value=actual_value,
                    confidence=confidence
                )
            
            # Stop-loss veya take-profit kontrolü
            should_stop, reason = manager.should_stop()
            if should_stop:
                print(f"\n⚠️ {reason} - Simülasyon durduruluyor (Bahis {i+1})")
                break
        
        # Rapor
        manager.print_report()
    
    print("="*70)
    print("✅ Test tamamlandı!")
    print("="*70)
