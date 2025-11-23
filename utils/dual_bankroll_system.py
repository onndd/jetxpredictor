"""
JetX Predictor - Dual Bankroll System (v2.0)

İki ayrı sanal kasa sistemi (Normal ve Rolling Mod):
1. KASA 1 (NORMAL): 0.85 Güven, Dinamik Çıkış
2. KASA 2 (ROLLING): 0.95 Güven, Sabit 1.5x Çıkış

GÜNCELLEME:
- 2 Modlu Yapı entegre edildi.
- Threshold Manager'dan eşikler alınıyor.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from utils.threshold_manager import get_threshold_manager

class DualBankrollSystem:
    """
    İki ayrı sanal kasa sistemiyle tahmin performansını değerlendirir (2 Modlu).
    """
    
    def __init__(
        self,
        test_predictions: np.ndarray,
        actual_values: np.ndarray,
        bet_amount: float = 10.0,
        confidence_scores: Optional[np.ndarray] = None
    ):
        """
        Args:
            test_predictions: Model tahminleri (regresyon değerleri)
            actual_values: Gerçek değerler
            bet_amount: Her oyunda yatırılacak bahis miktarı (TL)
            confidence_scores: Model güven skorları (0-1 arası, classifier olasılıkları)
        """
        self.test_predictions = test_predictions
        self.actual_values = actual_values
        self.bet_amount = bet_amount
        self.confidence_scores = confidence_scores
        
        # Threshold Manager'dan eşikleri al
        tm = get_threshold_manager()
        self.THRESHOLD_NORMAL = tm.get_normal_threshold()   # 0.85
        self.THRESHOLD_ROLLING = tm.get_rolling_threshold() # 0.95
        
        # Dinamik kasa miktarı
        test_count = len(actual_values)
        self.initial_bankroll = test_count * bet_amount
        
        # Kasa sonuçları
        self.kasa1_results = None
        self.kasa2_results = None
    
    def run_normal_mode_simulation(self) -> Dict:
        """
        KASA 1: NORMAL MOD SİMÜLASYONU
        Kural: Güven >= 0.85 ise Oyna. Çıkış: Dinamik (Regresyon Tahmini * 0.8, Max 2.5x)
        """
        wallet = self.initial_bankroll
        total_bets = 0
        total_wins = 0
        total_losses = 0
        skipped_low_confidence = 0
        
        for i in range(len(self.actual_values)):
            confidence = self.confidence_scores[i] if self.confidence_scores is not None else 0.5
            
            # Güven Eşiği Kontrolü (0.85)
            if confidence < self.THRESHOLD_NORMAL:
                skipped_low_confidence += 1
                continue
            
            # Bahis Yap
            wallet -= self.bet_amount
            total_bets += 1
            
            # Dinamik Çıkış Noktası
            # Regresyon tahmini varsa kullan, yoksa 1.5 varsay
            pred_val = self.test_predictions[i] if self.test_predictions is not None else 1.5
            exit_point = min(max(1.5, pred_val * 0.8), 2.5)
            
            if self.actual_values[i] >= exit_point:
                wallet += self.bet_amount * exit_point
                total_wins += 1
            else:
                total_losses += 1
        
        profit_loss = wallet - self.initial_bankroll
        roi = (profit_loss / self.initial_bankroll) * 100
        win_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0
        
        self.kasa1_results = {
            'name': f'KASA 1: NORMAL MOD (≥{self.THRESHOLD_NORMAL})',
            'strategy': 'Dinamik Çıkış (Max 2.5x)',
            'initial_bankroll': self.initial_bankroll,
            'final_wallet': wallet,
            'total_bets': total_bets,
            'total_wins': total_wins,
            'total_losses': total_losses,
            'win_rate': win_rate,
            'profit_loss': profit_loss,
            'roi': roi,
            'skipped': skipped_low_confidence
        }
        return self.kasa1_results

    def run_rolling_mode_simulation(self) -> Dict:
        """
        KASA 2: ROLLING MOD SİMÜLASYONU
        Kural: Güven >= 0.95 ise Oyna. Çıkış: Sabit 1.50x
        """
        wallet = self.initial_bankroll
        total_bets = 0
        total_wins = 0
        total_losses = 0
        skipped_low_confidence = 0
        exit_point = 1.50 # Sabit
        
        for i in range(len(self.actual_values)):
            confidence = self.confidence_scores[i] if self.confidence_scores is not None else 0.5
            
            # Güven Eşiği Kontrolü (0.95)
            if confidence < self.THRESHOLD_ROLLING:
                skipped_low_confidence += 1
                continue
            
            # Bahis Yap
            wallet -= self.bet_amount
            total_bets += 1
            
            if self.actual_values[i] >= exit_point:
                wallet += self.bet_amount * exit_point
                total_wins += 1
            else:
                total_losses += 1
                
        profit_loss = wallet - self.initial_bankroll
        roi = (profit_loss / self.initial_bankroll) * 100
        win_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0
        
        self.kasa2_results = {
            'name': f'KASA 2: ROLLING MOD (≥{self.THRESHOLD_ROLLING})',
            'strategy': 'Sabit 1.50x Çıkış',
            'initial_bankroll': self.initial_bankroll,
            'final_wallet': wallet,
            'total_bets': total_bets,
            'total_wins': total_wins,
            'total_losses': total_losses,
            'win_rate': win_rate,
            'profit_loss': profit_loss,
            'roi': roi,
            'skipped': skipped_low_confidence
        }
        return self.kasa2_results

    def run_both_simulations(self) -> Tuple[Dict, Dict]:
        """Her iki simülasyonu da çalıştır"""
        kasa1 = self.run_normal_mode_simulation()
        kasa2 = self.run_rolling_mode_simulation()
        return kasa1, kasa2

    def print_detailed_report(self):
        """Detaylı rapor yazdır"""
        if self.kasa1_results is None or self.kasa2_results is None:
            print("❌ Henüz simülasyon çalıştırılmadı!")
            return
            
        print("\n" + "="*80)
        print("💰 ÇİFT SANAL KASA SİMÜLASYONU RAPORU (2 MODLU)")
        print("="*80)
        
        self._print_kasa_report(self.kasa1_results)
        self._print_kasa_report(self.kasa2_results)
        
        # Karşılaştırma
        roi1 = self.kasa1_results['roi']
        roi2 = self.kasa2_results['roi']
        
        print("-" * 80)
        if roi1 > roi2:
            print(f"🏆 KASA 1 (NORMAL) Daha Karlı! Fark: {roi1 - roi2:+.2f}% ROI")
        elif roi2 > roi1:
            print(f"🏆 KASA 2 (ROLLING) Daha Karlı! Fark: {roi2 - roi1:+.2f}% ROI")
        else:
            print("⚖️ İki kasa eşit performans gösterdi.")
        print("="*80)

    def _print_kasa_report(self, results: Dict):
        print(f"\n📌 {results['name']}")
        print(f"   Strateji: {results['strategy']}")
        print(f"   Bahis Sayısı: {results['total_bets']} (Atlanan: {results['skipped']})")
        print(f"   Kazanma Oranı: %{results['win_rate']:.1f}")
        
        emoji = "🚀" if results['roi'] > 0 else "❌"
        print(f"   Net Kar/Zarar: {results['profit_loss']:+,.2f} TL")
        print(f"   ROI: {results['roi']:+.2f}% {emoji}")


def simulate_dual_bankroll(
    test_predictions: np.ndarray,
    actual_values: np.ndarray,
    threshold_predictions: np.ndarray = None, # Legacy support
    bet_amount: float = 10.0,
    confidence_scores: Optional[np.ndarray] = None,
    print_report: bool = True,
    **kwargs # Legacy argumentleri yutmak için
) -> Tuple[Dict, Dict]:
    """
    Helper function: Çift kasa simülasyonunu tek satırda çalıştırır.
    """
    system = DualBankrollSystem(
        test_predictions=test_predictions,
        actual_values=actual_values,
        bet_amount=bet_amount,
        confidence_scores=confidence_scores
    )
    
    kasa1, kasa2 = system.run_both_simulations()
    
    if print_report:
        system.print_detailed_report()
        
    return kasa1, kasa2

if __name__ == "__main__":
    # Test
    print("Dual Bankroll System Test...")
    preds = np.random.uniform(1.0, 3.0, 100)
    actuals = np.random.uniform(1.0, 3.0, 100)
    confs = np.random.uniform(0.80, 0.99, 100)
    
    simulate_dual_bankroll(preds, actuals, confidence_scores=confs)
