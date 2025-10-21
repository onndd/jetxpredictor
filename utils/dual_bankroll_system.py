"""
JetX Predictor - Dual Bankroll System (Optimize Edildi)

İki ayrı sanal kasa sistemi:
1. KASA 1: 1.5x Eşik Sistemi (güven skoru filtreli)
2. KASA 2: %70 Çıkış Sistemi (optimize edildi - 2x+ tahminler için)

YENİ ÖZELLİKLER:
- Güven skoru filtresi: Sadece %90+ güvenli tahminlerde bahis yap
- Kasa 2 optimize: %80 → %70 (daha erken, daha güvenli çıkış)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class DualBankrollSystem:
    """
    İki ayrı sanal kasa sistemiyle tahmin performansını değerlendirir
    
    Kasa 1: 1.5x eşik - Model 1.5x üstü tahmin ederse 1.5x'te çıkış (güven skoru filtreli)
    Kasa 2: %70 çıkış - Model 2x+ tahmin ederse tahmin×0.70'de çıkış (optimize edildi)
    
    YENİ: Güven skoru filtresi ile sadece yüksek güvenli tahminlerde bahis yap
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
            test_predictions: Model tahminleri (değerler)
            actual_values: Gerçek değerler
            bet_amount: Her oyunda yatırılacak bahis miktarı (TL)
            confidence_scores: Opsiyonel - Model güven skorları (0-1 arası)
                              Sadece yüksek güvenli tahminlerde bahis yapmak için
        """
        self.test_predictions = test_predictions
        self.actual_values = actual_values
        self.bet_amount = bet_amount
        self.confidence_scores = confidence_scores
        
        # Dinamik kasa miktarı: Test veri sayısı × Bahis miktarı
        test_count = len(actual_values)
        self.initial_bankroll = test_count * bet_amount
        
        # Kasa sonuçları
        self.kasa1_results = None
        self.kasa2_results = None
    
    def run_kasa1_simulation(
        self,
        threshold_predictions: np.ndarray,
        confidence_threshold: float = 0.0
    ) -> Dict:
        """
        KASA 1: 1.5x Eşik Sistemi (Güven Skoru Filtreli)
        
        Model "1.5x üstü" tahmin ederse → 1.5x'te çıkış yap
        YENİ: Güven skoru filtresi ile sadece yüksek güvenli tahminlerde bahis yap
        
        Args:
            threshold_predictions: Threshold tahminleri (0 veya 1)
                                   1 = 1.5x üstü tahmin
                                   0 = 1.5x altı tahmin
            confidence_threshold: Minimum güven skoru (0-1 arası)
                                 Örnek: 0.9 = %90+ güvenle bahis yap
                                 Varsayılan: 0.0 (tüm tahminlerde bahis yap)
        
        Returns:
            Kasa 1 sonuçları dictionary
        """
        wallet = self.initial_bankroll
        total_bets = 0
        total_wins = 0
        total_losses = 0
        skipped_low_confidence = 0  # Düşük güven nedeniyle atlanan
        
        exit_point = 1.5  # Sabit çıkış noktası
        win_amount = self.bet_amount * exit_point  # 1.5 × 10 TL = 15 TL
        
        for i in range(len(self.actual_values)):
            model_pred = threshold_predictions[i]  # 0 veya 1
            actual_value = self.actual_values[i]
            
            # Güven skoru kontrolü (eğer sağlanmışsa)
            if self.confidence_scores is not None and confidence_threshold > 0:
                confidence = self.confidence_scores[i]
                if confidence < confidence_threshold:
                    # Güven düşük, bu oyunu atla
                    if model_pred == 1:
                        skipped_low_confidence += 1
                    continue
            
            # Model "1.5 üstü" tahmin ediyorsa bahis yap
            if model_pred == 1:
                wallet -= self.bet_amount  # Bahis yap
                total_bets += 1
                
                # Gerçek değer çıkış noktasından büyük veya eşitse kazandık
                if actual_value >= exit_point:
                    # Kazandık! 15 TL geri al
                    wallet += win_amount
                    total_wins += 1
                else:
                    # Kaybettik (bahis zaten kesildi)
                    total_losses += 1
        
        # Sonuçları hesapla
        profit_loss = wallet - self.initial_bankroll
        roi = (profit_loss / self.initial_bankroll) * 100
        win_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0
        accuracy = win_rate  # Kazanma oranı = Doğruluk
        
        # Strateji açıklaması
        if confidence_threshold > 0:
            strategy = f'Model 1.5x üstü tahmin ederse → 1.5x\'te çıkış (Güven: ≥{confidence_threshold*100:.0f}%)'
        else:
            strategy = 'Model 1.5x üstü tahmin ederse → 1.5x\'te çıkış'
        
        self.kasa1_results = {
            'name': 'KASA 1: 1.5x Eşik Sistemi',
            'strategy': strategy,
            'initial_bankroll': self.initial_bankroll,
            'final_wallet': wallet,
            'total_bets': total_bets,
            'total_wins': total_wins,
            'total_losses': total_losses,
            'win_rate': win_rate,
            'accuracy': accuracy,
            'profit_loss': profit_loss,
            'roi': roi,
            'exit_point': exit_point,
            'confidence_threshold': confidence_threshold,
            'skipped_low_confidence': skipped_low_confidence
        }
        
        return self.kasa1_results
    
    def run_kasa2_simulation(
        self,
        exit_multiplier: float = 0.70,
        confidence_threshold: float = 0.0
    ) -> Dict:
        """
        KASA 2: Dinamik Çıkış Sistemi (Optimize Edildi)
        
        Model 2.0x+ tahmin ederse → Tahmin × exit_multiplier'de çıkış yap
        OPTİMİZE: %80 → %70 (daha erken, daha güvenli çıkış)
        YENİ: Güven skoru filtresi ile sadece yüksek güvenli tahminlerde bahis yap
        
        Args:
            exit_multiplier: Çıkış noktası çarpanı (varsayılan: 0.70 = %70)
                            Örnek: Tahmin 3.0x ise → 3.0 × 0.70 = 2.1x'te çık
            confidence_threshold: Minimum güven skoru (0-1 arası)
                                 Örnek: 0.9 = %90+ güvenle bahis yap
                                 Varsayılan: 0.0 (tüm tahminlerde bahis yap)
        
        Returns:
            Kasa 2 sonuçları dictionary
        """
        wallet = self.initial_bankroll
        total_bets = 0
        total_wins = 0
        total_losses = 0
        exit_points = []  # Çıkış noktalarını kaydet
        skipped_low_confidence = 0  # Düşük güven nedeniyle atlanan
        
        for i in range(len(self.actual_values)):
            model_pred_value = self.test_predictions[i]  # Tahmin edilen değer
            actual_value = self.actual_values[i]
            
            # SADECE 2.0x ve üzeri tahminlerde oyna
            if model_pred_value >= 2.0:
                # Güven skoru kontrolü (eğer sağlanmışsa)
                if self.confidence_scores is not None and confidence_threshold > 0:
                    confidence = self.confidence_scores[i]
                    if confidence < confidence_threshold:
                        # Güven düşük, bu oyunu atla
                        skipped_low_confidence += 1
                        continue
                
                wallet -= self.bet_amount  # Bahis yap
                total_bets += 1
                
                # Çıkış noktası: Tahmin × exit_multiplier
                # ÖNEMLİ: Minimum 1.5x olmalı (kar eşiği)
                exit_point = max(1.5, model_pred_value * exit_multiplier)
                exit_points.append(exit_point)
                
                # Gerçek değer çıkış noktasından büyük veya eşitse kazandık
                if actual_value >= exit_point:
                    # Kazandık! exit_point × bet_amount geri al
                    win_amount = exit_point * self.bet_amount
                    wallet += win_amount
                    total_wins += 1
                else:
                    # Kaybettik (bahis zaten kesildi)
                    total_losses += 1
        
        # Sonuçları hesapla
        profit_loss = wallet - self.initial_bankroll
        roi = (profit_loss / self.initial_bankroll) * 100
        win_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0
        accuracy = win_rate  # Kazanma oranı = Doğruluk
        avg_exit_point = np.mean(exit_points) if exit_points else 0
        
        # Strateji açıklaması
        if confidence_threshold > 0:
            strategy = f'Model 2x+ tahmin ederse → Tahmin × {exit_multiplier:.0%}\'de çıkış (Güven: ≥{confidence_threshold*100:.0f}%)'
        else:
            strategy = f'Model 2x+ tahmin ederse → Tahmin × {exit_multiplier:.0%}\'de çıkış'
        
        self.kasa2_results = {
            'name': f'KASA 2: {exit_multiplier:.0%} Çıkış Sistemi',
            'strategy': strategy,
            'initial_bankroll': self.initial_bankroll,
            'final_wallet': wallet,
            'total_bets': total_bets,
            'total_wins': total_wins,
            'total_losses': total_losses,
            'win_rate': win_rate,
            'accuracy': accuracy,
            'profit_loss': profit_loss,
            'roi': roi,
            'exit_multiplier': exit_multiplier,
            'avg_exit_point': avg_exit_point,
            'min_exit_point': min(exit_points) if exit_points else 0,
            'max_exit_point': max(exit_points) if exit_points else 0,
            'confidence_threshold': confidence_threshold,
            'skipped_low_confidence': skipped_low_confidence
        }
        
        return self.kasa2_results
    
    def run_both_simulations(
        self,
        threshold_predictions: np.ndarray,
        kasa1_confidence_threshold: float = 0.0,
        kasa2_exit_multiplier: float = 0.70,
        kasa2_confidence_threshold: float = 0.0
    ) -> Tuple[Dict, Dict]:
        """
        Her iki kasa simülasyonunu da çalıştır
        
        Args:
            threshold_predictions: Threshold tahminleri (0 veya 1)
            kasa1_confidence_threshold: Kasa 1 için minimum güven skoru (0-1)
            kasa2_exit_multiplier: Kasa 2 için çıkış çarpanı (varsayılan: 0.70)
            kasa2_confidence_threshold: Kasa 2 için minimum güven skoru (0-1)
        
        Returns:
            (kasa1_results, kasa2_results) tuple
        """
        kasa1 = self.run_kasa1_simulation(
            threshold_predictions,
            confidence_threshold=kasa1_confidence_threshold
        )
        kasa2 = self.run_kasa2_simulation(
            exit_multiplier=kasa2_exit_multiplier,
            confidence_threshold=kasa2_confidence_threshold
        )
        
        return kasa1, kasa2
    
    def print_detailed_report(self):
        """Detaylı rapor yazdır"""
        if self.kasa1_results is None or self.kasa2_results is None:
            print("❌ Henüz simülasyon çalıştırılmadı!")
            return
        
        print("\n" + "="*80)
        print("💰 ÇİFT SANAL KASA SİMÜLASYONU RAPORU")
        print("="*80)
        
        # Test parametreleri
        test_count = len(self.actual_values)
        print(f"\n📊 TEST PARAMETRELERİ:")
        print(f"  Test Veri Sayısı: {test_count:,}")
        print(f"  Başlangıç Kasası: {self.initial_bankroll:,.2f} TL (dinamik)")
        print(f"  Bahis Tutarı: {self.bet_amount:.2f} TL (sabit)")
        
        # Güven skoru bilgisi
        if self.confidence_scores is not None:
            print(f"  ✅ Güven Skoru Filtresi: Aktif")
        else:
            print(f"  ⚠️ Güven Skoru Filtresi: Pasif (tüm tahminler kullanılıyor)")
        print()
        
        # KASA 1 Raporu
        self._print_kasa_report(self.kasa1_results, kasa_number=1)
        
        # KASA 2 Raporu
        self._print_kasa_report(self.kasa2_results, kasa_number=2)
        
        # Karşılaştırma
        self._print_comparison()
    
    def _print_kasa_report(self, kasa_results: Dict, kasa_number: int):
        """Tek bir kasa için rapor yazdır"""
        print("="*80)
        print(f"💰 {kasa_results['name']}")
        print("="*80)
        print(f"Strateji: {kasa_results['strategy']}")
        print()
        
        # Güven skoru bilgisi
        if 'confidence_threshold' in kasa_results and kasa_results['confidence_threshold'] > 0:
            print(f"🔒 GÜ VEN FİLTRESİ:")
            print(f"  Minimum Güven: %{kasa_results['confidence_threshold']*100:.0f}")
            print(f"  Atlanan Düşük Güven: {kasa_results['skipped_low_confidence']:,} oyun")
            print()
        
        # Oyun sonuçları
        print(f"📊 OYUN SONUÇLARI:")
        print(f"  Toplam Oyun: {kasa_results['total_bets']:,} el")
        print(f"  ✅ Kazanan: {kasa_results['total_wins']:,} oyun ({kasa_results['win_rate']:.1f}%)")
        print(f"  ❌ Kaybeden: {kasa_results['total_losses']:,} oyun ({100 - kasa_results['win_rate']:.1f}%)")
        print()
        
        # Finansal sonuçlar
        print(f"💰 FİNANSAL SONUÇLAR:")
        print(f"  Başlangıç Kasası: {kasa_results['initial_bankroll']:,.2f} TL")
        print(f"  Final Kasa: {kasa_results['final_wallet']:,.2f} TL")
        
        profit_emoji = "📈" if kasa_results['profit_loss'] > 0 else "📉"
        profit_sign = "+" if kasa_results['profit_loss'] > 0 else ""
        print(f"  {profit_emoji} Net Kar/Zarar: {profit_sign}{kasa_results['profit_loss']:,.2f} TL")
        
        roi_emoji = "🚀" if kasa_results['roi'] > 10 else "✅" if kasa_results['roi'] > 0 else "⚠️" if kasa_results['roi'] > -10 else "❌"
        print(f"  {roi_emoji} ROI: {profit_sign}{kasa_results['roi']:.2f}%")
        print()
        
        # Doğruluk
        acc_emoji = "✅" if kasa_results['accuracy'] >= 70 else "⚠️" if kasa_results['accuracy'] >= 60 else "❌"
        print(f"🎯 DOĞRULUK:")
        print(f"  {acc_emoji} Kazanma Oranı: {kasa_results['accuracy']:.1f}%")
        
        # Kasa 2 için ek bilgi
        if 'avg_exit_point' in kasa_results:
            print()
            print(f"📊 ÇIKIŞ NOKTALARI:")
            print(f"  Çıkış Çarpanı: {kasa_results.get('exit_multiplier', 0.70):.0%}")
            print(f"  Ortalama Çıkış: {kasa_results['avg_exit_point']:.2f}x")
            print(f"  Min Çıkış: {kasa_results['min_exit_point']:.2f}x")
            print(f"  Max Çıkış: {kasa_results['max_exit_point']:.2f}x")
        
        print()
    
    def _print_comparison(self):
        """İki kasa karşılaştırması"""
        print("="*80)
        print("📊 KASA KARŞILAŞTIRMASI")
        print("="*80)
        
        kasa1 = self.kasa1_results
        kasa2 = self.kasa2_results
        
        # Kasa isimleri
        kasa1_name = f"Kasa 1 (1.5x)"
        kasa2_multiplier = kasa2.get('exit_multiplier', 0.70)
        kasa2_name = f"Kasa 2 ({kasa2_multiplier:.0%})"
        
        # Tablo başlığı
        print(f"{'Metrik':<30} {kasa1_name:<25} {kasa2_name:<25}")
        print("-"*80)
        
        # Metrikler
        print(f"{'Toplam Oyun':<30} {kasa1['total_bets']:<25,} {kasa2['total_bets']:<25,}")
        print(f"{'Kazanan Oyun':<30} {kasa1['total_wins']:<25,} {kasa2['total_wins']:<25,}")
        print(f"{'Kazanma Oranı':<30} {kasa1['win_rate']:<24.1f}% {kasa2['win_rate']:<24.1f}%")
        print(f"{'Net Kar/Zarar':<30} {kasa1['profit_loss']:<24,.2f} TL {kasa2['profit_loss']:<24,.2f} TL")
        print(f"{'ROI':<30} {kasa1['roi']:<24.2f}% {kasa2['roi']:<24.2f}%")
        
        # Güven skoru filtreleri
        if 'skipped_low_confidence' in kasa1:
            print(f"{'Atlanan Düşük Güven':<30} {kasa1['skipped_low_confidence']:<25,} {kasa2.get('skipped_low_confidence', 0):<25,}")
        
        print("-"*80)
        
        # Hangi kasa daha karlı?
        if kasa1['profit_loss'] > kasa2['profit_loss']:
            diff = kasa1['profit_loss'] - kasa2['profit_loss']
            print(f"🏆 KASA 1 daha karlı (+{diff:,.2f} TL fark)")
        elif kasa2['profit_loss'] > kasa1['profit_loss']:
            diff = kasa2['profit_loss'] - kasa1['profit_loss']
            print(f"🏆 KASA 2 daha karlı (+{diff:,.2f} TL fark)")
        else:
            print(f"⚖️ Her iki kasa eşit karlılıkta")
        
        print("="*80)
        print()


def simulate_dual_bankroll(
    test_predictions: np.ndarray,
    actual_values: np.ndarray,
    threshold_predictions: np.ndarray,
    bet_amount: float = 10.0,
    confidence_scores: Optional[np.ndarray] = None,
    kasa1_confidence_threshold: float = 0.0,
    kasa2_exit_multiplier: float = 0.70,
    kasa2_confidence_threshold: float = 0.0,
    print_report: bool = True
) -> Tuple[Dict, Dict]:
    """
    Çift kasa simülasyonu için yardımcı fonksiyon
    
    Args:
        test_predictions: Model tahminleri (değerler)
        actual_values: Gerçek değerler
        threshold_predictions: Threshold tahminleri (0 veya 1)
        bet_amount: Bahis miktarı (TL)
        confidence_scores: Opsiyonel - Model güven skorları (0-1 arası)
        kasa1_confidence_threshold: Kasa 1 için minimum güven skoru
        kasa2_exit_multiplier: Kasa 2 için çıkış çarpanı (varsayılan: 0.70)
        kasa2_confidence_threshold: Kasa 2 için minimum güven skoru
        print_report: Rapor yazdır mı?
    
    Returns:
        (kasa1_results, kasa2_results) tuple
    """
    # Sistem oluştur
    dual_system = DualBankrollSystem(
        test_predictions=test_predictions,
        actual_values=actual_values,
        bet_amount=bet_amount,
        confidence_scores=confidence_scores
    )
    
    # Simülasyonları çalıştır
    kasa1, kasa2 = dual_system.run_both_simulations(
        threshold_predictions=threshold_predictions,
        kasa1_confidence_threshold=kasa1_confidence_threshold,
        kasa2_exit_multiplier=kasa2_exit_multiplier,
        kasa2_confidence_threshold=kasa2_confidence_threshold
    )
    
    # Rapor yazdır
    if print_report:
        dual_system.print_detailed_report()
    
    return kasa1, kasa2


# Örnek kullanım
if __name__ == "__main__":
    print("🎯 Dual Bankroll System - Test (Optimize Edilmiş)")
    print("✨ YENİ: Güven skoru filtresi + %70 çıkış optimize")
    print()
    
    # Örnek veri
    np.random.seed(42)
    test_count = 1000
    
    # Örnek tahminler ve gerçek değerler
    test_predictions = np.random.uniform(1.0, 5.0, test_count)
    actual_values = np.random.uniform(1.0, 5.0, test_count)
    threshold_predictions = (test_predictions >= 1.5).astype(int)
    
    # Örnek güven skorları (0.5 ile 1.0 arası)
    confidence_scores = np.random.uniform(0.5, 1.0, test_count)
    
    print("📊 Test 1: Güven filtresi YOK")
    print("="*50)
    kasa1, kasa2 = simulate_dual_bankroll(
        test_predictions=test_predictions,
        actual_values=actual_values,
        threshold_predictions=threshold_predictions,
        bet_amount=10.0,
        confidence_scores=None,  # Güven skoru yok
        kasa2_exit_multiplier=0.70,  # %70 çıkış
        print_report=True
    )
    
    print("\n" + "="*80)
    print("📊 Test 2: %90 Güven Filtresi AKTİF")
    print("="*50)
    kasa1_filtered, kasa2_filtered = simulate_dual_bankroll(
        test_predictions=test_predictions,
        actual_values=actual_values,
        threshold_predictions=threshold_predictions,
        bet_amount=10.0,
        confidence_scores=confidence_scores,  # Güven skoru var
        kasa1_confidence_threshold=0.90,  # %90 güven filtresi
        kasa2_exit_multiplier=0.70,  # %70 çıkış
        kasa2_confidence_threshold=0.90,  # %90 güven filtresi
        print_report=True
    )
    
    print("\n✅ Test tamamlandı!")
    print("💡 Güven filtresi sayesinde daha az ama daha güvenli bahisler yapıldı.")
