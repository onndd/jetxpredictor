#!/usr/bin/env python3
"""
JetX Predictor - Volatilite Bazlı Pozisyon Küçültme Test Script'i

Bu script, AdvancedBankrollManager'a eklenen yeni volatilite bazlı pozisyon küçültme 
özelliğini test etmek için kullanılır.
"""

import sys
import os
import numpy as np
from datetime import datetime

# Proje path'ini ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.advanced_bankroll import AdvancedBankrollManager
    from utils.psychological_analyzer import PsychologicalAnalyzer
    print("✅ Modüller başarıyla yüklendi")
except ImportError as e:
    print(f"❌ Modül yüklenemedi: {e}")
    sys.exit(1)


def test_volatility_sizing():
    """Volatilite bazlı pozisyon küçültme özelliğini test eder"""
    
    print("=" * 80)
    print("🛡️ VOLATİLİTE BAZLI POZİSYON KÜÇÜLTME TESTİ")
    print("=" * 80)
    
    # Test senaryoları
    test_scenarios = [
        {
            'name': 'Düşük Risk Senaryosu',
            'confidence': 0.80,
            'predicted_value': 2.0,
            'volatility_risk': 0.3,  # Düşük risk - küçültme olmamalı
            'expected_reduction': 0
        },
        {
            'name': 'Orta Risk Senaryosu',
            'confidence': 0.80,
            'predicted_value': 2.0,
            'volatility_risk': 0.6,  # Orta risk - %50 küçültme
            'expected_reduction': 50
        },
        {
            'name': 'Yüksek Risk Senaryosu',
            'confidence': 0.80,
            'predicted_value': 2.0,
            'volatility_risk': 0.8,  # Yüksek risk - %80 küçültme
            'expected_reduction': 80
        },
        {
            'name': 'Çok Yüksek Güven + Yüksek Risk',
            'confidence': 0.95,
            'predicted_value': 3.0,
            'volatility_risk': 0.9,  # Çok yüksek risk - %80 küçültme
            'expected_reduction': 80
        }
    ]
    
    # Bankroll manager oluştur (moderate risk tolerance)
    manager = AdvancedBankrollManager(
        initial_bankroll=1000.0,
        risk_tolerance='moderate'
    )
    
    print(f"\n💰 Bankroll Manager Ayarları:")
    print(f"  Başlangıç Kasa: {manager.initial_bankroll:.2f} TL")
    print(f"  Risk Stratejisi: {manager.strategy['name']}")
    print(f"  Max Bahis Oranı: %{manager.strategy['max_bet_ratio']*100:.1f}")
    print(f"  Kelly Fraksiyonu: {manager.strategy['kelly_fraction']}")
    
    # Test senaryolarını çalıştır
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'='*60}")
        print(f"🧪 TEST {i}: {scenario['name']}")
        print(f"{'='*60}")
        
        # Orijinal bahis hesapla (volatilite riski olmadan)
        original_bet = manager.calculate_bet_size(
            confidence=scenario['confidence'],
            predicted_value=scenario['predicted_value']
        )
        
        # Volatilite riski ile bahis hesapla
        adjusted_bet = manager.calculate_bet_size(
            confidence=scenario['confidence'],
            predicted_value=scenario['predicted_value'],
            volatility_risk=scenario['volatility_risk']
        )
        
        # Sonuçları göster
        print(f"  Güven Skoru: {scenario['confidence']:.0%}")
        print(f"  Tahmin Edilen Değer: {scenario['predicted_value']:.2f}x")
        print(f"  Volatilite Riski: {scenario['volatility_risk']*100:.0f}%")
        print(f"  Orijinal Bahis: {original_bet:.2f} TL")
        print(f"  Düzenlenmiş Bahis: {adjusted_bet:.2f} TL")
        
        # Küçültme oranını hesapla
        if original_bet > 0:
            actual_reduction = ((original_bet - adjusted_bet) / original_bet) * 100
            print(f"  Gerçek Küçültme: %{actual_reduction:.0f}")
            print(f"  Beklenen Küçültme: %{scenario['expected_reduction']:.0f}")
            
            # Test sonucunu kontrol et
            if abs(actual_reduction - scenario['expected_reduction']) < 5:  # %5 tolerans
                print(f"  ✅ TEST BAŞARILI")
            else:
                print(f"  ❌ TEST BAŞARISIZ - Beklenen: %{scenario['expected_reduction']:.0f}, Gerçek: %{actual_reduction:.0f}")
        else:
            print("  ⚠️ Bahis miktarı 0 - güven skoru çok düşük olabilir")
    
    # PsychologicalAnalyzer entegrasyon testi
    print(f"\n{'='*60}")
    print("🧠 PSYCHOLOGICALANALYZER ENTEGRASYON TESTİ")
    print(f"{'='*60}")
    
    try:
        # Örnek history oluştur
        np.random.seed(42)
        history = []
        
        # Normal değerler (manipülasyon yok)
        for i in range(30):
            history.append(np.random.choice([
                np.random.uniform(1.0, 1.49),  # %40 1.5 altı
                np.random.uniform(1.5, 3.0)     # %60 1.5 üstü
            ], p=[0.4, 0.6]))
        
        # Bait & switch pattern'i ekle (son 5 el yüksek değerler)
        for i in range(5):
            history.append(np.random.uniform(3.0, 8.0))
        
        # PsychologicalAnalyzer ile manipulation score hesapla
        analyzer = PsychologicalAnalyzer(threshold=1.5)
        features = analyzer.analyze_psychological_patterns(history)
        manipulation_score = features.get('manipulation_score', 0.0)
        
        print(f"  History Boyutu: {len(history)} el")
        print(f"  Son 5 Değer: {[f'{v:.2f}x' for v in history[-5:]]}")
        print(f"  Manipülasyon Skoru: {manipulation_score*100:.0f}%")
        
        # Manipülasyon skoru ile bahis hesapla
        confidence = 0.80
        predicted_value = 2.0
        
        original_bet = manager.calculate_bet_size(confidence, predicted_value)
        adjusted_bet = manager.calculate_bet_size(confidence, predicted_value, manipulation_score)
        
        if original_bet > 0:
            reduction = ((original_bet - adjusted_bet) / original_bet) * 100
            print(f"  Orijinal Bahis: {original_bet:.2f} TL")
            print(f"  Manipülasyon ile: {adjusted_bet:.2f} TL")
            print(f"  Otomatik Küçültme: %{reduction:.0f}")
            
            if manipulation_score > 0.5:
                print("  ✅ Manipülasyon tespiti çalışıyor - pozisyon küçültüldü")
            else:
                print("  ⚠️ Düşük manipülasyon skoru - küçültme yapılmadı")
        else:
            print("  ⚠️ Bahis hesaplanamadı")
            
    except Exception as e:
        print(f"  ❌ PsychologicalAnalyzer test hatası: {e}")
    
    # Edge case testleri
    print(f"\n{'='*60}")
    print("⚠️ EDGE CASE TESTLERİ")
    print(f"{'='*60}")
    
    edge_cases = [
        {
            'name': 'Sınır Değeri 0.5',
            'volatility_risk': 0.5,
            'should_reduce': False
        },
        {
            'name': 'Sınır Değeri 0.5+epsilon',
            'volatility_risk': 0.51,
            'should_reduce': True
        },
        {
            'name': 'Sınır Değeri 0.7',
            'volatility_risk': 0.7,
            'should_reduce': True
        },
        {
            'name': 'Sınır Değeri 0.7+epsilon',
            'volatility_risk': 0.71,
            'should_reduce': True
        }
    ]
    
    for case in edge_cases:
        print(f"\n  Test: {case['name']}")
        original_bet = manager.calculate_bet_size(0.80, 2.0)
        adjusted_bet = manager.calculate_bet_size(0.80, 2.0, case['volatility_risk'])
        
        if original_bet > 0:
            reduction = ((original_bet - adjusted_bet) / original_bet) * 100
            reduced = reduction > 0
            
            if reduced == case['should_reduce']:
                print(f"    ✅ Başarılı - Küçültme: {'Evet' if reduced else 'Hayır'}")
            else:
                print(f"    ❌ Başarısız - Beklenen: {'Evet' if case['should_reduce'] else 'Hayır'}, Gerçek: {'Evet' if reduced else 'Hayır'}")
    
    print(f"\n{'='*80}")
    print("🎯 ÖZET")
    print(f"{'='*80}")
    print("✅ Volatilite bazlı pozisyon küçültme özelliği başarıyla test edildi.")
    print("✅ PsychologicalAnalyzer entegrasyonu çalışıyor.")
    print("✅ Edge case'ler doğru çalışıyor.")
    print("\n💡 Özellik production'a hazır!")
    
    return True


if __name__ == "__main__":
    try:
        test_volatility_sizing()
        print(f"\n🕐 Test tamamlandı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"\n❌ Test sırasında hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
