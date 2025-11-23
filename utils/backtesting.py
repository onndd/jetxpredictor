"""
Backtesting Engine - Model Performans Testi ve Simülasyonu
Geçmiş veri üzerinde model stratejilerini test eder ve performans analizi yapar.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class BettingStrategy(Enum):
    """Bahis stratejileri"""
    FIXED = "fixed"  # Sabit bahis
    KELLY = "kelly"  # Kelly criterion
    MARTINGALE = "martingale"  # Martingale (risk!)
    CONSERVATIVE = "conservative"  # Muhafazakar


@dataclass
class BacktestResult:
    """Backtest sonucu"""
    # Temel metrikler
    total_games: int
    wins: int
    losses: int
    skipped: int  # Atlanan oyunlar (düşük güven vb.)
    
    # Finansal metrikler
    starting_capital: float
    ending_capital: float
    net_profit: float
    roi: float  # Return on Investment
    
    # Performans metrikleri
    win_rate: float
    avg_confidence: float
    avg_bet_size: float
    
    # Risk metrikleri
    max_drawdown: float  # En büyük düşüş
    max_drawdown_pct: float
    sharpe_ratio: float  # Risk-adjusted return
    
    # Streak analizi
    max_win_streak: int
    max_loss_streak: int
    current_streak: int
    
    # Detaylar
    trades: List[Dict] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Dict'e çevir"""
        return {
            'total_games': self.total_games,
            'wins': self.wins,
            'losses': self.losses,
            'skipped': self.skipped,
            'starting_capital': self.starting_capital,
            'ending_capital': self.ending_capital,
            'net_profit': self.net_profit,
            'roi': self.roi,
            'win_rate': self.win_rate,
            'avg_confidence': self.avg_confidence,
            'avg_bet_size': self.avg_bet_size,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'max_win_streak': self.max_win_streak,
            'max_loss_streak': self.max_loss_streak
        }


class BacktestEngine:
    """
    Backtesting motoru
    
    Özellikleri:
    1. Historical backtesting - Geçmiş veri testi
    2. Walk-forward validation - İleriye doğru test
    3. Monte Carlo simülasyon - Rastgele senaryo testi
    4. Strategy comparison - Strateji karşılaştırma
    """
    
    def __init__(
        self,
        starting_capital: float = 1000.0,
        bet_size: float = 10.0,
        threshold: float = 1.5,
        strategy: BettingStrategy = BettingStrategy.FIXED,
        min_confidence: float = 0.85,  # GÜNCELLENDİ: Varsayılan 0.5 -> 0.85
        risk_per_trade: float = 0.02
    ):
        """
        Args:
            starting_capital: Başlangıç sermayesi
            bet_size: Bahis büyüklüğü (fixed strategy için)
            threshold: Threshold değeri
            strategy: Bahis stratejisi
            min_confidence: Minimum güven skoru
            risk_per_trade: İşlem başına risk (sermayenin %)
        """
        self.starting_capital = starting_capital
        self.bet_size = bet_size
        self.threshold = threshold
        self.strategy = strategy
        self.min_confidence = min_confidence
        self.risk_per_trade = risk_per_trade
        
        # State
        self.capital = starting_capital
        self.trades = []
        self.equity_curve = [starting_capital]
        
        logger.info(f"Backtesting Engine oluşturuldu:")
        logger.info(f"  • Başlangıç sermayesi: {starting_capital}")
        logger.info(f"  • Bahis stratejisi: {strategy.value}")
        logger.info(f"  • Threshold: {threshold}")
    
    def run(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None
    ) -> BacktestResult:
        """
        Backtest çalıştır
        
        Args:
            predictions: Model tahminleri
            actuals: Gerçek değerler
            confidences: Güven skorları (opsiyonel)
            timestamps: Zaman damgaları (opsiyonel)
            
        Returns:
            BacktestResult
        """
        # Reset state
        self.capital = self.starting_capital
        self.trades = []
        self.equity_curve = [self.starting_capital]
        
        # Confidences yoksa 1.0 kabul et
        if confidences is None:
            confidences = np.ones(len(predictions))
        
        # Timestamps yoksa index kullan
        if timestamps is None:
            timestamps = np.arange(len(predictions))
        
        # Her tahmin için işlem yap
        wins = 0
        losses = 0
        skipped = 0
        
        win_streak = 0
        loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        current_streak = 0
        
        for i, (pred, actual, conf, ts) in enumerate(zip(predictions, actuals, confidences, timestamps)):
            # Güven düşükse atla
            if conf < self.min_confidence:
                skipped += 1
                continue
            
            # Bahis tahmini yap
            will_exceed_threshold = pred >= self.threshold
            
            # Eğer tahmin threshold üstüyse bahse gir
            if will_exceed_threshold:
                # Bahis büyüklüğünü hesapla
                bet_amount = self._calculate_bet_size(conf)
                
                # Gerçek sonuç
                actually_exceeded = actual >= self.threshold
                
                # Kazanç/kayıp hesapla
                if actually_exceeded:
                    # Kazandık!
                    profit = bet_amount * (self.threshold - 1.0)
                    self.capital += profit
                    wins += 1
                    win_streak += 1
                    loss_streak = 0
                    current_streak = win_streak
                    max_win_streak = max(max_win_streak, win_streak)
                else:
                    # Kaybettik
                    self.capital -= bet_amount
                    losses += 1
                    loss_streak += 1
                    win_streak = 0
                    current_streak = -loss_streak
                    max_loss_streak = max(max_loss_streak, loss_streak)
                
                # Trade kaydı
                trade = {
                    'timestamp': ts,
                    'prediction': float(pred),
                    'actual': float(actual),
                    'confidence': float(conf),
                    'bet_amount': bet_amount,
                    'profit': profit if actually_exceeded else -bet_amount,
                    'won': actually_exceeded,
                    'capital': self.capital
                }
                self.trades.append(trade)
                self.equity_curve.append(self.capital)
        
        # Sonuç hesapla
        result = self._calculate_results(
            wins=wins,
            losses=losses,
            skipped=skipped,
            max_win_streak=max_win_streak,
            max_loss_streak=max_loss_streak,
            current_streak=current_streak
        )
        
        return result
    
    def _calculate_bet_size(self, confidence: float) -> float:
        """
        Bahis büyüklüğünü hesapla (strateji bazlı)
        
        Args:
            confidence: Güven skoru
            
        Returns:
            Bet amount
        """
        if self.strategy == BettingStrategy.FIXED:
            # Sabit bahis
            return self.bet_size
        
        elif self.strategy == BettingStrategy.KELLY:
            # Kelly criterion
            # f* = (bp - q) / b
            # b = odds (threshold - 1)
            # p = win probability (confidence)
            # q = loss probability (1 - confidence)
            
            b = self.threshold - 1.0  # Kazanç oranı
            p = confidence
            q = 1 - confidence
            
            kelly_fraction = (b * p - q) / b
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            return self.capital * kelly_fraction
        
        elif self.strategy == BettingStrategy.MARTINGALE:
            # Martingale (RİSKLİ!)
            # Her kayıptan sonra bahsi ikiye katla
            if len(self.trades) > 0 and not self.trades[-1]['won']:
                last_bet = self.trades[-1]['bet_amount']
                return min(last_bet * 2, self.capital * 0.5)  # Max %50
            else:
                return self.bet_size
        
        elif self.strategy == BettingStrategy.CONSERVATIVE:
            # Muhafazakar: Güvene göre ama düşük
            return self.capital * self.risk_per_trade * confidence
        
        else:
            return self.bet_size
    
    def _calculate_results(
        self,
        wins: int,
        losses: int,
        skipped: int,
        max_win_streak: int,
        max_loss_streak: int,
        current_streak: int
    ) -> BacktestResult:
        """Sonuçları hesapla"""
        
        total_games = wins + losses
        net_profit = self.capital - self.starting_capital
        roi = (net_profit / self.starting_capital) * 100 if self.starting_capital > 0 else 0
        win_rate = wins / total_games if total_games > 0 else 0
        
        # Average confidence ve bet size
        if self.trades:
            avg_confidence = np.mean([t['confidence'] for t in self.trades])
            avg_bet_size = np.mean([t['bet_amount'] for t in self.trades])
        else:
            avg_confidence = 0.0
            avg_bet_size = 0.0
        
        # Max drawdown
        max_drawdown, max_drawdown_pct = self._calculate_max_drawdown()
        
        # Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        return BacktestResult(
            total_games=total_games,
            wins=wins,
            losses=losses,
            skipped=skipped,
            starting_capital=self.starting_capital,
            ending_capital=self.capital,
            net_profit=net_profit,
            roi=roi,
            win_rate=win_rate,
            avg_confidence=avg_confidence,
            avg_bet_size=avg_bet_size,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            max_win_streak=max_win_streak,
            max_loss_streak=max_loss_streak,
            current_streak=current_streak,
            trades=self.trades,
            equity_curve=self.equity_curve
        )
    
    def _calculate_max_drawdown(self) -> Tuple[float, float]:
        """
        Maximum drawdown hesapla
        
        Returns:
            (max_drawdown_amount, max_drawdown_percentage)
        """
        if len(self.equity_curve) < 2:
            return 0.0, 0.0
        
        equity = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = running_max - equity
        max_dd = np.max(drawdown)
        max_dd_pct = (max_dd / np.max(running_max)) * 100 if np.max(running_max) > 0 else 0
        
        return float(max_dd), float(max_dd_pct)
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Sharpe ratio hesapla (risk-adjusted return)
        
        Args:
            risk_free_rate: Risksiz getiri oranı (yıllık)
            
        Returns:
            Sharpe ratio
        """
        if len(self.trades) < 2:
            return 0.0
        
        # Her işlemin getirisini hesapla
        returns = [t['profit'] / t['bet_amount'] for t in self.trades if t['bet_amount'] > 0]
        
        if not returns:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Sharpe ratio
        sharpe = (mean_return - risk_free_rate) / std_return
        
        return float(sharpe)
    
    def walk_forward_validation(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        train_window: int = 100,
        test_window: int = 20
    ) -> List[BacktestResult]:
        """
        Walk-forward validation
        
        Veriyi pencereler halinde test eder:
        - İlk 100 veri ile eğit (burada kullanmıyoruz, sadece simüle)
        - Sonraki 20 veri ile test
        - Pencereyi kaydır, tekrarla
        
        Args:
            predictions: Tahminler
            actuals: Gerçek değerler
            confidences: Güven skorları
            train_window: Eğitim penceresi
            test_window: Test penceresi
            
        Returns:
            List of BacktestResult (her pencere için)
        """
        results = []
        
        for i in range(train_window, len(predictions), test_window):
            # Test window
            test_start = i
            test_end = min(i + test_window, len(predictions))
            
            test_preds = predictions[test_start:test_end]
            test_actuals = actuals[test_start:test_end]
            test_confs = confidences[test_start:test_end] if confidences is not None else None
            
            # Backtest yap
            result = self.run(test_preds, test_actuals, test_confs)
            results.append(result)
            
            logger.info(f"Walk-forward window {len(results)}: ROI={result.roi:.2f}%, Win rate={result.win_rate:.2%}")
        
        return results
    
    def monte_carlo_simulation(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        n_simulations: int = 1000,
        sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Monte Carlo simülasyon
        
        Rastgele örnekleme ile çoklu senaryo testi
        
        Args:
            predictions: Tahminler
            actuals: Gerçek değerler
            confidences: Güven skorları
            n_simulations: Simülasyon sayısı
            sample_size: Her simülasyondaki örnek sayısı (None = tüm veri)
            
        Returns:
            Simülasyon sonuçları
        """
        if sample_size is None:
            sample_size = len(predictions)
        
        roi_distribution = []
        win_rate_distribution = []
        max_dd_distribution = []
        
        for sim in range(n_simulations):
            # Rastgele örnekle
            indices = np.random.choice(len(predictions), size=sample_size, replace=True)
            
            sim_preds = predictions[indices]
            sim_actuals = actuals[indices]
            sim_confs = confidences[indices] if confidences is not None else None
            
            # Backtest yap
            result = self.run(sim_preds, sim_actuals, sim_confs)
            
            roi_distribution.append(result.roi)
            win_rate_distribution.append(result.win_rate)
            max_dd_distribution.append(result.max_drawdown_pct)
        
        # İstatistikler
        return {
            'n_simulations': n_simulations,
            'roi': {
                'mean': np.mean(roi_distribution),
                'std': np.std(roi_distribution),
                'min': np.min(roi_distribution),
                'max': np.max(roi_distribution),
                'percentile_5': np.percentile(roi_distribution, 5),
                'percentile_95': np.percentile(roi_distribution, 95),
                'positive_ratio': sum(1 for r in roi_distribution if r > 0) / n_simulations
            },
            'win_rate': {
                'mean': np.mean(win_rate_distribution),
                'std': np.std(win_rate_distribution),
                'min': np.min(win_rate_distribution),
                'max': np.max(win_rate_distribution)
            },
            'max_drawdown_pct': {
                'mean': np.mean(max_dd_distribution),
                'std': np.std(max_dd_distribution),
                'min': np.min(max_dd_distribution),
                'max': np.max(max_dd_distribution)
            }
        }
    
    def save_results(self, result: BacktestResult, filepath: str):
        """Sonuçları kaydet"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.info(f"Backtest sonuçları kaydedildi: {filepath}")
    
    def print_summary(self, result: BacktestResult):
        """Özet rapor yazdır"""
        print("\n" + "="*70)
        print("BACKTEST SONUÇLARI")
        print("="*70)
        
        print(f"\n📊 GENEL PERFORMANS:")
        print(f"  Toplam Oyun: {result.total_games}")
        print(f"  Kazanan: {result.wins} ({result.win_rate:.1%})")
        print(f"  Kaybeden: {result.losses} ({(result.losses/result.total_games if result.total_games > 0 else 0):.1%})")
        print(f"  Atlanan: {result.skipped}")
        
        print(f"\n💰 FİNANSAL:")
        print(f"  Başlangıç: {result.starting_capital:.2f} TL")
        print(f"  Bitiş: {result.ending_capital:.2f} TL")
        print(f"  Net Kar/Zarar: {result.net_profit:+.2f} TL")
        print(f"  ROI: {result.roi:+.2f}%")
        
        print(f"\n📈 RİSK METRİKLERİ:")
        print(f"  Max Drawdown: {result.max_drawdown:.2f} TL ({result.max_drawdown_pct:.1f}%)")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.3f}")
        print(f"  Ortalama Güven: {result.avg_confidence:.2%}")
        print(f"  Ortalama Bahis: {result.avg_bet_size:.2f} TL")
        
        print(f"\n🎯 STREAK ANALİZİ:")
        print(f"  En Uzun Kazanma Serisi: {result.max_win_streak}")
        print(f"  En Uzun Kaybetme Serisi: {result.max_loss_streak}")
        print(f"  Mevcut Seri: {result.current_streak}")
        
        # Değerlendirme
        print(f"\n💡 DEĞERLENDİRME:")
        if result.roi >= 5:
            print("  ✅ MÜKEMMEL! Çok karlı strateji.")
        elif result.roi >= 2:
            print("  ✅ İYİ! Karlı ve sürdürülebilir.")
        elif result.roi >= 0:
            print("  ⚠️ ORTA. Minimal kar, dikkatli olun.")
        elif result.roi >= -5:
            print("  ❌ KÖTÜ. Zarar ediyor, strateji değişmeli.")
        else:
            print("  ❌ ÇOK KÖTÜ! Ciddi kayıp, hemen durdurun!")
        
        if result.win_rate >= 0.70:
            print("  ✅ Kazanma oranı yüksek.")
        elif result.win_rate >= 0.67:
            print("  ✅ Kazanma oranı başabaş seviyesinde.")
        else:
            print("  ❌ Kazanma oranı yetersiz (hedef: %67+).")
        
        if result.max_drawdown_pct < 20:
            print("  ✅ Risk kontrol altında.")
        elif result.max_drawdown_pct < 40:
            print("  ⚠️ Orta seviye risk.")
        else:
            print("  ❌ Yüksek risk! Sermaye koruması gerekli.")
        
        print("="*70 + "\n")


def compare_strategies(
    predictions: np.ndarray,
    actuals: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    starting_capital: float = 1000.0,
    strategies: Optional[List[BettingStrategy]] = None
) -> Dict[str, BacktestResult]:
    """
    Farklı stratejileri karşılaştır
    
    Args:
        predictions: Tahminler
        actuals: Gerçek değerler
        confidences: Güven skorları
        starting_capital: Başlangıç sermayesi
        strategies: Test edilecek stratejiler (None = hepsi)
        
    Returns:
        Strategy -> BacktestResult mapping
    """
    if strategies is None:
        strategies = list(BettingStrategy)
    
    results = {}
    
    for strategy in strategies:
        engine = BacktestEngine(
            starting_capital=starting_capital,
            strategy=strategy
        )
        
        result = engine.run(predictions, actuals, confidences)
        results[strategy.value] = result
        
        logger.info(f"Strateji: {strategy.value} → ROI: {result.roi:.2f}%, Win rate: {result.win_rate:.2%}")
    
    return results


def create_backtest_engine(
    starting_capital: float = 1000.0,
    strategy: str = 'fixed',
    **kwargs
) -> BacktestEngine:
    """
    Backtest engine factory
    
    Args:
        starting_capital: Başlangıç sermayesi
        strategy: Strateji ('fixed', 'kelly', 'martingale', 'conservative')
        **kwargs: Additional parameters
        
    Returns:
        BacktestEngine instance
    """
    strategy_enum = BettingStrategy(strategy.lower())
    
    return BacktestEngine(
        starting_capital=starting_capital,
        strategy=strategy_enum,
        **kwargs
    )
