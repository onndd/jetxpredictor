"""
Backtest Engine - Model Performans Testi ve Simülasyonu

GÜNCELLEME:
- 2 Modlu Yapı (Normal/Rolling) entegre edildi.
- Normal Mod Eşik: 0.85
- Rolling Mod Eşik: 0.95
- Threshold Manager entegrasyonu
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
from utils.threshold_manager import get_threshold_manager

logger = logging.getLogger(__name__)

class BettingStrategy(Enum):
    """Bahis stratejileri"""
    FIXED = "fixed"          # Sabit bahis
    KELLY = "kelly"          # Kelly criterion (optimize edilmiş)
    MARTINGALE = "martingale" # Martingale (riskli!)
    CONSERVATIVE = "conservative" # Muhafazakar (düşük risk)

@dataclass
class BacktestResult:
    """Backtest sonucu"""
    # Temel metrikler
    total_games: int
    wins: int
    losses: int
    skipped: int
    
    # Finansal metrikler
    starting_capital: float
    ending_capital: float
    net_profit: float
    roi: float
    
    # Performans metrikleri
    win_rate: float
    avg_confidence: float
    avg_bet_size: float
    
    # Risk metrikleri
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    
    # Streak analizi
    max_win_streak: int
    max_loss_streak: int
    current_streak: int
    
    # Detaylar
    trades: List[Dict] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
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
    Backtesting motoru (2 Modlu)
    """
    
    def __init__(
        self,
        starting_capital: float = 1000.0,
        bet_size: float = 10.0,
        target_multiplier: float = 1.5, # Çıkış hedefi (örn: 1.5x)
        strategy: BettingStrategy = BettingStrategy.FIXED,
        mode: str = 'normal', # 'normal' veya 'rolling'
        risk_per_trade: float = 0.02
    ):
        self.starting_capital = starting_capital
        self.bet_size = bet_size
        self.target_multiplier = target_multiplier
        self.strategy = strategy
        self.mode = mode
        self.risk_per_trade = risk_per_trade
        
        # Threshold Manager'dan eşiği al
        tm = get_threshold_manager()
        if mode == 'rolling':
            self.min_confidence = tm.get_rolling_threshold() # 0.95
        else:
            self.min_confidence = tm.get_normal_threshold()  # 0.85
            
        # State
        self.capital = starting_capital
        self.trades = []
        self.equity_curve = [starting_capital]
        
        logger.info(f"BacktestEngine ({mode.upper()} MOD - Eşik: {self.min_confidence})")

    def run(
        self,
        predictions: np.ndarray, # Modelin confidence değerleri (classifier proba)
        actuals: np.ndarray,     # Gerçek değerler
        timestamps: Optional[np.ndarray] = None
    ) -> BacktestResult:
        """Backtest çalıştır"""
        # Reset state
        self.capital = self.starting_capital
        self.trades = []
        self.equity_curve = [self.starting_capital]
        
        if timestamps is None:
            timestamps = np.arange(len(predictions))
            
        wins = 0
        losses = 0
        skipped = 0
        
        win_streak = 0
        loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        current_streak = 0
        
        for i, (conf, actual, ts) in enumerate(zip(predictions, actuals, timestamps)):
            # Güven kontrolü (Mod bazlı)
            if conf < self.min_confidence:
                skipped += 1
                continue
            
            # Bahis büyüklüğünü hesapla
            bet_amount = self._calculate_bet_size(conf)
            
            # Bahis yap
            # Kazandık mı? (Gerçek değer >= Hedef Çarpan)
            won = actual >= self.target_multiplier
            
            if won:
                profit = bet_amount * (self.target_multiplier - 1.0)
                self.capital += profit
                wins += 1
                win_streak += 1
                loss_streak = 0
                current_streak = win_streak
                max_win_streak = max(max_win_streak, win_streak)
            else:
                profit = -bet_amount
                self.capital -= bet_amount
                losses += 1
                loss_streak += 1
                win_streak = 0
                current_streak = -loss_streak
                max_loss_streak = max(max_loss_streak, loss_streak)
            
            self.trades.append({
                'timestamp': ts,
                'confidence': float(conf),
                'actual': float(actual),
                'bet_amount': bet_amount,
                'profit': profit,
                'won': won,
                'capital': self.capital
            })
            self.equity_curve.append(self.capital)
            
            # İflas kontrolü
            if self.capital <= 0:
                break
        
        return self._calculate_results(wins, losses, skipped, max_win_streak, max_loss_streak, current_streak)

    def _calculate_bet_size(self, confidence: float) -> float:
        if self.strategy == BettingStrategy.FIXED:
            return self.bet_size
            
        elif self.strategy == BettingStrategy.KELLY:
            b = self.target_multiplier - 1.0
            p = confidence # Modelin verdiği olasılığı gerçek kazanma olasılığı varsayıyoruz
            q = 1 - p
            kelly_fraction = (b * p - q) / b
            kelly_fraction = max(0, min(kelly_fraction, 0.25)) # Max %25 risk
            return self.capital * kelly_fraction
            
        elif self.strategy == BettingStrategy.MARTINGALE:
            if len(self.trades) > 0 and not self.trades[-1]['won']:
                last_bet = self.trades[-1]['bet_amount']
                return min(last_bet * 2, self.capital * 0.5)
            return self.bet_size
            
        elif self.strategy == BettingStrategy.CONSERVATIVE:
            # Normal modda %4, Rolling modda %2 (AdvancedBankroll ile uyumlu)
            if self.mode == 'rolling':
                 return self.capital * 0.02
            return self.capital * 0.04
            
        return self.bet_size

    def _calculate_results(self, wins, losses, skipped, max_win, max_loss, curr_streak) -> BacktestResult:
        total_games = wins + losses
        net_profit = self.capital - self.starting_capital
        roi = (net_profit / self.starting_capital) * 100 if self.starting_capital > 0 else 0
        win_rate = wins / total_games if total_games > 0 else 0
        
        avg_conf = np.mean([t['confidence'] for t in self.trades]) if self.trades else 0
        avg_bet = np.mean([t['bet_amount'] for t in self.trades]) if self.trades else 0
        
        max_dd, max_dd_pct = self._calculate_max_drawdown()
        sharpe = self._calculate_sharpe_ratio()
        
        return BacktestResult(
            total_games=total_games, wins=wins, losses=losses, skipped=skipped,
            starting_capital=self.starting_capital, ending_capital=self.capital,
            net_profit=net_profit, roi=roi, win_rate=win_rate,
            avg_confidence=avg_conf, avg_bet_size=avg_bet,
            max_drawdown=max_dd, max_drawdown_pct=max_dd_pct, sharpe_ratio=sharpe,
            max_win_streak=max_win, max_loss_streak=max_loss, current_streak=curr_streak,
            trades=self.trades, equity_curve=self.equity_curve
        )

    def _calculate_max_drawdown(self) -> Tuple[float, float]:
        if len(self.equity_curve) < 2: return 0.0, 0.0
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = peak - equity
        max_dd = np.max(drawdown)
        max_dd_pct = (max_dd / np.max(peak)) * 100 if np.max(peak) > 0 else 0
        return float(max_dd), float(max_dd_pct)

    def _calculate_sharpe_ratio(self, risk_free_rate=0.02) -> float:
        if len(self.trades) < 2: return 0.0
        returns = [t['profit'] / t['bet_amount'] for t in self.trades if t['bet_amount'] > 0]
        if not returns: return 0.0
        return float((np.mean(returns) - risk_free_rate) / (np.std(returns) + 1e-9))
    
    def print_summary(self, result: BacktestResult):
        print("\n" + "="*60)
        print(f"BACKTEST SONUÇLARI ({self.mode.upper()} MOD)")
        print("="*60)
        print(f"Net Kar: {result.net_profit:+.2f} TL | ROI: {result.roi:+.2f}%")
        print(f"Win Rate: {result.win_rate:.1%} ({result.wins}/{result.total_games})")
        print(f"Max Drawdown: {result.max_drawdown_pct:.1f}%")
        print(f"Atlanan Oyun: {result.skipped}")
        print("="*60)

def compare_strategies(
    predictions: np.ndarray,
    actuals: np.ndarray,
    mode: str = 'normal'
) -> Dict[str, BacktestResult]:
    """Farklı stratejileri karşılaştır"""
    strategies = list(BettingStrategy)
    results = {}
    
    print(f"\n📊 STRATEJİ KARŞILAŞTIRMA ({mode.upper()} MOD)\n")
    
    for strat in strategies:
        engine = BacktestEngine(strategy=strat, mode=mode)
        res = engine.run(predictions, actuals)
        results[strat.value] = res
        print(f"Strategy: {strat.value:<15} -> ROI: {res.roi:>6.2f}% | WR: {res.win_rate:.1%}")
        
    return results
