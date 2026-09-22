"""
tests/test_metrics.py — Tests unitaires des métriques de backtest
(evaluation/metrics.py).
"""

import numpy as np
import pandas as pd
import pytest

from evaluation.metrics import (
    compute_returns,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    win_rate,
    profit_factor,
    summarize,
)


class TestComputeReturns:
    def test_simple_returns(self):
        equity = np.array([100.0, 110.0, 99.0])
        returns = compute_returns(equity)
        assert returns == pytest.approx([0.10, -0.10], rel=1e-6)


class TestSharpeRatio:
    def test_zero_volatility_returns_zero(self):
        # Constant returns -> undefined ratio, guarded to 0.0 (same
        # convention as env/reward.py's sharpe_bonus guard).
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        assert sharpe_ratio(returns) == 0.0

    def test_matches_hand_formula(self):
        returns = np.array([0.02, -0.01, 0.02, -0.01])
        periods_per_year = 252
        expected = (np.mean(returns) / np.std(returns)) * np.sqrt(periods_per_year)
        assert sharpe_ratio(returns, periods_per_year=periods_per_year) == pytest.approx(expected, rel=1e-6)

    def test_empty_returns_zero(self):
        assert sharpe_ratio(np.array([])) == 0.0


class TestSortinoRatio:
    def test_no_downside_returns_zero(self):
        # All-positive returns -> no downside deviation -> guarded to 0.0.
        returns = np.array([0.01, 0.02, 0.03])
        assert sortino_ratio(returns) == 0.0

    def test_matches_hand_formula(self):
        returns = np.array([0.02, -0.01, 0.03, -0.02])
        periods_per_year = 252
        downside = returns[returns < 0]
        expected = (np.mean(returns) / np.std(downside)) * np.sqrt(periods_per_year)
        assert sortino_ratio(returns, periods_per_year=periods_per_year) == pytest.approx(expected, rel=1e-6)


class TestMaxDrawdown:
    def test_known_sequence(self):
        equity = np.array([100.0, 120.0, 90.0, 110.0, 80.0, 130.0])
        # Peaks: 100,120,120,120,120,130. Worst drawdown at 80 vs peak 120 = 1/3.
        assert max_drawdown(equity) == pytest.approx(1 / 3, rel=1e-6)

    def test_monotonic_increase_has_zero_drawdown(self):
        equity = np.array([100.0, 110.0, 120.0, 130.0])
        assert max_drawdown(equity) == pytest.approx(0.0)


class TestCalmarRatio:
    def test_matches_hand_formula(self):
        equity = np.array([100.0, 120.0, 90.0, 130.0])
        periods_per_year = 252
        n = len(equity) - 1
        annualized_return = (equity[-1] / equity[0]) ** (periods_per_year / n) - 1
        mdd = max_drawdown(equity)
        expected = annualized_return / mdd
        assert calmar_ratio(equity, periods_per_year=periods_per_year) == pytest.approx(expected, rel=1e-6)

    def test_zero_drawdown_returns_zero(self):
        equity = np.array([100.0, 110.0, 120.0])
        assert calmar_ratio(equity) == 0.0


class TestWinRateProfitFactor:
    @pytest.fixture
    def trade_history(self):
        return pd.DataFrame({
            "type": ["buy", "sell", "short", "cover", "cover"],
            "pnl_pct": [np.nan, 0.05, np.nan, -0.02, 0.03],
        })

    def test_win_rate_only_counts_closed_trades(self, trade_history):
        # 3 closed trades (sell, cover, cover): 2 wins (0.05, 0.03), 1 loss (-0.02).
        assert win_rate(trade_history) == pytest.approx(2 / 3, rel=1e-6)

    def test_profit_factor(self, trade_history):
        expected = (0.05 + 0.03) / abs(-0.02)
        assert profit_factor(trade_history) == pytest.approx(expected, rel=1e-6)

    def test_no_closed_trades_returns_nan(self):
        th = pd.DataFrame({"type": ["buy"], "pnl_pct": [np.nan]})
        assert np.isnan(win_rate(th))
        assert np.isnan(profit_factor(th))

    def test_no_losses_profit_factor_is_inf(self):
        th = pd.DataFrame({"type": ["sell"], "pnl_pct": [0.05]})
        assert profit_factor(th) == float("inf")


class TestSummarize:
    def test_returns_all_expected_keys(self):
        equity = np.array([100.0, 105.0, 98.0, 110.0])
        trade_history = pd.DataFrame({
            "type": ["sell", "cover"],
            "pnl_pct": [0.02, -0.01],
        })
        result = summarize(equity, trade_history, periods_per_year=252)
        for key in ["sharpe", "sortino", "max_drawdown", "calmar", "win_rate", "profit_factor", "total_return_pct"]:
            assert key in result

    def test_total_return_pct(self):
        equity = np.array([100.0, 150.0])
        trade_history = pd.DataFrame({"type": [], "pnl_pct": []})
        result = summarize(equity, trade_history)
        assert result["total_return_pct"] == pytest.approx(50.0, rel=1e-6)
