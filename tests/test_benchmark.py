"""
tests/test_benchmark.py — Tests unitaires des références de backtest
(evaluation/benchmark.py) : Buy & Hold et baseline aléatoire.
"""

import os

import numpy as np
import pandas as pd
import pytest
import yaml

from evaluation.benchmark import buy_and_hold_curve, random_baseline_curve

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "config.yaml")


@pytest.fixture
def config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def synthetic_df(config):
    """Same synthetic-data recipe as tests/test_env.py, kept small on purpose."""
    np.random.seed(42)
    n = 2000
    dates = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    close = 40000 + np.cumsum(np.random.randn(n) * 10)
    df = pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 5,
            "high": close + abs(np.random.randn(n) * 20),
            "low": close - abs(np.random.randn(n) * 20),
            "close": close,
            "volume": np.random.uniform(10, 500, n),
        },
        index=dates,
    )

    from features.indicators import add_all_indicators
    from features.multi_timeframe import add_multi_timeframe_features
    from features.normalizer import rolling_normalize

    df = add_all_indicators(df, config.get("observation", {}))

    mtf_config = config.get("observation", {}).get("multi_timeframe", {})
    if mtf_config.get("enabled", False):
        df = add_multi_timeframe_features(df, source_tf="1m", config=mtf_config)

    mtf_cols = [c for c in df.columns if c.startswith("mtf_")]
    if mtf_cols:
        df[mtf_cols] = df[mtf_cols].ffill().fillna(0.0)
    df = df.dropna()

    raw_cols = df[["open", "high", "low", "close"]].copy()
    lookback = config.get("data", {}).get("lookback_window", 60)
    df_norm = rolling_normalize(df, window=lookback)

    for col in raw_cols.columns:
        df_norm[f"raw_{col}"] = raw_cols[col]

    df_norm = df_norm.dropna()
    return df_norm


class TestBuyAndHoldCurve:
    def test_matches_hand_computation(self):
        close_prices = np.array([100.0, 110.0, 90.0])
        curve = buy_and_hold_curve(close_prices, initial_capital=1000.0, fee_rate=0.01)
        # units = (1000 * 0.99) / 100 = 9.9
        expected = np.array([990.0, 1089.0, 891.0])
        assert curve == pytest.approx(expected, rel=1e-6)

    def test_same_length_as_input(self):
        close_prices = np.linspace(100, 200, 50)
        curve = buy_and_hold_curve(close_prices, initial_capital=10_000.0, fee_rate=0.001)
        assert len(curve) == len(close_prices)

    def test_empty_input_returns_empty(self):
        assert len(buy_and_hold_curve(np.array([]))) == 0


class TestRandomBaselineCurve:
    def test_runs_full_episode_and_returns_history(self, synthetic_df, config):
        equity_curve, trade_history = random_baseline_curve(
            {"BTC/USDT": synthetic_df}, config, seed=0,
        )
        assert len(equity_curve) > 0
        assert equity_curve[0] > 0
        # trade_history may be empty by chance but must be a DataFrame with the right shape when non-empty
        if len(trade_history):
            assert "pnl_pct" in trade_history.columns

    def test_seed_is_reproducible(self, synthetic_df, config):
        curve_a, _ = random_baseline_curve({"BTC/USDT": synthetic_df}, config, seed=42)
        curve_b, _ = random_baseline_curve({"BTC/USDT": synthetic_df}, config, seed=42)
        assert curve_a == pytest.approx(curve_b, rel=1e-6)
