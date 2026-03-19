"""
tests/test_env.py — Tests unitaires de l'environnement de trading.
"""

import pytest
import numpy as np
import pandas as pd
import yaml
import os

from env.trading_env import CryptoTradingEnv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "config.yaml")

@pytest.fixture
def config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

@pytest.fixture
def synthetic_df(config):
    """Create a synthetic OHLCV DataFrame fully processed for Phase 4."""
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
        df = add_multi_timeframe_features(
            df, source_tf="1m", config=mtf_config
        )

    # ffill for mtf
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

@pytest.fixture
def env(synthetic_df, config):
    """Create a trading environment with Phase 4 config."""
    return CryptoTradingEnv(synthetic_df, config=config, mode="test")


class TestEnvCreation:
    def test_spaces(self, env):
        lookback = env.lookback_window
        assert env.observation_space.shape == (lookback, env.n_features + 4)
        assert env.action_space.shape == (1,)

    def test_reset(self, env):
        obs, info = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        assert info["portfolio_value"] > 0
        assert info["n_trades"] == 0


class TestActions:
    def test_hold(self, env):
        env.reset(seed=0)
        obs, r, term, trunc, info = env.step(np.array([0.0]))
        assert info["trade"]["type"] == "hold"
        assert info["n_trades"] == 0

    def test_buy(self, env):
        env.reset(seed=0)
        initial_usdt = env.balance_usdt
        obs, r, term, trunc, info = env.step(np.array([0.8]))
        assert info["trade"]["type"] == "buy"
        assert env.balance_usdt < initial_usdt
        assert env.balance_asset > 0

    def test_sell_after_buy(self, env):
        env.max_position_pct = 1.0  # Bypass position cap for this test
        env.reset(seed=0)
        env.step(np.array([1.0]))  # Buy all
        obs, r, term, trunc, info = env.step(np.array([-1.0]))  # Sell all
        assert info["trade"]["type"] == "sell"
        assert env.balance_asset == pytest.approx(0.0, abs=1e-10)

    def test_dead_zone(self, env):
        env.reset(seed=0)
        obs, r, term, trunc, info = env.step(np.array([0.03]))
        assert info["trade"]["type"] == "hold"
        obs, r, term, trunc, info = env.step(np.array([-0.04]))
        assert info["trade"]["type"] == "hold"

    def test_fees_deducted(self, env):
        env.reset(seed=0)
        initial = env.balance_usdt
        env.step(np.array([1.0]))  # Buy
        env.step(np.array([-1.0]))  # Sell
        # After round trip, we should have LESS due to fees
        assert env.balance_usdt < initial


class TestReward:
    def test_hold_reward_near_zero(self, env):
        env.reset(seed=0)
        _, r, _, _, _ = env.step(np.array([0.0]))
        # Hold reward should be small (just market movement + penalties)
        assert abs(r) < 1.0

    def test_reward_dict_keys(self, env):
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([0.5]))
        assert "log_return" in info
        assert "fee_penalty" in info
        assert "drawdown_penalty" in info


class TestEpisode:
    def test_full_episode(self, env):
        obs, info = env.reset(seed=0)
        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            steps += 1
        assert steps > 0
        assert info["portfolio_value"] > 0

    def test_trade_history(self, env):
        env.reset(seed=0)
        env.step(np.array([0.8]))
        env.step(np.array([-0.5]))
        history = env.get_trade_history()
        assert len(history) == 2
        assert "price" in history.columns


class TestSB3Compat:
    def test_check_env(self, env):
        from stable_baselines3.common.env_checker import check_env
        check_env(env, warn=True)
