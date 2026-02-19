"""
tests/test_env.py — Tests unitaires de l'environnement de trading.
"""

import pytest
import numpy as np
import pandas as pd

from env.trading_env import CryptoTradingEnv


@pytest.fixture
def synthetic_df():
    """Create a synthetic OHLCV DataFrame with indicators."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 40000 + np.cumsum(np.random.randn(n) * 50)
    df = pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 20,
            "high": close + abs(np.random.randn(n) * 80),
            "low": close - abs(np.random.randn(n) * 80),
            "close": close,
            "volume": np.random.uniform(100, 1000, n),
        },
        index=dates,
    )
    from features.indicators import add_all_indicators
    df = add_all_indicators(df)
    df = df.dropna()
    return df


@pytest.fixture
def env(synthetic_df):
    """Create a trading environment."""
    return CryptoTradingEnv(synthetic_df, mode="test")


class TestEnvCreation:
    def test_spaces(self, env):
        assert env.observation_space.shape == (30, env.n_features + 4)
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
