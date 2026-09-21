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
        env.cooldown_steps = 0  # This test is about buy/sell mechanics, not cooldown spacing
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
        env.cooldown_steps = 0  # This test is about fee mechanics, not cooldown spacing
        env.reset(seed=0)
        initial = env.balance_usdt
        env.step(np.array([1.0]))  # Buy
        env.step(np.array([-1.0]))  # Sell
        # After round trip, we should have LESS due to fees
        assert env.balance_usdt < initial


class TestCooldown:
    """
    Regression tests for action.cooldown_steps, enabled in config.yaml to
    structurally cap trade frequency. Measured on the live curriculum runs
    (W&B trading/trades_count): the agent was trading on ~96-97% of all
    steps regardless of fees, and widening dead_zone alone barely moved
    that (with train/std kept near 1.0 by target_kl, most of the action
    distribution's mass falls outside even a doubled dead_zone). Unlike
    dead_zone, cooldown_steps enforces a hard minimum spacing between
    trades independently of the action distribution's shape.
    """

    def test_blocks_trade_too_soon(self, env):
        env.cooldown_steps = 3
        env.max_position_pct = 1.0
        env.reset(seed=0)
        env.step(np.array([1.0]))  # Buy - starts the cooldown
        _, _, _, _, info = env.step(np.array([-1.0]))  # Immediately after: too soon
        assert info["trade"]["type"] == "hold"
        assert env.balance_asset > 0  # The buy still executed, only the sell was blocked

    def test_allows_trade_after_elapsed(self, env):
        env.cooldown_steps = 2
        env.max_position_pct = 1.0
        env.reset(seed=0)
        env.step(np.array([1.0]))  # Buy - starts the cooldown
        env.step(np.array([0.0]))  # Hold (1 step elapsed)
        env.step(np.array([0.0]))  # Hold (2 steps elapsed)
        _, _, _, _, info = env.step(np.array([-1.0]))  # Cooldown elapsed: allowed
        assert info["trade"]["type"] == "sell"

    def test_disabled_by_default_value_zero(self, env):
        env.cooldown_steps = 0
        env.max_position_pct = 1.0
        env.reset(seed=0)
        env.step(np.array([1.0]))
        _, _, _, _, info = env.step(np.array([-1.0]))
        assert info["trade"]["type"] == "sell"


class TestShortSelling:
    """
    Regression tests for short-selling, gated by config short.enabled
    (default False — see docs/superpowers/specs/2026-09-21-short-selling-design.md).
    The default `env` fixture loads the real config.yaml, where
    short.enabled=False, so these tests explicitly flip env.short_enabled
    for the duration of the test.
    """

    def test_short_disabled_by_default(self, env):
        assert env.short_enabled is False

    def test_sell_while_flat_opens_short(self, env):
        env.short_enabled = True
        env.max_position_pct = 1.0
        env.cooldown_steps = 0
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([-0.8]))
        assert info["trade"]["type"] == "short"
        assert env.balance_asset < 0
        assert env.balance_usdt > 0  # a recu le produit net de la vente a decouvert

    def test_cover_closes_short_and_reports_pnl(self, env):
        env.short_enabled = True
        env.max_position_pct = 1.0
        env.cooldown_steps = 0
        env.reset(seed=0)
        env.step(np.array([-1.0]))  # ouvre un short
        entry_price = env.entry_price
        cover_price = env.close_prices[env.current_step]
        expected_pnl_pct = (entry_price - cover_price) / entry_price

        _, _, _, _, info = env.step(np.array([1.0]))  # couvre entierement

        assert info["trade"]["type"] == "cover"
        assert "pnl_pct" in info["trade"]
        assert info["trade"]["pnl_pct"] == pytest.approx(expected_pnl_pct, rel=1e-6)
        assert env.balance_asset == pytest.approx(0.0, abs=1e-9)
        assert env.entry_price == 0.0

    def test_short_pnl_positive_when_price_falls(self, env):
        env.short_enabled = True
        env.max_position_pct = 1.0
        env.cooldown_steps = 0
        env.reset(seed=0)
        env.step(np.array([-1.0]))
        # Force artificiellement une baisse de prix pour un test deterministe
        env.close_prices[env.current_step] = env.entry_price * 0.9
        _, _, _, _, info = env.step(np.array([1.0]))
        assert info["trade"]["pnl_pct"] > 0


class TestTradePnl:
    """
    Regression tests for trade['pnl_pct'] — required by
    training/callbacks.py's TensorboardCallback to compute
    trading/win_rate, trading/avg_win, trading/avg_loss and
    trading/profit_factor. Without it, that callback silently never logs
    anything (its `if 'pnl_pct' in t` filter always drops every trade),
    which happened on every training run to date without raising an error.
    """

    def test_buy_has_no_pnl_pct(self, env):
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([0.8]))
        assert info["trade"]["type"] == "buy"
        assert "pnl_pct" not in info["trade"]

    def test_sell_reports_realized_pnl_pct(self, env):
        env.max_position_pct = 1.0
        env.cooldown_steps = 0  # This test is about pnl_pct, not cooldown spacing
        env.reset(seed=0)
        env.step(np.array([1.0]))  # Buy all
        entry_price = env.entry_price
        sell_price = env.close_prices[env.current_step]
        expected_pnl_pct = (sell_price - entry_price) / entry_price

        _, _, _, _, info = env.step(np.array([-1.0]))  # Sell all

        assert info["trade"]["type"] == "sell"
        assert "pnl_pct" in info["trade"]
        assert info["trade"]["pnl_pct"] == pytest.approx(expected_pnl_pct, rel=1e-6)

    def test_hold_has_no_pnl_pct(self, env):
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([0.0]))
        assert "pnl_pct" not in info["trade"]


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
        env.cooldown_steps = 0  # This test is about trade logging, not cooldown spacing
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


class TestMaxEpisodeSteps:
    """
    Regression tests for bounded, randomized-start episodes during training.

    Without this, an episode always starts at the same point and only ends
    at the end of the dataframe. On the current 1-year/1-minute dataset
    that's ~367k steps per episode — longer than an entire curriculum
    level's budget, so ep_rew_mean never populates and the agent replays
    the exact same slice every reset (see Next_step.md 3b.3).
    """

    @pytest.fixture
    def train_config(self, config):
        cfg = dict(config)
        cfg["training"] = dict(config["training"])
        cfg["training"]["max_episode_steps"] = 200
        cfg["training"]["domain_randomization"] = dict(config["training"]["domain_randomization"])
        cfg["training"]["domain_randomization"]["enabled"] = False
        return cfg

    def test_episode_truncates_at_max_episode_steps(self, synthetic_df, train_config):
        env = CryptoTradingEnv(synthetic_df, config=train_config, mode="train")
        env.reset(seed=0)
        steps = 0
        done = False
        while not done:
            _, _, term, trunc, _ = env.step(np.array([0.0]))
            done = term or trunc
            steps += 1
        # Dataset has ~1900 usable rows; without the cap the episode would
        # run that long. With max_episode_steps=200 it must stop much sooner.
        assert steps <= 201

    def test_start_point_randomized_across_resets(self, synthetic_df, train_config):
        env = CryptoTradingEnv(synthetic_df, config=train_config, mode="train")
        starts = set()
        for seed in range(10):
            env.reset(seed=seed)
            starts.add(env.current_step)
        # At least some resets must land on different starting points.
        assert len(starts) > 1

    def test_eval_mode_ignores_max_episode_steps(self, synthetic_df, train_config):
        """mode='test' must still traverse the full dataset deterministically."""
        env = CryptoTradingEnv(synthetic_df, config=train_config, mode="test")
        obs, _ = env.reset(seed=0)
        assert env.current_step == env.lookback_window
        steps = 0
        done = False
        while not done:
            _, _, term, trunc, _ = env.step(np.array([0.0]))
            done = term or trunc
            steps += 1
        assert steps > 201  # full dataset, not capped
