"""
tests/test_phase4_validation.py — Validation tests for Phase 4 configuration.

Validates that config.yaml, feature pipeline, multi-timeframe, curriculum,
and environment are all consistent and ready for 1m BTC/USDT training.

Run with: python -m pytest tests/test_phase4_validation.py -v
"""

import os
import pytest
import yaml
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "config.yaml")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    """Load config.yaml."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def synthetic_1m_df():
    """Create a synthetic 1m OHLCV DataFrame (simulating 1 day = 1440 candles)."""
    np.random.seed(42)
    n = 2000  # ~1.4 days of 1m data
    dates = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
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
    return df


@pytest.fixture
def processed_1m_df(synthetic_1m_df, config):
    """Apply full feature pipeline to synthetic 1m data."""
    from features.indicators import add_all_indicators
    from features.multi_timeframe import add_multi_timeframe_features
    from features.normalizer import rolling_normalize

    df = synthetic_1m_df.copy()

    # 1. Indicators
    df = add_all_indicators(df, config.get("observation", {}))

    # 2. Multi-timeframe
    mtf_config = config.get("observation", {}).get("multi_timeframe", {})
    if mtf_config.get("enabled", False):
        df = add_multi_timeframe_features(
            df, source_tf=config["market"]["timeframe"], config=mtf_config
        )

    # 3. Forward-fill MTF columns, then drop NaNs
    mtf_cols = [c for c in df.columns if c.startswith("mtf_")]
    if mtf_cols:
        df[mtf_cols] = df[mtf_cols].ffill().fillna(0.0)
    df = df.dropna()

    # 4. Save raw columns before normalization
    raw_cols = df[["open", "high", "low", "close"]].copy()

    # 5. Normalize
    lookback = config.get("data", {}).get("lookback_window", 60)
    df_norm = rolling_normalize(df, window=lookback)

    for col in raw_cols.columns:
        df_norm[f"raw_{col}"] = raw_cols[col]

    df_norm = df_norm.dropna()
    return df_norm


# ---------------------------------------------------------------------------
# Config Validation
# ---------------------------------------------------------------------------

class TestConfigConsistency:
    """Verify config.yaml values are coherent for Phase 4."""

    def test_single_pair(self, config):
        """Phase 4: only BTC/USDT."""
        assert config["market"]["pairs"] == ["BTC/USDT"]

    def test_timeframe_is_1m(self, config):
        assert config["market"]["timeframe"] == "1m"

    def test_lookback_window_sensible(self, config):
        """Lookback >= 30 at 1m for meaningful context."""
        lookback = config["data"]["lookback_window"]
        assert lookback >= 30, f"Lookback {lookback} too small for 1m candles"

    def test_multi_timeframe_targets_valid(self, config):
        """Target TFs must be larger than source TF (1m)."""
        mtf = config["observation"]["multi_timeframe"]
        assert mtf["source_tf"] == "1m"
        valid_targets = {"5m", "15m", "30m", "1h", "4h", "1d", "1w"}
        for tf in mtf["target_tfs"]:
            assert tf in valid_targets, f"Unknown target TF: {tf}"

    def test_net_arch_not_too_large(self, config):
        """Network should fit in 2Go VRAM."""
        net_arch = config["training"]["ppo"]["policy_kwargs"]["net_arch"]
        total_params_approx = sum(net_arch)
        assert total_params_approx <= 1024, f"Network too large: {net_arch}"

    def test_learning_rate_constant(self, config):
        """Phase 4: constant LR (not linear schedule)."""
        lr = config["training"]["ppo"]["learning_rate"]
        assert isinstance(lr, (int, float)), f"LR should be constant, got: {lr}"
        assert 1e-5 < lr < 1e-2, f"LR {lr} out of reasonable range"

    def test_batch_size_divides_buffer(self, config):
        """batch_size should cleanly divide n_steps * n_envs."""
        ppo = config["training"]["ppo"]
        n_envs = config["training"].get("n_envs", 1)
        buffer_size = ppo["n_steps"] * n_envs
        assert buffer_size % ppo["batch_size"] == 0, (
            f"Buffer ({ppo['n_steps']}*{n_envs}={buffer_size}) "
            f"not divisible by batch_size ({ppo['batch_size']})"
        )


# ---------------------------------------------------------------------------
# Feature Pipeline
# ---------------------------------------------------------------------------

class TestFeaturePipeline:
    """Verify indicators + multi-TF + normalization work on 1m data."""

    def test_indicators_no_crash(self, synthetic_1m_df):
        """Indicators compute without errors on 1m data."""
        from features.indicators import add_all_indicators
        df = add_all_indicators(synthetic_1m_df)
        assert len(df.columns) > 5  # At least OHLCV + some indicators

    def test_multi_timeframe_resample(self, synthetic_1m_df, config):
        """5m and 15m resampling works correctly from 1m data."""
        from features.multi_timeframe import add_multi_timeframe_features
        mtf_config = config["observation"]["multi_timeframe"]
        df = add_multi_timeframe_features(
            synthetic_1m_df, source_tf="1m", config=mtf_config
        )
        # Check that MTF columns were created
        mtf_cols = [c for c in df.columns if c.startswith("mtf_")]
        assert len(mtf_cols) > 0, "No multi-timeframe features created"
        # Check specific timeframes
        assert any("5m" in c for c in mtf_cols), "Missing 5m features"
        assert any("15m" in c for c in mtf_cols), "Missing 15m features"
        assert any("1h" in c for c in mtf_cols), "Missing 1h features"

    def test_normalization_bounded(self, synthetic_1m_df, config):
        """Normalized values should be roughly bounded after z-score."""
        from features.indicators import add_all_indicators
        from features.normalizer import rolling_normalize, clip_outliers
        
        df = add_all_indicators(synthetic_1m_df)
        df = df.dropna()
        lookback = config["data"]["lookback_window"]
        df_norm = rolling_normalize(df, window=lookback)
        df_norm = df_norm.dropna()
        df_norm = clip_outliers(df_norm, n_std=5.0)
        
        # All values should be within [-5, 5] after clipping
        numeric_cols = df_norm.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert df_norm[col].max() <= 5.0 + 1e-6, f"{col} exceeds 5.0"
            assert df_norm[col].min() >= -5.0 - 1e-6, f"{col} below -5.0"

    def test_no_nan_in_processed(self, processed_1m_df):
        """Fully processed DataFrame should have zero NaN values."""
        nan_count = processed_1m_df.isna().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values in processed data"

    def test_raw_columns_preserved(self, processed_1m_df):
        """raw_open/high/low/close must survive the pipeline."""
        for col in ["raw_open", "raw_high", "raw_low", "raw_close"]:
            assert col in processed_1m_df.columns, f"Missing {col}"

    def test_enough_rows_after_processing(self, processed_1m_df):
        """After dropna, we should still have enough rows for training."""
        assert len(processed_1m_df) > 500, (
            f"Only {len(processed_1m_df)} rows after processing — too few"
        )


# ---------------------------------------------------------------------------
# Environment with 1m data
# ---------------------------------------------------------------------------

class TestEnvWith1mData:
    """Test environment creation/stepping with processed 1m data."""

    def test_env_creation(self, processed_1m_df, config):
        """Environment creates without errors."""
        from env.trading_env import CryptoTradingEnv
        env = CryptoTradingEnv(processed_1m_df, config=config, mode="train")
        assert env is not None
        lookback = config["data"]["lookback_window"]
        assert env.observation_space.shape[0] == lookback

    def test_reset_and_step(self, processed_1m_df, config):
        """Can reset and step through 100 steps without crash."""
        from env.trading_env import CryptoTradingEnv
        env = CryptoTradingEnv(processed_1m_df, config=config, mode="test")
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape

        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert not np.isnan(reward), "Reward is NaN!"
            assert not np.any(np.isnan(obs)), "Observation contains NaN!"
            if terminated or truncated:
                break

    def test_obs_no_inf(self, processed_1m_df, config):
        """Observations should never contain inf values."""
        from env.trading_env import CryptoTradingEnv
        env = CryptoTradingEnv(processed_1m_df, config=config, mode="test")
        obs, _ = env.reset(seed=42)

        for _ in range(50):
            action = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(action)
            assert not np.any(np.isinf(obs)), "Observation contains Inf!"
            if term or trunc:
                break

    def test_sb3_check_env(self, processed_1m_df, config):
        """SB3 check_env passes on 1m environment."""
        from env.trading_env import CryptoTradingEnv
        from stable_baselines3.common.env_checker import check_env
        env = CryptoTradingEnv(processed_1m_df, config=config, mode="test")
        check_env(env, warn=True)


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

class TestCurriculum:
    """Verify curriculum levels are correctly configured for Phase 4."""

    def test_all_levels_single_pair(self, config):
        """All 3 curriculum levels should use BTC/USDT only."""
        from training.curriculum import set_curriculum_level
        for level in [1, 2, 3]:
            cfg = set_curriculum_level(config, level)
            assert cfg["market"]["pairs"] == ["BTC/USDT"], (
                f"Level {level} has pairs: {cfg['market']['pairs']}"
            )

    def test_level1_no_fees(self, config):
        from training.curriculum import set_curriculum_level
        cfg = set_curriculum_level(config, 1)
        assert cfg["fees"]["maker"] == 0.0
        assert cfg["fees"]["taker"] == 0.0

    def test_level2_real_fees(self, config):
        from training.curriculum import set_curriculum_level
        cfg = set_curriculum_level(config, 2)
        assert cfg["fees"]["taker"] > 0

    def test_level3_domain_randomization(self, config):
        from training.curriculum import set_curriculum_level
        cfg = set_curriculum_level(config, 3)
        assert cfg["training"]["domain_randomization"]["enabled"] is True

    def test_timesteps_increase(self, config):
        """Each level should train for more steps than the previous."""
        from training.curriculum import set_curriculum_level
        timesteps = []
        for level in [1, 2, 3]:
            cfg = set_curriculum_level(config, level)
            timesteps.append(cfg["training"]["total_timesteps"])
        assert timesteps[0] < timesteps[1] < timesteps[2], (
            f"Timesteps should increase: {timesteps}"
        )
