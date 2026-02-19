"""
tests/test_reward.py — Tests unitaires de la reward function.
"""

import pytest
import numpy as np
from env.reward import RewardCalculator


@pytest.fixture
def reward_calc():
    """Create a default RewardCalculator."""
    config = {
        "fee_penalty_weight": 1.0,
        "volatility_penalty": 0.1,
        "drawdown_penalty": {"enabled": True, "threshold_pct": 0.05, "penalty_factor": 2.0},
        "sharpe_bonus": {"enabled": True, "window": 100, "weight": 0.01},
        "trend_alignment_bonus": {"enabled": True, "weight": 0.005},
    }
    calc = RewardCalculator(config)
    calc.reset(10000.0)
    return calc


class TestLogReturn:
    def test_positive_return(self, reward_calc):
        result = reward_calc.calculate(10100, 10000, 0.0, 0.0, 0.0)
        assert result["log_return"] > 0

    def test_negative_return(self, reward_calc):
        result = reward_calc.calculate(9900, 10000, 0.0, 0.0, 0.0)
        assert result["log_return"] < 0

    def test_zero_return(self, reward_calc):
        result = reward_calc.calculate(10000, 10000, 0.0, 0.0, 0.0)
        assert result["log_return"] == pytest.approx(0.0)


class TestFeePenalty:
    def test_fee_applied(self, reward_calc):
        result = reward_calc.calculate(10000, 10000, 0.8, 10.0, 0.0)
        assert result["fee_penalty"] < 0

    def test_no_fee_on_hold(self, reward_calc):
        result = reward_calc.calculate(10000, 10000, 0.0, 0.0, 0.0)
        assert result["fee_penalty"] == pytest.approx(0.0)


class TestDrawdownPenalty:
    def test_no_penalty_at_peak(self, reward_calc):
        result = reward_calc.calculate(10500, 10000, 0.0, 0.0, 0.0)
        assert result["drawdown_penalty"] == 0.0

    def test_penalty_after_big_drop(self, reward_calc):
        # Set peak high
        reward_calc.calculate(11000, 10000, 0.0, 0.0, 0.0)
        # Then drop below threshold
        result = reward_calc.calculate(10000, 11000, 0.0, 0.0, 0.0)
        assert result["drawdown_penalty"] < 0
        assert result["drawdown_pct"] > 0.05


class TestTrendBonus:
    def test_aligned(self, reward_calc):
        result = reward_calc.calculate(10000, 10000, 0.0, 0.0, 0.0,
                                       trend_direction=1.0, position_direction=1.0)
        assert result["trend_bonus"] > 0

    def test_misaligned(self, reward_calc):
        result = reward_calc.calculate(10000, 10000, 0.0, 0.0, 0.0,
                                       trend_direction=1.0, position_direction=-1.0)
        assert result["trend_bonus"] < 0

    def test_neutral(self, reward_calc):
        result = reward_calc.calculate(10000, 10000, 0.0, 0.0, 0.0,
                                       trend_direction=0.0, position_direction=1.0)
        assert result["trend_bonus"] == 0.0


class TestRewardComponents:
    def test_all_keys_present(self, reward_calc):
        result = reward_calc.calculate(10050, 10000, 0.5, 5.0, 0.02)
        expected_keys = ["total", "log_return", "fee_penalty", "vol_penalty",
                         "drawdown_penalty", "sharpe_bonus", "trend_bonus", "drawdown_pct"]
        for key in expected_keys:
            assert key in result

    def test_total_is_sum(self, reward_calc):
        result = reward_calc.calculate(10050, 10000, 0.5, 5.0, 0.02)
        computed = (result["log_return"] + result["fee_penalty"] + result["vol_penalty"]
                    + result["drawdown_penalty"] + result["sharpe_bonus"] + result["trend_bonus"])
        assert result["total"] == pytest.approx(computed, abs=1e-8)
