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
        "drawdown_penalty": {"enabled": True, "threshold_pct": 0.03, "penalty_factor": 5.0},
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


class TestNonPositiveCurrentValue:
    def test_non_positive_current_value_is_finite(self, reward_calc):
        """
        Regression test: short-selling can push NAV to zero or negative
        (impossible in the old long-only code). np.log() of a non-positive
        current_value produces NaN/-inf, which would silently corrupt PPO
        training. Both total and log_return must stay finite.
        """
        result = reward_calc.calculate(-100, 10000, 0.0, 0.0, 0.0)
        assert np.isfinite(result["total"])
        assert np.isfinite(result["log_return"])


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
        assert result["drawdown_pct"] > 0.03

    def test_no_permanent_penalty_while_flat_after_drop(self, reward_calc):
        """
        Regression test: once the drawdown stops worsening, an agent that
        holds perfectly flat (no trade, NAV unchanged) must stop paying the
        drawdown penalty. Before the fix, this penalty was re-applied every
        single step for the rest of the episode based on the stale historical
        peak, dwarfing every other reward term and making passivity the
        reward-optimal policy.
        """
        # Establish a peak, then a single drop past the threshold.
        reward_calc.calculate(12000, 10000, 0.0, 0.0, 0.0)
        first_drop = reward_calc.calculate(9000, 12000, 0.0, 0.0, 0.0)
        assert first_drop["drawdown_penalty"] < 0

        # Agent goes flat: NAV no longer moves, drawdown depth is unchanged.
        for _ in range(5000):
            result = reward_calc.calculate(9000, 9000, 0.0, 0.0, 0.0)

        assert result["drawdown_penalty"] == pytest.approx(0.0)

    def test_penalty_scales_with_incremental_worsening_only(self, reward_calc):
        """A deeper new low must still be penalized (only the increment)."""
        reward_calc.calculate(12000, 10000, 0.0, 0.0, 0.0)
        reward_calc.calculate(9000, 12000, 0.0, 0.0, 0.0)  # -25% drawdown

        # Drawdown worsens further to -40%: should be penalized again.
        worse = reward_calc.calculate(7200, 9000, 0.0, 0.0, 0.0)
        assert worse["drawdown_penalty"] < 0

        # Recovering slightly (still below peak, but less deep) must not
        # incur any further penalty.
        better = reward_calc.calculate(8000, 7200, 0.0, 0.0, 0.0)
        assert better["drawdown_penalty"] == pytest.approx(0.0)


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
