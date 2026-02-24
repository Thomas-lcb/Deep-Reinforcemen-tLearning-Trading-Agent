"""
env/reward.py — Fonctions de récompense modulaires.

Reward principale basée sur les log-returns avec pénalités :
- Frais de transaction
- Volatilité (risk aversion λ)
- Drawdown excessif
- Bonus : Sharpe glissant + alignement de tendance

cf. agent.md §5.C
"""

import numpy as np
from collections import deque


class RewardCalculator:
    """
    Calculates the reward signal for the trading environment.

    Formula:
        R_t = log(v_t / v_{t-1}) - c * |action| - λ * σ_t
              + sharpe_bonus + trend_bonus - drawdown_penalty
    """

    def __init__(self, config: dict | None = None):
        """
        Args:
            config: Reward config from config.yaml.
        """
        cfg = config or {}

        # Core weights
        self.fee_penalty_weight = cfg.get("fee_penalty_weight", 1.0)
        self.volatility_penalty = cfg.get("volatility_penalty", 0.1)  # λ
        self.reward_scale = cfg.get("reward_scale", 1.0)

        # Drawdown penalty
        dd_cfg = cfg.get("drawdown_penalty", {})
        self.dd_enabled = dd_cfg.get("enabled", True)
        self.dd_threshold = dd_cfg.get("threshold_pct", 0.05)
        self.dd_factor = dd_cfg.get("penalty_factor", 2.0)

        # Sharpe bonus
        sharpe_cfg = cfg.get("sharpe_bonus", {})
        self.sharpe_enabled = sharpe_cfg.get("enabled", True)
        self.sharpe_window = sharpe_cfg.get("window", 100)
        self.sharpe_weight = sharpe_cfg.get("weight", 0.01)

        # Trend alignment bonus
        trend_cfg = cfg.get("trend_alignment_bonus", {})
        self.trend_enabled = trend_cfg.get("enabled", True)
        self.trend_weight = trend_cfg.get("weight", 0.005)

        # State tracking
        self.portfolio_values = deque(maxlen=self.sharpe_window + 1)
        self.returns_history = deque(maxlen=self.sharpe_window)
        self.peak_value = 0.0

    def reset(self, initial_value: float):
        """Reset the reward calculator at the start of a new episode."""
        self.portfolio_values.clear()
        self.returns_history.clear()
        self.portfolio_values.append(initial_value)
        self.peak_value = initial_value

    def calculate(
        self,
        current_value: float,
        previous_value: float,
        action_magnitude: float,
        fee: float,
        current_volatility: float,
        trend_direction: float = 0.0,
        position_direction: float = 0.0,
    ) -> dict:
        """
        Calculate the reward for the current step.

        Args:
            current_value: Current portfolio NAV.
            previous_value: Previous portfolio NAV.
            action_magnitude: |action| for fee penalty (0 = hold).
            fee: Actual fee paid this step.
            current_volatility: Recent volatility (e.g. ATR or rolling std).
            trend_direction: +1 if macro trend is up, -1 if down, 0 if neutral.
            position_direction: +1 if agent is long, -1 if short, 0 if flat.

        Returns:
            Dict with 'total' reward and breakdown of components.
        """
        # Track portfolio value
        self.portfolio_values.append(current_value)

        # --- 1. Log-return ---
        if previous_value > 0:
            log_return = np.log(current_value / previous_value)
        else:
            log_return = 0.0
        self.returns_history.append(log_return)

        # --- 2. Fee penalty ---
        fee_penalty = -self.fee_penalty_weight * action_magnitude * (fee / (previous_value + 1e-10))

        # --- 3. Volatility penalty ---
        vol_penalty = -self.volatility_penalty * current_volatility

        # --- 4. Drawdown penalty ---
        drawdown_penalty = 0.0
        if self.dd_enabled:
            self.peak_value = max(self.peak_value, current_value)
            if self.peak_value > 0:
                drawdown = (self.peak_value - current_value) / self.peak_value
                if drawdown > self.dd_threshold:
                    drawdown_penalty = -self.dd_factor * (drawdown - self.dd_threshold)

        # --- 5. Rolling Sharpe bonus ---
        sharpe_bonus = 0.0
        if self.sharpe_enabled and len(self.returns_history) >= 20:
            returns_arr = np.array(self.returns_history)
            mean_r = returns_arr.mean()
            std_r = returns_arr.std()
            if std_r > 1e-8:
                rolling_sharpe = mean_r / std_r
                sharpe_bonus = self.sharpe_weight * rolling_sharpe

        # --- 6. Trend alignment bonus ---
        trend_bonus = 0.0
        if self.trend_enabled and trend_direction != 0.0 and position_direction != 0.0:
            # Positive if agent is aligned with the macro trend
            alignment = trend_direction * position_direction
            trend_bonus = self.trend_weight * alignment

        # --- Total ---
        total = log_return + fee_penalty + vol_penalty + drawdown_penalty + sharpe_bonus + trend_bonus
        if self.reward_scale and self.reward_scale != 0.0:
            total = total / self.reward_scale

        return {
            "total": float(total),
            "log_return": float(log_return),
            "fee_penalty": float(fee_penalty),
            "vol_penalty": float(vol_penalty),
            "drawdown_penalty": float(drawdown_penalty),
            "sharpe_bonus": float(sharpe_bonus),
            "trend_bonus": float(trend_bonus),
            "drawdown_pct": float(
                (self.peak_value - current_value) / self.peak_value if self.peak_value > 0 else 0.0
            ),
        }
