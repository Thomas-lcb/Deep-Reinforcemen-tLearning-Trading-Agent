"""
env/trading_env.py — Environnement de trading principal (Gymnasium).

CryptoTradingEnv simule le trading sur des données historiques OHLCV :
- Observation : fenêtre glissante de features normalisées + état du portefeuille
- Action : valeur continue [-1, 1] avec dead zone
- Reward : log-return pénalisé par frais, volatilité, drawdown
- Domain randomization : capital initial, frais, point de départ variables

cf. agent.md §5 et §6 Phase 2
"""

import os
import numpy as np
import pandas as pd
import gymnasium as gym
import yaml

from env.observation import build_observation_space, get_observation, PORTFOLIO_FEATURES
from env.action import build_action_space, interpret_action, apply_cooldown
from env.reward import RewardCalculator


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CryptoTradingEnv(gym.Env):
    """
    A Gymnasium environment for crypto trading with DRL.

    Supports:
    - Continuous action space with dead zone
    - Rolling normalized observations
    - Configurable reward function
    - Domain randomization
    - Trade logging
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        config: dict | None = None,
        mode: str = "train",
        render_mode: str | None = None,
    ):
        """
        Args:
            df: Processed DataFrame with normalized market features.
                Must contain an 'open', 'high', 'low', 'close' column (raw prices)
                and normalized feature columns.
            config: Full config dict from config.yaml.
            mode: 'train', 'val', or 'test'. Affects domain randomization.
            render_mode: Gymnasium render mode.
        """
        super().__init__()

        # Multi-asset support
        if isinstance(df, dict):
            self.dfs = df
            self.pairs = list(df.keys())
            self.current_pair = self.pairs[0]
            self.df = self.dfs[self.current_pair]
            self.multi_asset = True
        else:
            self.dfs = None
            self.multi_asset = False
            self.df = df
            self.current_pair = "Asset"
            
        # Load config
        if config is None:
            config_path = os.path.join(ROOT_DIR, "config", "config.yaml")
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

        self.config = config
        self.mode = mode
        self.render_mode = render_mode

        # --- Data ---
        # self.df is already set above (either from dict or direct df)
        self.df = self.df.reset_index(drop=True)
        self.n_steps = len(self.df)

        # Identify price columns (need raw close for portfolio valuation)
        if "raw_close" in self.df.columns:
            self.close_prices = self.df["raw_close"].values
        else:
            # Fallback (only if raw prices are missing, which shouldn't happen with new download)
            print("⚠️ Warning: 'raw_close' not found, using 'close'. Valuation may be wrong if normalized.")
            self.close_prices = self.df["close"].values

        # Feature columns (everything except raw OHLCV for the observation)
        self._feature_cols = [
            col for col in self.df.columns
            if col not in ["timestamp", "date"] and not col.startswith("raw_")
        ]
        self.n_features = len(self._feature_cols)

        # --- Spaces ---
        obs_cfg = config.get("observation", {})
        self.lookback_window = config.get("data", {}).get("lookback_window", 30)

        # Observation space
        total_obs_features = self.n_features + len(PORTFOLIO_FEATURES)
        self.observation_space = build_observation_space(
            n_features=self.n_features,
            lookback_window=self.lookback_window,
        )

        # Action space
        self.action_space = build_action_space()

        # --- Action config ---
        action_cfg = config.get("action", {})
        self.dead_zone = action_cfg.get("dead_zone", 0.05)
        self.cooldown_steps = action_cfg.get("cooldown_steps", 0)
        self.max_position_pct = action_cfg.get("max_position_pct", 1.0)

        # --- Fee config ---
        fee_cfg = config.get("fees", {})
        self.base_fee_rate = fee_cfg.get("taker", 0.001)

        # --- Domain randomization config ---
        dr_cfg = config.get("training", {}).get("domain_randomization", {})
        self.dr_enabled = dr_cfg.get("enabled", False) and mode == "train"
        self.dr_capital_var = dr_cfg.get("capital_variation_pct", 0.20)
        self.dr_fee_range = dr_cfg.get("fee_range", [0.0005, 0.0015])

        # --- Reward ---
        reward_cfg = config.get("reward", {})
        self.reward_calc = RewardCalculator(reward_cfg)

        # --- Default capital ---
        self.default_capital = 10_000.0  # $10,000 USDT

        # --- Internal state (initialized in reset) ---
        self.current_step = 0
        self.balance_usdt = 0.0
        self.balance_asset = 0.0
        self.entry_price = 0.0
        self.steps_since_trade = 0
        self.fee_rate = self.base_fee_rate
        self.initial_capital = self.default_capital
        self.trade_history = []
        self.portfolio_history = []

    def reset(self, seed=None, options=None):
        """Reset the environment to the start of a new episode."""
        super().reset(seed=seed)

        # Select random asset if multi-asset mode
        if self.multi_asset:
            self.current_pair = self.np_random.choice(self.pairs)
            self.df = self.dfs[self.current_pair]
            # Update price array references
            if "raw_close" in self.df.columns:
                self.close_prices = self.df["raw_close"].values
            else:
                self.close_prices = self.df["close"].values
            self.n_steps = len(self.df)

        # Domain randomization
        if self.dr_enabled:
            # Randomize starting capital
            cap_var = self.np_random.uniform(-self.dr_capital_var, self.dr_capital_var)
            self.initial_capital = self.default_capital * (1.0 + cap_var)

            # Randomize fee rate
            self.fee_rate = self.np_random.uniform(*self.dr_fee_range)

            # Randomize starting position in data
            max_start = max(0, self.n_steps - self.lookback_window - 500)
            if max_start > self.lookback_window:
                self.current_step = self.np_random.integers(
                    self.lookback_window, max_start
                )
            else:
                self.current_step = self.lookback_window
        else:
            self.initial_capital = self.default_capital
            self.fee_rate = self.base_fee_rate
            self.current_step = self.lookback_window

        # Reset portfolio
        self.balance_usdt = self.initial_capital
        self.balance_asset = 0.0
        self.entry_price = 0.0
        self.steps_since_trade = 0
        self.trade_history = []
        self.portfolio_history = []

        # Reset reward calculator
        self.reward_calc.reset(self.initial_capital)

        # Get initial observation
        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action: np.ndarray of shape (1,) with value in [-1, 1].

        Returns:
            obs, reward, terminated, truncated, info
        """
        raw_action = float(action[0])
        current_price = self.close_prices[self.current_step]
        prev_value = self._portfolio_value()

        # --- Interpret action ---
        trade = interpret_action(
            raw_action=raw_action,
            balance_usdt=self.balance_usdt,
            balance_asset=self.balance_asset,
            asset_price=current_price,
            dead_zone=self.dead_zone,
            fee_rate=self.fee_rate,
            max_position_pct=self.max_position_pct,
        )

        # Apply cooldown
        trade = apply_cooldown(self.steps_since_trade, self.cooldown_steps, trade)

        # --- Execute trade ---
        if trade["type"] == "buy" and trade["amount_usdt"] > 0:
            self.balance_usdt -= trade["amount_usdt"]
            self.balance_asset += trade["amount_asset"]

            # Update entry price (weighted average)
            if self.balance_asset > 0:
                total_cost = self.entry_price * (self.balance_asset - trade["amount_asset"]) + \
                             trade["amount_usdt"]
                self.entry_price = total_cost / self.balance_asset

            self.steps_since_trade = 0
            self._log_trade(trade, current_price)

        elif trade["type"] == "sell" and trade["amount_asset"] > 0:
            self.balance_usdt += trade["amount_usdt"]  # net of fees
            self.balance_asset -= trade["amount_asset"]

            # Reset entry price if fully closed
            if self.balance_asset < 1e-10:
                self.balance_asset = 0.0
                self.entry_price = 0.0

            self.steps_since_trade = 0
            self._log_trade(trade, current_price)

        else:
            self.steps_since_trade += 1

        # --- Advance time ---
        self.current_step += 1

        # --- Calculate reward ---
        current_value = self._portfolio_value()
        current_price_now = self.close_prices[min(self.current_step, self.n_steps - 1)]

        # Get current volatility from ATR if available
        current_volatility = self._get_current_volatility()

        # Get trend direction from multi-TF features if available
        trend_dir = self._get_trend_direction()

        # Position direction: +1 if long (holding asset), 0 if flat
        pos_dir = 1.0 if self.balance_asset > 1e-10 else 0.0

        reward_info = self.reward_calc.calculate(
            current_value=current_value,
            previous_value=prev_value,
            action_magnitude=abs(raw_action) if trade["type"] != "hold" else 0.0,
            fee=trade["fee"],
            current_volatility=current_volatility,
            trend_direction=trend_dir,
            position_direction=pos_dir,
        )

        reward = reward_info["total"]

        # --- Track portfolio ---
        self.portfolio_history.append({
            "step": self.current_step,
            "value": current_value,
            "action": raw_action,
            "trade_type": trade["type"],
            "reward": reward,
        })

        # --- Check termination ---
        terminated = False
        truncated = False

        # End of data
        if self.current_step >= self.n_steps - 1:
            truncated = True

        # Portfolio collapsed (lost > 95% of initial capital)
        if current_value < self.initial_capital * 0.05:
            terminated = True

        # Get observation
        obs = self._get_obs()
        info = self._get_info()
        info.update(reward_info)
        info["trade"] = trade

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _portfolio_value(self) -> float:
        """Calculate the current Net Asset Value."""
        price_idx = min(self.current_step, self.n_steps - 1)
        current_price = self.close_prices[price_idx]
        return self.balance_usdt + self.balance_asset * current_price

    def _get_obs(self) -> np.ndarray:
        """Get the current observation."""
        # Ensure we have enough lookback
        start = max(0, self.current_step - self.lookback_window)
        end = self.current_step

        # Market data (normalized features)
        market_data = self.df[self._feature_cols].values[start:end]

        # Pad if we don't have enough history
        if len(market_data) < self.lookback_window:
            pad_size = self.lookback_window - len(market_data)
            padding = np.zeros((pad_size, self.n_features), dtype=np.float32)
            market_data = np.vstack([padding, market_data])

        current_price = self.close_prices[min(self.current_step, self.n_steps - 1)]

        obs = get_observation(
            market_data=market_data,
            balance_usdt=self.balance_usdt,
            balance_asset=self.balance_asset,
            asset_price=current_price,
            entry_price=self.entry_price,
            steps_since_trade=self.steps_since_trade,
            initial_capital=self.initial_capital,
            lookback_window=self.lookback_window,
        )

        return obs

    def _get_current_volatility(self) -> float:
        """Get the current volatility from ATR or rolling std."""
        # Try ATR column
        atr_cols = [c for c in self._feature_cols if "atr" in c.lower() and "pct" in c.lower()]
        if atr_cols:
            idx = min(self.current_step, self.n_steps - 1)
            val = self.df[atr_cols[0]].iloc[idx]
            return float(val) if not np.isnan(val) else 0.0

        # Fallback: rolling std of returns
        window = min(20, self.current_step)
        if window > 1:
            recent = self.close_prices[max(0, self.current_step - window):self.current_step]
            returns = np.diff(recent) / (recent[:-1] + 1e-10)
            return float(np.std(returns))

        return 0.0

    def _get_trend_direction(self) -> float:
        """Get macro trend direction from multi-TF EMA if available."""
        ema_cols = [c for c in self._feature_cols if "ema200_dir" in c.lower()]
        if ema_cols:
            idx = min(self.current_step, self.n_steps - 1)
            val = self.df[ema_cols[0]].iloc[idx]
            if not np.isnan(val):
                return float(val)
        return 0.0

    def _get_info(self) -> dict:
        """Get info dict for the current state."""
        value = self._portfolio_value()
        return {
            "portfolio_value": value,
            "balance_usdt": self.balance_usdt,
            "balance_asset": self.balance_asset,
            "entry_price": self.entry_price,
            "steps_since_trade": self.steps_since_trade,
            "return_pct": (value - self.initial_capital) / self.initial_capital * 100,
            "n_trades": len(self.trade_history),
            "current_step": self.current_step,
            "total_steps": self.n_steps,
        }

    def _log_trade(self, trade: dict, price: float):
        """Log a trade for analysis."""
        self.trade_history.append({
            "step": self.current_step,
            "type": trade["type"],
            "price": price,
            "amount_usdt": trade["amount_usdt"],
            "amount_asset": trade["amount_asset"],
            "proportion": trade["proportion"],
            "fee": trade["fee"],
            "portfolio_value": self._portfolio_value(),
            "balance_usdt": self.balance_usdt,
            "balance_asset": self.balance_asset,
        })

    def render(self):
        """Render the current state."""
        if self.render_mode == "human":
            info = self._get_info()
            price = self.close_prices[min(self.current_step, self.n_steps - 1)]
            print(
                f"Step {info['current_step']:5d}/{info['total_steps']} | "
                f"Price: ${price:,.2f} | "
                f"NAV: ${info['portfolio_value']:,.2f} | "
                f"Return: {info['return_pct']:+.2f}% | "
                f"Trades: {info['n_trades']}"
            )

    def get_trade_history(self) -> pd.DataFrame:
        """Return trade history as DataFrame."""
        if not self.trade_history:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_history)

    def get_portfolio_history(self) -> pd.DataFrame:
        """Return portfolio value history as DataFrame."""
        if not self.portfolio_history:
            return pd.DataFrame()
        return pd.DataFrame(self.portfolio_history)
