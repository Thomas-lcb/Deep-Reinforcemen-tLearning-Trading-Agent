"""
env/observation.py — Construction de l'Observation Space.

Gère la création de la fenêtre d'observation que l'agent reçoit à chaque step :
- OHLCV normalisé (fenêtre glissante)
- Indicateurs techniques
- Features multi-timeframe
- État du portefeuille (solde, PnL, temps depuis dernier trade)

cf. agent.md §5.A
"""

import numpy as np
import gymnasium as gym


# Colonnes OHLCV brutes
OHLCV_COLS = ["open", "high", "low", "close", "volume"]

# Colonnes du portefeuille ajoutées à l'observation
PORTFOLIO_FEATURES = [
    "balance_usdt_pct",      # % du capital en USDT
    "balance_asset_pct",     # % du capital en asset
    "unrealized_pnl_pct",    # PnL non-réalisé en %
    "steps_since_trade",     # Steps depuis le dernier trade (normalisé)
]


def build_observation_space(
    n_features: int,
    lookback_window: int,
) -> gym.spaces.Box:
    """
    Create the observation space for the trading environment.

    Shape: (lookback_window, n_features + len(PORTFOLIO_FEATURES))
    All values are normalized to roughly [-5, 5] after z-score + clipping.

    Args:
        n_features: Number of market features (OHLCV + indicators + multi-TF).
        lookback_window: Number of past timesteps in the observation window.

    Returns:
        gym.spaces.Box observation space.
    """
    total_features = n_features + len(PORTFOLIO_FEATURES)
    return gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(lookback_window, total_features),
        dtype=np.float32,
    )


def get_observation(
    market_data: np.ndarray,
    balance_usdt: float,
    balance_asset: float,
    asset_price: float,
    entry_price: float,
    steps_since_trade: int,
    initial_capital: float,
    lookback_window: int,
    max_steps_normalize: int = 100,
) -> np.ndarray:
    """
    Construct the observation array for the current timestep.

    Args:
        market_data: 2D array of shape (lookback_window, n_market_features).
                     Already normalized (rolling z-score).
        balance_usdt: Current USDT balance.
        balance_asset: Current asset (e.g. BTC) quantity.
        asset_price: Current asset price (raw, not normalized).
        entry_price: Average entry price of the current position.
        steps_since_trade: Number of steps since the last trade.
        initial_capital: Starting capital (for normalization).
        lookback_window: Window size.
        max_steps_normalize: Normalization constant for steps_since_trade.

    Returns:
        Observation array of shape (lookback_window, total_features).
    """
    # Portfolio value
    total_value = balance_usdt + balance_asset * asset_price

    # Portfolio features (repeated across the lookback window)
    balance_usdt_pct = balance_usdt / (total_value + 1e-10)
    balance_asset_pct = (balance_asset * asset_price) / (total_value + 1e-10)

    # Unrealized PnL
    if balance_asset > 0 and entry_price > 0:
        unrealized_pnl_pct = (asset_price - entry_price) / entry_price
    else:
        unrealized_pnl_pct = 0.0

    # Normalize steps since trade
    steps_norm = min(steps_since_trade / max_steps_normalize, 1.0)

    # Build portfolio vector
    portfolio_vec = np.array([
        balance_usdt_pct,
        balance_asset_pct,
        unrealized_pnl_pct,
        steps_norm,
    ], dtype=np.float32)

    # Repeat portfolio features for each timestep in the window
    portfolio_matrix = np.tile(portfolio_vec, (lookback_window, 1))

    # Concatenate market data + portfolio
    observation = np.concatenate([market_data, portfolio_matrix], axis=1)

    # Sanitize NaN/Inf that can come from indicator warm-up periods
    observation = np.nan_to_num(observation, nan=0.0, posinf=5.0, neginf=-5.0)

    return observation.astype(np.float32)


def get_feature_columns(df_columns: list[str]) -> list[str]:
    """
    Extract the list of market feature columns from a processed DataFrame.
    Excludes raw OHLCV (we use normalized versions) and metadata columns.

    Args:
        df_columns: List of column names from the processed DataFrame.

    Returns:
        Filtered list of feature column names.
    """
    # Exclude raw timestamp or non-feature columns
    exclude_patterns = ["timestamp"]
    features = [
        col for col in df_columns
        if not any(pat in col.lower() for pat in exclude_patterns)
    ]
    return features
