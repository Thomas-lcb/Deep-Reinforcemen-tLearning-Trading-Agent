"""
features/normalizer.py — Rolling normalization (anti data-leakage).

Au lieu d'un MinMaxScaler global (qui utilise des infos futures),
on applique un z-score glissant : chaque valeur est normalisée
par rapport à la moyenne et l'écart-type des N dernières bougies.

cf. agent.md §5.A.1 — WARNING sur le data leakage
"""

import pandas as pd
import numpy as np


def rolling_normalize(
    df: pd.DataFrame,
    window: int = 30,
    columns: list[str] | None = None,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    Apply rolling z-score normalization to the DataFrame.

    z_t = (x_t - μ_{t-window:t}) / (σ_{t-window:t} + ε)

    Args:
        df: Input DataFrame.
        window: Lookback window for rolling statistics.
        columns: Specific columns to normalize (default: all numeric).
        min_periods: Minimum periods for rolling calculation (default: window).

    Returns:
        Normalized DataFrame (same shape).
    """
    df = df.copy()
    min_p = min_periods or window

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue

        rolling_mean = df[col].rolling(window=window, min_periods=min_p).mean()
        rolling_std = df[col].rolling(window=window, min_periods=min_p).std()

        # z-score with epsilon to avoid division by zero
        df[col] = (df[col] - rolling_mean) / (rolling_std + 1e-8)

    return df


def normalize_ohlcv_as_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Alternative normalization: convert OHLCV prices to percentage returns.
    This is naturally stationary and avoids leakage.

    - open, high, low, close → pct_change()
    - volume → pct_change()

    Args:
        df: DataFrame with OHLCV columns.

    Returns:
        DataFrame with returns instead of absolute prices.
    """
    df = df.copy()
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        if col in df.columns:
            df[f"{col}_return"] = df[col].pct_change()

    if "volume" in df.columns:
        df["volume_return"] = df["volume"].pct_change()
        # Clip extreme volume spikes
        df["volume_return"] = df["volume_return"].clip(-5, 5)

    return df


def clip_outliers(
    df: pd.DataFrame,
    n_std: float = 5.0,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Clip extreme outliers to ±n_std standard deviations.
    Applied after normalization to prevent extreme values from
    destabilizing the neural network.

    Args:
        df: Normalized DataFrame.
        n_std: Number of standard deviations for clipping.
        columns: Columns to clip (default: all numeric).

    Returns:
        Clipped DataFrame.
    """
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col in df.columns:
            df[col] = df[col].clip(-n_std, n_std)

    return df
