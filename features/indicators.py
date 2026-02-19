"""
features/indicators.py — Calcul des indicateurs techniques.

Ajoute les colonnes d'indicateurs au DataFrame OHLCV :
- Tendance   : MACD, EMA (50, 200)
- Oscillateurs : RSI, CCI
- Volatilité : ATR, Bandes de Bollinger
- Volume     : OBV

cf. agent.md §5.A.2
"""

import pandas as pd
import pandas_ta as ta


def add_all_indicators(df: pd.DataFrame, obs_config: dict | None = None) -> pd.DataFrame:
    """
    Add all technical indicators to the OHLCV DataFrame.

    Args:
        df: DataFrame with columns [open, high, low, close, volume].
        obs_config: Observation config from config.yaml (optional overrides).

    Returns:
        DataFrame with indicator columns appended.
    """
    df = df.copy()
    cfg = (obs_config or {}).get("indicators", {})

    # --- Trend ---
    df = _add_ema(df, cfg)
    df = _add_macd(df, cfg)

    # --- Oscillators ---
    df = _add_rsi(df, cfg)
    df = _add_cci(df, cfg)

    # --- Volatility ---
    df = _add_atr(df, cfg)
    df = _add_bollinger(df, cfg)

    # --- Volume ---
    if cfg.get("obv", True):
        df = _add_obv(df)

    # --- Returns (for normalization later) ---
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = df["close"].apply(lambda x: x).pct_change()  # will be log in normalizer

    return df


# ---------------------------------------------------------------------------
# Individual indicator functions
# ---------------------------------------------------------------------------

def _add_ema(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Exponential Moving Averages."""
    periods = cfg.get("ema_periods", [50, 200])
    for p in periods:
        df[f"ema_{p}"] = ta.ema(df["close"], length=p)
    # Price relative to EMAs (useful feature)
    for p in periods:
        df[f"close_vs_ema_{p}"] = (df["close"] - df[f"ema_{p}"]) / df[f"ema_{p}"]
    return df


def _add_macd(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """MACD (Moving Average Convergence Divergence)."""
    fast = cfg.get("macd_fast", 12)
    slow = cfg.get("macd_slow", 26)
    signal = cfg.get("macd_signal", 9)

    macd = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
    if macd is not None:
        df = pd.concat([df, macd], axis=1)
    return df


def _add_rsi(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Relative Strength Index."""
    period = cfg.get("rsi_period", 14)
    df[f"rsi_{period}"] = ta.rsi(df["close"], length=period)
    return df


def _add_cci(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Commodity Channel Index."""
    period = cfg.get("cci_period", 20)
    df[f"cci_{period}"] = ta.cci(df["high"], df["low"], df["close"], length=period)
    return df


def _add_atr(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Average True Range (volatility)."""
    period = cfg.get("atr_period", 14)
    df[f"atr_{period}"] = ta.atr(df["high"], df["low"], df["close"], length=period)
    # Normalize ATR by close price for comparability
    df[f"atr_{period}_pct"] = df[f"atr_{period}"] / df["close"]
    return df


def _add_bollinger(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Bollinger Bands."""
    period = cfg.get("bollinger_period", 20)
    std = cfg.get("bollinger_std", 2.0)

    bbands = ta.bbands(df["close"], length=period, std=std)
    if bbands is not None:
        df = pd.concat([df, bbands], axis=1)
        # Bollinger %B — position of price within bands
        upper_col = [c for c in bbands.columns if "BBU" in c]
        lower_col = [c for c in bbands.columns if "BBL" in c]
        if upper_col and lower_col:
            df["bb_pct_b"] = (df["close"] - df[lower_col[0]]) / (
                df[upper_col[0]] - df[lower_col[0]] + 1e-10
            )
    return df


def _add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """On-Balance Volume."""
    df["obv"] = ta.obv(df["close"], df["volume"])
    # Normalize OBV with its own rolling mean for scale invariance
    df["obv_normalized"] = df["obv"] / (df["obv"].rolling(50).mean() + 1e-10)
    return df
