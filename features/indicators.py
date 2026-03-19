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
import ta


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
        df[f"ema_{p}"] = ta.trend.EMAIndicator(close=df["close"], window=p).ema_indicator()
    # Price relative to EMAs (useful feature)
    for p in periods:
        df[f"close_vs_ema_{p}"] = (df["close"] - df[f"ema_{p}"]) / df[f"ema_{p}"]
    return df


def _add_macd(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """MACD (Moving Average Convergence Divergence)."""
    fast = cfg.get("macd_fast", 12)
    slow = cfg.get("macd_slow", 26)
    signal = cfg.get("macd_signal", 9)

    macd = ta.trend.MACD(close=df["close"], window_fast=fast, window_slow=slow, window_sign=signal)
    df[f"MACD_{fast}_{slow}_{signal}"] = macd.macd()
    df[f"MACDh_{fast}_{slow}_{signal}"] = macd.macd_diff()
    df[f"MACDs_{fast}_{slow}_{signal}"] = macd.macd_signal()
    return df


def _add_rsi(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Relative Strength Index."""
    period = cfg.get("rsi_period", 14)
    df[f"rsi_{period}"] = ta.momentum.RSIIndicator(close=df["close"], window=period).rsi()
    return df


def _add_cci(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Commodity Channel Index."""
    period = cfg.get("cci_period", 20)
    df[f"cci_{period}"] = ta.trend.CCIIndicator(high=df["high"], low=df["low"], close=df["close"], window=period).cci()
    return df


def _add_atr(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Average True Range (volatility)."""
    period = cfg.get("atr_period", 14)
    df[f"atr_{period}"] = ta.volatility.AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=period).average_true_range()
    # Normalize ATR by close price for comparability
    df[f"atr_{period}_pct"] = df[f"atr_{period}"] / df["close"]
    return df


def _add_bollinger(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Bollinger Bands."""
    period = cfg.get("bollinger_period", 20)
    std = cfg.get("bollinger_std", 2.0)

    bb = ta.volatility.BollingerBands(close=df["close"], window=period, window_dev=std)
    df[f"BBL_{period}_{std}"] = bb.bollinger_lband()
    df[f"BBM_{period}_{std}"] = bb.bollinger_mavg()
    df[f"BBU_{period}_{std}"] = bb.bollinger_hband()
    df["bb_pct_b"] = bb.bollinger_pband()
    return df


def _add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """On-Balance Volume."""
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"]).on_balance_volume()
    # Normalize OBV with its own rolling mean for scale invariance
    df["obv_normalized"] = df["obv"] / (df["obv"].rolling(50).mean() + 1e-10)
    return df
