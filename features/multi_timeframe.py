"""
features/multi_timeframe.py — Agrégation multi-timeframe.

L'agent trade en 1H mais reçoit aussi des features agrégées
sur des timeframes supérieurs (4H, 1D) pour capter la tendance de fond.

cf. agent.md §5.A.3
"""

import pandas as pd
import numpy as np
import pandas_ta as ta


def add_multi_timeframe_features(
    df: pd.DataFrame,
    source_tf: str = "1h",
    config: dict | None = None,
) -> pd.DataFrame:
    """
    Add multi-timeframe features to the 1H DataFrame.

    Resamples the data to higher timeframes (4H, 1D), computes indicators
    on those, then merges back to the original 1H index via forward-fill
    (to avoid look-ahead bias).

    Args:
        df: 1H OHLCV DataFrame with DatetimeIndex.
        source_tf: Source timeframe string (e.g. "1h").
        config: Multi-timeframe config from config.yaml.

    Returns:
        DataFrame with additional multi-TF columns.
    """
    df = df.copy()
    cfg = config or {}
    target_tfs = cfg.get("timeframes", ["4h", "1d"])
    features = cfg.get("features", ["ema_200_direction", "rsi", "atr"])

    # Mapping from config timeframe strings to pandas resample rules
    tf_to_resample = {
        "4h": "4h",
        "1d": "1D",
        "1D": "1D",
        "4H": "4h",
    }

    for tf in target_tfs:
        resample_rule = tf_to_resample.get(tf, tf)
        prefix = f"mtf_{tf}_"

        # Resample OHLCV
        df_resampled = _resample_ohlcv(df, resample_rule)

        if df_resampled.empty:
            continue

        # Compute features on resampled data
        tf_features = _compute_tf_features(df_resampled, features, prefix)

        # Merge back to original index (forward-fill to avoid look-ahead)
        df = df.join(tf_features, how="left")
        for col in tf_features.columns:
            df[col] = df[col].ffill()

    return df


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV data to a higher timeframe."""
    try:
        resampled = df.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        return resampled
    except Exception as e:
        print(f"  ⚠️ Resample error ({rule}): {e}")
        return pd.DataFrame()


def _compute_tf_features(
    df: pd.DataFrame,
    features: list[str],
    prefix: str,
) -> pd.DataFrame:
    """
    Compute specified features on a resampled DataFrame.

    Available features:
    - ema_200_direction : direction (+1/-1) of EMA 200
    - rsi : RSI 14
    - atr : ATR 14 normalized by close
    - macd_hist : MACD histogram
    - trend_strength : abs(close - ema_200) / atr (how strong the trend is)
    """
    result = pd.DataFrame(index=df.index)

    for feat in features:
        if feat == "ema_200_direction":
            ema = ta.ema(df["close"], length=200)
            if ema is not None:
                # +1 if price above EMA 200, -1 otherwise
                result[f"{prefix}ema200_dir"] = np.where(
                    df["close"] > ema, 1.0, -1.0
                )
                # Distance from EMA 200 as a ratio
                result[f"{prefix}ema200_dist"] = (df["close"] - ema) / (ema + 1e-10)

        elif feat == "rsi":
            rsi = ta.rsi(df["close"], length=14)
            if rsi is not None:
                # Normalize RSI to [-1, 1] range (originally 0-100)
                result[f"{prefix}rsi"] = (rsi - 50) / 50

        elif feat == "atr":
            atr = ta.atr(df["high"], df["low"], df["close"], length=14)
            if atr is not None:
                # ATR as percentage of close price
                result[f"{prefix}atr_pct"] = atr / (df["close"] + 1e-10)

        elif feat == "macd_hist":
            macd = ta.macd(df["close"])
            if macd is not None:
                hist_col = [c for c in macd.columns if "MACDh" in c or "MACD_12_26_9" in c]
                if hist_col:
                    result[f"{prefix}macd_hist"] = macd[hist_col[0]]

        elif feat == "trend_strength":
            ema = ta.ema(df["close"], length=200)
            atr = ta.atr(df["high"], df["low"], df["close"], length=14)
            if ema is not None and atr is not None:
                result[f"{prefix}trend_str"] = (
                    (df["close"] - ema).abs() / (atr + 1e-10)
                )

    return result
