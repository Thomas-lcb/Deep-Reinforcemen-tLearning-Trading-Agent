"""
data/download.py — Téléchargement des données OHLCV historiques via CCXT.

Télécharge les données pour toutes les paires configurées dans config.yaml,
les sauvegarde en CSV dans data/raw/, puis lance le pipeline de features
(indicateurs + normalisation + multi-timeframe) et sauvegarde dans data/processed/.

Usage:
    python -m data.download
    python -m data.download --pair BTC/USDT --timeframe 1h --years 5
"""

import os
import time
import argparse
from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "config.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _init_exchange(exchange_id: str) -> ccxt.Exchange:
    """Initialize a CCXT exchange instance (public endpoints only)."""
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    exchange.load_markets()
    return exchange


def _fetch_ohlcv_chunked(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    limit: int = 1000,
) -> list:
    """
    Fetch OHLCV data in chunks to work around exchange API limits.
    Binance allows max 1000 candles per request.
    """
    all_candles = []
    current_since = since_ms

    # Calculate ms per candle for the given timeframe
    tf_map = {
        "1m": 60_000, "5m": 300_000, "15m": 900_000,
        "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000,
        "1d": 86_400_000,
    }
    ms_per_candle = tf_map.get(timeframe, 3_600_000)

    total_expected = (until_ms - since_ms) // ms_per_candle
    fetched = 0

    while current_since < until_ms:
        try:
            candles = exchange.fetch_ohlcv(
                symbol, timeframe, since=current_since, limit=limit
            )
        except ccxt.RateLimitExceeded:
            print("  ⏳ Rate limit hit, sleeping 60s...")
            time.sleep(60)
            continue
        except ccxt.NetworkError as e:
            print(f"  ⚠️ Network error: {e}, retrying in 10s...")
            time.sleep(10)
            continue

        if not candles:
            break

        all_candles.extend(candles)
        fetched += len(candles)

        # Progress
        pct = min(100, int(fetched / max(total_expected, 1) * 100))
        print(f"  📊 {symbol} {timeframe}: {fetched} candles ({pct}%)", end="\r")

        # Move to next chunk
        last_ts = candles[-1][0]
        current_since = last_ts + ms_per_candle

        # Respect rate limits
        time.sleep(exchange.rateLimit / 1000)

    print()  # Newline after progress
    return all_candles


def download_pair(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    years: int,
    output_dir: str,
) -> pd.DataFrame:
    """Download OHLCV data for a single pair and save to CSV."""
    print(f"\n{'='*60}")
    print(f"📥 Downloading {symbol} | {timeframe} | {years} years")
    print(f"{'='*60}")

    # Date range
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=years * 365)
    since_ms = int(since.timestamp() * 1000)
    until_ms = int(now.timestamp() * 1000)

    print(f"  From: {since.strftime('%Y-%m-%d')} → To: {now.strftime('%Y-%m-%d')}")

    # Fetch
    candles = _fetch_ohlcv_chunked(exchange, symbol, timeframe, since_ms, until_ms)

    if not candles:
        print(f"  ❌ No data returned for {symbol}")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")

    # Remove duplicates
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    # Save raw
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{symbol.replace('/', '_')}_{timeframe}.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath)

    print(f"  ✅ Saved {len(df)} candles → {filepath}")
    print(f"  📅 Range: {df.index[0]} → {df.index[-1]}")

    return df


# ---------------------------------------------------------------------------
# Processing pipeline
# ---------------------------------------------------------------------------

def process_and_save(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    config: dict,
    output_dir: str,
    sentiment_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Apply the full feature engineering pipeline:
    1. Technical indicators
    2. Multi-timeframe aggregation
    3. Sentiment integration
    4. Rolling normalization
    5. Train/Val/Test split
    """
    from features.indicators import add_all_indicators
    from features.normalizer import rolling_normalize
    from features.multi_timeframe import add_multi_timeframe_features
    from data.sentiment import add_sentiment_data

    print(f"\n🔧 Processing {symbol} {timeframe}...")

    # 1. Add technical indicators
    df = add_all_indicators(df, config.get("observation", {}))
    print(f"  ✅ Indicators added: {len(df.columns)} columns")

    # 2. Multi-timeframe features
    mtf_config = config.get("observation", {}).get("multi_timeframe", {})
    if mtf_config.get("enabled", False):
        df = add_multi_timeframe_features(df, source_tf=timeframe, config=mtf_config)
        print(f"  ✅ Multi-TF features added: {len(df.columns)} columns")

    # 3. Sentiment Data
    if sentiment_df is not None:
        df = add_sentiment_data(df, sentiment_df)
        print(f"  ✅ Sentiment data merged: {len(df.columns)} columns")

    # 3. Drop NaN rows (from indicator warm-up)
    rows_before = len(df)
    df = df.dropna()
    print(f"  🧹 Dropped {rows_before - len(df)} NaN rows (indicator warm-up)")

    # 4. Rolling normalization
    # Preserve raw columns for env simulation
    raw_cols = df[["open", "high", "low", "close"]].copy()

    lookback = config.get("data", {}).get("lookback_window", 30)
    df_normalized = rolling_normalize(df, window=lookback)
    
    # Re-attach raw columns with 'raw_' prefix
    for col in raw_cols.columns:
        df_normalized[f"raw_{col}"] = raw_cols[col]

    print(f"  ✅ Rolling z-score normalization applied (window={lookback})")

    # 5. Save processed data (before normalization for flexibility)
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{symbol.replace('/', '_')}_{timeframe}_processed.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath)
    print(f"  💾 Processed data saved → {filepath}")

    # 6. Save normalized data
    filename_norm = f"{symbol.replace('/', '_')}_{timeframe}_normalized.csv"
    filepath_norm = os.path.join(output_dir, filename_norm)
    df_normalized.to_csv(filepath_norm)
    print(f"  💾 Normalized data saved → {filepath_norm}")

    # 7. Train / Val / Test split info
    data_cfg = config.get("data", {})
    n = len(df)
    train_end = int(n * data_cfg.get("train_ratio", 0.70))
    val_end = train_end + int(n * data_cfg.get("val_ratio", 0.15))

    print(f"\n  📊 Split (chronologique) :")
    print(f"     Train : {df.index[0]} → {df.index[train_end - 1]} ({train_end} rows)")
    print(f"     Val   : {df.index[train_end]} → {df.index[val_end - 1]} ({val_end - train_end} rows)")
    print(f"     Test  : {df.index[val_end]} → {df.index[-1]} ({n - val_end} rows)")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download & process OHLCV data")
    parser.add_argument("--pair", type=str, default=None, help="Single pair (e.g. BTC/USDT)")
    parser.add_argument("--timeframe", type=str, default=None, help="Timeframe (e.g. 1h)")
    parser.add_argument("--years", type=int, default=None, help="Years of history")
    parser.add_argument("--skip-process", action="store_true", help="Skip processing pipeline")
    args = parser.parse_args()

    config = load_config()
    market = config["market"]

    exchange_id = market["exchange"]
    timeframe = args.timeframe or market["timeframe"]
    years = args.years or market["history_years"]
    pairs = [args.pair] if args.pair else market["pairs"]

    raw_dir = os.path.join(ROOT_DIR, config["paths"]["raw_data"])
    processed_dir = os.path.join(ROOT_DIR, config["paths"]["processed_data"])

    # Fetch Sentiment Data (Global)
    sentiment_df = None
    if config.get("observation", {}).get("sentiment", {}).get("fear_and_greed", False):
        print("🌍 Fetching Global Sentiment Data (Fear & Greed)...")
        from data.sentiment import fetch_fear_and_greed_history
        sentiment_df = fetch_fear_and_greed_history()

    # Init exchange
    exchange = _init_exchange(exchange_id)
    print(f"✅ Connected to {exchange_id.upper()}")

    # Download each pair
    for symbol in pairs:
        df = download_pair(exchange, symbol, timeframe, years, raw_dir)

        if not args.skip_process and not df.empty:
            process_and_save(df, symbol, timeframe, config, processed_dir, sentiment_df)

    print(f"\n{'='*60}")
    print("🎉 All downloads complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
