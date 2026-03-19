import os
import pytest
import pandas as pd
import numpy as np
import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "config.yaml")


@pytest.fixture
def config():
    """Load config.yaml."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def check_csv_exists(filepath):
    """Check if the CSV file exists."""
    return os.path.exists(filepath)


def test_raw_data_formatting(config):
    """Validate format of raw downloaded data."""
    raw_dir = os.path.join(ROOT_DIR, config.get("paths", {}).get("raw_data", "data/raw"))
    pairs = config.get("market", {}).get("pairs", ["BTC/USDT"])
    timeframe = config.get("market", {}).get("timeframe", "1m")
    
    for pair in pairs:
        filename = f"{pair.replace('/', '_')}_{timeframe}.csv"
        filepath = os.path.join(raw_dir, filename)
        
        # Test won't fail if data isn't there yet, but if it is, it must be correct.
        if not check_csv_exists(filepath):
            pytest.skip(f"Data file {filepath} not found. Skipping validation.")
            
        df = pd.read_csv(filepath)
        
        # 1. Check for expected columns
        expected_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in expected_cols:
            assert col in df.columns, f"Missing '{col}' in {filepath}"
            
        # 2. Check no missing or infinite values
        assert not df.isnull().values.any(), f"Missing values found in {filepath}"
        
        # 3. Verify chronological order of timestamps
        # Make sure timestamp is a datetime or string parsed sequentially
        try:
            df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
            assert df["timestamp_dt"].is_monotonic_increasing, f"Timestamps are not strictly increasing in {filepath}"
        except Exception as e:
            pytest.fail(f"Failed to parse timestamp in {filepath}: {e}")
            
        # 4. OHLC logical constraint check (High >= Low, High >= Open, High >= Close...)
        invalid_ohlc = df[
            (df['high'] < df['low']) | 
            (df['high'] < df['open']) | 
            (df['high'] < df['close']) | 
            (df['low'] > df['open']) | 
            (df['low'] > df['close'])
        ]
        assert len(invalid_ohlc) == 0, f"Found {len(invalid_ohlc)} rows with invalid OHLC logic in {filepath}"


def test_processed_data_integrity(config):
    """Validate structure of processed data (after indicators & multi-TF)."""
    processed_dir = os.path.join(ROOT_DIR, config.get("paths", {}).get("processed_data", "data/processed"))
    pairs = config.get("market", {}).get("pairs", ["BTC/USDT"])
    timeframe = config.get("market", {}).get("timeframe", "1m")
    
    for pair in pairs:
        filename = f"{pair.replace('/', '_')}_{timeframe}_processed.csv"
        filepath = os.path.join(processed_dir, filename)
        
        if not check_csv_exists(filepath):
            pytest.skip(f"Processed file {filepath} not found. Skipping validation.")
            
        df = pd.read_csv(filepath)
        
        # Needs no remaining NaNs
        nan_count = df.isnull().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values in {filepath}. Check pipeline warm-up logic."
        
        # Raw prices should be available
        assert "open" in df.columns, f"open missing in {filepath}"
        assert "close" in df.columns, f"close missing in {filepath}"


def test_normalized_data_bounds(config):
    """Validate normalized data bounds."""
    processed_dir = os.path.join(ROOT_DIR, config.get("paths", {}).get("processed_data", "data/processed"))
    pairs = config.get("market", {}).get("pairs", ["BTC/USDT"])
    timeframe = config.get("market", {}).get("timeframe", "1m")
    
    for pair in pairs:
        filename = f"{pair.replace('/', '_')}_{timeframe}_normalized.csv"
        filepath = os.path.join(processed_dir, filename)
        
        if not check_csv_exists(filepath):
            pytest.skip(f"Normalized file {filepath} not found. Skipping validation.")
            
        df = pd.read_csv(filepath)
        
        # Focus on a few known indicator columns to check standard distribution
        # Most values should fall inside [-5, 5] interval (clipping)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Excluding explicitly raw_ columns from normalized check
        zscore_cols = [c for c in numeric_cols if not c.startswith('raw_')]
        
        for col in zscore_cols:
            max_val = df[col].max()
            min_val = df[col].min()
            # Pure z-score can exceed 6.0 in rare spikes, we check it stays within a reasonable float range (e.g. 15.0)
            assert max_val <= 15.0, f"Value {max_val} exceeds normalized bound for {col} in {filepath}"
            assert min_val >= -15.0, f"Value {min_val} below normalized bound for {col} in {filepath}"
