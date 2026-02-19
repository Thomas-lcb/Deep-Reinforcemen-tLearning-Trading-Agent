
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone

def fetch_fear_and_greed_history(limit=0):
    """
    Fetch historical Fear & Greed Index data from alternative.me API.
    limit=0 means fetch all available history.
    """
    url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data['metadata']['error'] is not None:
             print(f"Error fetching sentiment: {data['metadata']['error']}")
             return None

        # Convert to DataFrame
        records = []
        for item in data['data']:
            records.append({
                'timestamp': int(item['timestamp']),
                'value': int(item['value']),
                'value_classification': item['value_classification']
            })
        
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        # Set index to date for easy joining
        df = df.set_index('date').sort_index()
        
        # We only have daily data. Resample to fill hourly if needed, 
        # but for now just returning the daily DF.
        # The calling function (download.py or env) should handle the merge (ffill).
        
        return df[['value']] # value is 0-100

    except Exception as e:
        print(f"Failed to fetch Fear & Greed Index: {e}")
        return None

def add_sentiment_data(ohlcv_df, sentiment_df):
    """
    Merges daily sentiment data into hourly OHLCV data using forward fill.
    """
    if sentiment_df is None or sentiment_df.empty:
        print("Warning: No sentiment data to merge. Filling with neutral (50).")
        ohlcv_df['sentiment'] = 50
        return ohlcv_df

    # Ensure indexes are timezone-aware UTC and have same precision (ns)
    if ohlcv_df.index.tz is None:
        ohlcv_df.index = ohlcv_df.index.tz_localize('UTC')
    
    ohlcv_df.index = ohlcv_df.index.astype('datetime64[ns, UTC]')
    sentiment_df.index = sentiment_df.index.astype('datetime64[ns, UTC]')
    
    # Merge using merge_asof logic (or simple reindex + ffill)
    # Since sentiment is daily, we want to forward fill it to all hours of that day
    
    # First, reindex sentiment to hourly? No, merge_asof is better strictly.
    # But pandas merge_asof needs sorted.
    
    ohlcv_df = ohlcv_df.sort_index()
    sentiment_df = sentiment_df.sort_index()
    
    combined = pd.merge_asof(
        ohlcv_df,
        sentiment_df,
        left_index=True,
        right_index=True,
        direction='backward', # Use last available sentiment
        tolerance=pd.Timedelta(days=2) # Don't use too old data
    )
    
    combined = combined.rename(columns={'value': 'sentiment'})
    
    # Fill remaining NaNs (e.g. before history starts) with neutral 50
    combined['sentiment'] = combined['sentiment'].fillna(50).astype(float)
    
    # Normalize sentiment to [-1, 1] range commonly used in RL? 
    # Or keep 0-100? Let's keep 0-100 for now, normalizer will handle it if configured.
    # Actually, RL agents prefer normalized inputs.
    # But our normalizer (rolling z-score) will handle it automatically if we include it in columns.
    
    return combined

if __name__ == "__main__":
    print("Fetching Fear & Greed Index...")
    df = fetch_fear_and_greed_history()
    if df is not None:
        print(df.tail())
        print(f"Fetched {len(df)} days of sentiment data.")
    else:
        print("Failed.")
