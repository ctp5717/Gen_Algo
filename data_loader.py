# data_loader.py

"""
Data Loader & Caching Module
============================

(This final version flattens MultiIndex columns from yfinance and ensures
the DatetimeIndex is correctly loaded from the cache, which is critical
for the indicator and backtesting engines.)
"""

import os
import pandas as pd
import yfinance as yf
from binance.client import Client
import config

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'data_cache')

def _get_binance_data(ticker: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    """ Fetches historical kline data from Binance and formats it. """
    print(f"Loading '{ticker}' data from Binance.{config.API_KEYS['binance']['tld']} API...")
    
    # --- MODIFIED: Added the tld parameter to correctly connect to Binance.US ---
    client = Client(
        api_key=config.API_KEYS['binance']['api_key'],
        api_secret=config.API_KEYS['binance']['api_secret'],
        tld=config.API_KEYS['binance']['tld']
    )
    
    # Fetch the data
    klines = client.get_historical_klines(ticker.replace('-',''), interval, start_str=start_date, end_str=end_date)
    
    if not klines:
        print(
            "No data returned from Binance for "
            f"{ticker}. It may not be listed on Binance.US or have history in this range."
        )
        return pd.DataFrame()

    # Create a pandas DataFrame
    data = pd.DataFrame(klines, columns=[
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
        'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
        'Taker buy quote asset volume', 'Ignore'
    ])
    
    # --- Data Cleaning and Formatting ---
    data['Date'] = pd.to_datetime(data['Open time'], unit='ms')
    data.set_index('Date', inplace=True)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data = data.apply(pd.to_numeric, errors='coerce')
    
    print("Binance data loaded and formatted successfully.")
    return data

def _load_single_asset_data(ticker: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    """Load data for a single ticker handling caching and routing."""
    source = config.DATA_SOURCE.lower()

    cache_filename = f"{ticker}_{source}_{start_date}_{end_date}_{interval}.csv"
    cache_filepath = os.path.join(CACHE_DIR, cache_filename)

    if os.path.exists(cache_filepath):
        print(f"Loading '{ticker}' data from local cache: {cache_filename}")
        try:
            data = pd.read_csv(cache_filepath, index_col=0, parse_dates=True)
            if not isinstance(data.index, pd.DatetimeIndex):
                raise TypeError("Loaded data index is not a DatetimeIndex.")
            print("Cache loaded successfully.")
            return data
        except Exception as e:
            print(f"Error loading from cache file {cache_filepath}: {e}. Re-downloading.")

    try:
        if source == 'binance':
            data = _get_binance_data(ticker, start_date, end_date, interval)
        elif source == 'yfinance':
            print(f"Cache not found. Downloading '{ticker}' data from Yahoo Finance...")
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
        else:
            raise ValueError(
                f"Unknown data source '{source}' in config file. Use 'yfinance' or 'binance'."
            )

        if data.empty:
            print(
                f"No data returned for ticker '{ticker}' from source '{source}'. Check parameters."
            )
            return pd.DataFrame()

        data.columns = [col.capitalize() for col in data.columns]

        os.makedirs(CACHE_DIR, exist_ok=True)
        data.to_csv(cache_filepath)
        print(f"Saved data to cache: {cache_filename}")

        return data
    except Exception as e:
        print(f"An error occurred while downloading data for {ticker}: {e}")
        return pd.DataFrame()


def get_data(ticker, start_date: str, end_date: str, interval: str = '1d') -> pd.DataFrame:
    """Fetch data for a single ticker or a list of tickers."""
    if isinstance(ticker, (list, tuple)):
        frames = []
        for t in ticker:
            df = _load_single_asset_data(t, start_date, end_date, interval)
            if df.empty:
                continue
            df.columns = pd.MultiIndex.from_product([[t], df.columns])
            frames.append(df)
        return pd.concat(frames, axis=1) if frames else pd.DataFrame()

    return _load_single_asset_data(ticker, start_date, end_date, interval)
