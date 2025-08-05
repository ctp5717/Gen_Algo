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
    """Fetch historical kline data from Binance and return a standardised frame."""
    
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

    return data

def _load_single_asset(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str,
    idx: int = 1,
    total: int = 1,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Load data for one asset with unified progress logging.

    Retries the download a limited number of times and raises an exception if
    all attempts fail so callers can surface the failure instead of silently
    returning an empty frame.
    """

    prefix = f"[{idx}/{total}] "
    source = config.DATA_SOURCE.lower()

    cache_filename = f"{ticker}_{source}_{start_date}_{end_date}_{interval}.csv"
    cache_filepath = os.path.join(CACHE_DIR, cache_filename)

    if os.path.exists(cache_filepath):
        print(f"{prefix}{ticker} from cache")
        try:
            data = pd.read_csv(cache_filepath, index_col=0, parse_dates=True)
            if not isinstance(data.index, pd.DatetimeIndex):
                raise TypeError("Loaded data index is not a DatetimeIndex.")
            return data
        except Exception as e:
            print(f"Error loading from cache file {cache_filepath}: {e}. Re-downloading.")

    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            if source == "binance":
                print(f"{prefix}{ticker} from Binance")
                data = _get_binance_data(ticker, start_date, end_date, interval)
            elif source == "yfinance":
                print(f"{prefix}{ticker} from Yahoo Finance")
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False,
                )
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
            else:
                raise ValueError(
                    f"Unknown data source '{source}' in config file. Use 'yfinance' or 'binance'."
                )

            if data.empty:
                raise ValueError(
                    f"No data returned for ticker '{ticker}' from source '{source}'."
                )

            data.columns = [col.capitalize() for col in data.columns]
            os.makedirs(CACHE_DIR, exist_ok=True)
            data.to_csv(cache_filepath)
            return data
        except Exception as e:
            last_exception = e
            print(f"Attempt {attempt} for {ticker} failed: {e}")

    raise RuntimeError(
        f"Failed to load data for {ticker} after {max_retries} attempts"
    ) from last_exception


def get_data(ticker, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
    """Download historical data for a single ticker or a list of tickers."""

    if isinstance(ticker, (list, tuple)):
        tickers = list(ticker)
        print(f"Assets: {', '.join(tickers)}")
        frames = []
        skipped = []
        for i, tk in enumerate(tickers, 1):
            try:
                df = _load_single_asset(tk, start_date, end_date, interval, i, len(tickers))
            except Exception as e:
                print(f"Skipping {tk}: {e}")
                skipped.append(tk)
                continue
            df.columns = pd.MultiIndex.from_product([[tk], df.columns])
            frames.append(df)
        if skipped:
            print(f"Skipped assets: {', '.join(skipped)}")
        return pd.concat(frames, axis=1) if frames else pd.DataFrame()

    return _load_single_asset(ticker, start_date, end_date, interval)
