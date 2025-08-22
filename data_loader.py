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

# Track how many times group data has been requested so we can adjust
# verbosity on subsequent calls.
_group_load_count = 0

# Keep track of assets excluded during the most recent ``get_group_data``
# invocation so other modules (e.g. walk-forward diagnostics) can report them
# in their summaries and persisted ``details.json`` files.
_last_excluded_assets: list[dict] = []


def _normalize_ticker(ticker: str, source: str) -> str:
    """Normalize ticker symbols for different data sources.

    For Binance, tickers may be provided in formats like ``BTC-USD`` or
    ``BTCUSD``. Binance expects the dash to be removed and uses ``USDT`` as the
    USD trading pair. This helper converts tickers accordingly so that cache
    filenames and API calls use a consistent symbol.

    Parameters
    ----------
    ticker : str
        The original ticker symbol.
    source : str
        Lowercase name of the data source (e.g. ``"binance"``).

    Returns
    -------
    str
        Normalized ticker symbol appropriate for the requested source.
    """

    if source == 'binance':
        # Remove any dashes and ensure USD pairs use USDT unless already USDT
        ticker = ticker.replace('-', '')
        if ticker.endswith('USD') and not ticker.endswith('USDT'):
            ticker = ticker[:-3] + 'USDT'
    return ticker

def _get_binance_data(ticker: str, start_date: str, end_date: str, interval: str, *, verbose: bool = True) -> pd.DataFrame:
    """Fetch historical kline data from Binance and format it.

    Parameters
    ----------
    ticker : str
        Asset symbol to download.
    start_date, end_date : str
        Date boundaries for the request.
    interval : str
        Candle interval to request from Binance.
    verbose : bool, optional
        If ``True`` (default) print progress messages; otherwise remain
        silent.
    """
    if verbose:
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
        if verbose:
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

    if verbose:
        print("Binance data loaded and formatted successfully.")
    return data

def get_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = '1d',
    *,
    verbose: bool = True,
    return_source: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, str]:
    """Acts as a router to fetch data from the selected source.

    Parameters
    ----------
    ticker : str
        Symbol to download.
    start_date, end_date : str
        Date range for the request.
    interval : str, optional
        Candle interval, default ``'1d'``.
    verbose : bool, optional
        If ``True`` print progress messages.
    return_source : bool, optional
        When ``True`` also return the data source used (``"cache"`` or
        ``"api"``).
    """

    source = config.DATA_SOURCE.lower()
    normalized_ticker = _normalize_ticker(ticker, source)

    cache_filename = f"{normalized_ticker}_{source}_{start_date}_{end_date}_{interval}.csv"
    cache_filepath = os.path.join(CACHE_DIR, cache_filename)

    source_used = 'api'

    if os.path.exists(cache_filepath):
        try:
            data = pd.read_csv(cache_filepath, index_col=0, parse_dates=True)
            if not isinstance(data.index, pd.DatetimeIndex):
                raise TypeError("Loaded data index is not a DatetimeIndex.")
            start_ts = pd.to_datetime(start_date)
            end_ts = pd.to_datetime(end_date)
            if interval.endswith("d"):
                tol = pd.Timedelta("1D")
            else:
                tol = pd.to_timedelta(interval)
            if data.index.min() <= start_ts and data.index.max() >= end_ts - tol:
                source_used = 'cache'
                if verbose:
                    print(f"Loading '{normalized_ticker}' data from cache: {cache_filepath}")
                return (data, source_used) if return_source else data
            else:
                if verbose:
                    print("Cache exists but does not cover requested range. Re-downloading.")
        except Exception as e:
            if verbose:
                print(f"Error loading from cache file {cache_filepath}: {e}. Re-downloading.")

    # --- Router Logic ---
    try:
        if source == 'binance':
            data = _get_binance_data(
                normalized_ticker,
                start_date,
                end_date,
                interval,
                verbose=verbose,
            )
        elif source == 'yfinance':
            if verbose:
                print(
                    f"Cache not found. Downloading '{ticker}' data from Yahoo Finance..."
                )
            data = yf.download(
                normalized_ticker,
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
            if verbose:
                print(
                    f"No data returned for ticker '{ticker}' from source '{source}'. Check parameters."
                )
            return (pd.DataFrame(), source_used) if return_source else pd.DataFrame()

        data.columns = [col.capitalize() for col in data.columns]

        os.makedirs(CACHE_DIR, exist_ok=True)
        data.to_csv(cache_filepath)
        if verbose:
            print(
                f"Downloaded '{normalized_ticker}' from API; saved cache to {cache_filepath}"
            )

        return (data, source_used) if return_source else data

    except Exception as e:
        if verbose:
            print(f"An error occurred while downloading data for {ticker}: {e}")
        return (pd.DataFrame(), source_used) if return_source else pd.DataFrame()


def get_group_data(asset_group, start_date, end_date, interval, coverage_threshold=None):
    """Load and align OHLCV data for a group of assets.

    The first time this function is called it prints per‑asset information and
    then remains quiet on subsequent calls, emitting a concise summary instead.

    Parameters
    ----------
    asset_group : iterable
        Iterable of ``(name, ticker)`` pairs describing the assets.
    start_date, end_date : str
        Date boundaries to pass to :func:`get_data`.
    interval : str
        Timeframe to request from the data source.
    coverage_threshold : float, optional
        Minimum fraction of bars required for an asset to be kept. Default is
        ``0.85`` (85%).

    Returns
    -------
    dict
        Mapping of ticker -> aligned OHLCV dataframe.  If no assets pass the
        coverage filter an empty dict is returned.
    """

    global _group_load_count, _last_excluded_assets
    _group_load_count += 1
    first_call = _group_load_count == 1
    # Reset record of excluded assets each time the loader is invoked
    _last_excluded_assets = []

    if coverage_threshold is None:
        coverage_threshold = getattr(config, "COVERAGE_THRESHOLD", 0.85)

    raw_data = {}
    sources = []
    for _, ticker in asset_group:
        df, src = get_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            verbose=first_call,
            return_source=True,
        )
        if not df.empty:
            raw_data[ticker] = df
            sources.append(src)

    if not raw_data:
        return {}

    # Determine the union of timestamps across all assets to evaluate coverage.
    union_index = None
    for df in raw_data.values():
        union_index = df.index if union_index is None else union_index.union(df.index)

    # Filter out assets with insufficient data coverage.
    filtered = {}
    possible_bars = len(union_index)
    for ticker, df in raw_data.items():
        bars_present = len(df.index)
        coverage = bars_present / possible_bars
        if first_call:
            print(f"{ticker}: {bars_present}/{possible_bars} bars ({coverage:.0%} coverage)")
        if coverage >= coverage_threshold:
            filtered[ticker] = df
        else:
            pct = round(coverage * 100, 2)
            _last_excluded_assets.append({
                "ticker": ticker,
                "coverage": pct,
                "reason": "low_coverage",
            })
            print(f"Excluded: {ticker} ({pct:.0f}%)")
            if first_call:
                print(
                    f"WARNING: {ticker} below coverage threshold ({coverage_threshold:.0%})"
                )

    if not filtered:
        return {}

    # Align remaining assets to the intersection of their timestamps.
    common_index = None
    for df in filtered.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)

    aligned = {ticker: df.loc[common_index] for ticker, df in filtered.items()}

    if not first_call and sources:
        unique_sources = set(sources)
        source_str = unique_sources.pop() if len(unique_sources) == 1 else 'mixed'
        print(
            f"Loading asset data for {len(asset_group)} assets ({start_date}–{end_date}) from {source_str}"
        )

    return aligned


def get_last_excluded_assets() -> list[dict]:
    """Return assets excluded during the last :func:`get_group_data` call."""

    return list(_last_excluded_assets)

