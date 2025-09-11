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

import config

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")

# Tracks whether the first group data load has occurred so we can automatically
# show verbose output once.
_first_group_load = True


def _validate_volume(df: pd.DataFrame) -> bool:
    """Ensure a valid ``Volume`` column exists and is numeric."""
    if "Volume" not in df.columns:
        return False
    vol = df["Volume"]
    if not pd.api.types.is_numeric_dtype(vol):
        raise KeyError("Volume column must be numeric")
    if (vol < 0).any():
        raise KeyError("Volume column contains negative values")
    return True


def _normalize_ticker(ticker: str) -> str:
    """Normalize ticker symbols based on the configured data source.

    When using Binance the framework operates on ``USDT``-settled pairs.  The
    configuration defines tickers with a ``-USD`` suffix for generality, so we
    convert those to ``USDT`` and drop any dashes before requesting data.  For
    other data sources the ticker is returned unchanged.
    """

    if getattr(config, "DATA_SOURCE", "").lower() == "binance":
        ticker = ticker.replace("-", "")
        if ticker.endswith("USD") and not ticker.endswith("USDT"):
            ticker = ticker[:-3] + "USDT"
    return ticker


def _get_binance_data(
    ticker: str, start_date: str, end_date: str, interval: str, *, verbose: bool = True
) -> pd.DataFrame:
    """Fetch historical kline data from Binance and format it.

    The real ``binance`` package is an optional dependency and importing it
    at module import time causes deprecation warnings in the test
    environment. We therefore import the client lazily inside this
    function.
    """
    if verbose:
        print(
            f"Loading '{ticker}' data from Binance.{config.API_KEYS['binance']['tld']} API..."
        )

    from binance.client import Client  # type: ignore

    # --- MODIFIED: Added the tld parameter to correctly connect to Binance.US ---
    client = Client(
        api_key=config.API_KEYS["binance"]["api_key"],
        api_secret=config.API_KEYS["binance"]["api_secret"],
        tld=config.API_KEYS["binance"]["tld"],
    )

    # Fetch the data
    klines = client.get_historical_klines(
        ticker.replace("-", ""), interval, start_str=start_date, end_str=end_date
    )

    if not klines:
        if verbose:
            print(
                "No data returned from Binance for "
                f"{ticker}. It may not be listed on Binance.US or have history in this range."
            )
        return pd.DataFrame()

    # Create a pandas DataFrame
    data = pd.DataFrame(
        klines,
        columns=[
            "Open time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Close time",
            "Quote asset volume",
            "Number of trades",
            "Taker buy base asset volume",
            "Taker buy quote asset volume",
            "Ignore",
        ],
    )

    # --- Data Cleaning and Formatting ---
    data["Date"] = pd.to_datetime(data["Open time"], unit="ms")
    data.set_index("Date", inplace=True)
    data = data[["Open", "High", "Low", "Close", "Volume"]]
    data = data.apply(pd.to_numeric, errors="coerce")

    if verbose:
        print("Binance data loaded and formatted successfully.")
    return data


def get_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    *,
    verbose: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Fetch OHLCV data for a single ticker.

    Parameters
    ----------
    ticker : str
        Asset symbol to download.
    start_date, end_date : str
        Date range for the request.
    interval : str
        Sampling interval, e.g. ``'1d'``.
    verbose : bool, keyword-only
        When ``True`` (default) prints status messages; otherwise silent.

    Returns
    -------
    tuple[pd.DataFrame, str]
        The data frame and a marker indicating ``'cache'`` or ``'API'`` for the
        data source used.
    """
    source = config.DATA_SOURCE.lower()
    ticker = _normalize_ticker(ticker)

    # Include the source in the cache filename to prevent conflicts
    cache_filename = f"{ticker}_{source}_{start_date}_{end_date}_{interval}.csv"
    cache_filepath = os.path.join(CACHE_DIR, cache_filename)

    if os.path.exists(cache_filepath):
        if verbose:
            print(f"Loading '{ticker}' data from local cache: {cache_filename}")
        try:
            data = pd.read_csv(cache_filepath, index_col=0, parse_dates=True)
            if not isinstance(data.index, pd.DatetimeIndex):
                raise TypeError("Loaded data index is not a DatetimeIndex.")
            if not _validate_volume(data):
                try:
                    from strategy_engine import VOLUME_INDICATORS  # noqa: WPS433
                except Exception:
                    VOLUME_INDICATORS = set()
                if VOLUME_INDICATORS and verbose:
                    print(
                        "Warning: Volume column missing; volume-based indicators will fail."
                    )
            if verbose:
                print("Cache loaded successfully.")
            return data, "cache"
        except KeyError:
            raise
        except Exception as e:
            if verbose:
                print(
                    f"Error loading from cache file {cache_filepath}: {e}. Re-downloading."
                )

    # --- Router Logic ---
    try:
        if source == "binance":
            data = _get_binance_data(
                ticker, start_date, end_date, interval, verbose=verbose
            )
        elif source == "yfinance":
            if verbose:
                print(
                    f"Cache not found. Downloading '{ticker}' data from Yahoo Finance..."
                )
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
            if verbose:
                print(
                    f"No data returned for ticker '{ticker}' from source '{source}'. Check parameters."
                )
            return pd.DataFrame(), "API"

        # Standardize column names without mangling "Adj Close"
        rename = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adj close": "Adj Close",
            "volume": "Volume",
        }
        data.columns = [rename.get(str(col).lower(), col) for col in data.columns]

        if not _validate_volume(data):
            try:
                from strategy_engine import VOLUME_INDICATORS  # noqa: WPS433
            except Exception:
                VOLUME_INDICATORS = set()
            if VOLUME_INDICATORS and verbose:
                print(
                    "Warning: Volume column missing; volume-based indicators will fail."
                )

        # Save the newly fetched data to cache
        os.makedirs(CACHE_DIR, exist_ok=True)
        data.to_csv(cache_filepath)
        if verbose:
            print(f"Saved data to cache: {cache_filename}")

        return data, "API"

    except KeyError:
        raise
    except Exception as e:
        if verbose:
            print(f"An error occurred while downloading data for {ticker}: {e}")
        return pd.DataFrame(), "API"


def get_group_data(
    asset_group,
    start_date,
    end_date,
    interval,
    coverage_threshold=None,
    *,
    verbose: bool = False,
):
    """Load and align OHLCV data for a group of assets.

    Each asset is downloaded via :func:`get_data`. The resulting dataframes are
    aligned to the **intersection** of their timestamps to ensure fairness when
    comparing strategies across assets. Assets that lack sufficient coverage of
    the overall date range are discarded. Coverage is measured as the fraction
    of bars present for an asset relative to the union of timestamps across all
    assets. Assets with coverage below ``coverage_threshold`` are excluded.

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
        ``0.8`` (80%).

    Returns
    -------
    dict
        Mapping of ticker -> aligned OHLCV dataframe.  If no assets pass the
        coverage filter an empty dict is returned.
    """

    if coverage_threshold is None:
        coverage_threshold = getattr(config, "COVERAGE_THRESHOLD", 0.8)

    global _first_group_load
    asset_verbose = verbose or _first_group_load

    raw_data = {}
    sources = {}
    for _, ticker in asset_group:
        norm = _normalize_ticker(ticker)
        df, src = get_data(
            ticker=norm,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            verbose=asset_verbose,
        )
        if not df.empty:
            raw_data[norm] = df
            sources[norm] = src

    if not raw_data:
        if not verbose:
            print(f"Loaded 0 assets from {start_date} to {end_date} [cache:0, API:0]")
        if asset_verbose:
            _first_group_load = False
        return {}

    # Determine the union of timestamps across all assets to evaluate coverage.
    union_index = None
    for df in raw_data.values():
        union_index = df.index if union_index is None else union_index.union(df.index)
    union_len = len(union_index)

    # Filter out assets with insufficient data coverage.
    filtered = {}
    for ticker, df in raw_data.items():
        coverage = len(df.index) / union_len
        if coverage >= coverage_threshold:
            filtered[ticker] = df
        else:
            if asset_verbose:
                print(
                    f"Excluding {ticker} due to insufficient coverage ({coverage:.0%})"
                )

    if not filtered:
        if not verbose:
            print(f"Loaded 0 assets from {start_date} to {end_date} [cache:0, API:0]")
        if asset_verbose:
            _first_group_load = False
        return {}

    # Align remaining assets to the intersection of their timestamps.
    common_index = None
    for df in filtered.values():
        common_index = (
            df.index if common_index is None else common_index.intersection(df.index)
        )

    aligned = {ticker: df.loc[common_index] for ticker, df in filtered.items()}

    if not verbose:
        used_sources = {t: sources[t] for t in aligned.keys()}
        cache_count = sum(1 for s in used_sources.values() if s == "cache")
        api_count = sum(1 for s in used_sources.values() if s == "API")
        print(
            f"Loaded {len(aligned)} assets from {start_date} to {end_date} "
            f"[cache:{cache_count}, API:{api_count}]"
        )

    if asset_verbose:
        _first_group_load = False

    return aligned
