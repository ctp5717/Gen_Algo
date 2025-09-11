# data_loader.py

"""Data loading and caching utilities."""

import logging
import os

import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")


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
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str,
    *,
    verbose: bool = True,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Fetch historical kline data from Binance and format it."""
    logger = logger or logging.getLogger(__name__)

    if verbose:
        logger.info(
            "Loading '%s' data from Binance.%s API...",
            ticker,
            config.API_KEYS["binance"]["tld"],
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
            logger.warning(
                "No data returned from Binance for %s. It may not be listed on Binance.US or have history in this range.",
                ticker,
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
        logger.info("Binance data loaded and formatted successfully.")
    return data


def get_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    *,
    verbose: bool = True,
    logger: logging.Logger | None = None,
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
    logger = logger or logging.getLogger(__name__)

    source = config.DATA_SOURCE.lower()
    ticker = _normalize_ticker(ticker)

    # Include the source in the cache filename to prevent conflicts
    cache_filename = f"{ticker}_{source}_{start_date}_{end_date}_{interval}.csv"
    cache_filepath = os.path.join(CACHE_DIR, cache_filename)

    if os.path.exists(cache_filepath):
        if verbose:
            logger.info(
                "Loading '%s' data from local cache: %s", ticker, cache_filename
            )
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
                    logger.warning(
                        "Volume column missing; volume-based indicators will fail."
                    )
            if verbose:
                logger.info("Cache loaded successfully.")
            return data, "cache"
        except KeyError:
            raise
        except Exception as e:
            if verbose:
                logger.warning(
                    "Error loading from cache file %s: %s. Re-downloading.",
                    cache_filepath,
                    e,
                )

    # --- Router Logic ---
    try:
        if source == "binance":
            data = _get_binance_data(
                ticker,
                start_date,
                end_date,
                interval,
                verbose=verbose,
                logger=logger,
            )
        elif source == "yfinance":
            if verbose:
                logger.info(
                    "Cache not found. Downloading '%s' data from Yahoo Finance...",
                    ticker,
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
                logger.warning(
                    "No data returned for ticker '%s' from source '%s'. Check parameters.",
                    ticker,
                    source,
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
                logger.warning(
                    "Volume column missing; volume-based indicators will fail."
                )

        # Save the newly fetched data to cache
        os.makedirs(CACHE_DIR, exist_ok=True)
        data.to_csv(cache_filepath)
        if verbose:
            logger.info("Saved data to cache: %s", cache_filename)

        return data, "API"

    except KeyError:
        raise
    except Exception as e:
        if verbose:
            logger.warning(
                "An error occurred while downloading data for %s: %s", ticker, e
            )
        return pd.DataFrame(), "API"


def get_group_data(
    asset_group,
    start_date,
    end_date,
    interval,
    coverage_threshold=None,
    *,
    verbose: bool = False,
    logger: logging.Logger | None = None,
):
    """Load and align OHLCV data for a group of assets."""

    logger = logger or logging.getLogger(__name__)

    if coverage_threshold is None:
        coverage_threshold = getattr(config, "COVERAGE_THRESHOLD", 0.8)

    asset_verbose = verbose

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
            logger=logger,
        )
        if not df.empty:
            raw_data[norm] = df
            sources[norm] = src

    if not raw_data:
        logger.info(
            "Loaded 0 assets from %s to %s [cache:0, API:0]", start_date, end_date
        )
        return {}

    union_index = None
    for df in raw_data.values():
        union_index = df.index if union_index is None else union_index.union(df.index)
    union_len = len(union_index)

    filtered = {}
    for ticker, df in raw_data.items():
        coverage = len(df.index) / union_len
        if coverage >= coverage_threshold:
            filtered[ticker] = df
        else:
            if asset_verbose:
                logger.info(
                    "Excluding %s due to insufficient coverage (%.0f%%)",
                    ticker,
                    coverage * 100,
                )

    if not filtered:
        logger.info(
            "Loaded 0 assets from %s to %s [cache:0, API:0]", start_date, end_date
        )
        return {}

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
        logger.info(
            "Loaded %d assets from %s to %s [cache:%d, API:%d]",
            len(aligned),
            start_date,
            end_date,
            cache_count,
            api_count,
        )

    return aligned
