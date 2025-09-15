# data_loader.py

"""Data loading and caching utilities."""

import functools
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")
CACHE_EXTENSION = ".parquet"
LEGACY_CACHE_EXTENSION = ".csv"

_BINANCE_CLIENT: Any | None = None
_BINANCE_CLIENT_LOCK = threading.Lock()


def set_binance_client(client: Any | None) -> None:
    """Override the cached Binance client (mainly for testing)."""

    global _BINANCE_CLIENT
    with _BINANCE_CLIENT_LOCK:
        _BINANCE_CLIENT = client


def _get_cached_binance_client() -> Any:
    """Return a cached Binance client instance, creating it if necessary."""

    global _BINANCE_CLIENT
    with _BINANCE_CLIENT_LOCK:
        if _BINANCE_CLIENT is not None:
            return _BINANCE_CLIENT
        from binance.client import Client  # type: ignore

        _BINANCE_CLIENT = Client(
            api_key=config.API_KEYS["binance"]["api_key"],
            api_secret=config.API_KEYS["binance"]["api_secret"],
            tld=config.BINANCE_TLD,
        )
        return _BINANCE_CLIENT


def _load_legacy_cache(
    legacy_path: str,
    *,
    ticker: str,
    cache_filepath: str,
    cache_filename: str,
    verbose: bool,
    logger: logging.Logger,
    migrate: bool = False,
) -> pd.DataFrame | None:
    """Attempt to load a legacy CSV cache and migrate it to Parquet."""

    if verbose:
        logger.info("Loading '%s' data from legacy cache: %s", ticker, legacy_path)
    try:
        data = pd.read_csv(legacy_path, index_col=0, parse_dates=True)
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
        if migrate:
            os.makedirs(CACHE_DIR, exist_ok=True)
            data.to_parquet(cache_filepath)
            if verbose:
                logger.info("Migrated cache to Parquet: %s", cache_filename)
        return data
    except KeyError:
        raise
    except Exception as exc:
        if verbose:
            logger.warning(
                "Error loading from legacy cache file %s: %s. Re-downloading.",
                legacy_path,
                exc,
            )
        return None


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
    client: Any | None = None,
) -> pd.DataFrame:
    """Fetch historical kline data from Binance and format it."""
    logger = logger or logging.getLogger(__name__)

    if verbose:
        logger.info(
            "Loading '%s' data from Binance.%s API...",
            ticker,
            config.BINANCE_TLD,
        )

    client = client or _get_cached_binance_client()

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
    cache_stem = f"{ticker}_{source}_{start_date}_{end_date}_{interval}"
    cache_filename = f"{cache_stem}{CACHE_EXTENSION}"
    cache_filepath = os.path.join(CACHE_DIR, cache_filename)
    legacy_filepath = os.path.join(
        CACHE_DIR, f"{cache_stem}{LEGACY_CACHE_EXTENSION}"
    )

    legacy_data: pd.DataFrame | None = None

    if os.path.exists(cache_filepath):
        if verbose:
            logger.info(
                "Loading '%s' data from local cache: %s", ticker, cache_filename
            )
        try:
            data = pd.read_parquet(cache_filepath)
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
            legacy_data = _load_legacy_cache(
                legacy_filepath,
                ticker=ticker,
                cache_filepath=cache_filepath,
                cache_filename=cache_filename,
                verbose=verbose,
                logger=logger,
                migrate=getattr(config, "MIGRATE_CACHE_TO_PARQUET", False),
            )
            if legacy_data is not None:
                return legacy_data, "cache"

    if legacy_data is None and os.path.exists(legacy_filepath):
        legacy_data = _load_legacy_cache(
            legacy_filepath,
            ticker=ticker,
            cache_filepath=cache_filepath,
            cache_filename=cache_filename,
            verbose=verbose,
            logger=logger,
            migrate=getattr(config, "MIGRATE_CACHE_TO_PARQUET", False),
        )
        if legacy_data is not None:
            return legacy_data, "cache"

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
        data.to_parquet(cache_filepath)
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

    assets = list(asset_group)
    if not assets:
        return {}

    raw_data: dict[str, pd.DataFrame] = {}
    sources: dict[str, str] = {}
    ordered_norms = [_normalize_ticker(ticker) for _, ticker in assets]

    max_workers = getattr(config, "DATA_LOADER_MAX_WORKERS", None)
    if max_workers is None:
        cpu = os.cpu_count() or 1
        max_workers = min(len(ordered_norms), max(1, cpu * 2))
    else:
        max_workers = max(1, int(max_workers))

    if max_workers <= 1 or len(ordered_norms) <= 1:
        for norm in ordered_norms:
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
        loaded_norms = [n for n in ordered_norms if n in raw_data]
    else:
        futures: dict[Any, str] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for norm in ordered_norms:
                futures[
                    executor.submit(
                        get_data,
                        ticker=norm,
                        start_date=start_date,
                        end_date=end_date,
                        interval=interval,
                        verbose=asset_verbose,
                        logger=logger,
                    )
                ] = norm
            for future in as_completed(futures):
                norm = futures[future]
                try:
                    df, src = future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    if verbose:
                        logger.warning("Error loading %s: %s", norm, exc)
                    continue
                if not df.empty:
                    raw_data[norm] = df
                    sources[norm] = src
        loaded_norms = [n for n in ordered_norms if n in raw_data]

    if not loaded_norms:
        logger.info(
            "Loaded 0 assets from %s to %s [cache:0, API:0]", start_date, end_date
        )
        return {}
    raw_values = [raw_data[norm] for norm in loaded_norms]
    index_lengths = np.fromiter((len(df.index) for df in raw_values), dtype=np.int64)
    if raw_values:
        index_arrays = [df.index.asi8 for df in raw_values if len(df.index) > 0]
        if index_arrays:
            concatenated = np.concatenate(index_arrays)
            union_len = len(np.unique(concatenated))
        else:
            union_len = 0
    else:
        union_len = 0

    if union_len == 0:
        coverages = np.zeros_like(index_lengths, dtype=float)
    else:
        coverages = index_lengths / union_len
    mask = coverages >= coverage_threshold

    filtered = {
        ticker: df for ticker, df, keep in zip(loaded_norms, raw_values, mask) if keep
    }

    if asset_verbose:
        excluded = [
            f"{ticker} ({coverage * 100:.0f}%)"
            for ticker, coverage, keep in zip(loaded_norms, coverages, mask)
            if not keep
        ]
        if excluded:
            logger.info(
                "Excluding tickers due to insufficient coverage: %s",
                ", ".join(excluded),
            )

    if not filtered:
        logger.info(
            "Loaded 0 assets from %s to %s [cache:0, API:0]", start_date, end_date
        )
        return {}

    common_index = functools.reduce(
        pd.Index.intersection, [df.index for df in filtered.values()]
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
