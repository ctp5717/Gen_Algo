# indicator_library.py

"""
Indicator Library
=================

This script serves as the central "toolbox" for all technical indicators
used in the trading framework.
Each function in this file is responsible for calculating a specific indicator and returning its
values as a pandas Series or DataFrame.

Design Philosophy:
- Each function is self-contained and only responsible for one indicator.
- We use the high-performance 'pandas-ta' library to ensure calculations are fast and accurate.
- Functions are designed to be called by the 'strategy_engine.py' module.
- The function signatures are standardized for easy integration: they take a pandas DataFrame
  of price data (ohlc) and the indicator's specific parameters as arguments.

To Add a New Indicator:
1. Find the desired indicator in the 'pandas-ta' library documentation or implement your own.
2. Create a new function in this file (e.g., `calculate_macd(...)`).
3. Ensure it takes the ohlc DataFrame and necessary parameters as arguments.
4. Call the 'pandas-ta' function and return the resulting series/dataframe.
5. The new indicator can now be called from the `strategy_engine.py` by referencing
   the function name in the `config.py` file.
"""

import logging
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

import indicator_contracts as contracts

# -- Compatibility shim -------------------------------------------------------
# Some versions of pandas_ta expect ``numpy.NaN`` to be defined, but newer
# numpy releases expose only ``numpy.nan``. Importing pandas_ta without this
# attribute raises ``ImportError: cannot import name 'NaN'``. To keep the
# library working across numpy versions, ensure ``np.NaN`` exists before
# importing pandas_ta.
if not hasattr(np, "NaN"):
    np.NaN = np.nan

import pandas_ta as ta

logger = logging.getLogger(__name__)
logger.info(
    "Indicator backend: pandas-ta %s, pandas %s, numpy %s, TA-Lib %s",
    getattr(ta, "__version__", "unknown"),
    pd.__version__,
    np.__version__,
    "enabled" if getattr(ta, "USE_TALIB", False) else "disabled",
)


@dataclass
class Constraint:
    kind: str
    a: str
    b: str | float

    def enforce(self, params: Dict[str, float]) -> None:
        va = params.get(self.a)
        vb = params.get(self.b) if isinstance(self.b, str) else self.b
        if va is None or vb is None:
            raise ValueError(f"Missing params for constraint {self}")
        if self.kind == "GT" and not va > vb:
            params[self.a] = int(vb) + 1
        elif self.kind == "GE" and not va >= vb:
            params[self.a] = int(vb)
        elif self.kind == "LT" and not va < vb:
            params[self.a] = int(vb) - 1
        elif self.kind == "LE" and not va <= vb:
            params[self.a] = int(vb)


def GT(a: str, b: str | float) -> Constraint:
    return Constraint("GT", a, b)


def GE(a: str, b: str | float) -> Constraint:
    return Constraint("GE", a, b)


def LT(a: str, b: str | float) -> Constraint:
    return Constraint("LT", a, b)


INDICATOR_REGISTRY: Dict[str, Callable] = {}
INDICATOR_CONSTRAINTS: Dict[str, List[Constraint]] = {}


def indicator(name: str, constraints: List[Constraint] | None = None):
    def decorator(func: Callable) -> Callable:
        INDICATOR_REGISTRY[name] = func
        if constraints:
            INDICATOR_CONSTRAINTS[name] = constraints
        return func

    return decorator


def calculate_ema(ohlc_data: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculates the Exponential Moving Average (EMA).

    Args:
        ohlc_data (pd.DataFrame): DataFrame with columns 'Open', 'High', 'Low', 'Close'.
        period (int): The lookback period for the EMA.

    Returns:
        pd.Series: A pandas Series containing the EMA values.
    """
    if period is None:
        raise ValueError("EMA 'period' parameter cannot be None.")

    # pandas-ta automatically finds the 'Close' column to perform the calculation
    ema_series = ohlc_data.ta.ema(length=period)

    if ema_series is None:
        raise ConnectionError(
            "Failed to calculate EMA. Check input data and parameters."
        )

    return ema_series


def calculate_atr(ohlc_data: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculates the Average True Range (ATR).

    Args:
        ohlc_data (pd.DataFrame): DataFrame with columns 'Open', 'High', 'Low', 'Close'.
        period (int): The lookback period for the ATR.

    Returns:
        pd.Series: A pandas Series containing the ATR values.
    """
    if period is None:
        raise ValueError("ATR 'period' parameter cannot be None.")

    # pandas-ta uses the high, low, and close columns for the ATR calculation
    atr_series = ohlc_data.ta.atr(length=period)

    if atr_series is None:
        raise ConnectionError(
            "Failed to calculate ATR. Check input data and parameters."
        )

    return atr_series


def calculate_rsi(ohlc_data: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI).
    """
    if period is None:
        raise ValueError("RSI 'period' parameter cannot be None.")

    rsi_series = ohlc_data.ta.rsi(length=period)

    if rsi_series is None:
        raise ConnectionError(
            "Failed to calculate RSI. Check input data and parameters."
        )

    return rsi_series


@indicator(
    "macd",
    constraints=[GT("slow", "fast"), GE("signal", 1), LT("signal", "slow")],
)
def calculate_macd(
    ohlc_data: pd.DataFrame, fast: int, slow: int, signal: int
) -> pd.DataFrame:
    """
    Calculates the Moving Average Convergence Divergence (MACD).

    Returns:
        pd.DataFrame: A DataFrame containing MACD line, histogram, and signal line.
    """
    params = {"fast": fast, "slow": slow, "signal": signal}
    for name, val in params.items():
        if not isinstance(val, int):
            raise TypeError(f"MACD '{name}' must be int, got {type(val).__name__}")
    if fast < 1 or slow < 2:
        raise ValueError("MACD 'fast' must be ≥1 and 'slow' ≥2.")
    if fast >= slow or signal < 1 or signal >= slow:
        raise ValueError(
            "MACD invalid: fast < slow and 1 ≤ signal < slow "
            f"(got fast={fast}, slow={slow}, signal={signal})"
        )

    # The .ta.macd() function returns a DataFrame with multiple columns
    macd_df = ohlc_data.ta.macd(fast=fast, slow=slow, signal=signal)

    if macd_df is None:
        raise ConnectionError(
            "Failed to calculate MACD. Check input data and parameters."
        )

    return contracts.normalize_output("macd", macd_df, params, index=ohlc_data.index)


def calculate_bbands(
    ohlc_data: pd.DataFrame, period: int, std_dev: float
) -> pd.DataFrame:
    """
    Calculates Bollinger Bands (BBands).

    Returns:
        pd.DataFrame: A DataFrame containing the upper, middle, and lower bands.
    """
    if period is None or std_dev is None:
        raise ValueError("BBands 'period' and 'std_dev' parameters cannot be None.")

    # pandas_ta.bbands uses separate ``lower_std`` and ``upper_std`` parameters
    # (it does not accept ``std``). Passing ``std`` previously resulted in
    # pandas_ta defaulting both values to ``2.0`` which yielded column names
    # like ``BBL_10_2.0_2.0`` regardless of the requested deviation. Our
    # contract expects the standard deviation to be reflected in the column
    # names (e.g. ``BBL_10_0.5_0.5``). Use the correct parameter names so the
    # output aligns with the contract.

    bbands_df = ohlc_data.ta.bbands(length=period, lower_std=std_dev, upper_std=std_dev)

    if bbands_df is None:
        raise ConnectionError(
            "Failed to calculate BBands. Check input data and parameters."
        )

    return contracts.normalize_output(
        "bbands",
        bbands_df,
        {"period": period, "std_dev": std_dev},
        index=ohlc_data.index,
    )


def calculate_sma(ohlc_data: pd.DataFrame, period: int) -> pd.Series:
    """Calculate the Simple Moving Average (SMA)."""
    if not isinstance(period, int):
        raise TypeError("SMA 'period' must be int")
    if period < 1:
        raise ValueError("SMA 'period' must be ≥1")
    sma = ohlc_data.ta.sma(length=period)
    if sma is None:
        raise ConnectionError("Failed to calculate SMA")
    return sma


def calculate_wma(ohlc_data: pd.DataFrame, period: int) -> pd.Series:
    """Calculate the Weighted Moving Average (WMA)."""
    if not isinstance(period, int):
        raise TypeError("WMA 'period' must be int")
    if period < 1:
        raise ValueError("WMA 'period' must be ≥1")
    wma = ohlc_data.ta.wma(length=period)
    if wma is None:
        raise ConnectionError("Failed to calculate WMA")
    return wma


def calculate_hma(ohlc_data: pd.DataFrame, period: int) -> pd.Series:
    """Calculate the Hull Moving Average (HMA)."""
    if not isinstance(period, int):
        raise TypeError("HMA 'period' must be int")
    if period < 1:
        raise ValueError("HMA 'period' must be ≥1")
    hma = ohlc_data.ta.hma(length=period)
    if hma is None:
        raise ConnectionError("Failed to calculate HMA")
    return hma


def calculate_stoch(
    ohlc_data: pd.DataFrame, k: int, d: int, smooth_k: int = 3
) -> pd.DataFrame:
    """Calculate the Stochastic Oscillator (%K and %D)."""
    params = {"k": k, "d": d, "smooth_k": smooth_k}
    for name, val in params.items():
        if not isinstance(val, int):
            raise TypeError(f"Stochastic '{name}' must be int")
        if val < 1:
            raise ValueError(f"Stochastic '{name}' must be ≥1")
    stoch = ohlc_data.ta.stoch(k=k, d=d, smooth_k=smooth_k)
    if stoch is None:
        raise ConnectionError("Failed to calculate Stochastic Oscillator")
    return contracts.normalize_output("stoch", stoch, params, index=ohlc_data.index)


def calculate_cci(ohlc_data: pd.DataFrame, period: int) -> pd.Series:
    """Calculate the Commodity Channel Index (CCI) without pandas-ta overhead."""

    if not isinstance(period, int):
        raise TypeError("CCI 'period' must be int")
    if period < 1:
        raise ValueError("CCI 'period' must be ≥1")

    required = {"High", "Low", "Close"}
    missing = required.difference(ohlc_data.columns)
    if missing:
        raise KeyError(f"CCI calculation requires columns: {sorted(missing)}")

    typical_price = (
        ohlc_data["High"].to_numpy(dtype=float, copy=False)
        + ohlc_data["Low"].to_numpy(dtype=float, copy=False)
        + ohlc_data["Close"].to_numpy(dtype=float, copy=False)
    ) / 3.0

    tp_series = pd.Series(typical_price, index=ohlc_data.index)
    rolling_mean = tp_series.rolling(window=period, min_periods=period).mean()

    if len(typical_price) < period:
        mad_values = np.full_like(typical_price, np.nan, dtype=float)
    else:
        window_view = np.lib.stride_tricks.sliding_window_view(typical_price, period)
        means = window_view.mean(axis=-1, keepdims=True)
        mad_core = np.abs(window_view - means).mean(axis=-1)
        mad_values = np.concatenate(
            (np.full(period - 1, np.nan, dtype=float), mad_core.astype(float))
        )

    mean_dev = pd.Series(mad_values, index=ohlc_data.index)
    denominator = 0.015 * mean_dev.replace(0.0, np.nan)
    cci = (tp_series - rolling_mean) / denominator
    cci.name = f"CCI_{period}"
    return cci


def calculate_williams_r(ohlc_data: pd.DataFrame, period: int) -> pd.Series:
    """Calculate the Williams %R indicator."""
    if not isinstance(period, int):
        raise TypeError("Williams %R 'period' must be int")
    if period < 1:
        raise ValueError("Williams %R 'period' must be ≥1")
    wr = ohlc_data.ta.willr(length=period)
    if wr is None:
        raise ConnectionError("Failed to calculate Williams %R")
    return wr


def calculate_tsi(
    ohlc_data: pd.DataFrame, long: int, short: int, signal: int | None = None
) -> pd.Series | pd.DataFrame:
    """Calculate the True Strength Index (TSI).

    Parameters
    ----------
    long : int
        The long lookback period. Must be ``> short``.
    short : int
        The short lookback period.
    signal : int, optional
        Period for an optional signal line. Must be ``>=1``.
    """
    params = {"long": long, "short": short}
    for name, val in params.items():
        if not isinstance(val, int):
            raise TypeError(f"TSI '{name}' must be int")
        if val < 1:
            raise ValueError(f"TSI '{name}' must be ≥1")
    if long <= short:
        raise ValueError("TSI requires 'long' > 'short'")
    if signal is not None:
        if not isinstance(signal, int):
            raise TypeError("TSI 'signal' must be int")
        if signal < 1:
            raise ValueError("TSI 'signal' must be ≥1")
        tsi = ohlc_data.ta.tsi(long=long, short=short, signal=signal)
    else:
        tsi = ohlc_data.ta.tsi(long=long, short=short)
    if tsi is None:
        raise ConnectionError("Failed to calculate TSI")
    return tsi


def calculate_ultimate_oscillator(
    ohlc_data: pd.DataFrame, short: int, medium: int, long: int
) -> pd.Series:
    """Calculate the Ultimate Oscillator.

    Ensures the standard configuration ``short < medium < long``.
    """
    params = {"short": short, "medium": medium, "long": long}
    for name, val in params.items():
        if not isinstance(val, int):
            raise TypeError(f"Ultimate Oscillator '{name}' must be int")
        if val < 1:
            raise ValueError(f"Ultimate Oscillator '{name}' must be ≥1")
    if not (short < medium < long):
        raise ValueError("Ultimate Oscillator requires short < medium < long")
    uo = ohlc_data.ta.uo(length1=short, length2=medium, length3=long)
    if uo is None:
        raise ConnectionError("Failed to calculate Ultimate Oscillator")
    return uo


def calculate_adx(ohlc_data: pd.DataFrame, period: int) -> pd.DataFrame:
    """Calculate the Average Directional Index (ADX/DMI)."""
    if not isinstance(period, int):
        raise TypeError("ADX 'period' must be int")
    if period < 1:
        raise ValueError("ADX 'period' must be ≥1")
    adx = ohlc_data.ta.adx(length=period)
    if adx is None:
        raise ConnectionError("Failed to calculate ADX")
    return contracts.normalize_output(
        "adx", adx, {"period": period}, index=ohlc_data.index
    )


def calculate_psar(
    ohlc_data: pd.DataFrame, acceleration: float = 0.02, maximum: float = 0.2
) -> pd.DataFrame:
    """Calculate the Parabolic SAR indicator."""
    for name, val in {"acceleration": acceleration, "maximum": maximum}.items():
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise TypeError(f"PSAR '{name}' must be numeric")
        if val <= 0:
            raise ValueError(f"PSAR '{name}' must be > 0")

    # pandas-ta uses the raw float values in column names.  When parameters
    # include floating point representation noise (e.g. ``0.15000000000000002``)
    # the resulting column names would not match our contract formatting.  To
    # keep column names deterministic, quantize inputs to six decimal places
    # before invoking the underlying indicator.
    acceleration = float(
        Decimal(str(acceleration)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
    )
    maximum = float(
        Decimal(str(maximum)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
    )

    psar = ohlc_data.ta.psar(af=acceleration, max_af=maximum)
    if psar is None:
        raise ConnectionError("Failed to calculate PSAR")
    return contracts.normalize_output(
        "psar", psar, {"acc": acceleration, "maximum": maximum}, index=ohlc_data.index
    )


def calculate_keltner(
    ohlc_data: pd.DataFrame, period: int, multiplier: float = 2.0
) -> pd.DataFrame:
    """Calculate Keltner Channels.

    Parameters
    ----------
    period : int
        Lookback period for the moving average and ATR.
    multiplier : float, default 2.0
        Band width multiplier applied to the ATR.
    """
    if not isinstance(period, int):
        raise TypeError("Keltner 'period' must be int")
    if period < 1:
        raise ValueError("Keltner 'period' must be ≥1")
    if not isinstance(multiplier, (int, float)) or isinstance(multiplier, bool):
        raise TypeError("Keltner 'multiplier' must be numeric")
    if multiplier <= 0:
        raise ValueError("Keltner 'multiplier' must be > 0")
    kc = ohlc_data.ta.kc(length=period, scalar=multiplier)
    if kc is None:
        raise ConnectionError("Failed to calculate Keltner Channels")
    return contracts.normalize_output(
        "keltner",
        kc,
        {"period": period, "multiplier": multiplier},
        index=ohlc_data.index,
    )


def calculate_donchian(ohlc_data: pd.DataFrame, period: int) -> pd.DataFrame:
    """Calculate Donchian Channels."""
    if not isinstance(period, int):
        raise TypeError("Donchian 'period' must be int")
    if period < 1:
        raise ValueError("Donchian 'period' must be ≥1")
    dc = ohlc_data.ta.donchian(lower_length=period, upper_length=period)
    if dc is None:
        raise ConnectionError("Failed to calculate Donchian Channels")
    return contracts.normalize_output(
        "donchian", dc, {"period": period}, index=ohlc_data.index
    )


def calculate_stdev_channel(ohlc_data: pd.DataFrame, period: int) -> pd.Series:
    """Calculate Standard Deviation of price over a period."""
    if not isinstance(period, int):
        raise TypeError("Standard Deviation 'period' must be int")
    if period < 1:
        raise ValueError("Standard Deviation 'period' must be ≥1")
    stdev = ohlc_data.ta.stdev(length=period)
    if stdev is None:
        raise ConnectionError("Failed to calculate Standard Deviation")
    return stdev


def calculate_cmo(ohlc_data: pd.DataFrame, period: int) -> pd.Series:
    """Calculate the Chande Momentum Oscillator (CMO)."""
    if not isinstance(period, int):
        raise TypeError("CMO 'period' must be int")
    if period < 1:
        raise ValueError("CMO 'period' must be ≥1")
    cmo = ohlc_data.ta.cmo(length=period)
    if cmo is None:
        raise ConnectionError("Failed to calculate CMO")
    return cmo


def calculate_obv(ohlc_data: pd.DataFrame) -> pd.Series:
    """Calculate On-Balance Volume (OBV).

    Requires a ``'Volume'`` column in ``ohlc_data``.
    """
    if "Volume" not in ohlc_data.columns:
        raise ValueError("OBV requires a 'Volume' column")
    obv = ohlc_data.ta.obv()
    if obv is None:
        raise ConnectionError("Failed to calculate OBV")
    return obv


def calculate_mfi(ohlc_data: pd.DataFrame, period: int) -> pd.Series:
    """Calculate the Money Flow Index (MFI).

    Requires a ``'Volume'`` column in ``ohlc_data``.
    """
    if not isinstance(period, int):
        raise TypeError("MFI 'period' must be int")
    if period < 1:
        raise ValueError("MFI 'period' must be ≥1")
    if "Volume" not in ohlc_data.columns:
        raise ValueError("MFI requires a 'Volume' column")
    mfi = ohlc_data.ta.mfi(length=period)
    if mfi is None:
        raise ConnectionError("Failed to calculate MFI")
    return mfi


def calculate_adl(ohlc_data: pd.DataFrame) -> pd.Series:
    """Calculate the Accumulation/Distribution Line (ADL).

    Requires a ``'Volume'`` column in ``ohlc_data``.
    """
    if "Volume" not in ohlc_data.columns:
        raise ValueError("ADL requires a 'Volume' column")
    adl = ohlc_data.ta.ad()
    if adl is None:
        raise ConnectionError("Failed to calculate ADL")
    return adl


def calculate_cmf(ohlc_data: pd.DataFrame, period: int) -> pd.Series:
    """Calculate Chaikin Money Flow (CMF).

    Requires a ``'Volume'`` column in ``ohlc_data``.
    """
    if not isinstance(period, int):
        raise TypeError("CMF 'period' must be int")
    if period < 1:
        raise ValueError("CMF 'period' must be ≥1")
    if "Volume" not in ohlc_data.columns:
        raise ValueError("CMF requires a 'Volume' column")
    cmf = ohlc_data.ta.cmf(length=period)
    if cmf is None:
        raise ConnectionError("Failed to calculate CMF")
    return cmf


def calculate_ma_envelope(
    ohlc_data: pd.DataFrame, period: int, percent: float, ma: str = "sma"
) -> pd.DataFrame:
    """Calculate Moving Average Envelopes.

    Parameters
    ----------
    ohlc_data : pd.DataFrame
        Price data containing at least a ``"Close"`` column.
    period : int
        Lookback period for the base moving average.
    percent : float
        Distance from the base moving average. ``2.0`` and ``0.02`` are
        treated equivalently and normalised to a fractional float.
    ma : str, default ``"sma"``
        Moving average method. Any ``pandas_ta`` moving average name is
        accepted. If unavailable, a simple rolling mean is used.

    Returns
    -------
    pd.DataFrame
        Columns ``MAE_U_<period>_<percent>``, ``MAE_M_<period>_<percent>``, and
        ``MAE_L_<period>_<percent>`` for the upper, middle (default), and lower
        bands respectively.
    """

    if "Close" not in ohlc_data.columns:
        raise ValueError("MA Envelope requires a 'Close' column")
    if not isinstance(period, int):
        raise TypeError("MA Envelope 'period' must be int")
    if period < 1:
        raise ValueError("MA Envelope 'period' must be ≥1")
    if not isinstance(percent, (int, float)) or isinstance(percent, bool):
        raise TypeError("MA Envelope 'percent' must be numeric")
    if percent <= 0:
        raise ValueError("MA Envelope 'percent' must be > 0")
    if not isinstance(ma, str):
        raise TypeError("MA Envelope 'ma' must be str")

    # normalise percent for caching: accept 2.0 or 0.02 as the same value
    pct_frac = percent / 100 if percent > 1 else percent
    pct_frac = round(pct_frac, 10)
    pct_display = round(pct_frac * 100, 10)
    percent_str = f"{pct_display:.10g}"
    if "." not in percent_str:
        percent_str += ".0"

    ma_lower = ma.lower()
    ta_obj = ohlc_data.ta
    base = None
    if hasattr(ta_obj, ma_lower):
        base = getattr(ta_obj, ma_lower)(length=period)
    elif hasattr(ta_obj, "ma"):
        try:
            base = ta_obj.ma(mamode=ma_lower, length=period)
        except Exception:  # pragma: no cover - fallback handles errors
            base = None
    if base is None:
        base = ohlc_data["Close"].rolling(window=period, min_periods=period).mean()

    upper = base * (1 + pct_frac)
    lower = base * (1 - pct_frac)
    df = pd.concat(
        [
            upper.rename(f"MAE_U_{period}_{percent_str}"),
            base.rename(f"MAE_M_{period}_{percent_str}"),
            lower.rename(f"MAE_L_{period}_{percent_str}"),
        ],
        axis=1,
    )
    return df


def calculate_ichimoku(
    ohlc_data: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou: int = 52
) -> pd.DataFrame:
    """Calculate the Ichimoku Cloud indicator."""
    params = {"tenkan": tenkan, "kijun": kijun, "senkou": senkou}
    for name, val in params.items():
        if not isinstance(val, int):
            raise TypeError(f"Ichimoku '{name}' must be int")
        if val < 1:
            raise ValueError(f"Ichimoku '{name}' must be ≥1")
    ich = ohlc_data.ta.ichimoku(tenkan=tenkan, kijun=kijun, senkou=senkou)
    if isinstance(ich, tuple):
        ich = ich[0]
    if ich is None:
        raise ConnectionError("Failed to calculate Ichimoku")
    return contracts.normalize_output("ichimoku", ich, params, index=ohlc_data.index)


def calculate_pivot_points(ohlc_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate Pivot Points.

    Attempts to call ``pivot_points`` or ``pivots`` depending on the
    ``pandas-ta`` version.
    """
    ta_mod = ohlc_data.ta
    if hasattr(ta_mod, "pivot_points"):
        piv = ta_mod.pivot_points()
    elif hasattr(ta_mod, "pivots"):
        piv = ta_mod.pivots()
    else:
        raise ConnectionError("pandas-ta pivot function not found")
    if piv is None:
        raise ConnectionError("Failed to calculate Pivot Points")
    return piv


def calculate_trix(
    ohlc_data: pd.DataFrame, period: int, signal: int | None = None
) -> pd.Series | pd.DataFrame:
    """Calculate the TRIX indicator.

    Parameters
    ----------
    period : int
        Lookback period for the TRIX line.
    signal : int, optional
        Period for an optional signal line. Must be ``>=1`` and ``<= period``.
    """
    if not isinstance(period, int):
        raise TypeError("TRIX 'period' must be int")
    if period < 1:
        raise ValueError("TRIX 'period' must be ≥1")
    if signal is not None:
        if not isinstance(signal, int):
            raise TypeError("TRIX 'signal' must be int")
        if signal < 1:
            raise ValueError("TRIX 'signal' must be ≥1")
        if signal > period:
            raise ValueError("TRIX 'signal' must be ≤ 'period'")
        trix = ohlc_data.ta.trix(length=period, signal=signal)
    else:
        trix = ohlc_data.ta.trix(length=period)
    if trix is None:
        raise ConnectionError("Failed to calculate TRIX")
    params = {"period": period}
    if signal is not None:
        params["signal"] = signal
    return contracts.normalize_output("trix", trix, params, index=ohlc_data.index)


def calculate_roc(ohlc_data: pd.DataFrame, period: int) -> pd.Series:
    """Calculate Rate of Change (ROC)."""
    if not isinstance(period, int):
        raise TypeError("ROC 'period' must be int")
    if period < 1:
        raise ValueError("ROC 'period' must be ≥1")
    roc = ohlc_data.ta.roc(length=period)
    if roc is None:
        raise ConnectionError("Failed to calculate ROC")
    return roc


# -- Auto-registration of indicator functions -------------------------------
for name, obj in list(globals().items()):
    if (
        name.startswith("calculate_")
        and callable(obj)
        and getattr(obj, "__module__", None) == __name__
    ):
        key = name[len("calculate_") :]
        if key not in INDICATOR_REGISTRY:
            INDICATOR_REGISTRY[key] = obj
# Backward-compatible alias for standard deviation channel
if "stdev_channel" in INDICATOR_REGISTRY:
    INDICATOR_REGISTRY["stdev"] = INDICATOR_REGISTRY["stdev_channel"]
