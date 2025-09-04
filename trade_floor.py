import math


def scale_floor(base_floor, start_ts, end_ts, trading_days_per_year: int = 252):
    """Scale a per-asset trade floor to the given window.

    Parameters
    ----------
    base_floor : int | float
        Desired trades per asset on an annual basis.
    start_ts, end_ts : datetime-like
        Window bounds.
    trading_days_per_year : int, optional
        Annualisation base. Defaults to ``252`` trading days.

    Returns
    -------
    tuple[int, dict]
        Scaled floor (minimum of 1 if ``base_floor`` > 0) and an info dict.
    """

    window_days = (end_ts - start_ts).total_seconds() / 86400
    years = window_days / trading_days_per_year if trading_days_per_year else 0
    raw = base_floor * years
    ceil_val = max(1, math.ceil(raw)) if base_floor > 0 else 0
    info = {
        "base_floor": base_floor,
        "window_days": window_days,
        "trading_days_per_year": trading_days_per_year,
        "years": years,
        "raw": raw,
        "ceil": ceil_val,
    }
    return ceil_val, info
