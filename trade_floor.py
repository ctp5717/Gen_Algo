import math


def scale_floor(rate_per_year, start_ts, end_ts):
    years = (end_ts - start_ts).days / 365.25
    raw = rate_per_year * years
    scaled = math.ceil(raw) if rate_per_year > 0 else 0
    return scaled, {
        "rate_per_year": rate_per_year,
        "years": years,
        "raw": raw,
        "ceil": scaled,
    }
