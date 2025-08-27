import datetime as dt
import pytest
from trade_floor import scale_floor


def test_exact_quarter_scaling():
    start = dt.datetime(2021, 1, 1)
    end = dt.datetime(2021, 4, 1)
    floor, info = scale_floor(4, start, end)
    assert floor == 1
    assert info["ceil"] == 1


@pytest.mark.parametrize("days,expected", [(89, 1), (90, 1), (91, 1), (92, 2)])
def test_span_scaling(days, expected):
    start = dt.datetime(2021, 1, 1)
    end = start + dt.timedelta(days=days)
    floor, _ = scale_floor(4, start, end)
    assert floor == expected


def test_leap_year_window():
    rate = 1
    start_non = dt.datetime(2019, 1, 1)
    end_non = dt.datetime(2020, 1, 1)
    floor_non, _ = scale_floor(rate, start_non, end_non)
    assert floor_non == 1
    start_leap = dt.datetime(2020, 1, 1)
    end_leap = dt.datetime(2021, 1, 1)
    floor_leap, _ = scale_floor(rate, start_leap, end_leap)
    assert floor_leap == 2
