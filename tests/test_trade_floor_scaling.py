import datetime as dt

import pytest

from trade_floor import scale_floor


@pytest.mark.parametrize(
    "days,expected",
    [
        (15, 1),
        (30, 1),
        (90, 2),
        (365, 6),
        (400, 7),
    ],
)
def test_scale_floor_various_windows(days, expected):
    start = dt.datetime(2021, 1, 1)
    end = start + dt.timedelta(days=days)
    floor, info = scale_floor(4, start, end)
    assert floor == expected
    assert info["ceil"] == expected
    assert info["window_days"] == days


def test_minimum_one():
    start = dt.datetime(2021, 1, 1)
    end = start + dt.timedelta(days=1)
    floor, info = scale_floor(4, start, end)
    assert floor == 1
    assert info["ceil"] == 1


def test_scale_floor_custom_base():
    start = dt.datetime(2021, 1, 1)
    end = start + dt.timedelta(days=365)
    floor_default, _ = scale_floor(4, start, end)
    floor_alt, info = scale_floor(4, start, end, trading_days_per_year=365)
    assert floor_default != floor_alt
    assert floor_default == 6
    assert floor_alt == 4
    assert info["trading_days_per_year"] == 365


def test_scale_floor_zero_base():
    start = dt.datetime(2021, 1, 1)
    end = start + dt.timedelta(days=365)
    floor, info = scale_floor(0, start, end)
    assert floor == 0
    assert info["ceil"] == 0
