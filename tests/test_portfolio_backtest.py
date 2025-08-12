import sys
import types
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))

import fitness  # noqa: E402


class DummyTrades:
    records_readable = pd.DataFrame()


class DummyPortfolio:
    def __init__(self, close):
        self._close = close
        self.trades = DummyTrades()

    def value(self):
        return self._close

    def total(self):
        return self

    def returns(self, column=None):
        data = self._close if column is None else self._close[column]
        return data.pct_change().fillna(0)

    def stats(self, column=None, silence_warnings=False):
        ret = self.returns(column=column)
        return pd.Series({"Total Trades": 1, "Volatility": ret.std()})


def _fake_from_signals(close, *args, **kwargs):
    return DummyPortfolio(close)


def _make_ohlc(close_a, close_b):
    idx = close_a.index
    data_a = pd.DataFrame(
        {
            "Open": close_a,
            "High": close_a,
            "Low": close_a,
            "Close": close_a,
            "Volume": 1,
        },
        index=idx,
    )
    data_b = pd.DataFrame(
        {
            "Open": close_b,
            "High": close_b,
            "Low": close_b,
            "Close": close_b,
            "Volume": 1,
        },
        index=idx,
    )
    return pd.concat({"A": data_a, "B": data_b}, axis=1)


def _make_entries(idx):
    return pd.DataFrame({"A": [True] * len(idx), "B": [True] * len(idx)}, index=idx)


def test_custom_weights_affect_aggregation(monkeypatch):
    monkeypatch.setattr(fitness.vbt.Portfolio, "from_signals", _fake_from_signals)
    dates = pd.date_range("2020", periods=3, freq="D")
    ohlc = _make_ohlc(pd.Series([1, 2, 3], dates), pd.Series([1, 1, 1], dates))
    entries = _make_entries(dates)
    _, agg_eq, _, _ = fitness.run_portfolio_backtest(ohlc, entries)
    _, agg_w, _, _ = fitness.run_portfolio_backtest(ohlc, entries, weights=[0.8, 0.2])
    assert agg_eq.iloc[-1] != agg_w.iloc[-1]


def test_volatility_non_zero(monkeypatch):
    monkeypatch.setattr(fitness.vbt.Portfolio, "from_signals", _fake_from_signals)
    dates = pd.date_range("2020", periods=3, freq="D")
    ohlc = _make_ohlc(pd.Series([1, 2, 3], dates), pd.Series([1, 2, 1], dates))
    entries = _make_entries(dates)
    _, _, agg_stats, _ = fitness.run_portfolio_backtest(ohlc, entries)
    assert agg_stats["Volatility"] > 0


def test_timeframe_to_freq():
    assert fitness._timeframe_to_freq("15m") == "15min"
    assert fitness._timeframe_to_freq("1h") == "1H"
    assert fitness._timeframe_to_freq("1d") == "1D"
    assert fitness._timeframe_to_freq("1wk") == "1W"
    assert fitness._timeframe_to_freq("1mo") == "1M"


def test_timeframe_to_freq_no_deprecation():
    import warnings

    with warnings.catch_warnings(record=True) as w:
        freq = fitness._timeframe_to_freq("15m")
        # Force pandas to parse the frequency to trigger any potential warning
        pd.Timedelta(freq)

    assert not any("deprecated" in str(warn.message).lower() for warn in w)


def test_agg_stats_contains_standard_keys(monkeypatch):
    ohlc = pd.DataFrame(
        {
            ("A", "Close"): [1, 2, 3],
            ("B", "Close"): [4, 5, 4],
        },
        index=pd.date_range("2020-01-01", periods=3),
    )
    ohlc.columns = pd.MultiIndex.from_tuples(ohlc.columns)
    entries = pd.DataFrame(
        [[True, True], [False, True], [False, False]],
        index=ohlc.index,
        columns=["A", "B"],
    )

    class DummyPortfolio:
        def stats(self, column=None, silence_warnings=None):
            return pd.Series({"Total Trades": 1, "Win Rate [%]": np.nan})

        def value(self):
            return pd.DataFrame({"A": [100, 110, 105], "B": [200, 190, 195]}, index=ohlc.index)

        def returns(self):
            return pd.DataFrame(
                {"A": [0.1, -0.0454545, 0.0], "B": [-0.05, 0.0263158, 0.0]},
                index=ohlc.index,
            )

        @property
        def trades(self):
            class T:
                @property
                def records_readable(self):
                    return pd.DataFrame({"PnL": [10, -5], "Column": ["A", "B"]})

            return T()

        def total(self):
            return self

    monkeypatch.setattr(fitness.vbt.Portfolio, "from_signals", lambda *a, **k: DummyPortfolio())
    _, _, agg_stats, _ = fitness.run_portfolio_backtest(ohlc, entries, weights=[0.6, 0.4])

    keys = [
        "Total Return [%]",
        "Max Drawdown [%]",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Volatility",
        "Total Trades",
        "Win Rate [%]",
        "Profit Factor",
    ]
    for k in keys:
        assert k in agg_stats.index


def test_weights_zero_sum_raises(monkeypatch):
    monkeypatch.setattr(fitness.vbt.Portfolio, "from_signals", _fake_from_signals)
    dates = pd.date_range("2020", periods=2, freq="D")
    ohlc = _make_ohlc(pd.Series([1, 2], dates), pd.Series([1, 2], dates))
    entries = _make_entries(dates)
    with pytest.raises(ValueError):
        fitness.run_portfolio_backtest(ohlc, entries, weights=[0.0, 0.0])
