import sys
import types
from pathlib import Path
import pandas as pd

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
    data_a = pd.DataFrame({"Open": close_a, "High": close_a, "Low": close_a, "Close": close_a, "Volume": 1}, index=idx)
    data_b = pd.DataFrame({"Open": close_b, "High": close_b, "Low": close_b, "Close": close_b, "Volume": 1}, index=idx)
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
    assert fitness._timeframe_to_freq("15m") == "15T"
    assert fitness._timeframe_to_freq("1h") == "1H"
    assert fitness._timeframe_to_freq("1d") == "1D"
    assert fitness._timeframe_to_freq("1wk") == "1W"
    assert fitness._timeframe_to_freq("1mo") == "1M"
