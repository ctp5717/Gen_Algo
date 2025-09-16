import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import metrics_contract


@pytest.fixture(autouse=True)
def reset_metrics_contract():
    metrics_contract.reset_cache()


class DummyPortfolio:
    def __init__(self, stats: dict[str, float], returns=None):
        self._stats = stats
        self._returns = returns
        self.trades = SimpleNamespace(count=lambda: 1)

    def stats(self, metrics=None):
        if metrics is None:
            return self._stats
        result = {}
        for metric in metrics:
            if metric not in self._stats:
                raise KeyError(metric)
            result[metric] = self._stats[metric]
        return result

    def returns(self):
        if self._returns is None:
            return None
        return pd.Series(self._returns)


def test_resolve_metrics_handles_aliases_and_units():
    stats = {
        "sortino_ratio": 1.25,
        "profit_factor": 2.5,
        "Max Drawdown": 0.12,
        "total_return": 0.4,
    }
    portfolio = DummyPortfolio(stats, returns=[0.1, -0.05, 0.02])

    metrics, aliases = metrics_contract.resolve_metrics(portfolio)

    assert pytest.approx(1.25) == metrics["sortino"]
    assert pytest.approx(2.5) == metrics["profit_factor"]
    assert pytest.approx(12.0) == metrics["max_drawdown"]  # converted to %
    assert pytest.approx(40.0) == metrics["total_return"]
    assert aliases["sortino"] == "sortino_ratio"
    assert aliases["max_drawdown"] in {"Max Drawdown", "Max Drawdown [%]"}


@pytest.mark.parametrize(
    ("missing", "alias", "expected"),
    [
        ("sortino", "sortino_ratio", 7.4081036709808545),
        ("profit_factor", "profit_factor", 2.4),
        ("max_drawdown", "Max Drawdown", 5.0),
        ("total_return", "total_return", 6.589999999999986),
    ],
)
def test_compute_fallbacks_fill_missing_metrics(missing, alias, expected):
    stats = {
        "sortino_ratio": 1.25,
        "profit_factor": 2.5,
        "Max Drawdown": 0.12,
        "total_return": 0.4,
    }
    stats.pop(alias)
    portfolio = DummyPortfolio(stats, returns=[0.1, -0.05, 0.02])

    metrics, _ = metrics_contract.resolve_metrics(portfolio)
    assert metrics[missing] is None

    metrics, computed = metrics_contract.compute_fallbacks(portfolio, metrics)

    assert computed[missing] == "computed"
    assert pytest.approx(expected) == metrics[missing]


def test_compute_fallbacks_with_no_returns_leaves_missing_metrics():
    portfolio = DummyPortfolio({}, returns=None)

    metrics, _ = metrics_contract.resolve_metrics(portfolio)
    metrics, computed = metrics_contract.compute_fallbacks(portfolio, metrics)

    assert computed == {}
    assert all(value is None for value in metrics.values())


def test_to_pct_scaling_stays_within_expected_bounds():
    rng = np.random.default_rng(1234)
    for raw in rng.uniform(-500, 500, size=500):
        scaled = metrics_contract._to_pct(raw)
        if isinstance(scaled, (float, np.floating)) and not math.isnan(raw):
            baseline = max(1.0, abs(raw))
            assert abs(scaled) <= baseline * 100.0 + 1e-9
