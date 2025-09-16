import math
from types import SimpleNamespace

import itertools
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


_BASE_VALUES = {
    "sortino": 1.25,
    "profit_factor": 2.5,
    "max_drawdown": 0.12,
    "total_return": 0.4,
}

_EXPECTED_VALUES = {
    "sortino": _BASE_VALUES["sortino"],
    "profit_factor": _BASE_VALUES["profit_factor"],
    "max_drawdown": _BASE_VALUES["max_drawdown"] * 100.0,
    "total_return": _BASE_VALUES["total_return"] * 100.0,
}

_RETURNS = [0.1, -0.05, 0.02]


@pytest.mark.parametrize(
    "sortino_alias, pf_alias, dd_alias, tr_alias",
    list(
        itertools.product(
            metrics_contract.METRIC_ALIASES["sortino"],
            metrics_contract.METRIC_ALIASES["profit_factor"],
            metrics_contract.METRIC_ALIASES["max_drawdown"],
            metrics_contract.METRIC_ALIASES["total_return"],
        )
    ),
)
def test_resolve_metrics_handles_all_alias_variants(
    sortino_alias, pf_alias, dd_alias, tr_alias
):
    stats = {
        sortino_alias: _BASE_VALUES["sortino"],
        pf_alias: _BASE_VALUES["profit_factor"],
        dd_alias: _BASE_VALUES["max_drawdown"],
        tr_alias: _BASE_VALUES["total_return"],
    }
    portfolio = DummyPortfolio(stats, returns=_RETURNS)

    metrics, aliases = metrics_contract.resolve_metrics(portfolio)

    assert pytest.approx(_EXPECTED_VALUES["sortino"]) == metrics["sortino"]
    assert pytest.approx(_EXPECTED_VALUES["profit_factor"]) == metrics["profit_factor"]
    assert pytest.approx(_EXPECTED_VALUES["max_drawdown"]) == metrics["max_drawdown"]
    assert pytest.approx(_EXPECTED_VALUES["total_return"]) == metrics["total_return"]
    assert aliases["sortino"] == sortino_alias
    assert aliases["profit_factor"] == pf_alias
    assert aliases["max_drawdown"] == dd_alias
    assert aliases["total_return"] == tr_alias


def test_resolve_metrics_recovers_after_alias_drift():
    initial_stats = {
        "sortino": _BASE_VALUES["sortino"],
        "profit_factor": _BASE_VALUES["profit_factor"],
        "max_drawdown": _BASE_VALUES["max_drawdown"],
        "total_return": _BASE_VALUES["total_return"],
    }
    portfolio = DummyPortfolio(initial_stats, returns=_RETURNS)

    metrics, aliases = metrics_contract.resolve_metrics(portfolio)
    assert aliases["sortino"] == "sortino"

    renamed_stats = {
        "Sortino Ratio": _BASE_VALUES["sortino"],
        "Profit Factor": _BASE_VALUES["profit_factor"],
        "Max Drawdown [%]": _BASE_VALUES["max_drawdown"],
        "Return [%]": _BASE_VALUES["total_return"],
    }
    portfolio._stats = renamed_stats

    metrics, aliases = metrics_contract.resolve_metrics(portfolio)

    assert pytest.approx(_EXPECTED_VALUES["sortino"]) == metrics["sortino"]
    assert aliases["sortino"] == "Sortino Ratio"
    assert aliases["profit_factor"] == "Profit Factor"
    assert aliases["max_drawdown"] == "Max Drawdown [%]"
    assert aliases["total_return"] == "Return [%]"


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
