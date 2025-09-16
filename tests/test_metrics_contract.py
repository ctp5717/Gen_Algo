import logging
import math
import random
import re
import string
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import config
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

_EXPECTED_METRICS = {
    "sortino": 1.25,
    "profit_factor": 2.5,
    "max_drawdown": 12.0,
    "total_return": 40.0,
}

_DEFAULT_RETURNS = [0.1, -0.05, 0.02]


def _build_stats(alias_overrides: dict[str, str] | None = None) -> dict[str, float]:
    alias_map = {
        metric: metrics_contract.METRIC_ALIASES[metric][0]
        for metric in _BASE_VALUES
    }
    if alias_overrides:
        alias_map.update(alias_overrides)
    return {alias_map[key]: _BASE_VALUES[key] for key in _BASE_VALUES}


def test_resolve_metrics_handles_aliases_and_units():
    stats = {
        "sortino_ratio": _BASE_VALUES["sortino"],
        "profit_factor": _BASE_VALUES["profit_factor"],
        "Max Drawdown": _BASE_VALUES["max_drawdown"],
        "total_return": _BASE_VALUES["total_return"],
    }
    portfolio = DummyPortfolio(stats, returns=_DEFAULT_RETURNS)

    metrics, aliases = metrics_contract.resolve_metrics(portfolio)

    assert pytest.approx(1.25) == metrics["sortino"]
    assert pytest.approx(2.5) == metrics["profit_factor"]
    assert pytest.approx(12.0) == metrics["max_drawdown"]  # converted to %
    assert pytest.approx(40.0) == metrics["total_return"]
    assert aliases["sortino"] == "sortino_ratio"
    assert aliases["max_drawdown"] in {"Max Drawdown", "Max Drawdown [%]"}


@pytest.mark.parametrize(
    "metric, alias",
    [
        (metric, alias)
        for metric, aliases in metrics_contract.METRIC_ALIASES.items()
        for alias in aliases
    ],
)
def test_resolve_metrics_handles_alias_variants(metric, alias):
    stats = _build_stats({metric: alias})
    portfolio = DummyPortfolio(stats, returns=_DEFAULT_RETURNS)

    metrics, aliases = metrics_contract.resolve_metrics(portfolio)

    for key, expected in _EXPECTED_METRICS.items():
        assert pytest.approx(expected) == metrics[key]
    assert aliases[metric] == alias


def test_resolve_metrics_collapses_wrapped_scalars():
    stats = {
        "sortino": pd.Series([_BASE_VALUES["sortino"]]),
        "profit_factor": {"pf": _BASE_VALUES["profit_factor"]},
        "max_drawdown": pd.Series([np.nan, _BASE_VALUES["max_drawdown"]]),
        "total_return": {"value": _BASE_VALUES["total_return"]},
    }
    portfolio = DummyPortfolio(stats, returns=_DEFAULT_RETURNS)

    metrics, aliases = metrics_contract.resolve_metrics(portfolio)

    for key, expected in _EXPECTED_METRICS.items():
        assert pytest.approx(expected) == metrics[key]
    assert aliases["sortino"] == "sortino"
    assert aliases["profit_factor"] == "profit_factor"


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
        "sortino_ratio": _BASE_VALUES["sortino"],
        "profit_factor": _BASE_VALUES["profit_factor"],
        "Max Drawdown": _BASE_VALUES["max_drawdown"],
        "total_return": _BASE_VALUES["total_return"],
    }
    stats.pop(alias)
    portfolio = DummyPortfolio(stats, returns=_DEFAULT_RETURNS)

    metrics, _ = metrics_contract.resolve_metrics(portfolio)
    assert metrics[missing] is None

    metrics, computed = metrics_contract.compute_fallbacks(portfolio, metrics)

    assert computed[missing] == "computed"
    assert pytest.approx(expected) == metrics[missing]


def test_discovery_logs_debounced(caplog):
    portfolio = DummyPortfolio(_build_stats(), returns=_DEFAULT_RETURNS)
    signature = metrics_contract._provider_signature(portfolio)

    with caplog.at_level(logging.INFO):
        metrics_contract._discover_aliases(portfolio, signature)

    discovery = [
        rec for rec in caplog.records if "Discovering metric aliases" in rec.message
    ]
    assert discovery and discovery[0].levelno == logging.INFO

    caplog.clear()

    with caplog.at_level(logging.DEBUG):
        metrics_contract._discover_aliases(portfolio, signature)

    discovery = [
        rec for rec in caplog.records if "Discovering metric aliases" in rec.message
    ]
    assert discovery and discovery[0].levelno == logging.DEBUG


def test_resolve_metrics_refreshes_cache_on_alias_drift():
    initial = _build_stats()
    portfolio = DummyPortfolio(initial, returns=_DEFAULT_RETURNS)
    metrics_contract.resolve_metrics(portfolio)

    alias_updates = {
        "sortino": "Sortino Ratio",
        "profit_factor": "Profit Factor",
        "max_drawdown": "Max Drawdown [%]",
        "total_return": "Total Return [%]",
    }
    portfolio._stats = _build_stats(alias_updates)

    metrics, aliases = metrics_contract.resolve_metrics(portfolio)

    for key, expected in _EXPECTED_METRICS.items():
        assert pytest.approx(expected) == metrics[key]
        assert aliases[key] == alias_updates[key]


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


def test_resolve_metrics_handles_multiindex_dataframe():
    class DataFramePortfolio:
        def __init__(self):
            index = pd.MultiIndex.from_tuples(
                [
                    ("Ratios", "Sortino Ratio!"),
                    ("Ratios", "Profit Factor"),
                    ("Drawdown", "Max Drawdown (%)"),
                    ("Performance", "Total Return (%)"),
                ]
            )
            self._series = pd.Series(
                [
                    _BASE_VALUES["sortino"],
                    _BASE_VALUES["profit_factor"],
                    _BASE_VALUES["max_drawdown"],
                    _BASE_VALUES["total_return"],
                ],
                index=index,
            )
            self._returns = _DEFAULT_RETURNS
            self.trades = SimpleNamespace(count=lambda: 1)

        def stats(self, metrics=None):
            if metrics is not None:
                raise KeyError(metrics[0] if metrics else "metrics")
            return self._series

        def returns(self):
            return pd.Series(self._returns)

    portfolio = DataFramePortfolio()
    metrics, aliases = metrics_contract.resolve_metrics(portfolio)

    for key, expected in _EXPECTED_METRICS.items():
        assert pytest.approx(expected) == metrics[key]
    assert "Sortino Ratio" in aliases["sortino"]
    assert "Max Drawdown" in aliases["max_drawdown"]
    assert "Total Return" in aliases["total_return"]


def _manual_key_norm(value: object) -> str:
    lowered = str(value).lower()
    sanitized = re.sub(r"[^0-9a-z]+", "_", lowered)
    collapsed = re.sub(r"_+", "_", sanitized)
    return collapsed.strip("_")


def test_key_norm_matches_reference_implementation():
    rng = random.Random(42)
    alphabet = string.ascii_letters + string.digits + string.punctuation + " \t\n"
    for _ in range(200):
        length = rng.randint(1, 32)
        raw = "".join(rng.choice(alphabet) for _ in range(length))
        expected = _manual_key_norm(raw)
        assert metrics_contract._key_norm(raw) == expected


def test_assert_metric_aliases_fail_mode_raises(monkeypatch):
    portfolio = DummyPortfolio({"profit_factor": _BASE_VALUES["profit_factor"]})
    monkeypatch.setattr(
        config,
        "METRICS_PREFLIGHT",
        {"mode": "fail", "missing_threshold": 0},
        raising=False,
    )

    with pytest.raises(metrics_contract.MetricsAliasError):
        metrics_contract.assert_metric_aliases(portfolio)
