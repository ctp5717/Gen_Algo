import logging
import math
from types import SimpleNamespace

import pandas as pd
import pytest

import config
import fitness
import metrics_contract


@pytest.fixture(autouse=True)
def reset_metric_cache():
    metrics_contract.reset_cache()


def _fake_process_rules(ohlc, rules, collect_counts=False):
    entries = pd.Series([True, False, True], index=ohlc.index)
    if collect_counts:
        return entries, {"entries": int(entries.sum())}
    return entries


class _AliasPortfolio:
    def __init__(self, returns, stats):
        self._returns = pd.Series(returns)
        self._value = (1 + self._returns).cumprod()
        self._stats = stats
        self.trades = SimpleNamespace(count=lambda: 2)

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
        return self._returns

    def value(self):
        return self._value


class _MissingPortfolio(_AliasPortfolio):
    def returns(self):  # pragma: no cover - explicit override
        return None

    def value(self):  # pragma: no cover - explicit override
        return pd.Series(dtype=float)


def _multi_asset_settings():
    return {
        "enabled": True,
        "min_included_assets": 1,
        "per_asset_min_trades": 0,
        "min_total_trades": 0,
        "lambda_dispersion": 0.0,
        "winsorize_pf_cap": 5.0,
        "zero_trade_policy": "ignore",
        "parallel": {"enabled": False},
    }


def test_multi_asset_handles_alias_fallback(monkeypatch):
    config.initialize_config(force=True)
    monkeypatch.setattr(fitness.engine, "process_strategy_rules", _fake_process_rules)

    stats = {
        "sortino_ratio": 1.0,
        "Max Drawdown": 0.04,
    }

    def fake_from_signals(*args, **kwargs):
        return _AliasPortfolio([0.1, -0.05, 0.02], stats)

    monkeypatch.setattr(fitness.vbt.Portfolio, "from_signals", fake_from_signals)

    ohlc = pd.DataFrame({"Close": [1.0, 1.1, 1.05]})
    evaluator = fitness.MultiAssetFitnessEvaluator(
        {"AAA": ohlc},
        {},
        {},
        settings=_multi_asset_settings(),
    )

    results, _ = evaluator._evaluate_assets({})
    record = results["AAA"]
    assert record.get("evaluation_reason") is None

    score = evaluator(None, [], 0)
    assert math.isfinite(score)
    detail = evaluator.last_details["per_asset"]["AAA"]
    assert detail.get("reason") is None
    assert detail.get("metric_sources")


def test_metrics_missing_reason_when_fallback_unavailable(monkeypatch):
    config.initialize_config(force=True)
    monkeypatch.setattr(fitness.engine, "process_strategy_rules", _fake_process_rules)

    def fake_from_signals(*args, **kwargs):
        return _MissingPortfolio([], {})

    monkeypatch.setattr(fitness.vbt.Portfolio, "from_signals", fake_from_signals)

    ohlc = pd.DataFrame({"Close": [1.0, 1.1, 1.05]})
    evaluator = fitness.MultiAssetFitnessEvaluator(
        {"BBB": ohlc},
        {},
        {},
        settings=_multi_asset_settings(),
    )

    results, _ = evaluator._evaluate_assets({})
    record = results["BBB"]
    assert record.get("evaluation_reason") == "metrics_missing"
    assert "sortino" in (record.get("reason_detail") or "")


def test_metrics_error_falls_back_to_missing(monkeypatch, caplog):
    config.initialize_config(force=True)
    monkeypatch.setattr(fitness.engine, "process_strategy_rules", _fake_process_rules)

    stats = {
        "sortino": 1.0,
        "profit_factor": 1.5,
        "max_drawdown": 0.08,
        "total_return": 0.25,
    }

    def fake_from_signals(*args, **kwargs):
        return _AliasPortfolio([0.1, -0.05, 0.02], stats)

    def explode(*args, **kwargs):
        raise KeyError("sortino")

    monkeypatch.setattr(fitness.vbt.Portfolio, "from_signals", fake_from_signals)
    monkeypatch.setattr(metrics_contract, "evaluate_metrics", explode)

    caplog.set_level(logging.WARNING, logger=fitness.logger.name)

    ohlc = pd.DataFrame({"Close": [1.0, 1.1, 1.05]})
    evaluator = fitness.MultiAssetFitnessEvaluator(
        {"AAA": ohlc},
        {},
        {},
        settings=_multi_asset_settings(),
    )

    results, _ = evaluator._evaluate_assets({})
    record = results["AAA"]
    assert record.get("evaluation_reason") == "metrics_missing"
    assert set((record.get("metric_sources") or {}).values()) == {"missing"}

    score = evaluator(None, [], 0)
    assert score != -999.0
    assert any("Metric evaluation failed" in msg for msg in caplog.messages)
