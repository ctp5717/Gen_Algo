import copy
import math
from collections import Counter
from types import SimpleNamespace

import pandas as pd
import pytest

import config
import fitness
import metrics_contract


@pytest.fixture(autouse=True)
def reset_metric_cache():
    metrics_contract.reset_cache()


class _ImmediateFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result

    def add_done_callback(self, fn):
        fn(self)


@pytest.fixture(autouse=True)
def _sequential_executor(monkeypatch):
    monkeypatch.setattr(
        fitness.global_executor,
        "submit",
        lambda fn, *args, **kwargs: _ImmediateFuture(fn(*args, **kwargs)),
    )
    monkeypatch.setattr(
        fitness.global_executor,
        "metrics",
        lambda: {
            "submitted": 0,
            "completed": 0,
            "total_runtime": 0.0,
            "pending": 0,
            "max_pending": 0,
            "in_flight_cap": 0,
            "base_in_flight_cap": 0,
            "bytes_avg": 0.0,
        },
    )

    def _sync_population(self, solutions, indices):
        if not solutions:
            return [], Counter()

        fitness_values: list[float] = []
        err_counts: Counter = Counter()

        for vector, idx in zip(solutions, indices, strict=False):
            results, errors = self._evaluate_assets({"solution": vector})
            err_counts.update(errors)
            assets_map = {
                ticker: results.get(ticker, self._build_evaluation_record())
                for ticker in self._sorted_tickers
            }
            summary = self._score_assets(assets_map)
            score = self._aggregate_scores(summary)
            fitness_values.append(score)
            self.batch_details[idx] = copy.deepcopy(self.last_details)

        record = {
            "tasks_submitted": 0,
            "evaluations": len(fitness_values) * len(self._sorted_tickers),
            "latency": [],
            "latency_mean": 0.0,
            "latency_p95": 0.0,
            "latency_target": getattr(self, "_latency_target", 0.0),
            "throughput": 0.0,
            "cpu_time": 0.0,
            "occupancy": 0.0,
            "pending": 0,
            "max_pending": 0,
            "queue_depth": 0,
            "queue_ratio": 0.0,
            "serialization_bytes": 0,
            "rows_processed": 0,
            "submitted": 0,
            "completed": 0,
            "batch_size": getattr(self, "_batch_size", 1),
            "next_batch_size": getattr(self, "_batch_size", 1),
            "in_flight_cap": 0,
            "base_in_flight_cap": 0,
            "bytes_avg": 0.0,
            "worker_count": 0,
            "worker_seeds": [],
            "reducer_timeouts": 0,
            "error_counts": {},
            "error_top": [],
        }
        self.instrumentation = record
        if hasattr(self, "_generation_records"):
            self._generation_records.append(dict(record))

        return fitness_values, err_counts

    monkeypatch.setattr(
        fitness.MultiAssetFitnessEvaluator,
        "_evaluate_population",
        _sync_population,
    )


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
    signature = metrics_contract._provider_signature(_MissingPortfolio([], {}))
    detail = record.get("reason_detail") or ""
    assert signature in detail
    assert "sortino" in detail


def test_evaluation_warning_and_metrics_missing_on_failure(monkeypatch, caplog):
    config.initialize_config(force=True)
    monkeypatch.setattr(fitness.engine, "process_strategy_rules", _fake_process_rules)

    stats = {
        "sortino": 1.0,
        "profit_factor": 1.5,
        "max_drawdown": 0.1,
        "total_return": 0.2,
    }

    def fake_from_signals(*args, **kwargs):
        return _AliasPortfolio([0.1, -0.05, 0.02], stats)

    monkeypatch.setattr(fitness.vbt.Portfolio, "from_signals", fake_from_signals)

    def boom(portfolio):
        raise KeyError("sortino")

    monkeypatch.setattr(metrics_contract, "evaluate_metrics", boom)

    evaluator = fitness.MultiAssetFitnessEvaluator(
        {"CCC": pd.DataFrame({"Close": [1.0, 1.1, 1.05]})},
        {},
        {},
        settings=_multi_asset_settings(),
    )

    with caplog.at_level("WARNING"):
        results, _ = evaluator._evaluate_assets({})

    record = results["CCC"]
    assert record.get("evaluation_reason") == "metrics_missing"
    assert set(record.get("missing_metrics") or ()) == set(
        metrics_contract.METRIC_ALIASES
    )
    signature = metrics_contract._provider_signature(_AliasPortfolio([], {}))
    detail = record.get("reason_detail") or ""
    assert signature in detail
    score = evaluator(None, [], 0)
    assert score != -999.0
    assert any("Metric evaluation failed for" in msg for msg in caplog.messages)
    assert any(signature in msg for msg in caplog.messages)
    detail = evaluator.last_details["per_asset"]["CCC"]
    assert detail.get("missing_metrics")
    assert detail.get("metric_sources")
    assert detail.get("metric_provider") == signature
