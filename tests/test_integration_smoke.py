import copy
from collections import Counter

import pandas as pd
import pytest

import fitness


@pytest.fixture(autouse=True)
def _sequential_executor(monkeypatch):
    class _ImmediateFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

        def add_done_callback(self, fn):
            fn(self)

    def _submit(fn, *args, **kwargs):
        return _ImmediateFuture(fn(*args, **kwargs))

    monkeypatch.setattr(fitness.global_executor, "submit", _submit)
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


def _df():
    idx = pd.date_range("2020", periods=5)
    return pd.DataFrame(
        {
            "Open": [1, 2, 3, 4, 5],
            "High": [1, 2, 3, 4, 5],
            "Low": [1, 2, 3, 4, 5],
            "Close": [1, 2, 3, 4, 5],
        },
        index=idx,
    )


def test_integration_smoke():
    data = {"A": _df()}
    rules = {"entry_rules": {"conditions": []}}
    evaluator = fitness.MultiAssetFitnessEvaluator(
        data, rules, {}, settings={"zero_trade_policy": "penalize"}
    )
    score = evaluator(None, [], 0)
    details = evaluator.last_details
    per_asset = details["per_asset"].get("A", {})
    assert per_asset.get("reason") != "evaluation_error"
    assert details.get("assets_included", 0) >= 1
    assert score != -999
