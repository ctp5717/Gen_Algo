import sys
import types
from pathlib import Path

# Ensure repository root is on the import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy optional dependencies
sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))
sys.modules.setdefault("vectorbt", types.ModuleType("vectorbt"))

import copy
from collections import Counter

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import fitness  # noqa: E402


def _make_evaluator(stats, settings=None):
    """Utility to construct a MultiAssetFitnessEvaluator with patched stats."""
    group_data = {
        "A": pd.DataFrame({"Close": [1, 2, 3]}),
    }
    evaluator = fitness.MultiAssetFitnessEvaluator(group_data, {}, {}, settings or {})

    def fake_eval(self, ohlc, rules):
        return stats

    evaluator._evaluate_single_asset = types.MethodType(fake_eval, evaluator)
    return evaluator


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
                ticker: results.get(
                    ticker, self._build_evaluation_record()
                )
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

def test_near_zero_losses_winsorized():
    profit = 1000.0
    loss = 1e-9
    pf_raw = profit / loss
    stats = {
        "sortino": 1.0,
        "profit_factor": pf_raw,
        "max_drawdown": 10.0,
        "trades": 5,
        "total_return": 1.0,
    }
    settings = {
        "winsorize_pf_cap": 5.0,
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 0,
        "lambda_dispersion": 0.0,
    }
    ev = _make_evaluator(stats, settings)
    ev(None, [], 0)
    details = ev.last_details["per_asset"]["A"]
    assert details["profit_factor"] == pf_raw
    assert details["profit_factor_capped"] == 5.0


def test_negative_profit_factor_not_capped():
    profit = -100.0
    loss = 10.0
    pf_raw = profit / loss
    stats = {
        "sortino": 1.0,
        "profit_factor": pf_raw,
        "max_drawdown": 10.0,
        "trades": 5,
        "total_return": 1.0,
    }
    settings = {
        "winsorize_pf_cap": 5.0,
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 0,
        "lambda_dispersion": 0.0,
    }
    ev = _make_evaluator(stats, settings)
    ev(None, [], 0)
    details = ev.last_details["per_asset"]["A"]
    assert details["profit_factor"] == pf_raw
    assert details["profit_factor_capped"] == pf_raw


def test_nan_profit_factor_fallback():
    pf_raw = float("nan")
    stats = {
        "sortino": 1.0,
        "profit_factor": pf_raw,
        "max_drawdown": 10.0,
        "trades": 5,
        "total_return": 1.0,
    }
    settings = {
        "winsorize_pf_cap": 5.0,
        "nan_fallback": -1.0,
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 0,
        "lambda_dispersion": 0.0,
    }
    ev = _make_evaluator(stats, settings)
    ev(None, [], 0)
    details = ev.last_details["per_asset"]["A"]
    assert np.isnan(details["profit_factor"])
    assert details["profit_factor_capped"] == -1.0
