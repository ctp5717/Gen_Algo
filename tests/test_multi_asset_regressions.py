import copy
import json
import sys
import types
from collections import Counter

import numpy as np
import pandas as pd
import pytest

try:
    import vectorbt  # noqa: F401
except Exception:  # pragma: no cover
    sys.modules.setdefault("vectorbt", types.ModuleType("vectorbt"))

import analysis
import config
import fitness
from utils.math import weighted_mean_std

config.initialize_config()


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


def _make_evaluator(stats_list, settings=None, group_data=None):
    group_data = group_data or {
        "A": pd.DataFrame({"Close": [1, 2, 3]}),
        "B": pd.DataFrame({"Close": [1, 2, 3]}),
        "C": pd.DataFrame({"Close": [1, 2, 3]}),
    }
    base = {
        "per_asset_min_trades": 1,
        "min_included_assets": 1,
        "coverage_penalty": 0.0,
    }
    if settings:
        base.update(settings)
    evaluator = fitness.MultiAssetFitnessEvaluator(group_data, {}, {}, base)
    stats_iter = iter(stats_list)

    def fake_eval(self, ohlc, rules):
        return next(stats_iter)

    evaluator._evaluate_single_asset = types.MethodType(fake_eval, evaluator)
    return evaluator


def test_weighted_aggregation_regression():
    vals = [1.6, 1.0, 0.4]
    w = [1 / 3, 1 / 3, 1 / 3]
    mu, sigma = weighted_mean_std(vals, w)
    lam = 0.25
    F = mu - lam * sigma
    assert np.isclose(mu, 1.0)
    assert np.isclose(sigma, 0.4899, atol=1e-4)
    assert np.isclose(F, 0.8775, atol=1e-4)


def test_weights_sum_to_one():
    stats = [
        {"total_return": 1.0, "trades": 5},
        {"total_return": 0.5, "trades": 5},
        {"total_return": 0.0, "trades": 5},
    ]
    settings = {
        "metric": "return",
        "lambda_dispersion": 0.0,
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 0,
        "asset_weights": {"A": 3, "B": 1, "C": 1},
    }
    ev = _make_evaluator(stats, settings)
    ev(None, [], 0)
    details = ev.last_details
    inc = [t for t, d in details["per_asset"].items() if d["included"]]
    total = sum(details["per_asset"][t]["asset_weight"] for t in inc)
    assert np.isclose(total, 1.0)


def test_hard_floor_no_coverage_penalty():
    stats = [
        {"total_return": 1.0, "trades": 1},
        {"total_return": 1.0, "trades": 1},
        {"total_return": 1.0, "trades": 1},
    ]
    settings = {
        "metric": "return",
        "lambda_dispersion": 0.0,
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 10,
        "zero_trade_policy": "ignore",
        "coverage_penalty": 0.3,
        "poor_score": -999.0,
    }
    ev = _make_evaluator(stats, settings)
    score = ev(None, [], 0)
    assert score == -999.0
    assert ev.last_details["penalties"]["coverage"] == 0.0


def test_deterministic_order_and_score():
    group = {
        "B": pd.DataFrame({"Close": [1, 2, 3]}),
        "A": pd.DataFrame({"Close": [1, 2, 3]}),
        "C": pd.DataFrame({"Close": [1, 2, 3]}),
    }
    stats = [
        {"total_return": 1.0, "trades": 5},
        {"total_return": 2.0, "trades": 5},
        {"total_return": 3.0, "trades": 5},
    ]
    settings = {
        "metric": "return",
        "lambda_dispersion": 0.0,
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 0,
    }
    ev1 = _make_evaluator(stats, settings, group)
    f1 = ev1(None, [], 0)
    rows1 = pd.DataFrame(
        [
            {"ticker": t, "score": d["score"]}
            for t, d in ev1.last_details["per_asset"].items()
            if d["score"] is not None
        ]
    ).sort_values("score", ascending=False)
    order1 = rows1["ticker"].tolist()

    ev2 = _make_evaluator(stats, settings, group)
    f2 = ev2(None, [], 0)
    rows2 = pd.DataFrame(
        [
            {"ticker": t, "score": d["score"]}
            for t, d in ev2.last_details["per_asset"].items()
            if d["score"] is not None
        ]
    ).sort_values("score", ascending=False)
    order2 = rows2["ticker"].tolist()

    assert f1 == f2
    assert order1 == order2


def test_csv_and_json_include_exclusions(tmp_path, monkeypatch):
    group = {
        "A": pd.DataFrame({"Close": [1, 2, 3]}),
        "B": pd.DataFrame({"Close": [1, 2, 3]}),
    }
    stats = [
        {"total_return": 0.0, "trades": 0},
        {"total_return": 1.0, "trades": 5},
    ]

    stats_iter = iter(stats)

    def fake_eval(self, ohlc, rules):
        return next(stats_iter)

    monkeypatch.setattr(
        fitness.MultiAssetFitnessEvaluator, "_evaluate_single_asset", fake_eval
    )
    monkeypatch.setitem(config.MULTI_ASSET, "metric", "return")
    monkeypatch.setitem(config.MULTI_ASSET, "lambda_dispersion", 0.0)
    monkeypatch.setitem(config.MULTI_ASSET, "trade_floor_policy", "hard_floor")
    monkeypatch.setitem(config.MULTI_ASSET, "min_total_trades", 0)
    monkeypatch.setitem(config.MULTI_ASSET, "min_total_trades_per_year", 0)
    monkeypatch.setitem(config.MULTI_ASSET, "asset_weights", {"A": 1, "B": 1})
    monkeypatch.setitem(config.MULTI_ASSET, "per_asset_min_trades", 1)
    monkeypatch.setitem(config.MULTI_ASSET, "min_included_assets", 1)
    monkeypatch.setitem(config.MULTI_ASSET, "coverage_penalty", 0.0)
    monkeypatch.setattr(analysis, "_plot_multi_asset_overview", lambda *a, **k: None)

    class _VBT:
        __version__ = "0.0.0"
        __file__ = __file__

    import sys

    monkeypatch.setitem(sys.modules, "vectorbt", _VBT)
    monkeypatch.setattr(analysis, "vbt", _VBT)
    monkeypatch.setattr(analysis, "_write_run_metadata", lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)
    analysis.set_run_dir(tmp_path)

    analysis._run_multi_asset_analysis([], {}, group, [])

    csv_file = next(tmp_path.glob("multi_asset_stats_*.csv"))
    df = pd.read_csv(csv_file)
    assert set(df["ticker"]) == {"A", "B"}
    assert not bool(df.loc[df["ticker"] == "A", "included"].item())
    assert df.loc[df["ticker"] == "A", "reason"].item() != ""
    assert np.isclose(df[df["included"]]["asset_weight"].sum(), 1.0)

    json_file = next(tmp_path.glob("multi_asset_summary_*.json"))
    summary = json.loads(json_file.read_text())
    assert np.isclose(sum(summary["asset_weights"].values()), 1.0)
    assert set(summary["asset_weights"].keys()) == {"B"}
    assert summary["assets_ignored"] == 1


def test_evaluation_error_reason(tmp_path, monkeypatch):
    group = {
        "A": pd.DataFrame({"Close": [1, 2, 3]}),
        "B": pd.DataFrame({"Close": [1, 2, 3]}),
    }
    stats = [Exception("boom"), {"total_return": 1.0, "trades": 5}]

    stats_iter = iter(stats)

    def fake_eval(self, ohlc, rules):
        val = next(stats_iter)
        if isinstance(val, Exception):
            raise val
        return val

    monkeypatch.setattr(
        fitness.MultiAssetFitnessEvaluator, "_evaluate_single_asset", fake_eval
    )
    monkeypatch.setitem(config.MULTI_ASSET, "metric", "return")
    monkeypatch.setitem(config.MULTI_ASSET, "lambda_dispersion", 0.0)
    monkeypatch.setitem(config.MULTI_ASSET, "trade_floor_policy", "hard_floor")
    monkeypatch.setitem(config.MULTI_ASSET, "min_total_trades", 0)
    monkeypatch.setitem(config.MULTI_ASSET, "min_total_trades_per_year", 0)
    monkeypatch.setitem(config.MULTI_ASSET, "asset_weights", {"A": 1, "B": 1})
    monkeypatch.setitem(config.MULTI_ASSET, "per_asset_min_trades", 1)
    monkeypatch.setitem(config.MULTI_ASSET, "min_included_assets", 1)
    monkeypatch.setitem(config.MULTI_ASSET, "coverage_penalty", 0.0)
    monkeypatch.chdir(tmp_path)

    evaluator = fitness.MultiAssetFitnessEvaluator(
        group, {}, {}, dict(config.MULTI_ASSET)
    )
    evaluator(None, [], 0)
    details = evaluator.last_details
    rows = []
    w_map = details.get("asset_weights", {})
    for t in sorted(details["per_asset"]):
        d = details["per_asset"][t]
        rows.append(
            {
                "ticker": t,
                "included": d.get("included", False),
                "asset_weight": w_map.get(t),
                "reason": d.get("reason", ""),
                "reason_detail": d.get("reason_detail", ""),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv("multi_asset_stats_test.csv", index=False)
    row_a = df.loc[df["ticker"] == "A"].iloc[0]
    assert row_a["reason"] == "evaluation_error"
    assert "boom" in row_a["reason_detail"]
    assert np.isclose(df[df["included"]]["asset_weight"].sum(), 1.0)


def test_asset_counts_ordering():
    stats = [
        {"total_return": 1.0, "trades": 0},
        {"total_return": 1.0, "trades": 2},
    ]
    group = {
        "A": pd.DataFrame({"Close": [1, 2, 3]}),
        "B": pd.DataFrame({"Close": [1, 2, 3]}),
    }
    settings = {
        "metric": "return",
        "lambda_dispersion": 0.0,
        "per_asset_min_trades": 1,
        "min_total_trades": 0,
        "trade_floor_policy": "hard_floor",
        "zero_trade_policy": "ignore",
    }
    ev = _make_evaluator(stats, settings, group)
    ev(None, [], 0)
    details = ev.last_details
    included = details.get("assets_included")
    traded = details.get("assets_traded")
    total = len(group)
    assert traded <= included <= total
