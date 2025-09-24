import sys
import types
from pathlib import Path

# Ensure repository root is on the import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy optional dependencies before importing modules that use them
sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))
sys.modules.setdefault("vectorbt", types.ModuleType("vectorbt"))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import fitness  # noqa: E402


def test_exception_logging(capsys, monkeypatch):
    """FitnessEvaluator prints exception messages"""
    ohlc = pd.DataFrame({"Close": [1, 2, 3]})
    evaluator = fitness.FitnessEvaluator(ohlc, {}, {})

    def raise_error(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(fitness.engine, "process_strategy_rules", raise_error)

    score = evaluator(None, [], 0)

    captured = capsys.readouterr()
    assert "boom" in captured.out
    assert score == -999.0


def test_invalid_exit_config_penalised(monkeypatch):
    ohlc = pd.DataFrame({"Close": [100.0, 101.0, 102.0]})
    entries = pd.Series([True, False, False], index=ohlc.index)

    monkeypatch.setattr(
        fitness.engine, "process_strategy_rules", lambda *a, **k: entries
    )
    monkeypatch.setitem(fitness.config.FITNESS_WEIGHTS, "min_trades", 0)
    monkeypatch.setattr(
        fitness.config, "USE_DYNAMIC_EXIT_SIMULATOR", True, raising=False
    )
    monkeypatch.setattr(fitness.config, "MAX_HOLD_PERIOD", 10, raising=False)
    monkeypatch.setattr(fitness.config, "TIMEFRAME", "1h", raising=False)

    invalid_rules = {
        "exit_rules": {
            "stop_loss": {"params": {"value": 0.05}},
            "trade_management": {
                "num_tp_levels": 4,
                "tp_pct_cap": 0.01,
            },
        }
    }

    evaluator = fitness.FitnessEvaluator(ohlc, invalid_rules, {})
    score = evaluator(None, [], 0)

    assert score == -999.0


def test_composite_score_matches_helper(monkeypatch):
    ohlc = pd.DataFrame({"Close": [1.0, 1.1, 1.2]})
    entries = pd.Series([True, False, False], index=ohlc.index)

    monkeypatch.setattr(
        fitness.engine, "process_strategy_rules", lambda *a, **k: entries
    )
    monkeypatch.setattr(
        fitness.config, "USE_DYNAMIC_EXIT_SIMULATOR", False, raising=False
    )
    empty = pd.Series(False, index=ohlc.index)
    monkeypatch.setattr(
        fitness,
        "extract_exit_params",
        lambda *a, **k: (empty, None, None, None),
    )

    class DummyPortfolio:
        def __init__(self):
            self.trades = types.SimpleNamespace(count=lambda: 3)

        @classmethod
        def from_signals(cls, *args, **kwargs):
            return cls()

    monkeypatch.setattr(fitness.vbt, "Portfolio", DummyPortfolio)
    monkeypatch.setattr(
        fitness.metrics_contract, "assert_metric_aliases", lambda *a, **k: None
    )
    monkeypatch.setattr(
        fitness.metrics_contract,
        "_provider_signature",
        lambda *a, **k: "dummy",
    )

    metrics_payload = {
        "sortino": 0.8,
        "profit_factor": 7.0,
        "max_drawdown": 12.5,
    }

    def fake_metrics(portfolio):
        return (metrics_payload, {}, [])

    monkeypatch.setattr(fitness.metrics_contract, "evaluate_metrics", fake_metrics)

    evaluator = fitness.FitnessEvaluator(ohlc, {}, {})
    score = evaluator(None, [], 0)

    cap = getattr(fitness.config, "MULTI_ASSET", {}).get("winsorize_pf_cap", 5.0)
    expected = fitness._composite_score(
        metrics_payload["sortino"],
        metrics_payload["profit_factor"],
        metrics_payload["max_drawdown"],
        weights=fitness.config.FITNESS_WEIGHTS,
        pf_cap=cap,
        nan_fallback=0.0,
        max_drawdown_fallback=100.0,
        exit_usage=None,
        exit_weights=getattr(fitness.config, "FITNESS_EXIT_USAGE", None),
    )

    assert score == pytest.approx(expected)


def test_composite_score_penalizes_high_timeout():
    weights = {"sortino_ratio": 0.5, "profit_factor": 0.3, "max_drawdown": 0.2}
    exit_weights = {
        "timeout_weight": 1.0,
        "timeout_target": 0.2,
        "tp_hit_weight": 0.0,
        "tp_hit_target": 0.0,
        "avg_tp_level_weight": 0.0,
        "avg_tp_level_target": 0.0,
        "trailing_tp_weight": 0.0,
        "trailing_tp_target": 0.0,
    }
    base = fitness._composite_score(
        1.0,
        2.0,
        10.0,
        weights=weights,
        pf_cap=5.0,
        nan_fallback=0.0,
        max_drawdown_fallback=100.0,
        exit_usage=None,
        exit_weights=exit_weights,
    )
    penalised = fitness._composite_score(
        1.0,
        2.0,
        10.0,
        weights=weights,
        pf_cap=5.0,
        nan_fallback=0.0,
        max_drawdown_fallback=100.0,
        exit_usage={"sl_timeout_usage_rate": 0.8, "trades_evaluated": 10.0},
        exit_weights=exit_weights,
    )
    assert penalised < base


def test_composite_score_penalizes_missing_tp_hits():
    weights = {"sortino_ratio": 0.5, "profit_factor": 0.3, "max_drawdown": 0.2}
    exit_weights = {
        "timeout_weight": 0.0,
        "timeout_target": 1.0,
        "tp_hit_weight": 0.8,
        "tp_hit_target": 0.5,
        "avg_tp_level_weight": 0.2,
        "avg_tp_level_target": 1.5,
        "trailing_tp_weight": 0.0,
        "trailing_tp_target": 0.0,
    }
    healthy = fitness._composite_score(
        0.8,
        2.0,
        12.0,
        weights=weights,
        pf_cap=5.0,
        nan_fallback=0.0,
        max_drawdown_fallback=100.0,
        exit_usage={
            "trades_evaluated": 10.0,
            "tp_trades_evaluated": 6.0,
            "avg_tp_level_reached": 2.0,
        },
        exit_weights=exit_weights,
    )
    poor = fitness._composite_score(
        0.8,
        2.0,
        12.0,
        weights=weights,
        pf_cap=5.0,
        nan_fallback=0.0,
        max_drawdown_fallback=100.0,
        exit_usage={
            "trades_evaluated": 10.0,
            "tp_trades_evaluated": 0.0,
            "avg_tp_level_reached": 0.5,
        },
        exit_weights=exit_weights,
    )
    assert poor < healthy
