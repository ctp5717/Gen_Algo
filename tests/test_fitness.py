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


def test_composite_score_matches_helper(monkeypatch):
    ohlc = pd.DataFrame({"Close": [1.0, 1.1, 1.2]})
    entries = pd.Series([True, False, False], index=ohlc.index)

    monkeypatch.setattr(
        fitness.engine, "process_strategy_rules", lambda *a, **k: entries
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
    )

    assert score == pytest.approx(expected)
