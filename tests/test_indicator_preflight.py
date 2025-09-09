import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy dependencies
sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))

import analysis  # noqa: E402
import config  # noqa: E402
import fitness  # noqa: E402
import indicator_library  # noqa: E402
import main  # noqa: E402
import strategy_engine  # noqa: E402


def test_indicator_preflight_failure(monkeypatch):
    data = pd.DataFrame({"Close": [1, 2, 3]})
    rules = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "ema",
                    "params": {},
                    "condition": {"type": "price_is_above_indicator"},
                }
            ]
        }
    }

    monkeypatch.setattr(main, "ensure_real_vectorbt", lambda *a, **k: None)
    monkeypatch.setattr(
        main.analysis, "_write_run_metadata", lambda *a, **k: None, raising=False
    )
    monkeypatch.setitem(
        strategy_engine.INDICATOR_MAPPING,
        "ema",
        lambda df, **p: pd.DataFrame({"EMA": [1, 2, 3]}),
    )

    def fake_process(*a, **k):  # noqa: ANN001, ANN002
        raise KeyError("EMA column missing")

    monkeypatch.setattr(strategy_engine, "process_strategy_rules", fake_process)
    with pytest.raises(SystemExit):
        main.indicator_preflight(data, rules)


def test_indicator_preflight_combination_logic_gene_dict(monkeypatch):
    data = pd.DataFrame({"Close": [1, 2, 3]})
    rules = {
        "entry_rules": {
            "combination_logic": {"name": "cl", "options": ["AND", "OR"]},
            "conditions": [],
        }
    }

    a = pd.Series([True, False, True], index=data.index)
    b = pd.Series([False, True, True], index=data.index)

    captured = {"logics": []}

    def fake_process(ohlc, rule_set):  # noqa: ANN001
        logic = rule_set["entry_rules"].get("combination_logic", "AND")
        captured["logics"].append(logic)
        return a | b if logic == "OR" else a & b

    monkeypatch.setattr(main, "ensure_real_vectorbt", lambda *a, **k: None)
    monkeypatch.setattr(
        main.analysis, "_write_run_metadata", lambda *a, **k: None, raising=False
    )
    monkeypatch.setattr(strategy_engine, "process_strategy_rules", fake_process)

    main.indicator_preflight(data, rules)
    # preflight should not mutate the original gene dict
    assert isinstance(rules["entry_rules"]["combination_logic"], dict)

    gene_map = {0: {"path": ["entry_rules", "combination_logic"]}}
    evaluator = fitness.FitnessEvaluator(data, rules, gene_map)

    trade_counts = []

    class DummyPF:
        def __init__(self, trades):
            self._trades = types.SimpleNamespace(count=lambda: trades)

        def stats(self):
            return {
                "Sortino Ratio": 1.0,
                "Profit Factor": 1.0,
                "Max Drawdown [%]": 0.0,
            }

    def fake_from_signals(**kwargs):  # noqa: ANN003
        trades = int(kwargs["entries"].sum())
        trade_counts.append(trades)
        return DummyPF(trades)

    monkeypatch.setattr(
        fitness.vbt.Portfolio, "from_signals", fake_from_signals, raising=False
    )

    evaluator(None, ["OR"], 0)

    assert captured["logics"] == ["AND", "OR"]
    assert trade_counts == [3]


def test_indicator_preflight_logs_and_metadata(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    data = pd.DataFrame({"Close": [1, 2, 3]})
    rules = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "ema",
                    "params": {},
                    "condition": {"type": "indicator_is_above_value", "value": 0},
                },
                {
                    "indicator": "bbands",
                    "params": {},
                    "condition": {"type": "price_is_above_indicator"},
                },
            ]
        }
    }

    monkeypatch.setattr(main, "ensure_real_vectorbt", lambda *a, **k: None)

    def ema(df, **p):
        return pd.Series([1, 2, 3], name="EMA")

    def bb(df, **p):
        return pd.DataFrame({"u": [1, 1, 1], "l": [0, 0, 0]})

    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "ema", ema)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "bbands", bb)
    monkeypatch.setattr(strategy_engine, "process_strategy_rules", lambda *a, **k: None)

    captured = {}

    def fake_write(start, arts, extra=None):
        captured.update(extra or {})

    monkeypatch.setattr(main.analysis, "_write_run_metadata", fake_write, raising=False)

    main.indicator_preflight(data, rules)
    out = capsys.readouterr().out
    assert "ema: (Series)" in out
    assert "bbands: ['u', 'l']" in out
    ic = captured["indicator_columns"]
    assert ic["ema"]["type"] == "Series"
    assert ic["bbands"]["type"] == "DataFrame"
    assert captured["preflight_all"] is False
    assert captured["preflight_sample_len"] == 3


def test_indicator_preflight_all_unused_failures_recorded(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setattr(config, "PREFLIGHT_ALL_INDICATORS", True, raising=False)
    analysis.set_run_dir(tmp_path)
    data = pd.DataFrame({"Close": [1, 2], "Volume": [1, 1]})

    def good(df, **p):
        return pd.Series([1, 2])

    def bad(df, **p):
        raise ValueError("boom")

    monkeypatch.setattr(
        indicator_library, "INDICATOR_REGISTRY", {"good": good, "bad": bad}
    )
    monkeypatch.setattr(
        strategy_engine, "INDICATOR_MAPPING", {"good": good, "bad": bad}
    )
    captured = {}

    def fake_write(start, arts, extra=None):
        captured.update(extra or {})

    monkeypatch.setattr(main, "analysis", analysis, raising=False)
    monkeypatch.setattr(main.analysis, "_write_run_metadata", fake_write, raising=False)

    main.indicator_preflight(data, {"entry_rules": {"conditions": []}})

    out = capsys.readouterr().out
    assert "bad failed" in out
    results = captured["indicator_results"]
    assert not results["bad"]["success"]
    assert "ValueError" in results["bad"]["error"]
    assert results["good"]["success"]


def test_indicator_preflight_all_active_failure_exits(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(config, "PREFLIGHT_ALL_INDICATORS", True, raising=False)
    analysis.set_run_dir(tmp_path)
    data = pd.DataFrame({"Close": [1, 2], "Volume": [1, 1]})

    def bad(df, **p):
        raise ValueError("boom")

    monkeypatch.setattr(
        indicator_library, "INDICATOR_REGISTRY", {"bad": bad}, raising=False
    )
    monkeypatch.setattr(
        strategy_engine, "INDICATOR_MAPPING", {"bad": bad}, raising=False
    )
    monkeypatch.setattr(main, "analysis", analysis, raising=False)
    monkeypatch.setattr(
        main.analysis, "_write_run_metadata", lambda *a, **k: None, raising=False
    )

    with pytest.raises(SystemExit):
        main.indicator_preflight(
            data,
            {
                "entry_rules": {
                    "conditions": [
                        {
                            "indicator": "bad",
                            "params": {},
                            "condition": {
                                "type": "indicator_is_above_value",
                                "value": 0,
                            },
                        }
                    ]
                }
            },
        )

    out = capsys.readouterr().out
    assert "bad failed" in out


def test_indicator_preflight_all_success(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PREFLIGHT_ALL_INDICATORS", True, raising=False)
    analysis.set_run_dir(tmp_path)
    data = pd.DataFrame({"Close": [1, 2], "Volume": [1, 1]})

    def good(df, **p):
        return pd.Series([1, 2])

    monkeypatch.setattr(indicator_library, "INDICATOR_REGISTRY", {"good": good})
    monkeypatch.setattr(strategy_engine, "INDICATOR_MAPPING", {"good": good})
    captured = {}

    def fake_write(start, arts, extra=None):
        captured.update(extra or {})

    monkeypatch.setattr(main, "analysis", analysis, raising=False)
    monkeypatch.setattr(main.analysis, "_write_run_metadata", fake_write, raising=False)

    main.indicator_preflight(data, {"entry_rules": {"conditions": []}})
    assert captured["indicator_results"]["good"]["success"]
