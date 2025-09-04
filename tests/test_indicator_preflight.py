import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy dependencies
sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))
sys.modules.setdefault("vectorbt", types.ModuleType("vectorbt"))

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
    monkeypatch.setattr(main, "ensure_real_vectorbt", lambda *a, **k: None)
    monkeypatch.setattr(
        main.analysis, "_write_run_metadata", lambda *a, **k: None, raising=False
    )
    monkeypatch.setattr(
        strategy_engine, "process_strategy_rules", lambda *a, **k: pd.Series([1, 0, 1])
    )
    main.indicator_preflight(data, rules)


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
