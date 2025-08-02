import sys
import types
from pathlib import Path
import pandas as pd

# Ensure repository root is on the import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy optional dependencies before importing modules
sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))
sys.modules.setdefault('vectorbt', types.ModuleType('vectorbt'))

import indicator_library  # noqa: E402
import strategy_engine  # noqa: E402


def test_process_strategy_rules_simple(monkeypatch):
    data = pd.DataFrame({
        'Open': [1, 2, 3, 4, 5],
        'High': [1, 2, 3, 4, 5],
        'Low': [1, 2, 3, 4, 5],
        'Close': [10, 11, 12, 13, 14],
        'Volume': [100, 100, 100, 100, 100],
    }, index=pd.date_range('2020-01-01', periods=5, freq='D'))

    # Patch indicator calculations with simple deterministic series
    def ema_func(ohlc, period):
        return ohlc['Close'] - 1

    def rsi_func(ohlc, period):
        return pd.Series(60, index=ohlc.index)
    monkeypatch.setattr(indicator_library, 'calculate_ema', ema_func)
    monkeypatch.setattr(indicator_library, 'calculate_rsi', rsi_func)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, 'ema', ema_func)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, 'rsi', rsi_func)

    rules = {
        'entry_rules': {
            'combination_logic': 'AND',
            'conditions': [
                {
                    'indicator': 'ema',
                    'params': {'period': 3},
                    'condition': {'type': 'price_is_above_indicator'}
                },
                {
                    'indicator': 'rsi',
                    'params': {'period': 2},
                    'condition': {
                        'type': 'indicator_is_above_value',
                        'value': 50
                    }
                },
            ]
        }
    }

    signal = strategy_engine.process_strategy_rules(data, rules)

    assert signal.all()


def _multi_ohlc():
    base = pd.DataFrame(
        {
            "Open": [1, 2, 3],
            "High": [1, 2, 3],
            "Low": [1, 2, 3],
            "Close": [10, 11, 12],
            "Volume": [100, 100, 100],
        },
        index=pd.date_range("2020-01-01", periods=3),
    )
    return pd.concat({"AAA": base, "BBB": base}, axis=1)


def test_process_strategy_rules_multiindex_alignment(monkeypatch):
    df = _multi_ohlc()

    def ema_func(ohlc, period):
        frames = []
        for tk in ohlc.columns.get_level_values(0).unique():
            ema = ohlc[tk]["Close"] - 1
            frames.append(pd.DataFrame({(tk, "EMA"): ema}))
        return pd.concat(frames, axis=1)

    monkeypatch.setattr(indicator_library, "calculate_ema", ema_func)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "ema", ema_func)

    rules = {
        "entry_rules": {
            "combination_logic": "AND",
            "conditions": [
                {
                    "indicator": "ema",
                    "params": {"period": 2},
                    "condition": {"type": "price_is_above_indicator"},
                }
            ],
        }
    }

    signal = strategy_engine.process_strategy_rules(df, rules)

    assert isinstance(signal, pd.DataFrame)
    assert list(signal.columns) == ["AAA", "BBB"]
    assert signal.all().all()
