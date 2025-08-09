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


def test_process_strategy_rules_with_vectorbt_multiindex(monkeypatch):
    """RSI from vectorbt returns MultiIndex columns; ensure they are flattened."""
    dates = pd.date_range('2020-01-01', periods=3, freq='D')
    base = pd.DataFrame({
        'Open': [1, 1, 1],
        'High': [1, 1, 1],
        'Low': [1, 1, 1],
        'Close': [1, 2, 3],
        'Volume': [100, 100, 100],
    }, index=dates)
    # Portfolio-style multi-index columns
    ohlc = pd.concat({'AAA': base, 'BBB': base}, axis=1)

    class DummyRSI:
        @staticmethod
        def run(close, window):
            cols = pd.MultiIndex.from_product([[window], close.columns], names=['rsi_window', None])
            data = pd.DataFrame(60, index=close.index, columns=cols)
            return types.SimpleNamespace(rsi=data)

    monkeypatch.setattr(indicator_library, 'vbt', types.SimpleNamespace(RSI=DummyRSI))
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, 'rsi', indicator_library.calculate_rsi)

    rules = {
        'entry_rules': {
            'combination_logic': 'AND',
            'conditions': [
                {
                    'indicator': 'rsi',
                    'params': {'period': 14},
                    'condition': {'type': 'indicator_is_above_value', 'value': 50},
                }
            ],
        }
    }

    signal = strategy_engine.process_strategy_rules(ohlc, rules)

    assert list(signal.columns) == ['AAA', 'BBB']
    assert signal.all().all()
