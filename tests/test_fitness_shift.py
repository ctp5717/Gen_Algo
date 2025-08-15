import sys
from pathlib import Path

import pandas as pd
import vectorbt as vbt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fitness  # noqa: E402
import config  # noqa: E402


def test_entries_shifted(monkeypatch):
    idx = pd.date_range('2020', periods=4, freq='D')
    ohlc = pd.DataFrame({'Close': [100, 110, 120, 130]}, index=idx)

    def fake_process(df, rules):
        return pd.Series([True, False, False, False], index=df.index)

    monkeypatch.setattr(fitness.engine, 'process_strategy_rules', fake_process)

    captured = []
    orig_from_signals = vbt.Portfolio.from_signals

    def capture_entries(*args, **kwargs):
        entries = kwargs.get('entries') if 'entries' in kwargs else args[1]
        captured.append(entries.copy())
        return orig_from_signals(*args, **kwargs)

    monkeypatch.setattr(vbt.Portfolio, 'from_signals', capture_entries)

    orig_hold = config.MAX_HOLD_PERIOD
    config.MAX_HOLD_PERIOD = 1
    orig_min_trades = config.FITNESS_WEIGHTS['min_trades']
    config.FITNESS_WEIGHTS['min_trades'] = 0

    evaluator = fitness.FitnessEvaluator(ohlc, {}, {})
    evaluator(None, [], 0)

    used_entries = captured[0]
    assert used_entries.index[used_entries].tolist() == [idx[1]]

    config.MAX_HOLD_PERIOD = orig_hold
    config.FITNESS_WEIGHTS['min_trades'] = orig_min_trades
