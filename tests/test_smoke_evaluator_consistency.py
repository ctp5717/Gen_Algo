import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub optional heavy dependencies
dummy_ta = types.ModuleType('pandas_ta')
sys.modules.setdefault('pandas_ta', dummy_ta)

dummy_vbt = types.ModuleType('vectorbt')
class DummyPortfolio:
    @classmethod
    def from_signals(cls, **kwargs):
        return cls()
    def stats(self):
        return {
            'Sortino Ratio': 1.0,
            'Profit Factor': 1.0,
            'Max Drawdown [%]': 0.0,
        }
dummy_vbt.Portfolio = DummyPortfolio
sys.modules.setdefault('vectorbt', dummy_vbt)

import pandas as pd  # noqa: E402
import fitness  # noqa: E402
import tuner  # noqa: E402


def test_tuner_and_ga_consistency(monkeypatch):
    df = pd.DataFrame(
        {
            'Open': [1, 1],
            'High': [1, 1],
            'Low': [1, 1],
            'Close': [1, 1],
            'Volume': [1, 1],
        },
        index=pd.date_range('2020-01-01', periods=2),
    )

    # Ensure pandas_ta attribute exists
    monkeypatch.setattr(pd.DataFrame, 'ta', property(lambda self: None), raising=False)

    # Simplified strategy rule evaluation
    entries = pd.Series([True, False], index=df.index)
    monkeypatch.setattr(fitness.engine, 'process_strategy_rules', lambda data, rules: entries)
    monkeypatch.setattr(tuner.engine, 'process_strategy_rules', lambda data, rules: entries)

    gene_map = {0: {'name': 'x', 'path': [], 'type': float}}
    solution = [0.5]

    evaluator = fitness.FitnessEvaluator(df, {}, gene_map)
    ga_score = evaluator(None, solution, 0)

    monkeypatch.setattr(tuner.data_loader, 'get_data', lambda **kwargs: df)
    monkeypatch.setattr(tuner.config, 'TICKER', 'T', raising=False)
    monkeypatch.setattr(
        tuner.config,
        'VALIDATION_PERIOD',
        {'start': '2020-01-01', 'end': '2020-01-02'},
        raising=False,
    )

    tuner_score = tuner._evaluate_on_validation(solution, gene_map)
    assert ga_score == tuner_score
