import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repository root is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fitness


def test_fitness_handles_multi_asset_stats(monkeypatch):
    # Create sample OHLC data for two assets with MultiIndex columns
    index = pd.date_range('2024', periods=3, freq='D')
    columns = pd.MultiIndex.from_product([
        ['A', 'B'], ['Open', 'High', 'Low', 'Close', 'Volume']
    ])
    # Populate with deterministic values
    data = pd.DataFrame(
        np.tile(np.arange(1, 11), (3, 1)), index=index, columns=columns
    )

    # Provide entry signals for each asset
    entries = pd.DataFrame({'A': [1, 0, 0], 'B': [1, 0, 0]}, index=index).astype(bool)

    # Return the prepared entries regardless of rules
    monkeypatch.setattr(
        fitness.engine,
        'process_strategy_rules',
        lambda ohlc, rules: entries,
    )

    class DummyPortfolio:
        def stats(self, agg_func=None):
            # Simulate per-asset stats coming from vectorbt
            return pd.DataFrame(
                {
                    'A': {
                        'Sortino Ratio': 1.0,
                        'Profit Factor': 1.5,
                        'Max Drawdown [%]': 10,
                        'Total Trades': 1,
                    },
                    'B': {
                        'Sortino Ratio': 1.2,
                        'Profit Factor': 1.2,
                        'Max Drawdown [%]': 8,
                        'Total Trades': 1,
                    },
                }
            )

    # Vectorbt's from_signals will return our dummy portfolio
    monkeypatch.setattr(
        fitness.vbt.Portfolio,
        'from_signals',
        lambda **kwargs: DummyPortfolio(),
    )

    # Simplify configuration to avoid trade-count penalties and long holds
    monkeypatch.setattr(fitness.config, 'MAX_HOLD_PERIOD', 1)
    monkeypatch.setattr(
        fitness.config,
        'FITNESS_WEIGHTS',
        {'sortino_ratio': 0.5, 'profit_factor': 0.3, 'max_drawdown': 0.2, 'min_trades': 1},
    )

    evaluator = fitness.FitnessEvaluator(data, {}, {})
    score = evaluator(None, [], 0)

    # With the fix, the evaluation should produce a positive score
    assert score > 0
