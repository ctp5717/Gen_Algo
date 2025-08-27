import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))
sys.modules.setdefault('vectorbt', types.ModuleType('vectorbt'))

import json
import numpy as np
import pandas as pd
import walk_forward
import config


def test_walk_forward_determinism(monkeypatch, tmp_path):
    """Repeated walk-forward runs with fixed seeds yield identical results."""
    monkeypatch.chdir(tmp_path)

    index = pd.date_range('2020-01-01', periods=6, freq='D')
    data = pd.DataFrame(
        {
            'Open': 1,
            'High': 1,
            'Low': 1,
            'Close': 1,
            'Volume': 1,
        },
        index=index,
    )
    group_data = {'AAA': data, 'BBB': data}

    # Return preloaded data to simulate cache usage
    monkeypatch.setattr(
        walk_forward.data_loader,
        'get_group_data',
        lambda *a, **k: group_data,
    )

    periods = [
        {
            'train_start': index[0],
            'train_end': index[2],
            'test_start': index[2],
            'test_end': index[4],
        },
        {
            'train_start': index[1],
            'train_end': index[3],
            'test_start': index[3],
            'test_end': index[5],
        },
    ]
    monkeypatch.setattr(walk_forward, '_generate_periods', lambda *a, **k: periods)

    monkeypatch.setattr(walk_forward, 'parse_genes_from_config', lambda *a, **k: ([], {}, []))

    class DummyEval:
        def __init__(self, *a, **k):
            self.last_details = {
                'mu': 0.1,
                'sigma': 0.2,
                'lambda_sigma': 0.05,
                'total_trades': 4,
                'assets_included': 2,
                'assets_traded': 2,
                'min_total_trades': 1,
                'penalties': {'coverage': 0.0},
                'per_asset': {
                    'AAA': {'score': 0.5, 'trades': 2, 'included': True},
                    'BBB': {'score': 0.6, 'trades': 2, 'included': True},
                },
            }

        def __call__(self, *a, **k):
            return 1.0

    monkeypatch.setattr(walk_forward.fitness, 'MultiAssetFitnessEvaluator', DummyEval)

    class DummyGA:
        def __init__(self, *a, **k):
            self.population = np.zeros((1, 0))
            self.initial_population = self.population.copy()

        def run(self):
            pass

        def best_solution(self, **k):
            return [], 1.0, None

    monkeypatch.setattr(walk_forward.pygad, 'GA', DummyGA)

    def run_once():
        np.random.seed(config.SEED)
        summary = walk_forward.run_walk_forward_validation()
        with open('walk_forward_summary.json', 'r') as fh:
            blob = json.load(fh)
        cols = [
            'Fitness',
            'Mu',
            'Sigma',
            'Lambda Sigma',
            'Total Trades',
            'Assets Included',
            'Assets Traded',
        ]
        df = summary['folds'][cols].reset_index(drop=True)
        return df, blob

    df1, j1 = run_once()
    df2, j2 = run_once()

    pd.testing.assert_frame_equal(df1, df2)
    assert j1 == j2
