import sys
import types
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy optional dependencies
sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))
sys.modules.setdefault('vectorbt', types.ModuleType('vectorbt'))

import main  # noqa: E402


def test_asset_list_normalized(monkeypatch):
    df_ok = pd.DataFrame(
        {
            'Open': [1, 1, 1, 1, 1],
            'High': [1, 1, 1, 1, 1],
            'Low': [1, 1, 1, 1, 1],
            'Close': [1, 1, 1, 1, 1],
            'Volume': [1, 1, 1, 1, 1],
        },
        index=pd.date_range('2020-01-01', periods=5),
    )
    df_short = df_ok.iloc[:2]

    def load_group_data_stub(group, start, end, interval):
        if start == 'train':
            return {'A': df_ok, 'B': df_short}
        else:
            return {'A': df_ok}

    monkeypatch.setattr(main.data_loader, 'load_group_data', load_group_data_stub)

    gene_space = [{'low': 0, 'high': 1}]
    gene_map = {0: {'name': 'x', 'path': [], 'type': float}}
    gene_types = [float]
    monkeypatch.setattr(main, 'parse_genes_from_config', lambda *a, **k: (gene_space, gene_map, gene_types))

    class DummyGA:
        def __init__(self, *a, **k):
            self.num_generations = 1
            self.generations_completed = 1
            self.last_generation_fitness = [1.0]

        def run(self):
            return None

        def best_solution(self, **_):
            return [0], 1.0, None

        def plot_fitness(self):
            return None

    monkeypatch.setattr(main.pygad, 'GA', DummyGA)
    monkeypatch.setattr(main.analysis, 'run_champion_analysis_multi', lambda *a, **k: None)
    monkeypatch.setattr(main.analysis, 'run_champion_analysis', lambda *a, **k: None)

    monkeypatch.setattr(main.config, 'ASSET_GROUP', [('A', 'A'), ('B', 'B')], raising=False)
    monkeypatch.setattr(main.config, 'MIN_BARS', 3, raising=False)
    monkeypatch.setattr(main.config, 'ENABLE_WALK_FORWARD_VALIDATION', False, raising=False)
    monkeypatch.setattr(main.config, 'WALK_FORWARD_SETTINGS', {'enabled': False}, raising=False)
    monkeypatch.setattr(main.config, 'FITNESS_WEIGHTS', {'min_trades': 0}, raising=False)
    monkeypatch.setattr(main.config, 'GA_NUM_GENERATIONS', 1, raising=False)
    monkeypatch.setattr(main.config, 'GA_POPULATION_SIZE', 1, raising=False)
    monkeypatch.setattr(main.config, 'GA_PARENTS_MATING', 1, raising=False)
    monkeypatch.setattr(main.config, 'GA_MUTATION_NUM_GENES', 1, raising=False)
    monkeypatch.setattr(main.config, 'TRAINING_PERIOD', {'start': 'train', 'end': 't_end'}, raising=False)
    monkeypatch.setattr(main.config, 'VALIDATION_PERIOD', {'start': 'val', 'end': 'v_end'}, raising=False)
    monkeypatch.setattr(main.config, 'TIMEFRAME', '1d', raising=False)
    monkeypatch.setattr(main.config, 'SELECTED_ASSET_NAME', 'Test', raising=False)
    monkeypatch.setattr(main.config, 'TICKER', 'TEST', raising=False)

    main.main()

    assert main.config.ASSET_GROUP == [('A', 'A')]
