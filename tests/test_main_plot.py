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


def test_plot_shows_and_pauses(monkeypatch):
    df = pd.DataFrame(
        {
            'Open': [1, 2],
            'High': [1, 2],
            'Low': [1, 2],
            'Close': [1, 2],
            'Volume': [100, 100],
        },
        index=pd.date_range('2020-01-01', periods=2),
    )

    monkeypatch.setattr(main.data_loader, 'get_data', lambda *a, **k: df)

    gene_space = [{'low': 0, 'high': 1}]
    gene_map = {0: {'name': 'x', 'path': [], 'type': float}}
    gene_types = [float]
    monkeypatch.setattr(
        main, 'parse_genes_from_config', lambda *a, **k: (gene_space, gene_map, gene_types)
    )

    class DummyGA:
        def __init__(self, *args, **kwargs):
            self.num_generations = 1
            self.generations_completed = 1
            self.last_generation_fitness = [1.0]
            self.best_solutions_fitness = [1.0]

        def run(self):
            return None

        def best_solution(self, **kwargs):
            return [0], 1.0, None

    monkeypatch.setattr(main.pygad, 'GA', DummyGA)
    monkeypatch.setattr(main.analysis, 'run_champion_analysis', lambda *a, **k: None)

    class DummyEvaluator:
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            return 1.0

    monkeypatch.setattr(main.fitness, 'FitnessEvaluator', DummyEvaluator)

    monkeypatch.setattr(main.config, 'ENABLE_WALK_FORWARD_VALIDATION', False, raising=False)
    monkeypatch.setattr(main.config, 'WALK_FORWARD_SETTINGS', {'enabled': False}, raising=False)
    monkeypatch.setattr(main.config, 'FITNESS_WEIGHTS', {'min_trades': 0}, raising=False)
    monkeypatch.setattr(main.config, 'GA_NUM_GENERATIONS', 1, raising=False)
    monkeypatch.setattr(main.config, 'GA_POPULATION_SIZE', 1, raising=False)
    monkeypatch.setattr(main.config, 'GA_PARENTS_MATING', 1, raising=False)
    monkeypatch.setattr(main.config, 'GA_MUTATION_NUM_GENES', 1, raising=False)
    train_period = {'start': '2020-01-01', 'end': '2020-01-02'}
    valid_period = {'start': '2020-01-02', 'end': '2020-01-03'}
    monkeypatch.setattr(main.config, 'TRAINING_PERIOD', train_period, raising=False)
    monkeypatch.setattr(main.config, 'VALIDATION_PERIOD', valid_period, raising=False)
    monkeypatch.setattr(main.config, 'SELECTED_ASSET_NAME', 'Test', raising=False)
    monkeypatch.setattr(main.config, 'TICKER', 'TEST', raising=False)
    monkeypatch.setattr(main.config, 'TIMEFRAME', '1d', raising=False)

    calls = {}

    def fake_show(*args, **kwargs):
        calls['show'] = kwargs

    def fake_pause(interval):
        calls['pause'] = interval

    monkeypatch.setattr(
        main,
        'plt',
        types.SimpleNamespace(
            ion=lambda: None,
            plot=lambda *a, **k: None,
            xlabel=lambda *a, **k: None,
            ylabel=lambda *a, **k: None,
            legend=lambda *a, **k: None,
            show=fake_show,
            pause=fake_pause,
        ),
    )

    main.main()

    assert calls.get('show', {}).get('block') is False
    assert calls.get('pause') == 0.001
