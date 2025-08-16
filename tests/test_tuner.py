import sys
import types
from pathlib import Path
import pandas as pd
import pickle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault('vectorbt', types.ModuleType('vectorbt'))
sys.modules.setdefault('binance', types.ModuleType('binance'))
bin_client = types.ModuleType('binance.client')
bin_client.Client = object
sys.modules.setdefault('binance.client', bin_client)

import tuner  # noqa: E402


def test_find_best_hyperparameters_selects_best(monkeypatch):
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

    gene_space = [{'low': 0, 'high': 1}]
    gene_map = {0: {'name': 'x', 'path': [], 'type': float}}
    gene_types = [float]

    search = [
        {'sol_per_pop': 1, 'num_parents_mating': 1, 'mutation_num_genes': 1},
        {'sol_per_pop': 2, 'num_parents_mating': 1, 'mutation_num_genes': 1},
    ]
    monkeypatch.setattr(tuner.config, 'HYPERPARAMETER_SEARCH_SPACE', search, raising=False)
    monkeypatch.setattr(tuner.config, 'GENERATIONS_PER_TUNE', 1, raising=False)

    scores = [0.1, 0.2]

    class DummyGA:
        def __init__(self, *a, **k):
            self.score = scores.pop(0)

        def run(self):
            pass

        def best_solution(self, **kwargs):
            return [self.score], self.score, None

    monkeypatch.setattr(tuner.pygad, 'GA', DummyGA)
    monkeypatch.setattr(tuner, '_evaluate_on_validation', lambda sol, gm: sol[0])

    best = tuner.find_best_hyperparameters(df, gene_space, gene_map, gene_types)
    assert best == search[1]


def test_find_best_hyperparameters_uses_selected_asset(monkeypatch):
    df_a = pd.DataFrame(
        {
            'Open': [1],
            'High': [1],
            'Low': [1],
            'Close': [1],
            'Volume': [1],
        },
        index=pd.date_range('2020-01-01', periods=1),
    )
    df_b = df_a * 2
    data_map = {'A': df_a, 'B': df_b}

    gene_space = [{'low': 0, 'high': 1}]
    gene_map = {0: {'name': 'x', 'path': [], 'type': float}}
    gene_types = [float]

    monkeypatch.setattr(
        tuner.config,
        'HYPERPARAMETER_SEARCH_SPACE',
        [{'sol_per_pop': 1, 'num_parents_mating': 1, 'mutation_num_genes': 1}],
        raising=False,
    )
    monkeypatch.setattr(tuner.config, 'GENERATIONS_PER_TUNE', 1, raising=False)
    monkeypatch.setattr(tuner.config, 'SELECTED_ASSET_NAME', 'B', raising=False)
    monkeypatch.setattr(tuner.config, 'TICKER', 'B', raising=False)

    captured = {}

    class DummyEval:
        def __init__(self, ohlc_data, base_rules, gene_map):
            captured['data'] = ohlc_data

        def __call__(self, ga_instance, solution, sol_idx):
            return 0.0

    class DummyGA:
        def __init__(self, *a, **k):
            pass

        def run(self):
            pass

        def best_solution(self, **kwargs):
            return [0], 0.0, None

    monkeypatch.setattr(tuner.fitness, 'FitnessEvaluator', DummyEval)
    monkeypatch.setattr(tuner.pygad, 'GA', DummyGA)
    monkeypatch.setattr(tuner, '_evaluate_on_validation', lambda s, g: 0)

    tuner.find_best_hyperparameters(data_map, gene_space, gene_map, gene_types)
    assert captured['data'] is df_b


def test_find_best_hyperparameters_preserves_gene_types(monkeypatch):
    df = pd.DataFrame({
        'Open': [1],
        'High': [1],
        'Low': [1],
        'Close': [1],
        'Volume': [1],
    }, index=pd.date_range('2020-01-01', periods=1))

    gene_space = [{'low': 0, 'high': 1}]
    gene_map = {0: {'name': 'x', 'path': [], 'type': float}}
    gene_types = [float]

    search = [{
        'sol_per_pop': 1,
        'num_parents_mating': 1,
        'mutation_num_genes': 1,
    }, {
        'sol_per_pop': 1,
        'num_parents_mating': 1,
        'mutation_num_genes': 1,
    }]

    monkeypatch.setattr(tuner.config, 'HYPERPARAMETER_SEARCH_SPACE', search, raising=False)
    monkeypatch.setattr(tuner.config, 'GENERATIONS_PER_TUNE', 1, raising=False)

    class DummyGA:
        def __init__(self, *a, **k):
            # Simulate PyGAD mutating the list in-place
            gt = k.get('gene_type')
            if isinstance(gt, list) and gt:
                gt[0] = [gt[0], None]

        def run(self):
            pass

        def best_solution(self, **kwargs):
            return [0], 0, None

    monkeypatch.setattr(tuner.pygad, 'GA', DummyGA)
    monkeypatch.setattr(tuner, '_evaluate_on_validation', lambda sol, gm: 0)

    original = list(gene_types)
    tuner.find_best_hyperparameters(df, gene_space, gene_map, gene_types)
    assert gene_types == original


def test_evaluate_on_validation_imports_pandas_ta(monkeypatch):
    df = pd.DataFrame(
        {
            'Open': [1, 1, 1],
            'High': [1, 1, 1],
            'Low': [1, 1, 1],
            'Close': [1, 1, 1],
            'Volume': [1, 1, 1],
        },
        index=pd.date_range('2020-01-01', periods=3),
    )

    if hasattr(pd.DataFrame, 'ta'):
        delattr(pd.DataFrame, 'ta')

    class PandasTaStub(types.ModuleType):
        def __init__(self):  # pragma: no cover - side effects only
            super().__init__('pandas_ta')

            class _Accessor:
                def __init__(self, df):
                    self._df = df

                def ema(self, length):  # noqa: D401 - simple stub
                    return pd.Series(1.0, index=self._df.index)

            pd.DataFrame.ta = property(lambda self: _Accessor(self))

    monkeypatch.setitem(sys.modules, 'pandas_ta', PandasTaStub())

    class DummyPF:
        def stats(self):
            return {'Sortino Ratio': 1.0}

    class DummyPortfolio:
        @staticmethod
        def from_signals(*a, **k):
            return DummyPF()

    monkeypatch.setattr(tuner, 'vbt', types.SimpleNamespace(Portfolio=DummyPortfolio))
    monkeypatch.setattr(tuner.data_loader, 'get_data', lambda **kwargs: df)
    monkeypatch.setattr(
        tuner.fitness,
        '_inject_genes_into_rules',
        lambda rules, gm, sol: {'entry_rules': {}, 'exit_rules': {}},
    )
    monkeypatch.setattr(
        tuner.engine,
        'process_strategy_rules',
        lambda data, rules: pd.Series([True, False, True], index=df.index),
    )
    monkeypatch.setattr(
        tuner.scanner_sim,
        'gate_entries',
        lambda entries, exits, mc: (pd.DataFrame(entries), None, {'accepted': 2}),
    )
    monkeypatch.setattr(tuner.config, 'TICKER', 'X', raising=False)
    monkeypatch.setattr(
        tuner.config,
        'VALIDATION_PERIOD',
        {'start': '2020-01-01', 'end': '2020-01-03'},
        raising=False,
    )
    monkeypatch.setattr(tuner.config, 'TIMEFRAME', '1D', raising=False)
    monkeypatch.setattr(tuner.config, 'STRATEGY_RULES', {}, raising=False)
    monkeypatch.setattr(tuner.config, 'MAX_HOLD_PERIOD', 1, raising=False)
    monkeypatch.setattr(tuner.config, 'FEES', 0.0, raising=False)
    monkeypatch.setattr(tuner.config, 'MIN_TRADES', 0, raising=False)
    monkeypatch.setattr(tuner.config, 'FITNESS_WEIGHTS', {'min_trades': 0}, raising=False)

    score = tuner._evaluate_on_validation([], {})
    assert score == 1.0


def test_evaluate_on_validation_respects_concurrency_limit(monkeypatch):
    df = pd.DataFrame(
        {
            'Open': [1, 1, 1],
            'High': [1, 1, 1],
            'Low': [1, 1, 1],
            'Close': [1, 1, 1],
            'Volume': [1, 1, 1],
        },
        index=pd.date_range('2020-01-01', periods=3),
    )

    if hasattr(pd.DataFrame, 'ta'):
        delattr(pd.DataFrame, 'ta')

    class PandasTaStub(types.ModuleType):
        def __init__(self):  # pragma: no cover - side effects only
            super().__init__('pandas_ta')

            class _Accessor:
                def __init__(self, df):
                    self._df = df

                def ema(self, length):  # noqa: D401 - simple stub
                    return pd.Series(1.0, index=self._df.index)

            pd.DataFrame.ta = property(lambda self: _Accessor(self))

    monkeypatch.setitem(sys.modules, 'pandas_ta', PandasTaStub())

    class DummyPF:
        def stats(self):
            return {'Sortino Ratio': 1.0}

    class DummyPortfolio:
        @staticmethod
        def from_signals(*a, **k):
            return DummyPF()

    monkeypatch.setattr(tuner, 'vbt', types.SimpleNamespace(Portfolio=DummyPortfolio))
    monkeypatch.setattr(tuner.data_loader, 'get_data', lambda **kwargs: df)
    monkeypatch.setattr(
        tuner.fitness,
        '_inject_genes_into_rules',
        lambda rules, gm, sol: {'entry_rules': {}, 'exit_rules': {}},
    )
    monkeypatch.setattr(
        tuner.engine,
        'process_strategy_rules',
        lambda data, rules: pd.Series([True, True, True], index=df.index),
    )

    captured = {}

    def gate_entries(entries, exits, mc):
        captured['mc'] = mc
        return pd.DataFrame(entries), None, {'accepted': mc}

    monkeypatch.setattr(tuner.scanner_sim, 'gate_entries', gate_entries)

    monkeypatch.setattr(tuner.config, 'TICKER', 'X', raising=False)
    monkeypatch.setattr(
        tuner.config,
        'VALIDATION_PERIOD',
        {'start': '2020-01-01', 'end': '2020-01-03'},
        raising=False,
    )
    monkeypatch.setattr(tuner.config, 'TIMEFRAME', '1D', raising=False)
    monkeypatch.setattr(tuner.config, 'STRATEGY_RULES', {}, raising=False)
    monkeypatch.setattr(tuner.config, 'MAX_HOLD_PERIOD', 1, raising=False)
    monkeypatch.setattr(tuner.config, 'FEES', 0.0, raising=False)
    monkeypatch.setattr(tuner.config, 'MIN_TRADES', 2, raising=False)
    monkeypatch.setattr(tuner.config, 'FITNESS_WEIGHTS', {'min_trades': 2}, raising=False)

    monkeypatch.setitem(tuner.config.SCANNER, 'max_concurrent_trades', 1)
    score = tuner._evaluate_on_validation([], {})
    assert score == -1e6
    assert captured['mc'] == 1

    monkeypatch.setitem(tuner.config.SCANNER, 'max_concurrent_trades', 2)
    score = tuner._evaluate_on_validation([], {})
    assert score == 1.0
    assert captured['mc'] == 2


def test_find_best_hyperparameters_pickleable_callback(monkeypatch):
    df = pd.DataFrame({
        'Open': [1],
        'High': [1],
        'Low': [1],
        'Close': [1],
        'Volume': [1],
    }, index=pd.date_range('2020-01-01', periods=1))

    gene_space = [{'low': 0, 'high': 1}]
    gene_map = {0: {'name': 'x', 'path': [], 'type': float}}
    gene_types = [float]

    search = [{
        'sol_per_pop': 1,
        'num_parents_mating': 1,
        'mutation_num_genes': 1,
    }]

    monkeypatch.setattr(tuner.config, 'HYPERPARAMETER_SEARCH_SPACE', search, raising=False)
    monkeypatch.setattr(tuner.config, 'GENERATIONS_PER_TUNE', 1, raising=False)

    class DummyGA:
        def __init__(self, *a, **k):
            pickle.dumps(k['on_generation'])

        def run(self):
            pass

        def best_solution(self, **kwargs):
            return [0], 0, None

    monkeypatch.setattr(tuner.pygad, 'GA', DummyGA)
    monkeypatch.setattr(tuner, '_evaluate_on_validation', lambda sol, gm: 0)

    tuner.find_best_hyperparameters(df, gene_space, gene_map, gene_types)
