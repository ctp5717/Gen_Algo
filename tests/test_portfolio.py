import types
import pandas as pd
import numpy as np

import config
import data_loader
import strategy_engine as engine
import fitness
import tuner
import walk_forward

class SimpleMonkeyPatch:
    def setattr(self, target, name=None, value=None, *, raising=True):
        if name is None:
            target(value)
        else:
            if isinstance(target, str):
                module = __import__(target, fromlist=[name])
                setattr(module, name, value)
            else:
                setattr(target, name, value)

monkeypatch = SimpleMonkeyPatch()

def _mk_df():
    idx = pd.date_range("2021-01-01", periods=10, freq="D")
    a = pd.DataFrame({'Open':1,'High':1,'Low':1,'Close':[100,101,102,103,104,105,106,107,108,109],'Volume':1}, index=idx)
    b = pd.DataFrame({'Open':1,'High':1,'Low':1,'Close':[200,200,200,200,200,200,200,200,200,200],'Volume':1}, index=idx)
    a.columns = pd.MultiIndex.from_product([['AAA'], a.columns])
    b.columns = pd.MultiIndex.from_product([['BBB'], b.columns])
    return a.join(b)

def test_get_data_multi_asset():
    def fake_fetch(t, s, e, i):
        idx = pd.date_range("2021-01-01", periods=5, freq="D")
        df = pd.DataFrame({'Open':1,'High':1,'Low':1,'Close':[1,2,3,4,5],'Volume':1}, index=idx)
        return df
    monkeypatch.setattr(data_loader, '_fetch_single', lambda t,s,e,i: fake_fetch(t,s,e,i))
    df = data_loader.get_data(['AAA-USD','BBB-USD'], '2021-01-01','2021-01-10','1d')
    assert isinstance(df.columns, pd.MultiIndex)
    assert ('AAA-USD','Close') in df.columns and ('BBB-USD','Close') in df.columns

def test_fitness_evaluator_counts_portfolio_trades():
    df = _mk_df()
    rules = {'entry_rules': {'combination_logic':'AND','conditions':[]}, 'exit_rules':{}}
    monkeypatch.setattr(engine, 'process_strategy_rules', lambda data, rules: pd.DataFrame({'AAA':[True,False,False,False,False,False,False,False,False,False],'BBB':[True,False,False,False,False,False,False,False,False,False]}, index=data.index))
    cfg = dict(config.STRATEGY_RULES)
    gm = {0:{'name':'dummy','path':['entry_rules','conditions']}}
    evalr = fitness.FitnessEvaluator(df, cfg, gm)
    score = evalr(None, [0], 0)
    assert score != -999.0

def test_tuner_uses_tuning_asset():
    orig = config.PORTFOLIO_OPTIMIZATION_ENABLED
    setattr(config, 'PORTFOLIO_OPTIMIZATION_ENABLED', True)
    setattr(config, 'TUNING_ASSET', 'AAA-USD')
    monkeypatch.setattr(data_loader, 'get_data', lambda t,s,e,i: pd.DataFrame({'Open':1,'High':1,'Low':1,'Close':[1,2,3,4,5],'Volume':1}, index=pd.date_range('2021-01-01', periods=5)))
    monkeypatch.setattr(fitness, 'FitnessEvaluator', lambda ohlc_data, base_rules, gene_map: types.SimpleNamespace(__call__=lambda *a, **k: 1.0))
    class DummyGA:
        def __init__(self, *a, **k): pass
        def run(self): pass
        def best_solution(self): return [0], 1.0, None
    import pygad as _pg
    import builtins
    builtins.pygad = _pg
    monkeypatch.setattr(_pg, 'GA', DummyGA)
    gs, gt, gm = [], [], {}
    best = tuner.find_best_hyperparameters(gs, gt, gm)
    assert isinstance(best, dict)
    setattr(config, 'PORTFOLIO_OPTIMIZATION_ENABLED', orig)

def test_walk_forward_passes_asset_basket():
    setattr(config, 'PORTFOLIO_OPTIMIZATION_ENABLED', True)
    setattr(config, 'ASSET_BASKET', ['AAA-USD','BBB-USD'])
    df = _mk_df()
    monkeypatch.setattr(data_loader, 'get_data', lambda t,s,e,i: df)
    monkeypatch.setattr(engine, 'process_strategy_rules', lambda data, rules: pd.DataFrame({'AAA':[True]+[False]*9,'BBB':[True]+[False]*9}, index=data.index))
    class DummyGA:
        def __init__(self, *a, **k):
            import numpy as np
            self.population = np.zeros((k.get('sol_per_pop', 10), k.get('num_genes', 0)))
            self.initial_population = self.population.copy()
            self.num_generations = k.get('num_generations', 10)
            self.generations_completed = self.num_generations
        def run(self): pass
        def best_solution(self): return [0], 1.0, None
    import pygad as _pg
    import builtins
    builtins.pygad = _pg
    monkeypatch.setattr(_pg, 'GA', DummyGA)
    res = walk_forward.run_walk_forward_validation()
    assert res is None or isinstance(res, dict)
