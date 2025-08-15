import os
import sys
import types
import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))
import config  # noqa: E402
import scanner_sim  # noqa: E402
from multi_asset_fitness import MultiAssetFitnessEvaluator  # noqa: E402
import vectorbt  # noqa: E402


def _make_idx(n=3):
    return pd.date_range('2020', periods=n, freq='D')


def test_dict_inputs():
    idx = _make_idx()
    entries = {'A': pd.Series([True, False, False], index=idx)}
    exits = {'A': pd.Series([False, True, False], index=idx)}
    gated, _, _ = scanner_sim.gate_entries(entries, exits, max_concurrent=1, price_index=idx)
    assert gated.loc[idx[0], 'A']


def test_series_inputs():
    idx = _make_idx()
    entries = pd.Series([True, False, False], index=idx)
    exits = pd.Series([False, True, False], index=idx)
    gated, _, _ = scanner_sim.gate_entries(entries, exits, max_concurrent=1, price_index=idx)
    assert gated.iloc[0, 0]


def test_wrong_type_raises():
    with pytest.raises(TypeError):
        scanner_sim.gate_entries([True, False], [False, True], max_concurrent=1)


def test_single_asset_parity(monkeypatch):
    idx = _make_idx(4)
    data = pd.DataFrame({'Open': [1, 2, 3, 4], 'Close': [1, 2, 3, 4]}, index=idx)

    def fake_process(df, rules):
        return pd.Series([True, False, True, False], index=df.index)

    monkeypatch.setattr('strategy_engine.process_strategy_rules', fake_process)

    ma_eval = MultiAssetFitnessEvaluator({'a': data}, {}, {})
    _, entries_df, exits_df, _scores, sl, tp, tr = ma_eval._build_signals([], ['a'])
    gated, _, _ = scanner_sim.gate_entries(
        entries_df, exits_df, max_concurrent=1, price_index=data.index
    )
    asset_entries = gated.iloc[:, 0].reindex(data.index, fill_value=False)
    shifted_entries = asset_entries.shift(1, fill_value=False)
    time_exit = shifted_entries.shift(config.MAX_HOLD_PERIOD, fill_value=False)
    pf_multi = vectorbt.Portfolio.from_signals(
        close=data['Close'],
        entries=shifted_entries,
        exits=time_exit,
        sl_stop=sl,
        tp_stop=tp,
        sl_trail=tr,
        fees=config.FEES,
        freq=config.TIMEFRAME,
    )
    entries_single = fake_process(data, {})
    shifted_single = entries_single.shift(1, fill_value=False)
    time_exit_single = shifted_single.shift(config.MAX_HOLD_PERIOD, fill_value=False)
    pf_single = vectorbt.Portfolio.from_signals(
        close=data['Close'],
        entries=shifted_single,
        exits=time_exit_single,
        fees=config.FEES,
        freq=config.TIMEFRAME,
    )
    pd.testing.assert_series_equal(pf_multi.returns(), pf_single.returns())


def test_all_false_entries():
    idx = _make_idx()
    entries = pd.Series([False, False, False], index=idx, name='asset')
    exits = pd.Series([False, False, False], index=idx, name='asset')
    gated, open_count, diag = scanner_sim.gate_entries(
        entries, exits, max_concurrent=1, price_index=idx
    )
    assert not gated.any().any()
    assert diag['total_candidates'] == 0
