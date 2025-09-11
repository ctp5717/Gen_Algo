import pandas as pd
import pytest

from portfolio_utils import extract_exit_params


def test_extract_exit_params_none():
    entries = pd.Series([True, False, True], index=pd.date_range("2020", periods=3))
    t_exit, sl_stop, sl_trail, tp_stop = extract_exit_params(entries, {}, 1)
    expected = entries.shift(1, fill_value=False).reindex(entries.index, fill_value=False)
    assert t_exit.equals(expected)
    assert sl_stop is None and sl_trail is None and tp_stop is None


@pytest.mark.parametrize("rule_key", ["stop_loss", "trailing_stop", "take_profit"])
def test_extract_exit_params_single_rule(rule_key):
    entries = pd.Series([True, True, False], index=pd.date_range("2020", periods=3))
    exit_rules = {rule_key: {"is_active": True, "params": {"value": 0.1}}}
    t_exit, sl_stop, sl_trail, tp_stop = extract_exit_params(entries, exit_rules, 1)
    expected = entries.shift(1, fill_value=False).reindex(entries.index, fill_value=False)
    assert t_exit.equals(expected)
    assert sl_stop == (0.1 if rule_key == "stop_loss" else None)
    assert sl_trail == (0.1 if rule_key == "trailing_stop" else None)
    assert tp_stop == (0.1 if rule_key == "take_profit" else None)


def test_extract_exit_params_all_rules():
    entries = pd.Series([True, False, True], index=pd.date_range("2020", periods=3))
    exit_rules = {
        "stop_loss": {"is_active": True, "params": {"value": 0.2}},
        "trailing_stop": {"is_active": True, "params": {"value": 0.3}},
        "take_profit": {"is_active": True, "params": {"value": 0.4}},
    }
    t_exit, sl_stop, sl_trail, tp_stop = extract_exit_params(entries, exit_rules, 2)
    expected = entries.shift(2, fill_value=False).reindex(entries.index, fill_value=False)
    assert t_exit.equals(expected)
    assert (sl_stop, sl_trail, tp_stop) == (0.2, 0.3, 0.4)
