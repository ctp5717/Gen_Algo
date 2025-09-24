import sys
import types
from pathlib import Path

import pytest

# Ensure repository root is on the import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy optional dependencies before importing main
sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))
sys.modules.setdefault("vectorbt", types.ModuleType("vectorbt"))

from gene_parser import (  # noqa: E402
    decode_solution,
    parse_genes_from_config,
    prepare_ga_inputs,
)
from strategy_rules import STRATEGY_RULES  # noqa: E402


def test_parse_genes_respects_is_active():
    sample_rules = {
        "entry_rules": {
            "conditions": [
                {
                    "is_active": True,
                    "indicator": "ema",
                    "params": {
                        "period": {
                            "gene": "ema_period",
                            "low": 5,
                            "high": 10,
                            "step": 1,
                        }
                    },
                    "condition": {},
                },
                {
                    "is_active": False,
                    "indicator": "sma",
                    "params": {
                        "period": {
                            "gene": "sma_period",
                            "low": 5,
                            "high": 10,
                            "step": 1,
                        }
                    },
                    "condition": {},
                },
            ]
        },
        "exit_rules": {
            "trailing_stop": {
                "is_active": True,
                "type": "percentage",
                "params": {
                    "value": {"gene": "tsl", "low": 0.01, "high": 0.1, "step": 0.01}
                },
            },
            "take_profit": {
                "is_active": False,
                "type": "percentage",
                "params": {
                    "value": {"gene": "tp", "low": 0.02, "high": 0.2, "step": 0.01}
                },
            },
        },
    }

    gene_space, gene_map, gene_types = parse_genes_from_config(sample_rules)

    gene_names = [info["name"] for info in gene_map.values()]

    assert "ema_period" in gene_names
    assert "tsl" in gene_names
    assert "sma_period" not in gene_names
    assert "tp" not in gene_names
    assert len(gene_space) == 2


def test_parse_top_level_combination_genes():
    sample_rules = {
        "entry_rules": {
            "combination_logic": {
                "gene": "logic",
                "options": ["AND", "OR"],
            },
            "vote_threshold": {
                "gene": "vt",
                "low": 1,
                "high": 3,
                "step": 1,
            },
            "conditions": [],
        }
    }

    gene_space, gene_map, gene_types = parse_genes_from_config(sample_rules)

    names = [info["name"] for info in gene_map.values()]
    assert "logic" in names
    assert "vt" in names
    assert ["AND", "OR"] in gene_space
    assert any(
        isinstance(gs, dict) and gs.get("low") == 1 and gs.get("high") == 1
        for gs in gene_space
    )
    assert str in gene_types and int in gene_types


def test_option_genes_emit_sequence_space():
    space, _, _ = parse_genes_from_config(STRATEGY_RULES)
    assert all(not (isinstance(spec, dict) and "options" in spec) for spec in space)


def test_prepare_ga_inputs_encodes_string_options():
    gene_space, gene_map, gene_types = parse_genes_from_config(STRATEGY_RULES)
    ga_space, ga_types = prepare_ga_inputs(gene_space, gene_map, gene_types)

    idx = next(
        i for i, info in gene_map.items() if info["name"] == "sl_break_even_mode"
    )

    assert ga_space[idx] == [0, 1, 2]
    assert ga_types[idx] is int

    decoded = decode_solution([0] * len(gene_map), gene_map)
    assert decoded[idx] == "none"
    assert gene_map[idx]["option_decode_map"][1] == "breakeven"


def test_vote_threshold_gene_present():
    _, gene_map, _ = parse_genes_from_config(STRATEGY_RULES)
    assert any(info["name"] == "vote_threshold" for info in gene_map.values())


@pytest.mark.parametrize("logic", ["AND", "OR"])
def test_vote_threshold_gene_absent_when_not_vote(logic):
    sample_rules = {
        "entry_rules": {
            "combination_logic": logic,
            "vote_threshold": {"gene": "vt", "low": 1, "high": 3, "step": 1},
            "conditions": [],
        }
    }
    _, gene_map, _ = parse_genes_from_config(sample_rules)
    assert all(info["name"] != "vt" for info in gene_map.values())


def test_vote_threshold_gene_only_for_vote_or_option_logic():
    vote_rules = {
        "entry_rules": {
            "combination_logic": "VOTE",
            "vote_threshold": {"gene": "vt", "low": 1, "high": 3, "step": 1},
            "conditions": [],
        }
    }
    _, gm_vote, _ = parse_genes_from_config(vote_rules)
    assert any(info["name"] == "vt" for info in gm_vote.values())

    option_rules = {
        "entry_rules": {
            "combination_logic": {"gene": "logic", "options": ["AND", "OR"]},
            "vote_threshold": {"gene": "vt", "low": 1, "high": 3, "step": 1},
            "conditions": [],
        }
    }
    _, gm_opt, _ = parse_genes_from_config(option_rules)
    assert any(info["name"] == "vt" for info in gm_opt.values())

    and_rules = {
        "entry_rules": {
            "combination_logic": "AND",
            "vote_threshold": {"gene": "vt", "low": 1, "high": 3, "step": 1},
            "conditions": [],
        }
    }
    _, gm_and, _ = parse_genes_from_config(and_rules)
    assert all(info["name"] != "vt" for info in gm_and.values())


def test_trade_management_genes_present():
    _, gene_map, _ = parse_genes_from_config(STRATEGY_RULES)
    names = {info["name"] for info in gene_map.values()}
    expected = {
        "stop_loss_pct",
        "num_tp_levels",
        "tp_pct_1",
        "tp_pct_2",
        "tp_pct_3",
        "tp_pct_4",
        "tp_trailing_pct",
        "sl_break_even_mode",
        "sl_timeout_bars",
        "sl_trailing_pct",
    }
    assert expected.issubset(names)


def test_trade_management_dependent_genes_skip_when_inactive():
    import copy

    rules = copy.deepcopy(STRATEGY_RULES)
    trade_mgmt = rules["exit_rules"]["trade_management"]
    trade_mgmt["tp_trailing_pct"]["is_active"] = False
    trade_mgmt["sl_timeout_bars"]["is_active"] = False
    trade_mgmt["sl_trailing_pct"]["is_active"] = False
    _, gene_map, _ = parse_genes_from_config(rules)
    names = {info["name"] for info in gene_map.values()}
    assert "tp_trailing_pct" not in names
    assert "sl_timeout_bars" not in names
    assert "sl_trailing_pct" not in names


def test_vote_threshold_gene_bounds_update():
    rules = {
        "entry_rules": {
            "combination_logic": "VOTE",
            "vote_threshold": {"gene": "vt", "low": 1, "high": 10, "step": 1},
            "conditions": [
                {"is_active": True},
                {"is_active": True},
                {"is_active": False},
            ],
        }
    }
    space, gene_map, _ = parse_genes_from_config(rules)
    idx = next(i for i, info in gene_map.items() if info["name"] == "vt")
    assert space[idx]["high"] == 2
    # deactivate one more condition
    rules["entry_rules"]["conditions"][1]["is_active"] = False
    space2, gene_map2, _ = parse_genes_from_config(rules)
    idx2 = next(i for i, info in gene_map2.items() if info["name"] == "vt")
    assert space2[idx2]["high"] == 1


def test_vote_threshold_high_clamped_and_other_genes_unaffected():
    rules = {
        "entry_rules": {
            "combination_logic": "VOTE",
            "vote_threshold": {"gene": "vt", "low": 1, "high": 10, "step": 1},
            "conditions": [
                {
                    "is_active": True,
                    "params": {"foo": {"gene": "foo", "low": 2, "high": 4, "step": 1}},
                    "condition": {},
                },
                {
                    "is_active": False,
                    "params": {"bar": {"gene": "bar", "low": 5, "high": 6, "step": 1}},
                    "condition": {},
                },
            ],
        }
    }
    space, gene_map, _ = parse_genes_from_config(rules)
    vt_idx = next(i for i, info in gene_map.items() if info["name"] == "vt")
    assert space[vt_idx]["high"] == 1
    foo_idx = next(i for i, info in gene_map.items() if info["name"] == "foo")
    assert space[foo_idx] == {"low": 2, "high": 4, "step": 1}
    assert all(info["name"] != "bar" for info in gene_map.values())
    assert len(space) == 2


def test_combination_logic_gene_clamps_vote_threshold_and_shrinks_with_inactive():
    rules = {
        "entry_rules": {
            "combination_logic": {"gene": "logic", "options": ["AND", "VOTE"]},
            "vote_threshold": {"gene": "vt", "low": 1, "high": 9, "step": 1},
            "conditions": [
                {"is_active": True},
                {"is_active": True},
                {"is_active": False},
            ],
        }
    }
    space, gene_map, _ = parse_genes_from_config(rules)
    vt_idx = next(i for i, info in gene_map.items() if info["name"] == "vt")
    assert space[vt_idx]["high"] == 2
    assert rules["entry_rules"]["vote_threshold"]["high"] == 2
    # deactivate another condition and ensure shrinkage
    rules["entry_rules"]["conditions"][0]["is_active"] = False
    space2, gene_map2, _ = parse_genes_from_config(rules)
    vt_idx2 = next(i for i, info in gene_map2.items() if info["name"] == "vt")
    assert space2[vt_idx2]["high"] == 1


def test_vote_threshold_warning_on_zero_active(caplog):
    import logging

    from params_resolver import resolve_effective_rules

    rules = {
        "entry_rules": {
            "combination_logic": {"gene": "logic", "options": ["AND", "VOTE"]},
            "vote_threshold": {"gene": "vt", "low": 1, "high": 5, "step": 1},
            "conditions": [
                {"is_active": {"gene": "c0", "options": [True, False]}},
                {"is_active": {"gene": "c1", "options": [True, False]}},
            ],
        }
    }

    space, gene_map, _ = parse_genes_from_config(rules)
    sol = [None] * len(gene_map)
    idx_logic = next(i for i, g in gene_map.items() if g["name"] == "logic")
    idx_vt = next(i for i, g in gene_map.items() if g["name"] == "vt")
    idx_c0 = next(i for i, g in gene_map.items() if g["name"] == "c0")
    idx_c1 = next(i for i, g in gene_map.items() if g["name"] == "c1")
    sol[idx_logic] = "VOTE"
    sol[idx_vt] = 5
    sol[idx_c0] = False
    sol[idx_c1] = False

    with caplog.at_level(logging.WARNING):
        resolved = resolve_effective_rules(rules, gene_map, sol)
    assert resolved["entry_rules"]["vote_threshold"] == 1
    assert any("vote_threshold set to 1" in msg for msg in caplog.messages)
