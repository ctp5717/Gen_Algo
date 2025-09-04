import sys
import types
from pathlib import Path

# Ensure repository root is on the import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy optional dependencies before importing main
sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))
sys.modules.setdefault("vectorbt", types.ModuleType("vectorbt"))

from gene_parser import parse_genes_from_config  # noqa: E402


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
    assert {"options": ["AND", "OR"]} in gene_space
    assert any(gs.get("low") == 1 and gs.get("high") == 3 for gs in gene_space)
    assert str in gene_types and int in gene_types


def test_vote_threshold_gene_present():
    import config

    _, gene_map, _ = parse_genes_from_config(config.STRATEGY_RULES)
    assert any(info["name"] == "vote_threshold" for info in gene_map.values())


def test_vote_threshold_gene_absent_when_not_vote():
    sample_rules = {
        "entry_rules": {
            "combination_logic": "AND",
            "vote_threshold": {"gene": "vt", "low": 1, "high": 3, "step": 1},
            "conditions": [],
        }
    }
    _, gene_map, _ = parse_genes_from_config(sample_rules)
    assert all(info["name"] != "vt" for info in gene_map.values())


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
