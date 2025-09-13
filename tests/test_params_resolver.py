import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))
sys.modules.setdefault("vectorbt", types.ModuleType("vectorbt"))

from gene_parser import parse_genes_from_config  # noqa: E402
from params_resolver import resolve_effective_rules  # noqa: E402


def _build_solution(gene_map, **values):
    sol = [None] * len(gene_map)
    for name, val in values.items():
        idx = next(i for i, info in gene_map.items() if info["name"] == name)
        sol[idx] = val
    return sol


def test_vote_threshold_reclamped_after_is_active_genes():
    rules = {
        "entry_rules": {
            "combination_logic": "VOTE",
            "vote_threshold": {"gene": "vt", "low": 1, "high": 3, "step": 1},
            "conditions": [
                {"is_active": {"gene": "c0", "options": [True, False]}},
                {"is_active": {"gene": "c1", "options": [True, False]}},
            ],
        }
    }
    _, gene_map, _ = parse_genes_from_config(rules)
    sol = _build_solution(gene_map, vt=2, c0=True, c1=False)
    resolved = resolve_effective_rules(rules, gene_map, sol)
    assert resolved["entry_rules"]["vote_threshold"] == 1


def test_vote_threshold_warning_when_all_inactive(caplog):
    import logging

    rules = {
        "entry_rules": {
            "combination_logic": "VOTE",
            "vote_threshold": {"gene": "vt", "low": 1, "high": 4, "step": 1},
            "conditions": [
                {"is_active": {"gene": "c0", "options": [True, False]}},
                {"is_active": {"gene": "c1", "options": [True, False]}},
            ],
        }
    }
    _, gene_map, _ = parse_genes_from_config(rules)
    sol = _build_solution(gene_map, vt=4, c0=False, c1=False)
    with caplog.at_level(logging.WARNING):
        resolved = resolve_effective_rules(rules, gene_map, sol)
    assert resolved["entry_rules"]["vote_threshold"] == 1
    assert any("vote_threshold set to 1" in m for m in caplog.messages)


def test_macd_constraints_enforced_deterministically():
    rules = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "macd",
                    "params": {"fast": 10, "slow": 5, "signal": 0},
                }
            ]
        }
    }
    repaired1 = resolve_effective_rules(rules, {}, [])
    repaired2 = resolve_effective_rules(rules, {}, [])
    params = repaired1["entry_rules"]["conditions"][0]["params"]
    assert params["slow"] > params["fast"]
    assert params["signal"] >= 1
    assert params["signal"] < params["slow"]
    assert repaired1 == repaired2
