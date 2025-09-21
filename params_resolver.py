"""Utilities for applying gene values and repairing indicator params."""

from __future__ import annotations

import copy
import logging

from gene_parser import decode_solution
from indicator_library import INDICATOR_CONSTRAINTS

logger = logging.getLogger(__name__)


def inject_genes_into_rules(base_rules: dict, gene_map: dict, solution: list) -> dict:
    """Inject gene values into a copy of strategy rules, resolving defaults."""

    decoded_solution = decode_solution(solution, gene_map)

    def _resolve_defaults(obj):
        if isinstance(obj, dict):
            if "gene" in obj:
                if "default" in obj:
                    return obj["default"]
                if "options" in obj:
                    return obj.get("options", [None])[0]
                return obj.get("low", obj.get("high"))
            return {k: _resolve_defaults(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_resolve_defaults(v) for v in obj]
        return obj

    injected_rules = _resolve_defaults(copy.deepcopy(base_rules))
    for i, gene_value in enumerate(decoded_solution):
        gene_info = gene_map.get(i)
        if not gene_info:
            continue
        path = gene_info.get("path", [])
        if not path:
            continue
        current_level = injected_rules
        for key in path[:-1]:
            current_level = current_level[key]
        current_level[path[-1]] = gene_value

    def _apply_constraints(obj):
        if isinstance(obj, dict):
            ind = obj.get("indicator")
            if ind:
                params = obj.get("params", {})
                constraints = INDICATOR_CONSTRAINTS.get(ind.lower(), [])
                for c in sorted(
                    constraints,
                    key=lambda c: (
                        getattr(c, "kind", ""),
                        getattr(c, "a", ""),
                        str(getattr(c, "b", "")),
                    ),
                ):
                    c.enforce(params)
            for val in obj.values():
                _apply_constraints(val)
        elif isinstance(obj, list):
            for item in obj:
                _apply_constraints(item)

    _apply_constraints(injected_rules)
    return injected_rules


def resolve_effective_rules(base_rules: dict, gene_map: dict, solution: list) -> dict:
    """Return strategy rules with genes applied and constraints repaired."""
    resolved = inject_genes_into_rules(base_rules, gene_map, solution)
    entry = resolved.get("entry_rules", {})
    comb_logic = entry.get("combination_logic", "AND")
    if str(comb_logic).upper() == "VOTE":
        conditions = [
            c for c in entry.get("conditions", []) if c.get("is_active", True)
        ]
        n_active = len(conditions)
        vt_val = entry.get("vote_threshold")
        if n_active == 0:
            entry["vote_threshold"] = 1
            logger.warning(
                "resolve_effective_rules: vote_threshold set to 1 due to zero active conditions"
            )
        else:
            if vt_val is None:
                vt_val = n_active
            vt_val = max(1, min(int(vt_val), n_active))
            entry["vote_threshold"] = vt_val
    return resolved
