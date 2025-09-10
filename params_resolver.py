"""Utilities for applying gene values and repairing indicator params."""

from __future__ import annotations

import copy
import logging

logger = logging.getLogger(__name__)

MACD_REPAIR_COUNT = 0


def _normalize_macd_params(params: dict) -> dict:
    """Ensure MACD params satisfy fast < slow and 1 <= signal < slow."""
    fast, slow, signal = (
        params.get("fast"),
        params.get("slow"),
        params.get("signal"),
    )
    if fast is None or slow is None or signal is None:
        raise ValueError("MACD params must be non-null: fast, slow, signal")
    original = (fast, slow, signal)
    if slow <= fast:
        slow = fast + 1
    if signal < 1:
        signal = 1
    if signal >= slow:
        signal = slow - 1
    fast, slow, signal = int(fast), int(slow), int(signal)
    params.update({"fast": fast, "slow": slow, "signal": signal})
    repaired = (fast, slow, signal)
    if repaired != original:
        logger.debug("Repaired MACD params %s -> %s", original, repaired)
        global MACD_REPAIR_COUNT
        MACD_REPAIR_COUNT += 1
    return params


def inject_genes_into_rules(base_rules: dict, gene_map: dict, solution: list) -> dict:
    """Inject gene values into a copy of strategy rules, resolving defaults."""

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
    for i, gene_value in enumerate(solution):
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

    def _apply_macd_repair(obj):
        if isinstance(obj, dict):
            if obj.get("indicator") == "macd":
                params = obj.get("params", {})
                if {"fast", "slow", "signal"} <= params.keys():
                    _normalize_macd_params(params)
            for val in obj.values():
                _apply_macd_repair(val)
        elif isinstance(obj, list):
            for item in obj:
                _apply_macd_repair(item)

    _apply_macd_repair(injected_rules)
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
