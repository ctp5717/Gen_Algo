# gene_parser.py
"""
Gene Parsing Utilities
This module contains helper functions for working with strategy rules.
"""

from __future__ import annotations

from numbers import Number
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

GeneSpaceItem = Union[Dict[str, Any], List[Any]]
GeneSpace = List[GeneSpaceItem]
GeneMap = Dict[int, Dict[str, Any]]


def parse_genes_from_config(
    rules: Dict[str, Any],
) -> Tuple[GeneSpace, GeneMap, List[type]]:
    """Parse STRATEGY_RULES and return gene_space, gene_map, and gene_types.

    Only rules with ``is_active`` set to ``True`` will be considered when
    searching for genes.
    """
    gene_space: GeneSpace = []
    gene_map: GeneMap = {}
    gene_types: List[type] = []
    gene_index = 0

    entry_rules = rules.get("entry_rules", {})
    active_conditions = [
        c for c in entry_rules.get("conditions", []) if c.get("is_active", True)
    ]
    n_active = len(active_conditions) or 1
    comb_logic = entry_rules.get("combination_logic", "AND")
    if str(comb_logic).upper() == "VOTE" or (
        isinstance(comb_logic, dict)
        and (comb_logic.get("gene") or comb_logic.get("options"))
    ):
        vt_gene = entry_rules.get("vote_threshold")
        if isinstance(vt_gene, dict) and vt_gene.get("gene"):
            high = vt_gene.get("high", n_active)
            vt_gene["high"] = min(high, n_active)
            vt_gene["low"] = max(1, min(vt_gene.get("low", 1), n_active))
            entry_rules["vote_threshold"] = vt_gene

    def find_genes(sub_config: Any, path: List[Any]) -> None:
        nonlocal gene_index
        # Skip entire branch if rule is explicitly inactive
        if isinstance(sub_config, dict) and sub_config.get("is_active") is False:
            return

        if isinstance(sub_config, dict):
            for key, value in sub_config.items():
                current_path = path + [key]
                if (
                    isinstance(value, dict)
                    and value.get("is_active", True) is not False
                    and "gene" in value
                ):
                    gene_info = value
                    gene_name = gene_info["gene"]

                    if key == "vote_threshold" and not (
                        str(comb_logic).upper() == "VOTE"
                        or (
                            isinstance(comb_logic, dict)
                            and (comb_logic.get("gene") or comb_logic.get("options"))
                        )
                    ):
                        continue

                    if "options" in gene_info:
                        options = list(gene_info["options"])
                        space_item: GeneSpaceItem = options
                        gene_type = type(options[0]) if options else str
                    else:
                        if key == "vote_threshold":
                            gene_type = int
                        else:
                            gene_type = (
                                int
                                if isinstance(gene_info.get("step", 1.0), int)
                                else float
                            )
                        high = gene_info.get("high")
                        low = gene_info.get("low", 1)
                        space_dict: Dict[str, Any] = {"low": low, "high": high}
                        if "step" in gene_info:
                            space_dict["step"] = gene_info["step"]
                        space_item = space_dict

                    gene_space.append(space_item)
                    gene_types.append(gene_type)
                    gene_map[gene_index] = {
                        "name": gene_name,
                        "path": current_path,
                        "type": gene_type,
                    }
                    if "options" in gene_info:
                        gene_map[gene_index]["options"] = options
                    gene_index += 1
                elif isinstance(value, dict) or isinstance(value, list):
                    find_genes(value, current_path)
        elif isinstance(sub_config, list):
            for i, item in enumerate(sub_config):
                current_path = path + [i]
                find_genes(item, current_path)

    find_genes(rules, [])
    return gene_space, gene_map, gene_types


def _is_numeric_gene_value(value: Any) -> bool:
    """Return ``True`` if ``value`` can be consumed by PyGAD as-is."""

    if value is None:
        return True
    if isinstance(value, (bool, Number)):
        return True
    if isinstance(value, str):
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def prepare_ga_inputs(
    gene_space: Sequence[GeneSpaceItem],
    gene_map: GeneMap,
    gene_types: Sequence[type],
) -> Tuple[List[GeneSpaceItem], List[type]]:
    """Return PyGAD-compatible ``gene_space`` and ``gene_type`` lists.

    PyGAD 3.x rejects string values inside ``gene_space``.  Strategy rules
    frequently declare categorical genes using string options (for example,
    ``"sl_break_even_mode"``).  This helper converts such options into
    integer enumerations while recording the mapping on ``gene_map`` so the
    original value can be restored during injection and display.
    """

    converted_space: List[GeneSpaceItem] = []
    converted_types: List[type] = []

    for idx, spec in enumerate(gene_space):
        gene_info = gene_map.get(idx, {})
        gene_type = gene_types[idx] if idx < len(gene_types) else float

        if isinstance(spec, (list, tuple)):
            options = list(spec)
            if options and not all(_is_numeric_gene_value(opt) for opt in options):
                decode_map = dict(enumerate(options))
                # Store both the original options and the decoder to allow
                # downstream code to recover the categorical values.
                gene_info.setdefault("options", options)
                gene_info["option_decode_map"] = decode_map
                converted_space.append(list(decode_map.keys()))
                converted_types.append(int)
                gene_map[idx] = gene_info
                continue
            converted_space.append(options)
            converted_types.append(gene_type)
            gene_map[idx] = gene_info
            continue

        if isinstance(spec, dict):
            converted_space.append(dict(spec))
            converted_types.append(gene_type)
            gene_map[idx] = gene_info
            continue

        converted_space.append(spec)
        converted_types.append(gene_type)
        gene_map[idx] = gene_info

    return converted_space, converted_types


def decode_solution(
    solution: Sequence[Any] | Iterable[Any],
    gene_map: GeneMap,
) -> List[Any]:
    """Convert an encoded GA solution back to declarative gene values."""

    if hasattr(solution, "tolist"):
        decoded = list(solution.tolist())  # type: ignore[assignment]
    else:
        decoded = list(solution)

    for idx, value in enumerate(decoded):
        info = gene_map.get(idx)
        if not info:
            continue
        mapping = info.get("option_decode_map")
        if not mapping:
            continue
        try:
            key = int(round(float(value)))
        except (TypeError, ValueError):
            continue
        decoded[idx] = mapping.get(key, value)
    return decoded
