# gene_parser.py
"""Gene Parsing Utilities used to discover GA genes from strategy rules."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Number
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

GeneSpaceItem = Union[Dict[str, Any], List[Any]]
GeneSpace = List[GeneSpaceItem]
GeneMap = Dict[int, Dict[str, Any]]


class _GeneAccumulator:
    """State container that records discovered gene definitions."""

    def __init__(self, vote_enabled: bool) -> None:
        self.vote_enabled = vote_enabled
        self.gene_space: GeneSpace = []
        self.gene_map: GeneMap = {}
        self.gene_types: List[type] = []
        self._index = 0

    def add_gene(
        self,
        *,
        path: List[Any],
        key: Any,
        gene_info: Mapping[str, Any],
    ) -> None:
        space_item, gene_type, options = _normalise_gene_spec(key, gene_info)
        idx = self._index
        self._index += 1

        self.gene_space.append(space_item)
        self.gene_types.append(gene_type)

        entry: Dict[str, Any] = {
            "name": gene_info["gene"],
            "path": path,
            "type": gene_type,
        }
        if options is not None:
            entry["options"] = options
        self.gene_map[idx] = entry


def _traverse_rules(config: Any, path: List[Any], accumulator: _GeneAccumulator) -> None:
    """Depth-first traversal that collects gene declarations."""

    if isinstance(config, Mapping):
        if config.get("is_active") is False:
            return
        for key, value in config.items():
            current_path = path + [key]
            if _is_gene_declaration(value):
                if key == "vote_threshold" and not accumulator.vote_enabled:
                    continue
                accumulator.add_gene(path=current_path, key=key, gene_info=value)
                continue
            if isinstance(value, (Mapping, list, tuple)):
                _traverse_rules(value, current_path, accumulator)
        return

    if isinstance(config, (list, tuple)):
        for index, item in enumerate(config):
            _traverse_rules(item, path + [index], accumulator)


def _is_gene_declaration(candidate: Any) -> bool:
    return (
        isinstance(candidate, Mapping)
        and candidate.get("is_active", True) is not False
        and "gene" in candidate
    )


def _combination_allows_vote(value: Any) -> bool:
    if isinstance(value, str):
        return value.upper() == "VOTE"
    if isinstance(value, Mapping):
        return bool(value.get("gene") or value.get("options"))
    return False


def _count_active_conditions(entry_rules: Mapping[str, Any]) -> int:
    conditions = entry_rules.get("conditions", [])
    active = [c for c in conditions if c.get("is_active", True)]
    return len(active) or 1


def _clamp_vote_threshold_gene(entry_rules: Mapping[str, Any], n_active: int) -> None:
    if n_active <= 0:
        n_active = 1

    vt_gene = entry_rules.get("vote_threshold")
    if not isinstance(vt_gene, Mapping) or not vt_gene.get("gene"):
        return

    vt_copy = dict(vt_gene)
    raw_high = vt_copy.get("high")
    high = n_active if raw_high is None else min(raw_high, n_active)
    low = vt_copy.get("low", 1)
    low = max(1, min(low, high))

    vt_copy["high"] = high
    vt_copy["low"] = low
    entry_rules["vote_threshold"] = vt_copy


def _normalise_gene_spec(
    key: Any, gene_info: Mapping[str, Any]
) -> Tuple[GeneSpaceItem, type, List[Any] | None]:
    if "options" in gene_info:
        options = list(gene_info["options"])
        gene_type = type(options[0]) if options else str
        return options, gene_type, options

    if key == "vote_threshold":
        gene_type = int
    else:
        gene_type = int if isinstance(gene_info.get("step", 1.0), int) else float

    space_dict: Dict[str, Any] = {
        "low": gene_info.get("low", 1),
        "high": gene_info.get("high"),
    }
    if "step" in gene_info:
        space_dict["step"] = gene_info["step"]
    return space_dict, gene_type, None


def parse_genes_from_config(
    rules: Dict[str, Any],
) -> Tuple[GeneSpace, GeneMap, List[type]]:
    """Parse ``STRATEGY_RULES`` and return ``gene_space``, ``gene_map`` and types."""

    entry_rules = rules.get("entry_rules", {})
    vote_enabled = _combination_allows_vote(entry_rules.get("combination_logic"))
    if vote_enabled:
        _clamp_vote_threshold_gene(entry_rules, _count_active_conditions(entry_rules))

    accumulator = _GeneAccumulator(vote_enabled)
    _traverse_rules(rules, [], accumulator)
    return accumulator.gene_space, accumulator.gene_map, accumulator.gene_types


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
