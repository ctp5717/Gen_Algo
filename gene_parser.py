# gene_parser.py
"""
Gene Parsing Utilities
This module contains helper functions for working with strategy rules.
"""

from typing import Any, Dict, List, Tuple


def resolve_gene_value(value: Any) -> Any:
    """Return a deterministic default for a gene configuration value.

    Parameters
    ----------
    value : Any
        Either a primitive or a mapping containing gene metadata such as
        ``low``/``high`` bounds or a set of ``options``.

    Returns
    -------
    Any
        The resolved default value. If ``value`` is a mapping with an
        ``options`` list, the first option is returned. Otherwise ``low`` takes
        precedence over ``high`` when present. Non-mapping inputs are returned
        unchanged.
    """

    if isinstance(value, dict):
        if "options" in value:
            options = value.get("options", [])
            return options[0] if options else None
        if "low" in value or "high" in value:
            return value.get("low", value.get("high"))
    return value


def parse_genes_from_config(
    rules: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]], List[type]]:
    """Parse STRATEGY_RULES and return gene_space, gene_map, and gene_types.

    Only rules with ``is_active`` set to ``True`` will be considered when
    searching for genes.
    """
    gene_space: List[Dict[str, Any]] = []
    gene_map: Dict[int, Dict[str, Any]] = {}
    gene_types: List[type] = []
    gene_index = 0

    entry_rules = rules.get("entry_rules", {})
    active_conditions = [
        c for c in entry_rules.get("conditions", []) if c.get("is_active", True)
    ]
    n_active = len(active_conditions) or 1
    comb_logic = entry_rules.get("combination_logic", "AND")

    def find_genes(sub_config: Any, path: List[Any]) -> None:
        nonlocal gene_index
        # Skip entire branch if rule is explicitly inactive
        if isinstance(sub_config, dict) and sub_config.get("is_active") is False:
            return

        if isinstance(sub_config, dict):
            for key, value in sub_config.items():
                current_path = path + [key]
                if isinstance(value, dict) and "gene" in value:
                    gene_info = value
                    gene_name = gene_info["gene"]

                    if key == "vote_threshold" and not (
                        isinstance(comb_logic, dict)
                        or str(comb_logic).upper() == "VOTE"
                    ):
                        continue

                    if "options" in gene_info:
                        space_item = {"options": gene_info["options"]}
                        gene_type = (
                            type(gene_info["options"][0])
                            if gene_info["options"]
                            else str
                        )
                    else:
                        gene_type = (
                            int
                            if isinstance(gene_info.get("step", 1.0), int)
                            else float
                        )
                        if key == "vote_threshold":
                            if isinstance(comb_logic, dict):
                                high = gene_info.get("high")
                                low = gene_info.get("low", 1)
                            else:
                                high = gene_info.get("high")
                                low = gene_info.get("low", 1)
                                high = (
                                    min(high, n_active)
                                    if high is not None
                                    else n_active
                                )
                                low = max(1, low)
                        else:
                            high = gene_info.get("high")
                            low = gene_info.get("low", 1)
                        space_item = {"low": low, "high": high}
                        if "step" in gene_info:
                            space_item["step"] = gene_info["step"]

                    gene_space.append(space_item)
                    gene_types.append(gene_type)
                    gene_map[gene_index] = {
                        "name": gene_name,
                        "path": current_path,
                        "type": gene_type,
                    }
                    gene_index += 1
                elif isinstance(value, dict) or isinstance(value, list):
                    find_genes(value, current_path)
        elif isinstance(sub_config, list):
            for i, item in enumerate(sub_config):
                current_path = path + [i]
                find_genes(item, current_path)

    find_genes(rules, [])
    return gene_space, gene_map, gene_types
