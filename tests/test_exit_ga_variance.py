import numpy as np
import pygad

from gene_parser import parse_genes_from_config
from strategy_rules import STRATEGY_RULES


def _to_numeric(value):
    if isinstance(value, (int, float, np.number)):
        return float(value)
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, str):
        mapping = {"none": 0.0, "breakeven": 1.0, "follow_tp": 2.0}
        return mapping.get(value.lower(), 0.0)
    return 0.0


def test_exit_genes_exhibit_variance_after_ga_run():
    gene_space, gene_map, gene_types = parse_genes_from_config(STRATEGY_RULES)
    target_order = [
        "num_tp_levels",
        "tp_pct_1",
        "tp_pct_2",
        "tp_pct_3",
        "tp_pct_4",
        "sl_timeout_bars",
        "sl_timeout_enabled",
        "sl_trailing_pct",
        "sl_trailing_enabled",
        "sl_break_even_mode",
    ]

    target_specs: list = []
    target_types: list[type] = []
    indices: dict[str, int] = {}
    option_mappings: dict[int, dict[float, object]] = {}

    for idx, info in gene_map.items():
        name = info["name"]
        if name in target_order:
            gene_idx = len(target_specs)
            spec = gene_space[idx]
            mapping: dict[float, object] | None = None
            if isinstance(spec, dict) and "options" in spec:
                options = list(spec["options"])
                if all(isinstance(opt, (int, float, bool)) for opt in options):
                    normalized = [float(opt) for opt in options]
                else:
                    mapping = {float(i): opt for i, opt in enumerate(options)}
                    normalized = list(mapping)
            elif isinstance(spec, dict):
                low = spec.get("low")
                high = spec.get("high")
                if low is not None and high is not None:
                    normalized = {"low": low, "high": high}
                else:
                    normalized = spec
            else:
                normalized = spec

            indices[name] = gene_idx
            target_specs.append(normalized)
            target_types.append(float)
            if mapping is not None:
                option_mappings[gene_idx] = mapping

    assert len(target_specs) == len(target_order)

    def fitness_func(ga_instance, solution, solution_idx):
        score = 0.0
        for value in solution:
            score += _to_numeric(value)
        return float(score)

    ga = pygad.GA(
        num_generations=3,
        sol_per_pop=12,
        num_parents_mating=6,
        mutation_num_genes=4,
        mutation_probability=0.2,
        random_seed=42,
        gene_space=target_specs,
        gene_type=target_types,
        num_genes=len(target_specs),
        fitness_func=fitness_func,
    )
    ga.run()

    population = ga.population
    assert population is not None and len(population) > 1

    for name in target_order:
        idx = indices[name]
        values = [chrom[idx] for chrom in population]
        if idx in option_mappings:
            mapped = [option_mappings[idx].get(float(val), val) for val in values]
            assert len(set(mapped)) > 1, f"GA failed to vary {name}"
            continue
        if all(isinstance(v, str) for v in values):
            assert len(set(values)) > 1, f"GA failed to vary {name}"
        else:
            arr = np.asarray(values, dtype=float)
            assert np.var(arr) > 0, f"GA variance for {name} is zero"
