"""Utility helpers for Genetic Algorithm runs."""

import config


def make_stagnation_callback():
    """Return a callback that restarts the GA when fitness stagnates at ``-999``.

    The returned callable tracks consecutive generations where the best fitness
    equals ``-999``. Once the number of stagnant generations reaches
    ``config.GA_STAGNATION_THRESHOLD``, the policy specified by
    ``config.GA_RESTART_POLICY`` is applied:

    - ``"restart"``: the population is simply randomised.
    - ``"expand"``: each gene's ``low`` and ``high`` bounds are widened by
      ``config.GA_GENE_RANGE_EXPANSION`` (as a fraction of the current range)
      before randomising the population.
    """

    counter = {"count": 0}

    def _callback(ga_instance):
        best_fitness = ga_instance.best_solution(
            pop_fitness=ga_instance.last_generation_fitness
        )[1]
        if best_fitness == -999:
            counter["count"] += 1
            if counter["count"] >= getattr(config, "GA_STAGNATION_THRESHOLD", 0):
                policy = getattr(config, "GA_RESTART_POLICY", "restart").lower()
                if policy == "expand":
                    expansion = getattr(config, "GA_GENE_RANGE_EXPANSION", 0.5)
                    for gene in ga_instance.gene_space:
                        if isinstance(gene, dict):
                            span = gene["high"] - gene["low"]
                            gene["low"] -= span * expansion
                            gene["high"] += span * expansion
                if hasattr(ga_instance, "initialize_population"):
                    ga_instance.initialize_population()
                counter["count"] = 0
                print(
                    "\nPopulation restarted due to stagnant fitness (-999)."
                )
        else:
            counter["count"] = 0

    return _callback
