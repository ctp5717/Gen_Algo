"""Utility helpers for Genetic Algorithm runs."""

import config


class StagnationCallback:
    """Callable used by ``pygad`` to handle stagnant fitness values.

    Whenever the best fitness of a generation equals ``-999`` the callback
    increments an internal counter.  After ``GA_STAGNATION_THRESHOLD``
    consecutive stagnant generations the mutation strength of the running GA
    instance is increased to encourage exploration.  If ``GA_RESTART_POLICY`` is
    set to ``"expand"`` each gene's search range is widened before the mutation
    escalation is applied.  This avoids calling internal ``pygad`` methods whose
    signatures frequently change between releases.
    """

    def __init__(self):
        self.count = 0

    def __call__(self, ga_instance):
        best_fitness = ga_instance.best_solution(
            pop_fitness=ga_instance.last_generation_fitness
        )[1]
        if best_fitness == -999:
            self.count += 1
            threshold = getattr(config, "GA_STAGNATION_THRESHOLD", 0)
            if self.count >= threshold:
                policy = getattr(config, "GA_RESTART_POLICY", "restart").lower()
                if policy == "expand":
                    expansion = getattr(config, "GA_GENE_RANGE_EXPANSION", 0.5)
                    for gene in ga_instance.gene_space:
                        if isinstance(gene, dict):
                            span = gene["high"] - gene["low"]
                            gene["low"] -= span * expansion
                            gene["high"] += span * expansion

                factor = getattr(config, "GA_MUTATION_ESCALATION_FACTOR", 2.0)
                current = getattr(ga_instance, "mutation_num_genes", 1)
                ga_instance.mutation_num_genes = min(
                    ga_instance.num_genes,
                    max(1, int(current * factor)),
                )
                self.count = 0
                print(
                    "\nMutation strength increased due to stagnant fitness (-999)."
                )
        else:
            self.count = 0


def make_stagnation_callback():
    """Return a picklable stagnation callback function.

    ``pygad.GA`` expects lifecycle callbacks like ``on_generation`` to either be
    a plain function or a bound method. Previously this helper returned an
    instance of :class:`StagnationCallback`. While the instance is callable, it
    does not expose a ``__code__`` attribute which ``pygad`` requires during
    initialization, leading to an ``AttributeError``.  By returning the bound
    ``__call__`` method instead, the callback satisfies ``pygad``'s interface
    (it is recognised as a method with the correct signature) while still
    maintaining internal state and remaining picklable.
    """

    # Return the bound ``__call__`` method so ``pygad`` treats it as a method.
    return StagnationCallback().__call__
