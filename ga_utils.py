"""Utility helpers for Genetic Algorithm runs."""

import config


class StagnationCallback:
    """Callable object used to restart the GA when fitness stagnates.

    The object tracks consecutive generations where the best fitness equals
    ``-999``. Once the number of stagnant generations reaches
    ``config.GA_STAGNATION_THRESHOLD``, the policy specified by
    ``config.GA_RESTART_POLICY`` is applied:

    - ``"restart"``: the population is simply randomised.
    - ``"expand"``: each gene's ``low`` and ``high`` bounds are widened by
      ``config.GA_GENE_RANGE_EXPANSION`` (as a fraction of the current range)
      before randomising the population.

    The class is defined at module level so that instances are picklable. This
    is important because ``pygad`` evaluates fitness functions in worker
    processes and thus needs to pickle the GA instance, including any
    callbacks.
    """

    def __init__(self):
        self.count = 0

    def __call__(self, ga_instance):
        best_fitness = ga_instance.best_solution(
            pop_fitness=ga_instance.last_generation_fitness
        )[1]
        if best_fitness == -999:
            self.count += 1
            if self.count >= getattr(config, "GA_STAGNATION_THRESHOLD", 0):
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
                self.count = 0
                print(
                    "\nPopulation restarted due to stagnant fitness (-999)."
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
