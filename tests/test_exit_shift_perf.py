import time

import numpy as np
import pandas as pd


def _bench(fn, repeat=5):
    """Return the best runtime for ``fn`` across ``repeat`` executions."""

    times: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def test_shift_without_reindex_is_faster():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.random(100_000) > 0.5)

    def shift_only():
        for _ in range(200):
            s.shift(1, fill_value=False)

    def shift_then_reindex():
        for _ in range(200):
            s.shift(1, fill_value=False).reindex(s.index, fill_value=False)

    no_reindex = _bench(shift_only)
    with_reindex = _bench(shift_then_reindex)

    # The shift-only path should not be measurably slower than performing an
    # extra reindex. Allow a small tolerance to account for timing noise on
    # shared CI runners.
    assert no_reindex <= with_reindex * 1.05
