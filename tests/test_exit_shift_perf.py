import time

import numpy as np
import pandas as pd


def test_shift_without_reindex_is_faster():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.random(100_000) > 0.5)

    start = time.perf_counter()
    for _ in range(200):
        s.shift(1, fill_value=False)
    no_reindex = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(200):
        tmp = s.shift(1, fill_value=False)
        tmp.reindex(s.index, fill_value=False)
    with_reindex = time.perf_counter() - start

    assert no_reindex <= with_reindex * 0.8
