import time

import numpy as np
import pandas as pd

import strategy_engine


def test_combine_signals_performance():
    rng = np.random.default_rng(0)
    n_signals = 20
    length = 10000
    signals = [pd.Series(rng.random(length) > 0.5) for _ in range(n_signals)]

    start = time.perf_counter()
    strategy_engine._combine_signals(signals, "VOTE")
    vote_time = time.perf_counter() - start

    start = time.perf_counter()
    strategy_engine._combine_signals(signals, "AND")
    and_time = time.perf_counter() - start

    start = time.perf_counter()
    strategy_engine._combine_signals(signals, "OR")
    or_time = time.perf_counter() - start

    assert vote_time < 0.05
    assert and_time < 0.02
    assert or_time < 0.02
