import multiprocessing as mp
import sys
import types

import pandas as pd


def _worker(q):
    ta_stub = types.ModuleType("pandas_ta")

    class TAStub:
        def ema(self, length=1):  # noqa: ANN001, D401
            return pd.Series([1, 2], name="EMA")

    pd.DataFrame.ta = TAStub()
    sys.modules["pandas_ta"] = ta_stub
    import indicator_library as il  # noqa: WPS433, F401

    df = pd.DataFrame(
        {
            "Open": [1, 2],
            "High": [1, 2],
            "Low": [1, 2],
            "Close": [1, 2],
        }
    )
    try:
        il.calculate_ema(df, period=1)
        q.put(True)
    except AttributeError as e:
        q.put(str(e))


def test_indicator_in_worker_has_ta_accessor():
    mp.set_start_method("spawn", force=True)
    q: mp.Queue = mp.Queue()
    p = mp.Process(target=_worker, args=(q,))
    p.start()
    p.join(timeout=5)
    assert p.exitcode == 0
    result = q.get()
    assert result is True, result
