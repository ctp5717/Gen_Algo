import types

import pytest

import global_executor


@pytest.fixture(autouse=True)
def _reset_executor():
    global_executor.shutdown()
    yield
    global_executor.shutdown()


def test_record_batch_metrics_respects_memory_target(monkeypatch):
    class DummyExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def submit(self, *args, **kwargs):  # pragma: no cover - no submissions expected
            raise AssertionError("submit should not be called in this test")

        def shutdown(self, wait=True):  # pragma: no cover - cleanup hook
            return None

    monkeypatch.setattr(global_executor, "cf", types.SimpleNamespace(ProcessPoolExecutor=DummyExecutor))
    monkeypatch.setattr(global_executor.mp, "get_context", lambda method: types.SimpleNamespace())

    global_executor.create(
        {
            "max_workers": 2,
            "memory_target_gib": 0.001,
            "in_flight_cap": 8,
        },
        force=True,
    )

    cap_before = global_executor.current_in_flight_cap()
    assert cap_before == 8

    new_cap = global_executor.record_batch_metrics(800_000)
    assert new_cap < cap_before
    metrics = global_executor.metrics()
    assert metrics["in_flight_cap"] == new_cap
    assert metrics["bytes_avg"] > 0
