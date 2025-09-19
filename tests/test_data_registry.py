import pandas as pd

from data_registry import DataRegistry


def test_columnar_backend_roundtrip():
    registry = DataRegistry(backend="auto", columnar_threshold=2)
    df = pd.DataFrame({f"col{i}": [float(i), float(i) + 1.0] for i in range(4)})
    descriptor = registry.register_slice("window-1", "asset-A", df)
    assert descriptor["backend"] == "columnar"
    restored = DataRegistry.attach(descriptor)
    pd.testing.assert_frame_equal(restored, df, check_flags=False)
    registry.release_window("window-1")
    registry.cleanup()


def test_records_backend_for_object_columns():
    registry = DataRegistry(backend="auto", columnar_threshold=2)
    df = pd.DataFrame({"Close": [1.0, 2.0], "Label": ["buy", "sell"]})
    descriptor = registry.register_slice("window-2", "asset-B", df)
    assert descriptor["backend"] == "records"
    restored = DataRegistry.attach(descriptor)
    expected = df.copy()
    expected.index.name = "index"
    pd.testing.assert_frame_equal(restored, expected, check_flags=False)
    registry.release_window("window-2")
    registry.cleanup()
