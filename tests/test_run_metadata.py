import json

from run_metadata import merge_run_metadata


def test_repeated_merges_accumulate(tmp_path):
    path = tmp_path / "run_metadata.json"
    merge_run_metadata(path, {"artifacts": ["a"], "settings": {"x": 1}})
    merge_run_metadata(path, {"artifacts": ["b"], "settings": {"y": 2}})
    data = json.loads(path.read_text())
    assert data["artifacts"] == ["a", "b"]
    assert data["settings"] == {"x": 1, "y": 2}
    assert data["metadata_version"] == 1


def test_corrupt_file_quarantined(tmp_path):
    path = tmp_path / "run_metadata.json"
    path.write_text("{bad json")
    merge_run_metadata(path, {"artifacts": ["x"]})
    data = json.loads(path.read_text())
    assert data["artifacts"] == ["x"]
    corrupts = list(tmp_path.glob("run_metadata.corrupt-*.json"))
    assert len(corrupts) == 1


def test_interleaved_writes_keep_artifacts(tmp_path):
    path = tmp_path / "run_metadata.json"
    merge_run_metadata(path, {"artifacts": ["a"], "settings": {"x": 1}})
    merge_run_metadata(path, {"artifacts": ["b"], "settings": {"x": 2}})
    data = json.loads(path.read_text())
    assert data["artifacts"] == ["a", "b"]
    assert data["settings"] == {"x": 2}


def test_duplicate_artifacts_deduped(tmp_path):
    path = tmp_path / "run_metadata.json"
    merge_run_metadata(path, {"artifacts": ["a", "b"]})
    merge_run_metadata(path, {"artifacts": ["b", "c"]})
    data = json.loads(path.read_text())
    assert data["artifacts"] == ["a", "b", "c"]


def test_shallow_merge_duplicate_keys(tmp_path):
    path = tmp_path / "run_metadata.json"
    merge_run_metadata(path, {"artifacts": [], "settings": {"a": 1, "b": 2}})
    merge_run_metadata(path, {"artifacts": [], "settings": {"b": 3, "c": 4}})
    data = json.loads(path.read_text())
    assert data["settings"] == {"a": 1, "b": 3, "c": 4}
