import json
from datetime import datetime, timezone

import analysis


def test_write_run_metadata_extra(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(analysis, "_get_cache_hashes", lambda: {})
    analysis.set_run_dir(tmp_path)

    class _VBT:
        __version__ = "0.0.0"
        __file__ = __file__

    monkeypatch.setattr(analysis, "vbt", _VBT)
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    extra = {
        "combination_logic": "AND",
        "vote_threshold": None,
        "per_asset_signal_counts": {"A": {"rule": 3}},
    }
    (tmp_path / "foo.png").write_text("x")
    analysis._write_run_metadata(start, ["foo.png"], extra)
    with open("run_metadata.json") as fh:
        meta = json.load(fh)
    assert meta["combination_logic"] == "AND"
    assert meta["per_asset_signal_counts"]["A"]["rule"] == 3
    assert "foo.png" in meta["artifacts"]


def test_write_run_metadata_skips_missing_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(analysis, "_get_cache_hashes", lambda: {})
    analysis.set_run_dir(tmp_path)

    class _VBT:
        __version__ = "0.0.0"
        __file__ = __file__

    monkeypatch.setattr(analysis, "vbt", _VBT)
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)

    existing = tmp_path / "exists.png"
    existing.write_text("x")
    analysis._write_run_metadata(start, [str(existing), str(tmp_path / "missing.png")])
    with open("run_metadata.json") as fh:
        meta = json.load(fh)
    assert meta["artifacts"] == [str(existing)]
