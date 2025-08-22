import sys
import types
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.modules.setdefault('vectorbt', types.ModuleType('vectorbt'))
sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))

import analysis  # noqa: E402


def test_persist_details_creates_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dummy_eval = types.SimpleNamespace(last_details={'a': 1})
    monkeypatch.setattr(analysis.subprocess, 'check_output', lambda *a, **k: 'abc123\n')
    out = analysis.persist_details(dummy_eval, charts_cfg={'run_ts': '123'})
    expected = Path('reports/123/details_abc123.json')
    assert out == expected
    assert expected.exists()
    with expected.open() as f:
        assert json.load(f) == {'a': 1}


def test_persist_details_respects_diag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dummy_eval = types.SimpleNamespace(last_details={'a': 1})
    monkeypatch.setattr(analysis.subprocess, 'check_output', lambda *a, **k: 'abc123\n')
    monkeypatch.delattr(analysis.config, 'DIAG', raising=False)
    monkeypatch.setattr(analysis.config, 'DIAGNOSTICS', {'persist_json': False})
    out = analysis.persist_details(dummy_eval, charts_cfg={'run_ts': '123'})
    assert out is None
    assert not Path('reports').exists()
