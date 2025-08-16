import sys
import types
from pathlib import Path
import pytest

# Ensure repository root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy optional dependencies
sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))
sys.modules.setdefault('vectorbt', types.ModuleType('vectorbt'))

import main  # noqa: E402


def test_abort_when_limits_exceeded(monkeypatch):
    monkeypatch.setattr(main.config, 'LIMITS', {'max_assets': 1, 'max_mc_runs': 1}, raising=False)
    monkeypatch.setattr(main.config, 'ASSET_GROUP', [('A', 'A-USD'), ('B', 'B-USD')], raising=False)
    scanner = {
        'max_concurrent_trades': 1,
        'tie_break_policy': 'fifo',
        'score_func': 'pct_change',
        'score_scaling': None,
        'monte_carlo_runs': 2,
        'seed': 123,
        'verbose': False,
        'verbose_top_n': 5,
    }
    monkeypatch.setattr(main.config, 'SCANNER', scanner, raising=False)

    with pytest.warns(UserWarning):
        main.main()
