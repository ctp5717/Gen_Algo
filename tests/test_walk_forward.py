import sys
import types
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy optional dependencies
sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))
sys.modules.setdefault('vectorbt', types.ModuleType('vectorbt'))

import walk_forward  # noqa: E402
from dateutil.relativedelta import relativedelta  # noqa: E402


def test_generate_periods_produces_windows():
    start = datetime(2020, 1, 1)
    end = datetime(2021, 6, 1)
    periods = walk_forward._generate_periods(start, end, train_months=12, test_months=3)
    assert len(periods) > 0


def test_generate_periods_insufficient_data():
    start = datetime(2020, 1, 1)
    end = datetime(2020, 3, 1)
    periods = walk_forward._generate_periods(start, end, train_months=3, test_months=3)
    assert periods == []

def test_generate_periods_window_consistency():
    start = datetime(2020, 1, 1)
    end = datetime(2020, 12, 31)
    train_months = 6
    test_months = 2
    periods = walk_forward._generate_periods(start, end, train_months, test_months)
    assert periods
    for idx, p in enumerate(periods):
        assert p['train_end'] == p['train_start'] + relativedelta(months=train_months)
        assert p['test_start'] == p['train_end']
        assert p['test_end'] == p['test_start'] + relativedelta(months=test_months)
        if idx > 0:
            expected_start = periods[idx-1]['train_start'] + relativedelta(months=test_months)
            assert p['train_start'] == expected_start
