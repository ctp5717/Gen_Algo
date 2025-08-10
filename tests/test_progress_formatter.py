import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))
sys.modules.setdefault('vectorbt', types.ModuleType('vectorbt'))

from main import format_progress_line  # noqa: E402


def test_progress_formatter_no_trailing_characters():
    for est in [0, 5, 123]:
        line = format_progress_line(1, 10, 0.1234, est)
        assert line == line.rstrip()
        assert line.endswith('s')
        assert '\n' not in line and '\r' not in line
