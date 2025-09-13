import sys


def test_python_is_312_or_newer():
    assert sys.version_info >= (3, 12), f"Python 3.12+ required, found {sys.version}"
