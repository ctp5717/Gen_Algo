import sys

def test_python_is_313_or_newer():
    assert sys.version_info >= (3, 13), f"Python 3.13+ required, found {sys.version}"
