import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if os.getenv("USE_VBT_STUB") == "1":
    import vbt_stub as vbt

    # Ensure tests use the lightweight vectorbt stub
    sys.modules["vectorbt"] = vbt
