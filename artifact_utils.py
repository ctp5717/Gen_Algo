import json
from pathlib import Path

ARTIFACTS_DIR = Path('artifacts')
MANIFEST_PATH = ARTIFACTS_DIR / 'manifest.json'


def append_to_manifest(path: Path) -> None:
    """Append a file path to the artifacts manifest."""
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    if MANIFEST_PATH.exists():
        try:
            data = json.loads(MANIFEST_PATH.read_text())
        except json.JSONDecodeError:
            data = []
    else:
        data = []
    data.append(str(path))
    MANIFEST_PATH.write_text(json.dumps(data, indent=2))
