from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def merge_run_metadata(path: str | os.PathLike, new_meta: Dict[str, Any]) -> None:
    """Atomically merge ``new_meta`` into the JSON file at ``path``."""

    path = Path(path)
    lock_path = path.with_suffix(path.suffix + ".lock")

    deadline = time.time() + 5
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if time.time() > deadline:
                try:
                    os.remove(lock_path)
                except FileNotFoundError:
                    pass
                continue
            time.sleep(0.05)

    try:
        existing: Dict[str, Any] = {}
        if path.exists():
            try:
                with path.open() as fh:
                    existing = json.load(fh)
            except json.JSONDecodeError:
                ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                corrupt = path.with_name(f"{path.stem}.corrupt-{ts}.json")
                os.replace(path, corrupt)
                existing = {}

        arts = list(existing.get("artifacts", []))
        arts.extend(new_meta.get("artifacts", []))

        # Merge dictionaries shallowly, but keep nested dicts
        merged = existing
        for k, v in new_meta.items():
            if k == "artifacts":
                continue
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k].update(v)
            else:
                merged[k] = v

        seen: set[str] = set()
        uniq: list[str] = []
        for a in arts:
            if a in seen:
                continue
            seen.add(a)
            uniq.append(a)
        merged["artifacts"] = uniq
        merged["metadata_version"] = 1

        tmp_fd, tmp_name = tempfile.mkstemp(dir=str(path.parent))
        with os.fdopen(tmp_fd, "w") as fh:
            json.dump(merged, fh, indent=2)
        os.replace(tmp_name, path)
    finally:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass
