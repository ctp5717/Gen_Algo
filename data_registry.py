"""Shared, read-only data registry backed by NumPy memmaps."""

from __future__ import annotations

import atexit
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


class DataRegistry:
    """Materialise OHLCV windows as read-only memory maps."""

    def __init__(
        self,
        backend: str = "records",
        *,
        columnar_threshold: int = 12,
        schema_version: int = 1,
    ) -> None:
        self._lock = threading.Lock()
        self._root = Path(tempfile.mkdtemp(prefix="ga_registry_"))
        self._descriptors: dict[tuple[str, str], dict[str, Any]] = {}
        self._backend_preference = (backend or "records").lower()
        self._columnar_threshold = int(columnar_threshold)
        self._schema_version = int(schema_version)
        atexit.register(self.cleanup)

    # ------------------------------------------------------------------
    def _make_path(self, window_id: str, asset_id: str) -> Path:
        safe_asset = asset_id.replace(os.sep, "_")
        return self._root / f"{window_id}__{safe_asset}.npy"

    # ------------------------------------------------------------------
    def register_window(
        self, window_id: str, data_map: dict[str, pd.DataFrame]
    ) -> dict[str, dict[str, Any]]:
        descriptors: dict[str, dict[str, Any]] = {}
        for asset_id, df in data_map.items():
            descriptors[asset_id] = self.register_slice(window_id, asset_id, df)
        return descriptors

    # ------------------------------------------------------------------
    def _determine_backend(self, df: pd.DataFrame) -> str:
        if self._backend_preference == "auto":
            if df.shape[1] >= max(1, self._columnar_threshold):
                return "columnar"
            return "records"
        return self._backend_preference

    # ------------------------------------------------------------------
    def _store_records(
        self, window_id: str, asset_id: str, df: pd.DataFrame
    ) -> dict[str, Any]:
        records = df.to_records(index=True)
        path = self._make_path(window_id, asset_id)
        dtype = np.dtype(records.dtype)
        pickle = False
        if dtype.hasobject:
            np.save(str(path), records, allow_pickle=True)
            pickle = True
        else:
            memmap = np.lib.format.open_memmap(
                str(path), mode="w+", dtype=records.dtype, shape=records.shape
            )
            memmap[:] = records
            memmap.flush()
        return {
            "schema_version": self._schema_version,
            "window_id": window_id,
            "asset_id": asset_id,
            "backend": "records",
            "empty": False,
            "path": str(path),
            "dtype": records.dtype.descr,
            "fields": list(records.dtype.names or []),
            "index_field": records.dtype.names[0] if records.dtype.names else None,
            "columns": list(df.columns),
            "index_name": df.index.name,
            "pickle": pickle,
        }

    # ------------------------------------------------------------------
    def _store_columnar(
        self, window_id: str, asset_id: str, df: pd.DataFrame
    ) -> dict[str, Any]:
        if any(df[col].dtype.kind == "O" for col in df.columns):
            return self._store_records(window_id, asset_id, df)
        index_arr = df.index.to_numpy()
        if index_arr.dtype.kind == "O":
            return self._store_records(window_id, asset_id, df)

        safe_asset = asset_id.replace(os.sep, "_")
        column_paths: list[str] = []
        column_dtypes: list[str] = []
        for idx, col in enumerate(df.columns):
            arr = df[col].to_numpy()
            path = self._root / f"{window_id}__{safe_asset}__col{idx}.npy"
            memmap = np.lib.format.open_memmap(
                str(path), mode="w+", dtype=arr.dtype, shape=arr.shape
            )
            memmap[:] = arr
            memmap.flush()
            column_paths.append(str(path))
            column_dtypes.append(arr.dtype.str)

        index_path = self._root / f"{window_id}__{safe_asset}__index.npy"
        index_mem = np.lib.format.open_memmap(
            str(index_path), mode="w+", dtype=index_arr.dtype, shape=index_arr.shape
        )
        index_mem[:] = index_arr
        index_mem.flush()

        return {
            "schema_version": self._schema_version,
            "window_id": window_id,
            "asset_id": asset_id,
            "backend": "columnar",
            "empty": False,
            "columns": list(df.columns),
            "column_paths": column_paths,
            "column_dtypes": column_dtypes,
            "index_path": str(index_path),
            "index_dtype": index_arr.dtype.str,
            "index_name": df.index.name,
        }

    # ------------------------------------------------------------------
    def register_slice(
        self, window_id: str, asset_id: str, df: pd.DataFrame | None
    ) -> dict[str, Any]:
        key = (window_id, asset_id)
        with self._lock:
            if df is None or df.empty:
                descriptor = {
                    "schema_version": self._schema_version,
                    "window_id": window_id,
                    "asset_id": asset_id,
                    "backend": "records",
                    "empty": True,
                    "path": None,
                    "fields": [],
                    "columns": list(df.columns) if df is not None else [],
                    "index_name": None if df is None else df.index.name,
                }
                self._descriptors[key] = descriptor
                return descriptor

            backend = self._determine_backend(df)
            if backend == "columnar":
                descriptor = self._store_columnar(window_id, asset_id, df)
            else:
                descriptor = self._store_records(window_id, asset_id, df)
            self._descriptors[key] = descriptor
            return descriptor

    # ------------------------------------------------------------------
    def describe(self, window_id: str, asset_id: str) -> dict[str, Any]:
        return dict(self._descriptors[(window_id, asset_id)])

    # ------------------------------------------------------------------
    @staticmethod
    def attach(descriptor: dict[str, Any]) -> pd.DataFrame:
        if descriptor.get("empty"):
            columns = descriptor.get("columns", [])
            idx_name = descriptor.get("index_name")
            df = pd.DataFrame(columns=columns)
            if idx_name is not None:
                df.index.name = idx_name
            return df

        backend = descriptor.get("backend", "records")
        if backend == "columnar":
            columns = descriptor.get("columns", [])
            arrays: dict[str, Any] = {}
            for col, path, _dtype_str in zip(
                columns,
                descriptor.get("column_paths", []),
                descriptor.get("column_dtypes", []),
                strict=False,
            ):
                arrays[col] = np.lib.format.open_memmap(str(path), mode="r")

            index_mem = np.lib.format.open_memmap(
                str(descriptor.get("index_path")), mode="r"
            )
            df = pd.DataFrame(arrays, index=pd.Index(index_mem))
            idx_name = descriptor.get("index_name")
            if idx_name is not None:
                df.index.name = idx_name
            return df.set_flags(allows_duplicate_labels=False)

        path = descriptor["path"]
        if descriptor.get("pickle"):
            arr = np.load(path, allow_pickle=True)
        else:
            arr = np.lib.format.open_memmap(path, mode="r")
        index_field = descriptor.get("index_field")
        df = pd.DataFrame.from_records(arr, index=index_field)
        if descriptor.get("columns"):
            df.columns = descriptor["columns"]
        idx_name = descriptor.get("index_name")
        if idx_name is not None:
            df.index.name = idx_name
        return df.set_flags(allows_duplicate_labels=False)

    # ------------------------------------------------------------------
    def release_window(self, window_id: str) -> None:
        with self._lock:
            to_remove = [key for key in self._descriptors if key[0] == window_id]
            for key in to_remove:
                descriptor = self._descriptors.pop(key)
                path = descriptor.get("path")
                if path:
                    try:
                        os.remove(path)
                    except FileNotFoundError:  # pragma: no cover - benign race
                        pass
                if descriptor.get("backend") == "columnar":
                    for extra in descriptor.get("column_paths", []):
                        try:
                            os.remove(extra)
                        except FileNotFoundError:  # pragma: no cover - benign race
                            pass
                    index_path = descriptor.get("index_path")
                    if index_path:
                        try:
                            os.remove(index_path)
                        except FileNotFoundError:  # pragma: no cover - benign race
                            pass

    # ------------------------------------------------------------------
    def cleanup(self) -> None:
        with self._lock:
            for descriptor in self._descriptors.values():
                path = descriptor.get("path")
                if path:
                    try:
                        os.remove(path)
                    except FileNotFoundError:  # pragma: no cover - benign race
                        pass
                if descriptor.get("backend") == "columnar":
                    for extra in descriptor.get("column_paths", []):
                        try:
                            os.remove(extra)
                        except FileNotFoundError:  # pragma: no cover - benign race
                            pass
                    index_path = descriptor.get("index_path")
                    if index_path:
                        try:
                            os.remove(index_path)
                        except FileNotFoundError:  # pragma: no cover - benign race
                            pass
            self._descriptors.clear()
        try:
            self._root.rmdir()
        except OSError:  # pragma: no cover - directory may not be empty
            pass


_cfg = getattr(config, "DATA_REGISTRY", {})
_backend_choice = _cfg.get("backend", "auto")
_threshold = int(_cfg.get("columnar_threshold", 12) or 12)
_schema_version = int(_cfg.get("schema_version", 1) or 1)

registry = DataRegistry(
    backend=_backend_choice,
    columnar_threshold=_threshold,
    schema_version=_schema_version,
)
