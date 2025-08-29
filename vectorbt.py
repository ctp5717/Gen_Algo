"""A lightweight stub of the :mod:`vectorbt` package used in tests.

This stub avoids importing the real heavy dependency while exposing the
very small surface that the project needs. It defines a minimal
``Portfolio`` class with a ``from_signals`` constructor and provides a
``__version__`` attribute so metadata helpers can record it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

__all__ = ["Portfolio", "__version__"]

__version__ = "0.0.0"


@dataclass
class _Trades:
    count_value: int = 0

    def count(self) -> int:
        return self.count_value


class Portfolio:
    """Tiny substitute for :class:`vectorbt.Portfolio`.

    Only the pieces exercised in the unit tests are implemented. Instances
    expose ``trades.count()`` and a ``stats`` method. ``from_signals``
    simply returns an empty portfolio but is designed to be monkeypatched
    in tests.
    """

    def __init__(self, trades: int = 0, stats: Dict[str, Any] | None = None):
        self._trades = _Trades(trades)
        self._stats = stats or {}

    @property
    def trades(self) -> _Trades:  # pragma: no cover - simple property
        return self._trades

    def stats(
        self, *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:  # pragma: no cover - passthrough
        return self._stats

    @classmethod
    def from_signals(
        cls, *args: Any, **kwargs: Any
    ) -> "Portfolio":  # pragma: no cover - deterministic default
        return cls()
