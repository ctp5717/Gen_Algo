"""Indicator column contracts and output normalization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Callable, Dict, List, Mapping as TypingMapping

import pandas as pd


class IndicatorContractError(Exception):
    """Raised when an indicator output violates its column contract."""

    def __init__(
        self, indicator: str, return_type: str, length: int | None, message: str
    ):
        self.indicator = indicator
        self.return_type = return_type
        self.length = length
        super().__init__(f"{indicator} returned {return_type} len={length}: {message}")


def _fmt(x: float | int) -> str:
    """Format floats/ints so 2 and 2.0 yield '2.0'."""
    if isinstance(x, int):
        return f"{x}.0"
    d = Decimal(str(x)).quantize(Decimal("0.000000"), rounding=ROUND_HALF_UP)
    s = f"{d.normalize()}"
    return s if "." in s else s + ".0"


def _macd_contract(
    fast: int = 12, slow: int = 26, signal: int = 9, **_: Any
) -> List[str]:
    base = f"{fast}_{slow}_{signal}"
    return [f"MACD_{base}", f"MACDh_{base}", f"MACDs_{base}"]


def _stoch_contract(k: int = 14, d: int = 3, smooth_k: int = 3, **_: Any) -> List[str]:
    base = f"{k}_{d}_{smooth_k}"
    return [f"STOCHk_{base}", f"STOCHd_{base}", f"STOCHh_{base}"]


def _adx_contract(period: int = 14, **_: Any) -> List[str]:
    p = period
    return [f"ADX_{p}", f"ADXR_{p}_2", f"DMP_{p}", f"DMN_{p}"]


def _bbands_contract(period: int = 20, std_dev: float = 2.0, **_: Any) -> List[str]:
    s = _fmt(std_dev)
    base = f"{period}_{s}_{s}"
    return [
        f"BBL_{base}",
        f"BBM_{base}",
        f"BBU_{base}",
        f"BBB_{base}",
        f"BBP_{base}",
    ]


def _psar_contract(
    acc: float = 0.02,
    maximum: float = 0.2,
    acceleration: float | None = None,
    **_: Any,
) -> List[str]:
    if acceleration is not None:
        acc = acceleration
    base = f"{_fmt(acc)}_{_fmt(maximum)}"
    return [
        f"PSARl_{base}",
        f"PSARs_{base}",
        f"PSARaf_{base}",
        f"PSARr_{base}",
    ]


def _keltner_contract(period: int = 20, multiplier: float = 2.0, **_: Any) -> List[str]:
    base = f"{period}_{_fmt(multiplier)}"
    return [f"KCLe_{base}", f"KCBe_{base}", f"KCUe_{base}"]


def _donchian_contract(
    period: int = 20, offset: int | None = None, **_: Any
) -> List[str]:
    off = offset if offset is not None else period
    base = f"{period}_{off}"
    return [f"DCL_{base}", f"DCM_{base}", f"DCU_{base}"]


def _ma_envelope_contract(
    period: int = 20, percent: float = 2.0, **_: Any
) -> List[str]:
    try:
        pct = float(percent)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        pct = percent
    pct_frac = pct / 100 if isinstance(pct, (int, float)) and pct > 1 else pct
    if isinstance(pct_frac, (int, float)):
        pct_frac = round(pct_frac, 10)
        pct_display = round(pct_frac * 100, 10)
    else:  # pragma: no cover - fallback when percent cannot be coerced
        pct_display = pct_frac
    percent_str = f"{pct_display:.10g}" if isinstance(pct_display, (int, float)) else str(pct_display)
    if "." not in percent_str:
        percent_str += ".0"
    base = f"{period}_{percent_str}"
    return [f"MAE_L_{base}", f"MAE_M_{base}", f"MAE_U_{base}"]


def _trix_contract(
    period: int = 15, signal: int | None = None, **_: Any
) -> List[str] | None:
    if signal is None:
        return None
    base = f"{period}_{signal}"
    return [f"TRIX_{base}", f"TRIXs_{base}"]


def _ichimoku_contract(
    tenkan: int = 9, kijun: int = 26, senkou: int = 52, **_: Any
) -> List[str]:
    return [
        f"IKS_{kijun}",
        f"ITS_{tenkan}",
        f"ISA_{tenkan}",
        f"ISB_{kijun}",
    ]


CONTRACTS: Dict[str, Callable[..., List[str] | None]] = {
    "macd": _macd_contract,
    "stoch": _stoch_contract,
    "adx": _adx_contract,
    "bbands": _bbands_contract,
    "psar": _psar_contract,
    "keltner": _keltner_contract,
    "donchian": _donchian_contract,
    "ma_envelope": _ma_envelope_contract,
    "trix": _trix_contract,
    "ichimoku": _ichimoku_contract,
}


@dataclass(frozen=True)
class OutputSchema:
    """Describe indicator output columns and semantic roles."""

    columns: list[str] = field(default_factory=list)
    default: str | None = None
    roles: dict[str, str] = field(default_factory=dict)
    priority: list[str] = field(default_factory=list)


def describe_output(indicator: str, params: TypingMapping[str, Any]) -> OutputSchema:
    """Return canonical column metadata for ``indicator`` given ``params``."""

    name = indicator.lower()
    contract = CONTRACTS.get(name)
    expected = contract(**params) if contract else None
    columns = list(expected) if expected else []
    roles: dict[str, str] = {}
    priority: list[str] = []
    default: str | None = None

    if name == "bbands" and len(columns) >= 3:
        roles = {"lower": columns[0], "middle": columns[1], "upper": columns[2]}
        default = columns[1]
        priority = [columns[1], columns[0], columns[2]]
    elif name == "keltner" and len(columns) >= 3:
        roles = {"lower": columns[0], "middle": columns[1], "upper": columns[2]}
        default = columns[1]
        priority = [columns[1], columns[0], columns[2]]
    elif name == "donchian" and len(columns) >= 3:
        roles = {"lower": columns[0], "middle": columns[1], "upper": columns[2]}
        default = columns[1]
        priority = [columns[1], columns[0], columns[2]]
    elif name == "ma_envelope" and len(columns) >= 3:
        roles = {"lower": columns[0], "middle": columns[1], "upper": columns[2]}
        default = columns[1]
        priority = [columns[1], columns[0], columns[2]]
    elif name == "macd" and len(columns) >= 3:
        roles = {
            "line": columns[0],
            "histogram": columns[1],
            "signal": columns[2],
        }
        default = columns[1]
        priority = [columns[1], columns[0], columns[2]]
    elif name == "stoch" and columns:
        roles = {"k": columns[0]}
        if len(columns) > 1:
            roles["d"] = columns[1]
        if len(columns) > 2:
            roles["histogram"] = columns[2]
        default = columns[0]
        priority = [col for col in columns[:3] if col is not None]
    elif name == "adx" and columns:
        roles = {"main": columns[0]}
        default = columns[0]
        priority = [columns[0]]
    elif name == "ichimoku" and columns:
        roles = {"baseline": columns[0]}
        if len(columns) > 1:
            roles["conversion"] = columns[1]
        if len(columns) > 2:
            roles["span_a"] = columns[2]
        if len(columns) > 3:
            roles["span_b"] = columns[3]
        default = columns[0]
        priority = [columns[0]]
    elif name == "trix" and columns:
        roles = {"line": columns[0]}
        default = columns[0]
        priority = [columns[0]]
        if len(columns) > 1:
            roles["signal"] = columns[1]
            priority.append(columns[1])
    elif name in {"pivot_points", "pivots"}:
        roles = {"pivot": "P"}
        default = "P"
        priority = ["P"]
    else:
        default = columns[0] if columns else None
        if default:
            priority = [default]

    roles = {key: value for key, value in roles.items() if value}
    return OutputSchema(columns=columns, default=default, roles=roles, priority=priority)


def normalize_output(
    indicator: str,
    output: Any,
    params: Mapping[str, Any],
    index: pd.Index | None = None,
) -> pd.Series | pd.DataFrame:
    """Normalize an indicator output to a pandas object using its contract.

    Parameters
    ----------
    indicator : str
        Indicator name used to look up the contract.
    output : Any
        Raw output from the indicator function.
    params : Mapping[str, Any]
        Parameters passed to the indicator.
    """

    contract = CONTRACTS.get(indicator)
    expected = contract(**params) if contract else None

    if isinstance(output, pd.Series):
        if expected and len(expected) != 1:
            raise IndicatorContractError(
                indicator,
                type(output).__name__,
                1,
                f"expected {len(expected)} columns, got 1",
            )
        if index is not None and not output.index.equals(index):
            output = output.copy()
            output.index = index
        return output

    if isinstance(output, pd.DataFrame):
        if expected:
            cols = list(output.columns)
            mapping: Dict[str, str] = {}
            missing = []
            for exp in expected:
                if exp in cols:
                    mapping[exp] = exp
                else:
                    alt = exp.replace(".0", "")
                    if alt in cols:
                        mapping[exp] = alt
                    else:
                        missing.append(exp)
            if missing:
                raise IndicatorContractError(
                    indicator,
                    "DataFrame",
                    output.shape[1],
                    f"missing columns: {sorted(missing)}",
                )
            output = output.loc[:, list(mapping.values())].copy()
            output.columns = list(mapping.keys())
        if index is not None and not output.index.equals(index):
            output = output.copy()
            output.index = index
        return output

    if hasattr(output, "_asdict"):
        output = output._asdict()

    if isinstance(output, Mapping):
        if expected:
            try:
                data = {col: output[col] for col in expected}
            except KeyError as ke:
                raise IndicatorContractError(
                    indicator, "dict", len(output), f"missing key: {ke}"
                ) from ke
        else:
            data = dict(output)
        df = pd.DataFrame(data)
        if index is not None and not df.index.equals(index):
            df.index = index
        return df

    if isinstance(output, tuple):
        if not expected:
            raise IndicatorContractError(
                indicator,
                "tuple",
                len(output),
                "no contract for tuple output",
            )
        if len(output) != len(expected):
            raise IndicatorContractError(
                indicator,
                "tuple",
                len(output),
                f"expected {len(expected)} values, got {len(output)}",
            )
        data = {col: output[i] for i, col in enumerate(expected)}
        df = pd.DataFrame(data)
        if index is not None and not df.index.equals(index):
            df.index = index
        return df

    length = len(output) if hasattr(output, "__len__") else None
    raise IndicatorContractError(
        indicator,
        type(output).__name__,
        length,
        f"unsupported return type {type(output).__name__}",
    )
