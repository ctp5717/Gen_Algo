# config.py

"""
Configuration File for the GA Trading Framework
(This version includes automated rolling date ranges that adapt to the selected timeframe)
"""

import math
import os
import warnings
from datetime import datetime
from typing import Any

from dateutil.relativedelta import relativedelta

from strategy_rules import STRATEGY_RULES


class ConfigurationError(ValueError):
    """Raised when configuration options are invalid."""


KNOWN_ASSET_CLASSES = {
    "Stars",
    "Stalwarts",
    "Gambles",
    "Borderline",
    "Drags",
    "Insufficient Data",
}


def validate_final_strategy_config(cfg: dict[str, Any] | None = None) -> None:
    """Validate final strategy synthesis configuration values."""

    cfg = cfg or globals().get("FINAL_STRATEGY", {})
    if not isinstance(cfg, dict):
        raise ConfigurationError("FINAL_STRATEGY must be a dictionary")

    include_classes = cfg.get("INCLUDE_CLASSES", [])
    if not isinstance(include_classes, (list, tuple, set)):
        raise ConfigurationError("INCLUDE_CLASSES must be a sequence of class names")
    include_classes_list = list(include_classes)
    include_unknown: list[str] = []
    known_lower = {cls.lower() for cls in KNOWN_ASSET_CLASSES}
    for cls in include_classes_list:
        if not isinstance(cls, str):
            raise ConfigurationError("INCLUDE_CLASSES entries must be strings")
        normalized = cls.strip()
        if not normalized:
            raise ConfigurationError(
                "INCLUDE_CLASSES entries must not be empty strings"
            )
        if normalized.lower() not in known_lower:
            include_unknown.append(normalized)
    if include_unknown:
        warnings.warn(
            "Unknown FINAL_STRATEGY INCLUDE_CLASSES entries: "
            + ", ".join(sorted(set(include_unknown))),
            UserWarning,
            stacklevel=2,
        )

    watch = float(cfg.get("PARAM_RCV_WATCHLIST", 0.0))
    unstable = float(cfg.get("PARAM_RCV_UNSTABLE", 0.0))
    if watch >= unstable:
        raise ConfigurationError("PARAM_RCV_WATCHLIST must be < PARAM_RCV_UNSTABLE")

    scheme = cfg.get("WEIGHTING_SCHEME", "risk_adjusted")
    allowed_schemes = {"equal", "proportional", "risk_adjusted", "override"}
    if scheme not in allowed_schemes:
        raise ConfigurationError(
            "WEIGHTING_SCHEME must be one of: equal | proportional | risk_adjusted | override"
        )

    overrides = cfg.get("ASSET_WEIGHTS_OVERRIDE", {})
    if scheme == "override":
        if not overrides:
            raise ConfigurationError(
                "ASSET_WEIGHTS_OVERRIDE must define weights when WEIGHTING_SCHEME is 'override'"
            )
        total = sum(float(v) for v in overrides.values())
        if total <= 0 or not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ConfigurationError(
                "ASSET_WEIGHTS_OVERRIDE weights must sum to 1.0 when override scheme is used"
            )
        if any(float(v) < 0 for v in overrides.values()):
            raise ConfigurationError("Override weights must be non-negative")
    else:
        if overrides and any(float(v) for v in overrides.values()):
            raise ConfigurationError(
                "ASSET_WEIGHTS_OVERRIDE is only respected when WEIGHTING_SCHEME=='override'"
            )

    shrink = float(cfg.get("SHRINK_TO_EQUAL", 0.0))
    if shrink < 0 or shrink > 1:
        raise ConfigurationError("SHRINK_TO_EQUAL must be between 0 and 1")

    max_cap = float(cfg.get("MAX_WEIGHT_CAP", 1.0))
    min_floor = float(cfg.get("MIN_WEIGHT_FLOOR", 0.0))
    if not (0 < max_cap <= 1):
        raise ConfigurationError("MAX_WEIGHT_CAP must be in the (0, 1] interval")
    if not (0 <= min_floor < max_cap):
        raise ConfigurationError(
            "MIN_WEIGHT_FLOOR must satisfy 0 <= floor < MAX_WEIGHT_CAP"
        )

    if cfg.get("USE_RECENCY_WEIGHTING") and cfg.get("FOLD_DECAY_RATE", 0.0) <= 0:
        raise ConfigurationError(
            "FOLD_DECAY_RATE must be > 0 when recency weighting is enabled"
        )

    param_sensitivity = float(cfg.get("PARAM_SENSITIVITY_THRESHOLD", 0.0))
    if param_sensitivity < 0 or param_sensitivity > 1:
        raise ConfigurationError("PARAM_SENSITIVITY_THRESHOLD must be between 0 and 1")

    weight_sensitivity = float(cfg.get("WEIGHT_SENSITIVITY_THRESHOLD", 0.0))
    if weight_sensitivity < 0 or weight_sensitivity > 1:
        raise ConfigurationError("WEIGHT_SENSITIVITY_THRESHOLD must be between 0 and 1")

    ratio_threshold = cfg.get("WEIGHT_SENSITIVITY_RATIO_THRESHOLD")
    if ratio_threshold is not None:
        ratio_val = float(ratio_threshold)
        if ratio_val < 0:
            raise ConfigurationError(
                "WEIGHT_SENSITIVITY_RATIO_THRESHOLD must be >= 0 when provided"
            )

    decimals_cfg = cfg.get("PARAM_VALUE_DECIMALS", {"default": 3})
    if not isinstance(decimals_cfg, dict):
        raise ConfigurationError(
            "PARAM_VALUE_DECIMALS must be a mapping of parameter names to decimal precision"
        )
    for key, value in decimals_cfg.items():
        try:
            decimals = int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"PARAM_VALUE_DECIMALS[{key!r}] must be an integer"
            ) from exc
        if decimals < 0:
            raise ConfigurationError(f"PARAM_VALUE_DECIMALS[{key!r}] must be >= 0")


def _env_flag(name: str, default: bool) -> bool:
    """Return a boolean from environment variables with a sensible default."""

    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


ENV_NAME = os.environ.get("ENV", "")
IS_PROD = ENV_NAME.strip().lower() in {"prod", "production"}

# Global random seed for deterministic runs. Can be overridden via the
# GA_SEED environment variable which acts like a CLI flag.
SEED = int(os.environ.get("GA_SEED", 42))

# Centralised trading fee (percentage). All modules should reference this
# constant rather than hard-coding fee rates.
FEES = 0.001

# When True, preflight computes all indicators to surface latent errors.
PREFLIGHT_ALL_INDICATORS = False

# Behaviour when metric aliases are missing in vectorbt stats.
METRICS_PREFLIGHT = {
    "mode": "warn",  # "warn" | "fail"
    "missing_threshold": 0,
}

# --- DATA SOURCE AND API CONFIGURATION ---
# Select your data source: 'yfinance' or 'binance'
DATA_SOURCE = "binance"

# Binance connection settings are loaded from environment variables so the
# repository never contains sensitive information.  Provide empty-string
# placeholders if the variables are not set.
BINANCE_TLD = os.environ.get("BINANCE_TLD", "us")
API_KEYS = {
    "binance": {
        "api_key": os.environ.get("BINANCE_API_KEY", ""),
        "api_secret": os.environ.get("BINANCE_API_SECRET", ""),
    }
}

# --- 1. CRYPTOCURRENCY PAIR SELECTION ---
CRYPTO_UNIVERSE = {
    # Tier 1 / Major Pairs
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    # Major Altcoins
    "Solana": "SOL-USD",
    "XRP": "XRP-USD",
    "Cardano": "ADA-USD",
    "Avalanche": "AVAX-USD",
    "Dogecoin": "DOGE-USD",
    "Chainlink": "LINK-USD",
    "Polkadot": "DOT-USD",
    "Polygon": "MATIC-USD",
    "Litecoin": "LTC-USD",
    "Bitcoin_Cash": "BCH-USD",
    "Shiba_Inu": "SHIB-USD",
    "Toncoin": "TON-USD",
    # Other Prominent L1s / L2s / DeFi
    "Uniswap": "UNI-USD",
    "TRON": "TRX-USD",
    "Stellar": "XLM-USD",
    "Near_Protocol": "NEAR-USD",
    "Internet_Computer": "ICP-USD",
    "Ethereum_Classic": "ETC-USD",
    "VeChain": "VET-USD",
    "Filecoin": "FIL-USD",
    "Optimism": "OP-USD",
    "The_Graph": "GRT-USD",
}

# Default asset group used when multi-asset optimisation is enabled.  Each
# entry is a tuple of (friendly name, ticker).  By default we keep the group
# very small to avoid additional API calls during normal single-asset runs.
# Minimum fraction of bars required for an asset to be retained when aligning
# group data across multiple tickers. Assets with coverage below this threshold
# are dropped. Used consistently across training, tuning, analysis and
# walk-forward phases.
COVERAGE_THRESHOLD = 0.8

ASSET_GROUP = [
    ("Bitcoin", CRYPTO_UNIVERSE["Bitcoin"]),
    ("Ethereum", CRYPTO_UNIVERSE["Ethereum"]),
    ("Solana", CRYPTO_UNIVERSE["Solana"]),
    ("XRP", CRYPTO_UNIVERSE["XRP"]),
    ("Cardano", CRYPTO_UNIVERSE["Cardano"]),
    ("Avalanche", CRYPTO_UNIVERSE["Avalanche"]),
    ("Dogecoin", CRYPTO_UNIVERSE["Dogecoin"]),
    ("Chainlink", CRYPTO_UNIVERSE["Chainlink"]),
    ("Uniswap", CRYPTO_UNIVERSE["Uniswap"]),
    ("TRON", CRYPTO_UNIVERSE["TRON"]),
]

# --- 2. DYNAMIC DATE & TIMEFRAME SETTINGS ---

SELECTED_ASSET_NAME = "Bitcoin"

# Set your desired timeframe here. This now controls everything.
# Allow overrides by checking for an existing global.
TIMEFRAME = globals().get("TIMEFRAME", "4h")
TICKER = CRYPTO_UNIVERSE.get(SELECTED_ASSET_NAME, "BTC-USD")

# --- 2. DYNAMIC DATE & RISK CALCULATION ---
MAX_HOLD_DAYS = 14
VALIDATION_MONTHS = 3
DEFAULT_MAX_PERIOD = 200
ENABLE_WALK_FORWARD_VALIDATION = True
RSI_PERIOD_BOUNDS = (7, 21)


_UNSET = object()
_INITIALIZED = False

today: datetime | None = None
VALIDATION_BARS: int | None = None
MAX_HOLD_PERIOD: int | None = None
TRAINING_PERIOD: dict[str, str] | object = _UNSET
VALIDATION_PERIOD: dict[str, str] | object = _UNSET
WALK_FORWARD_SETTINGS: dict[str, Any] | object = _UNSET


def _apply_rsi_bounds() -> None:
    low, high = RSI_PERIOD_BOUNDS
    for _cond in STRATEGY_RULES.get("entry_rules", {}).get("conditions", []):
        if _cond.get("rule_name") == "RSI_Momentum_Filter":
            _cond.setdefault("params", {}).setdefault("period", {})
            _cond["params"]["period"]["low"] = low
            _cond["params"]["period"]["high"] = high
            break


def _clamp_long_term_trend(high: int) -> None:
    for _cond in STRATEGY_RULES.get("entry_rules", {}).get("conditions", []):
        if _cond.get("rule_name") == "Long_Term_Trend_Filter":
            params = _cond.setdefault("params", {})
            period_cfg = params.setdefault("period", {})
            if isinstance(period_cfg, dict):
                period_cfg["high"] = high
            break


def _clamp_vote_threshold() -> None:
    _active = len(
        [
            c
            for c in STRATEGY_RULES["entry_rules"].get("conditions", [])
            if c.get("is_active", True)
        ]
    )
    _vt = STRATEGY_RULES["entry_rules"].get("vote_threshold")
    if isinstance(_vt, dict) and "high" in _vt:
        _vt["high"] = max(1, min(_vt["high"], _active))


def initialize_config(force: bool = False) -> None:
    """Populate derived configuration values lazily."""

    global _INITIALIZED, today, VALIDATION_BARS, MAX_HOLD_PERIOD
    global TRAINING_PERIOD, VALIDATION_PERIOD, WALK_FORWARD_SETTINGS, TICKER

    if _INITIALIZED and not force:
        return

    now = datetime.now()
    today = now

    tf = str(TIMEFRAME)
    tf_lower = tf.lower()
    is_hour = tf_lower.endswith("h")
    is_minute = tf_lower.endswith("m") and not tf_lower.endswith("mo")

    validation_bars = 91.0
    if is_hour or is_minute:
        if is_hour:
            validation_bars = 91.0 * 24
        else:
            try:
                minute_value = int(tf_lower[:-1])
            except ValueError:
                minute_value = 1
            validation_bars = 91.0 * (24 * (60 / max(minute_value, 1)))

    if force or VALIDATION_BARS is None:
        VALIDATION_BARS = int(validation_bars)

    max_lookback_period = max(20, min(DEFAULT_MAX_PERIOD, int(validation_bars - 2)))
    _clamp_long_term_trend(max_lookback_period)
    _apply_rsi_bounds()

    if is_hour or is_minute:
        if is_hour:
            minutes_per_bar = 60 * int(tf_lower[:-1])
        else:
            minutes_per_bar = int(tf_lower[:-1])
        minutes_per_bar = max(minutes_per_bar, 1)
        bars_per_day = int(24 * 60 / minutes_per_bar)
        max_hold = MAX_HOLD_DAYS * bars_per_day
    else:
        max_hold = MAX_HOLD_DAYS

    if force or MAX_HOLD_PERIOD is None:
        MAX_HOLD_PERIOD = max_hold

    training_years_daily, training_months_intraday = 3, 20
    training_end_date = now - relativedelta(months=VALIDATION_MONTHS)
    if tf_lower in {"1d", "1wk", "1mo"}:
        training_start_date = training_end_date - relativedelta(
            years=training_years_daily
        )
    else:
        training_start_date = training_end_date - relativedelta(
            months=training_months_intraday
        )

    ticker = TICKER
    if DATA_SOURCE == "binance":
        ticker = ticker.replace("-", "")
        if ticker.endswith("USD") and not ticker.endswith("USDT"):
            ticker = ticker[:-3] + "USDT"
    TICKER = ticker

    training_period = {
        "start": training_start_date.strftime("%Y-%m-%d"),
        "end": training_end_date.strftime("%Y-%m-%d"),
    }
    validation_period = {
        "start": training_end_date.strftime("%Y-%m-%d"),
        "end": now.strftime("%Y-%m-%d"),
    }

    if force or TRAINING_PERIOD is _UNSET:
        TRAINING_PERIOD = training_period
    if force or VALIDATION_PERIOD is _UNSET:
        VALIDATION_PERIOD = validation_period

    wf_start_date = (now - relativedelta(years=3)).strftime("%Y-%m-%d")
    wf_defaults = {
        "enabled": ENABLE_WALK_FORWARD_VALIDATION,
        "total_data_range": {
            "start": wf_start_date,
            "end": validation_period["end"],
        },
        "training_period_length": 12,
        "validation_period_length": 3,
    }

    if (
        force
        or WALK_FORWARD_SETTINGS is _UNSET
        or not isinstance(WALK_FORWARD_SETTINGS, dict)
    ):
        WALK_FORWARD_SETTINGS = wf_defaults
    else:
        wf_cfg = WALK_FORWARD_SETTINGS
        wf_cfg.setdefault("enabled", wf_defaults["enabled"])
        total_range = wf_cfg.setdefault("total_data_range", {})
        total_range.setdefault("start", wf_defaults["total_data_range"]["start"])
        total_range.setdefault("end", wf_defaults["total_data_range"]["end"])
        wf_cfg.setdefault(
            "training_period_length", wf_defaults["training_period_length"]
        )
        wf_cfg.setdefault(
            "validation_period_length", wf_defaults["validation_period_length"]
        )

    _clamp_vote_threshold()
    _validate_combination_logic(STRATEGY_RULES)

    _INITIALIZED = True


def to_pandas_freq(tf: str) -> str:
    """Convert common timeframe strings to pandas frequency aliases.

    Monthly ("1mo") and weekly ("1wk") inputs map to pandas' "M" and "W"
    offsets which are anchored to month-end and week-end respectively.
    """
    tf = tf.strip().lower()
    if tf.endswith("mo"):
        return tf[:-2] + "M"
    if tf.endswith("wk"):
        return tf[:-2] + "W"
    if tf.endswith("d"):
        return tf[:-1] + "D"
    if tf.endswith("h"):
        # Pandas deprecated the uppercase 'H' alias; use lowercase
        return tf[:-1] + "h"
    if tf.endswith("m"):
        # Pandas deprecated the 'T' alias for minutes; use 'min' instead
        return tf[:-1] + "min"
    return tf


# --- 4. GENETIC ALGORITHM PARAMETERS ---
# Default to discovery-run settings but allow quick-test overrides via env.
if os.getenv("GA_QUICK_TEST", "").lower() in {"1", "true", "yes"}:
    GA_POPULATION_SIZE = 50
    GA_NUM_GENERATIONS = 25
    GA_PARENTS_MATING = 20
    GA_MUTATION_NUM_GENES = 1
else:
    GA_POPULATION_SIZE = 200
    GA_NUM_GENERATIONS = 100
    GA_PARENTS_MATING = 50
    GA_MUTATION_NUM_GENES = 3  # Mutate more genes with a more complex strategy

# --- AUTO-TUNER SETTINGS ---
AUTO_TUNE_ENABLED = True
GENERATIONS_PER_TUNE = 5
HYPERPARAMETER_SEARCH_SPACE = [
    {"sol_per_pop": 50, "num_parents_mating": 20, "mutation_num_genes": 1},
    {"sol_per_pop": 100, "num_parents_mating": 30, "mutation_num_genes": 2},
    {"sol_per_pop": 150, "num_parents_mating": 40, "mutation_num_genes": 3},
    {"sol_per_pop": 200, "num_parents_mating": 50, "mutation_num_genes": 4},
    {"sol_per_pop": 250, "num_parents_mating": 60, "mutation_num_genes": 5},
]

# --- 5. COMPOSITE FITNESS FUNCTION WEIGHTS ---
FITNESS_WEIGHTS = {
    "sortino_ratio": 0.5,
    "profit_factor": 0.3,
    "max_drawdown": 0.2,
    "min_trades": 0,
}

# --- 5a. MULTI-ASSET EVALUATION SETTINGS ---
# These options control the behaviour of the multi-asset fitness evaluator.  By
# default the framework behaves exactly as before (single asset) until
# `MULTI_ASSET['enabled']` is set to True.
MULTI_ASSET = {
    # Master switch
    "enabled": True,
    # Optional per-ticker weights; if None, all assets are weighted equally
    "asset_weights": None,
    # Penalty multiplier for dispersion across assets
    "lambda_dispersion": 0.25,
    # Optional coarse tuning grid for lambda. If provided the tuner can try
    # multiple values and pick the best one.
    "lambda_grid": [0.2, 0.3, 0.4],
    # Number of top lambda candidates to re-score after the initial sweep
    "lambda_top_k": 2,
    # Seeds used when re-scoring lambda candidates without mutation
    "lambda_rescore_seeds": [SEED, SEED + 1],
    # Number of GA generations to run when sweeping or rescoring lambda
    "lambda_grid_generations": 10,
    # Which per-asset metric to aggregate; typically "composite"
    "metric": "composite",  # composite | sortino | profit_factor | return
    # Profit factor cap to avoid outliers
    "winsorize_pf_cap": 5.0,
    # Substitute value for NaN metrics
    "nan_fallback": 0.0,
    # Group trade floor configuration
    "min_total_trades": 0,
    "trade_floor_policy": "soft_penalty",  # hard_floor | soft_penalty
    "soft_penalty_strength": 0.75,
    "soft_penalty_mode": "multiplicative",  # multiplicative | additive
    # How to handle assets with zero trades
    "zero_trade_policy": "ignore",  # penalize | ignore
    "zero_trade_penalty": -1.0,
    # Penalty applied when ignoring assets
    "coverage_penalty": 0.25,
    # Minimal trades to consider an asset as traded
    "per_asset_min_trades": 5,
    # Minimal number of assets that must be included
    "min_included_assets": 3,
    # Annualisation base for trade floor scaling
    "trading_days_per_year": 252,
    # Optional scaling of the group trade floor based on fold length (years)
    "min_total_trades_per_year": 50,
    # Parallel evaluation of per-asset statistics
    "parallel": {
        "enabled": False,  # when True uses concurrent.futures
        "backend": "thread",  # "thread" or "process"
        "max_workers": None,
    },
    # Verbose logging of per-asset evaluation errors (can be noisy)
    "verbose_asset_errors": False,
    # Fitness score returned when the hard floor triggers or an error occurs
    "poor_score": -999.0,
}

# Validate core multi-asset parameters on import so that misconfigured
# values fail fast regardless of whether the evaluator runs.
assert MULTI_ASSET["lambda_dispersion"] >= 0, "lambda_dispersion must be >= 0"
assert MULTI_ASSET["winsorize_pf_cap"] >= 1, "winsorize_pf_cap must be >= 1"
assert MULTI_ASSET["soft_penalty_strength"] >= 0, "soft_penalty_strength must be >= 0"
assert MULTI_ASSET["min_total_trades"] >= 0, "min_total_trades must be >= 0"

# --- Optional stability regularizer for parameter CoV ---
ENABLE_STABILITY_REG = False
STABILITY_ALPHA = 0.1
STABILITY_GENES = ["rsi_period"]

# Basic charting options for the multi-asset analysis overview.
CHARTS = {
    "max_assets_in_overview": 20,
    "save_pngs": True,
    "show_distribution": True,
    "save_csv": True,
}

# Settings for the Strategy Recommendation Engine.
RECOMMENDATION = {
    "MEDIAN_TARGET": 1.5,
    "TAIL_PENALTY_REF": 2.0,
    "DOWNSIDE_REF": 1.0,
    "WEIGHTS": {
        "median": 0.35,
        "consistency": 0.35,
        "tail": 0.15,
        "downside": 0.15,
    },
    "CATEGORY_CUTOFFS": {"high": 80, "medium": 50},
    "MIN_TRADES_FOR_SAMPLE": 3,
    "MIN_SAMPLES_FOR_ASSET": 3,
    "PARAM_COV_UNSTABLE": 0.5,
    "PARAM_COV_WATCHLIST": 0.35,
    "PARAM_COV_DDOF": 0,
    "USE_RETURN_AS_FITNESS": False,
    "LOG_UNKNOWN_COLUMNS_ON_SUCCESS": _env_flag("SRE_LOG_UNKNOWN_COLS", not IS_PROD),
    "ASSET_CLASS_THRESHOLDS": {
        "star": {"performance": 1.0, "consistency": 70},
        "stalwart": {
            "performance_low": 0.0,
            "performance_high": 1.0,
            "consistency": 60,
        },
        "gamble": {"performance": 1.0, "consistency": 50},
        "drag": {"performance": 0.0, "consistency": 50},
    },
}

FINAL_STRATEGY = {
    # --- Fold weighting ---
    "USE_RECENCY_WEIGHTING": True,
    "FOLD_DECAY_RATE": 0.139,
    # --- Safety gates ---
    "MIN_CONFIDENCE_FOR_FINAL": 60,
    # --- Asset selection & weighting ---
    "INCLUDE_CLASSES": ["Stars", "Stalwarts"],
    "MIN_ASSET_CONSISTENCY": 60.0,
    "WEIGHTING_SCHEME": "risk_adjusted",
    "ASSET_WEIGHTS_OVERRIDE": {},
    "MAX_WEIGHT_CAP": 0.35,
    "MIN_WEIGHT_FLOOR": 0.02,
    "SHRINK_TO_EQUAL": 0.25,
    # --- Parameter stability (robust) ---
    "PARAM_RCV_UNSTABLE": 0.50,
    "PARAM_RCV_WATCHLIST": 0.35,
    "PARAM_RCV_DDOF": 0,
    "MULTIMODAL_MIN_SEPARATION": 0.75,
    "MULTIMODAL_MIN_CLUSTER_WEIGHT": 0.2,
    # --- Sensitivity thresholds ---
    "PARAM_SENSITIVITY_THRESHOLD": 0.15,
    "WEIGHT_SENSITIVITY_THRESHOLD": 0.05,
    "WEIGHT_SENSITIVITY_RATIO_THRESHOLD": 0.25,
    # --- Reporting knobs ---
    "SHOW_PARAM_DISTS": True,
    "SHOW_RECENCY_HALFLIFE": True,
    # --- Parameter rounding ---
    "PARAM_VALUE_DECIMALS": {"default": 3},
}

validate_final_strategy_config(FINAL_STRATEGY)

# Global NaN handling policy for signal combination. Individual rule sets may
# override ``nan_policy`` and ``ffill_lookback``.
NAN_POLICY = "FALSE"  # FALSE | PROPAGATE | FORWARD_FILL
NAN_FFILL_LOOKBACK = 0  # 0 disables the lookback cap

# Simple guardrails for the indicator cache used by ``strategy_engine``.
CACHE_GUARDRAILS = {
    "MAX_CACHE_KEYS": 1000,
    "MAX_CACHE_ROWS": 1_000_000,
    "clear_cache_between_assets": False,
}

# Settings controlling how walk-forward champions are kept or discarded
CHAMPION_SELECTION_SETTINGS = {
    # Minimum validation fitness required for a champion to survive
    "survival_threshold": 0.5,
    # Threshold at which a champion is considered elite and cloned
    "cloning_threshold": 1.5,
    # Number of clones to make for elite champions
    "num_clones": 5,
    # Probability of mutating each gene on a clone
    "clone_mutation_rate": 0.20,
}


def _validate_combination_logic(rules: dict) -> None:
    entry = rules.get("entry_rules", {})
    conditions = [c for c in entry.get("conditions", []) if c.get("is_active", True)]
    n = len(conditions) or 1
    logic = entry.get("combination_logic", "AND")
    if isinstance(logic, dict):
        if "options" in logic:
            # Gene-driven dict; allow GA to explore provided options
            return
        raise ConfigurationError("combination_logic must be AND, OR, or VOTE")
    logic_u = str(logic).upper()
    if logic_u not in {"AND", "OR", "VOTE"}:
        raise ConfigurationError(
            f"combination_logic must be AND, OR, or VOTE (got {logic})"
        )
    entry["combination_logic"] = logic_u
    if logic_u == "VOTE":
        vt = entry.get("vote_threshold")
        if isinstance(vt, dict):
            return
        if vt is None:
            entry["vote_threshold"] = math.ceil(n / 2)
        elif vt < 1 or vt > n:
            raise ConfigurationError(f"vote_threshold {vt} must be between 1 and {n}")
