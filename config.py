# config.py

"""
Configuration File for the GA Trading Framework
(This version includes automated rolling date ranges that adapt to the selected timeframe)
"""

import math
import os

# Import necessary libraries for date calculation
from datetime import datetime

from dateutil.relativedelta import relativedelta

# Global random seed for deterministic runs. Can be overridden via the
# GA_SEED environment variable which acts like a CLI flag.
SEED = int(os.environ.get("GA_SEED", 42))

# Centralised trading fee (percentage). All modules should reference this
# constant rather than hard-coding fee rates.
FEES = 0.001

# --- DATA SOURCE AND API CONFIGURATION ---
# Select your data source: 'yfinance' or 'binance'
DATA_SOURCE = "binance"

# Binance API credentials are now loaded from environment variables so the
# repository never contains sensitive information.  Provide empty-string
# placeholders if the variables are not set.
API_KEYS = {
    "binance": {
        "tld": os.environ.get("BINANCE_TLD", "us"),
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
    ("Polkadot", CRYPTO_UNIVERSE["Polkadot"]),
    ("Polygon", CRYPTO_UNIVERSE["Polygon"]),
    ("Litecoin", CRYPTO_UNIVERSE["Litecoin"]),
    ("Uniswap", CRYPTO_UNIVERSE["Uniswap"]),
    ("TRON", CRYPTO_UNIVERSE["TRON"]),
    ("Stellar", CRYPTO_UNIVERSE["Stellar"]),
    ("Near_Protocol", CRYPTO_UNIVERSE["Near_Protocol"]),
]

# --- 2. DYNAMIC DATE & TIMEFRAME SETTINGS ---

SELECTED_ASSET_NAME = "Dogecoin"

# Set your desired timeframe here. This now controls everything.
# Allow overrides by checking for an existing global.
TIMEFRAME = globals().get("TIMEFRAME", "1h")
TICKER = CRYPTO_UNIVERSE.get(SELECTED_ASSET_NAME, "BTC-USD")

# --- 2. DYNAMIC DATE & RISK CALCULATION ---
MAX_HOLD_DAYS = 14
VALIDATION_MONTHS = 3
DEFAULT_MAX_PERIOD = 200
ENABLE_WALK_FORWARD_VALIDATION = True
today = datetime.now()
if "h" in TIMEFRAME.lower() or "m" in TIMEFRAME.lower():
    VALIDATION_BARS = 91 * (
        24 if "h" in TIMEFRAME.lower() else 24 * (60 / int(TIMEFRAME.replace("m", "")))
    )
else:
    VALIDATION_BARS = 91
max_lookback_period = max(20, min(DEFAULT_MAX_PERIOD, int(VALIDATION_BARS - 2)))
if "m" in TIMEFRAME.lower() or "h" in TIMEFRAME.lower():
    minutes = 60 if TIMEFRAME.endswith("h") else int(TIMEFRAME[:-1])
    bars_per_day = int(24 * 60 / minutes)
    MAX_HOLD_PERIOD = MAX_HOLD_DAYS * bars_per_day
else:
    MAX_HOLD_PERIOD = MAX_HOLD_DAYS
training_years_daily, training_months_intraday = 3, 20
training_end_date = today - relativedelta(months=VALIDATION_MONTHS)
if TIMEFRAME in ["1d", "1wk", "1mo"]:
    training_start_date = training_end_date - relativedelta(years=training_years_daily)
else:
    training_start_date = training_end_date - relativedelta(
        months=training_months_intraday
    )

# --- 3. FINAL CONFIGURATION OUTPUTS ---
if DATA_SOURCE == "binance":
    TICKER = TICKER.replace("-", "")
    # Binance typically provides deep history for USDT pairs rather than USD.
    # Convert "BTCUSD" -> "BTCUSDT" to ensure sufficient historical data.
    if TICKER.endswith("USD") and not TICKER.endswith("USDT"):
        TICKER = TICKER[:-3] + "USDT"
TRAINING_PERIOD = {
    "start": training_start_date.strftime("%Y-%m-%d"),
    "end": training_end_date.strftime("%Y-%m-%d"),
}
VALIDATION_PERIOD = {
    "start": training_end_date.strftime("%Y-%m-%d"),
    "end": today.strftime("%Y-%m-%d"),
}


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
        return tf[:-1] + "h"
    if tf.endswith("m"):
        # Pandas deprecated the 'T' alias for minutes; use 'min' instead
        return tf[:-1] + "min"
    return tf


# Walk-forward validation will leverage a longer history than the main
# optimisation phase.  Start three years back from today regardless of the
# optimisation window above.
walk_forward_start_date = (today - relativedelta(years=3)).strftime("%Y-%m-%d")

WALK_FORWARD_SETTINGS = {
    "enabled": ENABLE_WALK_FORWARD_VALIDATION,
    "total_data_range": {
        # Use the extended three year lookback for walk-forward windows
        "start": walk_forward_start_date,
        "end": VALIDATION_PERIOD["end"],
    },
    # Each window trains on one year of data and tests on the following three
    # months.
    "training_period_length": 12,  # months
    "validation_period_length": 3,
}

# --- 4. GENETIC ALGORITHM PARAMETERS ---
# Use these settings for quick tests
GA_POPULATION_SIZE = 50
GA_NUM_GENERATIONS = 30
GA_PARENTS_MATING = 20
GA_MUTATION_NUM_GENES = 1

# For serious, overnight "Discovery" runs, comment out the block above
# and uncomment the block below.
# GA_POPULATION_SIZE = 200
# GA_NUM_GENERATIONS = 100
# GA_PARENTS_MATING = 50
# GA_MUTATION_NUM_GENES = 3 # Mutate more genes with a more complex strategy

# --- AUTO-TUNER SETTINGS ---
AUTO_TUNE_ENABLED = True
GENERATIONS_PER_TUNE = 5
HYPERPARAMETER_SEARCH_SPACE = [
    {"sol_per_pop": 50, "num_parents_mating": 20, "mutation_num_genes": 1},
    {"sol_per_pop": 100, "num_parents_mating": 30, "mutation_num_genes": 2},
    {"sol_per_pop": 150, "num_parents_mating": 40, "mutation_num_genes": 3},
    {"sol_per_pop": 200, "num_parents_mating": 50, "mutation_num_genes": 4},
    {"sol_per_pop": 250, "num_parents_mating": 60, "mutation_num_genes": 4},
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
    "lambda_dispersion": 0.20,
    # Optional coarse tuning grid for lambda. If provided the tuner can try
    # multiple values and pick the best one.
    "lambda_grid": [0.1, 0.2, 0.3, 0.4, 0.5],
    # Number of top candidates considered when applying the elbow heuristic
    "lambda_shortlist_size": 3,
    # Percentile threshold for the optional dispersion screen
    "lambda_sigma_pctl": 0.75,
    # Seeds used for the lambda sweep to smooth variance
    "lambda_seeds": [SEED, SEED + 1, SEED + 2],
    # Optional minimum average coverage fraction for lambda candidates
    "lambda_coverage_min": None,
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
    "per_asset_min_trades": 10,
    # Minimal number of assets that must be included
    "min_included_assets": 4,
    # Annualisation base for trade floor scaling
    "trading_days_per_year": 252,
    # Optional scaling of the group trade floor based on fold length (years)
    "min_total_trades_per_year": 36,
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

# Basic charting options for the multi-asset analysis overview.
CHARTS = {
    "max_assets_in_overview": 20,
    "save_pngs": True,
    "show_distribution": True,
    "save_csv": True,
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

# --- 6. STRATEGY RULES DEFINITION ---
# Here you can define a "master list" of all potential conditions.
# Use the `is_active` flag to control which ones are used in a given run.
STRATEGY_RULES = {
    "entry_rules": {
        # Optional keys:
        #   combination_logic (str): "AND" | "OR" | "VOTE" (default "AND")
        #   vote_threshold (int | None): min signals for "VOTE"; ``None`` uses
        #       ``ceil(N/2)`` and values outside ``1..N`` raise ``ValueError``
        #   treat_nan_as_false (bool): replace NaNs before combining (default True)
        "combination_logic": "VOTE",
        "vote_threshold": {
            "gene": "vote_threshold",
            "low": 2,
            "high": 3,
            "step": 1,
        },
        "treat_nan_as_false": True,
        "conditions": [
            {
                "is_active": True,  # This rule is ON
                "rule_name": "Long_Term_Trend_Filter",
                "indicator": "ema",
                "params": {
                    "period": {
                        "gene": "ema_period",
                        "low": 30,
                        "high": max_lookback_period,
                        "step": 5,
                    }
                },
                "condition": {"type": "price_is_above_indicator"},
            },
            {
                "is_active": True,  # This rule is ON
                "rule_name": "RSI_Momentum_Filter",
                "indicator": "rsi",
                "params": {
                    "period": {"gene": "rsi_period", "low": 5, "high": 35, "step": 1}
                },
                "condition": {
                    "type": "indicator_is_above_value",
                    "value": {
                        "gene": "rsi_threshold",
                        "low": 45,
                        "high": 70,
                        "step": 1,
                    },
                },
            },
            {
                "is_active": True,
                "rule_name": "MACD_Momentum_Cross",
                "indicator": "macd",
                "params": {
                    "fast": {"gene": "macd_fast", "low": 4, "high": 20, "step": 1},
                    "slow": {"gene": "macd_slow", "low": 15, "high": 35, "step": 1},
                    "signal": {"gene": "macd_signal", "low": 4, "high": 16, "step": 1},
                },
                "condition": {"type": "indicator_is_above_value", "value": 0},
            },
            {
                "is_active": False,
                "rule_name": "Bollinger_Band_Breakout",
                "indicator": "bbands",
                "params": {
                    "period": {
                        "gene": "bband_period",
                        "low": 10,
                        "high": 35,
                        "step": 1,
                    },
                    "std_dev": {
                        "gene": "bband_std",
                        "low": 0.5,
                        "high": 5,
                        "step": 0.25,
                    },
                },
                "condition": {
                    "type": "price_crosses_above_indicator",
                    "band": "upper",
                },
            },
        ],
    },
    "exit_rules": {
        "stop_loss": {
            "is_active": True,  # Turn off regular stop to use trailing stop
            "type": "percentage",
            "params": {  # Correctly nested
                "value": {
                    "gene": "stop_loss_pct",
                    "low": 0.01,
                    "high": 0.10,
                    "step": 0.005,
                }
            },
        },
        "trailing_stop": {
            "is_active": True,
            "type": "percentage",
            "params": {  # Correctly nested
                "value": {
                    "gene": "tsl_pct",
                    "low": 0.02,
                    "high": 0.06,
                    "step": 0.005,
                }
            },
        },
        "take_profit": {
            "is_active": True,
            "type": "percentage",
            "params": {  # Correctly nested
                "value": {
                    "gene": "take_profit_pct",
                    "low": 0.02,
                    "high": 0.20,
                    "step": 0.01,
                }
            },
        },
    },
}

# Clamp vote_threshold gene so the upper bound never exceeds the number of active
# conditions. This prevents GA runs from sampling impossible thresholds if rules
# are toggled off later.
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


class ConfigurationError(ValueError):
    """Raised when configuration options are invalid."""


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


_validate_combination_logic(STRATEGY_RULES)
