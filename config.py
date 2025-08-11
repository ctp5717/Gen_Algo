# config.py

"""
Configuration File for the GA Trading Framework
(This version includes automated rolling date ranges that adapt to the selected timeframe)
"""

# Import necessary libraries for date calculation
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os

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
        "api_secret": os.environ.get("BINANCE_API_SECRET", "")
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
    "Dai": "DAI-USD",
    "Stellar": "XLM-USD",
    "Near_Protocol": "NEAR-USD",
    "Internet_Computer": "ICP-USD",
    "Ethereum_Classic": "ETC-USD",
    "VeChain": "VET-USD",
    "Filecoin": "FIL-USD",
    "Optimism": "OP-USD",
    "The_Graph": "GRT-USD"
}

# --- 2. DYNAMIC DATE & TIMEFRAME SETTINGS ---
# ---------------------------------------------------------------------------
# Portfolio settings
# ---------------------------------------------------------------------------
# The framework now supports optimising a single strategy across a basket of
# assets.  When ``PORTFOLIO_OPTIMIZATION_ENABLED`` is ``True`` the data loader
# and backtesting logic expect a list of tickers instead of a single string.
# ``TUNING_ASSET`` defines which asset is used during the fast hyperparameter
# tuning phase before the main portfolio optimisation.

PORTFOLIO_OPTIMIZATION_ENABLED = True

# Define a basket of tickers when portfolio optimisation is enabled.  Leaving
# this list empty will result in an error if ``PORTFOLIO_OPTIMIZATION_ENABLED``
# is set to True.
ASSET_BASKET: list[str] = ['ETH-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD', 'LINK-USD']  # e.g. ['BTC-USD', 'ETH-USD']

# Optional custom weights corresponding to ASSET_BASKET.  If ``None`` each
# asset is equally weighted during portfolio backtests.  The weights will be
# normalised automatically.
PORTFOLIO_WEIGHTS: list[float] | None = None  # e.g. [0.6, 0.4]

# Asset to use for the express tuning phase.  When the basket is provided this
# should typically be one of its members.
TUNING_ASSET = CRYPTO_UNIVERSE["Bitcoin"]

if PORTFOLIO_OPTIMIZATION_ENABLED and not ASSET_BASKET:
    raise ValueError("ASSET_BASKET cannot be empty when portfolio optimisation is enabled")

# ---------------------------------------------------------------------------
# Legacy single asset settings
# ---------------------------------------------------------------------------
SELECTED_ASSET_NAME = "Dogecoin"

# Set your desired timeframe here. This now controls everything.
TIMEFRAME = "15m"
TICKER = CRYPTO_UNIVERSE.get(SELECTED_ASSET_NAME, "BTC-USD")

# --- 2. DYNAMIC DATE & RISK CALCULATION ---
MAX_HOLD_DAYS = 14
VALIDATION_MONTHS = 3
DEFAULT_MAX_PERIOD = 200
ENABLE_WALK_FORWARD_VALIDATION = True
today = datetime.now()
if 'h' in TIMEFRAME.lower() or 'm' in TIMEFRAME.lower():
    VALIDATION_BARS = 91 * (24 if 'h' in TIMEFRAME.lower() else 24 * (60 / int(TIMEFRAME.replace('m',''))))
else: VALIDATION_BARS = 91
max_lookback_period = max(20, min(DEFAULT_MAX_PERIOD, int(VALIDATION_BARS - 2)))
if 'm' in TIMEFRAME.lower() or 'h' in TIMEFRAME.lower():
    minutes = 60 if TIMEFRAME.endswith('h') else int(TIMEFRAME[:-1])
    bars_per_day = int(24 * 60 / minutes)
    MAX_HOLD_PERIOD = MAX_HOLD_DAYS * bars_per_day
else:
    MAX_HOLD_PERIOD = MAX_HOLD_DAYS
training_years_daily, training_months_intraday = 3, 20
training_end_date = today - relativedelta(months=VALIDATION_MONTHS)
if TIMEFRAME in ['1d', '1wk', '1mo']:
    training_start_date = training_end_date - relativedelta(years=training_years_daily)
else:
    training_start_date = training_end_date - relativedelta(months=training_months_intraday)

# --- 3. FINAL CONFIGURATION OUTPUTS ---
TRAINING_PERIOD = {
    "start": training_start_date.strftime("%Y-%m-%d"),
    "end": training_end_date.strftime("%Y-%m-%d"),
}
VALIDATION_PERIOD = {
    "start": training_end_date.strftime("%Y-%m-%d"),
    "end": today.strftime("%Y-%m-%d"),
}

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
GA_NUM_GENERATIONS = 25
GA_PARENTS_MATING = 20
GA_MUTATION_NUM_GENES = 1

# For serious, overnight "Discovery" runs, comment out the block above
# and uncomment the block below.
# GA_POPULATION_SIZE = 200
# GA_NUM_GENERATIONS = 100
# GA_PARENTS_MATING = 50
# GA_MUTATION_NUM_GENES = 3 # Mutate more genes with a more complex strategy

# When the GA repeatedly produces invalid solutions (fitness ``-999``), the
# optimisation can be configured to restart or expand the search space.
# ``GA_STAGNATION_THRESHOLD`` controls how many consecutive generations of
# ``-999`` fitness are tolerated before the policy triggers.  Set
# ``GA_RESTART_POLICY`` to ``"restart"`` to simply randomise the population or
# ``"expand"`` to widen each gene's range before randomising.  The expansion
# applied when using ``"expand"`` is governed by ``GA_GENE_RANGE_EXPANSION``
# and is expressed as a fraction of the current range (default 0.5 = 50%).
GA_STAGNATION_THRESHOLD = 5
GA_RESTART_POLICY = "restart"  # or "expand"
GA_GENE_RANGE_EXPANSION = 0.5

# --- AUTO-TUNER SETTINGS ---
AUTO_TUNE_ENABLED = True
GENERATIONS_PER_TUNE = 10
HYPERPARAMETER_SEARCH_SPACE = [
    {"sol_per_pop": 50, "num_parents_mating": 20, "mutation_num_genes": 1},
    {"sol_per_pop": 100, "num_parents_mating": 30, "mutation_num_genes": 2},
    {"sol_per_pop": 150, "num_parents_mating": 40, "mutation_num_genes": 3},
    {"sol_per_pop": 200, "num_parents_mating": 50, "mutation_num_genes": 4},
]

# --- 5. COMPOSITE FITNESS FUNCTION WEIGHTS ---
FITNESS_WEIGHTS = {
    "sortino_ratio": 0.5, "profit_factor": 0.3, "max_drawdown": 0.2, "min_trades": 10
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
    'entry_rules': {
        'combination_logic': 'AND',
        'conditions': [
            {
                'is_active': True, # This rule is ON
                'rule_name': 'Long_Term_Trend_Filter',
                'indicator': 'ema',
                'params': {
                    'period': {'gene': 'ema_period', 'low': 20, 'high': max_lookback_period, 'step': 5}
                },
                'condition': {'type': 'price_is_above_indicator'}
            },
            {
                'is_active': True, # This rule is ON
                'rule_name': 'RSI_Momentum_Filter',
                'indicator': 'rsi',
                'params': {
                    'period': {'gene': 'rsi_period', 'low': 3, 'high': 35, 'step': 1}
                },
                'condition': {
                    'type': 'indicator_is_above_value',
                    'value': {'gene': 'rsi_threshold', 'low': 30, 'high': 84, 'step': 2}
                }
            },
            {
                'is_active': False,
                'rule_name': 'MACD_Momentum_Cross',
                'indicator': 'macd',
                'params': {
                    'fast': {'gene': 'macd_fast', 'low': 4, 'high': 20, 'step': 1},
                    'slow': {'gene': 'macd_slow', 'low': 15, 'high': 35, 'step': 1},
                    'signal': {'gene': 'macd_signal', 'low': 4, 'high': 16, 'step': 1}
                },
                'condition': {'type': 'indicator_crosses_above_value', 'value': 0}
            },
            {
                'is_active': False,
                'rule_name': 'Bollinger_Band_Breakout',
                'indicator': 'bbands',
                'params': {
                    'period': {'gene': 'bband_period', 'low': 10, 'high': 35, 'step': 1},
                    'std_dev': {'gene': 'bband_std', 'low': 0.5, 'high': 5, 'step': 0.25}
                },
                'condition': {
                    'type': 'price_crosses_above_upper_band',
                    'column': 'BBU_20_2.0' # Specify which band to check against
                }
            }
        ]
    },
    'exit_rules': {
        'stop_loss': {
            'is_active': True, # Turn off regular stop to use trailing stop
            'type': 'percentage',
            'params': { # Correctly nested
                'value': {'gene': 'stop_loss_pct', 'low': 0.01, 'high': 0.10, 'step': 0.005}
            }
        },
        'trailing_stop': {
            'is_active': False, # Turn on trailing stop
            'type': 'percentage',
            'params': { # Correctly nested
                'value': {'gene': 'tsl_pct', 'low': 0.01, 'high': 0.10, 'step': 0.005}
            }
        },
        'take_profit': {
            'is_active': True,
            'type': 'percentage',
            'params': { # Correctly nested
                'value': {'gene': 'take_profit_pct', 'low': 0.02, 'high': 0.20, 'step': 0.01}
            }
        }
    }
}
