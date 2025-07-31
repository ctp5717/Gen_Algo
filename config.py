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
if TIMEFRAME == "1h": MAX_HOLD_PERIOD = MAX_HOLD_DAYS * 24
else: MAX_HOLD_PERIOD = MAX_HOLD_DAYS
training_years_daily, training_months_intraday = 3, 20
training_end_date = today - relativedelta(months=VALIDATION_MONTHS)
if TIMEFRAME in ['1d', '1wk', '1mo']:
    training_start_date = training_end_date - relativedelta(years=training_years_daily)
else:
    training_start_date = training_end_date - relativedelta(months=training_months_intraday)

# --- 3. FINAL CONFIGURATION OUTPUTS ---
if DATA_SOURCE == 'binance':
    TICKER = TICKER.replace('-', '')
    # Binance typically provides deep history for USDT pairs rather than USD.
    # Convert "BTCUSD" -> "BTCUSDT" to ensure sufficient historical data.
    if TICKER.endswith('USD') and not TICKER.endswith('USDT'):
        TICKER = TICKER[:-3] + 'USDT'
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
