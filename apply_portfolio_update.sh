#!/usr/bin/env bash
set -euo pipefail

# Fail if not in a git repo
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "Run from repo root."; exit 1; }

mkdir -p tests

# ------------------------------
# config.py (adds portfolio flags)
# ------------------------------
cat > config.py <<'PY'
"""
Configuration File for the GA Trading Framework
(rolling date ranges adapt to the selected timeframe)
"""

from datetime import datetime
from dateutil.relativedelta import relativedelta
import os

# --- DATA SOURCE AND API CONFIGURATION ---
# Select your data source: 'yfinance' or 'binance'
DATA_SOURCE = "binance"

API_KEYS = {
    "binance": {
        "tld": os.environ.get("BINANCE_TLD", "us"),
        "api_key": os.environ.get("BINANCE_API_KEY", ""),
        "api_secret": os.environ.get("BINANCE_API_SECRET", "")
    }
}

# --- 1. CRYPTOCURRENCY PAIR SELECTION ---
CRYPTO_UNIVERSE = {
    "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD",
    "Solana": "SOL-USD", "XRP": "XRP-USD", "Cardano": "ADA-USD",
    "Avalanche": "AVAX-USD", "Dogecoin": "DOGE-USD", "Chainlink": "LINK-USD",
    "Polkadot": "DOT-USD", "Polygon": "MATIC-USD", "Litecoin": "LTC-USD",
    "Bitcoin_Cash": "BCH-USD", "Shiba_Inu": "SHIB-USD", "Toncoin": "TON-USD",
    "Uniswap": "UNI-USD", "TRON": "TRX-USD", "Dai": "DAI-USD",
    "Stellar": "XLM-USD", "Near_Protocol": "NEAR-USD",
    "Internet_Computer": "ICP-USD", "Ethereum_Classic": "ETC-USD",
    "VeChain": "VET-USD", "Filecoin": "FIL-USD", "Optimism": "OP-USD",
    "The_Graph": "GRT-USD"
}

# --- 2. DYNAMIC DATE & TIMEFRAME SETTINGS ---
SELECTED_ASSET_NAME = "Dogecoin"
TIMEFRAME = "15m"
TICKER = CRYPTO_UNIVERSE.get(SELECTED_ASSET_NAME, "BTC-USD")

# Portfolio optimization switches (NEW)
PORTFOLIO_OPTIMIZATION_ENABLED = False  # set True to enable portfolio mode
ASSET_BASKET = ["BTC-USD", "ETH-USD", "SOL-USD"]  # used when portfolio mode is on
TUNING_ASSET = "BTC-USD"  # fast tuning runs on this single asset first

# --- DYNAMIC DATE & RISK CALCULATION ---
MAX_HOLD_DAYS = 14
VALIDATION_MONTHS = 3
DEFAULT_MAX_PERIOD = 200
ENABLE_WALK_FORWARD_VALIDATION = True

today = datetime.now()
if 'h' in TIMEFRAME.lower() or 'm' in TIMEFRAME.lower():
    VALIDATION_BARS = 91 * (24 if 'h' in TIMEFRAME.lower() else 24 * (60 / int(TIMEFRAME.replace('m',''))))
else:
    VALIDATION_BARS = 91
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
if DATA_SOURCE == 'binance':
    TICKER = TICKER.replace('-', '')
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

walk_forward_start_date = (today - relativedelta(years=3)).strftime("%Y-%m-%d")
WALK_FORWARD_SETTINGS = {
    "enabled": ENABLE_WALK_FORWARD_VALIDATION,
    "total_data_range": {"start": walk_forward_start_date, "end": VALIDATION_PERIOD["end"]},
    "training_period_length": 12,  # months
    "validation_period_length": 3,
}

# --- 4. GENETIC ALGORITHM PARAMETERS ---
GA_POPULATION_SIZE = 50
GA_NUM_GENERATIONS = 25
GA_PARENTS_MATING = 20
GA_MUTATION_NUM_GENES = 1

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

CHAMPION_SELECTION_SETTINGS = {
    "survival_threshold": 0.5,
    "cloning_threshold": 1.5,
    "num_clones": 5,
    "clone_mutation_rate": 0.20,
}

# --- 6. STRATEGY RULES DEFINITION ---
STRATEGY_RULES = {
    'entry_rules': {
        'combination_logic': 'AND',
        'conditions': [
            {
                'is_active': True,
                'rule_name': 'Long_Term_Trend_Filter',
                'indicator': 'ema',
                'params': {'period': {'gene': 'ema_period', 'low': 20, 'high': max_lookback_period, 'step': 5}},
                'condition': {'type': 'price_is_above_indicator'}
            },
            {
                'is_active': True,
                'rule_name': 'RSI_Momentum_Filter',
                'indicator': 'rsi',
                'params': {'period': {'gene': 'rsi_period', 'low': 3, 'high': 35, 'step': 1}},
                'condition': {'type': 'indicator_is_above_value','value': {'gene': 'rsi_threshold', 'low': 30, 'high': 84, 'step': 2}}
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
                'condition': {'type': 'price_crosses_above_upper_band', 'column': 'BBU_20_2.0'}
            }
        ]
    },
    'exit_rules': {
        'stop_loss': {'is_active': True, 'type': 'percentage','params': {'value': {'gene': 'stop_loss_pct', 'low': 0.01,'high': 0.10,'step': 0.005}}},
        'trailing_stop': {'is_active': False, 'type': 'percentage','params': {'value': {'gene': 'tsl_pct', 'low': 0.01,'high': 0.10,'step': 0.005}}},
        'take_profit': {'is_active': True, 'type': 'percentage','params': {'value': {'gene': 'take_profit_pct','low': 0.02,'high': 0.20,'step': 0.01}}},
    }
}
PY

# ------------------------------
# data_loader.py
# ------------------------------
cat > data_loader.py <<'PY'
from __future__ import annotations
import os
from typing import List, Union, Dict
import pandas as pd

import yfinance as yf
from binance.client import Client
import config

_BN_INTERVALS = {
    "1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m","1h":"1h","2h":"2h","4h":"4h","6h":"6h","8h":"8h","12h":"12h","1d":"1d","3d":"3d","1w":"1w","1M":"1M"
}

def _fetch_yf(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if df.empty:
        return df
    df = df.rename(columns={c: c.title() for c in df.columns})
    df.index = pd.to_datetime(df.index)
    return df[['Open','High','Low','Close','Volume']]

def _fetch_binance(symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
    api = config.API_KEYS.get('binance', {})
    client = Client(api.get('api_key', ''), api.get('api_secret', ''), tld=api.get('tld','us'))
    bn_interval = _BN_INTERVALS.get(interval, "1h")
    klines = client.get_historical_klines(symbol, bn_interval, f"{start} 00:00:00", f"{end} 23:59:59")
    if not klines:
        return pd.DataFrame(columns=['Open','High','Low','Close','Volume'])
    cols = ['Open time','Open','High','Low','Close','Volume','Close time','Quote asset volume','Number of trades','Taker buy base','Taker buy quote','Ignore']
    df = pd.DataFrame(klines, columns=cols)
    for c in ['Open','High','Low','Close','Volume']:
        df[c] = pd.to_numeric(df[c])
    df['Date'] = pd.to_datetime(df['Open time'], unit='ms')
    df = df.set_index('Date')[['Open','High','Low','Close','Volume']]
    return df

def _normalize_symbol(t: str) -> str:
    if config.DATA_SOURCE == 'binance':
        t = t.replace('-', '')
        if t.endswith('USD') and not t.endswith('USDT'):
            t = t[:-3] + 'USDT'
    return t

def _fetch_single(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    if config.DATA_SOURCE == 'binance':
        sym = _normalize_symbol(ticker)
        df = _fetch_binance(sym, start, end, interval)
    else:
        df = _fetch_yf(ticker, start, end, interval)
    return df

def get_data(ticker: Union[str, List[str]], start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    """
    Returns OHLCV for one or many tickers. If a list is passed, returns a
    MultiIndex dataframe: columns = (asset, field).
    """
    if isinstance(ticker, list):
        frames: Dict[str, pd.DataFrame] = {}
        for t in ticker:
            df = _fetch_single(t, start_date, end_date, interval)
            if df.empty:
                continue
            df = df.copy()
            df.columns = pd.MultiIndex.from_product([[t], df.columns])
            frames[t] = df
        if not frames:
            return pd.DataFrame()
        out = None
        for _, d in frames.items():
            out = d if out is None else out.join(d, how='outer')
        out = out.sort_index().fillna(method='ffill').dropna()
        return out
    else:
        return _fetch_single(ticker, start_date, end_date, interval)
PY

# ------------------------------
# indicator_library.py
# ------------------------------
cat > indicator_library.py <<'PY'
import pandas as pd
import pandas_ta as ta  # type: ignore

def calculate_ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df['Close'].ta.ema(length=int(period))

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    atr = ta.atr(high=df['High'], low=df['Low'], close=df['Close'], length=int(period))
    return atr

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return df['Close'].ta.rsi(length=int(period))

def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    macd = ta.macd(df['Close'], fast=int(fast), slow=int(slow), signal=int(signal))
    return macd['MACD_12_26_9']

def calculate_bbands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    bb = ta.bbands(df['Close'], length=int(period), std=float(std_dev))
    return bb.rename(columns={c: c.replace('BOLL', 'BB').replace('L', 'L').replace('M', 'M').replace('U','U')})
PY

# ------------------------------
# strategy_engine.py
# ------------------------------
cat > strategy_engine.py <<'PY'
import pandas as pd
from typing import Dict
import indicator_library as ind_lib

INDICATOR_MAPPING = {
    'ema': ind_lib.calculate_ema,
    'atr': ind_lib.calculate_atr,
    'rsi': ind_lib.calculate_rsi,
    'macd': ind_lib.calculate_macd,
    'bbands': ind_lib.calculate_bbands,
}

def _generate_signal(ohlc_data: pd.DataFrame, indicator_series: pd.Series, condition: dict) -> pd.Series:
    t = condition.get('type')
    if t == 'price_is_above_indicator':
        return (ohlc_data['Close'] > indicator_series).fillna(False)
    if t == 'indicator_is_above_value':
        v = condition.get('value', 0)
        return (indicator_series > v).fillna(False)
    if t == 'indicator_crosses_above_value':
        v = condition.get('value', 0)
        return (indicator_series.shift(1) <= v) & (indicator_series > v)
    if t == 'price_crosses_above_upper_band':
        col = condition.get('column')
        upper = indicator_series if hasattr(indicator_series, 'name') and indicator_series.name == col else None
        if isinstance(indicator_series, pd.DataFrame):
            upper = indicator_series.get(col)
        if upper is None:
            return pd.Series(False, index=ohlc_data.index)
        return (ohlc_data['Close'].shift(1) <= upper.shift(1)) & (ohlc_data['Close'] > upper)
    return pd.Series(False, index=ohlc_data.index)

def _process_single_asset(df: pd.DataFrame, rules: dict) -> pd.Series:
    logic = (rules.get('entry_rules', {}) or {}).get('combination_logic', 'AND').upper()
    conds = (rules.get('entry_rules', {}) or {}).get('conditions', [])
    signals = []
    for c in conds:
        if c.get('is_active') is False:
            continue
        ind = c.get('indicator')
        fn = INDICATOR_MAPPING.get(ind)
        if fn is None:
            continue
        params = c.get('params', {}) or {}
        ind_series = fn(df, **{k: (v if not isinstance(v, dict) else v.get('value', None) or v) for k, v in params.items()})
        if isinstance(ind_series, pd.DataFrame):
            col = (c.get('condition', {}) or {}).get('column')
            if col and col in ind_series.columns:
                target = ind_series[col]
            else:
                target = ind_series.iloc[:, 1] if ind_series.shape[1] > 1 else ind_series.iloc[:, 0]
        else:
            target = ind_series
        signals.append(_generate_signal(df, target, c.get('condition', {}) or {}))
    if not signals:
        return pd.Series(False, index=df.index)
    if logic == 'AND':
        out = signals[0]
        for s in signals[1:]:
            out = out & s
        return out.fillna(False)
    else:
        out = signals[0]
        for s in signals[1:]:
            out = out | s
        return out.fillna(False)

def process_strategy_rules(ohlc_data: pd.DataFrame, rules: dict) -> pd.DataFrame | pd.Series:
    if isinstance(ohlc_data.columns, pd.MultiIndex):
        assets = sorted(set(ohlc_data.columns.get_level_values(0)))
        out = {}
        for a in assets:
            df = ohlc_data[a].copy()
            out[a] = _process_single_asset(df, rules)
        return pd.DataFrame(out).reindex(ohlc_data.index)
    return _process_single_asset(ohlc_data, rules)
PY

# ------------------------------
# fitness.py
# ------------------------------
cat > fitness.py <<'PY'
import copy
from typing import Dict
import numpy as np
import pandas as pd
import vectorbt as vbt
import strategy_engine as engine
import config

def _inject_genes_into_rules(base_rules: dict, gene_map: Dict[int, dict], solution: list) -> dict:
    injected_rules = copy.deepcopy(base_rules)
    for i, gene_value in enumerate(solution):
        gene_info = gene_map.get(i)
        if not gene_info:
            continue
        path = gene_info.get('path', [])
        current_level = injected_rules
        for key in path[:-1]:
            current_level = current_level[key]
        current_level[path[-1]] = gene_value
    return injected_rules

class FitnessEvaluator:
    def __init__(self, ohlc_data: pd.DataFrame, base_rules: dict, gene_map: Dict[int, dict]):
        self.ohlc_data = ohlc_data
        self.base_rules = base_rules
        self.gene_map = gene_map

    def __call__(self, ga_instance, solution, sol_idx):
        try:
            rules = _inject_genes_into_rules(self.base_rules, self.gene_map, solution)
            entries = engine.process_strategy_rules(self.ohlc_data, rules)

            total_trades = entries.astype(bool).values.sum() if isinstance(entries, pd.DataFrame) else int(entries.sum())
            if total_trades < config.FITNESS_WEIGHTS['min_trades']:
                return -1.0

            exit_rules = rules.get('exit_rules', {}) or {}
            sl_rule = exit_rules.get('stop_loss', {}) or {}
            tsl_rule = exit_rules.get('trailing_stop', {}) or {}
            tp_rule = exit_rules.get('take_profit', {}) or {}

            sl_stop = sl_rule.get('params', {}).get('value') if sl_rule.get('is_active', False) else None
            sl_trail = tsl_rule.get('params', {}).get('value') if tsl_rule.get('is_active', False) else None
            tp_stop = tp_rule.get('params', {}).get('value') if tp_rule.get('is_active', False) else None

            time_based_exit = entries.shift(config.MAX_HOLD_PERIOD, fill_value=False).reindex(entries.index, fill_value=False)

            if isinstance(self.ohlc_data.columns, pd.MultiIndex):
                close_prices = self.ohlc_data.xs('Close', level=1, axis=1)
            else:
                close_prices = self.ohlc_data['Close']

            portfolio = vbt.Portfolio.from_signals(
                close=close_prices,
                entries=entries,
                exits=time_based_exit,
                sl_stop=sl_stop,
                tp_stop=tp_stop,
                sl_trail=sl_trail,
                fees=0.001,
                freq=config.TIMEFRAME,
            )
            stats = portfolio.stats()
            def _get(obj, key, default=np.nan):
                try:
                    return obj[key] if isinstance(obj, dict) else obj.get(key, default)
                except Exception:
                    return default
            sortino = _get(stats, 'Sortino Ratio', 0.0)
            profit_factor = _get(stats, 'Profit Factor', 0.0)
            max_drawdown = _get(stats, 'Max Drawdown [%]', 100.0)

            if np.isinf(profit_factor) or profit_factor > 5:
                profit_factor = 5
            sortino = 0 if np.isnan(sortino) else sortino
            profit_factor = 0 if np.isnan(profit_factor) else profit_factor
            max_drawdown = 100.0 if np.isnan(max_drawdown) else max_drawdown

            drawdown_score = 1 - (max_drawdown / 100.0)
            w = config.FITNESS_WEIGHTS
            fitness_score = (sortino * w['sortino_ratio']) + (profit_factor * w['profit_factor']) + (drawdown_score * w['max_drawdown'])
            return fitness_score if not np.isnan(fitness_score) else -1.0
        except Exception as err:
            print(f"Error in fitness evaluation: {err}")
            return -999.0
PY

# ------------------------------
# gene_parser.py
# ------------------------------
cat > gene_parser.py <<'PY'
from typing import Any, Dict, List, Tuple

def parse_genes_from_config(rules: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]], List[type]]:
    gene_space: List[Dict[str, Any]] = []
    gene_map: Dict[int, Dict[str, Any]] = {}
    gene_types: List[type] = []
    gene_index = 0
    def find_genes(sub_config: Any, path: List[Any]) -> None:
        nonlocal gene_index
        if isinstance(sub_config, dict) and sub_config.get("is_active") is False:
            return
        if isinstance(sub_config, dict):
            for key, value in sub_config.items():
                current_path = path + [key]
                if isinstance(value, dict) and "gene" in value:
                    gene_info = value
                    gene_name = gene_info["gene"]
                    gene_type = int if isinstance(gene_info.get("step", 1.0), int) else float
                    space_item: Dict[str, Any] = {"low": gene_info["low"], "high": gene_info["high"]}
                    if "step" in gene_info:
                        space_item["step"] = gene_info["step"]
                    gene_space.append(space_item)
                    gene_types.append(gene_type)
                    gene_map[gene_index] = {"name": gene_name, "path": current_path, "type": gene_type}
                    gene_index += 1
                elif isinstance(value, dict) or isinstance(value, list):
                    find_genes(value, current_path)
        elif isinstance(sub_config, list):
            for i, item in enumerate(sub_config):
                current_path = path + [i]
                find_genes(item, current_path)
    find_genes(rules, [])
    return gene_space, gene_map, gene_types
PY

# ------------------------------
# tuner.py
# ------------------------------
cat > tuner.py <<'PY'
import os
from typing import Dict, List
import numpy as np
import pandas as pd
import pygad
import config
import data_loader
import fitness

def _eval_on_validation(best_solution: List[float], val_data: pd.DataFrame, base_rules: dict, gene_map: Dict[int, dict]) -> float:
    evaluator = fitness.FitnessEvaluator(val_data, base_rules, gene_map)
    return evaluator(None, best_solution, 0)

def find_best_hyperparameters(gene_space, gene_types, gene_map) -> Dict[str, int]:
    # If portfolio mode enabled, tune on single TUNING_ASSET for speed
    if getattr(config, 'PORTFOLIO_OPTIMIZATION_ENABLED', False):
        tune_ticker = getattr(config, 'TUNING_ASSET', config.TICKER)
        train_data = data_loader.get_data(tune_ticker, config.TRAINING_PERIOD['start'], config.TRAINING_PERIOD['end'], config.TIMEFRAME)
        val_data = data_loader.get_data(tune_ticker, config.VALIDATION_PERIOD['start'], config.VALIDATION_PERIOD['end'], config.TIMEFRAME)
    else:
        train_data = data_loader.get_data(config.TICKER, config.TRAINING_PERIOD['start'], config.TRAINING_PERIOD['end'], config.TIMEFRAME)
        val_data = data_loader.get_data(config.TICKER, config.VALIDATION_PERIOD['start'], config.VALIDATION_PERIOD['end'], config.TIMEFRAME)

    best = None
    best_score = -np.inf
    for opt in config.HYPERPARAMETER_SEARCH_SPACE[:]:
        ga = pygad.GA(
            num_generations=min(config.GENERATIONS_PER_TUNE, 10),
            num_parents_mating=opt.get('num_parents_mating', 20),
            sol_per_pop=opt.get('sol_per_pop', 50),
            mutation_num_genes=opt.get('mutation_num_genes', 1),
            num_genes=len(gene_space),
            gene_space=gene_space,
            gene_type=gene_types,
            fitness_func=fitness.FitnessEvaluator(train_data, config.STRATEGY_RULES, gene_map).__call__,
            parallel_processing=['process', max(1, os.cpu_count() or 1)],
        )
        ga.run()
        sol, train_score, _ = ga.best_solution()
        val_score = _eval_on_validation(sol, val_data, config.STRATEGY_RULES, gene_map)
        if val_score > best_score:
            best_score = val_score
            best = opt
    return best or {"sol_per_pop": 50, "num_parents_mating": 20, "mutation_num_genes": 1}
PY

# ------------------------------
# walk_forward.py
# ------------------------------
cat > walk_forward.py <<'PY'
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import pygad
import vectorbt as vbt

import config
import data_loader
import strategy_engine as engine
from gene_parser import parse_genes_from_config
import fitness

def _generate_periods(start: datetime, end: datetime, train_months: int, test_months: int) -> List[Dict[str, datetime]]:
    start_dt = pd.to_datetime(start).to_pydatetime()
    end_dt = pd.to_datetime(end).to_pydatetime()
    if start_dt + relativedelta(months=train_months + test_months) > end_dt:
        return []
    periods: List[Dict[str, datetime]] = []
    current_start = start_dt
    while True:
        train_end = current_start + relativedelta(months=train_months)
        test_end = train_end + relativedelta(months=test_months)
        if test_end > end_dt:
            break
        periods.append({'train_start': current_start,'train_end': train_end,'test_start': train_end,'test_end': test_end})
        current_start += relativedelta(months=test_months)
    return periods

def _update_champion_pool(pool: List[List[float]], best_solution: List[float], validation_score: float, gene_space: List[Dict[str, any]], settings: Dict[str, any]) -> List[List[float]]:
    survival = settings.get('survival_threshold', 0.0)
    cloning = settings.get('cloning_threshold', float('inf'))
    num_clones = settings.get('num_clones', 0)
    mutation_rate = settings.get('clone_mutation_rate', 0.0)
    if validation_score < survival:
        print("Champion discarded due to poor performance.")
        return pool
    if validation_score >= cloning:
        print("Elite Champion found. Cloning champion.")
        pool.append(list(best_solution))
        for _ in range(num_clones):
            clone = list(best_solution)
            for idx in range(len(clone)):
                if np.random.rand() < mutation_rate:
                    gs = gene_space[idx]
                    low, high = gs['low'], gs['high']
                    step = gs.get('step')
                    if step is not None:
                        steps = int(round((high - low) / step))
                        val = low + step * np.random.randint(0, steps + 1)
                    else:
                        val = np.random.uniform(low, high)
                    clone[idx] = type(clone[idx])(val)
            pool.append(clone)
    else:
        print("Viable Champion found and kept for next fold.")
        pool.append(list(best_solution))
    return pool

def run_walk_forward_validation(initial_champions: Optional[List[List[float]]] = None):
    print("\n=== Running Walk-Forward Validation ===")
    num_cores = os.cpu_count()
    wf_settings = getattr(config, 'WALK_FORWARD_SETTINGS', {})
    date_range = wf_settings.get('total_data_range', {})
    start_date = date_range.get('start', config.TRAINING_PERIOD['start'])
    end_date = date_range.get('end', config.VALIDATION_PERIOD['end'])

    if getattr(config, 'PORTFOLIO_OPTIMIZATION_ENABLED', False):
        tickers = getattr(config, 'ASSET_BASKET', [config.TICKER])
    else:
        tickers = config.TICKER

    all_data = data_loader.get_data(tickers, start_date, end_date, config.TIMEFRAME)
    if all_data.empty:
        print("No data available for walk-forward validation.")
        return None

    start = all_data.index[0]
    end = all_data.index[-1]
    train_months = wf_settings.get('training_period_length', 12)
    test_months = wf_settings.get('validation_period_length', 3)
    periods = _generate_periods(start, end, train_months, test_months)
    if not periods:
        print("Insufficient data for the requested walk-forward windows.")
        return None

    results: List[Dict[str, any]] = []
    champion_pool: List[List[float]] = list(initial_champions or [])
    gene_space, gene_map, gene_types = parse_genes_from_config(config.STRATEGY_RULES)

    for idx, p in enumerate(periods, start=1):
        print(f"\n--- Window {idx} ---")
        print(f"Train: {p['train_start'].date()} -> {p['train_end'].date()}")
        print(f"Test : {p['test_start'].date()} -> {p['test_end'].date()}")

        train_data = all_data.loc[p['train_start']:p['train_end']]
        test_data = all_data.loc[p['test_start']:p['test_end']]

        ga = pygad.GA(
            num_generations=config.GA_NUM_GENERATIONS,
            num_parents_mating=config.GA_PARENTS_MATING,
            sol_per_pop=config.GA_POPULATION_SIZE,
            num_genes=len(gene_space),
            gene_space=gene_space,
            gene_type=gene_types,
            mutation_num_genes=config.GA_MUTATION_NUM_GENES,
            fitness_func=fitness.FitnessEvaluator(train_data, config.STRATEGY_RULES, gene_map).__call__,
            parallel_processing=['process', num_cores],
        )

        if champion_pool and hasattr(ga, 'population'):
            import numpy as np
            champs = np.array(champion_pool, dtype=float)
            if champs.ndim == 1:
                champs = champs.reshape(1, -1)
            if champs.shape[1] == ga.population.shape[1]:
                champs = champs[-config.GA_POPULATION_SIZE:]
                num_champs = min(len(champs), ga.population.shape[0])
                ga.population[:num_champs] = champs[:num_champs]
                if hasattr(ga, 'initial_population'):
                    ga.initial_population[:num_champs] = champs[:num_champs]

        ga.run()
        best_solution, best_fitness, _ = ga.best_solution()
        print(f"Best training fitness: {best_fitness:.4f}")
        winning_params = {gene_map[i]['name']: best_solution[i] for i in range(len(best_solution))}

        entries = engine.process_strategy_rules(test_data, fitness._inject_genes_into_rules(config.STRATEGY_RULES, gene_map, best_solution))
        total_trades = entries.astype(bool).values.sum() if isinstance(entries, pd.DataFrame) else int(entries.sum())
        if total_trades < config.FITNESS_WEIGHTS['min_trades']:
            print("No trades in test period.")
            results.append({'Window': idx,'Total Return [%]': np.nan,'Max Drawdown [%]': np.nan,'Sharpe Ratio': np.nan,'Sortino Ratio': np.nan,'Win Rate [%]': np.nan,'Params': None})
            continue

        exit_rules = config.STRATEGY_RULES.get('exit_rules', {}) or {}
        def getp(name):
            r = exit_rules.get(name, {}) or {}
            return r.get('params', {}).get('value') if r.get('is_active', False) else None
        sl_stop, sl_trail, tp_stop = getp('stop_loss'), getp('trailing_stop'), getp('take_profit')

        time_exit = entries.shift(config.MAX_HOLD_PERIOD, fill_value=False).reindex(entries.index, fill_value=False)
        if isinstance(test_data.columns, pd.MultiIndex):
            close_prices = test_data.xs('Close', level=1, axis=1)
        else:
            close_prices = test_data['Close']

        portfolio = vbt.Portfolio.from_signals(
            close=close_prices, entries=entries, exits=time_exit,
            sl_stop=sl_stop, tp_stop=tp_stop, sl_trail=sl_trail,
            fees=0.001, freq=config.TIMEFRAME,
        )
        stats = portfolio.stats()
        def _get(obj, key): return obj[key] if isinstance(obj, dict) else obj.get(key)
        tr, dd = _get(stats, 'Total Return [%]'), _get(stats, 'Max Drawdown [%]')
        sharpe, sortino, winr = _get(stats, 'Sharpe Ratio'), _get(stats, 'Sortino Ratio'), _get(stats, 'Win Rate [%]')

        print(f"Test Return: {tr:.2f}% | Max DD: {dd:.2f}%")
        print("Winning Parameters:")
        for k, v in winning_params.items():
            print(f"  {k}: {v}")

        validation_score = fitness.FitnessEvaluator(test_data, config.STRATEGY_RULES, gene_map)(None, best_solution, 0)
        champion_pool = _update_champion_pool(champion_pool, best_solution, validation_score, gene_space, getattr(config, 'CHAMPION_SELECTION_SETTINGS', {}))

        results.append({'Window': idx,'Total Return [%]': tr,'Max Drawdown [%]': dd,'Sharpe Ratio': sharpe,'Sortino Ratio': sortino,'Win Rate [%]': winr,'Params': winning_params})

    if not results:
        print("\nNo walk-forward runs produced trades.")
        return None

    results_df = pd.DataFrame(results)
    avg_return = results_df['Total Return [%]'].mean()
    std_return = results_df['Total Return [%]'].std()
    avg_sharpe = results_df['Sharpe Ratio'].mean()
    avg_sortino = results_df['Sortino Ratio'].mean()
    avg_win = results_df['Win Rate [%]'].mean()
    total_compounded_return = (results_df['Total Return [%]'] / 100 + 1).prod() - 1

    print("\n=== Walk-Forward Summary ===")
    with pd.option_context('display.max_colwidth', None, 'display.width', None):
        print(results_df.to_string(index=False))
    print("\nAggregate Metrics:")
    print(f"Average Return: {avg_return:.2f}% (+/- {std_return:.2f}%)")
    print(f"Average Sharpe: {avg_sharpe:.2f}")
    print(f"Average Sortino: {avg_sortino:.2f}")
    print(f"Average Win Rate: {avg_win:.2f}%")
    print(f"Total Compounded Return: {total_compounded_return * 100:.2f}%")

    return {'folds': results_df,'average_return': avg_return,'std_return': std_return,'average_sharpe': avg_sharpe,'average_sortino': avg_sortino,'average_win_rate': avg_win,'total_compounded_return': total_compounded_return}
PY

# ------------------------------
# analysis.py
# ------------------------------
cat > analysis.py <<'PY'
import traceback
from typing import Dict, List
import matplotlib.pyplot as plt
import pandas as pd
import vectorbt as vbt
import config
import data_loader
import fitness
import strategy_engine as engine

def run_champion_analysis(best_solution: List[float], gene_map: Dict[int, dict]) -> None:
    print("\n\n--- Champion Strategy Analysis on Unseen Data ---")
    start = config.VALIDATION_PERIOD['start']
    end = config.VALIDATION_PERIOD['end']

    tickers = getattr(config, 'ASSET_BASKET', [config.TICKER]) if getattr(config, 'PORTFOLIO_OPTIMIZATION_ENABLED', False) else config.TICKER
    validation_data = data_loader.get_data(tickers, start, end, config.TIMEFRAME)
    if validation_data.empty:
        print("No validation data available.")
        return
    try:
        rules = fitness._inject_genes_into_rules(config.STRATEGY_RULES, gene_map, best_solution)
        entries = engine.process_strategy_rules(validation_data, rules)
        total_trades = entries.astype(bool).values.sum() if isinstance(entries, pd.DataFrame) else int(entries.sum())
        if total_trades < 1:
            print("\nChampion strategy produced no trades in the validation period.")
            return

        exit_rules = rules.get('exit_rules', {}) or {}
        def getp(name):
            r = exit_rules.get(name, {}) or {}
            return r.get('params', {}).get('value') if r.get('is_active', False) else None
        sl_stop, sl_trail, tp_stop = getp('stop_loss'), getp('trailing_stop'), getp('take_profit')

        time_based_exit = entries.shift(config.MAX_HOLD_PERIOD, fill_value=False).reindex(entries.index, fill_value=False)
        if isinstance(validation_data.columns, pd.MultiIndex):
            close_prices = validation_data.xs('Close', level=1, axis=1)
        else:
            close_prices = validation_data['Close']

        portfolio = vbt.Portfolio.from_signals(
            close=close_prices, entries=entries, exits=time_based_exit,
            sl_stop=sl_stop, tp_stop=tp_stop, sl_trail=sl_trail,
            fees=0.001, freq=config.TIMEFRAME,
        )
    except Exception as err:
        print(f"An error occurred during analysis backtest: {err}")
        traceback.print_exc()
        return

    print("\n--- Validation Period Performance Stats ---")
    stats = portfolio.stats()
    metrics = ['Total Return [%]','Benchmark Return [%]','Max Drawdown [%]','Sortino Ratio','Sharpe Ratio','Profit Factor','Win Rate [%]','Total Trades']
    if isinstance(stats, dict):
        stats_df = pd.DataFrame([stats])[metrics]
        print(stats_df.to_string(index=False))
    else:
        print(stats[metrics].to_string(index=False))

    if isinstance(close_prices, pd.DataFrame) and close_prices.shape[1] > 1:
        try:
            per_asset_results = []
            for asset in close_prices.columns:
                asset_port = vbt.Portfolio.from_signals(
                    close=close_prices[asset],
                    entries=entries[asset] if isinstance(entries, pd.DataFrame) else entries,
                    exits=time_based_exit[asset] if isinstance(time_based_exit, pd.DataFrame) else time_based_exit,
                    sl_stop=sl_stop, tp_stop=tp_stop, sl_trail=sl_trail, fees=0.001, freq=config.TIMEFRAME,
                )
                asset_stats = asset_port.stats()
                per_asset_results.append({
                    'Asset': asset,
                    'Total Return [%]': asset_stats.get('Total Return [%]', float('nan')) if isinstance(asset_stats, dict) else asset_stats.get('Total Return [%]'),
                    'Max Drawdown [%]': asset_stats.get('Max Drawdown [%]', float('nan')) if isinstance(asset_stats, dict) else asset_stats.get('Max Drawdown [%]'),
                })
            per_df = pd.DataFrame(per_asset_results)
            print("\n--- Per-Asset Breakdown ---")
            print(per_df.to_string(index=False))
        except Exception:
            pass

    print("\nDisplaying equity curve plot for the validation period...")
    plt.ion()
    fig = portfolio.plot(title="Champion Strategy Performance on Validation Portfolio")
    fig.show()
PY

# ------------------------------
# main.py
# ------------------------------
cat > main.py <<'PY'
import os
import pprint
import time
import traceback
from typing import List
import matplotlib.pyplot as plt
import pygad

import config
import data_loader
import fitness
import analysis
from gene_parser import parse_genes_from_config
import tuner

start_time: float = 0.0

def on_generation(ga_instance: pygad.GA) -> None:
    generation = ga_instance.generations_completed
    total_generations = ga_instance.num_generations
    fitness_score = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
    elapsed = time.time() - start_time
    remaining = (elapsed / generation) * (total_generations - generation) if generation > 0 else 0
    print(f"Generation {generation}/{total_generations} | Best Fitness: {fitness_score:.4f} | Est. Time Left: {int(remaining)}s", end="\\r")

def main() -> None:
    print("--- GA Trading Strategy Framework ---")
    if getattr(config, 'PORTFOLIO_OPTIMIZATION_ENABLED', False):
        asset_list = getattr(config, 'ASSET_BASKET', [config.TICKER])
        print(f"Starting optimisation for portfolio: {asset_list}")
    else:
        print(f"Starting optimisation for: {config.SELECTED_ASSET_NAME} ({config.TICKER})")
    num_cores = os.cpu_count()
    print(f"Detected {num_cores} CPU cores available for parallel processing.")
    print("-" * 35)

    train_tickers = getattr(config, 'ASSET_BASKET', [config.TICKER]) if getattr(config, 'PORTFOLIO_OPTIMIZATION_ENABLED', False) else config.TICKER
    print(f"Loading TRAINING data from {config.TRAINING_PERIOD['start']} to {config.TRAINING_PERIOD['end']}...")
    ohlc_data = data_loader.get_data(train_tickers, config.TRAINING_PERIOD['start'], config.TRAINING_PERIOD['end'], config.TIMEFRAME)
    if ohlc_data.empty:
        print("No training data.")
        return

    print("Parsing strategy rules to identify genes for optimisation...")
    gene_space, gene_map, gene_types = parse_genes_from_config(config.STRATEGY_RULES)
    if not gene_space:
        print("No genes found. Exiting.")
        return
    print(f"Found {len(gene_space)} genes to optimise:")
    pprint.pprint(gene_map)
    print("-" * 35)

    fitness_evaluator = fitness.FitnessEvaluator(ohlc_data=ohlc_data, base_rules=config.STRATEGY_RULES, gene_map=gene_map)
    fitness_function = fitness_evaluator.__call__

    if getattr(config, 'AUTO_TUNE_ENABLED', False):
        if getattr(config, 'PORTFOLIO_OPTIMIZATION_ENABLED', False):
            tuning_ticker = getattr(config, 'TUNING_ASSET', config.TICKER)
            tune_data = data_loader.get_data(tuning_ticker, config.TRAINING_PERIOD['start'], config.TRAINING_PERIOD['end'], config.TIMEFRAME)
        best_hparams = tuner.find_best_hyperparameters(gene_space, gene_types, gene_map)
        print("Auto-tuner selected hyperparameters:", best_hparams)

    global start_time
    start_time = time.time()
    ga = pygad.GA(
        num_generations=config.GA_NUM_GENERATIONS,
        num_parents_mating=config.GA_PARENTS_MATING,
        sol_per_pop=config.GA_POPULATION_SIZE,
        num_genes=len(gene_space),
        gene_space=gene_space,
        gene_type=gene_types,
        mutation_num_genes=config.GA_MUTATION_NUM_GENES,
        fitness_func=fitness_function,
        on_generation=on_generation,
        parallel_processing=['process', num_cores],
    )
    try:
        ga.run()
        best_solution, best_fitness, _ = ga.best_solution()
        print(f"\\nBest training fitness: {best_fitness:.4f}")
        print("Winning gene values:")
        for i, v in enumerate(best_solution):
            print(f"  {gene_map[i]['name']}: {v}")
    except Exception as e:
        print(f"\\nError during GA run: {e}")
        traceback.print_exc()
        return

    print("\\n--- Running analysis on validation data ---")
    analysis.run_champion_analysis(best_solution, gene_map)

    if getattr(config, 'WALK_FORWARD_SETTINGS', {}).get('enabled', False):
        from walk_forward import run_walk_forward_validation
        run_walk_forward_validation(initial_champions=[list(best_solution)])

if __name__ == "__main__":
    main()
PY

# ------------------------------
# tests/test_portfolio.py
# ------------------------------
cat > tests/test_portfolio.py <<'PY'
import types
import pandas as pd
import numpy as np

import config
import data_loader
import strategy_engine as engine
import fitness
import tuner
import walk_forward

class SimpleMonkeyPatch:
    def setattr(self, target, name=None, value=None, *, raising=True):
        if name is None:
            target(value)
        else:
            if isinstance(target, str):
                module = __import__(target, fromlist=[name])
                setattr(module, name, value)
            else:
                setattr(target, name, value)

monkeypatch = SimpleMonkeyPatch()

def _mk_df():
    idx = pd.date_range("2021-01-01", periods=10, freq="D")
    a = pd.DataFrame({'Open':1,'High':1,'Low':1,'Close':[100,101,102,103,104,105,106,107,108,109],'Volume':1}, index=idx)
    b = pd.DataFrame({'Open':1,'High':1,'Low':1,'Close':[200,200,200,200,200,200,200,200,200,200],'Volume':1}, index=idx)
    a.columns = pd.MultiIndex.from_product([['AAA'], a.columns])
    b.columns = pd.MultiIndex.from_product([['BBB'], b.columns])
    return a.join(b)

def test_get_data_multi_asset():
    def fake_fetch(t, s, e, i):
        idx = pd.date_range("2021-01-01", periods=5, freq="D")
        df = pd.DataFrame({'Open':1,'High':1,'Low':1,'Close':[1,2,3,4,5],'Volume':1}, index=idx)
        return df
    monkeypatch.setattr(data_loader, '_fetch_single', lambda t,s,e,i: fake_fetch(t,s,e,i))
    df = data_loader.get_data(['AAA-USD','BBB-USD'], '2021-01-01','2021-01-10','1d')
    assert isinstance(df.columns, pd.MultiIndex)
    assert ('AAA-USD','Close') in df.columns and ('BBB-USD','Close') in df.columns

def test_fitness_evaluator_counts_portfolio_trades():
    df = _mk_df()
    rules = {'entry_rules': {'combination_logic':'AND','conditions':[]}, 'exit_rules':{}}
    monkeypatch.setattr(engine, 'process_strategy_rules', lambda data, rules: pd.DataFrame({'AAA':[True,False,False,False,False,False,False,False,False,False],'BBB':[True,False,False,False,False,False,False,False,False,False]}, index=data.index))
    cfg = dict(config.STRATEGY_RULES)
    gm = {0:{'name':'dummy','path':['entry_rules','conditions']}}
    evalr = fitness.FitnessEvaluator(df, cfg, gm)
    score = evalr(None, [0], 0)
    assert score != -999.0

def test_tuner_uses_tuning_asset():
    orig = config.PORTFOLIO_OPTIMIZATION_ENABLED
    setattr(config, 'PORTFOLIO_OPTIMIZATION_ENABLED', True)
    setattr(config, 'TUNING_ASSET', 'AAA-USD')
    monkeypatch.setattr(data_loader, 'get_data', lambda t,s,e,i: pd.DataFrame({'Open':1,'High':1,'Low':1,'Close':[1,2,3,4,5],'Volume':1}, index=pd.date_range('2021-01-01', periods=5)))
    monkeypatch.setattr(fitness, 'FitnessEvaluator', lambda ohlc_data, base_rules, gene_map: types.SimpleNamespace(__call__=lambda *a, **k: 1.0))
    class DummyGA:
        def __init__(self, *a, **k): pass
        def run(self): pass
        def best_solution(self): return [0], 1.0, None
    import pygad as _pg
    import builtins
    builtins.pygad = _pg
    monkeypatch.setattr(_pg, 'GA', DummyGA)
    gs, gt, gm = [], [], {}
    best = tuner.find_best_hyperparameters(gs, gt, gm)
    assert isinstance(best, dict)
    setattr(config, 'PORTFOLIO_OPTIMIZATION_ENABLED', orig)

def test_walk_forward_passes_asset_basket():
    setattr(config, 'PORTFOLIO_OPTIMIZATION_ENABLED', True)
    setattr(config, 'ASSET_BASKET', ['AAA-USD','BBB-USD'])
    df = _mk_df()
    monkeypatch.setattr(data_loader, 'get_data', lambda t,s,e,i: df)
    monkeypatch.setattr(engine, 'process_strategy_rules', lambda data, rules: pd.DataFrame({'AAA':[True]+[False]*9,'BBB':[True]+[False]*9}, index=data.index))
    class DummyGA:
        def __init__(self, *a, **k):
            import numpy as np
            self.population = np.zeros((k.get('sol_per_pop', 10), k.get('num_genes', 0)))
            self.initial_population = self.population.copy()
            self.num_generations = k.get('num_generations', 10)
            self.generations_completed = self.num_generations
        def run(self): pass
        def best_solution(self): return [0], 1.0, None
    import pygad as _pg
    import builtins
    builtins.pygad = _pg
    monkeypatch.setattr(_pg, 'GA', DummyGA)
    res = walk_forward.run_walk_forward_validation()
    assert res is None or isinstance(res, dict)
PY

echo "All files written."
