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
