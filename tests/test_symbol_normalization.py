from pathlib import Path
import sys
import types
import pandas as pd


# Ensure repository root is on the import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy optional dependencies before importing modules
sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))
sys.modules.setdefault("vectorbt", types.ModuleType("vectorbt"))

# Stub binance client to avoid import errors
binance_mod = types.ModuleType("binance")
client_mod = types.ModuleType("binance.client")
client_mod.Client = object
binance_mod.client = client_mod
sys.modules.setdefault("binance", binance_mod)
sys.modules.setdefault("binance.client", client_mod)

import data_loader  # noqa: E402
import tuner  # noqa: E402
import walk_forward  # noqa: E402


def test_normalize_symbol_binance(monkeypatch):
    monkeypatch.setattr(data_loader.config, "DATA_SOURCE", "binance", raising=False)
    assert data_loader.normalize_symbol("BTC-USD") == "BTCUSDT"


def test_normalize_symbol_yfinance(monkeypatch):
    monkeypatch.setattr(data_loader.config, "DATA_SOURCE", "yfinance", raising=False)
    assert data_loader.normalize_symbol("BTC-USD") == "BTC-USD"


def test_get_data_normalizes_single_ticker(monkeypatch):
    df = pd.DataFrame(
        {
            "Open": [1],
            "High": [1],
            "Low": [1],
            "Close": [1],
            "Volume": [1],
        },
        index=pd.date_range("2020-01-01", periods=1),
    )

    captured = {}

    monkeypatch.setattr(data_loader.config, "DATA_SOURCE", "binance", raising=False)
    monkeypatch.setattr(data_loader.os.path, "exists", lambda path: False)
    monkeypatch.setattr(data_loader.os, "makedirs", lambda *a, **k: None)
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda *a, **k: None, raising=False)

    def fake_get_binance_data(ticker, *a, **k):
        captured["ticker"] = ticker
        return df

    monkeypatch.setattr(data_loader, "_get_binance_data", fake_get_binance_data)

    data_loader.get_data("BTC-USD", "2020-01-01", "2020-01-02")
    assert captured["ticker"] == "BTCUSDT"


def test_get_data_normalizes_asset_basket(monkeypatch):
    df = pd.DataFrame(
        {
            "Open": [1],
            "High": [1],
            "Low": [1],
            "Close": [1],
            "Volume": [1],
        },
        index=pd.date_range("2020-01-01", periods=1),
    )

    captured = []

    monkeypatch.setattr(data_loader.config, "DATA_SOURCE", "binance", raising=False)
    monkeypatch.setattr(data_loader.os.path, "exists", lambda path: False)
    monkeypatch.setattr(data_loader.os, "makedirs", lambda *a, **k: None)
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda *a, **k: None, raising=False)

    def fake_get_binance_data(ticker, *a, **k):
        captured.append(ticker)
        return df

    monkeypatch.setattr(data_loader, "_get_binance_data", fake_get_binance_data)

    result = data_loader.get_data(["ETH-USD", "XRP-USD"], "2020-01-01", "2020-01-02")
    assert captured == ["ETHUSDT", "XRPUSDT"]
    assert isinstance(result.columns, pd.MultiIndex)
    assert set(result.columns.get_level_values(0)) == {"ETHUSDT", "XRPUSDT"}


def test_tuner_validation_loads_normalized_symbol(monkeypatch):
    df = pd.DataFrame(
        {
            "Open": [1, 1],
            "High": [1, 1],
            "Low": [1, 1],
            "Close": [1, 1],
            "Volume": [1, 1],
        },
        index=pd.date_range("2020-01-01", periods=2),
    )

    captured = {}

    def fake_get_binance(ticker, *a, **k):
        captured["ticker"] = ticker
        return df

    monkeypatch.setattr(tuner.data_loader, "_get_binance_data", fake_get_binance)
    monkeypatch.setattr(tuner.data_loader.os.path, "exists", lambda path: False)
    monkeypatch.setattr(tuner.data_loader.os, "makedirs", lambda *a, **k: None)
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda *a, **k: None, raising=False)

    monkeypatch.setattr(tuner.config, "DATA_SOURCE", "binance", raising=False)
    monkeypatch.setattr(tuner.config, "PORTFOLIO_OPTIMIZATION_ENABLED", True, raising=False)
    monkeypatch.setattr(tuner.config, "TUNING_ASSET", "ADA-USD", raising=False)

    monkeypatch.setattr(tuner.fitness, "_inject_genes_into_rules", lambda *a, **k: {})
    monkeypatch.setattr(
        tuner.engine,
        "process_strategy_rules",
        lambda *a, **k: pd.Series([True, True], index=df.index),
    )
    monkeypatch.setattr(tuner.fitness, "_count_trades", lambda *a, **k: 1)
    monkeypatch.setattr(
        tuner.fitness,
        "run_portfolio_backtest",
        lambda *a, **k: (None, None, pd.Series({"Sortino Ratio": 1.0}), None),
    )

    monkeypatch.setattr(pd.DataFrame, "ta", None, raising=False)
    monkeypatch.setattr(tuner.vbt, "Portfolio", object, raising=False)

    tuner._evaluate_on_validation([], {})
    assert captured["ticker"] == "ADAUSDT"


def test_walk_forward_loads_normalized_symbol(monkeypatch):
    df = pd.DataFrame(
        {
            "Open": [1, 1],
            "High": [1, 1],
            "Low": [1, 1],
            "Close": [1, 1],
            "Volume": [1, 1],
        },
        index=pd.date_range("2020-01-01", periods=2),
    )

    captured = {}

    def fake_get_binance(ticker, *a, **k):
        captured["ticker"] = ticker
        return df

    monkeypatch.setattr(walk_forward.data_loader, "_get_binance_data", fake_get_binance)
    monkeypatch.setattr(walk_forward.data_loader.os.path, "exists", lambda path: False)
    monkeypatch.setattr(walk_forward.data_loader.os, "makedirs", lambda *a, **k: None)
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda *a, **k: None, raising=False)

    monkeypatch.setattr(walk_forward.config, "DATA_SOURCE", "binance", raising=False)
    monkeypatch.setattr(walk_forward.config, "PORTFOLIO_OPTIMIZATION_ENABLED", False, raising=False)
    monkeypatch.setattr(walk_forward.config, "TICKER", "DOGE-USD", raising=False)
    monkeypatch.setattr(walk_forward.config, "FITNESS_WEIGHTS", {"min_trades": 0}, raising=False)

    monkeypatch.setattr(
        walk_forward,
        "_generate_periods",
        lambda *a, **k: [
            {
                "train_start": df.index[0],
                "train_end": df.index[1],
                "test_start": df.index[0],
                "test_end": df.index[1],
            }
        ],
    )
    monkeypatch.setattr(walk_forward, "parse_genes_from_config", lambda *a, **k: ([], {}, []))

    class DummyEvaluator:
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            return 1.0

    monkeypatch.setattr(walk_forward.fitness, "FitnessEvaluator", DummyEvaluator)
    monkeypatch.setattr(
        walk_forward.engine,
        "process_strategy_rules",
        lambda *a, **k: pd.Series([True, True], index=df.index),
    )

    class DummyGA:
        def __init__(self, *a, **k):
            pass

        def run(self):
            pass

        def best_solution(self, **k):
            return [], 0, None

    monkeypatch.setattr(walk_forward.pygad, "GA", DummyGA)
    monkeypatch.setattr(
        walk_forward.fitness,
        "run_portfolio_backtest",
        lambda *a, **k: (
            None,
            None,
            pd.Series(
                {
                    "Total Return [%]": 0,
                    "Max Drawdown [%]": 0,
                    "Sharpe Ratio": 0,
                    "Sortino Ratio": 0,
                    "Win Rate [%]": 0,
                }
            ),
            None,
        ),
    )

    walk_forward.run_walk_forward_validation()
    assert captured["ticker"] == "DOGEUSDT"
