import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_asset_basket_converted_for_binance(monkeypatch):
    sys.modules.pop('config', None)
    import config
    monkeypatch.setattr(config, 'DATA_SOURCE', 'binance', raising=False)
    importlib.reload(config)

    assert all('-' not in t for t in config.ASSET_BASKET)
    assert all(t.endswith('USDT') for t in config.ASSET_BASKET)
    assert '-' not in config.TUNING_ASSET
    assert config.TUNING_ASSET.endswith('USDT')
