import pytest

import config


def _base_final_cfg() -> dict:
    return {
        "INCLUDE_CLASSES": ["Stars"],
        "PARAM_RCV_WATCHLIST": 0.1,
        "PARAM_RCV_UNSTABLE": 0.2,
        "WEIGHTING_SCHEME": "equal",
        "ASSET_WEIGHTS_OVERRIDE": {},
        "SHRINK_TO_EQUAL": 0.25,
        "MAX_WEIGHT_CAP": 0.6,
        "MIN_WEIGHT_FLOOR": 0.1,
        "PARAM_SENSITIVITY_THRESHOLD": 0.4,
        "WEIGHT_SENSITIVITY_THRESHOLD": 0.3,
        "WEIGHT_SENSITIVITY_RATIO_THRESHOLD": 0.1,
        "PARAM_VALUE_DECIMALS": {"default": 3},
    }


def test_validate_combination_logic_errors():
    rules = {"entry_rules": {"combination_logic": "XOR", "conditions": []}}
    with pytest.raises(config.ConfigurationError):
        config._validate_combination_logic(rules)
    rules = {
        "entry_rules": {
            "combination_logic": "VOTE",
            "vote_threshold": 2,
            "conditions": [{}],
        }
    }
    with pytest.raises(config.ConfigurationError):
        config._validate_combination_logic(rules)


def test_import_bad_combination_logic_raises(tmp_path, monkeypatch):
    mod = tmp_path / "bad_logic_mod.py"
    mod.write_text(
        "from config import _validate_combination_logic\n"
        "STRATEGY_RULES = {'entry_rules': {'combination_logic': 'XOR', 'conditions': []}}\n"
        "_validate_combination_logic(STRATEGY_RULES)\n"
    )
    monkeypatch.syspath_prepend(tmp_path)
    with pytest.raises(Exception) as exc:
        import importlib

        importlib.import_module("bad_logic_mod")
    assert exc.value.__class__.__name__ == "ConfigurationError"


def test_import_gene_driven_combination_logic(tmp_path, monkeypatch):
    mod = tmp_path / "gene_logic_mod.py"
    mod.write_text(
        "from config import _validate_combination_logic\n"
        "STRATEGY_RULES = {'entry_rules': {'combination_logic': {'name':'cl',"
        "'options':['AND','OR']}, 'conditions': []}}\n"
        "_validate_combination_logic(STRATEGY_RULES)\n"
    )
    monkeypatch.syspath_prepend(tmp_path)
    import importlib

    importlib.import_module("gene_logic_mod")


def test_validate_final_strategy_config_override_errors():
    cfg = _base_final_cfg()
    cfg.update(
        {
            "WEIGHTING_SCHEME": "override",
            "ASSET_WEIGHTS_OVERRIDE": {"A": 0.7, "B": 0.3, "C": -0.05},
        }
    )
    with pytest.raises(config.ConfigurationError):
        config.validate_final_strategy_config(cfg)


def test_validate_final_strategy_config_warns_unknown_class():
    cfg = _base_final_cfg()
    cfg["INCLUDE_CLASSES"] = ["Stars", "Unknown"]
    with pytest.warns(UserWarning):
        config.validate_final_strategy_config(cfg)


def test_validate_final_strategy_config_accepts_valid_config():
    cfg = _base_final_cfg()
    cfg.update(
        {
            "INCLUDE_CLASSES": ["Stars", "Stalwarts"],
            "WEIGHTING_SCHEME": "proportional",
            "USE_RECENCY_WEIGHTING": True,
            "FOLD_DECAY_RATE": 0.25,
            "WEIGHT_SENSITIVITY_RATIO_THRESHOLD": 0.1,
            "PARAM_VALUE_DECIMALS": {"default": 3, "tp_pct_1": 2},
        }
    )
    config.validate_final_strategy_config(cfg)


def test_validate_final_strategy_config_rejects_non_mapping():
    with pytest.raises(config.ConfigurationError):
        config.validate_final_strategy_config(123)  # type: ignore[arg-type]


def test_validate_final_strategy_config_requires_sequence():
    cfg = _base_final_cfg()
    cfg["INCLUDE_CLASSES"] = "Stars"  # type: ignore[assignment]
    with pytest.raises(config.ConfigurationError):
        config.validate_final_strategy_config(cfg)


def test_validate_final_strategy_config_requires_strings():
    cfg = _base_final_cfg()
    cfg["INCLUDE_CLASSES"] = ["Stars", 5]  # type: ignore[list-item]
    with pytest.raises(config.ConfigurationError):
        config.validate_final_strategy_config(cfg)


def test_validate_final_strategy_config_rejects_empty_class_name():
    cfg = _base_final_cfg()
    cfg["INCLUDE_CLASSES"] = [" "]
    with pytest.raises(config.ConfigurationError):
        config.validate_final_strategy_config(cfg)


def test_validate_final_strategy_config_watchlist_boundaries():
    cfg = _base_final_cfg()
    cfg["PARAM_RCV_WATCHLIST"] = 0.3
    cfg["PARAM_RCV_UNSTABLE"] = 0.2
    with pytest.raises(config.ConfigurationError):
        config.validate_final_strategy_config(cfg)


def test_validate_final_strategy_config_shrink_bounds():
    cfg = _base_final_cfg()
    cfg["SHRINK_TO_EQUAL"] = 1.5
    with pytest.raises(config.ConfigurationError):
        config.validate_final_strategy_config(cfg)


def test_validate_final_strategy_config_recency_requires_decay():
    cfg = _base_final_cfg()
    cfg.update({"USE_RECENCY_WEIGHTING": True, "FOLD_DECAY_RATE": 0.0})
    with pytest.raises(config.ConfigurationError):
        config.validate_final_strategy_config(cfg)


def test_validate_final_strategy_config_sensitivity_bounds():
    cfg = _base_final_cfg()
    cfg["PARAM_SENSITIVITY_THRESHOLD"] = 1.5
    with pytest.raises(config.ConfigurationError):
        config.validate_final_strategy_config(cfg)


def test_validate_final_strategy_config_weight_sensitivity_bounds():
    cfg = _base_final_cfg()
    cfg["WEIGHT_SENSITIVITY_THRESHOLD"] = -0.1
    with pytest.raises(config.ConfigurationError):
        config.validate_final_strategy_config(cfg)


def test_validate_final_strategy_config_ratio_threshold_bounds():
    cfg = _base_final_cfg()
    cfg["WEIGHT_SENSITIVITY_RATIO_THRESHOLD"] = -0.2
    with pytest.raises(config.ConfigurationError):
        config.validate_final_strategy_config(cfg)


def test_validate_final_strategy_config_param_decimals_mapping():
    cfg = _base_final_cfg()
    cfg["PARAM_VALUE_DECIMALS"] = ["bad"]  # type: ignore[assignment]
    with pytest.raises(config.ConfigurationError):
        config.validate_final_strategy_config(cfg)


def test_validate_final_strategy_config_param_decimals_negative():
    cfg = _base_final_cfg()
    cfg["PARAM_VALUE_DECIMALS"] = {"default": -1}
    with pytest.raises(config.ConfigurationError):
        config.validate_final_strategy_config(cfg)
