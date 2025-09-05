import pytest

import config


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
