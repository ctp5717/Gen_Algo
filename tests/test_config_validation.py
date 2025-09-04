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
