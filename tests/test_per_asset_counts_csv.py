import analysis
from strategy_rules import STRATEGY_RULES


def test_per_asset_counts_columns_defaults_and_sorting():
    per_asset_signal_counts = {
        # Deliberately unsorted to verify sorting
        "BBB": {"MACD_Momentum_Cross": 3},
        "AAA": {"Long_Term_Trend_Filter": 2, "RSI_Momentum_Filter": 1},
    }
    df = analysis._build_per_asset_counts(per_asset_signal_counts, STRATEGY_RULES)
    slugs, _ = analysis._canonical_rule_slugs(STRATEGY_RULES)
    expected_cols = [
        "asset",
        "combination_logic",
        "vote_threshold",
        "nan_policy",
    ] + [f"count_{s}" for s in slugs]
    assert list(df.columns) == expected_cols
    # Rows should be sorted by asset name
    assert df["asset"].tolist() == sorted(per_asset_signal_counts)

    row_aaa = df[df["asset"] == "AAA"].iloc[0]
    assert row_aaa["count_long_term_trend_filter"] == 2
    assert row_aaa["count_macd_momentum_cross"] == 0

    row_bbb = df[df["asset"] == "BBB"].iloc[0]
    assert row_bbb["count_rsi_momentum_filter"] == 0

    # Inactive rule should be present with zero counts for all assets
    assert "count_bollinger_band_breakout" in df.columns
    assert row_aaa["count_bollinger_band_breakout"] == 0
    assert row_bbb["count_bollinger_band_breakout"] == 0


def test_slug_collision_unique():
    rules = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "ema",
                    "params": {"period": 10},
                    "condition": {"type": "price_is_above_indicator"},
                },
                {
                    "indicator": "ema",
                    "params": {"period": 20},
                    "condition": {"type": "price_is_above_indicator"},
                },
            ]
        }
    }
    slugs, _ = analysis._canonical_rule_slugs(rules)
    assert slugs == [
        "ema_price_is_above_indicator",
        "ema_price_is_above_indicator__1",
    ]
    df = analysis._build_per_asset_counts(
        {"AAA": {"ema:price_is_above_indicator": 2}}, rules
    )
    assert list(df.columns) == [
        "asset",
        "combination_logic",
        "vote_threshold",
        "nan_policy",
        "count_ema_price_is_above_indicator",
        "count_ema_price_is_above_indicator__1",
    ]
