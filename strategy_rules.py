"""Strategy rules for the GA trading framework."""

STRATEGY_RULES = {
    "entry_rules": {
        # Optional keys:
        #   combination_logic (str): "AND" | "OR" | "VOTE" (default "AND")
        #   vote_threshold (int | None): min signals for "VOTE"; ``None`` uses
        #       ``ceil(N/2)`` and values outside ``1..N`` raise ``ValueError``
        #   nan_policy (str): "FALSE" (fill), "PROPAGATE", or "FORWARD_FILL"
        #   strict_column (bool): when False, missing columns/bands fall back
        #       to the first available column with a warning (default True).
        #       Individual conditions may override via
        #       condition['strict_column'].
        # Multi-output indicators have these default selections when
        # "column"/"band" is omitted:
        #   - MACD → histogram
        #   - BBands/Keltner/Donchian → middle band
        #   - ADX/DMI → ADX line
        #   - Stoch → %K
        #   - Ichimoku → baseline (IKS_*)
        #   - Pivot Points → P
        #   - TRIX (with signal) → TRIX line
        "combination_logic": "VOTE",
        "vote_threshold": {
            "gene": "vote_threshold",
            "low": 2,
            "high": 5,
            "step": 1,
        },
        "conditions": [
            {
                "is_active": True,  # This rule is ON
                "rule_name": "Long_Term_Trend_Filter",
                "indicator": "ema",
                "params": {
                    "period": {
                        "gene": "ema_period",
                        "low": 30,
                        "high": 200,
                        "step": 5,
                    }
                },
                "condition": {"type": "price_is_above_indicator"},
            },
            {
                "is_active": True,  # This rule is ON
                "rule_name": "RSI_Momentum_Filter",
                "indicator": "rsi",
                "params": {
                    "period": {
                        "gene": "rsi_period",
                        "step": 1,
                    }
                },
                "condition": {
                    "type": "indicator_is_above_value",
                    "value": {
                        "gene": "rsi_threshold",
                        "low": 45,
                        "high": 70,
                        "step": 1,
                    },
                },
            },
            {
                "is_active": True,
                "rule_name": "MACD_Momentum_Cross",
                "indicator": "macd",
                "params": {
                    "fast": {"gene": "macd_fast", "low": 4, "high": 20, "step": 1},
                    "slow": {"gene": "macd_slow", "low": 15, "high": 35, "step": 1},
                    "signal": {"gene": "macd_signal", "low": 4, "high": 16, "step": 1},
                },
                "condition": {"type": "indicator_is_above_value", "value": 0},
            },
            {
                "is_active": False,
                "rule_name": "Bollinger_Band_Breakout",
                "indicator": "bbands",
                "params": {
                    "period": {
                        "gene": "bband_period",
                        "low": 10,
                        "high": 35,
                        "step": 1,
                    },
                    "std_dev": {
                        "gene": "bband_std",
                        "low": 0.5,
                        "high": 5,
                        "step": 0.25,
                    },
                },
                "condition": {
                    "type": "price_crosses_above_indicator",
                    "band": "upper",
                },
            },
            {
                "is_active": False,
                "rule_name": "ATR_Volatility_Filter",
                "indicator": "atr",
                "params": {
                    "period": {
                        "gene": "atr_period",
                        "low": 5,
                        "high": 35,
                        "step": 1,
                    }
                },
                "condition": {
                    "type": "indicator_is_above_value",
                    "value": {
                        "gene": "atr_threshold",
                        "low": 0.5,
                        "high": 5.0,
                        "step": 0.5,
                    },
                },
            },
            {
                "is_active": False,
                "rule_name": "SMA_Trend_Filter",
                "indicator": "sma",
                "params": {
                    "period": {
                        "gene": "sma_period",
                        "low": 5,
                        "high": 60,
                        "step": 1,
                    }
                },
                "condition": {"type": "price_is_above_indicator"},
            },
            {
                "is_active": False,
                "rule_name": "WMA_Trend_Filter",
                "indicator": "wma",
                "params": {
                    "period": {
                        "gene": "wma_period",
                        "low": 5,
                        "high": 60,
                        "step": 1,
                    }
                },
                "condition": {"type": "price_is_above_indicator"},
            },
            {
                "is_active": False,
                "rule_name": "HMA_Trend_Filter",
                "indicator": "hma",
                "params": {
                    "period": {
                        "gene": "hma_period",
                        "low": 5,
                        "high": 60,
                        "step": 1,
                    }
                },
                "condition": {"type": "price_is_above_indicator"},
            },
            {
                "is_active": False,
                "rule_name": "Stoch_Momentum_Filter",
                "indicator": "stoch",
                "params": {
                    "k": {
                        "gene": "stoch_k",
                        "low": 5,
                        "high": 20,
                        "step": 1,
                    },
                    "d": {
                        "gene": "stoch_d",
                        "low": 3,
                        "high": 20,
                        "step": 1,
                    },
                    "smooth_k": {
                        "gene": "stoch_smooth_k",
                        "low": 1,
                        "high": 5,
                        "step": 1,
                    },
                },
                "condition": {
                    "type": "indicator_is_above_value",
                    "value": {
                        "gene": "stoch_threshold",
                        "low": 20,
                        "high": 80,
                        "step": 5,
                    },
                },
            },
            {
                "is_active": True,
                "rule_name": "CCI_Momentum_Filter",
                "indicator": "cci",
                "params": {
                    "period": {
                        "gene": "cci_period",
                        "low": 10,
                        "high": 40,
                        "step": 1,
                    }
                },
                "condition": {
                    "type": "indicator_is_above_value",
                    "value": {
                        "gene": "cci_threshold",
                        "low": 50,
                        "high": 200,
                        "step": 10,
                    },
                },
            },
            {
                "is_active": False,
                "rule_name": "WilliamsR_Momentum_Filter",
                "indicator": "williams_r",
                "params": {
                    "period": {
                        "gene": "williams_period",
                        "low": 10,
                        "high": 40,
                        "step": 1,
                    }
                },
                "condition": {
                    "type": "indicator_is_above_value",
                    "value": {
                        "gene": "williams_threshold",
                        "low": -80,
                        "high": -20,
                        "step": 5,
                    },
                },
            },
            {
                "is_active": False,
                "rule_name": "TSI_Momentum_Filter",
                "indicator": "tsi",
                "params": {
                    "long": {
                        "gene": "tsi_long",
                        "low": 25,
                        "high": 60,
                        "step": 1,
                    },
                    "short": {
                        "gene": "tsi_short",
                        "low": 5,
                        "high": 24,
                        "step": 1,
                    },
                    "signal": {
                        "gene": "tsi_signal",
                        "low": 5,
                        "high": 25,
                        "step": 1,
                    },
                },
                "condition": {
                    "type": "indicator_is_above_value",
                    "value": {
                        "gene": "tsi_threshold",
                        "low": -50,
                        "high": 50,
                        "step": 5,
                    },
                },
            },
            {
                "is_active": False,
                "rule_name": "Ultimate_Oscillator_Filter",
                "indicator": "ultimate_oscillator",
                "params": {
                    "short": {
                        "gene": "uo_short",
                        "low": 7,
                        "high": 14,
                        "step": 1,
                    },
                    "medium": {
                        "gene": "uo_medium",
                        "low": 15,
                        "high": 28,
                        "step": 1,
                    },
                    "long": {
                        "gene": "uo_long",
                        "low": 29,
                        "high": 60,
                        "step": 1,
                    },
                },
                "condition": {
                    "type": "indicator_is_above_value",
                    "value": {
                        "gene": "uo_threshold",
                        "low": 40,
                        "high": 60,
                        "step": 1,
                    },
                },
            },
            {
                "is_active": True,
                "rule_name": "ADX_Trend_Strength",
                "indicator": "adx",
                "params": {
                    "period": {
                        "gene": "adx_period",
                        "low": 5,
                        "high": 30,
                        "step": 1,
                    }
                },
                "condition": {
                    "type": "indicator_is_above_value",
                    "value": {
                        "gene": "adx_threshold",
                        "low": 20,
                        "high": 50,
                        "step": 1,
                    },
                },
            },
            {
                "is_active": False,
                "rule_name": "PSAR_Trend_Follow",
                "indicator": "psar",
                "params": {
                    "acceleration": {
                        "gene": "psar_acceleration",
                        "low": 0.01,
                        "high": 0.2,
                        "step": 0.01,
                    },
                    "maximum": {
                        "gene": "psar_maximum",
                        "low": 0.1,
                        "high": 0.5,
                        "step": 0.05,
                    },
                },
                "condition": {"type": "price_crosses_above_indicator"},
            },
            {
                "is_active": False,
                "rule_name": "Keltner_Channel_Breakout",
                "indicator": "keltner",
                "params": {
                    "period": {
                        "gene": "keltner_period",
                        "low": 10,
                        "high": 40,
                        "step": 1,
                    },
                    "multiplier": {
                        "gene": "keltner_multiplier",
                        "low": 1.0,
                        "high": 3.0,
                        "step": 0.1,
                    },
                },
                "condition": {
                    "type": "price_crosses_above_indicator",
                    "band": "upper",
                },
            },
            {
                "is_active": False,
                "rule_name": "Donchian_Channel_Breakout",
                "indicator": "donchian",
                "params": {
                    "period": {
                        "gene": "donchian_period",
                        "low": 10,
                        "high": 60,
                        "step": 1,
                    }
                },
                "condition": {
                    "type": "price_crosses_above_indicator",
                    "band": "upper",
                },
            },
            {
                "is_active": False,
                "rule_name": "Stdev_Channel_Filter",
                "indicator": "stdev_channel",
                "params": {
                    "period": {
                        "gene": "stdev_period",
                        "low": 10,
                        "high": 40,
                        "step": 1,
                    }
                },
                "condition": {
                    "type": "indicator_is_above_value",
                    "value": {
                        "gene": "stdev_threshold",
                        "low": 0.5,
                        "high": 5.0,
                        "step": 0.5,
                    },
                },
            },
            {
                "is_active": False,
                "rule_name": "CMO_Momentum_Filter",
                "indicator": "cmo",
                "params": {
                    "period": {
                        "gene": "cmo_period",
                        "low": 5,
                        "high": 40,
                        "step": 1,
                    }
                },
                "condition": {
                    "type": "indicator_is_above_value",
                    "value": {
                        "gene": "cmo_threshold",
                        "low": -50,
                        "high": 50,
                        "step": 5,
                    },
                },
            },
            {
                "is_active": False,
                "rule_name": "OBV_Trend_Filter",
                "indicator": "obv",
                "condition": {
                    "type": "indicator_is_above_value",
                    "value": {
                        "gene": "obv_threshold",
                        "low": -100000,
                        "high": 100000,
                        "step": 10000,
                    },
                },
            },
            {
                "is_active": False,
                "rule_name": "MFI_Momentum_Filter",
                "indicator": "mfi",
                "params": {
                    "period": {
                        "gene": "mfi_period",
                        "low": 5,
                        "high": 40,
                        "step": 1,
                    }
                },
                "condition": {
                    "type": "indicator_is_above_value",
                    "value": {
                        "gene": "mfi_threshold",
                        "low": 20,
                        "high": 80,
                        "step": 5,
                    },
                },
            },
            {
                "is_active": False,
                "rule_name": "ADL_Accumulation_Filter",
                "indicator": "adl",
                "condition": {
                    "type": "indicator_is_above_value",
                    "value": {
                        "gene": "adl_threshold",
                        "low": -100000,
                        "high": 100000,
                        "step": 10000,
                    },
                },
            },
            {
                "is_active": False,
                "rule_name": "CMF_Momentum_Filter",
                "indicator": "cmf",
                "params": {
                    "period": {
                        "gene": "cmf_period",
                        "low": 5,
                        "high": 40,
                        "step": 1,
                    }
                },
                "condition": {
                    "type": "indicator_is_above_value",
                    "value": {
                        "gene": "cmf_threshold",
                        "low": -0.5,
                        "high": 0.5,
                        "step": 0.05,
                    },
                },
            },
            {
                "is_active": False,
                "rule_name": "MA_Envelope_Breakout",
                "indicator": "ma_envelope",
                "params": {
                    "period": {
                        "gene": "mae_period",
                        "low": 10,
                        "high": 60,
                        "step": 1,
                    },
                    "percent": {
                        "gene": "mae_percent",
                        "low": 0.5,
                        "high": 5.0,
                        "step": 0.5,
                    },
                },
                "condition": {
                    "type": "price_crosses_above_indicator",
                    "band": "upper",
                },
            },
            {
                "is_active": True,
                "rule_name": "Ichimoku_Trend_Filter",
                "indicator": "ichimoku",
                "params": {
                    "tenkan": {
                        "gene": "ichimoku_tenkan",
                        "low": 7,
                        "high": 12,
                        "step": 1,
                    },
                    "kijun": {
                        "gene": "ichimoku_kijun",
                        "low": 20,
                        "high": 30,
                        "step": 1,
                    },
                    "senkou": {
                        "gene": "ichimoku_senkou",
                        "low": 40,
                        "high": 60,
                        "step": 5,
                    },
                },
                "condition": {"type": "price_is_above_indicator"},
            },
            {
                "is_active": False,
                "rule_name": "Pivot_Point_Filter",
                "indicator": "pivot_points",
                "condition": {"type": "price_is_above_indicator"},
            },
            {
                "is_active": False,
                "rule_name": "TRIX_Momentum_Filter",
                "indicator": "trix",
                "params": {
                    "period": {
                        "gene": "trix_period",
                        "low": 5,
                        "high": 50,
                        "step": 1,
                    },
                    "signal": {
                        "gene": "trix_signal",
                        "low": 2,
                        "high": 20,
                        "step": 1,
                    },
                },
                "condition": {
                    "type": "indicator_is_above_value",
                    "value": {
                        "gene": "trix_threshold",
                        "low": -0.5,
                        "high": 0.5,
                        "step": 0.05,
                    },
                },
            },
            {
                "is_active": False,
                "rule_name": "ROC_Momentum_Filter",
                "indicator": "roc",
                "params": {
                    "period": {
                        "gene": "roc_period",
                        "low": 5,
                        "high": 30,
                        "step": 1,
                    }
                },
                "condition": {
                    "type": "indicator_is_above_value",
                    "value": {
                        "gene": "roc_threshold",
                        "low": -5.0,
                        "high": 5.0,
                        "step": 0.5,
                    },
                },
            },
        ],
    },
    "exit_rules": {
        "stop_loss": {
            "is_active": True,
            "type": "percentage",
            "params": {
                "value": {
                    "gene": "stop_loss_pct",
                    "low": 0.01,
                    "high": 0.20,
                    "step": 0.005,
                }
            },
        },
        "trade_management": {
            "num_tp_levels": {
                "gene": "num_tp_levels",
                "low": 1,
                "high": 4,
                "step": 1,
            },
            "tp_pct_1": {
                "gene": "tp_pct_1",
                "low": 0.005,
                "high": 0.80,
                "step": 0.005,
            },
            "tp_pct_2": {
                "gene": "tp_pct_2",
                "low": 0.010,
                "high": 0.80,
                "step": 0.005,
            },
            "tp_pct_3": {
                "gene": "tp_pct_3",
                "low": 0.015,
                "high": 0.80,
                "step": 0.005,
            },
            "tp_pct_4": {
                "gene": "tp_pct_4",
                "low": 0.020,
                "high": 0.80,
                "step": 0.005,
            },
            "tp_trailing_enabled": {
                "gene": "tp_trailing_enabled",
                "options": [0, 1],
            },
            "tp_trailing_pct": {
                "gene": "tp_trailing_pct",
                "low": 0.002,
                "high": 0.10,
                "step": 0.001,
            },
            "sl_timeout_enabled": {
                "gene": "sl_timeout_enabled",
                "options": [0, 1],
            },
            "sl_timeout_bars": {
                "gene": "sl_timeout_bars",
                "low": 1,
                "high": 12,
                "step": 1,
            },
            "sl_break_even_mode": {
                "gene": "sl_break_even_mode",
                "options": ["none", "breakeven", "follow_tp"],
            },
            "sl_trailing_enabled": {
                "gene": "sl_trailing_enabled",
                "options": [0, 1],
            },
            "sl_trailing_pct": {
                "gene": "sl_trailing_pct",
                "low": 0.005,
                "high": 0.20,
                "step": 0.001,
            },
        },
    },
}
