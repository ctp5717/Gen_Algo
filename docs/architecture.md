# Architecture

**Audience:** Developers and maintainers extending the framework.

The project is organised as a set of composable modules with clear inputs,
outputs, and metadata contracts.

```mermaid
flowchart TD
    subgraph Signals
        config[config.py] --> loader[data_loader.get_data]
        loader --> engine[strategy_engine.build_signals]
        indicators[indicator_library.py] --> engine
        contracts[indicator_contracts.py] --> preflight[preflight.check_indicator_contracts]
        engine -. columns .-> preflight
    end
    engine --> fitness[fitness.FitnessEvaluator]
    fitness --> analysis[analysis.run_champion_analysis]
    analysis --> metadata[run_metadata.merge_run_metadata]
    analysis --> walk[walk_forward.run_walk_forward_validation]
    walk --> recommendation[recommendation.generate_recommendation]
    recommendation --> final[final_strategy.generate_final_strategy]
    walk --> metadata
    recommendation --> metadata
    final --> metadata
```

### Fitness evaluation

`fitness.get_fitness_evaluator` returns either a single-asset or multi-asset
evaluator. Both wrap vectorbt and honour the composite score weights defined in
`config.FITNESS_WEIGHTS`.

```python
portfolio = vbt.Portfolio.from_signals(
    close=self.ohlc_data["Close"],
    entries=entries,
    exits=time_based_exit,
    sl_stop=sl_stop,
    tp_stop=tp_stop,
    sl_trail=sl_trail,
    fees=config.FEES,
    freq=config.to_pandas_freq(config.TIMEFRAME),
)
metrics, sources, missing = metrics_contract.evaluate_metrics(portfolio)
```

### Recommendation and final strategy

Walk-forward validation writes schema v1.0 JSON/CSV artifacts that feed the
recommendation and final-strategy stages. Both are callable utilities designed
for orchestration scripts:

```python
from pathlib import Path

from recommendation import generate_recommendation
from final_strategy import generate_final_strategy

run_dir = Path("./runs/2024-01-15")

# Produce strategy_recommendation.md and update run_metadata.json
generate_recommendation({"run_dir": run_dir})

# Synthesize final_strategy.md from recommendation + walk-forward outputs
generate_final_strategy({"run_dir": run_dir})
```

`final_strategy.generate_final_strategy` enforces gating rules from
`config.FINAL_STRATEGY`, aggregates per-fold parameters, and computes portfolio
weights via `_bounded_simplex_projection` with optional recency weighting.

### Metric contract

`metrics_contract.py` centralises statistic aliases, unit normalisation, and
fallback logic. The canonical metrics and fallbacks are:

| Canonical key   | Accepted aliases (subset)                          | Unit            | Fallback formula                                      |
|-----------------|-----------------------------------------------------|-----------------|------------------------------------------------------|
| `sortino`       | `Sortino Ratio`, `sortino_ratio`, `Sortino`         | ratio           | Mean excess return ÷ downside deviation × √252       |
| `profit_factor` | `Profit Factor`, `PF`, `profit_factor`              | ratio           | Σ positive returns ÷ Σ absolute negative returns     |
| `max_drawdown`  | `Max Drawdown [%]`, `Max Drawdown`, `max_drawdown`  | percent (0–100) | Max peak-to-trough drop of cumulative returns        |
| `total_return`  | `Total Return [%]`, `Return [%]`, `total_return`    | percent (0–100) | `(1 + returns).prod() - 1`                           |

`resolve_metrics` tries the fast-path labels, falls back to the cached alias map, and `_to_pct` harmonises fractional/percentage units. `compute_fallbacks` populates missing metrics from raw returns. Prefer `evaluate_metrics` for a single call returning the metrics, their sources (alias vs. `"computed"`), and any remaining missing keys.

Before large runs, `assert_metric_aliases` verifies that at least one alias per metric exists. Its behaviour is controlled via `config.METRICS_PREFLIGHT` (`mode`: `"warn"|"fail"`, `missing_threshold`: tolerated missing aliases). During evaluation the resolved mapping is logged once (e.g. `sortino→sortino_ratio`) and the first asset records `metric_sources` in `MultiAssetFitnessEvaluator.last_details`. When trades execute but metrics remain unavailable the evaluator surfaces `evaluation_reason="metrics_missing"` for that asset.

Continuous integration runs the test suite across a pinned environment (`vectorbt==0.28.1`, `quantstats>=0.0.62`) and floating environments with and without QuantStats to catch alias drift early.
