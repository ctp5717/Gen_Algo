# Agents Guide

> **Purpose**: Equip coding agents (e.g., ChatGPT Codex) to safely modify, extend, and operate this repo with minimal back-and-forth, deterministic behavior, and green CI.

---

## Contents

- [1) Project at a glance](#1-project-at-a-glance)
- [2) Repo map (authoritative)](#2-repo-map-authoritative)
- [3) Environment & setup](#3-environment--setup)
- [Using the real `vectorbt` (important)](#using-the-real-vectorbt-important)
- [4) How agents should work here (rules of the road)](#4-how-agents-should-work-here-rules-of-the-road)
- [5) Common agent tasks (recipes)](#5-common-agent-tasks-recipes)
- [Exit rules schema (quick reference)](#exit-rules-schema-quick-reference)
- [Universe & multi-asset quickstart](#universe--multi-asset-quickstart)
- [6) Running & validating](#6-running--validating)
- [7) Invariants & pitfalls (tests enforce these)](#7-invariants--pitfalls-tests-enforce-these)
- [8) Secrets, data, reproducibility](#8-secrets-data-reproducibility)
- [9) Style & contrib](#9-style--contrib)
- [10) Good prompts for agents](#10-good-prompts-for-agents)
- [Do not](#do-not)
- [Hardware & runtime](#hardware--runtime)
- [Foundational changes & overrides](#foundational-changes--overrides)

## Documentation quick reference

- `README.md` – high-level overview, workflow links, and run instructions.
- `docs/getting_started.md` – environment setup and first optimisation walkthrough.
- `docs/configuration.md` – detailed reference for `config.py` toggles and environment flags.
- `docs/strategy_authoring.md` – schema, defaults, and tips for editing `STRATEGY_RULES`.
- `docs/architecture.md` – module responsibilities and metadata contract.

Keep these documents in sync with behaviour changes—tests and reviews assume they describe the current pipeline.

## 1) Project at a glance

- **What this is**: A modular, GA-driven trading research framework in Python that:
  - Loads historical OHLCV (Binance US or yfinance) with **caching**.
  - Builds entry signals from **declarative rules** (`strategy_rules.STRATEGY_RULES`) using functions in `indicator_library.py`.
  - Backtests via **vectorbt** and scores with a **composite fitness** (Sortino, Profit Factor w/ winsorization, Max DD, etc.).
  - Supports **multi-asset fitness** with dispersion/coverage & trade-floor policies.
  - Includes **auto-tuning**, **walk-forward**, and a **champion analysis** step that writes reproducibility metadata.

- **Primary entry points**
  - `python main.py` — standard GA optimisation, champion analysis, and optional walk-forward/recommendation/final strategy synthesis.
  - `tuner.find_best_hyperparameters(...)` — express GA tuner invoked automatically when `config.AUTO_TUNE_ENABLED` is true; call it manually from a Python session for bespoke sweeps.
  - `walk_forward.run_walk_forward_validation(...)` — roll training/testing windows programmatically when you need to reuse cached champions without rerunning the GA.

---

## 2) Repo map (authoritative)

- `config.py` – **Control panel**:
  - Seed: `SEED` (overridable via `GA_SEED` env).
  - Data source: `DATA_SOURCE` = `"binance"` or `"yfinance"`, creds in env (`BINANCE_TLD`, `BINANCE_API_KEY`, `BINANCE_API_SECRET`).
  - Timeframe: `TIMEFRAME` (e.g. `"15m"`, `"1h"`, `"1d"`); use `to_pandas_freq()` for pandas-safe frequency.
  - Periods: `TRAINING_PERIOD`, `VALIDATION_PERIOD` (auto-computed rolling windows).
  - Risk: `MAX_HOLD_DAYS` → converted to bars (intraday aware).
  - GA knobs: population, generations, mutation, etc.; optional **auto-tuner** settings.
  - Strategy: `STRATEGY_RULES` (in `strategy_rules.py`) with `is_active` flags and per-param **genes** (`low`, `high`, `step`, `name`).
    - `RSI_Momentum_Filter` draws its `rsi_period` bounds from `config.RSI_PERIOD_BOUNDS`
      (default 7–21) to curb instability from overly wide windows.
  - Fitness: `FITNESS_WEIGHTS` for composite score.
  - Multi-asset: `MULTI_ASSET` (lambda dispersion, trade-floor, zero-trade policy, coverage penalty, etc.).
  - Stability regularizer knobs: `ENABLE_STABILITY_REG`, `STABILITY_ALPHA`, `STABILITY_GENES`
    (default [`rsi_period`]) apply an optional penalty for high parameter CoV.

- `data_loader.py` – **Cache + fetch**:
  - Cache dir: `data_cache/`; Parquet caches with restored `DatetimeIndex` (legacy CSV still readable).
  - `_normalize_ticker()` converts e.g. `BTC-USD` → `BTCUSDT` for Binance.
  - `get_data(ticker, start, end, interval)` returns `(DataFrame, "cache"|"API")`.

- `indicator_library.py` – **Indicator toolbox**:
  - Uses `pandas_ta` (imported as `ta`); standard signatures like `def ema(df, period: int) -> pd.Series`.
  - Return Series/DataFrames with index aligned to OHLCV.

- `strategy_engine.py` – **Rule interpreter**:
  - Consumes `STRATEGY_RULES`, calls indicators by name, applies `condition`:
    - Examples: `price_is_above_indicator`, `indicator_is_above_value`,
      `indicator_crosses_above_value`, and symmetric variants.
    - Combines conditions by case-insensitive `combination_logic` (`AND`/`OR`/`VOTE`, default `AND`).
      `VOTE` uses a majority threshold when `vote_threshold` is `None`; values
      outside `1..N` raise. With a single condition all modes behave identically.
      NaNs in signals default to `False`; configure `nan_policy` to `PROPAGATE` or
      `FORWARD_FILL` in `entry_rules` to propagate them instead. Forward-fill
      respects `ffill_lookback` (global `config.NAN_FFILL_LOOKBACK`; `0`
      means unlimited).
  - Handles **multi-output indicators** (pick the right column).

- `gene_parser.py` – Finds all **active** gene definitions, including top-level
  options like `combination_logic` or `vote_threshold`, and returns
  `gene_space`, `gene_map`, `gene_types`.

- `fitness.py` – **Fitness evaluators**:
  - Single-asset `FitnessEvaluator` uses **vectorbt** to backtest, exit handling with **`.shift()`** for time-based rules, stats (Sortino, Profit Factor with cap/winsorization, Max DD, total return, trades).
  - `MultiAssetFitnessEvaluator` aggregates per-asset stats with dispersion penalties, trade floor (via `trade_floor.scale_floor()`), zero-trade policy, coverage penalties, winsorization caps, etc.
  - `_inject_genes()` overlays GA genes into `STRATEGY_RULES`.
- `metrics_contract.py` – Canonical metric aliases, unit normalisation, and fallbacks.
  - `METRIC_ALIASES` captures label drift between vectorbt and QuantStats; `resolve_metrics` caches the discovered mapping.
  - `compute_fallbacks` synthesises missing metrics from raw returns; prefer `evaluate_metrics` to obtain `(metrics, sources, missing)`.
  - Configure alias preflight via `config.METRICS_PREFLIGHT` (`mode`: `"warn"|"fail"`, `missing_threshold`: tolerated missing aliases).

| Canonical key   | Accepted aliases (subset)                          | Unit            | Fallback formula                                      |
|-----------------|-----------------------------------------------------|-----------------|------------------------------------------------------|
| `sortino`       | `Sortino Ratio`, `sortino_ratio`, `Sortino`         | ratio           | Mean excess return ÷ downside deviation × √252       |
| `profit_factor` | `Profit Factor`, `PF`, `profit_factor`              | ratio           | Σ positive returns ÷ Σ absolute negative returns     |
| `max_drawdown`  | `Max Drawdown [%]`, `Max Drawdown`, `max_drawdown`  | percent (0–100) | Max peak-to-trough drop of cumulative returns        |
| `total_return`  | `Total Return [%]`, `Return [%]`, `total_return`    | percent (0–100) | `(1 + returns).prod() - 1`                           |

Metric mappings are logged once per run (e.g. `sortino→sortino_ratio`) and the first asset records a `metric_sources` map in `last_details`. Assets with trades but missing metrics emit `evaluation_reason="metrics_missing"`.

- `trade_floor.py` – Scales a required minimum trades **over elapsed years**.

- `analysis.py` – **Champion analysis + artifacts**:
  - Plots (matplotlib), writes **`run_metadata.json`** with library versions, cache file hashes, wall time, etc.

- `tuner.py` – Coarse GA + **lambda sweep** with optional re-scoring at fixed seeds.

- `walk_forward.py` – Rolling training/testing window generation, writes metadata.

- `tests/` – Invariants and integration tests:
  - Determinism (seed), hold-period conversion, pandas freq strings,
  - Data loader cache behavior,
  - Profit factor edge cases (winsorization),
  - Multi-asset trade floor & penalties,
  - Rule parsing and engine condition logic.

- CI / Quality:
  - `.github/workflows/` (lint, tests, security),
  - `.pre-commit-config.yaml` (black, isort, flake8, mypy relaxed),
  - `.flake8` per-file ignores,
  - `requirements.txt`, `requirements-dev.txt`.
  - Tests run on a matrix covering `vectorbt==0.28.1` with QuantStats and floating `vectorbt>=0.28.1` environments (with/without QuantStats) to surface alias drift.

---

## 3) Environment & setup

```bash
python -V   # Use Python 3.12 or 3.13
python -m pip install -r requirements.txt -r requirements-dev.txt

# Optional heavy deps for real backtests (tests stub them if missing)
# (already in requirements.txt): vectorbt, pandas-ta

# Secrets (never commit):
export BINANCE_TLD=us
export BINANCE_API_KEY=...
export BINANCE_API_SECRET=...

# Reproducibility:
export GA_SEED=42  # overrides config.SEED
```

The CI matrix exercises Linux runners on Python 3.12 and 3.13; keep code and dependencies compatible with both.

### Using the real `vectorbt` (important)
This repo ships a lightweight `vbt_stub.py` for tests. The real `vectorbt` package is used by default; the stub only takes effect when explicitly injected.

**To use the stub in tests:**
1. Run tests with `USE_VBT_STUB=1` to inject the stub automatically via `tests/conftest.py`.
2. Or manually: `import vbt_stub as vbt; sys.modules["vectorbt"] = vbt`.

`main.main()` and `walk_forward.run_walk_forward_validation()` call `deps.ensure_real_vectorbt()` to fail fast if a local stub shadows the real package. If you invoke `walk_forward` directly, ensure this guard runs before importing `vectorbt`.

Quick check for real runs:

```bash
python -c "import vectorbt as vbt; import pandas as pd; assert hasattr(pd.Series, 'vbt'); print(vbt.__version__)"  # should not print 0.0.0
```

---

## 4) How agents should work here (rules of the road)

### Determinism & data
- Always respect `SEED`/`GA_SEED`.
- Don't change cache semantics or index alignment. `data_loader` must write/read CSV with a correct `DatetimeIndex`.
- For timeframe→pandas frequency, use `config.to_pandas_freq()` (tests forbid deprecated strings).

### Strategy & indicators
- **Adding an indicator**
  1. Implement in `indicator_library.py` with a typed signature returning a `pd.Series`/`pd.DataFrame` aligned to OHLCV index.
  2. Register it in `strategy_engine.py` by adding to `INDICATOR_MAPPING`, e.g.:
  ```python
  INDICATOR_MAPPING["vwap"] = ind_lib.calculate_vwap
  ```
  3. Name must match what `STRATEGY_RULES` references (e.g., `"indicator": "ema"`).
  4. Document expected column key if multi-output.
- **Adding a rule/condition type**
  - Extend comparison/cross helpers in `strategy_engine.py`.
  - Return a boolean `pd.Series` aligned to price index.
  - Common patterns: `indicator_is_above_value`, `price_is_above_indicator`,
    `indicator_crosses_above_value`, `indicator_is_below_value`, etc.
  - Add tests in `tests/test_strategy_engine.py`.

### Fitness & multi-asset
- If you modify fitness composition, update:
  - `config.FITNESS_WEIGHTS`,
  - stats computation in `fitness.py` (keep winsorization and NaN fallback),
  - multi-asset aggregation/penalties in `MULTI_ASSET`.
- Respect tests around:
  - Profit factor caps for near-zero losses,
  - Trade floor scaling across time,
  - Coverage penalties and zero-trade policy.

### Exits & holds
- Time-based exits must use `Series.shift()` (explicitly tested).
- `MAX_HOLD_DAYS` conversion to bars must account for intraday `TIMEFRAME`.

---

## 5) Common agent tasks (recipes)

### A) Add a new indicator (e.g., VWAP)
1. Implement `def calculate_vwap(df: pd.DataFrame, period: int) -> pd.Series` in `indicator_library.py`.
2. Register it in `strategy_engine.py` by adding to `INDICATOR_MAPPING`.
3. Activate a new rule in `strategy_rules.STRATEGY_RULES["entry_rules"]["conditions"]`:

```json
{
  "is_active": true,
  "rule_name": "VWAP_Filter",
  "indicator": "vwap",
  "params": { "period": { "gene": "vwap_period", "low": 5, "high": 50, "step": 1 } },
  "condition": { "type": "price_is_above_indicator" }
}
```

4. Run:

```bash
pytest -q
python main.py
```

### B) Introduce a new condition type (e.g., indicator crosses above another indicator)
1. In `strategy_engine.py`, add a handler that consumes two indicator series and returns `series_a.vbt.crossed_above(series_b)` (or equivalent).
2. Add a test case in `tests/test_strategy_engine.py`.

### C) Extend the composite fitness (e.g., add Calmar)
1. Compute Calmar ratio where stats are aggregated in `fitness.py`.
2. Add a weight to `config.FITNESS_WEIGHTS` and include it in the weighted score.
3. Update invariants tests if needed.

### D) Adjust multi-asset dispersion/coverage behavior
1. Tune `config.MULTI_ASSET` (`lambda_dispersion`, `trade_floor_policy`, `coverage_penalty`, etc.).
2. If behavior changes, add a focused test in `tests/test_multi_asset_fitness.py`.
3. Optionally sweep lambda via `tuner.py`.

---

## Exit rules schema (quick reference)
`strategy_rules.STRATEGY_RULES["exit_rules"]` supports:

```json
{
  "stop_loss": {
    "is_active": true,
    "type": "percentage",
    "params": { "value": { "gene": "stop_loss_pct", "low": 0.01, "high": 0.10, "step": 0.005 } }
  },
  "trailing_stop": {
    "is_active": false,
    "type": "percentage",
    "params": { "value": { "gene": "tsl_pct", "low": 0.01, "high": 0.10, "step": 0.005 } }
  },
  "take_profit": {
    "is_active": true,
    "type": "percentage",
    "params": { "value": { "gene": "take_profit_pct", "low": 0.01, "high": 0.20, "step": 0.01 } }
  }
}
```

Notes:
- Time exits must use `.shift()` as enforced by tests.
- `MAX_HOLD_DAYS` is converted to bars based on `TIMEFRAME`.

---

## Universe & multi-asset quickstart
- **Choose asset:** set `SELECTED_ASSET_NAME` in `config.py` (maps via `CRYPTO_UNIVERSE`).
- **Binance normalization:** `data_loader._normalize_ticker("BTC-USD") -> "BTCUSDT"`.
- **Multi-asset:** confirm `MULTI_ASSET["enabled"] = true` and tune:
  - `lambda_dispersion`, `min_total_trades_per_year`, `coverage_penalty`,
  - `zero_trade_policy` ("ignore" or "penalize"), `per_asset_min_trades` (per fold, not time-scaled).
- **Coverage threshold:** `COVERAGE_THRESHOLD` (default `0.8`) controls asset retention when aligning data.

**BBands tips:** Select specific bands either by adding a `band` hint
(`"upper"`, `"middle"`/`"mid"`/`"basis"`, `"lower"`) to the condition or by using the
`*_band` condition types such as `"price_is_above_upper_band"` or
`"price_crosses_below_lower_band"`. The engine maps these to the `BBU` / `BBM`
/ `BBL` columns.

**MACD tips:** The engine expects the histogram (`MACDh_*` or `MACD_Hist*`) by
default; override with `condition.column` to use another component.

---

## 6) Running & validating

**Unit/Integration tests**

```bash
pytest -q             # quick
pytest -q -n auto     # parallel (CI default)
```

**Lint & hooks**

```bash
pre-commit install
pre-commit run -a
```

**End-to-end**

```bash
python main.py
# then inspect run_metadata.json and analysis artifacts
```

**Walk-forward & tuning (examples)**
```bash
# Rolling windows & metadata
python walk_forward.py
# Coarse hyperparameter sweep (GA & lambda)
python tuner.py
```

The tuning module also supports a deeper GA candidate for exhaustive searches:

```python
{"sol_per_pop": 250, "num_parents_mating": 60, "mutation_num_genes": 4}
```
Add it to `config.HYPERPARAMETER_SEARCH_SPACE` when exploring larger populations.

CI expectations:
- Linux matrix (Python 3.12 & 3.13), `pytest -q -n auto` with coverage.
- Lint via pre-commit; basic security scan via CodeQL & Bandit.
- No secrets in tree.

---

## 7) Invariants & pitfalls (tests enforce these)
- Use `config.to_pandas_freq()`; avoid deprecated pandas offsets.
- Cached CSV must restore `DatetimeIndex` correctly.
- `.shift()` must be used for time-based exits.
- Profit factor must be winsorized (cap configured).
- Trade floor scales with elapsed years; zero-trade handling must match `MULTI_ASSET.zero_trade_policy`.
- `gene_parser.parse_genes_from_config()` only considers rules with `is_active=True`.
- Indicator library must import `pandas_ta as ta` and ensure `numpy.NaN` exists.

---

## 8) Secrets, data, reproducibility
- Never commit keys; pass via env (`BINANCE_*`).
- Data cache: `data_cache/` (hashed in `run_metadata.json` for reproducibility).
- Metadata: `run_metadata.json` captures start/end times, wall time, library versions, cache file hashes, commit hash (if available).

---

## 9) Style & contrib
- Formatting: black, isort; lint: flake8; typing: mypy (relaxed/error-code disables).
- Keep functions small with clear IO contracts; prefer pure functions where possible.
- Add/modify tests for any behavior change.
- Update `README.md` and files in `docs/` when behaviour or interfaces change.

---

## 10) Good prompts for agents
- "Add indicator X to `indicator_library.py` with signature (...)->`pd.Series`, wire it into `STRATEGY_RULES`, and create a unit test that validates its integration using a tiny DataFrame."
- "Implement condition type Y in `strategy_engine.py` and cover it in `test_strategy_engine.py`."
- "Extend composite fitness to include Z, update `FITNESS_WEIGHTS`, and ensure determinism with `GA_SEED`."

---

## Do not
- Hardcode secrets.
- Remove `.shift()` on time-based exits.
- Change the Parquet cache format or drop legacy CSV support without a spec.
- Bypass seed determinism or replace `config.to_pandas_freq()` with ad-hoc strings.

---

## Hardware & runtime
- Typical GA runs with vectorbt and matplotlib can require several GB of RAM and benefit from multi-core CPUs.

---

## Foundational changes & overrides
This guide is **descriptive, not a contract**. If a product/feature spec conflicts with this document, **the spec wins**. Agents may refactor or replace modules and public APIs provided they:

1) **Preserve causality & reproducibility**
   - No look-ahead. If changing exit timing, provide a causality-safe mechanism (e.g., `.shift()` or equivalent) and document it.
   - Determinism: keep seed behavior (`SEED`/`GA_SEED`) or document why it changes.

2) **Keep CI green with updated tests**
   - Update/add tests to match new behavior; remove obsolete ones.
   - Run `pytest -q -n auto` and pre-commit hooks; resolve lint failures.

3) **Provide migration & versioning**
   - For schema/cache changes, add a small migration script and bump a `SCHEMA_VERSION` (or add one) in `config.py`.
   - Offer a temporary feature flag (e.g., `EXECUTION_MODE="v2"` or `NEW_ENGINE=true`) during transition.

4) **Document the change**
   - Update `AGENTS.md` and `CHANGELOG.md` with rationale, new defaults, and any **breaking changes**.
