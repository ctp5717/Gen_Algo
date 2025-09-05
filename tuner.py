import json
import math
import os
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pygad
import vectorbt as vbt

import config
import fitness
import lambda_selector
import strategy_engine as engine
import trade_floor


def sample_macd_params(rng: np.random.Generator | None = None) -> dict:
    """Sample MACD parameters while enforcing basic constraints.

    Ensures ``fast < slow`` and ``1 <= signal < slow`` by repairing
    any invalid random draws. Returns a dictionary suitable for use
    in strategy rule parameters.
    """

    rng = rng or np.random.default_rng()

    fast = int(rng.integers(4, 21))
    slow = int(rng.integers(15, 36))
    signal = int(rng.integers(4, 17))

    slow = max(slow, fast + 1)
    signal = min(max(signal, 1), slow - 1)

    return {"fast": fast, "slow": slow, "signal": signal}


def _evaluate_on_validation(solution, gene_map, val_data):
    """Evaluate solution on preloaded validation data and return the score."""
    # Skip evaluation gracefully if optional heavy dependencies are missing.
    if not hasattr(pd.DataFrame(), "ta") or not hasattr(vbt, "Portfolio"):
        return -np.inf

    if getattr(config, "MULTI_ASSET", {}).get("enabled"):
        if not val_data:
            return -np.inf
        settings = dict(config.MULTI_ASSET)
        start = pd.to_datetime(config.VALIDATION_PERIOD["start"])
        end = pd.to_datetime(config.VALIDATION_PERIOD["end"])
        per_asset_base = settings.get("per_asset_min_trades")
        if per_asset_base:
            floor_pa, info_pa = trade_floor.scale_floor(
                per_asset_base,
                start,
                end,
                settings.get("trading_days_per_year", 252),
            )
            settings["per_asset_min_trades"] = floor_pa
            settings["per_asset_floor_info"] = info_pa
            print(
                f"Per-asset floor: base={per_asset_base} → scaled={floor_pa} "
                f"(window={info_pa['window_days']}d, base={info_pa['trading_days_per_year']}d)"
            )
        rate = settings.get("min_total_trades_per_year")
        if rate:
            floor, info = trade_floor.scale_floor(
                rate, start, end, settings.get("trading_days_per_year", 252)
            )
            settings["min_total_trades"] = floor
            print(f"Scaled min_total_trades (validation): {floor} | info={info}")
        settings["trade_floor_policy"] = "soft_penalty"
        settings["soft_penalty_mode"] = "multiplicative"
        print(
            "Tuner: using trade_floor_policy=soft_penalty (multiplicative) for validation."
        )
        evaluator = fitness.MultiAssetFitnessEvaluator(
            val_data, config.STRATEGY_RULES, gene_map, settings
        )
        return evaluator(None, solution, 0)

    if val_data is None or val_data.empty:
        return -np.inf

    rules = fitness._inject_genes_into_rules(config.STRATEGY_RULES, gene_map, solution)
    entries = engine.process_strategy_rules(val_data, rules)
    if entries.sum() < 1:
        return -np.inf

    exit_rules = rules.get("exit_rules", {})
    sl_rule = exit_rules.get("stop_loss", {})
    tsl_rule = exit_rules.get("trailing_stop", {})
    tp_rule = exit_rules.get("take_profit", {})

    sl_stop = (
        sl_rule.get("params", {}).get("value")
        if sl_rule.get("is_active", False)
        else None
    )
    sl_trail = (
        tsl_rule.get("params", {}).get("value")
        if tsl_rule.get("is_active", False)
        else None
    )
    tp_stop = (
        tp_rule.get("params", {}).get("value")
        if tp_rule.get("is_active", False)
        else None
    )

    time_exit = entries.shift(config.MAX_HOLD_PERIOD, fill_value=False)
    time_exit = time_exit.reindex(entries.index, fill_value=False)

    portfolio = vbt.Portfolio.from_signals(
        close=val_data["Close"],
        entries=entries,
        exits=time_exit,
        sl_stop=sl_stop,
        tp_stop=tp_stop,
        sl_trail=sl_trail,
        fees=config.FEES,
        freq=config.to_pandas_freq(config.TIMEFRAME),
    )
    stats = portfolio.stats()
    score = stats.get("Sortino Ratio")
    return -np.inf if np.isnan(score) else score


def _extract_metrics(evaluator, solution):
    """Return (mu, sigma, F, coverage) for the given solution."""

    score = evaluator(None, solution, 0)
    details = getattr(evaluator, "last_details", {}) or {}
    mu = float(details.get("mu", 0.0))
    sigma = float(details.get("sigma", 0.0))
    assets_included = details.get("assets_included")
    total_assets = len(getattr(evaluator, "group_data", []))
    coverage = 0.0
    if assets_included is not None and total_assets:
        coverage = assets_included / total_assets
    if not math.isfinite(mu):
        mu = 0.0
    if not math.isfinite(sigma):
        sigma = 0.0
    score = float(score)
    if not math.isfinite(score):
        score = 0.0
    if not math.isfinite(coverage):
        coverage = 0.0
    return mu, sigma, score, coverage


def find_best_hyperparameters(train_data, gene_space, gene_map, gene_types, val_data):
    """Run short GA optimisations using preloaded data."""
    print("\n--- Express Hyperparameter Tuning ---")
    np.random.seed(config.SEED)

    # Optional coarse tuning of lambda dispersion
    if getattr(config, "MULTI_ASSET", {}).get("enabled"):
        lam_grid = config.MULTI_ASSET.get("lambda_grid")
        if lam_grid:
            print("\n-- Lambda Dispersion Grid --")
            sweep_start = datetime.now(timezone.utc).isoformat()
            sweep_rows = []
            seeds = (
                config.MULTI_ASSET.get("lambda_seeds")
                or config.MULTI_ASSET.get("lambda_rescore_seeds")
                or [config.SEED, config.SEED + 1, config.SEED + 2]
            )
            shortlist_size = config.MULTI_ASSET.get("lambda_shortlist_size", 3)
            sigma_pctl = config.MULTI_ASSET.get(
                "lambda_sigma_pctl",
                config.MULTI_ASSET.get("lambda_sigma_pct", 0.75),
            )
            coverage_min = config.MULTI_ASSET.get("lambda_coverage_min")

            for lam in lam_grid:
                for seed in seeds:
                    settings = dict(config.MULTI_ASSET)
                    settings["lambda_dispersion"] = lam
                    settings["trade_floor_policy"] = "soft_penalty"
                    settings["soft_penalty_mode"] = "multiplicative"
                    evaluator = fitness.MultiAssetFitnessEvaluator(
                        train_data, config.STRATEGY_RULES, gene_map, settings
                    )
                    probe = pygad.GA(
                        num_generations=1,
                        num_parents_mating=2,
                        sol_per_pop=4,
                        num_genes=len(gene_space),
                        gene_space=gene_space,
                        gene_type=list(gene_types),
                        mutation_num_genes=1,
                        fitness_func=evaluator.__call__,
                        random_seed=seed,
                    )
                    probe.run()
                    best_solution, _, _ = probe.best_solution()
                    mu_tr, sigma_tr, F_tr, _ = _extract_metrics(
                        evaluator, best_solution
                    )

                    val_settings = dict(settings)
                    val_settings["lambda_dispersion"] = 0.0
                    val_evaluator = fitness.MultiAssetFitnessEvaluator(
                        val_data, config.STRATEGY_RULES, gene_map, val_settings
                    )
                    mu_val, sigma_val, _, coverage = _extract_metrics(
                        val_evaluator, best_solution
                    )
                    sweep_rows.append(
                        lambda_selector.LambdaSweepRow(
                            lambda_value=lam,
                            mu_val=mu_val,
                            sigma_val=sigma_val,
                            mu_tr=mu_tr,
                            sigma_tr=sigma_tr,
                            F_tr=F_tr,
                            coverage=coverage,
                            fold=0,
                            seed=seed,
                        )
                    )
                    gap = mu_tr - mu_val
                    print(
                        f"λ={lam} seed={seed} | μ_val={mu_val:.4f} | "
                        f"σ_val={sigma_val:.4f} | gap={gap:.4f} | "
                        f"coverage={coverage:.4f}"
                    )

            (
                selected_lam,
                sweep_table,
                shortlist_df,
            ) = lambda_selector.select_lambda_with_elbow(
                sweep_rows,
                shortlist_size=shortlist_size,
                sigma_pct_threshold=sigma_pctl,
                coverage_min=coverage_min,
            )
            config.MULTI_ASSET["lambda_dispersion"] = selected_lam
            print(f"Selected λ={selected_lam}")

            for _, row in shortlist_df.iterrows():
                print(
                    f"shortlist λ={row['lambda']} | μ_val={row['mu_val_mean']:.4f} | "
                    f"σ_val={row['sigma_val_mean']:.4f} | elbow={row['elbow_dist']:.4f}"
                )

            sweep_end = datetime.now(timezone.utc).isoformat()
            try:
                git_sha = (
                    subprocess.check_output(
                        [
                            "git",
                            "rev-parse",
                            "HEAD",
                        ]
                    )
                    .decode()
                    .strip()
                )
            except Exception:
                git_sha = "unknown"

            artifact = {
                "selected_lambda": selected_lam,
                "grid": list(lam_grid),
                "folds": 1,
                "seeds": list(seeds),
                "shortlist_size": shortlist_size,
                "sigma_pctl": sigma_pctl,
                "coverage_min": coverage_min,
                "started_at": sweep_start,
                "ended_at": sweep_end,
                "git_sha": git_sha,
                "rows": sweep_table.to_dict(orient="records"),
                "shortlist": shortlist_df.to_dict(orient="records"),
            }
            with open("lambda_sweep.json", "w", encoding="utf-8") as f:
                json.dump(artifact, f, indent=2)

    fitness_evaluator = fitness.get_fitness_evaluator(
        train_data, config.STRATEGY_RULES, gene_map
    )
    fitness_func = fitness_evaluator.__call__
    num_cores = os.cpu_count()

    results = []

    for idx, params in enumerate(config.HYPERPARAMETER_SEARCH_SPACE, 1):
        print(
            f"Tuning with config {idx} of {len(config.HYPERPARAMETER_SEARCH_SPACE)}: {params}"
        )
        ga = pygad.GA(
            num_generations=config.GENERATIONS_PER_TUNE,
            num_parents_mating=params["num_parents_mating"],
            sol_per_pop=params["sol_per_pop"],
            num_genes=len(gene_space),
            gene_space=gene_space,
            gene_type=list(gene_types),
            mutation_num_genes=params["mutation_num_genes"],
            fitness_func=fitness_func,
            parallel_processing=["process", num_cores],
            random_seed=config.SEED,
        )
        ga.run()
        best_solution, _, _ = ga.best_solution()
        score = _evaluate_on_validation(best_solution, gene_map, val_data)
        results.append({"params": params, "score": score})
        print(f"Validation score: {score}")

    print("\n-- Tuning Summary --")
    for r in results:
        print(f"{r['params']} => {r['score']}")

    best = max(results, key=lambda x: x["score"]) if results else {"params": None}
    print(f"Best hyperparameters found: {best['params']}")
    return best["params"]
