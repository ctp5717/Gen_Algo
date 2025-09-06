import dataclasses
import hashlib
import json
import logging
import math
import os
import subprocess
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygad
import vectorbt as vbt

import config
import fitness
import lambda_selector
import strategy_engine as engine
import trade_floor

logger = logging.getLogger(__name__)


def _hash_solution(solution: np.ndarray | list | tuple) -> str:
    """Return a stable hash for GA solutions with mixed dtypes.

    PyGAD returns the best solution as an ``object`` dtype array when genes
    have mixed types (e.g. both ``int`` and ``float``). Passing such an array
    directly to ``np.round`` triggers ``TypeError: loop of ufunc does not
    support argument 0 of type float which has no callable rint method``.

    Casting to ``float`` before rounding normalises the array and avoids this
    issue while keeping hash determinism.
    """

    arr = np.asarray(solution, dtype=float)
    arr = np.round(arr, 6)
    return hashlib.sha256(arr.tobytes()).hexdigest()


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
            sweep_rows_all = []
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
            gens_r1 = config.MULTI_ASSET.get("lambda_probe_generations_round1", 3)
            gens_r2 = config.MULTI_ASSET.get("lambda_probe_generations_round2", 5)
            max_gens = config.MULTI_ASSET.get("lambda_probe_generations_max", gens_r2)
            reprobe_on_dup = config.MULTI_ASSET.get(
                "lambda_probe_round2_on_duplicate", True
            )
            reprobe_shortlist = config.MULTI_ASSET.get(
                "lambda_probe_round2_for_shortlist", False
            )
            round2_extra = config.MULTI_ASSET.get("lambda_probe_seeds_round2_extra", [])
            finalist_max_passes = config.MULTI_ASSET.get(
                "lambda_finalist_max_passes", 3
            )
            pop_size = config.MULTI_ASSET.get("lambda_probe_population", 12)
            pop_size_r2 = config.MULTI_ASSET.get(
                "lambda_probe_population_round2", pop_size
            )
            dup_tol = config.MULTI_ASSET.get("lambda_duplicate_tol", 1e-6)
            ndigits = max(0, int(abs(np.log10(max(dup_tol, 1e-12)))))
            rank_stat = config.MULTI_ASSET.get("lambda_rank_stat", "mean")

            probe_cache: dict[
                tuple[float, int, int, int], lambda_selector.LambdaSweepRow
            ] = {}

            def _probe_lambda(lam, seed, generations, population, round_id):
                settings = dict(config.MULTI_ASSET)
                settings["lambda_dispersion"] = lam
                settings["trade_floor_policy"] = "soft_penalty"
                settings["soft_penalty_mode"] = "multiplicative"
                evaluator = fitness.MultiAssetFitnessEvaluator(
                    train_data, config.STRATEGY_RULES, gene_map, settings
                )
                try:
                    probe = pygad.GA(
                        num_generations=generations,
                        num_parents_mating=2,
                        sol_per_pop=population,
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
                    sol_hash = _hash_solution(best_solution)
                    row = lambda_selector.LambdaSweepRow(
                        lambda_value=lam,
                        mu_val=mu_val,
                        sigma_val=sigma_val,
                        mu_tr=mu_tr,
                        sigma_tr=sigma_tr,
                        F_tr=F_tr,
                        coverage=coverage,
                        fold=0,
                        seed=seed,
                        round=round_id,
                        solution_hash=sol_hash,
                    )
                    gap = mu_tr - mu_val
                    print(
                        f"λ={lam} seed={seed} | μ_val={mu_val:.4f} | "
                        f"σ_val={sigma_val:.4f} | gap={gap:.4f} | "
                        f"coverage={coverage:.4f}"
                    )
                except Exception as e:  # pragma: no cover - safety net
                    print(f"Probe error for λ={lam} seed={seed}: {e}")
                    row = lambda_selector.LambdaSweepRow(
                        lambda_value=lam,
                        mu_val=np.nan,
                        sigma_val=np.nan,
                        mu_tr=np.nan,
                        sigma_tr=np.nan,
                        F_tr=np.nan,
                        coverage=np.nan,
                        fold=0,
                        seed=seed,
                        note="probe_error",
                        round=round_id,
                    )
                sweep_rows.append(row)
                sweep_rows_all.append(dataclasses.replace(row))
                probe_cache[(lam, seed, generations, population)] = dataclasses.replace(
                    row
                )
                return row

            round_id = 1
            print(f"λ-grid round1: gens={gens_r1}, pop={pop_size}")
            for lam in lam_grid:
                for seed in seeds:
                    _probe_lambda(lam, seed, gens_r1, pop_size, round_id)

            (
                selected_lam,
                sweep_table,
                shortlist_df,
            ) = lambda_selector.select_lambda_with_elbow(
                sweep_rows,
                shortlist_size=shortlist_size,
                sigma_pct_threshold=sigma_pctl,
                coverage_min=coverage_min,
                duplicate_tol=dup_tol,
                rank_stat=rank_stat,
            )

            def _dup_mask_close(df):
                rounded = df[["mu_val_mean", "sigma_val_mean"]].round(ndigits)
                return rounded.duplicated(keep=False).to_numpy()

            if reprobe_on_dup and _dup_mask_close(shortlist_df).any():
                dup_values = shortlist_df.loc[
                    _dup_mask_close(shortlist_df), "lambda"
                ].unique()
                current_gen = gens_r2
                pop = pop_size_r2
                round_id += 1
                print(
                    f"λ-grid reprobe: λ in {list(map(float, dup_values))}, "
                    f"gens={current_gen}, pop={pop}"
                )
                while dup_values.size:
                    sweep_rows = [
                        r for r in sweep_rows if r.lambda_value not in dup_values
                    ]
                    for lam in dup_values:
                        for seed in seeds:
                            _probe_lambda(lam, seed, current_gen, pop, round_id)
                    (
                        selected_lam,
                        sweep_table,
                        shortlist_df,
                    ) = lambda_selector.select_lambda_with_elbow(
                        sweep_rows,
                        shortlist_size=shortlist_size,
                        sigma_pct_threshold=sigma_pctl,
                        coverage_min=coverage_min,
                        duplicate_tol=dup_tol,
                        rank_stat=rank_stat,
                    )
                    if _dup_mask_close(shortlist_df).any() and current_gen < max_gens:
                        dup_values = shortlist_df.loc[
                            _dup_mask_close(shortlist_df), "lambda"
                        ].unique()
                        current_gen = max_gens
                        round_id += 1
                        print(
                            f"λ-grid reprobe: λ in {list(map(float, dup_values))}, "
                            f"gens={current_gen}, pop={pop}"
                        )
                    else:
                        break

            degenerate = (shortlist_df["elbow_dist"].abs() < 1e-9).all()
            if degenerate:
                logger.warning(
                    "λ-grid: shortlist elbow is degenerate; tie-breakers may dominate."
                )
            if degenerate or reprobe_shortlist:
                finalist_lams = shortlist_df["lambda"].unique()
                seed_pool = list(seeds) + list(round2_extra)
                target_level = 2
                passes = 0
                shortlist_set = set(finalist_lams)
                while passes < finalist_max_passes:
                    current_gen = gens_r2 if target_level == 2 else max_gens
                    pop = pop_size_r2
                    round_id += 1
                    print(
                        "λ-grid finalist reprobe (fairness): "
                        f"λ in {list(map(float, finalist_lams))}, "
                        f"gens={current_gen}, pop={pop}, level={target_level}"
                    )
                    sweep_rows = [
                        r for r in sweep_rows if r.lambda_value not in finalist_lams
                    ]
                    for lam in finalist_lams:
                        for seed in seed_pool:
                            key = (lam, seed, current_gen, pop)
                            cached = probe_cache.get(key)
                            if cached is not None:
                                sweep_rows.append(
                                    dataclasses.replace(cached, round=round_id)
                                )
                            else:
                                _probe_lambda(lam, seed, current_gen, pop, round_id)
                    passes += 1
                    (
                        selected_lam,
                        sweep_table,
                        shortlist_df,
                    ) = lambda_selector.select_lambda_with_elbow(
                        sweep_rows,
                        shortlist_size=shortlist_size,
                        sigma_pct_threshold=sigma_pctl,
                        coverage_min=coverage_min,
                        duplicate_tol=dup_tol,
                        rank_stat=rank_stat,
                    )
                    finalist_lams = shortlist_df["lambda"].unique()
                    new_set = set(finalist_lams)
                    shortlist_changed = new_set != shortlist_set
                    shortlist_set = new_set
                    degenerate = (shortlist_df["elbow_dist"].abs() < 1e-9).all()
                    if shortlist_changed:
                        continue
                    if degenerate and target_level < 3:
                        target_level = 3
                        continue
                    break

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

            rows_all_df = pd.DataFrame([r.to_dict() for r in sweep_rows_all])
            rows_final_df = pd.DataFrame([r.to_dict() for r in sweep_rows])
            rows_all_df.to_csv("lambda_sweep.csv", index=False)

            idx_A = shortlist_df["sigma_val_mean"].idxmin()
            idx_B = shortlist_df["mu_val_mean"].idxmax()
            A_sig = shortlist_df.loc[idx_A, "sigma_val_mean"]
            A_mu = shortlist_df.loc[idx_A, "mu_val_mean"]
            B_sig = shortlist_df.loc[idx_B, "sigma_val_mean"]
            B_mu = shortlist_df.loc[idx_B, "mu_val_mean"]

            fig, ax = plt.subplots()
            ax.scatter(
                sweep_table["sigma_val_mean"],
                sweep_table["mu_val_mean"],
            )
            for _, row in sweep_table.iterrows():
                ax.annotate(
                    f"{row['lambda']}",
                    (row["sigma_val_mean"], row["mu_val_mean"]),
                )
            ax.plot([A_sig, B_sig], [A_mu, B_mu], linestyle="--", color="grey")
            chosen = shortlist_df.iloc[0]
            ax.scatter(
                [chosen["sigma_val_mean"]],
                [chosen["mu_val_mean"]],
                marker="x",
                s=80,
                color="red",
            )
            ax.annotate(
                f"λ*={chosen['lambda']}",
                (chosen["sigma_val_mean"], chosen["mu_val_mean"]),
            )
            ax.set_xlabel("sigma_val_mean")
            ax.set_ylabel("mu_val_mean")
            fig.tight_layout()
            fig.savefig("lambda_frontier.png")
            plt.close(fig)

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
                "rows_all": rows_all_df.to_dict(orient="records"),
                "rows_final": rows_final_df.to_dict(orient="records"),
                "rows_agg": sweep_table.to_dict(orient="records"),
                "shortlist": shortlist_df.to_dict(orient="records"),
                "elbow_AB": {
                    "A": {
                        "lambda": float(shortlist_df.loc[idx_A, "lambda"]),
                        "sigma": float(A_sig),
                        "mu": float(A_mu),
                    },
                    "B": {
                        "lambda": float(shortlist_df.loc[idx_B, "lambda"]),
                        "sigma": float(B_sig),
                        "mu": float(B_mu),
                    },
                },
                "chosen": {
                    "lambda": float(selected_lam),
                    "elbow_dist": float(chosen["elbow_dist"]),
                },
                "probe": {
                    "gens_round1": gens_r1,
                    "gens_round2": gens_r2,
                    "gens_max": max_gens,
                    "pop_r1": pop_size,
                    "pop_r2": pop_size_r2,
                    "duplicate_tol": dup_tol,
                    "round2_on_duplicate": reprobe_on_dup,
                    "round2_for_shortlist": reprobe_shortlist,
                    "rank_stat": rank_stat,
                },
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
