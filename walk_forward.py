"""Walk-Forward Validation Module."""

from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import os
import numpy as np
import pygad
import vectorbt as vbt
from utils import set_global_seed

import config
import data_loader
import strategy_engine as engine
from gene_parser import parse_genes_from_config
import fitness
import analysis
from utils import _norm_freq


def _sparkline(arr):
    """Return a tiny sparkline for a sequence of numbers."""
    ticks = "▁▂▃▄▅▆▇█"
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return ""
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx == mn:
        return ticks[0] * len(arr)
    scaled = (arr - mn) / (mx - mn) * (len(ticks) - 1)
    chars = [ticks[int(round(x))] for x in scaled]
    return "".join(chars)


def _generate_periods(start: datetime, end: datetime, train_months: int, test_months: int):
    """Generate rolling training and testing windows."""
    # Ensure plain Python datetimes for relativedelta calculations
    start = pd.to_datetime(start).to_pydatetime()
    end = pd.to_datetime(end).to_pydatetime()

    # Quick check to avoid an infinite loop when the dataset is too short
    if start + relativedelta(months=train_months + test_months) > end:
        return []

    periods = []
    current_start = start
    while True:
        train_end = current_start + relativedelta(months=train_months)
        test_end = train_end + relativedelta(months=test_months)
        if test_end > end:
            break
        periods.append({
            'train_start': current_start,
            'train_end': train_end,
            'test_start': train_end,
            'test_end': test_end,
        })
        current_start += relativedelta(months=test_months)
    return periods


def _update_champion_pool(pool, best_solution, validation_score, gene_space, settings):
    """Update champion pool based on validation fitness."""
    survival = settings.get("survival_threshold", 0.0)
    cloning = settings.get("cloning_threshold", float("inf"))
    num_clones = settings.get("num_clones", 0)
    mutation_rate = settings.get("clone_mutation_rate", 0.0)

    if validation_score < survival:
        print("Champion discarded due to poor performance.")
        return pool

    if validation_score >= cloning:
        print("Elite Champion found. Cloning champion.")
        pool.append(list(best_solution))
        for _ in range(num_clones):
            clone = list(best_solution)
            for idx in range(len(clone)):
                if np.random.rand() < mutation_rate:
                    gs = gene_space[idx]
                    low, high = gs["low"], gs["high"]
                    step = gs.get("step")
                    if step is not None:
                        steps = int(round((high - low) / step))
                        val = low + step * np.random.randint(0, steps + 1)
                    else:
                        val = np.random.uniform(low, high)
                    clone[idx] = type(clone[idx])(val)
            pool.append(clone)
    else:
        print("Viable Champion found and kept for next fold.")
        pool.append(list(best_solution))

    return pool


_wf_on_gen_best: dict | None = None
_wf_on_gen_eval = None
_wf_on_gen_func = None


def _wf_on_generation_cb(ga_instance):
    """Module-level ``on_generation`` callback for walk-forward runs.

    Storing required state in module-level globals keeps the callback picklable
    when the GA instance is serialized for process-based parallel fitness
    evaluation.
    """

    if _wf_on_gen_best is None:
        return

    best_sol, fit, _ = ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness
    )
    if fit > _wf_on_gen_best["fitness"]:
        _wf_on_gen_best["fitness"] = fit
        try:
            _wf_on_gen_func(None, best_sol, 0)
            analysis.log_asset_extremes(
                getattr(_wf_on_gen_eval, "last_details", {})
            )
        except Exception:
            pass


def _make_on_generation(fitness_eval, fitness_func):
    """Return a picklable ``on_generation`` callback that logs asset extremes."""

    global _wf_on_gen_best, _wf_on_gen_eval, _wf_on_gen_func
    _wf_on_gen_best = {"fitness": -float("inf")}
    _wf_on_gen_eval = fitness_eval
    _wf_on_gen_func = fitness_func
    return _wf_on_generation_cb


def run_walk_forward(initial_champions=None):
    """Execute walk-forward validation across the available data.

    Parameters
    ----------
    initial_champions : list[list[float]] or None
        Optional list of solutions to seed the first population. Each solution
        should be an iterable of gene values matching the strategy's genes.
    """
    print("\n=== Running Walk-Forward Validation ===")
    num_cores = os.cpu_count()
    print(f"Using {num_cores} CPU cores for GA optimisation during each window.")
    seed = None
    if getattr(config, "DETERMINISTIC", False):
        seed = getattr(config, "RANDOM_SEED", 42)
        set_global_seed(seed)
        print(f"Deterministic mode enabled. Seed={seed}")
    wf_settings = getattr(config, "WALK_FORWARD_SETTINGS", {})
    date_range = wf_settings.get("total_data_range", {})
    start_date = date_range.get("start", config.TRAINING_PERIOD["start"])
    end_date = date_range.get("end", config.VALIDATION_PERIOD["end"])

    multi = getattr(config, "MULTI_ASSET", {}).get("enabled")
    if multi:
        all_data = data_loader.get_group_data(
            asset_group=config.ASSET_GROUP,
            start_date=start_date,
            end_date=end_date,
            interval=config.TIMEFRAME,
            coverage_threshold=config.COVERAGE_THRESHOLD,
        )
        if not all_data:
            print("No data available for walk-forward validation.")
            return
        sample_df = next(iter(all_data.values()))
        start = sample_df.index[0]
        end = sample_df.index[-1]
        inclusion_counts = {t: 0 for t in all_data.keys()}
    else:
        all_data = data_loader.get_data(
            ticker=config.TICKER,
            start_date=start_date,
            end_date=end_date,
            interval=config.TIMEFRAME,
        )
        if all_data.empty:
            print("No data available for walk-forward validation.")
            return
        start = all_data.index[0]
        end = all_data.index[-1]
        inclusion_counts = {config.TICKER: 0}
    train_months = wf_settings.get(
        "training_period_length",
        getattr(config, "WALK_FORWARD_TRAINING_MONTHS", 12),
    )
    test_months = wf_settings.get(
        "validation_period_length",
        getattr(config, "WALK_FORWARD_TEST_MONTHS", 3),
    )

    periods = _generate_periods(start, end, train_months, test_months)
    if not periods:
        print("Insufficient data for the requested walk-forward windows.")
        return

    results = []
    champion_pool = list(initial_champions or [])

    for idx, p in enumerate(periods, start=1):
        print(f"\n--- Window {idx} ---")
        print(f"Train: {p['train_start'].date()} -> {p['train_end'].date()}")
        print(f"Test : {p['test_start'].date()} -> {p['test_end'].date()}")
        # fmt: off
        if multi:
            train_data = {t: df.loc[p['train_start']:p['train_end']] for t, df in all_data.items()}
            test_data = {t: df.loc[p['test_start']:p['test_end']] for t, df in all_data.items()}
        else:
            train_data = all_data.loc[p['train_start']:p['train_end']]
            test_data = all_data.loc[p['test_start']:p['test_end']]
        # fmt: on

        gene_space, gene_map, gene_types = parse_genes_from_config(config.STRATEGY_RULES)
        if multi:
            settings_train = dict(config.MULTI_ASSET)
            evaluator = fitness.MultiAssetFitnessEvaluator(train_data, config.STRATEGY_RULES, gene_map, settings_train)
            print(f"Training trade floor: {evaluator.settings.get('min_total_trades')}")
        else:
            evaluator = fitness.get_fitness_evaluator(train_data, config.STRATEGY_RULES, gene_map)
        ev_name = type(evaluator).__name__
        objective = getattr(evaluator, "settings", {}).get("metric", "composite")
        print(f"[WalkForward] Evaluator: {ev_name} | Objective: {objective}")
        assert objective, "Objective must be defined"
        ga_instance = pygad.GA(
            num_generations=config.GA_NUM_GENERATIONS,
            num_parents_mating=config.GA_PARENTS_MATING,
            sol_per_pop=config.GA_POPULATION_SIZE,
            num_genes=len(gene_space),
            gene_space=gene_space,
            gene_type=gene_types,
            mutation_num_genes=config.GA_MUTATION_NUM_GENES,
            fitness_func=evaluator.__call__,
            parallel_processing=['process', num_cores],
            random_seed=seed,
            on_generation=_make_on_generation(evaluator, evaluator.__call__),
        )
        if champion_pool and hasattr(ga_instance, "population"):
            champs = np.array(champion_pool, dtype=float)
            if champs.ndim == 1:
                champs = champs.reshape(1, -1)
            if champs.shape[1] == ga_instance.population.shape[1]:
                champs = champs[-config.GA_POPULATION_SIZE :]
                num_champs = min(len(champs), ga_instance.population.shape[0])
                ga_instance.population[:num_champs] = champs[:num_champs]
                if hasattr(ga_instance, "initial_population"):
                    ga_instance.initial_population[:num_champs] = champs[:num_champs]
        ga_instance.run()
        best_solution, best_fitness, _ = ga_instance.best_solution()
        print(f"Best training fitness: {best_fitness:.4f}")

        winning_params = {
            gene_map[i]["name"]: best_solution[i] for i in range(len(best_solution))
        }

        rules = fitness._inject_genes_into_rules(config.STRATEGY_RULES, gene_map, best_solution)
        if multi:
            settings_val = dict(config.MULTI_ASSET)
            test_eval = fitness.MultiAssetFitnessEvaluator(test_data, config.STRATEGY_RULES, gene_map, settings_val)
            ev_name = type(test_eval).__name__
            objective = getattr(test_eval, "settings", {}).get("metric", "composite")
            print(f"[WalkForward] Validation evaluator: {ev_name} | Objective: {objective}")
            assert objective, "Objective must be defined"
            validation_score = test_eval(None, best_solution, 0)
            analysis.persist_details(test_eval)
            details = test_eval.last_details
            cov_pen = details.get('penalties', {}).get('coverage')
            cov_pen = cov_pen if isinstance(cov_pen, (int, float)) else 0.0
            mu = details.get('mu')
            sigma = details.get('sigma')
            lam_sig = details.get('lambda_sigma')
            assets_incl = details.get('assets_included')
            total_assets = len(test_data)
            mu_str = f"{mu:.4f}" if isinstance(mu, (int, float)) else "nan"
            sigma_str = (
                f"{sigma:.4f}"
                if isinstance(sigma, (int, float))
                else "nan (no scored assets)"
            )
            lam_sig_str = (
                f"{lam_sig:.4f}" if isinstance(lam_sig, (int, float)) else "nan"
            )
            assets_str = f"{assets_incl}/{total_assets}"
            floor = test_eval.settings.get('min_total_trades')
            print(
                "Validation fitness: {val:.4f} | floor={floor} | mu={mu} | sigma={sig} | "
                "lambda*sigma={lam} | coverage_penalty={cov:.4f} | assets={assets}".format(
                    val=validation_score,
                    floor=floor,
                    mu=mu_str,
                    sig=sigma_str,
                    lam=lam_sig_str,
                    cov=cov_pen,
                    assets=assets_str,
                )
            )
            poor = config.MULTI_ASSET.get("poor_score", -999.0)
            if validation_score == poor:
                trade_pen = details.get('penalties', {}).get('trade_floor')
                if trade_pen == "hard_floor":
                    reason = "trade floor not met"
                elif trade_pen == "error":
                    reason = "evaluation error"
                elif trade_pen:
                    reason = f"trade floor penalty: {trade_pen}"
                else:
                    reason = "unspecified reason"
                print(f"Fitness equals poor_score ({poor}) due to {reason}.")
            analysis.log_asset_extremes(details)
            for t, d in details.get('per_asset', {}).items():
                if d.get('included'):
                    inclusion_counts[t] += 1
            # Compute combined equity curve for optional sparkline later
            equity_curves = [
                d.get('equity_curve')
                for d in details['per_asset'].values()
                if d.get('included') and isinstance(d.get('equity_curve'), pd.Series)
            ]
            combined_eq = None
            if equity_curves:
                eq_norms = [ec / ec.iloc[0] for ec in equity_curves if len(ec) > 0]
                if eq_norms:
                    combined_eq = sum(eq_norms) / len(eq_norms)
            champion_settings = getattr(config, "CHAMPION_SELECTION_SETTINGS", {})
            champion_pool = _update_champion_pool(
                champion_pool, best_solution, validation_score, gene_space, champion_settings
            )
            results.append({
                'Window': idx,
                'Fitness': validation_score,
                'Mu': details.get('mu'),
                'Sigma': details.get('sigma'),
                'Lambda Sigma': details.get('lambda_sigma'),
                'Total Trades': details.get('total_trades'),
                'Coverage Penalty': cov_pen,
                'Assets Traded': assets_str,
                'Equity Curve': combined_eq,
                'Params': winning_params,
            })
            continue

        entries = engine.process_strategy_rules(test_data, rules)
        if entries.sum() < config.FITNESS_WEIGHTS['min_trades']:
            print("No trades in test period.")
            results.append({
                'Window': idx,
                'Total Trades': int(entries.sum()),
                'Assets Traded': '0/1',
                'Total Return [%]': np.nan,
                'Max Drawdown [%]': np.nan,
                'Sharpe Ratio': np.nan,
                'Sortino Ratio': np.nan,
                'Win Rate [%]': np.nan,
                'Params': None,
            })
            continue
        exit_rules = rules.get('exit_rules', {})
        sl_rule = exit_rules.get('stop_loss', {})
        tsl_rule = exit_rules.get('trailing_stop', {})
        tp_rule = exit_rules.get('take_profit', {})

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
            close=test_data['Close'],
            entries=entries,
            exits=time_exit,
            sl_stop=sl_stop,
            tp_stop=tp_stop,
            sl_trail=sl_trail,
            fees=0.001,
            freq=_norm_freq(config.TIMEFRAME),
        )
        stats = portfolio.stats()
        tr = stats['Total Return [%]'] if isinstance(stats, dict) else stats.get('Total Return [%]')
        dd = stats['Max Drawdown [%]'] if isinstance(stats, dict) else stats.get('Max Drawdown [%]')
        sharpe = stats.get('Sharpe Ratio') if isinstance(stats, dict) else stats.get('Sharpe Ratio')
        sortino = stats.get('Sortino Ratio') if isinstance(stats, dict) else stats.get('Sortino Ratio')
        win_rate = stats.get('Win Rate [%]') if isinstance(stats, dict) else stats.get('Win Rate [%]')
        print(f"Test Return: {tr:.2f}% | Max DD: {dd:.2f}%")
        print("Winning Parameters:")
        for param_name, param_value in winning_params.items():
            print(f"  {param_name}: {param_value}")

        # Evaluate champion on validation data using composite fitness
        val_evaluator = fitness.FitnessEvaluator(test_data, config.STRATEGY_RULES, gene_map)
        ev_name = type(val_evaluator).__name__
        objective = getattr(val_evaluator, "settings", {}).get("metric", "composite")
        print(f"[WalkForward] Validation evaluator: {ev_name} | Objective: {objective}")
        assert objective, "Objective must be defined"
        validation_score = val_evaluator(None, best_solution, 0)
        analysis.persist_details(val_evaluator)
        champion_settings = getattr(config, "CHAMPION_SELECTION_SETTINGS", {})
        champion_pool = _update_champion_pool(
            champion_pool, best_solution, validation_score, gene_space, champion_settings
        )

        total_trades = int(entries.sum())
        results.append({
            'Window': idx,
            'Total Trades': total_trades,
            'Assets Traded': '1/1',
            'Total Return [%]': tr,
            'Max Drawdown [%]': dd,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Win Rate [%]': win_rate,
            'Params': winning_params,
        })

    if not results:
        print("\nNo walk-forward runs produced trades.")
        return None

    results_df = pd.DataFrame(results)
    print("\n=== Walk-Forward Summary ===")
    with pd.option_context('display.max_colwidth', None, 'display.width', None):
        print(results_df.to_string(index=False))

    if multi:
        traded = sum(1 for c in inclusion_counts.values() if c > 0)
        total_assets = len(inclusion_counts)
        print("\nInclusion Counts:")
        print(f"assets_traded = {traded}/{total_assets}")
        for t, c in inclusion_counts.items():
            print(f"  {t}: {c}")
        avg_fitness = results_df['Fitness'].mean()
        poor = config.MULTI_ASSET.get("poor_score", -999.0)
        total_folds = len(results_df)
        fails = (results_df['Fitness'] == poor).sum()
        floor_fail_rate = fails / total_folds if total_folds else float("nan")
        valid = results_df[results_df['Fitness'] != poor]
        mean_fit = valid['Fitness'].mean()
        median_fit = valid['Fitness'].median()
        median_mu = pd.to_numeric(valid['Mu'], errors='coerce').median()
        median_sigma = pd.to_numeric(valid['Sigma'], errors='coerce').median()
        median_lambda_sigma = pd.to_numeric(valid['Lambda Sigma'], errors='coerce').median()
        print("\nAggregate Metrics:")
        print(f"Average Fitness: {avg_fitness:.4f}")
        print(f"floor_fail_rate: {floor_fail_rate:.2%}")
        print(f"Mean Fitness (excl poor): {mean_fit:.4f}")
        print(f"Median Fitness (excl poor): {median_fit:.4f}")
        print(f"Median mu: {median_mu:.4f}")
        print(f"Median sigma: {median_sigma:.4f}")
        print(f"Median lambda*sigma: {median_lambda_sigma:.4f}")
        combined_eq = None
        if 'Equity Curve' in results_df.columns:
            try:
                combined_eq = pd.concat(
                    [s for s in results_df['Equity Curve'] if isinstance(s, pd.Series)]
                )
            except Exception:
                combined_eq = None
        if combined_eq is not None and not combined_eq.empty:
            print(f"Combined Equity: {_sparkline(combined_eq.values)}")
        return {
            'folds': results_df,
            'average_fitness': avg_fitness,
            'average_return': avg_fitness,
            'total_compounded_return': avg_fitness,
            'floor_fail_rate': floor_fail_rate,
            'mean_fitness': mean_fit,
            'median_fitness': median_fit,
            'median_mu': median_mu,
            'median_sigma': median_sigma,
            'median_lambda_sigma': median_lambda_sigma,
        }

    avg_return = results_df['Total Return [%]'].mean()
    std_return = results_df['Total Return [%]'].std()
    avg_sharpe = results_df['Sharpe Ratio'].mean()
    avg_sortino = results_df['Sortino Ratio'].mean()
    avg_win = results_df['Win Rate [%]'].mean()
    total_compounded_return = (results_df['Total Return [%]'] / 100 + 1).prod() - 1

    print("\nAggregate Metrics:")
    print(f"Average Return: {avg_return:.2f}% (+/- {std_return:.2f}%)")
    print(f"Average Sharpe: {avg_sharpe:.2f}")
    print(f"Average Sortino: {avg_sortino:.2f}")
    print(f"Average Win Rate: {avg_win:.2f}%")
    print(f"Total Compounded Return: {total_compounded_return * 100:.2f}%")

    return {
        'folds': results_df,
        'average_return': avg_return,
        'std_return': std_return,
        'average_sharpe': avg_sharpe,
        'average_sortino': avg_sortino,
        'average_win_rate': avg_win,
        'total_compounded_return': total_compounded_return,
    }


# Backwards compatibility for older imports
run_walk_forward_validation = run_walk_forward
