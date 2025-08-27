"""Walk-Forward Validation Module."""

from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import os
import numpy as np
import pygad
import vectorbt as vbt
import math
import json

import config
import data_loader
import strategy_engine as engine
from gene_parser import parse_genes_from_config
import fitness


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


def run_walk_forward_validation(initial_champions=None, data=None):
    """Execute walk-forward validation across the available data.

    Parameters
    ----------
    initial_champions : list[list[float]] or None
        Optional list of solutions to seed the first population. Each solution
        should be an iterable of gene values matching the strategy's genes.
    data : DataFrame or dict, optional
        Preloaded dataset to reuse instead of fetching from disk/API.
    """
    print("\n=== Running Walk-Forward Validation ===")
    np.random.seed(config.SEED)
    num_cores = os.cpu_count()
    print(f"Using {num_cores} CPU cores for GA optimisation during each window.")
    wf_settings = getattr(config, "WALK_FORWARD_SETTINGS", {})
    date_range = wf_settings.get("total_data_range", {})
    start_date = date_range.get("start", config.TRAINING_PERIOD["start"])
    end_date = date_range.get("end", config.VALIDATION_PERIOD["end"])

    multi = getattr(config, "MULTI_ASSET", {}).get("enabled")
    if data is not None:
        all_data = data
        if multi:
            if not all_data:
                print("No data available for walk-forward validation.")
                return
            sample_df = next(iter(all_data.values()))
            start = sample_df.index[0]
            end = sample_df.index[-1]
        else:
            if all_data.empty:
                print("No data available for walk-forward validation.")
                return
            start = all_data.index[0]
            end = all_data.index[-1]
    else:
        if multi:
            all_data = data_loader.get_group_data(
                asset_group=config.ASSET_GROUP,
                start_date=start_date,
                end_date=end_date,
                interval=config.TIMEFRAME,
                coverage_threshold=config.COVERAGE_THRESHOLD,
                verbose=False,
            )
            if not all_data:
                print("No data available for walk-forward validation.")
                return
            sample_df = next(iter(all_data.values()))
            start = sample_df.index[0]
            end = sample_df.index[-1]
        else:
            all_data, _ = data_loader.get_data(
                ticker=config.TICKER,
                start_date=start_date,
                end_date=end_date,
                interval=config.TIMEFRAME,
                verbose=False,
            )
            if all_data.empty:
                print("No data available for walk-forward validation.")
                return
            start = all_data.index[0]
            end = all_data.index[-1]
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
    per_asset_records = []
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
            rate = settings_train.get("min_total_trades_per_year")
            if rate:
                years = (p['train_end'] - p['train_start']).days / 365.25
                floor = math.ceil(rate * years)
                settings_train["min_total_trades"] = floor
                print(
                    f"Scaled min_total_trades (train): {floor} (rate={rate}/yr, span={years:.2f}y)"
                )
            # Explicitly propagate trade floor policy settings
            policy = settings_train.get("trade_floor_policy", "hard_floor")
            settings_train["trade_floor_policy"] = policy
            if policy == "soft_penalty":
                settings_train.setdefault("soft_penalty_mode", config.MULTI_ASSET.get("soft_penalty_mode", "multiplicative"))
                settings_train.setdefault("soft_penalty_strength", config.MULTI_ASSET.get("soft_penalty_strength", 1.0))
            print(f"Training lambda={settings_train.get('lambda_dispersion')}")
            evaluator = fitness.MultiAssetFitnessEvaluator(train_data, config.STRATEGY_RULES, gene_map, settings_train)
        else:
            evaluator = fitness.get_fitness_evaluator(train_data, config.STRATEGY_RULES, gene_map)
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
            random_seed=config.SEED,
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

        if multi and settings_train.get("trade_floor_policy") == "soft_penalty":
            _ = evaluator(None, best_solution, 0)
            train_details = evaluator.last_details
            pen = train_details.get("penalties", {}).get("trade_floor") or {}
            mode = settings_train.get("soft_penalty_mode", "multiplicative")
            floor_tr = train_details.get("min_total_trades")
            total_tr = train_details.get("total_trades")
            if mode == "additive":
                delta = pen.get("penalty", 0.0)
                print(
                    f"Training soft floor (additive): floor={floor_tr}, total trades={total_tr}, delta={delta:.4f}"
                )
            else:
                mult = pen.get("scale", 1.0)
                print(
                    f"Training soft floor (multiplicative): floor={floor_tr}, total trades={total_tr}, multiplier={mult:.4f}"
                )

        winning_params = {
            gene_map[i]["name"]: best_solution[i] for i in range(len(best_solution))
        }

        rules = fitness._inject_genes_into_rules(config.STRATEGY_RULES, gene_map, best_solution)
        if multi:
            settings_val = dict(config.MULTI_ASSET)
            rate = settings_val.get("min_total_trades_per_year")
            if rate:
                years_val = (p['test_end'] - p['test_start']).days / 365.25
                floor_val = math.ceil(rate * years_val)
                settings_val["min_total_trades"] = floor_val
                print(
                    f"Scaled min_total_trades (validation): {floor_val} (rate={rate}/yr, span={years_val:.2f}y)"
                )
            policy_val = settings_val.get("trade_floor_policy", "hard_floor")
            settings_val["trade_floor_policy"] = policy_val
            if policy_val == "soft_penalty":
                settings_val.setdefault("soft_penalty_mode", config.MULTI_ASSET.get("soft_penalty_mode", "multiplicative"))
                settings_val.setdefault(
                    "soft_penalty_strength", config.MULTI_ASSET.get("soft_penalty_strength", 1.0)
                )
            print(f"Validation lambda={settings_val.get('lambda_dispersion')}")
            test_eval = fitness.MultiAssetFitnessEvaluator(test_data, config.STRATEGY_RULES, gene_map, settings_val)
            validation_score = test_eval(None, best_solution, 0)
            details = test_eval.last_details
            if policy_val == "soft_penalty":
                pen = details.get("penalties", {}).get("trade_floor") or {}
                mode = settings_val.get("soft_penalty_mode", "multiplicative")
                floor_v = details.get("min_total_trades")
                total_v = details.get("total_trades")
                if mode == "additive":
                    delta = pen.get("penalty", 0.0)
                    print(
                        f"Validation soft floor (additive): floor={floor_v}, total trades={total_v}, delta={delta:.4f}"
                    )
                else:
                    mult = pen.get("scale", 1.0)
                    print(
                        f"Validation soft floor (multiplicative): floor={floor_v}, total trades={total_v}, multiplier={mult:.4f}"
                    )
            per_details = details.get('per_asset', {})
            for t, d in per_details.items():
                per_asset_records.append({
                    'fold': idx,
                    'ticker': t,
                    'included': d.get('included'),
                    'score': d.get('score'),
                    'trades': d.get('trades'),
                    'reason': d.get('reason'),
                })

            reason_counts = {}
            for d in per_details.values():
                r = d.get('reason')
                if r:
                    reason_counts[r] = reason_counts.get(r, 0) + 1
            if reason_counts:
                print("Exclusion reasons:")
                for r, c in reason_counts.items():
                    print(f"  {r}: {c}")

            kappa = settings_val.get('coverage_penalty', 0.0)
            included = details.get('assets_included', 0)
            total_assets = included + details.get('assets_ignored', 0)
            print(f"Coverage penalty applied: κ={kappa}, included={included}/{total_assets}")

            fitness.print_floor_failures(getattr(test_eval, "floor_failures", {}))
            cov_pen = details.get('penalties', {}).get('coverage', 0.0)
            print(
                (
                    f"Validation fitness: {validation_score:.4f} | "
                    f"Lambda={settings_val.get('lambda_dispersion'):.4f} | "
                    f"Mu: {details.get('mu'):.4f} | "
                    f"Lambda*Sigma: {details.get('lambda_sigma'):.4f} | "
                    f"Coverage Penalty: {cov_pen:.4f}"
                )
            )
            scored = [
                (t, d['score'], d.get('trades', 0))
                for t, d in details['per_asset'].items()
                if d['score'] is not None
            ]
            if scored:
                scored.sort(key=lambda x: x[1])
                n = min(3, len(scored) // 2)  # avoid overlap
                top = scored[-n:][::-1]
                bottom = scored[:n]
                print("Top assets:")
                for t, s, tr in top:
                    print(f"  {t}: score={s:.3f}, trades={tr}")
                print("Bottom assets:")
                for t, s, tr in bottom:
                    print(f"  {t}: score={s:.3f}, trades={tr}")
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
                'Lambda': settings_val.get('lambda_dispersion'),
                'Total Trades': details.get('total_trades'),
                'Scaled Floor': details.get('min_total_trades'),
                'Assets Included': details.get('assets_included'),
                'Assets Traded': details.get('assets_traded'),
                'Coverage Penalty': cov_pen,
                'Params': winning_params,
            })
            continue

        entries = engine.process_strategy_rules(test_data, rules)
        if entries.sum() < config.FITNESS_WEIGHTS['min_trades']:
            print("No trades in test period.")
            per_asset_records.append({
                'fold': idx,
                'ticker': config.TICKER,
                'included': False,
                'score': None,
                'trades': 0,
                'reason': 'no_trades',
            })
            results.append({
                'Window': idx,
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
            fees=config.FEES,
            freq=config.to_pandas_freq(config.TIMEFRAME),
        )
        stats = portfolio.stats()
        trades = int(portfolio.trades.count()) if hasattr(portfolio, 'trades') else 0
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
        validation_score = val_evaluator(None, best_solution, 0)
        champion_settings = getattr(config, "CHAMPION_SELECTION_SETTINGS", {})
        champion_pool = _update_champion_pool(
            champion_pool, best_solution, validation_score, gene_space, champion_settings
        )

        per_asset_records.append({
            'fold': idx,
            'ticker': config.TICKER,
            'included': True,
            'score': validation_score,
            'trades': trades,
            'reason': None,
        })

        results.append({
            'Window': idx,
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
    results_df.to_csv('walk_forward_results.csv', index=False)
    if per_asset_records:
        pd.DataFrame(per_asset_records).to_csv('walk_forward_per_asset.csv', index=False)
    print("\n=== Walk-Forward Summary ===")
    with pd.option_context('display.max_colwidth', None, 'display.width', None):
        if multi:
            cols = [
                'Window', 'Fitness', 'Mu', 'Sigma', 'Lambda', 'Lambda Sigma',
                'Total Trades', 'Scaled Floor', 'Assets Included',
                'Assets Traded', 'Coverage Penalty', 'Params'
            ]
        else:
            cols = [
                'Window', 'Total Return [%]', 'Max Drawdown [%]',
                'Sharpe Ratio', 'Sortino Ratio', 'Win Rate [%]', 'Params'
            ]
        print(results_df[cols].to_string(index=False))

    if multi:
        poor = getattr(config, 'MULTI_ASSET', {}).get('poor_score', -999.0)
        mask = results_df['Fitness'] != poor
        mask &= results_df['Fitness'].notna()
        avg_fitness = results_df.loc[mask, 'Fitness'].mean()
        failure_rate = 1 - mask.mean()
        print("\nAggregate Metrics:")
        print(f"Average Fitness: {avg_fitness:.4f} | Fold failure rate: {failure_rate:.2%}")
        summary = {
            'folds': results_df,
            'average_fitness': avg_fitness,
            'fold_failure_rate': failure_rate,
            'artifact_version': '1.0.0',
        }
        summary_json = summary.copy()
        summary_json['folds'] = json.loads(results_df.to_json(orient='records'))
        with open('walk_forward_summary.json', 'w') as f:
            json.dump(summary_json, f, indent=2)
        return summary

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

    summary = {
        'folds': results_df,
        'average_return': avg_return,
        'std_return': std_return,
        'average_sharpe': avg_sharpe,
        'average_sortino': avg_sortino,
        'average_win_rate': avg_win,
        'total_compounded_return': total_compounded_return,
        'artifact_version': '1.0.0',
    }
    summary_json = summary.copy()
    summary_json['folds'] = json.loads(results_df.to_json(orient='records'))
    with open('walk_forward_summary.json', 'w') as f:
        json.dump(summary_json, f, indent=2)
    return summary
