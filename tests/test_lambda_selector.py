import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lambda_selector as ls  # noqa: E402


def test_select_lambda_recovers_best_mu():
    rows = [
        ls.LambdaSweepRow(
            0.1,
            mu_val=1.0,
            sigma_val=0.5,
            mu_tr=1.0,
            sigma_tr=0.5,
            F_tr=0.75,
            coverage=1.0,
        ),
        ls.LambdaSweepRow(
            0.2,
            mu_val=1.5,
            sigma_val=0.6,
            mu_tr=1.5,
            sigma_tr=0.6,
            F_tr=1.14,
            coverage=1.0,
        ),
        ls.LambdaSweepRow(
            0.3,
            mu_val=1.2,
            sigma_val=0.7,
            mu_tr=1.2,
            sigma_tr=0.7,
            F_tr=0.99,
            coverage=1.0,
        ),
    ]
    lam, table, _ = ls.select_lambda_with_elbow(rows)
    assert lam == 0.2
    assert set(table["lambda"]) == {0.1, 0.2, 0.3}


def test_select_lambda_prefers_lower_sigma_on_tie(caplog):
    rows = [
        ls.LambdaSweepRow(
            0.1,
            mu_val=1.0,
            sigma_val=0.5,
            mu_tr=1.0,
            sigma_tr=0.5,
            F_tr=0.95,
            coverage=1.0,
        ),
        ls.LambdaSweepRow(
            0.2,
            mu_val=1.0,
            sigma_val=0.4,
            mu_tr=1.0,
            sigma_tr=0.4,
            F_tr=0.92,
            coverage=1.0,
        ),
    ]
    with caplog.at_level(logging.INFO, logger=ls.logger.name):
        lam, _, _ = ls.select_lambda_with_elbow(rows, sigma_pct_threshold=1.0)
    assert lam == 0.2
    assert "skipping elbow" in caplog.text.lower()


def test_select_lambda_elbow_choice():
    rows = [
        ls.LambdaSweepRow(
            0.1,
            mu_val=1.5,
            sigma_val=0.9,
            mu_tr=1.5,
            sigma_tr=0.9,
            F_tr=1.41,
            coverage=1.0,
        ),
        ls.LambdaSweepRow(
            0.2,
            mu_val=1.4,
            sigma_val=0.5,
            mu_tr=1.4,
            sigma_tr=0.5,
            F_tr=1.30,
            coverage=1.0,
        ),
        ls.LambdaSweepRow(
            0.3,
            mu_val=1.2,
            sigma_val=0.3,
            mu_tr=1.2,
            sigma_tr=0.3,
            F_tr=1.11,
            coverage=1.0,
        ),
    ]
    lam, _, _ = ls.select_lambda_with_elbow(rows)
    assert lam == 0.2


def test_select_lambda_respects_coverage_min():
    rows = [
        ls.LambdaSweepRow(
            0.1,
            mu_val=1.0,
            sigma_val=0.5,
            mu_tr=1.0,
            sigma_tr=0.5,
            F_tr=0.75,
            coverage=0.5,
        ),
        ls.LambdaSweepRow(
            0.2,
            mu_val=2.0,
            sigma_val=0.5,
            mu_tr=2.0,
            sigma_tr=0.5,
            F_tr=1.6,
            coverage=0.1,
        ),
    ]
    lam, _, _ = ls.select_lambda_with_elbow(rows, coverage_min=0.3)
    assert lam == 0.1


def test_select_lambda_warns_on_degenerate(caplog):
    rows = [
        ls.LambdaSweepRow(
            0.1,
            mu_val=1.0,
            sigma_val=0.5,
            mu_tr=1.0,
            sigma_tr=0.5,
            F_tr=0.75,
            coverage=1.0,
        ),
        ls.LambdaSweepRow(
            0.2,
            mu_val=2.0,
            sigma_val=0.5,
            mu_tr=2.0,
            sigma_tr=0.5,
            F_tr=1.5,
            coverage=1.0,
        ),
        ls.LambdaSweepRow(
            0.3,
            mu_val=3.0,
            sigma_val=0.5,
            mu_tr=3.0,
            sigma_tr=0.5,
            F_tr=2.25,
            coverage=1.0,
        ),
    ]
    with caplog.at_level(logging.WARNING, logger=ls.logger.name):
        lam, _, _ = ls.select_lambda_with_elbow(rows)
    assert "degenerate" in caplog.text.lower()
    assert lam == 0.3


def test_select_lambda_logs_and_drops_nan(caplog):
    rows = [
        ls.LambdaSweepRow(
            0.1,
            mu_val=1.0,
            sigma_val=0.5,
            mu_tr=1.0,
            sigma_tr=0.5,
            F_tr=0.75,
            coverage=1.0,
        ),
        ls.LambdaSweepRow(
            0.2,
            mu_val=float("nan"),
            sigma_val=float("nan"),
            mu_tr=1.0,
            sigma_tr=0.5,
            F_tr=0.75,
            coverage=1.0,
            seed=42,
        ),
    ]
    with caplog.at_level(logging.WARNING, logger=ls.logger.name):
        lam, table, _ = ls.select_lambda_with_elbow(rows)
    assert "NaN metrics for λ=0.2; dropped 1 rows" in caplog.text
    assert 0.2 not in set(table["lambda"])
    assert lam == 0.1


def test_shortlist_dedup_triggers_tiebreak(caplog):
    rows = [
        ls.LambdaSweepRow(
            0.1,
            mu_val=1.0 + 3e-7,
            sigma_val=0.5 + 3e-7,
            mu_tr=1.0,
            sigma_tr=0.5,
            F_tr=0.95,
            coverage=1.0,
        ),
        ls.LambdaSweepRow(
            0.2,
            mu_val=1.0 + 3e-7,
            sigma_val=0.5 + 3e-7,
            mu_tr=1.0,
            sigma_tr=0.5,
            F_tr=0.95,
            coverage=1.0,
        ),
        ls.LambdaSweepRow(
            0.3,
            mu_val=1.0 + 3e-7,
            sigma_val=0.4,
            mu_tr=1.0,
            sigma_tr=0.4,
            F_tr=0.92,
            coverage=1.0,
        ),
    ]
    with caplog.at_level(logging.INFO, logger=ls.logger.name):
        lam, _, _ = ls.select_lambda_with_elbow(rows, sigma_pct_threshold=1.0)
    assert lam == 0.3
    assert "Shortlist de-duplicated: 3→2" in caplog.text
    assert "Shortlist size 2;" in caplog.text


def test_shortlist_dedup_to_single_row(caplog):
    rows = [
        ls.LambdaSweepRow(
            0.1,
            mu_val=1.0 + 3e-7,
            sigma_val=0.5 + 3e-7,
            mu_tr=1.0,
            sigma_tr=0.5,
            F_tr=0.95,
            coverage=1.0,
        ),
        ls.LambdaSweepRow(
            0.2,
            mu_val=1.0 + 2e-7,
            sigma_val=0.5 + 2e-7,
            mu_tr=1.0,
            sigma_tr=0.5,
            F_tr=0.95,
            coverage=1.0,
        ),
        ls.LambdaSweepRow(
            0.3,
            mu_val=1.0 + 1e-7,
            sigma_val=0.5 + 1e-7,
            mu_tr=1.0,
            sigma_tr=0.5,
            F_tr=0.95,
            coverage=1.0,
        ),
    ]
    with caplog.at_level(logging.INFO, logger=ls.logger.name):
        lam, _, _ = ls.select_lambda_with_elbow(rows, sigma_pct_threshold=1.0)
    assert lam == 0.1
    assert "Shortlist de-duplicated: 3→1" in caplog.text
    assert "Shortlist size 1;" in caplog.text


def test_soft_sigma_prefers_lower_sigma_when_mu_close():
    rows = [
        ls.LambdaSweepRow(
            0.1,
            mu_val=1.0,
            sigma_val=0.4,
            mu_tr=1.0,
            sigma_tr=0.4,
            F_tr=0.9,
            coverage=1.0,
        ),
        ls.LambdaSweepRow(
            0.2,
            mu_val=1.002,
            sigma_val=0.6,
            mu_tr=1.002,
            sigma_tr=0.6,
            F_tr=0.91,
            coverage=1.0,
        ),
    ]
    lam_default, _, _ = ls.select_lambda_with_elbow(rows, sigma_pct_threshold=1.0)
    lam_soft, _, _ = ls.select_lambda_with_elbow(
        rows, sigma_pct_threshold=1.0, soft_sigma_enabled=True
    )
    assert lam_default == 0.2
    assert lam_soft == 0.1


def test_soft_sigma_keeps_ranking_when_mu_far():
    rows = [
        ls.LambdaSweepRow(
            0.1,
            mu_val=1.0,
            sigma_val=0.4,
            mu_tr=1.0,
            sigma_tr=0.4,
            F_tr=0.9,
            coverage=1.0,
        ),
        ls.LambdaSweepRow(
            0.2,
            mu_val=1.3,
            sigma_val=0.6,
            mu_tr=1.3,
            sigma_tr=0.6,
            F_tr=1.17,
            coverage=1.0,
        ),
    ]
    lam_default, _, _ = ls.select_lambda_with_elbow(rows, sigma_pct_threshold=1.0)
    lam_soft, _, _ = ls.select_lambda_with_elbow(
        rows, sigma_pct_threshold=1.0, soft_sigma_enabled=True
    )
    assert lam_default == 0.2
    assert lam_soft == lam_default


def test_select_lambda_median_rank_stat():
    rows = []
    for mu in [0.0, 10.0, 10.0]:
        rows.append(
            ls.LambdaSweepRow(
                0.1,
                mu_val=mu,
                sigma_val=0.5,
                mu_tr=mu,
                sigma_tr=0.5,
                F_tr=1.0,
                coverage=1.0,
            )
        )
    for mu in [7.0, 7.0, 7.0]:
        rows.append(
            ls.LambdaSweepRow(
                0.2,
                mu_val=mu,
                sigma_val=0.5,
                mu_tr=mu,
                sigma_tr=0.5,
                F_tr=1.0,
                coverage=1.0,
            )
        )
    lam_mean, _, _ = ls.select_lambda_with_elbow(
        rows, shortlist_size=2, rank_stat="mean"
    )
    lam_med, _, _ = ls.select_lambda_with_elbow(
        rows, shortlist_size=2, rank_stat="median"
    )
    assert lam_mean == 0.2
    assert lam_med == 0.1
