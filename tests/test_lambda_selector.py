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


def test_select_lambda_prefers_lower_sigma_on_tie():
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
    lam, _, _ = ls.select_lambda_with_elbow(rows)
    assert lam == 0.2


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
