import pytest

from utils.math import weighted_mean_std


def test_weighted_mean_std_handles_scalars():
    mu, sigma = weighted_mean_std(5, 2)
    assert mu == 5.0
    assert sigma == 0.0


def test_weighted_mean_std_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        weighted_mean_std([1, 2], [1])


def test_weighted_mean_std_handles_zero_total():
    mu, sigma = weighted_mean_std([1, 3], [0, 0])
    assert mu == pytest.approx(2.0)
    assert sigma == pytest.approx(1.0)


def test_weighted_mean_std_rejects_negative_weights():
    with pytest.raises(ValueError):
        weighted_mean_std([1, 3], [1, -1])
