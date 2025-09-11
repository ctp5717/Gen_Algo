"""Utility math functions used across the project."""

from __future__ import annotations

import numpy as np


def weighted_mean_std(values, weights):
    r"""Compute weighted mean and population standard deviation.

    Parameters
    ----------
    values : array-like
        Sequence of values :math:`m_i`.
    weights : array-like
        Corresponding weights :math:`u_i`. They do not need to be normalised.

    Returns
    -------
    tuple[float, float]
        ``(mu, sigma)`` where ``mu`` is the weighted mean and ``sigma`` is the
        weighted population standard deviation :math:`\sqrt{\sum u_i(m_i-\mu)^2}`.
        Weights are normalised internally so that ``sum(u_i)=1``.
    """

    w = np.asarray(weights, dtype=float)
    x = np.asarray(values, dtype=float)
    if w.ndim == 0:
        w = np.array([float(w)])
    if x.ndim == 0:
        x = np.array([float(x)])
    if len(w) != len(x) or len(w) == 0:
        raise ValueError("weighted_mean_std: values/weights length mismatch")
    if (w < 0).any():
        raise ValueError("weighted_mean_std: weights must be non-negative")
    total = w.sum()
    if total == 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / total
    mu = float(np.sum(w * x))
    variance = float(np.sum(w * (x - mu) ** 2))
    sigma_pop = float(np.sqrt(variance))
    return mu, sigma_pop
