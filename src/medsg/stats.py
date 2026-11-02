"""Statistics used across stages: bootstrap CIs, measurement noise floor, multiple-comparison control.

Plain-words glossary
- Bootstrap CI: resample the pairs with replacement many times, recompute the
  statistic each time; the middle 95% of those values is the 95% interval.
- SEM (standard error of measurement): how much a metric wobbles when the same
  face is measured again under harmless changes (shift, JPEG, flip...).
- MDC95 (minimal detectable change) = 1.96 * sqrt(2) * SEM: a pre->post change
  smaller than this cannot be told apart from measurement noise (95% level).
- Holm correction: when testing many metrics at once, makes p-values stricter
  so that the chance of *any* false positive stays at 5%.
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(x: np.ndarray, stat=np.mean, n_boot: int = 10_000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Return (estimate, lo, hi) percentile bootstrap CI of `stat` over a 1-D sample."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise ValueError("need at least 2 finite values")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = stat(x[idx], axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(stat(x)), float(lo), float(hi)


def sem_within(values: np.ndarray) -> float:
    """Pooled within-subject SD from an (n_subjects, n_repeats) array (one-way ANOVA residual)."""
    v = np.asarray(values, dtype=np.float64)
    assert v.ndim == 2 and v.shape[1] >= 2, v.shape
    v = v[np.isfinite(v).all(1)]
    resid = v - v.mean(1, keepdims=True)
    dof = v.shape[0] * (v.shape[1] - 1)
    return float(np.sqrt((resid ** 2).sum() / dof))


def mdc95(sem: float) -> float:
    return 1.96 * np.sqrt(2.0) * sem


def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=np.float64)
    order = np.argsort(p)
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
