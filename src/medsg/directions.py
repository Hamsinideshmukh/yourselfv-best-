"""Latent <-> anatomy maths (pure numpy): ridge Jacobians, constrained edit directions, leakage.

Plain words
- Whitened step u: a latent move divided, dimension by dimension, by the spread of w (std(w)).
  Stage 3/4a perturbations are delta = sigma * std(w) * eps, so u = delta / std(w) = sigma * eps.
- Local Jacobian J (one row per measured quantity, 512 columns): "if I move the latent by u inside
  layer group g, quantity q changes by about J[q] . u". Fitted by ridge regression across many faces,
  so it is the *average* local effect; the closed loop corrects face-specific differences.
- Constrained direction: the shortest move that changes the target by r while,
  to first order, leaving protected quantities N unchanged:  u = r * P J_t / (J_t P J_t),
  P = I - J_N^T (J_N J_N^T + eps I)^-1 J_N  (removes the part of a move that would change N).
- kappa = (J_t P J_t) / (J_t J_t): share of the target's own effect left after protection.
  Small kappa = the generator ties the target to a protected quantity ("entangled").
"""
from __future__ import annotations

import numpy as np

ALPHAS = np.logspace(-2, 5, 15)


def fit_ridge_cv(X: np.ndarray, Y: np.ndarray, seed: int, test_frac: float = 0.2,
                 n_folds: int = 5) -> dict[str, np.ndarray]:
    """Ridge Y ~ X with per-target alpha chosen by K-fold CV on a train part; held-out R² on the rest.

    Returns dict: coef (p, t) in the original X units, intercept (t,), alpha (t,), r2 (t,) held-out,
    and the train/test index arrays (into the finite rows).
    """
    assert X.ndim == 2 and Y.ndim == 2 and len(X) == len(Y), (X.shape, Y.shape)
    ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    X, Y = X[ok].astype(np.float64), Y[ok].astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_te = int(test_frac * len(X))
    te, tr = idx[:n_te], idx[n_te:]
    sx = X[tr].std(0) + 1e-12
    Xs = X / sx
    folds = np.array_split(rng.permutation(tr), n_folds)
    sse = np.zeros((len(ALPHAS), Y.shape[1]))
    eye = np.eye(X.shape[1])
    for k in range(n_folds):
        va = folds[k]
        fit = np.concatenate([folds[i] for i in range(n_folds) if i != k])
        xm, ym = Xs[fit].mean(0), Y[fit].mean(0)
        Xc, Yc = Xs[fit] - xm, Y[fit] - ym
        G, XtY = Xc.T @ Xc, Xc.T @ Yc
        for ai, a in enumerate(ALPHAS):
            B = np.linalg.solve(G + a * eye, XtY)
            sse[ai] += (((Xs[va] - xm) @ B + ym - Y[va]) ** 2).sum(0)
    best = ALPHAS[sse.argmin(0)]
    xm, ym = Xs[tr].mean(0), Y[tr].mean(0)
    Xc, Yc = Xs[tr] - xm, Y[tr] - ym
    G, XtY = Xc.T @ Xc, Xc.T @ Yc
    B = np.stack([np.linalg.solve(G + a * eye, XtY[:, t]) for t, a in enumerate(best)], 1)
    pred = (Xs[te] - xm) @ B + ym
    r2 = 1 - ((Y[te] - pred) ** 2).sum(0) / ((Y[te] - Y[te].mean(0)) ** 2).sum(0)
    coef = B / sx[:, None]
    return {"coef": coef, "intercept": ym - (xm * sx) @ coef, "alpha": best, "r2": r2, "train": tr, "test": te}


def fit_jacobian(U: np.ndarray, dY: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Local Jacobian J (t, 512) from whitened steps U (n, 512) and changes dY (n, t)."""
    assert U.shape[1] == 512, U.shape
    f = fit_ridge_cv(U, dY, seed)
    return {"J": f["coef"].T, "r2": f["r2"], "alpha": f["alpha"]}


def projector(JN: np.ndarray, scale: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Soft null-space projector of the protected rows JN (n, p), each row divided by its scale (e.g. MDC)."""
    A = JN / scale[:, None]
    p = JN.shape[1]
    if len(A) == 0:
        return np.eye(p)
    G = A @ A.T
    lam = eps * np.trace(G) / len(G)
    return np.eye(p) - A.T @ np.linalg.solve(G + lam * np.eye(len(G)), A)


def constrained_direction(Jt: np.ndarray, JN: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, float]:
    """Whitened step that changes the target by +1 unit with protected rows ~unchanged; returns (u, kappa)."""
    assert Jt.ndim == 1 and JN.ndim == 2 and JN.shape[1] == Jt.shape[0], (Jt.shape, JN.shape)
    P = projector(JN, scale)
    v = P @ Jt
    gain = float(Jt @ v)
    if gain <= 0:
        raise ValueError("target has no effect left after protection (gain <= 0)")
    return v / gain, gain / float(Jt @ Jt)


def global_direction(coef: np.ndarray) -> np.ndarray:
    """Baseline B4: minimum-norm move along a global ridge coefficient b, scaled to +1 predicted unit."""
    return coef / float(coef @ coef)
