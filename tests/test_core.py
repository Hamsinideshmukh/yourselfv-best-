"""CPU unit tests for measurement and statistics (no models, no dataset)."""
import math

import numpy as np
import pytest

from medsg.landmarks import euler_from_matrix
from medsg.metrics2d import compute_metrics_2d
from medsg.stats import bootstrap_ci, holm, mdc95, sem_within


def _fake_landmarks(seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(100, 300, size=(478, 3))
    p[468, :2], p[473, :2] = (150, 150), (250, 150)   # irises, horizontal
    p[152, :2] = (200, 320)                             # chin below
    p[105, :2], p[334, :2] = (150, 120), (250, 118)     # brows above irises
    return p


def test_metrics_similarity_invariant():
    p = _fake_landmarks()
    base = compute_metrics_2d(p)
    th = math.radians(17)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    q = p.copy()
    q[:, :2] = (p[:, :2] @ R.T) * 2.7 + np.array([40.0, -13.0])
    moved = compute_metrics_2d(q)
    for k in base:
        assert moved[k] == pytest.approx(base[k], rel=1e-9), k


def test_brow_height_sign():
    m = compute_metrics_2d(_fake_landmarks())
    assert m["brow_height"] == pytest.approx((30 + 32) / 2 / 100)


def test_tiny_ipd_raises():
    p = _fake_landmarks()
    p[473, :2] = p[468, :2] + 1
    with pytest.raises(ValueError):
        compute_metrics_2d(p)


def test_euler_identity_and_yaw():
    assert euler_from_matrix(np.eye(4)) == pytest.approx((0, 0, 0))
    th = math.radians(20)
    Ry = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]])
    yaw, pitch, roll = euler_from_matrix(3.0 * Ry)   # uniform scale must be removed
    assert yaw == pytest.approx(20) and pitch == pytest.approx(0, abs=1e-9)


def test_holm_known():
    adj = holm(np.array([0.01, 0.04, 0.03, 0.005]))
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_sem_recovers_noise():
    rng = np.random.default_rng(1)
    truth = rng.normal(0, 5, size=(2000, 1))
    obs = truth + rng.normal(0, 0.3, size=(2000, 6))
    assert sem_within(obs) == pytest.approx(0.3, rel=0.03)
    assert mdc95(0.3) == pytest.approx(1.96 * math.sqrt(2) * 0.3)


def test_bootstrap_covers_mean():
    x = np.random.default_rng(2).normal(1.0, 1.0, 400)
    est, lo, hi = bootstrap_ci(x, n_boot=2000)
    assert lo < 1.0 < hi and lo < est < hi


def test_ridge_heldout_r2_known_signal():
    from medsg.stages.stage3b_analysis import ridge_heldout_r2
    rng = np.random.default_rng(3)
    X = rng.normal(size=(3000, 50))
    b = rng.normal(size=50)
    signal = X @ b
    y_clean, y_noise = signal, signal + rng.normal(0, signal.std(), 3000)   # true R² 1.0 and 0.5
    r2, _ = ridge_heldout_r2(X, np.c_[y_clean, y_noise, rng.normal(size=3000)], seed=0)
    assert r2[0] > 0.99 and abs(r2[1] - 0.5) < 0.05 and r2[2] < 0.02


def test_slope_r():
    from medsg.stages.stage2b_analysis import _slope_r
    x = np.random.default_rng(4).normal(size=500)
    s, r = _slope_r(x, 0.6 * x)
    assert s == pytest.approx(0.6) and r == pytest.approx(1.0)


def test_clinical_metrics_signs_and_invariance():
    from medsg.metrics2d import compute_clinical_2d
    p = _fake_landmarks()
    p[159, :2], p[386, :2] = (150, 140), (250, 140)     # upper lids 10 px above iris centres
    p[145, :2], p[374, :2] = (150, 158), (250, 158)     # lower lids 8 px below
    p[107, :2], p[336, :2] = (170, 125), (230, 125)
    p[70, :2], p[300, :2] = (120, 128), (280, 128)
    m = compute_clinical_2d(p)
    assert m["mrd1"] == pytest.approx(0.10) and m["mrd2"] == pytest.approx(0.08)
    assert m["brow_central"] == pytest.approx(compute_metrics_2d(p)["brow_height"])
    assert m["brow_medial"] == pytest.approx(0.25) and m["brow_lateral"] == pytest.approx(0.22)
    th = math.radians(-11)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    q = p.copy(); q[:, :2] = (p[:, :2] @ R.T) * 0.6 + 5
    for k, v in compute_clinical_2d(q).items():
        assert v == pytest.approx(m[k], rel=1e-9), k
