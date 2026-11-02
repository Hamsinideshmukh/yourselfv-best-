"""Deformation transfer (A1, approved 2026-10-31): the patient's own photo, warped by an exactly computed field.

Plain words
1. Shape of the change: the GAN population pattern for the target metric (assets/patterns_v1.npz) says how every
   MediaPipe landmark moves per +1 unit of that metric (E7: best or tied prior on held-out real surgery).
2. Keep everything else fixed, exactly: metrics are explicit formulas of landmark positions, so their sensitivity
   to each landmark (the gradient) is computable. Removing from the pattern every component that would change a
   protected metric (projection onto the null space of their gradients) leaves those metrics unchanged to first
   order. The iris centres are pinned the same way, so the IPD normaliser cannot move.
3. Size: the amplitude a is solved so that the target metric, computed on the moved landmarks, changes by exactly
   the request (1-D root finding on the exact formula; no measurement noise involved).
4. Photo: the landmark motion is spread over the image with a thin-plate spline (Bookstein 1989) anchored to zero
   at the image border, and the photo is resampled (backward map with one fixed-point refinement).
Output: warped photo + the exact "recipe" (per-landmark motion, approx. mm via mean IPD 63.36 mm, ±6 %) + exact
before/after metrics. Zero request -> the input photo unchanged (identity field).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .deform import face_axes
from .metrics2d import CLINICAL_2D, METRICS_2D, compute_clinical_2d, compute_metrics_2d

ASSET = Path(__file__).resolve().parent / "assets" / "patterns_v1.npz"
IPD_MM = 63.36                      # adult mean IPD (Dodgson 2004); approx. mm only, ±6 % per person
NAMES = list(METRICS_2D) + list(CLINICAL_2D)
IRIS_RING = list(range(468, 478))
# Declared couplings (not protected when editing the key); extends editspec.COUPLED to the clinical metrics.
COUPLED = {
    "brow_height": ["palpebral_height", "brow_central", "brow_medial", "brow_lateral", "mrd1"],
    "brow_central": ["brow_height", "brow_medial", "brow_lateral", "palpebral_height", "mrd1"],
    "brow_medial": ["brow_height", "brow_central", "brow_lateral"],
    "brow_lateral": ["brow_height", "brow_central", "brow_medial"],
    "palpebral_height": ["brow_height", "palpebral_width", "mrd1", "mrd2", "brow_central"],
    "mrd1": ["palpebral_height", "brow_height", "brow_central"],
    "mrd2": ["palpebral_height"],
    "palpebral_width": ["palpebral_height", "intercanthal"],
    "intercanthal": ["palpebral_width"],
    "nose_length": ["lower_face_height"],
    "lower_face_height": ["nose_length"],
    "face_width": ["jaw_width"],
    "jaw_width": ["face_width"],
    "nose_width": [],
    "mouth_width": [],
}


def metric_values(L: np.ndarray) -> np.ndarray:
    """All 15 metrics (frontal + clinical) of (478, 2) pixel landmarks."""
    p = np.c_[L, np.zeros(len(L))]
    return np.array([*compute_metrics_2d(p).values(), *compute_clinical_2d(p).values()])


_ACTIVE: np.ndarray | None = None          # coordinates any metric depends on (found once from a full gradient)


def metric_gradients(L: np.ndarray, eps: float = 0.05) -> np.ndarray:
    """(15, 956) d metric / d landmark coordinate, central differences (metrics are smooth explicit formulas).
    The first call differentiates all 956 coordinates and records which ones matter; later calls only those."""
    global _ACTIVE
    flat = L.ravel().astype(np.float64)
    G = np.zeros((len(NAMES), flat.size))
    cols = range(flat.size) if _ACTIVE is None else _ACTIVE
    for j in cols:
        up, dn = flat.copy(), flat.copy()
        up[j] += eps
        dn[j] -= eps
        G[:, j] = (metric_values(up.reshape(L.shape)) - metric_values(dn.reshape(L.shape))) / (2 * eps)
    if _ACTIVE is None:
        _ACTIVE = np.flatnonzero(np.abs(G).max(0) > 0)
    return G


@dataclass
class Edit:
    target: str
    request: float              # in IPD units (metric units)
    field_px: np.ndarray        # (478, 2) landmark motion in pixels
    before: dict
    after: dict                 # exact metrics on the moved landmarks
    amplitude: float
    kappa: float                # share of the pattern's effect on the target left after protection

    def recipe_mm(self, L: np.ndarray) -> np.ndarray:
        """Per-landmark motion in approx. mm (±6 %), using the photo's IPD as 63.36 mm."""
        ipd = face_axes(L)[3]
        return self.field_px * IPD_MM / ipd


class Transfer:
    def __init__(self, asset: Path = ASSET):
        z = np.load(asset)
        assert list(z["names"]) == NAMES, "pattern asset metric order differs from metrics2d"
        self.P = z["P"].astype(np.float64)                         # (15, 956) iris-frame units per +1 metric unit

    def edit(self, L: np.ndarray, target: str, request: float, protect: list[str] | None = None) -> Edit:
        """Exact constrained landmark field for one request on landmarks L (478, 2) in pixels."""
        assert L.shape == (478, 2), L.shape
        L = L.astype(np.float64)
        c, u, v, ipd = face_axes(L)
        Pf = self.P[NAMES.index(target)].reshape(478, 2)
        F = ipd * (Pf[:, :1] * u + Pf[:, 1:] * v)                    # iris-frame pattern -> pixel motion
        G = metric_gradients(L)
        if protect is None:
            protect = [m for m in NAMES if m != target and m not in COUPLED.get(target, [])]
        rows = [G[NAMES.index(m)] for m in protect]
        for i in IRIS_RING:                                           # pin the iris ring (x and y)
            for k in range(2):
                e = np.zeros(956)
                e[2 * i + k] = 1.0
                rows.append(e / ipd)
        A = np.array(rows)
        f = F.ravel()
        Ga = A @ A.T
        f_p = f - A.T @ np.linalg.solve(Ga + 1e-12 * np.trace(Ga) * np.eye(len(Ga)), A @ f)
        g_t = G[NAMES.index(target)]
        kappa = float((g_t @ f_p) / (g_t @ f)) if abs(g_t @ f) > 0 else 0.0
        if abs(g_t @ f_p) < 1e-9:
            raise ValueError(f"{target}: no change left after protecting {len(protect)} metrics (entangled)")
        before = metric_values(L)
        ti = NAMES.index(target)

        def resid(a: float) -> float:
            return metric_values(L + a * f_p.reshape(478, 2))[ti] - before[ti] - request
        a0, a1 = 0.0, request / (g_t @ f_p)
        r0, r1 = -request, resid(a1)
        for _ in range(30):                                           # secant on the exact formula
            if abs(r1) < 1e-10 or r1 == r0:
                break
            a0, a1, r0 = a1, a1 - r1 * (a1 - a0) / (r1 - r0), r1
            r1 = resid(a1)
        if abs(r1) > 1e-6:
            raise ValueError(f"{target}: amplitude solve did not converge (residual {r1:.2e})")
        field = a1 * f_p.reshape(478, 2)
        after = metric_values(L + field)
        return Edit(target, request, field, dict(zip(NAMES, before)), dict(zip(NAMES, after)), float(a1), kappa)


def dense_field(L: np.ndarray, field: np.ndarray, hw: tuple[int, int], step: int = 8) -> np.ndarray:
    """Thin-plate-spline interpolation of landmark motion to every pixel; zero on the image border."""
    from scipy.interpolate import RBFInterpolator
    h, w = hw
    bx = np.r_[np.linspace(0, w - 1, 9), np.linspace(0, w - 1, 9), np.zeros(9), np.full(9, w - 1)]
    by = np.r_[np.zeros(9), np.full(9, h - 1), np.linspace(0, h - 1, 9), np.linspace(0, h - 1, 9)]
    pts = np.r_[L, np.c_[bx, by]]
    vals = np.r_[field, np.zeros((len(bx), 2))]
    rbf = RBFInterpolator(pts, vals, kernel="thin_plate_spline", smoothing=1e-3)
    gy, gx = np.mgrid[0:h:step, 0:w:step]
    coarse = rbf(np.c_[gx.ravel(), gy.ravel()]).reshape(gy.shape + (2,)).astype(np.float32)
    return cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)


def warp_photo(rgb: np.ndarray, L: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Forward-warp the photo so each landmark moves by `field` (backward map, one fixed-point refinement)."""
    h, w = rgb.shape[:2]
    if not np.any(field):
        return rgb.copy()
    D = dense_field(L, field, (h, w))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    D2 = cv2.remap(D, xx - D[..., 0], yy - D[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.remap(rgb, xx - D2[..., 0], yy - D2[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def _projected_patterns(self_P, L, targets, protect_extra=(), couplings: dict | None = None):
    c, u, v, ipd = face_axes(L)
    G = metric_gradients(L)
    cp = COUPLED if couplings is None else couplings
    coupled = set().union(*[cp.get(t, []) for t in targets]) if targets else set()
    protect = [m for m in NAMES if m not in targets and m not in coupled] + list(protect_extra)
    rows = [G[NAMES.index(m)] for m in protect]
    for i in IRIS_RING:
        for k in range(2):
            e = np.zeros(956)
            e[2 * i + k] = 1.0
            rows.append(e / ipd)
    A = np.array(rows)
    Ga = A @ A.T
    W = np.linalg.solve(Ga + 1e-12 * np.trace(Ga) * np.eye(len(Ga)), A)
    fs = []
    for t in targets:
        Pf = self_P[NAMES.index(t)].reshape(478, 2)
        f = (ipd * (Pf[:, :1] * u + Pf[:, 1:] * v)).ravel()
        fs.append(f - A.T @ (W @ f))
    return np.array(fs), G, protect


def edit_multi(self, L: np.ndarray, requests: dict[str, float], couplings: dict | None = None) -> Edit:
    """Several targets at once: each target's pattern is stripped of effects on protected metrics, then the
    amplitudes are solved jointly so every target hits its request exactly (Newton on the exact formulas).
    Targets that the face model ties together (near-singular system) raise ValueError instead of a silent compromise."""
    L = np.asarray(L, np.float64)
    assert L.shape == (478, 2), L.shape
    targets = [t for t, r in requests.items() if r != 0.0]
    before = metric_values(L)
    if not targets:
        z = dict(zip(NAMES, before))
        return Edit("", 0.0, np.zeros((478, 2)), z, dict(z), 0.0, 1.0)
    F, G, _ = _projected_patterns(self.P, L, targets, couplings=couplings)
    idx = [NAMES.index(t) for t in targets]
    r = np.array([requests[t] for t in targets])
    M = G[idx] @ F.T                                   # (k, k): target i change per unit amplitude j
    if np.linalg.cond(M) > 1e6:
        raise ValueError(f"targets {targets} are entangled for this face (cannot be set independently)")
    a = np.linalg.solve(M, r)
    for _ in range(30):
        field = (a @ F).reshape(478, 2)
        res = metric_values(L + field)[idx] - before[idx] - r
        if np.abs(res).max() < 1e-9:
            break
        a = a - np.linalg.solve(M, res)
    if np.abs(res).max() > 1e-6:
        raise ValueError(f"joint amplitude solve did not converge (residual {np.abs(res).max():.1e})")
    after = metric_values(L + field)
    return Edit("+".join(targets), float(np.abs(r).max()), field, dict(zip(NAMES, before)), dict(zip(NAMES, after)),
                float(np.abs(a).max()), 1.0)


Transfer.edit_multi = edit_multi
