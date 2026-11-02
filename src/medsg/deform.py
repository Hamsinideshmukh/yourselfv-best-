"""Smooth image deformations with exactly known landmark motion (phantom tests; base for deformation transfer).

Plain words
- A displacement field says, for every image point, where it moves (dx, dy in pixels).
- Gaussian field: a few control points each move by a vector; nearby points move by the same vector scaled
  by exp(-distance² / 2σ²), so motion fades smoothly with distance.
- Warping an image needs the *backward* map (for each output pixel, where it came from). For small smooth
  fields, source ≈ x − D(x − D(x)) (one fixed-point refinement), which is accurate to second order.
- Truth for a phantom: the metric computed on landmarks moved by the same field. So the "measured / true"
  ratio (gain) tells how faithfully a landmarker reports a known change.
"""
from __future__ import annotations

import cv2
import numpy as np

# Phantom specs: metric -> {MediaPipe landmark: (x, y) unit motion in the face frame}.
# x = "outward" (away from the face midline, sign chosen per side), y = +1 towards the chin.
PHANTOM_SPECS: dict[str, dict[int, tuple[float, float]]] = {
    "nose_width": {129: (1, 0), 358: (1, 0)},
    "mouth_width": {61: (1, 0), 291: (1, 0)},
    "jaw_width": {172: (1, 0), 397: (1, 0)},
    "lower_face_height": {152: (0, 1)},
    "brow_height": {105: (0, -1), 334: (0, -1)},
    "brow_lateral": {70: (0, -1), 300: (0, -1)},
    "palpebral_height": {159: (0, -0.5), 386: (0, -0.5), 145: (0, 0.5), 374: (0, 0.5)},
    "mrd1": {159: (0, -1), 386: (0, -1)},
}


def face_axes(lm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Iris midpoint c, unit x-axis u (along the iris line), unit v (towards the chin), IPD in pixels."""
    a, b = lm[468, :2].astype(np.float64), lm[473, :2].astype(np.float64)
    ipd = float(np.linalg.norm(b - a))
    u = (b - a) / ipd
    v = np.array([-u[1], u[0]])
    if np.dot(lm[152, :2] - (a + b) / 2, v) < 0:
        v = -v
    return (a + b) / 2, u, v, ipd


def phantom_controls(lm: np.ndarray, metric: str, amp_ipd: float) -> tuple[np.ndarray, np.ndarray]:
    """Control points (k,2) and displacement vectors (k,2) in pixels for a phantom of size amp_ipd (IPD units)."""
    c, u, v, ipd = face_axes(lm)
    pts, vecs = [], []
    for i, (sx, sy) in PHANTOM_SPECS[metric].items():
        p = lm[i, :2].astype(np.float64)
        side = np.sign(np.dot(p - c, u)) or 1.0
        pts.append(p)
        vecs.append(amp_ipd * ipd * (sx * side * u + sy * v))
    return np.array(pts), np.array(vecs)


def gaussian_field(points: np.ndarray, ctrl: np.ndarray, vecs: np.ndarray, sigma_px: float) -> np.ndarray:
    """Displacement (..., 2) at `points` (..., 2) from Gaussian-weighted control vectors."""
    d = np.zeros(points.shape[:-1] + (2,), np.float64)
    s2 = 2.0 * sigma_px ** 2
    for p, vec in zip(ctrl, vecs):
        w = np.exp(-((points - p) ** 2).sum(-1) / s2)
        d += w[..., None] * vec
    return d


def warp_gaussian(rgb: np.ndarray, ctrl: np.ndarray, vecs: np.ndarray, sigma_px: float) -> np.ndarray:
    """Forward-warp an image by the Gaussian field (backward map with one fixed-point refinement)."""
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    grid = np.stack([xx, yy], -1)
    D = gaussian_field(grid, ctrl, vecs, sigma_px)
    src = grid - gaussian_field(grid - D, ctrl, vecs, sigma_px)
    return cv2.remap(rgb, src[..., 0].astype(np.float32), src[..., 1].astype(np.float32), cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REFLECT)


def moved_points(pts: np.ndarray, ctrl: np.ndarray, vecs: np.ndarray, sigma_px: float) -> np.ndarray:
    """Exact positions of `pts` after the forward field (the phantom's ground truth)."""
    return pts + gaussian_field(pts, ctrl, vecs, sigma_px)
