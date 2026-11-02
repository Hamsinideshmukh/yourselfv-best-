"""Frontal-valid 2D facial ratios from MediaPipe landmarks.

Every metric is a distance divided by the inter-iris distance (IPD), so it is
unitless and invariant to image scale, translation and in-plane rotation.
Iris centres are used because no common facial surgery moves them (eye
corners, the v1 normaliser, can move with canthal/eyelid surgery).
Left/right metrics are averaged so that a horizontal flip gives the same value.

Deliberately excluded (profile-only, invalid from frontal photos): nasolabial
angle, nose-tip projection, chin projection, and anything using MediaPipe z.
"""
from __future__ import annotations

import numpy as np

IRIS = (468, 473)

# name -> (procedure region, description). Landmark indices follow the MediaPipe
# canonical face mesh; anatomical names are approximate (a mesh vertex is not
# a palpated anthropometric point).
METRICS_2D: dict[str, tuple[str, str]] = {
    # Key kept as "nose_width" so all stored outputs stay valid; report it as DISPLAY_NAMES["nose_width"].
    # Audit 2026-10-30: 129/358 lie on the alar-facial junction, ~39.8 mm approx.,
    # wider than published al-al norms (31.4 F / 34.7 M mm), so it is not anatomical al-al.
    "nose_width": ("nose", "alar-junction width, d(129,358)/IPD (not al-al)"),
    "nose_length": ("nose", "nasal bridge (6) to tip (4), /IPD"),
    "palpebral_height": ("eyelid", "mean eye opening d(159,145), d(386,374), /IPD"),
    "palpebral_width": ("eyelid", "mean eye width d(33,133), d(263,362), /IPD"),
    "intercanthal": ("eyelid", "inner-corner distance d(133,362)/IPD"),
    "brow_height": ("eyebrow", "mean brow-peak (105,334) height above iris centre, /IPD"),
    "face_width": ("face", "bizygomatic proxy d(234,454)/IPD"),
    "jaw_width": ("face", "lower jaw width d(172,397)/IPD"),
    "lower_face_height": ("face", "subnasale (2) to chin (152), /IPD"),
    "mouth_width": ("lips", "commissure distance d(61,291)/IPD"),
}


def _d(p: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(p[i] - p[j]))


def compute_metrics_2d(lm_px: np.ndarray) -> dict[str, float]:
    """Compute METRICS_2D from (478, >=2) pixel landmarks (only x, y are used)."""
    assert lm_px.shape[0] == 478, lm_px.shape
    p = np.asarray(lm_px, dtype=np.float64)[:, :2]
    iris_a, iris_b = p[IRIS[0]], p[IRIS[1]]
    ipd = float(np.linalg.norm(iris_a - iris_b))
    if ipd < 5.0:
        raise ValueError(f"inter-iris distance {ipd:.1f}px is implausibly small")
    # Face-vertical unit vector: perpendicular to the iris line, pointing towards the chin.
    u = (iris_b - iris_a) / ipd
    v = np.array([-u[1], u[0]])
    if np.dot(p[152] - (iris_a + iris_b) / 2, v) < 0:
        v = -v

    def brow(idx: int) -> float:
        iris = iris_a if np.linalg.norm(p[idx] - iris_a) < np.linalg.norm(p[idx] - iris_b) else iris_b
        return float(np.dot(iris - p[idx], v))      # positive = brow above iris

    m = {
        "nose_width": _d(p, 129, 358),
        "nose_length": _d(p, 6, 4),
        "palpebral_height": (_d(p, 159, 145) + _d(p, 386, 374)) / 2,
        "palpebral_width": (_d(p, 33, 133) + _d(p, 263, 362)) / 2,
        "intercanthal": _d(p, 133, 362),
        "brow_height": (brow(105) + brow(334)) / 2,
        "face_width": _d(p, 234, 454),
        "jaw_width": _d(p, 172, 397),
        "lower_face_height": _d(p, 2, 152),
        "mouth_width": _d(p, 61, 291),
    }
    assert set(m) == set(METRICS_2D)
    return {k: v_ / ipd for k, v_ in m.items()}


DISPLAY_NAMES = {"nose_width": "alar-junction width"}

# Clinically standard frontal measures (user-approved metric-set addition, 2026-10-31; A3).
# Landmarks audited on HDA photos: brow lateral 70/300, central 105/334, medial 107/336 lie on the brow;
# 159/386 upper-lid margin, 145/374 lower-lid margin. The iris centre stands in for the pupil / corneal light
# reflex, so MRD1/MRD2 here are *proxies* of the clinical margin reflex distances (Bodner et al., JAMA Ophthalmol).
# Multi-point brow height follows the validated multi-point photographic approach (Verduijn et al.).
CLINICAL_2D: dict[str, tuple[str, str]] = {
    "mrd1": ("eyelid", "iris centre to upper-lid margin (159/386), /IPD; MRD1 proxy"),
    "mrd2": ("eyelid", "iris centre to lower-lid margin (145/374), /IPD; MRD2 proxy"),
    "brow_medial": ("eyebrow", "brow head (107/336) height above iris centre, /IPD"),
    "brow_central": ("eyebrow", "brow peak (105/334) height above iris centre, /IPD (= brow_height)"),
    "brow_lateral": ("eyebrow", "brow tail (70/300) height above iris centre, /IPD"),
}
_PAIRS = {"mrd1": (159, 386), "mrd2": (145, 374), "brow_medial": (107, 336),
          "brow_central": (105, 334), "brow_lateral": (70, 300)}


def compute_clinical_2d(lm_px: np.ndarray) -> dict[str, float]:
    """CLINICAL_2D from (478, >=2) pixel landmarks. Heights along the face-vertical axis, nearest iris per side."""
    assert lm_px.shape[0] == 478, lm_px.shape
    p = np.asarray(lm_px, dtype=np.float64)[:, :2]
    a, b = p[IRIS[0]], p[IRIS[1]]
    ipd = float(np.linalg.norm(a - b))
    if ipd < 5.0:
        raise ValueError(f"inter-iris distance {ipd:.1f}px is implausibly small")
    u = (b - a) / ipd
    v = np.array([-u[1], u[0]])
    if np.dot(p[152] - (a + b) / 2, v) < 0:
        v = -v                                    # v points towards the chin

    def above(idx: int) -> float:                 # >0 when the point is above its iris centre
        iris = a if np.linalg.norm(p[idx] - a) < np.linalg.norm(p[idx] - b) else b
        return float(np.dot(iris - p[idx], v))
    out = {}
    for k, (i, j) in _PAIRS.items():
        h = (above(i) + above(j)) / 2
        out[k] = (-h if k == "mrd2" else h) / ipd     # MRD2: lower lid below iris -> positive
    return out
