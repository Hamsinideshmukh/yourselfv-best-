"""Torch-free MediaPipe measurement helpers shared by the GPU stage (Stage 5) and CPU evaluations (Stage 6).

measure     : one image -> landmarks (normalised to [0,1] at 512 px), pose, blendshapes, 15 ratios (10 frontal + 5 clinical).
measure_avg : mean ratios over k = 4 harmless variants (original, mirror, +-1 % shift), which roughly halves the
              landmark jitter; ratios are left/right averaged, so the mirrored photo measures the same quantities.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import editspec as E
from .landmarks import FaceNotFound
from .metrics2d import CLINICAL_2D, METRICS_2D, compute_clinical_2d, compute_metrics_2d

M10, C5 = list(METRICS_2D), list(CLINICAL_2D)
MCOLS = M10 + C5
MEAS_RES = 512
BLENDSHAPES = ["jawOpen", "mouthSmileLeft", "mouthSmileRight", "eyeBlinkLeft", "eyeBlinkRight",
               "eyeSquintLeft", "eyeSquintRight", "browInnerUp", "browDownLeft", "browDownRight"]   # = Stage 1 order


def measure(rgb: np.ndarray, lmk) -> dict | None:
    """MediaPipe at 512 px: landmarks (normalised), pose, blendshapes, 15 ratios. None if no usable face."""
    small = rgb if rgb.shape[:2] == (MEAS_RES, MEAS_RES) else cv2.resize(rgb, (MEAS_RES, MEAS_RES),
                                                                         interpolation=cv2.INTER_AREA)
    try:
        obs = lmk(np.ascontiguousarray(small))
        lm = obs.lm_px[:, :2]
        m, c = compute_metrics_2d(lm), compute_clinical_2d(lm)
    except (FaceNotFound, ValueError):
        return None
    return {"lm": (lm / MEAS_RES).astype(np.float32), "pose": np.array([obs.yaw, obs.pitch, obs.roll], np.float32),
            "bs": np.array([obs.blendshapes.get(b, np.nan) for b in BLENDSHAPES], np.float32),
            "m": np.array([m[k] for k in M10] + [c[k] for k in C5], np.float64)}


def _shift(img: np.ndarray, frac: float) -> np.ndarray:
    M = np.float32([[1, 0, frac * img.shape[1]], [0, 1, 0]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def measure_avg(rgb: np.ndarray, lmk) -> tuple[np.ndarray | None, dict | None]:
    """Mean ratios over k = 4 harmless variants (editspec LOOP_K_AVG); needs >= 3 usable variants."""
    small = cv2.resize(rgb, (MEAS_RES, MEAS_RES), interpolation=cv2.INTER_AREA) if rgb.shape[0] != MEAS_RES else rgb
    first = measure(small, lmk)
    vals = [first["m"]] if first else []
    for v in (cv2.flip(small, 1), _shift(small, 0.01), _shift(small, -0.01)):
        r = measure(v, lmk)
        if r:
            vals.append(r["m"])
    assert E.LOOP_K_AVG == 4
    if first is None or len(vals) < 3:
        return None, None
    return np.mean(vals, 0), first
