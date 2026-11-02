"""3D face fit with 3DDFA-V2 (Guo et al., ECCV 2020) on the Basel Face Model (BFM).

Why: HDA pre/post photos differ in head pose and expression. A 3D morphable
model explains an image as  shape (identity, 40 coeffs) + expression (10) + pose,
so measuring on the *canonical* mesh (shape only, frontal, neutral expression)
removes pose and expression from the measurement by construction.

Honest limits:
- 40 shape PCs are a coarse basis; small local surgical changes may not be
  representable. Stage 1 measures this rather than assuming it.
- Depth from one frontal photo is a statistical prior, not a measurement.
  Depth-axis metrics are tagged "exploratory" and never reported as clinical.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .landmarks import euler_from_matrix

# 68-point iBUG indices
_L_EYE, _R_EYE = list(range(36, 42)), list(range(42, 48))

METRICS_3D: dict[str, tuple[str, str, bool]] = {
    # name -> (region, description, exploratory)
    "nose_width_3d": ("nose", "alar base d(31,35)/IED", False),
    "nose_length_3d": ("nose", "nasion (27) to tip (30), /IED", False),
    "nose_projection_3d": ("nose", "tip depth ahead of alar base along face normal, /IED", True),
    "palpebral_height_3d": ("eyelid", "mean lid opening (37-41, 38-40, 43-47, 44-46), /IED", False),
    "palpebral_width_3d": ("eyelid", "mean d(36,39), d(42,45), /IED", False),
    "intercanthal_3d": ("eyelid", "d(39,42)/IED", False),
    "brow_height_3d": ("eyebrow", "mean brow peak (19,24) height above eye centre, /IED", False),
    "face_width_3d": ("face", "d(1,15)/IED", False),
    "jaw_width_3d": ("face", "d(4,12)/IED", False),
    "lower_face_height_3d": ("face", "subnasale (33) to menton (8), /IED", False),
    "chin_projection_3d": ("face", "menton depth relative to subnasale along face normal, /IED", True),
    "mouth_width_3d": ("lips", "d(48,54)/IED", False),
}


@dataclass
class Fit3D:
    param: np.ndarray        # (62,) de-normalised 3DDFA params
    roi_box: list[float]
    yaw: float
    pitch: float
    roll: float
    alpha_shp: np.ndarray    # (40,)
    alpha_exp: np.ndarray    # (10,)
    kpts_img: np.ndarray     # (3, 68) image-space landmarks


class Face3D:
    """3DDFA-V2 MobileNet-v1 (120x120) regressor + BFM; runs on CPU in ~10 ms/face."""

    def __init__(self, repo_dir: Path, gpu: bool = False):
        repo_dir = Path(repo_dir).resolve()
        if not (repo_dir / "TDDFA.py").exists():
            raise FileNotFoundError(f"3DDFA_V2 not found at {repo_dir}")
        # Upstream bfm.py uses np.long, removed in NumPy >= 1.24.
        if not hasattr(np, "long"):
            np.long = np.int64  # type: ignore[attr-defined]
        # Trusted upstream checkpoint containing a plain state_dict.
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        sys.path.insert(0, str(repo_dir))
        try:
            from TDDFA import TDDFA
            from utils.tddfa_util import _parse_param
        finally:
            sys.path.remove(str(repo_dir))
        self._parse = _parse_param
        grad_was = torch.is_grad_enabled()      # TDDFA.__init__ calls torch.set_grad_enabled(False) globally
        self.tddfa = TDDFA(
            arch="mobilenet", widen_factor=1.0, size=120, num_params=62,
            checkpoint_fp=str(repo_dir / "weights/mb1_120x120.pth"),
            bfm_fp=str(repo_dir / "configs/bfm_noneck_v3.pkl"),
            gpu_mode=gpu,
        )
        torch.set_grad_enabled(grad_was)
        for prm in self.tddfa.model.parameters():   # inference only
            prm.requires_grad_(False)
        bfm = self.tddfa.bfm
        assert bfm.w_shp.shape[1] == 40 and bfm.w_exp.shape[1] == 10, (bfm.w_shp.shape, bfm.w_exp.shape)
        assert bfm.u_base.shape == (204, 1), bfm.u_base.shape
        self._u_base, self._w_shp_base = bfm.u_base, bfm.w_shp_base
        # Self-check the canonical frame on the mean face: which axis is "up" and "forward".
        mean = self._kpts(np.zeros((40, 1), np.float32))
        self._up = 1.0 if mean[1, 27] > mean[1, 8] else -1.0          # nasion above menton
        eyes_z = (mean[2, 36] + mean[2, 45]) / 2
        self._fwd = 1.0 if mean[2, 30] > eyes_z else -1.0               # nose tip ahead of eyes
        assert abs(mean[2, 30] - eyes_z) > 0, "degenerate BFM depth axis"

    def _kpts(self, alpha_shp: np.ndarray) -> np.ndarray:
        return (self._u_base + self._w_shp_base @ alpha_shp).reshape(3, -1, order="F")

    @torch.no_grad()
    def fit(self, bgr: np.ndarray, lm_px: np.ndarray) -> Fit3D:
        """Two-pass fit: crop from MediaPipe landmarks, then re-crop from 3DDFA's own 68 points."""
        assert bgr.ndim == 3 and bgr.shape[2] == 3, bgr.shape
        obj = np.asarray(lm_px[:, :2].T, dtype=np.float32)
        params, boxes = self.tddfa(bgr, [obj], crop_policy="landmark")
        k1 = self.tddfa.recon_vers(params, boxes, dense_flag=False)[0]
        params, boxes = self.tddfa(bgr, [k1[:2]], crop_policy="landmark")
        k2 = self.tddfa.recon_vers(params, boxes, dense_flag=False)[0]
        param = params[0]
        assert param.shape == (62,), param.shape
        R, _off, a_shp, a_exp = self._parse(param)
        yaw, pitch, roll = euler_from_matrix(R)
        return Fit3D(param, list(map(float, boxes[0])), yaw, pitch, roll,
                     a_shp.ravel().copy(), a_exp.ravel().copy(), np.asarray(k2))

    def canonical_metrics(self, alpha_shp: np.ndarray) -> dict[str, float]:
        """Measure on the shape-only mesh: frontal, neutral expression, pose removed."""
        k = self._kpts(np.asarray(alpha_shp, np.float32).reshape(40, 1)).astype(np.float64)
        P = k.T                                               # (68, 3)
        le, re_ = P[_L_EYE].mean(0), P[_R_EYE].mean(0)
        ied = float(np.linalg.norm(le - re_))
        d = lambda i, j: float(np.linalg.norm(P[i] - P[j]))
        up = lambda i: self._up * P[i, 1]
        fwd = lambda i: self._fwd * P[i, 2]
        eye_up = self._up * (le[1] + re_[1]) / 2
        m = {
            "nose_width_3d": d(31, 35),
            "nose_length_3d": d(27, 30),
            "nose_projection_3d": fwd(30) - (fwd(31) + fwd(35)) / 2,
            "palpebral_height_3d": (d(37, 41) + d(38, 40) + d(43, 47) + d(44, 46)) / 4,
            "palpebral_width_3d": (d(36, 39) + d(42, 45)) / 2,
            "intercanthal_3d": d(39, 42),
            "brow_height_3d": (up(19) + up(24)) / 2 - eye_up,
            "face_width_3d": d(1, 15),
            "jaw_width_3d": d(4, 12),
            "lower_face_height_3d": d(33, 8),
            "chin_projection_3d": fwd(8) - fwd(33),
            "mouth_width_3d": d(48, 54),
        }
        assert set(m) == set(METRICS_3D)
        return {n: v / ied for n, v in m.items()}

    @torch.no_grad()
    def depth_map(self, fit: Fit3D, hw: tuple[int, int]) -> np.ndarray:
        """Visualisation-only depth image (NaN = background) by z-buffered splatting of the dense mesh."""
        import cv2
        v = self.tddfa.recon_vers([fit.param], [fit.roi_box], dense_flag=True)[0]
        h, w = hw
        x, y, z = np.round(v[0]).astype(int), np.round(v[1]).astype(int), v[2]
        ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        depth = np.full((h, w), -np.inf, np.float32)
        np.maximum.at(depth, (y[ok], x[ok]), z[ok].astype(np.float32))
        empty = ~np.isfinite(depth)
        filled = cv2.dilate(depth, np.ones((5, 5), np.uint8))   # cv2 maps -inf to -FLT_MAX
        hole = empty & (filled > -1e30)
        depth[hole] = filled[hole]
        depth[empty & ~hole] = np.nan
        return depth
