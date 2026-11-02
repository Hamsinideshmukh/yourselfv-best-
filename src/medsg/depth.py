"""Depth as an independent check of edits: a dense monocular depth network + region masks.

Two depth sources are used in this project, for different jobs:
- 3DDFA-V2 (face3d.py): fits a 3D face model; its canonical mesh removes head pose and expression, so it gives
  pose-robust shape measurements. Coarse (40 shape components).
- Depth Anything V2 Small (Yang et al., NeurIPS 2024; Apache-2.0; 24.8M params), here: a dense per-pixel relative
  depth map from one photo. Used to ask "where in 3D did the edit change the face?" — if a nose edit changes depth
  mostly around the nose and not the cheeks or jaw, the edit is local in a modality the editor never optimised.
Honest limits: depth from a single photo is a learned estimate, relative (unknown scale and offset), never a
measurement in mm. Profile quantities (tip projection, chin projection) are NOT reported from it (not valid from frontal photos).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

DA2_ID = "depth-anything/Depth-Anything-V2-Small-hf"

# MediaPipe landmarks that define each target region (metrics2d / CLINICAL_2D points + their anatomical neighbours)
REGION = {
    "nose_width": [129, 358, 98, 327, 64, 294, 4, 2, 5, 195],
    "brow_height": [70, 63, 105, 66, 107, 336, 296, 334, 293, 300],
    "palpebral_height": [159, 145, 386, 374, 33, 133, 263, 362],
    "mrd1": [159, 145, 386, 374, 33, 133, 263, 362],
    "jaw_width": [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397],
    "mouth_width": [61, 291, 0, 17, 13, 14, 37, 267, 84, 314],
}
RADIUS_IPD = 0.25                     # disc radius around each region landmark, in inter-pupil distances


class DepthNet:
    """Depth Anything V2 Small via transformers. `__call__(rgb)` -> (H, W) float32 relative depth (larger = nearer)."""

    def __init__(self, device: str):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.proc = AutoImageProcessor.from_pretrained(DA2_ID)
        self.net = AutoModelForDepthEstimation.from_pretrained(DA2_ID).eval().to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        x = self.proc(images=rgb, return_tensors="pt")["pixel_values"].to(self.device)
        d = self.net(pixel_values=x).predicted_depth[:, None]
        return F.interpolate(d, size=rgb.shape[:2], mode="bicubic", align_corners=False)[0, 0].cpu().numpy()


def masks(L: np.ndarray, target: str, hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(region mask, whole-face mask) as bool arrays. Face = convex hull of all 478 landmarks."""
    h, w = hw
    face = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(face, cv2.convexHull(L[:, :2].astype(np.int32)), 1)
    reg = np.zeros((h, w), np.uint8)
    r = int(round(RADIUS_IPD * np.linalg.norm(L[468, :2] - L[473, :2])))
    for i in REGION[target]:
        cv2.circle(reg, tuple(int(v) for v in L[i, :2]), r, 1, -1)
    return (reg & face).astype(bool), face.astype(bool)


def depth_change(d0: np.ndarray, d1: np.ndarray, L: np.ndarray, target: str) -> dict:
    """Where did depth change? d1 is first mapped onto d0's scale/offset (least squares over the face), since
    monocular depth is only relative. Returns the change size inside / outside the target region (in units of the
    face's own depth range) and the locality ratio = (share of change inside) / (share of face area inside);
    1 = spread evenly over the face, > 1 = concentrated where the surgery is."""
    reg, face = masks(L, target, d0.shape)
    A = np.c_[d1[face], np.ones(face.sum())]
    s, o = np.linalg.lstsq(A, d0[face], rcond=None)[0]
    delta = np.abs(s * d1 + o - d0)
    span = float(np.percentile(d0[face], 95) - np.percentile(d0[face], 5)) or 1.0
    inside, outside = delta[reg], delta[face & ~reg]
    share = float(inside.sum() / max(delta[face].sum(), 1e-12))
    area = float(reg.sum() / max(face.sum(), 1))
    return {"depth_in": float(inside.mean() / span), "depth_out": float(outside.mean() / span),
            "depth_share_in": share, "depth_area_in": area, "depth_locality": share / max(area, 1e-12)}
