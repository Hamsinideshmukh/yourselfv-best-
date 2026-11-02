"""FFHQ-style face alignment from MediaPipe landmarks (no dlib).

Re-implements the FFHQ crop rule (Karras et al., ffhq-dataset `recreate_aligned_images`):
an oriented square built from the eye centres and mouth corners. dlib's 68 points
are replaced by the equivalent MediaPipe points (eye contour means, mouth corners
61/291). Small systematic differences to dlib-based alignment are expected; the
same function is used for every image, so comparisons stay consistent.
"""
from __future__ import annotations

import cv2
import numpy as np

# MediaPipe equivalents of dlib eye points 36-41 (image-left eye) and 42-47 (image-right eye)
EYE_IMG_LEFT = [33, 160, 158, 133, 153, 144]
EYE_IMG_RIGHT = [362, 385, 387, 263, 373, 380]
MOUTH_L, MOUTH_R = 61, 291


def ffhq_quad(lm_px: np.ndarray) -> np.ndarray:
    """Return the 4 FFHQ quad corners (UL, LL, LR, UR) in pixel coords from (478, >=2) landmarks."""
    p = np.asarray(lm_px, np.float64)[:, :2]
    eye_l, eye_r = p[EYE_IMG_LEFT].mean(0), p[EYE_IMG_RIGHT].mean(0)
    if eye_l[0] > eye_r[0]:
        eye_l, eye_r = eye_r, eye_l
    eye_avg, eye_to_eye = (eye_l + eye_r) / 2, eye_r - eye_l
    mouth_avg = (p[MOUTH_L] + p[MOUTH_R]) / 2
    eye_to_mouth = mouth_avg - eye_avg
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    return np.stack([c - x - y, c - x + y, c + x + y, c + x - y])


def align_ffhq(rgb: np.ndarray, lm_px: np.ndarray, size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Warp an RGB image to an FFHQ-aligned size x size crop.

    Returns (aligned uint8 image, 2x3 affine matrix mapping original pixels -> aligned pixels).
    Out-of-image regions are filled by reflection (as FFHQ does before blurring).
    """
    quad = ffhq_quad(lm_px)
    qsize = np.hypot(*(quad[3] - quad[0]))
    img = rgb
    scale = 1.0
    shrink = int(np.floor(qsize / size * 0.5))
    if shrink > 1:                                  # anti-aliased pre-shrink for large photos
        scale = 1.0 / shrink
        img = cv2.resize(rgb, (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)),
                         interpolation=cv2.INTER_AREA)
    src = (quad[:3] * scale).astype(np.float32)
    dst = np.float32([[0, 0], [0, size], [size, size]])
    M = cv2.getAffineTransform(src, dst)
    out = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    # FFHQ-style padding treatment: blur the reflected region and fade it to the median colour,
    # so mirrored face copies do not confuse the detector or the encoder.
    valid = cv2.warpAffine(np.full(img.shape[:2], 255, np.uint8), M, (size, size),
                           flags=cv2.INTER_NEAREST, borderValue=0)
    if (valid == 0).any():
        dist = cv2.distanceTransform((valid == 0).astype(np.uint8), cv2.DIST_L2, 5)
        m = np.clip(dist / (0.1 * size), 0, 1)[..., None]
        pad = (valid == 0)[..., None]
        f = out.astype(np.float32)
        blurred = cv2.GaussianBlur(f, (0, 0), 0.02 * size)
        f = np.where(pad, blurred, f)
        median = np.median(f[valid > 0], axis=0)
        f = f + (median - f) * m
        out = np.clip(f + 0.5, 0, 255).astype(np.uint8)
    M_full = M.copy()
    M_full[:, :2] *= scale
    return out, M_full
