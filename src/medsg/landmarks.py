"""MediaPipe FaceLandmarker (Tasks API): 478 landmarks incl. iris, head pose, 52 blendshapes.

Coordinates are converted to pixels with the image's own width/height, so aspect
ratio is never distorted (the v1 notebook resized every image to 512x512).
"""
from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import MEDIAPIPE_TASK_URL


class FaceNotFound(RuntimeError):
    """Raised when no face is detected; callers must record it, never substitute output."""


@dataclass
class FaceObs:
    lm_px: np.ndarray          # (478, 3): x, y in pixels; z in MediaPipe relative units (not metric)
    yaw: float                 # degrees, from the facial transformation matrix
    pitch: float
    roll: float
    blendshapes: dict[str, float]
    n_faces: int
    img_hw: tuple[int, int]


def ensure_task_file(models_dir: Path) -> Path:
    """Download the official face_landmarker.task once."""
    path = models_dir / "face_landmarker.task"
    if not path.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MEDIAPIPE_TASK_URL, path)
    if path.stat().st_size < 1_000_000:
        raise RuntimeError(f"{path} looks truncated ({path.stat().st_size} bytes)")
    return path


def euler_from_matrix(m: np.ndarray) -> tuple[float, float, float]:
    """Yaw, pitch, roll in degrees from a 4x4 (or 3x3) transform, removing uniform scale."""
    r = np.asarray(m, dtype=np.float64)[:3, :3]
    r = r / np.cbrt(abs(np.linalg.det(r)))
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2])))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return yaw, pitch, roll


class Landmarker:
    """Thin wrapper; one instance per process."""

    def __init__(self, task_path: Path, max_faces: int = 3):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(task_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_faces,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.5,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)

    def close(self) -> None:
        """Release the graph before interpreter shutdown (avoids a noisy __del__ traceback)."""
        self._lm.close()

    def __call__(self, rgb: np.ndarray) -> FaceObs:
        """Detect the largest face in an RGB uint8 image. Raises FaceNotFound."""
        assert rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8, rgb.shape
        h, w = rgb.shape[:2]
        res = self._lm.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                             data=np.ascontiguousarray(rgb)))
        if not res.face_landmarks:
            raise FaceNotFound("MediaPipe found no face")
        arrs = [np.array([[p.x * w, p.y * h, p.z * w] for p in f], dtype=np.float64)
                for f in res.face_landmarks]
        areas = [np.ptp(a[:, 0]) * np.ptp(a[:, 1]) for a in arrs]
        i = int(np.argmax(areas))
        lm = arrs[i]
        assert lm.shape == (478, 3), f"expected 478 landmarks (iris model), got {lm.shape}"
        yaw, pitch, roll = euler_from_matrix(res.facial_transformation_matrixes[i])
        bs = {c.category_name: float(c.score) for c in res.face_blendshapes[i]}
        return FaceObs(lm, yaw, pitch, roll, bs, len(arrs), (h, w))
