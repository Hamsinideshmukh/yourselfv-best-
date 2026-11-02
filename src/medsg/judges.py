"""Independent judges: ArcFace identity and FAN 68-point landmarks.

Why independent: the editor is steered by MediaPipe and e4e was trained with an IR-SE50 ArcFace loss, so
judging with either would flatter the method. ArcFace here = insightface buffalo_l `w600k_r50.onnx`
(ResNet-50, WebFace600K; non-commercial research licence), run with onnxruntime on CPU. FAN = 2D-FAN
(Bulat & Tzimiropoulos 2017; `face-alignment` package, BSD-3).
If a judge cannot be loaded, the caller records it as unavailable; nothing is substituted.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# insightface 112x112 ArcFace template: image-left eye, image-right eye, nose tip, mouth left, mouth right
ARC_TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                         [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)


def five_points(lm478: np.ndarray) -> np.ndarray:
    """ArcFace 5 points from MediaPipe pixel landmarks (iris centres, nose tip 4, mouth corners 61/291)."""
    eyes = sorted([lm478[468, :2], lm478[473, :2]], key=lambda p: p[0])
    mouth = sorted([lm478[61, :2], lm478[291, :2]], key=lambda p: p[0])
    return np.array([eyes[0], eyes[1], lm478[4, :2], mouth[0], mouth[1]], np.float32)


class ArcFace:
    def __init__(self, onnx_path: Path):
        import onnxruntime as ort
        if not Path(onnx_path).exists():
            raise FileNotFoundError(onnx_path)
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def embed(self, rgb: np.ndarray, lm478_px: np.ndarray) -> np.ndarray:
        """RGB uint8 image + its MediaPipe pixel landmarks -> L2-normalised (512,) embedding."""
        M, _ = cv2.estimateAffinePartial2D(five_points(lm478_px), ARC_TEMPLATE, method=cv2.LMEDS)
        if M is None:
            raise ValueError("ArcFace alignment failed")
        crop = cv2.warpAffine(rgb, M, (112, 112), borderValue=0.0)
        x = ((crop.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        e = self.sess.run(None, {self.inp: x})[0][0].astype(np.float64)
        return e / np.linalg.norm(e)


def face_box(lm478_px: np.ndarray) -> np.ndarray:
    """Tight face box (x1, y1, x2, y2) from MediaPipe landmarks, the input FAN expects instead of a detector box."""
    x1, y1 = lm478_px[:, :2].min(0)
    x2, y2 = lm478_px[:, :2].max(0)
    return np.array([x1, y1, x2, y2], np.float64)


class FAN68:
    def __init__(self, device: str, face_detector: str = "sfd"):
        """face_detector is never used for landmarks (callers pass the face box) but the package loads one;
        the app uses the small "blazeface" to avoid an 85 MB download."""
        import face_alignment
        lt = getattr(face_alignment.LandmarksType, "TWO_D", None) or getattr(face_alignment.LandmarksType, "_2D")
        self.fa = face_alignment.FaceAlignment(lt, device=device, flip_input=False, face_detector=face_detector)

    def landmarks(self, rgb: np.ndarray, box: np.ndarray) -> np.ndarray:
        """(68, 2) pixel landmarks for the face inside `box`."""
        out = self.fa.get_landmarks_from_image(rgb, detected_faces=[np.r_[box, 1.0]])
        if not out:
            raise ValueError("FAN returned no landmarks")
        return np.asarray(out[0][:, :2], np.float64)


FAN_METRICS = ["nose_width", "mouth_width", "jaw_width", "lower_face_height", "brow_height", "palpebral_height"]


def metrics_fan(p: np.ndarray) -> dict[str, float]:
    """FAN-68 versions of the frontal metrics, normalised by the eye-centre distance (no iris in 68 points).

    nose_width = alar base d(31,35); mouth d(48,54); jaw d(4,12); lower face d(33,8) (subnasale-chin);
    brow = brow-peak (19/24) height above the eye centre; palpebral = mean lid gaps (37-41, 38-40, 43-47, 44-46).
    """
    assert p.shape == (68, 2), p.shape
    el, er = p[36:42].mean(0), p[42:48].mean(0)
    ied = float(np.linalg.norm(er - el))
    u = (er - el) / ied
    v = np.array([-u[1], u[0]])
    if np.dot(p[8] - (el + er) / 2, v) < 0:
        v = -v

    def d(i, j):
        return float(np.linalg.norm(p[i] - p[j]))
    m = {"nose_width": d(31, 35), "mouth_width": d(48, 54), "jaw_width": d(4, 12), "lower_face_height": d(33, 8),
         "brow_height": (float(np.dot(el - p[19], v)) + float(np.dot(er - p[24], v))) / 2,
         "palpebral_height": (d(37, 41) + d(38, 40) + d(43, 47) + d(44, 46)) / 4}
    return {k: x / ied for k, x in m.items()}
