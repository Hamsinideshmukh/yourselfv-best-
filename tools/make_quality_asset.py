"""Calibrate the app's photo check (medsg.quality) on HDA DEV photos, through exactly the app's code path.

Raw photo (max side 1280) -> MediaPipe -> FAN in MediaPipe's face box (blazeface build, CPU), dev pre + post photos of
included pairs. (A first calibration on Stage 5's aligned 512 px crops gave 23 % "caution" on clean raw test photos:
the crop changes the detectors' agreement, so the app path itself is used.) Each anchor's usual MediaPipe-FAN offset
= median; caution / warn = 95th / 99th percentile of the mean disagreement. Offsets are cached privately
(outputs/quality/private_dev_offsets.npz, HDA-derived); the asset holds summary numbers only.
Run (CPU, ~25 min): MEDSG_DATA_DIR=... uv run ... --with face-alignment python tools/make_quality_asset.py
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from medsg.dedup import EXCLUDE  # noqa: E402
from medsg.quality import ANCHORS, ASSET, offsets  # noqa: E402

MAX_SIDE = 1280                                               # = app/app.py MAX_SIDE
CACHE = ROOT / "outputs/quality/private_dev_offsets.npz"


def app_path_offsets(rgb: np.ndarray, lmk, fan) -> np.ndarray:
    """Same steps as the app: resize to max side 1280, MediaPipe, FAN in MediaPipe's box, anchor offsets."""
    from medsg.judges import face_box
    s = MAX_SIDE / max(rgb.shape[:2])
    if s < 1:
        rgb = cv2.resize(rgb, (round(rgb.shape[1] * s), round(rgb.shape[0] * s)), interpolation=cv2.INTER_AREA)
    L = lmk(rgb).lm_px[:, :2].astype(np.float64)
    return offsets(L, fan.landmarks(rgb, face_box(L)))


def dev_offsets() -> np.ndarray:
    if CACHE.exists():
        return np.load(CACHE)["offsets"]
    from medsg.judges import FAN68
    from medsg.landmarks import FaceNotFound, Landmarker
    data = Path(os.environ["MEDSG_DATA_DIR"])
    split = json.load(open(ROOT / "outputs/splits.json"))
    lmk, fan, out = Landmarker(ROOT / "models/face_landmarker.task"), FAN68("cpu", face_detector="blazeface"), []
    for n, pid in enumerate(p for p in split["dev"] if p not in EXCLUDE):
        for suf in ("_b", "_a"):
            rgb = cv2.cvtColor(cv2.imread(str(data / f"{pid}{suf}.jpg")), cv2.COLOR_BGR2RGB)
            try:
                out.append(app_path_offsets(rgb, lmk, fan))
            except (FaceNotFound, ValueError):
                pass
        if n % 25 == 0:
            print(f"  {n} dev pairs", flush=True)
    lmk.close()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, offsets=np.stack(out))
    return np.stack(out)


def main() -> None:
    dev = dev_offsets()
    bias = np.median(dev, 0)
    per_dev = np.linalg.norm(dev - bias, axis=2)
    s_dev = per_dev.mean(1)
    caution, warn = float(np.percentile(s_dev, 95)), float(np.percentile(s_dev, 99))
    q = {"source": "HDA dev photos (pre + post), app code path: raw photo max side 1280, MediaPipe 478, FAN 68 "
                   "(face_alignment, blazeface build) in MediaPipe's box; false-alarm rate on test: tools/validate_quality.py",
         "anchors": [[i, j] for i, j, _ in ANCHORS], "regions": [r for *_, r in ANCHORS],
         "bias": bias.round(5).tolist(), "anchor_p99": np.percentile(per_dev, 99, 0).round(5).tolist(),
         "caution": round(caution, 5), "warn": round(warn, 5), "n_dev_photos": len(dev),
         "dev_median": round(float(np.median(s_dev)), 5)}
    ASSET.write_text(json.dumps(q, indent=1))
    print(json.dumps({k: v for k, v in q.items() if k not in ("bias", "anchor_p99", "anchors", "regions")}, indent=1))
    print("per-anchor usual offset (IPD units, across/down):")
    for (i, j, r), b, p in zip(ANCHORS, bias, q["anchor_p99"]):
        print(f"  MP {i:3d} / FAN {j:2d} {r:6s} bias ({b[0]:+.3f}, {b[1]:+.3f})  p99 residual {p:.3f}")


if __name__ == "__main__":
    main()
