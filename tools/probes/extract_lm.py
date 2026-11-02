"""Extract MediaPipe 478 landmarks for all HDA images (CPU) -> outputs/stage1/private/landmarks.npz (never commit)."""
import sys
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "probes"

import time
sys.path.insert(0, str(ROOT / "src"))
import cv2, numpy as np
from medsg.config import get_config
from medsg.data import list_pairs
from medsg.landmarks import Landmarker, FaceNotFound
cfg = get_config()
lmk = Landmarker(cfg.models_dir / "face_landmarker.task")
out, fails, t = {}, [], time.time()
for p in list_pairs(cfg):
    for role, path in (("pre", p.pre), ("post", p.post)):
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        try:
            out[f"{p.pid}_{role}"] = lmk(rgb).lm_px[:, :2].astype(np.float32)
        except FaceNotFound:
            fails.append(f"{p.pid}_{role}")
lmk.close()
dst = cfg.out_dir / "stage1" / "private" / "landmarks.npz"
np.savez_compressed(dst, **out)
print(f"{len(out)} images, {len(fails)} failures {fails[:5]}, {time.time()-t:.0f}s -> {dst}")
