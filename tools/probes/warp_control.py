"""Control: Gaussian-warp real dev pre-op photos (moving t's defining landmarks), re-measure with MediaPipe,
and get the MediaPipe-measured warp pattern. If it matches real surgery as well as the GAN does, the high
GAN cosine is MediaPipe's mesh prior, not the GAN's anatomy. Images processed in memory only."""
import sys
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "probes"

import cv2, numpy as np
sys.path.insert(0, str(ROOT / "src"))
from medsg.config import get_config
from medsg.landmarks import Landmarker, FaceNotFound
from medsg.metrics2d import compute_metrics_2d
import importlib.util
spec = importlib.util.spec_from_file_location("pt", str(ROOT / "tools" / "probes" / "pattern.py"))
src = open(spec.origin).read().split("# ================= controls")[0]
ns = {"__file__": spec.origin}; exec(compile(src, "pt", "exec"), ns)          # reuse REAL / GAN patterns and helpers
frame, pattern, cos, DEF, dev, Dr, Nr, Dg, Ng, pairs, loc, g, idx, b, Mi, mean_face = (ns[k] for k in
    "frame pattern cos DEF dev Dr Nr Dg Ng pairs loc g idx b Mi mean_face".split())
cfg = get_config(); lmk = Landmarker(cfg.models_dir / "face_landmarker.task")
# displacement direction per defining landmark in the iris frame (x outward from midline, y downward = chin)
def dirs(t, F):
    if t in ("nose_width", "mouth_width", "jaw_width"):
        return {i: np.array([np.sign(F[i, 0]), 0.0]) for i in DEF[t]}
    if t == "brow_height":
        return {i: np.array([0.0, -1.0]) for i in DEF[t]}
    if t == "palpebral_height":
        return {159: np.array([0, -.5]), 386: np.array([0, -.5]), 145: np.array([0, .5]), 374: np.array([0, .5])}
    if t == "lower_face_height":
        return {152: np.array([0.0, 1.0])}
def warp(rgb, lm, t, amp):
    a, bb = lm[468], lm[473]; ipd = np.linalg.norm(bb - a); u = (bb - a) / ipd; v = np.array([-u[1], u[0]])
    if np.dot(lm[152] - (a + bb) / 2, v) < 0: v = -v
    F = frame(lm[None])[0]
    h, w = rgb.shape[:2]; yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = np.zeros_like(xx); dy = np.zeros_like(yy); sig = 0.15 * ipd
    for i, dvec in dirs(t, F).items():
        d = amp * ipd * (dvec[0] * u + dvec[1] * v)
        wgt = np.exp(-((xx - lm[i, 0]) ** 2 + (yy - lm[i, 1]) ** 2) / (2 * sig ** 2))
        dx += d[0] * wgt; dy += d[1] * wgt
    return cv2.remap(rgb, xx - dx, yy - dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
rng = np.random.default_rng(3); sel = list(rng.choice(dev, 60, replace=False))
A_modes = []
for k in range(2):
    e = np.zeros((478, 2)); e[:, k] = 1; A_modes.append(e.ravel())
for i in range(2):
    for j in range(2):
        e = np.zeros((478, 2)); e[:, i] = mean_face[:, j]; A_modes.append(e.ravel())
A = np.stack(A_modes, 1); Pa = np.eye(956) - A @ np.linalg.pinv(A)
allm = np.ones(478, bool)
for t in DEF:
    D, dt = [], []
    for p in sel:
        rgb = cv2.cvtColor(cv2.imread(str(cfg.data_dir / f"{p}{cfg.pre_suffix}{cfg.img_ext}")), cv2.COLOR_BGR2RGB)
        lm0 = lmk(rgb).lm_px[:, :2].astype(np.float64); F0 = frame(lm0[None])[0]
        m0 = compute_metrics_2d(np.c_[lm0, np.zeros(478)])[t]
        for amp in (-0.02, 0.02):
            try:
                lm1 = lmk(warp(rgb, lm0, t, amp)).lm_px[:, :2].astype(np.float64)
            except FaceNotFound:
                continue
            D.append((frame(lm1[None])[0] - F0).ravel()); dt.append(compute_metrics_2d(np.c_[lm1, np.zeros(478)])[t] - m0)
    D, dt = np.array(D), np.array(dt)
    P_w = np.linalg.lstsq(np.c_[np.ones(len(dt)), dt], D, rcond=None)[0][1]
    dt_r = pairs.loc[dev, f"d_{t}"].to_numpy(); P_r = pattern(Dr, dt_r, Nr)
    dt_g = loc["metrics"][idx, Mi[t]] - g["metrics"][b, Mi[t]]; P_g = pattern(Dg, dt_g, Ng)
    print(f"{t:18s} n={len(dt):3d} mean|dt|={np.abs(dt).mean():.3f}  affine-removed cos to REAL:  "
          f"GAN={cos(Pa@P_r, Pa@P_g, allm):.2f}  MP(gauss-warp)={cos(Pa@P_r, Pa@P_w, allm):.2f}   "
          f"GAN vs MP(warp)={cos(Pa@P_g, Pa@P_w, allm):.2f}")
lmk.close()
