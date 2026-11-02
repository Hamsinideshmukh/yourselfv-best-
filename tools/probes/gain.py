"""Known-deformation (phantom) test of MediaPipe: warp real dev pre-op photos with a known smooth field,
truth = metric(lm0 + field(lm0)) - metric(lm0) (exact landmark motion under the field), measured = MediaPipe
on the warped photo. Gain = measured/true (slope through 0). Images in memory only."""
import sys
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "probes"

import json
import cv2, numpy as np, pandas as pd
sys.path.insert(0, str(ROOT / "src"))
from medsg.config import get_config
from medsg.landmarks import Landmarker, FaceNotFound
from medsg.metrics2d import compute_metrics_2d
cfg = get_config(); R = cfg.out_dir
split = json.load(open(R / "splits.json")); pairs = pd.read_csv(R / "stage1/pairs.csv", dtype={"pid": str}).set_index("pid")
dev = [p for p in split["dev"] if p in pairs.index and pairs.loc[p, "include_pose"]]
sel = list(np.random.default_rng(11).choice(dev, int(sys.argv[1]) if len(sys.argv) > 1 else 25, replace=False))
DEF = {"nose_width": {129: (-1, 0), 358: (1, 0)}, "mouth_width": {61: (-1, 0), 291: (1, 0)},
       "jaw_width": {172: (-1, 0), 397: (1, 0)}, "lower_face_height": {152: (0, 1)},
       "brow_height": {105: (0, -1), 334: (0, -1)}, "palpebral_height": {159: (0, -1), 386: (0, -1), 145: (0, 1), 374: (0, 1)}}
lmk = Landmarker(cfg.models_dir / "face_landmarker.task")
def field(pts, lm, t, amp, sig_ipd):
    a, b = lm[468], lm[473]; ipd = np.linalg.norm(b - a); u = (b - a) / ipd; v = np.array([-u[1], u[0]])
    if np.dot(lm[152] - (a + b) / 2, v) < 0: v = -v
    side = np.sign((lm[list(DEF[t])] - (a + b) / 2) @ u)
    d = np.zeros_like(pts, dtype=np.float64); s2 = 2 * (sig_ipd * ipd) ** 2
    for (i, (sx, sy)), sd in zip(DEF[t].items(), side):
        vec = amp * ipd * ((abs(sx) * sd if sx else 0) * u + sy * v)
        wgt = np.exp(-((pts - lm[i]) ** 2).sum(-1) / s2)
        d += wgt[..., None] * vec
    return d
rows = []
for p in sel:
    rgb = cv2.cvtColor(cv2.imread(str(cfg.data_dir / f"{p}{cfg.pre_suffix}{cfg.img_ext}")), cv2.COLOR_BGR2RGB)
    lm0 = lmk(rgb).lm_px[:, :2].astype(np.float64); h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64); grid = np.stack([xx, yy], -1)
    m0 = compute_metrics_2d(np.c_[lm0, np.zeros(478)])
    for t in DEF:
        for sig in (0.1, 0.2, 0.35):
            for amp in (-0.02, 0.02):
                D = field(grid, lm0, t, amp, sig)
                # backward map with one fixed-point refinement: source = x - D(x - D(x))
                src = grid - field(grid - D, lm0, t, amp, sig)
                img = cv2.remap(rgb, src[..., 0].astype(np.float32), src[..., 1].astype(np.float32), cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REFLECT)
                true = compute_metrics_2d(np.c_[lm0 + field(lm0, lm0, t, amp, sig), np.zeros(478)])[t] - m0[t]
                try:
                    meas = compute_metrics_2d(np.c_[lmk(img).lm_px[:, :2], np.zeros(478)])[t] - m0[t]
                except FaceNotFound:
                    continue
                rows.append({"pid": p, "metric": t, "sigma_ipd": sig, "amp": amp, "true": true, "meas": meas})
lmk.close()
df = pd.DataFrame(rows)
df.to_csv(OUT / "gain_raw.csv", index=False)
out = []
for (t, s), d in df.groupby(["metric", "sigma_ipd"]):
    g = float(d.true @ d.meas / (d.true @ d.true))
    bs = []
    rng = np.random.default_rng(0); ids = d.pid.unique()
    for _ in range(1000):
        pick = rng.choice(ids, len(ids)); dd = pd.concat([d[d.pid == i] for i in pick])
        bs.append(dd.true @ dd.meas / (dd.true @ dd.true))
    out.append({"metric": t, "sigma_ipd": s, "n_images": len(ids), "mean_abs_true": d.true.abs().mean(),
                "gain": g, "ci": f"[{np.quantile(bs, .025):.2f}, {np.quantile(bs, .975):.2f}]",
                "resid_sd": float(np.std(d.meas - g * d.true))})
pd.set_option("display.width", 200)
pd.DataFrame(out).to_csv(OUT / "gain_summary.csv", index=False)
print(pd.DataFrame(out).round(3).to_string(index=False))
