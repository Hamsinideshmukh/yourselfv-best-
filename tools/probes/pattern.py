"""Deformation-pattern test: does the GAN's landmark deformation for a metric change look like real surgery's?

All landmarks in the iris frame (origin iris midpoint, x along the iris line, unit = IPD), as the metrics are.
Pattern for target t = partial regression coefficient of the 956-dim landmark displacement on d_t, adjusting for
nuisance changes (d smile, d jawOpen, d yaw, d pitch): "how every landmark moves per +1 unit of t".
  REAL : HDA dev pose-included pairs (post - pre)
  GAN  : Stage-3 synthetic local perturbations, layer group 4-7 (local - base)
  LOCAL: only the landmarks that define t move (pure metric edit)
  GAUSS: LOCAL spread to neighbours with a Gaussian of distance (sigma 0.15 IPD), like the v1 warp (B3)
Similarity = cosine over landmarks (a) all, (b) excluding t's defining landmarks (how surrounding tissue moves).
Ceiling = split-half cosine of REAL with itself, Spearman-Brown corrected. CIs: bootstrap over pairs."""
import sys
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "probes"

import json
import numpy as np, pandas as pd
sys.path.insert(0, str(ROOT / "src"))
from medsg.metrics2d import compute_metrics_2d
from medsg.stages.stage3b_analysis import load_chunks

R = ROOT / "outputs"
DEF = {"nose_width": [129, 358], "brow_height": [105, 334], "palpebral_height": [159, 145, 386, 374],
       "mouth_width": [61, 291], "lower_face_height": [2, 152], "jaw_width": [172, 397]}

def frame(lm):
    lm = np.asarray(lm, np.float64)[..., :2]
    a, b = lm[..., 468, :], lm[..., 473, :]
    c = (a + b) / 2; ipd = np.linalg.norm(b - a, axis=-1)
    u = (b - a) / ipd[..., None]; v = np.stack([-u[..., 1], u[..., 0]], -1)
    rel = lm - c[..., None, :]
    return np.stack([(rel * u[..., None, :]).sum(-1), (rel * v[..., None, :]).sum(-1)], -1) / ipd[..., None, None]

def pattern(D, dt, N):
    X = np.c_[np.ones(len(dt)), dt, N]
    return np.linalg.lstsq(X, D, rcond=None)[0][1]          # (956,) per +1 unit of t

def cos(a, b, mask):
    a, b = a.reshape(478, 2)[mask].ravel(), b.reshape(478, 2)[mask].ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

# ---- real HDA (dev, pose-included)
pairs = pd.read_csv(R / "stage1/pairs.csv", dtype={"pid": str}).set_index("pid")
img = pd.read_csv(R / "stage1/images.csv", dtype={"pid": str})
split = json.load(open(R / "splits.json"))
dev = [p for p in split["dev"] if p in pairs.index and pairs.loc[p, "include_pose"]]
L = np.load(R / "stage1/private/landmarks.npz")
pre = img[img.role == "pre"].set_index("pid").loc[dev]; post = img[img.role == "post"].set_index("pid").loc[dev]
Fpre = frame(np.stack([L[f"{p}_pre"] for p in dev])); Fpost = frame(np.stack([L[f"{p}_post"] for p in dev]))
# sanity: metrics from these landmarks vs Stage 1
chk = np.array([compute_metrics_2d(np.c_[L[f"{p}_pre"], np.zeros(478)])["nose_width"] for p in dev[:50]])
print("landmark re-measure vs Stage-1 nose_width, max |diff|:", np.abs(chk - pre.nose_width.to_numpy()[:50]).max().round(5))
Dr = (Fpost - Fpre).reshape(len(dev), -1)
sm = ["bs_mouthSmileLeft", "bs_mouthSmileRight"]
Nr = np.c_[post[sm].mean(1) - pre[sm].mean(1), post.bs_jawOpen - pre.bs_jawOpen, post.yaw - pre.yaw, post.pitch - pre.pitch]
# ---- GAN synthetic local, middle group
d3 = R / "stage3"; meta = json.load(open(d3 / "meta.json"))
g, loc = load_chunks(d3, "global"), load_chunks(d3, "local")
idx = np.where(loc["ok"] & g["ok"][loc["base_idx"]] & (loc["group"] == 1))[0]; b = loc["base_idx"][idx]
Fb, Fl = frame(g["lm"][b].astype(np.float64)), frame(loc["lm"][idx].astype(np.float64))
Dg = (Fl - Fb).reshape(len(idx), -1)
BS = meta["BLENDSHAPES"]; si = [BS.index("mouthSmileLeft"), BS.index("mouthSmileRight")]; ji = BS.index("jawOpen")
Ng = np.c_[loc["bs"][idx][:, si].mean(1) - g["bs"][b][:, si].mean(1), loc["bs"][idx, ji] - g["bs"][b, ji],
           loc["pose"][idx, 0] - g["pose"][b, 0], loc["pose"][idx, 1] - g["pose"][b, 1]]
Mi = {m: i for i, m in enumerate(meta["METRICS"])}
mean_face = Fpre.mean(0)
rng = np.random.default_rng(7)
rows = []
for t, lms in DEF.items():
    dt_r = pairs.loc[dev, f"d_{t}"].to_numpy()
    dt_g = loc["metrics"][idx, Mi[t]] - g["metrics"][b, Mi[t]]
    P_real = pattern(Dr, dt_r, Nr); P_gan = pattern(Dg, dt_g, Ng)
    # LOCAL: defining landmarks moved by the real pattern's own displacement at those points (same "metric part")
    loc_p = np.zeros((478, 2)); loc_p[lms] = P_real.reshape(478, 2)[lms]
    dist = np.min(np.linalg.norm(mean_face[:, None, :] - mean_face[None, lms, :], axis=-1), 1)
    near = np.argmin(np.linalg.norm(mean_face[:, None, :] - mean_face[None, lms, :], axis=-1), 1)
    gauss = np.exp(-0.5 * (dist / 0.15) ** 2)[:, None] * loc_p[np.array(lms)[near]]
    m_all = np.ones(478, bool); m_off = m_all.copy(); m_off[lms] = False
    def stats(mask):
        out = {"gan": cos(P_real, P_gan, mask), "gauss": cos(P_real, gauss.ravel(), mask)}
        if mask[lms].any(): out["local"] = cos(P_real, loc_p.ravel(), mask)
        bo = {k: [] for k in out}; sh = []
        for _ in range(500):
            s = rng.integers(0, len(dev), len(dev)); Pb = pattern(Dr[s], dt_r[s], Nr[s])
            bo["gan"].append(cos(Pb, P_gan, mask)); bo["gauss"].append(cos(Pb, gauss.ravel(), mask))
            if "local" in bo: bo["local"].append(cos(Pb, loc_p.ravel(), mask))
            h = rng.permutation(len(dev)) < len(dev) // 2
            r = cos(pattern(Dr[h], dt_r[h], Nr[h]), pattern(Dr[~h], dt_r[~h], Nr[~h]), mask)
            sh.append(2 * r / (1 + r))
        return {k: (v, *np.quantile(bo[k], [0.025, 0.975])) for k, v in out.items()}, float(np.median(sh))
    for name, mask in (("all", m_all), ("off_target", m_off)):
        st, ceil = stats(mask)
        rows.append({"metric": t, "landmarks": name, "n_pairs": len(dev), "real_reliability": ceil,
                     **{f"cos_{k}": v[0] for k, v in st.items()}, **{f"cos_{k}_ci": f"[{v[1]:.2f}, {v[2]:.2f}]" for k, v in st.items()}})
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(df.round(2).to_string(index=False))
df.to_csv(OUT / "pattern_test.csv", index=False)

# ================= controls =================
# (1) remove global affine motion (6 modes on the mean face: x/y translation, and the 4 linear-map entries)
mf = mean_face
A_modes = []
for k in range(2):
    e = np.zeros((478, 2)); e[:, k] = 1; A_modes.append(e.ravel())
for i in range(2):
    for j in range(2):
        e = np.zeros((478, 2)); e[:, i] = mf[:, j]; A_modes.append(e.ravel())
A = np.stack(A_modes, 1); Pa = np.eye(956) - A @ np.linalg.pinv(A)
def deaff(p): return Pa @ p
Preal, Pgan = {}, {}
for t in DEF:
    dt_r = pairs.loc[dev, f"d_{t}"].to_numpy(); dt_g = loc["metrics"][idx, Mi[t]] - g["metrics"][b, Mi[t]]
    Preal[t], Pgan[t] = pattern(Dr, dt_r, Nr), pattern(Dg, dt_g, Ng)
allm = np.ones(478, bool)
for label, f in (("raw", lambda p: p), ("affine_removed", deaff)):
    C = pd.DataFrame([[cos(f(Preal[t]), f(Pgan[s]), allm) for s in DEF] for t in DEF], index=[f"real:{t}" for t in DEF],
                     columns=[f"gan:{s}" for s in DEF])
    print(f"\n[{label}] cosine real pattern (rows) vs GAN pattern (cols); diagonal = same metric")
    print(C.round(2).to_string())
    CR = pd.DataFrame([[cos(f(Preal[t]), f(Preal[s]), allm) for s in DEF] for t in DEF], index=DEF, columns=DEF)
    print(f"[{label}] real vs real (other metrics)"); print(CR.round(2).to_string())
# (2) local region only (landmarks within 0.35 IPD of the defining ones), affine removed
print("\n[affine_removed, local region <=0.35 IPD] gan vs gauss vs local")
for t, lms in DEF.items():
    dist = np.min(np.linalg.norm(mf[:, None, :] - mf[None, lms, :], axis=-1), 1); reg = dist <= 0.35
    near = np.argmin(np.linalg.norm(mf[:, None, :] - mf[None, lms, :], axis=-1), 1)
    loc_p = np.zeros((478, 2)); loc_p[lms] = Preal[t].reshape(478, 2)[lms]
    gauss = (np.exp(-0.5 * (dist / 0.15) ** 2)[:, None] * loc_p[np.array(lms)[near]]).ravel()
    pr = deaff(Preal[t])
    bo = []
    dt_r = pairs.loc[dev, f"d_{t}"].to_numpy()
    for _ in range(300):
        s = rng.integers(0, len(dev), len(dev)); bo.append(cos(deaff(pattern(Dr[s], dt_r[s], Nr[s])), deaff(Pgan[t]), reg))
    print(f"  {t:18s} n_lm={reg.sum():3d}  gan={cos(pr, deaff(Pgan[t]), reg):.2f} [{np.quantile(bo,.025):.2f},{np.quantile(bo,.975):.2f}]"
          f"  gauss={cos(pr, deaff(gauss), reg):.2f}  local={cos(pr, deaff(loc_p.ravel()), reg):.2f}")
