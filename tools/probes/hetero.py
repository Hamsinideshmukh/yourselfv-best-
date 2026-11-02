"""Is the local Jacobian face-dependent? Compare J rows fitted on two halves split by a base-face property
against two random halves of the same size (the random split gives the pure sampling-noise agreement)."""
import sys
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "probes"

import json
import numpy as np
sys.path.insert(0, str(ROOT / "src"))
from medsg.directions import fit_jacobian
from medsg.stages.stage3b_analysis import load_chunks

d = ROOT / "outputs" / "stage3"
meta = json.load(open(d / "meta.json")); M = meta["METRICS"][:10]
g, loc = load_chunks(d, "global"), load_chunks(d, "local")
w_std = np.load(d / "w_std.npy").astype(np.float64)
U = loc["delta"] / w_std
dQ = loc["metrics"][:, :10] - g["metrics"][loc["base_idx"], :10]
ok = loc["ok"] & g["ok"][loc["base_idx"]] & np.isfinite(dQ).all(1)
idx = np.where(ok & (loc["group"] == 1))[0]
base = loc["base_idx"][idx]
bs = g["bs"][base]; pose = g["pose"][base]; bm = g["metrics"][base]
BSN = meta["BLENDSHAPES"]; si = [BSN.index("mouthSmileLeft"), BSN.index("mouthSmileRight")]
props = {"base_smile": bs[:, si].mean(1), "base_abs_yaw": np.abs(pose[:, 0]),
         "base_face_width": bm[:, M.index("face_width")], "base_nose_width": bm[:, M.index("nose_width")]}
def cosrows(A, B): return np.array([a @ b / (np.linalg.norm(a) * np.linalg.norm(b)) for a, b in zip(A, B)])
rng = np.random.default_rng(5)
rand_cos = []
for r in range(3):
    h = rng.permutation(len(idx)) % 2 == 0
    rand_cos.append(cosrows(fit_jacobian(U[idx[h]], dQ[idx[h]], 1)["J"], fit_jacobian(U[idx[~h]], dQ[idx[~h]], 1)["J"]))
rc = np.mean(rand_cos, 0)
print("metric".ljust(18), "random-split cos", *[k for k in props])
res = {}
for k, v in props.items():
    h = v <= np.median(v)
    res[k] = cosrows(fit_jacobian(U[idx[h]], dQ[idx[h]], 1)["J"], fit_jacobian(U[idx[~h]], dQ[idx[~h]], 1)["J"])
for j, m in enumerate(M):
    print(m.ljust(18), f"{rc[j]:.2f}".ljust(16), *[f"{res[k][j]:.2f}".ljust(len(k)) for k in props])
