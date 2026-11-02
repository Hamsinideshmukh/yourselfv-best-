"""Learning curve: held-out R² of the local Jacobian vs number of training perturbations (Stage 3, middle 4-7)."""
import sys
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "probes"

import json
import numpy as np
sys.path.insert(0, str(ROOT / "src"))
from medsg.directions import ALPHAS
from medsg.stages.stage3b_analysis import load_chunks

d = ROOT / "outputs" / "stage3"
M = json.load(open(d / "meta.json"))["METRICS"][:10]
g, loc = load_chunks(d, "global"), load_chunks(d, "local")
w_std = np.load(d / "w_std.npy").astype(np.float64)
U = loc["delta"] / w_std
dQ = loc["metrics"][:, :10] - g["metrics"][loc["base_idx"], :10]
ok = loc["ok"] & g["ok"][loc["base_idx"]] & np.isfinite(dQ).all(1)

def ridge_r2(Xtr, Ytr, Xte, Yte):
    # alpha chosen on the test set's *train-internal* split would be cleaner; use inner 5-fold on train
    rng = np.random.default_rng(0); idx = rng.permutation(len(Xtr)); folds = np.array_split(idx, 5)
    best = []
    for t in range(Ytr.shape[1]):
        sse = np.zeros(len(ALPHAS))
        for k in range(5):
            va = folds[k]; fi = np.concatenate([folds[i] for i in range(5) if i != k])
            xm, ym = Xtr[fi].mean(0), Ytr[fi, t].mean()
            G = (Xtr[fi]-xm).T @ (Xtr[fi]-xm); b = (Xtr[fi]-xm).T @ (Ytr[fi, t]-ym)
            for ai, a in enumerate(ALPHAS):
                B = np.linalg.solve(G + a*np.eye(G.shape[0]), b)
                sse[ai] += (((Xtr[va]-xm) @ B + ym - Ytr[va, t])**2).sum()
        best.append(ALPHAS[sse.argmin()])
    xm, ym = Xtr.mean(0), Ytr.mean(0); G = (Xtr-xm).T @ (Xtr-xm)
    r2 = []
    for t, a in enumerate(best):
        B = np.linalg.solve(G + a*np.eye(G.shape[0]), (Xtr-xm).T @ (Ytr[:, t]-ym[t]))
        p = (Xte-xm) @ B + ym[t]
        r2.append(1 - ((Yte[:, t]-p)**2).sum() / ((Yte[:, t]-Yte[:, t].mean())**2).sum())
    return np.array(r2)

for gi in (1,):
    m = np.where(ok & (loc["group"] == gi))[0]
    rng = np.random.default_rng(1); m = rng.permutation(m)
    te, tr = m[:400], m[400:]
    for n in (200, 400, 800, 1200, len(tr)):
        r = ridge_r2(U[tr[:n]], dQ[tr[:n]], U[te], dQ[te])
        print(f"n_train={n:5d}", " ".join(f"{k[:10]}={v:.2f}" for k, v in zip(M, r)))
