"""Reduced-rank landmark Jacobian vs direct metric Jacobian (Stage 3, one layer group).
Direct: ridge u -> d_metric (current method).  RRR: ridge u -> d_landmarks (956 outputs, /IPD_base),
truncate the fitted map to rank r (shared deformation modes), metrics recomputed from base + predicted landmarks."""
import sys
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "probes"

import numpy as np
sys.path.insert(0, str(ROOT / "src"))
from medsg.directions import fit_ridge_cv, ALPHAS
from medsg.metrics2d import compute_metrics_2d, METRICS_2D
from medsg.stages.stage3b_analysis import load_chunks

GI = int(sys.argv[1]) if len(sys.argv) > 1 else 1
d = ROOT / "outputs" / "stage3"
M = list(METRICS_2D)
g, loc = load_chunks(d, "global"), load_chunks(d, "local")
w_std = np.load(d / "w_std.npy").astype(np.float64)
lmg = g["lm"].astype(np.float64) * 512; lml = loc["lm"].astype(np.float64) * 512
def mets(lm): return np.array([list(compute_metrics_2d(np.c_[x, np.zeros(478)]).values()) for x in lm])
idx = np.where(loc["ok"] & g["ok"][loc["base_idx"]] & (loc["group"] == GI))[0]
b = loc["base_idx"][idx]
LB, LL = lmg[b], lml[idx]
mB, mL = mets(LB), mets(LL)
print("float16 landmark metrics vs stored metrics, max abs diff:", np.abs(mB - g["metrics"][b, :10]).max().round(5))
dM = mL - mB
U = loc["delta"][idx] / w_std
ipd = np.linalg.norm(LB[:, 468] - LB[:, 473], axis=1)
dL = ((LL - LB) / ipd[:, None, None]).reshape(len(idx), -1)
rng = np.random.default_rng(1); perm = rng.permutation(len(idx)); te, tr = perm[:400], perm[400:]
def r2(y, p): return 1 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)
# direct
f = fit_ridge_cv(U[tr], dM[tr], seed=1234, test_frac=0.0001)
pd_ = (U[te] @ f["coef"] + f["intercept"])
print(f"group {GI}: n_train={len(tr)}  direct metric ridge R²:", np.round(r2(dM[te], pd_), 2))
# landmark ridge: one shared alpha (CV over the mean SSE across outputs)
X, Y = U[tr], dL[tr]
xm, ym = X.mean(0), Y.mean(0)
folds = np.array_split(rng.permutation(len(tr)), 5)
sse = np.zeros(len(ALPHAS))
for k in range(5):
    va = folds[k]; fi = np.concatenate([folds[i] for i in range(5) if i != k])
    xm_, ym_ = X[fi].mean(0), Y[fi].mean(0)
    G = (X[fi]-xm_).T @ (X[fi]-xm_); XtY = (X[fi]-xm_).T @ (Y[fi]-ym_)
    for ai, a in enumerate(ALPHAS):
        B = np.linalg.solve(G + a*np.eye(512), XtY)
        sse[ai] += (((X[va]-xm_) @ B + ym_ - Y[va]) ** 2).sum()
a = ALPHAS[sse.argmin()]
G = (X-xm).T @ (X-xm); B = np.linalg.solve(G + a*np.eye(512), (X-xm).T @ (Y-ym))
Yhat = (X-xm) @ B
_, S, Vt = np.linalg.svd(Yhat, full_matrices=False)
for r in (5, 10, 20, 40, 80, 956):
    Br = B @ Vt[:r].T @ Vt[:r] if r < 956 else B
    pL = ((U[te]-xm) @ Br + ym).reshape(-1, 478, 2) * ipd[te, None, None] + LB[te]
    print(f"  RRR rank {r:3d}: metric R²:", np.round(r2(dM[te], mets(pL) - mB[te]), 2))
