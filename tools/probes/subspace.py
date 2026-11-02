"""Does fitting the local Jacobian in a low-dim latent subspace raise held-out R²? (Stage 3 local set)"""
import sys
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "probes"

import json
import numpy as np
sys.path.insert(0, str(ROOT / "src"))
from medsg.directions import fit_ridge_cv
from medsg.stages.stage3b_analysis import load_chunks, GROUP_NAMES

d = ROOT / "outputs" / "stage3"
meta = json.load(open(d / "meta.json"))
M = meta["METRICS"][:10]
g, loc = load_chunks(d, "global"), load_chunks(d, "local")
w_std = np.load(d / "w_std.npy").astype(np.float64)
U = loc["delta"] / w_std
dQ = loc["metrics"][:, :10] - g["metrics"][loc["base_idx"], :10]
ok = loc["ok"] & g["ok"][loc["base_idx"]] & np.isfinite(dQ).all(1)
# PCA basis of whitened W (global set)
Wz = (g["w"] / w_std); Wz = Wz - Wz.mean(0)
_, S, Vt = np.linalg.svd(Wz, full_matrices=False)
var = S**2 / (S**2).sum()
print("cum var of top-k PCs of whitened W:", {k: round(float(var[:k].sum()), 3) for k in (16, 32, 64, 128, 256)})
rng = np.random.default_rng(0)
Q, _ = np.linalg.qr(rng.standard_normal((512, 512)))
rows = []
for gi, gn in GROUP_NAMES.items():
    m = ok & (loc["group"] == gi)
    full = fit_ridge_cv(U[m], dQ[m], seed=1234)["r2"]
    print(f"\n{gn}  n={m.sum()}  full-512 R²:", dict(zip(M, np.round(full, 3))))
    for k in (16, 32, 64, 128):
        pc = fit_ridge_cv(U[m] @ Vt[:k].T, dQ[m], seed=1234)["r2"]
        rnd = fit_ridge_cv(U[m] @ Q[:, :k], dQ[m], seed=1234)["r2"]
        print(f"  k={k:3d} PCA R²:", dict(zip(M, np.round(pc, 3))))
        print(f"  k={k:3d} rand R²:", dict(zip(M, np.round(rnd, 3))))
