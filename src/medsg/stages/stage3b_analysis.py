"""Stage 3b: first analysis of the synthetic latent <-> anatomy data (CPU, ~1 min, no images).

Plain words
- Ridge regression: a linear fit y ≈ w·b with a penalty on large coefficients, so 512 inputs
  do not overfit. alpha = penalty strength, chosen by 5-fold cross-validation on the train part.
- R² on held-out faces: share of the metric's spread the fit explains on faces it never saw
  (1 = perfect, 0 = no better than predicting the average).

A) Domain gap: synthetic (psi=0.7) metric/pose/expression distributions vs real HDA (Stage 1).
B) Global set: held-out R² of  w -> metric,  nuisance (pose + blendshapes) -> metric,  and
   w -> metric after removing the nuisance part (does the latent carry shape beyond pose/expression?).
   Also w -> each nuisance variable (how strongly the latent encodes pose/expression).
C) Local set: within-face change  d_metric = metric(w + delta in one layer group) - metric(w).
   Held-out R² of delta -> d_metric per layer group (coarse 0-3, middle 4-7, fine 8-17):
   which layers move which anatomy, and how much they also move pose/expression (leakage).

Writes outputs/stage3/analysis_*.csv.  Run: python -m medsg.stages.stage3b_analysis
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import get_config
from ..directions import fit_ridge_cv

GROUP_NAMES = {0: "coarse_0-3", 1: "middle_4-7", 2: "fine_8-17"}


def load_chunks(d, kind: str) -> dict[str, np.ndarray]:
    files = sorted((d / "chunks").glob(f"{kind}_*.npz"))
    if not files:
        raise FileNotFoundError(f"no {kind} chunks in {d / 'chunks'}")
    parts = [dict(np.load(f)) for f in files]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def ridge_heldout_r2(X: np.ndarray, Y: np.ndarray, seed: int, test_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Held-out R² per target and the CV-chosen alpha (see directions.fit_ridge_cv)."""
    f = fit_ridge_cv(X, Y, seed, test_frac)
    return f["r2"], f["alpha"]


def _residualise(Y: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Remove the linear (OLS) part of Y explained by nuisance N."""
    A = np.c_[np.ones(len(N)), N]
    return Y - A @ np.linalg.lstsq(A, Y, rcond=None)[0]


def main() -> None:
    cfg = get_config()
    d = cfg.out_dir / "stage3"
    meta = json.load(open(d / "meta.json"))
    M, P, BS = meta["METRICS"], meta["POSE"], [f"bs_{b}" for b in meta["BLENDSHAPES"]]
    g = load_chunks(d, "global")
    loc = load_chunks(d, "local")
    assert g["w"].shape == (meta["N_GLOBAL"], 512), g["w"].shape
    assert loc["delta"].shape == (meta["N_BASES"], 512), loc["delta"].shape
    if not g["ok"].all() or not loc["ok"].all():
        print(f"note: measurement failures global {(~g['ok']).sum()}, local {(~loc['ok']).sum()} (dropped)")
    nuis = np.c_[g["pose"], g["bs"]]

    # A) domain gap vs real HDA (Stage 1, all measured photos)
    real = pd.read_csv(cfg.out_dir / "stage1" / "images.csv")
    syn = pd.DataFrame(np.c_[g["metrics"], nuis][g["ok"]], columns=M + P + BS)
    rows = []
    for c in M + P + BS:
        r, s = real[c].dropna(), syn[c]
        rows.append({"variable": c, "real_mean": r.mean(), "syn_mean": s.mean(), "real_sd": r.std(),
                     "syn_sd": s.std(), "sd_ratio_syn_over_real": s.std() / r.std(),
                     "std_mean_diff": (s.mean() - r.mean()) / r.std()})
    gap = pd.DataFrame(rows)
    gap.to_csv(d / "analysis_domain_gap.csv", index=False)

    # B) global: w -> metric, nuisance -> metric, w -> metric | nuisance, w -> nuisance
    ok = g["ok"]
    W, Y, N = g["w"][ok], g["metrics"][ok], nuis[ok]
    r2_w, a_w = ridge_heldout_r2(W, Y, cfg.seed)
    r2_n, _ = ridge_heldout_r2(N, Y, cfg.seed)
    r2_wres, _ = ridge_heldout_r2(W, _residualise(Y, N), cfg.seed)
    r2_wn, _ = ridge_heldout_r2(W, N, cfg.seed)
    glob = pd.DataFrame({"metric": M, "r2_w": r2_w, "alpha_w": a_w, "r2_nuisance_only": r2_n,
                         "r2_w_after_removing_nuisance": r2_wres})
    glob.to_csv(d / "analysis_global_r2.csv", index=False)
    pd.DataFrame({"nuisance": P + BS, "r2_w": r2_wn}).to_csv(d / "analysis_global_nuisance_r2.csv", index=False)

    # C) local: within-face finite differences per layer group
    base = loc["base_idx"]
    assert (g["idx"][base] == base).all(), "global chunks are not in idx order"
    okl = loc["ok"] & g["ok"][base]
    dM = loc["metrics"] - g["metrics"][base]
    dN = np.c_[loc["pose"], loc["bs"]] - nuis[base]
    sem = pd.read_csv(cfg.out_dir / "stage1" / "noise_floor.csv").set_index("metric")["sem"]
    rows = []
    for gi, gname in GROUP_NAMES.items():
        sel = okl & (loc["group"] == gi)
        X = loc["delta"][sel]
        r2m, _ = ridge_heldout_r2(X, dM[sel], cfg.seed)
        r2n, _ = ridge_heldout_r2(X, dN[sel], cfg.seed)
        for name, r2, sd in zip(M + P + BS, np.r_[r2m, r2n], np.r_[dM[sel].std(0), dN[sel].std(0)]):
            rows.append({"group": gname, "n": int(sel.sum()), "variable": name,
                         "kind": "metric" if name in M else "nuisance", "r2_delta": r2, "sd_change": sd,
                         # approx. noise ceiling 1 - 2 SEM^2 / Var(change), with the real-photo same-photo SEM
                         # (the synthetic noise floor is Stage 4a, not measured yet)
                         "r2_max_approx": 1 - 2 * sem[name] ** 2 / sd ** 2 if name in M else np.nan})
    local = pd.DataFrame(rows)
    local.to_csv(d / "analysis_local_r2.csv", index=False)

    pd.set_option("display.width", 220)
    print(gap.round(3).to_string(index=False))
    print(glob.round(3).to_string(index=False))
    print(pd.read_csv(d / "analysis_global_nuisance_r2.csv").round(3).to_string(index=False))
    print(local.pivot(index="variable", columns="group", values="r2_delta").loc[M + P + BS].round(3).to_string())
    print(local.pivot(index="variable", columns="group", values="r2_max_approx").loc[M].round(3).to_string())


if __name__ == "__main__":
    main()
