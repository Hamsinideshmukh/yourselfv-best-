"""E7 (pre-registered, outputs/eval_plan.json): which deformation prior matches real surgical change on held-out pairs?

For a test pair and metric t whose real change exceeds the realistic noise floor (Stage 1b), each prior predicts
how all 478 landmarks move per +1 unit of t (a 956-number "pattern" P); the prediction for the pair is Δt_real · P
(oracle size: the question is the *shape* of the change, not its size). Compared with the real displacement
(post - pre, iris frame: origin iris midpoint, x along the iris line, unit = IPD), after removing global affine
motion (head pose, scale, shear). Two scores per pair:
  cosine  : agreement in shape (+1 identical direction, 0 unrelated);
  rmse_gain: how much the prediction reduces landmark error compared with predicting no change (B0); > 0 = better.
Priors: GAUSS_MP (smooth warp of real dev photos, re-measured by MediaPipe), GAN_POP (Stage 3 synthetic faces),
GAN_PERS (the test face's own e4e code, Stage 5 C3, antithetic), HDA_EMP (real dev pairs). All patterns adjust
for expression/pose changes (Δsmile, ΔjawOpen, Δyaw, Δpitch) by regression, as in the earlier CPU probes (tools/probes).
Writes outputs/stage5/e7_pairs.csv, e7_summary.csv, e7_comparisons.csv.
"""
from __future__ import annotations

import glob
import json

import cv2
import numpy as np
import pandas as pd

from ..config import get_config
from ..data import list_pairs
from ..dedup import EXCLUDE
from ..deform import face_axes, phantom_controls, warp_gaussian
from ..landmarks import FaceNotFound, Landmarker
from ..metrics2d import compute_metrics_2d
from ..stats import holm
from .stage3b_analysis import load_chunks
from .stage5b_analysis import paired_boot

TARGETS = ["nose_width", "brow_height", "palpebral_height", "mouth_width", "jaw_width", "lower_face_height"]
BSN = ["jawOpen", "mouthSmileLeft", "mouthSmileRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft",
       "eyeSquintRight", "browInnerUp", "browDownLeft", "browDownRight"]      # Stage 1 / C3 blendshape order
N_GAUSS, GAUSS_SIGMA, GAUSS_AMP = 60, 0.15, 0.02


def frame(lm: np.ndarray) -> np.ndarray:
    """(..., 478, 2) pixel landmarks -> iris frame (origin iris midpoint, x along iris line, unit IPD)."""
    lm = np.asarray(lm, np.float64)[..., :2]
    a, b = lm[..., 468, :], lm[..., 473, :]
    c = (a + b) / 2
    ipd = np.linalg.norm(b - a, axis=-1)
    u = (b - a) / ipd[..., None]
    v = np.stack([-u[..., 1], u[..., 0]], -1)
    rel = lm - c[..., None, :]
    return np.stack([(rel * u[..., None, :]).sum(-1), (rel * v[..., None, :]).sum(-1)], -1) / ipd[..., None, None]


def affine_projector(mean_face: np.ndarray) -> np.ndarray:
    modes = []
    for k in range(2):
        e = np.zeros((478, 2)); e[:, k] = 1; modes.append(e.ravel())
    for i in range(2):
        for j in range(2):
            e = np.zeros((478, 2)); e[:, i] = mean_face[:, j]; modes.append(e.ravel())
    A = np.stack(modes, 1)
    return np.eye(956) - A @ np.linalg.pinv(A)


def pattern(D: np.ndarray, dt: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Landmark motion per +1 unit of t, adjusting for nuisance changes N (least squares with intercept)."""
    ok = np.isfinite(D).all(1) & np.isfinite(dt) & np.isfinite(N).all(1)
    X = np.c_[np.ones(ok.sum()), dt[ok], N[ok]]
    return np.linalg.lstsq(X, D[ok], rcond=None)[0][1]


def _nuis_from_bs(bs: np.ndarray, pose: np.ndarray) -> np.ndarray:
    i = {b: k for k, b in enumerate(BSN)}
    smile = bs[..., [i["mouthSmileLeft"], i["mouthSmileRight"]]].mean(-1)
    return np.stack([smile, bs[..., i["jawOpen"]], pose[..., 0], pose[..., 1]], -1)


def main() -> None:
    cfg = get_config()
    out = cfg.out_dir / "stage5"
    split = json.load(open(cfg.out_dir / "splits.json"))
    pairs_tab = pd.read_csv(cfg.out_dir / "stage1/pairs.csv", dtype={"pid": str}).set_index("pid")
    img = pd.read_csv(cfg.out_dir / "stage1/images.csv", dtype={"pid": str})
    noise = pd.read_csv(cfg.out_dir / "stage1/followup_realistic_noise.csv").set_index("metric")
    L = np.load(cfg.out_dir / "stage1/private/landmarks.npz")
    pre_i, post_i = img[img.role == "pre"].set_index("pid"), img[img.role == "post"].set_index("pid")
    keep = lambda w: [p for p in split[w] if p not in EXCLUDE and p in pairs_tab.index and pairs_tab.loc[p, "include_pose"]]  # noqa: E731
    dev, test = keep("dev"), keep("test")

    def real_set(ids):
        D = (frame(np.stack([L[f"{p}_post"] for p in ids])) - frame(np.stack([L[f"{p}_pre"] for p in ids]))).reshape(len(ids), -1)
        cols = ["bs_mouthSmileLeft", "bs_mouthSmileRight"]
        N = np.c_[post_i.loc[ids, cols].mean(1) - pre_i.loc[ids, cols].mean(1),
                  post_i.loc[ids, "bs_jawOpen"] - pre_i.loc[ids, "bs_jawOpen"],
                  post_i.loc[ids, "yaw"] - pre_i.loc[ids, "yaw"], post_i.loc[ids, "pitch"] - pre_i.loc[ids, "pitch"]]
        return D, N
    Dd, Nd = real_set(dev)
    Dt, Nt = real_set(test)
    mean_face = frame(np.stack([L[f"{p}_pre"] for p in dev])).mean(0)
    Pa = affine_projector(mean_face)

    # --- population priors -----------------------------------------------------------------------
    P = {"HDA_EMP": {}, "GAN_POP": {}, "GAUSS_MP": {}}
    for t in TARGETS:
        P["HDA_EMP"][t] = pattern(Dd, pairs_tab.loc[dev, f"d_{t}"].to_numpy(), Nd)
    d3 = cfg.out_dir / "stage3"
    meta3 = json.load(open(d3 / "meta.json"))
    g, loc = load_chunks(d3, "global"), load_chunks(d3, "local")
    ok = loc["ok"] & g["ok"][loc["base_idx"]]
    b = loc["base_idx"][ok]
    Dg = (frame(loc["lm"][ok].astype(np.float64)) - frame(g["lm"][b].astype(np.float64))).reshape(ok.sum(), -1)
    Ng = _nuis_from_bs(loc["bs"][ok], loc["pose"][ok]) - _nuis_from_bs(g["bs"][b], g["pose"][b])
    Mi = {m: k for k, m in enumerate(meta3["METRICS"])}
    for t in TARGETS:
        P["GAN_POP"][t] = pattern(Dg, loc["metrics"][ok, Mi[t]] - g["metrics"][b, Mi[t]], Ng)
    # Gaussian warp of real dev pre-op photos, re-measured by MediaPipe (population pattern)
    lmk = Landmarker(cfg.models_dir / "face_landmarker.task")
    by_pid = {p.pid: p for p in list_pairs(cfg)}
    sel = list(np.random.default_rng(cfg.seed + 80).choice(dev, min(N_GAUSS, len(dev)), replace=False))
    spec = {"nose_width": "nose_width", "brow_height": "brow_height", "palpebral_height": "palpebral_height",
            "mouth_width": "mouth_width", "jaw_width": "jaw_width", "lower_face_height": "lower_face_height"}
    for t in TARGETS:
        D, dt = [], []
        for p in sel:
            rgb = cv2.cvtColor(cv2.imread(str(by_pid[p].pre)), cv2.COLOR_BGR2RGB)
            lm0 = L[f"{p}_pre"].astype(np.float64)
            F0, m0 = frame(lm0), compute_metrics_2d(np.c_[lm0, np.zeros(478)])[t]
            ipd = face_axes(lm0)[3]
            for amp in (-GAUSS_AMP, GAUSS_AMP):
                ctrl, vecs = phantom_controls(lm0, spec[t], amp)
                try:
                    lm1 = lmk(warp_gaussian(rgb, ctrl, vecs, GAUSS_SIGMA * ipd)).lm_px[:, :2]
                except FaceNotFound:
                    continue
                D.append((frame(lm1) - F0).ravel())
                dt.append(compute_metrics_2d(np.c_[lm1, np.zeros(478)])[t] - m0)
        D, dt = np.array(D), np.array(dt)
        P["GAUSS_MP"][t] = np.linalg.lstsq(np.c_[np.ones(len(dt)), dt], D, rcond=None)[0][1]
    lmk.close()

    # --- personalised GAN prior from C3 (antithetic half-differences of the face's own code) -----------
    pers = {}
    for f in sorted(glob.glob(str(out / "private" / "c3_chunk_*.npz"))):
        z = np.load(f)
        pid, grp, k, sign, okc = z["pid"], z["group"], z["k"], z["sign"], z["ok"]
        for p in np.unique(pid):
            if p not in test:
                continue
            sel_p = pid == p
            plus = {(int(gg), int(kk)): i for i, (gg, kk, ss) in enumerate(zip(grp, k, sign)) if sel_p[i] and ss == 1 and okc[i]}
            minus = {(int(gg), int(kk)): i for i, (gg, kk, ss) in enumerate(zip(grp, k, sign)) if sel_p[i] and ss == -1 and okc[i]}
            keys = sorted(set(plus) & set(minus))
            if len(keys) < 12:
                continue
            ip, im = np.array([plus[q] for q in keys]), np.array([minus[q] for q in keys])
            D = ((frame(z["lm"][ip].astype(np.float64)) - frame(z["lm"][im].astype(np.float64))) / 2).reshape(len(keys), -1)
            N = (_nuis_from_bs(z["bs"][ip], z["pose"][ip]) - _nuis_from_bs(z["bs"][im], z["pose"][im])) / 2
            M = (z["m"][ip] - z["m"][im]) / 2
            pers[p] = {t: pattern(D, M[:, k_], N) for k_, t in enumerate(["nose_width", "nose_length", "palpebral_height",
                       "palpebral_width", "intercanthal", "brow_height", "face_width", "jaw_width",
                       "lower_face_height", "mouth_width"]) if t in TARGETS}

    # --- score test pairs ---------------------------------------------------------------------------
    rows = []
    for i, p in enumerate(test):
        Dr = Pa @ Dt[i]
        for t in TARGETS:
            dt = pairs_tab.loc[p, f"d_{t}"]
            if not np.isfinite(dt) or abs(dt) <= noise.loc[t, "mdc_real"]:
                continue
            rec = {"pid": p, "target": t, "d_real": dt, "rmse_b0": float(np.sqrt((Dr ** 2).mean()))}
            cand = {k: v[t] for k, v in P.items()}
            if p in pers:
                cand["GAN_PERS"] = pers[p][t]
            for name, pat in cand.items():
                pred = Pa @ (dt * pat)
                rec[f"cos_{name}"] = float(pred @ Dr / (np.linalg.norm(pred) * np.linalg.norm(Dr) + 1e-12))
                rec[f"rmse_gain_{name}"] = rec["rmse_b0"] - float(np.sqrt(((Dr - pred) ** 2).mean()))
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(out / "e7_pairs.csv", index=False)
    names = ["GAUSS_MP", "GAN_POP", "GAN_PERS", "HDA_EMP"]
    summ, comps = [], []
    for t in TARGETS:
        s = df[df.target == t]
        for nme in names:
            c = s[f"cos_{nme}"].dropna()
            gn = s[f"rmse_gain_{nme}"].dropna()
            summ.append({"target": t, "prior": nme, "n_pairs": len(c), "mean_cos": c.mean(),
                         "mean_rmse_gain": gn.mean(), "frac_better_than_b0": float((gn > 0).mean()) if len(gn) else np.nan})
        both = s.dropna(subset=[f"cos_{n}" for n in names])
        for a_, b_ in (("GAN_PERS", "GAUSS_MP"), ("GAN_POP", "GAUSS_MP"), ("GAN_PERS", "GAN_POP"), ("HDA_EMP", "GAN_POP")):
            for metric in ("cos", "rmse_gain"):
                m, lo, hi, pv = paired_boot(both[f"{metric}_{a_}"].to_numpy(), both[f"{metric}_{b_}"].to_numpy(),
                                            both.pid.to_numpy())
                comps.append({"target": t, "comparison": f"{a_} - {b_}", "score": metric, "n_pairs": len(both),
                              "diff": m, "ci": f"[{lo:.3f}, {hi:.3f}]", "p": pv})
    S, C = pd.DataFrame(summ), pd.DataFrame(comps)
    for sc in ("cos", "rmse_gain"):
        k = C.score == sc
        C.loc[k, "p_holm"] = holm(C.loc[k, "p"].to_numpy())
    S.to_csv(out / "e7_summary.csv", index=False)
    C.to_csv(out / "e7_comparisons.csv", index=False)
    pd.set_option("display.width", 220)
    print(S.round(4).to_string(index=False))
    print(C.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
