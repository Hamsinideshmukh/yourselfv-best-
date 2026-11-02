"""E4 (pre-registered, outputs/eval_plan.json "E4_outcome"): how close does a simulated post-op face come to the
REAL post-op photo of the same person?

Plain words. For every held-out test pair we make three guesses of the post-op face and compare each with the real
post-op photo:
  B0      no change: the pre-op photo itself (the baseline any method must beat);
  B2      typical change: the pre-op photo warped by the average real change seen in the DEV pairs (a real prediction;
          one pooled request, because the dataset lost its procedure labels);
  ORACLE  the pair's own real change (uses the answer: shows how close the simulator gets when the size is right).
Scores: landmark error (all 478 points, iris frame, IPD units), per-feature error, ArcFace closeness to the real
post-op photo. CPU, warp mode, ~10-15 min with 4 processes. Writes outputs/e4/rows.csv, summary.csv, metrics.csv.
"""
from __future__ import annotations

import json
import os
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

from ..align import align_ffhq
from ..config import get_config
from ..data import list_pairs
from ..dedup import EXCLUDE
from ..measure import MEAS_RES, measure_avg
from ..stats import holm
from ..transfer import COUPLED, IPD_MM, NAMES, Transfer, warp_photo
from .stage5b_analysis import paired_boot
from .stage5c_e7 import frame

APP_COUPLED = {**COUPLED, "jaw_width": [], "face_width": []}          # = app/app.py
METHODS = ("B0", "B2", "ORACLE")
_W: dict = {}                                                           # per-process models


def _init(models_dir: str) -> None:
    from ..judges import ArcFace
    from ..landmarks import Landmarker
    _W.update(lmk=Landmarker(os.path.join(models_dir, "face_landmarker.task")),
              arc=ArcFace(os.path.join(models_dir, "w600k_r50.onnx")), T=Transfer())


def _side(path, L0) -> tuple:
    """Photo -> FFHQ-aligned 512 px, k=4 averaged metrics, landmarks (px), or None if measuring fails."""
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    al = cv2.resize(align_ffhq(rgb, np.c_[L0, np.zeros(478)], 1024)[0], (MEAS_RES, MEAS_RES),
                    interpolation=cv2.INTER_AREA)
    m, f = measure_avg(al, _W["lmk"])
    return (None, None, None) if m is None else (al, m, f["lm"].astype(np.float64) * MEAS_RES)


def measure_pair(job) -> dict:
    """Dev pairs: real change only."""
    pid, pre, post, L_pre, L_post = job
    _, m0, _ = _side(pre, L_pre)
    _, m1, _ = _side(post, L_post)
    if m0 is None or m1 is None:
        return {"pid": pid, "status": "measure_fail"}
    return {"pid": pid, "status": "ok", **{f"d_{m}": v for m, v in zip(NAMES, m1 - m0)}}


def test_pair(job) -> list[dict]:
    """Test pairs: the three guesses vs the real post-op photo."""
    pid, pre, post, L_pre, L_post, req_b2, targets = job
    al0, m0, L0 = _side(pre, L_pre)
    al1, m1, L1 = _side(post, L_post)
    if m0 is None or m1 is None:
        return [{"pid": pid, "method": k, "status": "measure_fail"} for k in METHODS]
    arc, T, lmk = _W["arc"], _W["T"], _W["lmk"]
    e_post = arc.embed(al1, L1)
    cos_pre_post = float(arc.embed(al0, L0) @ e_post)
    ti = [NAMES.index(t) for t in targets]
    reqs = {"B0": None, "B2": req_b2, "ORACLE": {t: float(m1[i] - m0[i]) for t, i in zip(targets, ti)}}
    rows = []
    for k, req in reqs.items():
        rec = {"pid": pid, "method": k, "status": "ok", "cos_pre_post": cos_pre_post}
        if req is None:
            ms, Ls, cos = m0, L0, cos_pre_post
        else:
            try:
                e = T.edit_multi(L0, req, couplings=APP_COUPLED)
            except ValueError as err:
                rows.append({**rec, "status": f"edit_fail: {err}"})
                continue
            w = warp_photo(al0, L0, e.field_px)
            ms, f = measure_avg(w, lmk)
            if ms is None:
                rows.append({**rec, "status": "measure_fail_sim"})
                continue
            Ls = f["lm"].astype(np.float64) * MEAS_RES
            cos = float(arc.embed(w, Ls) @ e_post)
        rec["lm_err"] = float(np.linalg.norm(frame(Ls) - frame(L1), axis=-1).mean())
        rec["cos_sim_post"] = cos
        rec.update({f"err_{t}": float(abs(ms[i] - m1[i])) for t, i in zip(targets, ti)})
        rec.update({f"real_d_{t}": float(m1[i] - m0[i]) for t, i in zip(targets, ti)})
        rows.append(rec)
    return rows


def analyse(R: pd.DataFrame, targets: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = R[R.status == "ok"]
    piv = {c: ok.pivot(index="pid", columns="method", values=c) for c in ["lm_err", "cos_sim_post",
                                                                           *[f"err_{t}" for t in targets]]}
    S = []
    for k in ("B2", "ORACLE"):
        lm = piv["lm_err"][["B0", k]].dropna()
        m, lo, hi, p = paired_boot(lm.B0.to_numpy(), lm[k].to_numpy(), lm.index.to_numpy())
        cs = piv["cos_sim_post"][["B0", k]].dropna()
        g, glo, ghi, gp = paired_boot(cs[k].to_numpy(), cs.B0.to_numpy(), cs.index.to_numpy())
        S.append({"method": k, "n_pairs": len(lm), "n_failed": int((R.method == k).sum() - (ok.method == k).sum()),
                  "lm_err_B0_median": lm.B0.median(), "lm_err_method_median": lm[k].median(),
                  "lm_improvement_mean": m, "lm_improvement_ci": f"[{lo:.4f}, {hi:.4f}]", "lm_p": p,
                  "lm_improvement_mm_approx": m * IPD_MM, "frac_pairs_closer": float((lm[k] < lm.B0).mean()),
                  "arc_gain_mean": g, "arc_gain_ci": f"[{glo:.4f}, {ghi:.4f}]", "arc_gain_p": gp,
                  "arc_cos_pre_post_median": cs.B0.median(),
                  "meets_prereg_rule": bool(lo > 0) if k == "B2" else "n/a (uses post-op information)"})
    M = []
    for k in ("B2", "ORACLE"):
        for t in targets:
            x = piv[f"err_{t}"][["B0", k]].dropna()
            m, lo, hi, p = paired_boot(x.B0.to_numpy(), x[k].to_numpy(), x.index.to_numpy())
            M.append({"method": k, "metric": t, "n": len(x), "err_B0_median_ipd": x.B0.median(),
                      "err_method_median_ipd": x[k].median(), "improvement_mean_ipd": m,
                      "improvement_ci": f"[{lo:.4f}, {hi:.4f}]", "improvement_mm_approx": m * IPD_MM, "p": p,
                      "frac_pairs_closer": float((x[k] < x.B0).mean())})
    M = pd.DataFrame(M)
    for k in ("B2", "ORACLE"):
        s = M.method == k
        M.loc[s, "p_holm"] = holm(M.loc[s, "p"].to_numpy())
    return pd.DataFrame(S), M


def main(procs: int = 4) -> None:
    cfg = get_config()
    out = cfg.out_dir / "e4"
    out.mkdir(exist_ok=True)
    plan = json.load(open(cfg.out_dir / "eval_plan.json"))["E4_outcome"]
    targets = plan["targets"]
    split = json.load(open(cfg.out_dir / "splits.json"))
    inc = pd.read_csv(cfg.out_dir / "stage1/pairs.csv", dtype={"pid": str}).set_index("pid").include_pose
    by_pid = {p.pid: p for p in list_pairs(cfg)}
    L = np.load(cfg.out_dir / "stage1/private/landmarks.npz")

    def ids(s):
        return [p for p in split[s] if p not in EXCLUDE and bool(inc.get(p, False))]

    def base(p):
        return (p, by_pid[p].pre, by_pid[p].post, L[f"{p}_pre"], L[f"{p}_post"])
    with Pool(procs, _init, (str(cfg.models_dir),)) as pool:
        D = pd.DataFrame(pool.map(measure_pair, [base(p) for p in ids("dev")]))
        D.to_csv(out / "dev_real_change.csv", index=False)
        dok = D[D.status == "ok"]
        req_b2 = {t: float(dok[f"d_{t}"].mean()) for t in targets}
        print(f"dev pairs {len(dok)}/{len(D)}; B2 request (IPD):", {t: round(v, 4) for t, v in req_b2.items()},
              flush=True)
        jobs = [(*base(p), req_b2, targets) for p in ids("test")]
        R = pd.DataFrame([r for rows in pool.imap(test_pair, jobs, chunksize=4) for r in rows])
    R.to_csv(out / "rows.csv", index=False)
    S, M = analyse(R, targets)
    S.to_csv(out / "summary.csv", index=False)
    M.to_csv(out / "metrics.csv", index=False)
    json.dump({"b2_request_ipd": req_b2, "n_dev": len(dok), "plan": plan}, open(out / "meta.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(S.round(4).to_string(index=False), "\n", M.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
