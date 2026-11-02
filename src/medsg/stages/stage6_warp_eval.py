"""Stage 6: pre-registered evaluation of the deformation-transfer output (A1; outputs/eval_plan.json "A1_warp_mode").

Test included pre-op photos, FFHQ-aligned 512 px (same crop as Stage 5). Requests in exact geometry, sized from
published mean surgical changes (approx. mm via IPD 63.36 mm). For each edit:
  exact achieved / protected change (by construction), MediaPipe re-measured change (k = 4 averaged) -> gain for
  anatomically shaped deformations, MediaPipe leakage in MDC units (E3 criterion ii), ArcFace identity vs the
  original photo, largest landmark motion in approx. mm. The GAN closed-loop identity on the same faces is joined
  for comparison (what the patient would see: edit vs their own photo).
CPU, ~20-40 min. Writes outputs/stage6/warp_rows.csv, warp_summary.csv; examples to outputs/stage6/private/.
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd

from ..align import align_ffhq
from ..config import get_config
from ..data import list_pairs
from ..dedup import EXCLUDE
from ..judges import ArcFace
from ..landmarks import Landmarker
from ..transfer import NAMES, Transfer, warp_photo
from ..measure import MEAS_RES, measure_avg
from .stage5b_analysis import BS, protected_cols, boot_rate

N_EXAMPLES = 8


def _save_if_chunk_end(rows: list, part, n: int, total: int, chunk: int) -> None:
    if n % chunk == chunk - 1 or n == total - 1:
        pd.DataFrame(rows).to_csv(part, index=False)
        print(f"  {n + 1}/{total}", flush=True)


def main() -> None:
    cfg = get_config()
    out = cfg.out_dir / "stage6"
    (out / "private").mkdir(parents=True, exist_ok=True)
    plan = json.load(open(cfg.out_dir / "eval_plan.json"))["A1_warp_mode"]
    reqs = plan["requests_ipd_units"]
    split = json.load(open(cfg.out_dir / "splits.json"))
    test = [p for p in split["test"] if p not in EXCLUDE]
    by_pid = {p.pid: p for p in list_pairs(cfg)}
    L0s = np.load(cfg.out_dir / "stage1/private/landmarks.npz")
    kaggle = np.load(cfg.out_dir / "stage5/private/c1_landmarks_embeddings.npz")
    asset = np.load(cfg.out_dir.parent / "src/medsg/assets/editor_v1.npz")
    mdc = dict(zip([str(m) for m in asset["mdc_metrics"]], asset["mdc"]))
    tau = float(pd.read_csv(cfg.out_dir / "stage5/analysis_identity.csv").tau_arc_dev_p05.iloc[0])
    lmk, arc, T = Landmarker(cfg.models_dir / "face_landmarker.task"), ArcFace(cfg.models_dir / "w600k_r50.onnx"), Transfer()
    rows, n_ex = [], 0
    parts = out / "parts"
    parts.mkdir(exist_ok=True)
    CHUNK = 40                                            # checkpoint: one CSV per 40 faces; finished chunks skipped
    for n, pid in enumerate(test):
        part = parts / f"warp_rows_{n // CHUNK:03d}.csv"
        if part.exists():
            continue
        if n % CHUNK == 0:
            rows = []
        rgb = cv2.cvtColor(cv2.imread(str(by_pid[pid].pre)), cv2.COLOR_BGR2RGB)
        al = cv2.resize(align_ffhq(rgb, np.c_[L0s[f"{pid}_pre"], np.zeros(478)], 1024)[0], (MEAS_RES, MEAS_RES),
                        interpolation=cv2.INTER_AREA)
        m0, f0 = measure_avg(al, lmk)
        if m0 is None:
            rows.append({"pid": pid, "status": "base_measure_fail"})
            _save_if_chunk_end(rows, part, n, len(test), CHUNK)
            continue
        L = f0["lm"].astype(np.float64) * MEAS_RES
        e0 = arc.embed(al, L)
        k_emb = kaggle[f"arc_real_{pid}_pre"] if f"arc_real_{pid}_pre" in kaggle.files else None
        for t, r in reqs.items():
            rec = {"pid": pid, "target": t, "request": r, "status": "ok",
                   "arc_local_vs_kaggle": float(e0 @ k_emb) if k_emb is not None else np.nan}
            try:
                e = T.edit(L, t, r)
            except ValueError as err:
                rows.append({**rec, "status": f"edit_fail: {err}"})
                continue
            w = warp_photo(al, L, e.field_px)
            mavg, f1 = measure_avg(w, lmk)
            rec.update({"kappa": e.kappa, "exact_achieved": e.after[t] - e.before[t],
                        "max_motion_mm": float(np.linalg.norm(e.recipe_mm(L), axis=1).max()),
                        **{f"exact_d_{m}": e.after[m] - e.before[m] for m in NAMES}})
            if mavg is None:
                rows.append({**rec, "status": "measure_fail"})
                continue
            rec.update({f"mp_d_{m}": v for m, v in zip(NAMES, mavg - m0)})
            rec.update({f"mp_d_{k}": v for k, v in zip(("yaw", "pitch", "roll"), f1["pose"] - f0["pose"])})
            bsd = dict(zip(["jawOpen", "mouthSmileLeft", "mouthSmileRight", "eyeBlinkLeft", "eyeBlinkRight",
                            "eyeSquintLeft", "eyeSquintRight", "browInnerUp", "browDownLeft", "browDownRight"],
                           f1["bs"] - f0["bs"]))
            for k, cols in BS.items():
                rec[f"mp_d_{k}"] = float(np.mean([bsd[c.replace("d_bs_", "")] for c in cols]))
            rec["arc_cos_edit_original"] = float(arc.embed(w, f1["lm"].astype(np.float64) * MEAS_RES) @ e0)
            if t in mdc:                                   # E3 criterion (ii) on the frontal-metric targets
                mets, nuis = protected_cols(t)
                rec["c2_metrics"] = bool(all(abs(rec[f"mp_d_{m}"]) / mdc[m] <= 1 for m in mets))
            rows.append(rec)
            if n_ex < N_EXAMPLES and n % 50 == 0:
                cv2.imwrite(str(out / "private" / f"{pid}_{t}.jpg"), cv2.cvtColor(w, cv2.COLOR_RGB2BGR))
        if n_ex < N_EXAMPLES and n % 50 == 0:
            cv2.imwrite(str(out / "private" / f"{pid}_original.jpg"), cv2.cvtColor(al, cv2.COLOR_RGB2BGR))
            n_ex += 1
        _save_if_chunk_end(rows, part, n, len(test), CHUNK)
    lmk.close()
    df = pd.concat([pd.read_csv(f, dtype={"pid": str}) for f in sorted(parts.glob("warp_rows_*.csv"))],
                   ignore_index=True)
    df.to_csv(out / "warp_rows.csv", index=False)
    gan = pd.read_csv(cfg.out_dir / "stage5/analysis_test_editor_rows.csv", dtype={"pid": str})
    gan = gan[(gan.method == "closed") & (gan.request_kind == "clinical_2mdc")]
    ok = df[df.status == "ok"]
    summ = []
    for t in reqs:
        s = ok[ok.target == t]
        ratio = s[f"mp_d_{t}"] / s.exact_achieved
        g_m, g_lo, g_hi = boot_rate(ratio.to_numpy(), s.pid.to_numpy())
        a_m, a_lo, a_hi = boot_rate((s.arc_cos_edit_original >= tau).astype(float).to_numpy(), s.pid.to_numpy())
        rec = {"target": t, "request_ipd": reqs[t], "request_mm_approx": plan["requests_exact_geometry_mm_approx"][t],
               "n": len(s), "n_failed": int((df.target == t).sum() - len(s)), "kappa_median": s.kappa.median(),
               "exact_achieved_median": s.exact_achieved.median(), "mp_measured_median": s[f"mp_d_{t}"].median(),
               "mp_gain_mean": g_m, "mp_gain_ci": f"[{g_lo:.2f}, {g_hi:.2f}]",
               "arc_cos_edit_original_median": s.arc_cos_edit_original.median(),
               "arc_frac_above_tau": a_m, "arc_frac_ci": f"[{a_lo:.2f}, {a_hi:.2f}]",
               "max_motion_mm_median": s.max_motion_mm.median(),
               "c2_metrics_mp": s.c2_metrics.mean() if "c2_metrics" in s and s.c2_metrics.notna().any() else np.nan,
               "arc_local_vs_kaggle_min": ok.arc_local_vs_kaggle.min()}
        gt = gan[gan.target == t]
        if len(gt):
            rec["gan_closed_arc_cos_edit_original_median"] = gt.arc_cos_edit_real.median()
            rec["gan_closed_arc_frac_above_tau"] = float((gt.arc_cos_edit_real >= tau).mean())
        summ.append(rec)
    S = pd.DataFrame(summ)
    S.to_csv(out / "warp_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(S.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
