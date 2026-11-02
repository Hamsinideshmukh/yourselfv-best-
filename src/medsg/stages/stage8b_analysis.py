"""Stage 8b: analysis of the consensus-steering run, following outputs/eval_plan.json "stage8" (written before the run).

Usage: python -m medsg.stages.stage8b_analysis      (CPU, seconds). Writes outputs/stage8/analysis_*.csv.
Plain words:
  confirmed (S8-1) = the 3D face model (pose and expression removed) sees the target change in the requested direction
                     by more than its noise floor, the edit still looks like the real person (ArcFace >= tau), and no
                     protected MediaPipe feature moved beyond its noise floor (E3 criterion ii);
  efficiency (S8-2) = 3D change (noise-floor units, signed toward the request) per unit of latent step: equal across arms
                     means consensus steering only went further; higher means it moved the face better.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import get_config
from ..stats import holm
from .stage5b_analysis import boot_rate, paired_boot, protected_cols

SIGN = {"nose_width": -1, "palpebral_height": 1, "brow_height": 1}
D3 = {t: f"d3_{t}_3d_d" for t in SIGN}
COMPARISONS = [("nose_width", "B", "A"), ("palpebral_height", "B", "A"), ("palpebral_height", "C", "A")]


def _load(cfg):
    asset = np.load(cfg.out_dir.parent / "src" / "medsg" / "assets" / "editor_v1.npz")
    mdc = dict(zip([str(m) for m in asset["mdc_metrics"]], asset["mdc"].astype(float)))
    tau = float(pd.read_csv(cfg.out_dir / "stage5" / "analysis_identity.csv").tau_arc_dev_p05.iloc[0])
    n7 = pd.read_csv(cfg.out_dir / "stage7" / "k7_judge_noise.csv")
    mdc3 = {m.replace("d3_", ""): v for m, v in zip(n7.metric, n7.mdc95) if m.startswith("d3_")}
    return mdc, tau, mdc3


def score(d: pd.DataFrame, mdc: dict, tau: float, mdc3: dict) -> pd.DataFrame:
    d = d.copy()
    z3 = []
    for _, r in d.iterrows():
        col = D3[r.target]
        z3.append(SIGN[r.target] * r[col] / mdc3[f"{r.target}_3d"] if col in r and np.isfinite(r[col]) else np.nan)
    d["z3_signed"] = z3                                         # 3D change toward the request, in 3D noise floors
    d["c3d"] = d.z3_signed > 1
    d["c_identity"] = d.arc_edit_real >= tau
    d["c_protected"] = [bool(all(abs(r[f"d_{m}"]) <= mdc[m] for m in protected_cols(r.target)[0]))
                        if all(f"d_{m}" in r and np.isfinite(r[f"d_{m}"]) for m in protected_cols(r.target)[0])
                        else False for _, r in d.iterrows()]
    d["confirmed"] = d.c3d & d.c_identity & d.c_protected
    d["efficiency"] = d.z3_signed / d.latent_norm
    d["fan_sees_it"] = fan_sees(d)
    return d


def fan_sees(d: pd.DataFrame) -> pd.Series:
    """FAN change toward the request beyond 1 FAN noise floor (a judge for arm A only; steered on in B and C)."""
    return pd.Series([bool(SIGN[t] * z > 1) if np.isfinite(z) else False for t, z in zip(d.target, d.z_fan)],
                     index=d.index)


def tables(d: pd.DataFrame, noise: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (t, arm), g in d.groupby(["target", "arm"]):
        rec = {"target": t, "arm": arm, "n": len(g)}
        for c in ("confirmed", "c3d", "c_identity", "c_protected", "fan_sees_it"):
            m, lo, hi = boot_rate(g[c].astype(float).to_numpy(), g.pid.to_numpy())
            rec[c], rec[f"{c}_ci"] = m, f"[{lo:.2f}, {hi:.2f}]"
        rec.update({"z3_median": g.z3_signed.median(), "efficiency_median": g.efficiency.median(),
                    "latent_step_median": g.latent_norm.median(), "z_mp_median": g.z_mp.median(),
                    "z_fan_median": g.z_fan.median(), "arc_edit_real_median": g.arc_edit_real.median(),
                    "depth_locality_median": g.depth_locality.median(), "renders_median": g.renders.median(),
                    "status": json.dumps(g.status.value_counts().to_dict())})
        nc = f"{t}_depth_in"
        if nc in noise:
            rec["depth_in_above_noise_p95"] = float((g.depth_in > noise[nc].quantile(0.95)).mean())
        rows.append(rec)
    comps = []
    for t, a, b in COMPARISONS:
        x = d[(d.target == t) & (d.arm == a)].set_index("pid")
        y = d[(d.target == t) & (d.arm == b)].set_index("pid")
        c = x.index.intersection(y.index)
        for crit in ("confirmed", "efficiency"):
            xv, yv = x.loc[c, crit].astype(float).to_numpy(), y.loc[c, crit].astype(float).to_numpy()
            ok = np.isfinite(xv) & np.isfinite(yv)
            m, lo, hi, p = paired_boot(xv[ok], yv[ok], np.array(c)[ok])
            comps.append({"target": t, "comparison": f"{a} - {b}", "criterion": crit, "n_faces": int(ok.sum()),
                          "diff": m, "ci": f"[{lo:.3f}, {hi:.3f}]", "ci_lo": lo, "p": p})
    C = pd.DataFrame(comps)
    k = C.criterion == "confirmed"
    C.loc[k, "p_holm"] = holm(C.loc[k, "p"].to_numpy())             # primary family: the 3 S8-1 comparisons
    return pd.DataFrame(rows), C


def decision(T: pd.DataFrame, C: pd.DataFrame) -> pd.DataFrame:
    """The pre-registered rule, applied mechanically."""
    ref = float(T[(T.target == "brow_height") & (T.arm == "REF")].confirmed.iloc[0])
    out = []
    for t in ("nose_width", "palpebral_height"):
        best = None
        for arm in ("B", "C"):
            c = C[(C.target == t) & (C.comparison == f"{arm} - A") & (C.criterion == "confirmed")]
            if len(c):
                rate = float(T[(T.target == t) & (T.arm == arm)].confirmed.iloc[0])
                ok = bool(c.ci_lo.iloc[0] > 0 and c.p_holm.iloc[0] < 0.05 and rate >= ref)
                if best is None or rate > best[1]:
                    best = (arm, rate, ok, float(c.ci_lo.iloc[0]))
        eff = C[(C.target == t) & (C.comparison == f"{best[0]} - A") & (C.criterion == "efficiency")]
        out.append({"target": t, "best_consensus_arm": best[0], "confirmed_rate": best[1], "brow_reference": ref,
                    "beats_A_ci_lo": best[3], "efficiency_diff_ci": eff.ci.iloc[0] if len(eff) else "",
                    "offer_in_gan_mode": best[2]})
    return pd.DataFrame(out)


def detector_ratio(d: pd.DataFrame, cfg, mdc: dict) -> pd.DataFrame:
    """EXPLORATORY (not pre-registered). Does an edit move the two landmarkers in the proportion a REAL shape change
    would? Observed: FAN change / MediaPipe change, both in the metric's own units (z * MDC). Expected: the ratio of the
    two detectors' gains on known-deformation phantoms (Stage 5 C2, mean over the three widths). A real anatomical
    change should match the calibrated ratio; an edit that moves MediaPipe's points more than the anatomy falls below."""
    n7 = pd.read_csv(cfg.out_dir / "stage7" / "k7_judge_noise.csv")      # source of stage8_consensus.FAN_MDC
    FAN_MDC = {m.replace("fan_", ""): v for m, v in zip(n7.metric, n7.mdc95) if m.startswith("fan_")}
    c2 = pd.read_csv(cfg.out_dir / "stage5" / "analysis_c2_gain.csv")
    rows = []
    for (t, arm), g in d.groupby(["target", "arm"]):
        cg = c2[c2.metric == t]
        r = (cg.gain_fan / cg.gain_mediapipe).to_numpy()            # one calibrated ratio per phantom width
        obs = float(((g.z_fan * FAN_MDC[t]) / (g.z_mp * mdc[t])).median())
        rows.append({"target": t, "arm": arm, "observed_fan_over_mp": obs, "expected_median": float(np.median(r)),
                     "expected_range": f"[{r.min():.2f}, {r.max():.2f}]", "observed_over_expected": obs / np.median(r),
                     "within_calibrated_range": bool(r.min() <= obs <= r.max()),
                     "fan_z_if_real_change_at_2mdc_mp": 2 * float(np.median(r)) * mdc[t] / FAN_MDC[t]})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = get_config()
    o = cfg.out_dir / "stage8"
    mdc, tau, mdc3 = _load(cfg)
    d = pd.read_csv(o / "s1_arms.csv", dtype={"pid": str})
    noise = pd.read_csv(o / "s1_judge_noise_rows.csv", dtype={"pid": str})
    d = score(d[d.status.notna()], mdc, tau, mdc3)
    T, C = tables(d, noise)
    D = decision(T, C)
    R = detector_ratio(d, cfg, mdc)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for name, df in (("arms", T), ("comparisons", C), ("decision", D), ("detector_ratio_exploratory", R)):
        df.to_csv(o / f"analysis_{name}.csv", index=False)
        print(f"\n== {name}\n{df.round(3).to_string(index=False)}")
    app = o / "s2_app_hyperstyle.csv"
    if app.exists():
        print("\n== app GAN path (HyperStyle)\n" + pd.read_csv(app).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
