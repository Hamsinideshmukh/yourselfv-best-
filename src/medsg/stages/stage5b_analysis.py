"""Stage 5b: analysis of the overnight run (CPU, minutes). Follows outputs/eval_plan.json (frozen 2026-10-31).

Usage: python -m medsg.stages.stage5b_analysis --split dev     (always first)
       python -m medsg.stages.stage5b_analysis --split test    (once, after the dev review)

Writes outputs/stage5/analysis_<split>_*.csv.
Plain words for the E3 success test of one edit (all four must hold; compared with the unedited rebuild G(E(x))):
 (i)   the target moved to the request within half its noise floor (|achieved - request| <= 0.5 MDC);
 (ii)  nothing protected moved beyond noise: every other metric (minus declared clinical couplings) <= 1 MDC,
       head pose <= 2 deg, expression read-outs <= 0.05 (editspec tolerances);
 (iii) identity kept: ArcFace cos(edit, rebuild) >= tau_arc (5th pct of real pre-vs-post ArcFace, dev pairs);
 (iv)  the achieved target change lies inside the HDA dev range (2.5-97.5 pct of real changes).
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import pandas as pd

from .. import editspec as E
from ..config import get_config
from ..dedup import EXCLUDE
from ..metrics2d import METRICS_2D
from ..stats import holm

M10 = list(METRICS_2D)
POSE = ["yaw", "pitch", "roll"]
BS = {"bs_smile": ["d_bs_mouthSmileLeft", "d_bs_mouthSmileRight"], "bs_blink": ["d_bs_eyeBlinkLeft", "d_bs_eyeBlinkRight"],
      "bs_squint": ["d_bs_eyeSquintLeft", "d_bs_eyeSquintRight"], "bs_browDown": ["d_bs_browDownLeft", "d_bs_browDownRight"],
      "bs_jawOpen": ["d_bs_jawOpen"], "bs_browInnerUp": ["d_bs_browInnerUp"]}
N_BOOT = 10_000


def boot_rate(x: np.ndarray, groups: np.ndarray, seed: int = 0, n: int = N_BOOT) -> tuple[float, float, float]:
    """Mean of x with a cluster (subject) bootstrap 95 % CI; x is 0/1 or continuous, one value per row."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, groups = x[ok], np.asarray(groups)[ok]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    ug, inv = np.unique(groups, return_inverse=True)
    sums, cnts = np.bincount(inv, x), np.bincount(inv)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ug), (n, len(ug)))
    est = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(x.mean()), float(np.quantile(est, 0.025)), float(np.quantile(est, 0.975))


def paired_boot(a: np.ndarray, b: np.ndarray, groups: np.ndarray, seed: int = 0, n: int = N_BOOT):
    """Mean difference a-b (paired rows) with subject-bootstrap CI and a two-sided bootstrap p-value."""
    d = np.asarray(a, float) - np.asarray(b, float)
    m, lo, hi = boot_rate(d, groups, seed, n)
    ok = np.isfinite(d)
    ug, inv = np.unique(np.asarray(groups)[ok], return_inverse=True)
    sums, cnts = np.bincount(inv, d[ok]), np.bincount(inv)
    idx = np.random.default_rng(seed + 1).integers(0, len(ug), (n, len(ug)))
    est = sums[idx].sum(1) / cnts[idx].sum(1)
    p = float(min(1.0, 2 * min((est <= 0).mean(), (est >= 0).mean())))
    return m, lo, hi, max(p, 1.0 / n)


def load_edits(cfg) -> pd.DataFrame:
    files = sorted(glob.glob(str(cfg.out_dir / "stage5" / "c4_edits_*.csv")))
    df = pd.concat([pd.read_csv(f, dtype={"pid": str}) for f in files], ignore_index=True)
    assert not df.pid.isin(list(EXCLUDE)).any(), "excluded pairs present in edits"
    for k, cols in BS.items():
        df[f"d_{k}"] = df[cols].mean(1)
    return df


def protected_cols(t: str) -> tuple[list[str], list[str]]:
    """Protected metric and nuisance names for target t (editspec; declared couplings removed)."""
    mets = [m for m in M10 if m != t and m not in E.COUPLED.get(t, [])]
    nuis = [n for n in E.NUISANCE if n not in E.COUPLED_NUISANCE.get(t, [])]
    return mets, nuis


def score(df: pd.DataFrame, mdc: dict, env: dict, tau: float) -> pd.DataFrame:
    """Add criterion columns c1..c4 and success to rows that represent a finished edit."""
    lo = dict(zip(env["metrics"], env["p025"]))
    hi = dict(zip(env["metrics"], env["p975"]))
    out = df.copy()
    c1, c2m, c2n, c3, c4, worst = [], [], [], [], [], []
    for _, r in out.iterrows():
        t = r.target
        ach = r.get(f"d_{t}", np.nan)
        c1.append(abs(ach - r.request) <= E.LOOP_TARGET_TOL_MDC * mdc[t] if np.isfinite(ach) else False)
        mets, nuis = protected_cols(t)
        lm = np.array([abs(r[f"d_{m}"]) / mdc[m] for m in mets])
        ln = np.array([abs(r[f"d_{n}"]) / E.NUISANCE_TOL[n] for n in nuis])
        c2m.append(bool(np.all(lm <= 1)) if np.isfinite(lm).all() else False)
        c2n.append(bool(np.all(ln <= 1)) if np.isfinite(ln).all() else False)
        allv = dict(zip(mets + nuis, np.r_[lm, ln]))
        worst.append(max(allv, key=lambda k: allv[k] if np.isfinite(allv[k]) else -1) if allv else "")
        c3.append(r.get("arc_cos_edit_base", np.nan) >= tau if np.isfinite(r.get("arc_cos_edit_base", np.nan)) else np.nan)
        c4.append(lo[t] <= ach <= hi[t] if np.isfinite(ach) else False)
    out["c1_target"], out["c2_metrics"], out["c2_nuisance"] = c1, c2m, c2n
    out["c2_all"] = out.c2_metrics & out.c2_nuisance
    out["c3_identity"], out["c4_envelope"], out["worst_protected"] = c3, c4, worst
    out["success"] = out.c1_target & out.c2_all & (out.c3_identity == True) & out.c4_envelope   # noqa: E712
    out["success_no_id"] = out.c1_target & out.c2_all & out.c4_envelope
    return out


def editor_tables(df: pd.DataFrame, split: str, mdc: dict, env: dict, tau: float, out_dir) -> pd.DataFrame:
    d = df[df.split == split]
    clo = d[(d.method == "closed")]
    fin = clo[clo.final == True].copy()                                 # noqa: E712
    first = clo[clo["iter"] == 0].copy()                                # open loop = closed loop's first render
    b4 = d[(d.method == "b4") & (d.status == "ok")].copy()
    S = {"closed": score(fin.dropna(subset=["iter"]), mdc, env, tau), "open": score(first, mdc, env, tau),
         "b4": score(b4, mdc, env, tau)}
    rows, comps = [], []
    for kind in sorted(d.request_kind.unique()):
        for t in sorted(d[d.request_kind == kind].target.unique()):
            sub = {k: v[(v.request_kind == kind) & (v.target == t)] for k, v in S.items()}
            n_req = int(((fin.request_kind == kind) & (fin.target == t)).sum())
            for meth, s in sub.items():
                rec = {"split": split, "request_kind": kind, "target": t, "method": meth, "n_edits": len(s),
                       "n_requested": n_req, "request": s.request.median() if len(s) else np.nan,
                       "median_achieved": s[f"d_{t}"].median() if len(s) else np.nan}
                for c in ("success", "success_no_id", "c1_target", "c2_metrics", "c2_nuisance", "c4_envelope"):
                    m, lo, hi = boot_rate(s[c].astype(float).to_numpy(), s.pid.to_numpy()) if len(s) else (np.nan,) * 3
                    rec[c], rec[f"{c}_ci"] = m, f"[{lo:.2f}, {hi:.2f}]"
                c3 = s.c3_identity.astype(float).to_numpy() if len(s) else np.array([])
                rec["c3_identity"] = np.nanmean(c3) if np.isfinite(c3).any() else np.nan
                rec["median_abs_err_mdc"] = float(np.median(np.abs(s[f"d_{t}"] - s.request) / mdc[t])) if len(s) else np.nan
                rec["median_arc_cos_edit_base"] = s.arc_cos_edit_base.median() if "arc_cos_edit_base" in s and len(s) else np.nan
                if meth == "closed" and len(s):
                    rec["median_renders"] = float((s["iter"] + 1).median())
                    rec["status_counts"] = json.dumps(s.status.value_counts().to_dict())
                    fan = s.get(f"fan_d_{t}")
                    rec["fan_spearman_with_mp"] = float(pd.Series(s[f"d_{t}"]).corr(fan, method="spearman")) if fan is not None and fan.notna().sum() > 5 else np.nan
                    rec["worst_protected_mode"] = s.worst_protected.mode().iloc[0] if len(s) else ""
                rows.append(rec)
            # comparisons (paired by face)
            for other, crit in (("open", "success_no_id"), ("b4", "success")):
                a = sub["closed"].set_index("pid")
                b = sub[other].set_index("pid")
                common = a.index.intersection(b.index)
                if len(common) < 5:
                    continue
                m, lo, hi, p = paired_boot(a.loc[common, crit].astype(float).to_numpy(),
                                           b.loc[common, crit].astype(float).to_numpy(), np.array(common))
                comps.append({"split": split, "request_kind": kind, "target": t, "comparison": f"closed - {other}",
                              "criterion": crit, "n_faces": len(common), "diff": m, "ci": f"[{lo:.2f}, {hi:.2f}]", "p": p})
    tab, cmp_ = pd.DataFrame(rows), pd.DataFrame(comps)
    if len(cmp_):
        cmp_["p_holm"] = np.nan
        for kind in cmp_.request_kind.unique():
            k = cmp_.request_kind == kind
            cmp_.loc[k, "p_holm"] = holm(cmp_.loc[k, "p"].to_numpy())
    tab.to_csv(out_dir / f"analysis_{split}_editor.csv", index=False)
    cmp_.to_csv(out_dir / f"analysis_{split}_editor_comparisons.csv", index=False)
    pd.concat([v.assign(method=k) for k, v in S.items()]).to_csv(out_dir / f"analysis_{split}_editor_rows.csv", index=False)
    return tab


def identity_table(cfg, split: dict, out_dir) -> tuple[pd.DataFrame, float]:
    """ArcFace yardstick (dev) and inversion identity; FaceNet values from Stage 4a for agreement (R10)."""
    Z = np.load(cfg.out_dir / "stage5" / "private" / "c1_landmarks_embeddings.npz")
    rows, tau = [], np.nan
    for which in ("dev", "test"):
        ids = [p for p in split[which] if p not in EXCLUDE and f"arc_real_{p}_pre" in Z.files and f"arc_real_{p}_post" in Z.files]
        g = np.array([Z[f"arc_real_{p}_pre"] @ Z[f"arc_real_{p}_post"] for p in ids])
        rec = np.array([Z[f"arc_real_{p}_pre"] @ Z[f"arc_rec_{p}_pre"] for p in ids])
        X = np.stack([Z[f"arc_real_{p}_pre"] for p in ids])
        imp = (X @ X.T)[np.triu_indices(len(ids), 1)]
        if which == "dev":
            tau = float(np.quantile(g, 0.05))
        rows.append({"split": which, "n": len(ids), "arc_real_pre_post_median": np.median(g),
                     "arc_real_pre_post_p05": np.quantile(g, 0.05), "arc_e4e_rec_vs_real_median": np.median(rec),
                     "arc_e4e_frac_above_tau": float((rec >= tau).mean()), "arc_diff_person_p99": np.quantile(imp, 0.99)})
    c5 = pd.read_csv(cfg.out_dir / "stage5" / "c5_pti.csv", dtype={"pid": str})
    reb = c5[c5.variant == "rebuild"]
    t = pd.DataFrame(rows)
    t["tau_arc_dev_p05"] = tau
    t["pti_arc_rebuild_vs_real_median_test30"] = reb.arc_cos_real.median()
    t["pti_frac_above_tau"] = float((reb.arc_cos_real >= tau).mean())
    t.to_csv(out_dir / "analysis_identity.csv", index=False)
    return t, tau


def _slope(x: pd.Series, y: pd.Series) -> float:
    ok = x.notna() & y.notna()
    return float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() > 10 else np.nan


def drift_table(cfg, out_dir) -> pd.DataFrame:
    """Inversion drift (rebuild - aligned photo) seen by MediaPipe and by FAN, included pairs."""
    c1 = pd.read_csv(cfg.out_dir / "stage5" / "c1_images.csv", dtype={"pid": str})
    c1 = c1[c1.ok & ~c1.excluded]
    rows = []
    for m in ["nose_width", "mouth_width", "jaw_width", "lower_face_height", "brow_height", "palpebral_height"]:
        dm = c1[f"mp_rec_{m}"] - c1[f"mp_real_{m}"]
        df_ = c1[f"fan_rec_{m}"] - c1[f"fan_real_{m}"]
        rows.append({"metric": m, "n": len(c1), "mp_bias": dm.mean(), "mp_median_abs": dm.abs().median(),
                     "fan_bias": df_.mean(), "fan_median_abs": df_.abs().median(),
                     "spearman_mp_vs_fan_drift": dm.corr(df_, method="spearman"),
                     "mp_rec_on_real_slope": _slope(c1[f"mp_real_{m}"], c1[f"mp_rec_{m}"]),
                     "fan_rec_on_real_slope": _slope(c1[f"fan_real_{m}"], c1[f"fan_rec_{m}"])})
    t = pd.DataFrame(rows)
    t.to_csv(out_dir / "analysis_drift_mp_vs_fan.csv", index=False)
    return t


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    a = ap.parse_args(argv)
    cfg = get_config()
    out = cfg.out_dir / "stage5"
    split = json.load(open(cfg.out_dir / "splits.json"))
    asset = np.load(cfg.out_dir.parent / "src" / "medsg" / "assets" / "editor_v1.npz")
    mdc = dict(zip([str(m) for m in asset["mdc_metrics"]], asset["mdc"]))
    env = json.load(open(cfg.out_dir / "stage4b" / "envelope.json"))
    pd.set_option("display.width", 250)
    idt, tau = identity_table(cfg, split, out)
    print(idt.round(3).to_string(index=False))
    if a.split == "dev":
        print(drift_table(cfg, out).round(4).to_string(index=False))
    df = load_edits(cfg)
    tab = editor_tables(df, a.split, mdc, env, tau, out)
    show = ["request_kind", "target", "method", "n_edits", "request", "median_achieved", "success", "success_ci",
            "success_no_id", "c1_target", "c2_metrics", "c2_nuisance", "c3_identity", "c4_envelope",
            "median_abs_err_mdc", "median_arc_cos_edit_base"]
    print(tab[show].round(3).to_string(index=False))
    extra = tab[tab.method == "closed"][["request_kind", "target", "median_renders", "status_counts",
                                         "fan_spearman_with_mp", "worst_protected_mode"]]
    print(extra.to_string(index=False))
    print(pd.read_csv(out / f"analysis_{a.split}_editor_comparisons.csv").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
