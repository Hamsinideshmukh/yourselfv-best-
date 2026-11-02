"""Stage 7b: analysis of the Stage 7 run, following outputs/eval_plan.json "stage7" (written before the run).

Usage: python -m medsg.stages.stage7b_analysis        (CPU, ~1 min). Writes outputs/stage7/analysis_*.csv.
Tables:
  inversion   K2: identity (ArcFace vs the real photo), LPIPS, landmark drift in noise-floor units, speed; paired
              subject-bootstrap differences vs official e4e (S7-1).
  editor      K3: the unchanged E3 scorer (stage5b.score) per encoder; differences vs e4e with Holm (S7-2, S7-3).
  repro       K3 e4e vs the Stage 5 e4e run on the same faces and requests (does the editor reproduce?).
  pti         K4: identity vs seconds for PTI 300 / 100 steps and both pivots, HyperStyle from K2 (S7-5).
  realism     K5: KID vs real photos (S7-4).
  depth_gan   K5: 3DDFA canonical shape change (MDC units) and Depth Anything locality for GAN edits (S7-7).
  warp        K7: FAN gain and FAN-measured leakage, 3D shape change, depth locality for warp edits (S7-6, S7-7).
"""
from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd

from ..config import get_config
from ..stats import holm
from .stage5b_analysis import BS, boot_rate, editor_tables, paired_boot, score

ENC = ["e4e", "e4e_hda", "hyperstyle"]
TARGET_FAN = {"nose_width": "nose_width", "brow_height": "brow_height", "jaw_width": "jaw_width",
              "mouth_width": "mouth_width", "palpebral_height": "palpebral_height", "mrd1": "palpebral_height"}
TARGET_3D = {k: f"{v}_3d" for k, v in TARGET_FAN.items()}
GAN_TARGETS = ["brow_height", "mouth_width", "jaw_width"]
SIGN = {"nose_width": -1, "jaw_width": -1, "brow_height": 1, "palpebral_height": 1, "mouth_width": 1, "mrd1": 1}
# = signs of the clinical requests (stage5_overnight.CLIN_REQ_MDC; warp requests in stage7_final.WARP_REQ_MM)


def _load(cfg):
    asset = np.load(cfg.out_dir.parent / "src" / "medsg" / "assets" / "editor_v1.npz")
    mdc = dict(zip([str(m) for m in asset["mdc_metrics"]], asset["mdc"].astype(float)))
    env = json.load(open(cfg.out_dir / "stage4b" / "envelope.json"))
    tau = float(pd.read_csv(cfg.out_dir / "stage5" / "analysis_identity.csv").tau_arc_dev_p05.iloc[0])
    return mdc, env, tau


def inversion(o, mdc, tau) -> pd.DataFrame:
    k = pd.read_csv(o / "k2_inversion.csv", dtype={"pid": str})
    k = k[k.encoder.isin(ENC)]
    piv = k.pivot(index="pid", columns="encoder", values="arc_cos_real_rec")
    rows = []
    for e in ENC:
        s = k[k.encoder == e]
        m, lo, hi = boot_rate((s.arc_cos_real_rec >= tau).astype(float).to_numpy(), s.pid.to_numpy())
        drift = {f"mp_drift_mdc_{t}": float((s[f"mp_drift_{t}"].abs() / mdc[t]).median()) for t in mdc}
        r = {"encoder": e, "n": len(s), "arc_median": s.arc_cos_real_rec.median(), "frac_above_tau": m,
             "frac_ci": f"[{lo:.2f}, {hi:.2f}]", "lpips_median": s.lpips_real_rec.median(),
             "seconds_median": s.seconds.median(), "mp_drift_mdc_median_over_metrics": float(np.median(list(drift.values()))),
             **drift}
        if e != "e4e":
            c = piv[[e, "e4e"]].dropna()
            d, dlo, dhi, p = paired_boot(c[e].to_numpy(), c["e4e"].to_numpy(), c.index.to_numpy())
            r.update(arc_diff_vs_e4e=d, arc_diff_ci=f"[{dlo:.3f}, {dhi:.3f}]", arc_diff_p=p)
        rows.append(r)
    return pd.DataFrame(rows)


def _edits(o, enc: str) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(f, dtype={"pid": str}) for f in sorted(glob.glob(str(o / f"k3_{enc}_edits_*.csv")))],
                   ignore_index=True)
    for k, cols in BS.items():
        df[f"d_{k}"] = df[cols].mean(axis=1)
    df["method"] = df["method"].fillna("closed")
    return df


def editor(o, mdc, env, tau) -> tuple[pd.DataFrame, pd.DataFrame]:
    tabs, fins = [], {}
    for e in ENC:
        df = _edits(o, e)
        sub = o / f"_tmp_{e}"
        sub.mkdir(exist_ok=True)
        t = editor_tables(df, "test", mdc, env, tau, sub)
        t["encoder"] = e
        clo = df[(df.method == "closed") & (df.final == True)].dropna(subset=["iter"])  # noqa: E712
        f = score(clo, mdc, env, tau)
        agg = f.groupby("target").agg(arc_edit_real_median=("arc_cos_edit_real", "median"),
                                      arc_edit_real_above_tau=("arc_cos_edit_real", lambda x: float((x >= tau).mean())))
        t = t.merge(agg, left_on="target", right_index=True, how="left")
        tabs.append(t)
        fins[e] = f.set_index(["pid", "target"])
        for x in sub.iterdir():
            x.unlink()
        sub.rmdir()
    tab = pd.concat(tabs, ignore_index=True)
    comps = []
    for e in ("e4e_hda", "hyperstyle"):
        for t in sorted(fins["e4e"].index.get_level_values(1).unique()):
            a, b = fins[e].xs(t, level=1), fins["e4e"].xs(t, level=1)
            common = a.index.intersection(b.index)
            for crit in ("success_no_id", "success"):
                m, lo, hi, p = paired_boot(a.loc[common, crit].astype(float).to_numpy(),
                                           b.loc[common, crit].astype(float).to_numpy(), np.array(common))
                comps.append({"encoder": e, "target": t, "criterion": crit, "n_faces": len(common), "diff_vs_e4e": m,
                              "ci": f"[{lo:.2f}, {hi:.2f}]", "ci_lo": lo, "p": p,
                              "noninferior_margin_0.10": bool(lo > -0.10) if crit == "success_no_id" else np.nan})
    c = pd.DataFrame(comps)
    for (e, crit), g in c.groupby(["encoder", "criterion"]):
        c.loc[g.index, "p_holm"] = holm(g.p.to_numpy())
    return tab, c


def editor_real(o, mdc, env, tau) -> pd.DataFrame:
    """POST-HOC composite (defined after seeing Stage 7, labelled as such): an edit succeeds for the patient if it
    passes E3 (i), (ii), (iv) AND ArcFace(edit, REAL photo) >= tau — S7-3's pre-registered identity criterion — instead
    of identity vs the encoder's own rebuild (which flatters an encoder whose rebuild is already not the person)."""
    F = {}
    for e in ENC:
        d = _edits(o, e)
        f = score(d[(d.method == "closed") & (d.final == True)].dropna(subset=["iter"]), mdc, env, tau)  # noqa: E712
        f["success_real"] = f.success_no_id & (f.arc_cos_edit_real >= tau)
        F[e] = f.set_index(["pid", "target"])
    rows = []
    for t in sorted(F["e4e"].index.get_level_values(1).unique()):
        r = {"target": t, "definition": "post-hoc: E3 (i),(ii),(iv) + ArcFace(edit, real) >= tau"}
        for e in ENC:
            x = F[e].xs(t, level=1)
            m, lo, hi = boot_rate(x.success_real.astype(float).to_numpy(), x.index.to_numpy())
            r.update({f"{e}": m, f"{e}_lo": lo, f"{e}_hi": hi})
        a, b = F["hyperstyle"].xs(t, level=1), F["e4e"].xs(t, level=1)
        c = a.index.intersection(b.index)
        m, lo, hi, p = paired_boot(a.loc[c, "success_real"].astype(float).to_numpy(),
                                   b.loc[c, "success_real"].astype(float).to_numpy(), np.array(c))
        r.update(hyperstyle_minus_e4e=m, diff_ci=f"[{lo:.2f}, {hi:.2f}]", p=p)
        rows.append(r)
    return pd.DataFrame(rows)


def repro(cfg, o, mdc, env, tau) -> pd.DataFrame:
    """Same editor, same faces and requests, two Kaggle sessions (Stage 5 vs Stage 7): success rates should match."""
    s5 = pd.read_csv(cfg.out_dir / "stage5" / "analysis_test_editor.csv")
    s5 = s5[(s5.method == "closed") & (s5.request_kind == "clinical_2mdc")][["target", "success", "success_no_id"]]
    s7 = _edits(o, "e4e")
    sub = o / "_tmp_repro"
    sub.mkdir(exist_ok=True)
    t7 = editor_tables(s7, "test", mdc, env, tau, sub)
    for x in sub.iterdir():
        x.unlink()
    sub.rmdir()
    t7 = t7[t7.method == "closed"][["target", "success", "success_no_id"]]
    return s5.merge(t7, on="target", suffixes=("_stage5", "_stage7"))


def pti(o, tau) -> pd.DataFrame:
    k4 = pd.read_csv(o / "k4_pti.csv", dtype={"pid": str})
    k2 = pd.read_csv(o / "k2_inversion.csv", dtype={"pid": str})
    hs = k2[(k2.encoder == "hyperstyle") & k2.pid.isin(k4.pid)]
    e4 = k2[(k2.encoder == "e4e") & k2.pid.isin(k4.pid)]
    rows = [{"method": "e4e (no tuning)", "n": len(e4), "seconds": e4.seconds.median(),
             "arc_median": e4.arc_cos_real_rec.median(), "frac_above_tau": (e4.arc_cos_real_rec >= tau).mean(),
             "lpips": e4.lpips_real_rec.median()},
            {"method": "HyperStyle", "n": len(hs), "seconds": hs.seconds.median(),
             "arc_median": hs.arc_cos_real_rec.median(), "frac_above_tau": (hs.arc_cos_real_rec >= tau).mean(),
             "lpips": hs.lpips_real_rec.median()}]
    for (pv, st), g in k4.groupby(["pivot", "steps"]):
        rows.append({"method": f"PTI {st} steps from {pv}", "n": len(g), "seconds": g.seconds.median(),
                     "arc_median": g.arc_cos_real_rec.median(), "frac_above_tau": (g.arc_cos_real_rec >= tau).mean(),
                     "lpips": g.lpips_real_rec.median()})
    return pd.DataFrame(rows)


def _mdc3d(o) -> dict:
    n = pd.read_csv(o / "k7_judge_noise.csv")
    return dict(zip(n.metric, n.mdc95))


def depth_gan(o) -> pd.DataFrame:
    d = pd.read_csv(o / "k5_gan_depth.csv", dtype={"pid": str})
    m3 = _mdc3d(o)
    nz = pd.read_csv(o / "k7_judge_noise_rows.csv", dtype={"pid": str})
    rows = []
    for (e, t), g in d.groupby(["encoder", "target"]):
        c3 = f"d3_{TARGET_3D[t]}"
        x = g.get(f"{c3}_d")
        rec = {"encoder": e, "target": t, "n": len(g)}
        if x is not None and c3 in m3:
            sign = SIGN[t]
            rec.update({"shape3d_median_mdc": float((x / m3[c3]).median()),
                        "shape3d_right_direction": float((np.sign(x) == sign).mean()),
                        "shape3d_beyond_mdc_right_dir": float(((x * sign) > m3[c3]).mean())})
        if "depth_locality" in g:
            rec.update({"depth_locality_median": g.depth_locality.median(), "depth_in_median": g.depth_in.median(),
                        "depth_out_median": g.depth_out.median()})
            ncol = f"noise_{t}_depth_in"
            if ncol in nz:
                rec["depth_in_noise_p95"] = float(nz[ncol].quantile(0.95))
                rec["depth_in_above_noise_p95"] = float((g.depth_in > nz[ncol].quantile(0.95)).mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def warp(o) -> pd.DataFrame:
    w = pd.read_csv(o / "k7_warp_judges.csv", dtype={"pid": str})
    n = pd.read_csv(o / "k7_judge_noise.csv")
    mdc = dict(zip(n.metric, n.mdc95))
    nz = pd.read_csv(o / "k7_judge_noise_rows.csv", dtype={"pid": str})
    rows = []
    for t, g in w.groupby("target"):
        g = g[g.get("ok", True) == True]  # noqa: E712
        f = TARGET_FAN[t]
        true, meas = g[f"true_fan_d_{f}"], g[f"meas_fan_{f}_d"]
        sign = SIGN[t]
        rec = {"target": t, "request_mm": g.request_mm.iloc[0], "n": len(g),
               "fan_true_median": true.median(), "fan_measured_median": meas.median(),
               "fan_gain_median": float((meas / true).median()) if t != "mrd1" else np.nan,
               "fan_right_direction": float((np.sign(meas) == sign).mean()),
               "fan_target_beyond_mdc": float(((meas * sign) > mdc[f"fan_{f}"]).mean())}
        leak = {}
        for k in ("nose_width", "mouth_width", "jaw_width", "lower_face_height", "brow_height", "palpebral_height"):
            if k == f:
                continue
            leak[k] = float(((g[f"meas_fan_{k}_d"] - g[f"true_fan_d_{k}"]).abs() / mdc[f"fan_{k}"]).median())
        rec["fan_leak_worst"] = max(leak, key=leak.get)
        rec["fan_leak_worst_median_mdc"] = max(leak.values())
        rec["fan_leak_share_all_le_1mdc"] = float(np.mean([
            all(abs(r[f"meas_fan_{k}_d"] - r[f"true_fan_d_{k}"]) <= mdc[f"fan_{k}"] for k in leak) for _, r in g.iterrows()]))
        c3 = f"d3_{TARGET_3D[t]}"
        x = g[f"meas_{c3}_d"]
        rec.update({"shape3d_median_mdc": float((x / mdc[c3]).median()),
                    "shape3d_right_direction": float((np.sign(x) == sign).mean()),
                    "shape3d_beyond_mdc_right_dir": float(((x * sign) > mdc[c3]).mean())})
        p3 = [c for c in g.columns if c.startswith("meas_d3_") and c != f"meas_{c3}_d"
              and c.replace("meas_", "").replace("_d", "") in mdc and "projection" not in c]
        rec["shape3d_other_median_mdc"] = float(np.median([(g[c].abs() / mdc[c.replace("meas_", "")[:-2]]).median()
                                                           for c in p3]))
        rec.update({"depth_locality_median": g.depth_locality.median(), "depth_in_median": g.depth_in.median(),
                    "depth_out_median": g.depth_out.median()})
        ncol = f"noise_{t}_depth_in"
        if ncol in nz:
            rec["depth_in_noise_p95"] = float(nz[ncol].quantile(0.95))
            rec["depth_in_above_noise_p95"] = float((g.depth_in > nz[ncol].quantile(0.95)).mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    cfg = get_config()
    o = cfg.out_dir / "stage7"
    mdc, env, tau = _load(cfg)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    out = {"inversion": inversion(o, mdc, tau)}
    out["editor"], out["editor_comparisons"] = editor(o, mdc, env, tau)
    out["editor_real"] = editor_real(o, mdc, env, tau)
    out["repro"] = repro(cfg, o, mdc, env, tau)
    out["pti"] = pti(o, tau)
    out["realism"] = pd.read_csv(o / "k5_kid.csv")
    out["depth_gan"] = depth_gan(o)
    out["warp"] = warp(o)
    out["app_timing"] = pd.read_csv(o / "k6_app_timing.csv")
    for k, v in out.items():
        v.to_csv(o / f"analysis_{k}.csv", index=False)
        print(f"\n== {k}")
        print(v.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
