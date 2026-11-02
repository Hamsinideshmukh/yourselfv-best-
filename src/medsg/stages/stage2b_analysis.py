"""Stage 2b: analysis of the Stage-2 inversion benchmark (CPU, seconds, tables only).

Questions (all per Stage-1 metric, against Stage-1 noise thresholds):
  A) Inversion quality: FaceNet identity cosine and LPIPS, e4e vs PTI (same 48 images).
  B) Anatomical drift: does reconstruction alone move a metric?  drift = rec - real (aligned photo).
     Compared to the same-photo MDC95 and to the realistic (re-photography) MDC from Stage 1b.
  C) Change preservation: on pose-matched pairs, is the real surgical change
     d_real = real_post - real_pre still present in d_e4e = e4e_post - e4e_pre?
     slope of d_e4e on d_real: 1 = change carried fully, 0 = change lost.
     Also the across-face slope of rec on real (does inversion pull every face towards the mean face?).
  D) Alignment consistency: metrics on the FFHQ-aligned photo vs Stage-1 metrics on the raw photo.

Writes outputs/stage2/analysis_*.csv.  Run: python -m medsg.stages.stage2b_analysis
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import get_config
from ..stats import bootstrap_ci


def _slope_r(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """OLS slope of y on x and Pearson r (finite rows only)."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    c = np.cov(x, y)
    return float(c[0, 1] / c[0, 0]), float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])


def _boot_spearman(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    s = [_spearman(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(n_boot))]
    return float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))


def _boot_slope(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    s = [_slope_r(x[i], y[i])[0] for i in (rng.integers(0, len(x), len(x)) for _ in range(n_boot))]
    return float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))


def main() -> None:
    cfg = get_config()
    s1, s2 = cfg.out_dir / "stage1", cfg.out_dir / "stage2"
    img = pd.read_csv(s2 / "images.csv")
    pti = pd.read_csv(s2 / "pti.csv")
    if not img.ok.all() or not pti.ok.all():
        raise ValueError("stage-2 tables contain failed rows; analysis assumes none")
    summ = pd.read_csv(s1 / "summary.csv").set_index("metric")
    real_mdc = pd.read_csv(s1 / "followup_realistic_noise.csv").set_index("metric").mdc_real
    pairs = pd.read_csv(s1 / "pairs.csv", index_col=0)
    s1img = pd.read_csv(s1 / "images.csv").set_index(["pid", "role"])
    metrics = list(summ.index)
    nb = 2000

    # A) inversion quality, e4e vs PTI on the same images
    j = pti.merge(img, on=["pid", "role"], suffixes=("_pti", "_e4e"))
    rows = []
    for name, x in [("id_cos_e4e_all", img.id_cos), ("lpips_e4e_all", img.lpips),
                    ("id_cos_e4e_ptisubset", j.id_cos_e4e), ("id_cos_pti", j.id_cos_pti),
                    ("id_cos_pti_minus_e4e", j.id_cos_pti - j.id_cos_e4e),
                    ("lpips_e4e_ptisubset", j.lpips_e4e), ("lpips_pti", j.lpips_pti)]:
        e, lo, hi = bootstrap_ci(x.to_numpy(), np.median, n_boot=nb, seed=cfg.seed)
        rows.append({"quantity": name, "n": len(x), "median": e, "ci_lo": lo, "ci_hi": hi,
                     "p05": x.quantile(0.05), "p95": x.quantile(0.95)})
    pd.DataFrame(rows).to_csv(s2 / "analysis_quality.csv", index=False)

    # B) drift per metric (all 1324 images for e4e; 48 for PTI, paired with their e4e drift)
    jp = j.set_index(["pid", "role"])
    rows = []
    for m in metrics:
        d = (img[f"e4e_{m}"] - img[f"real_{m}"]).to_numpy()
        d_e4e48 = (jp[f"e4e_{m}"] - jp[f"real_{m}"]).to_numpy()
        d_pti48 = (jp[f"pti_{m}"] - jp[f"real_{m}"]).to_numpy()
        bias, blo, bhi = bootstrap_ci(d, n_boot=nb, seed=cfg.seed)
        sd_surg = pairs.loc[pairs.include_pose, f"d_{m}"].std()
        shrink, shrink_r = _slope_r(img[f"real_{m}"].to_numpy(), img[f"e4e_{m}"].to_numpy())
        shrink_pti = _slope_r(jp[f"real_{m}"].to_numpy(), jp[f"pti_{m}"].to_numpy())[0]
        rows.append({"metric": m, "family": summ.loc[m, "family"], "exploratory": summ.loc[m, "exploratory"],
                     "n": len(d), "bias": bias, "bias_lo": blo, "bias_hi": bhi,
                     "median_abs_drift": np.median(np.abs(d)),
                     "mdc_same_photo": summ.loc[m, "mdc95"], "mdc_real": real_mdc[m],
                     "pct_gt_mdc_same": 100 * (np.abs(d) > summ.loc[m, "mdc95"]).mean(),
                     "pct_gt_mdc_real": 100 * (np.abs(d) > real_mdc[m]).mean(),
                     "sd_drift_over_sd_surgical_delta": d.std() / sd_surg,
                     # across-face slope of rec on real: 1 = individual differences kept, <1 = pulled to the mean face
                     "across_face_slope_e4e": shrink, "across_face_r_e4e": shrink_r,
                     "across_face_slope_pti48": shrink_pti,
                     "pti48_median_abs_drift": np.median(np.abs(d_pti48)),
                     "e4e48_median_abs_drift": np.median(np.abs(d_e4e48)),
                     "pti48_pct_gt_mdc_real": 100 * (np.abs(d_pti48) > real_mdc[m]).mean(),
                     "e4e48_pct_gt_mdc_real": 100 * (np.abs(d_e4e48) > real_mdc[m]).mean()})
    drift = pd.DataFrame(rows)
    drift.to_csv(s2 / "analysis_drift.csv", index=False)

    # C) change preservation on pose-matched pairs
    w = img.pivot(index="pid", columns="role")
    inc = pairs.index[pairs.include_pose]
    w = w.loc[w.index.intersection(inc)]
    pw = pti.set_index(["pid", "role"])
    pti_ids = sorted(set(pti.pid) & set(inc))
    rows = []
    for m in metrics:
        dr = (w[(f"real_{m}", "post")] - w[(f"real_{m}", "pre")]).to_numpy()
        de = (w[(f"e4e_{m}", "post")] - w[(f"e4e_{m}", "pre")]).to_numpy()
        slope, r = _slope_r(dr, de)
        slo, shi = _boot_slope(dr, de, nb, cfg.seed)
        rho = _spearman(dr, de)
        rlo, rhi = _boot_spearman(dr, de, nb, cfg.seed)
        mr, mrlo, mrhi = bootstrap_ci(dr, n_boot=nb, seed=cfg.seed)
        me, melo, mehi = bootstrap_ci(de, n_boot=nb, seed=cfg.seed)
        dr24 = np.array([w.loc[p, (f"real_{m}", "post")] - w.loc[p, (f"real_{m}", "pre")] for p in pti_ids])
        dp24 = np.array([pw.loc[(p, "post"), f"pti_{m}"] - pw.loc[(p, "pre"), f"pti_{m}"] for p in pti_ids])
        de24 = np.array([w.loc[p, (f"e4e_{m}", "post")] - w.loc[p, (f"e4e_{m}", "pre")] for p in pti_ids])
        rows.append({"metric": m, "family": summ.loc[m, "family"], "exploratory": summ.loc[m, "exploratory"],
                     "n": len(dr), "mean_d_real": mr, "mean_d_real_lo": mrlo, "mean_d_real_hi": mrhi,
                     "mean_d_e4e": me, "mean_d_e4e_lo": melo, "mean_d_e4e_hi": mehi,
                     "slope_e4e_on_real": slope, "slope_lo": slo, "slope_hi": shi, "r": r,
                     "spearman": rho, "spearman_lo": rlo, "spearman_hi": rhi,
                     "n_pti_pairs": len(pti_ids),
                     "slope_pti_on_real": _slope_r(dr24, dp24)[0], "r_pti": _slope_r(dr24, dp24)[1],
                     "slope_e4e_on_real_same24": _slope_r(dr24, de24)[0]})
    pres = pd.DataFrame(rows)
    pres.to_csv(s2 / "analysis_preservation.csv", index=False)

    # D) aligned-photo metrics vs Stage-1 raw-photo metrics (same image, two crops)
    rows = []
    for m in metrics:
        a = img.set_index(["pid", "role"])[f"real_{m}"]
        b = s1img.loc[a.index, m]
        rows.append({"metric": m, "r": np.corrcoef(a, b)[0, 1], "mean_diff": (a - b).mean(),
                     "median_abs_diff": (a - b).abs().median(), "mdc_same_photo": summ.loc[m, "mdc95"]})
    pd.DataFrame(rows).to_csv(s2 / "analysis_alignment.csv", index=False)

    # Pre-declared Stage-2 criteria. (a) needs the real pre-vs-post
    # identity yardstick, which is not computed yet -> reported as not evaluable, not as passed.
    dr_ = drift.set_index("metric")
    crit = pd.DataFrame([
        {"criterion": "a_pti_identity_gt_real_pre_post", "result": "NOT EVALUABLE (yardstick missing)"},
        {"criterion": "b_pti_median_abs_drift_le_mdc95_all_metrics",
         "result": str(bool((dr_.pti48_median_abs_drift <= dr_.mdc_same_photo).all())),
         "failing": ",".join(dr_.index[dr_.pti48_median_abs_drift > dr_.mdc_same_photo])},
        {"criterion": "b_e4e_median_abs_drift_le_mdc95_all_metrics",
         "result": str(bool((dr_.median_abs_drift <= dr_.mdc_same_photo).all())),
         "failing": ",".join(dr_.index[dr_.median_abs_drift > dr_.mdc_same_photo])},
        {"criterion": "c_brow_height_spearman_lo_gt_0",
         "result": str(bool(pres.set_index("metric").loc["brow_height", "spearman_lo"] > 0))}])
    crit.to_csv(s2 / "analysis_criteria.csv", index=False)

    pd.set_option("display.width", 220)
    print(crit.to_string(index=False))
    print(pd.read_csv(s2 / "analysis_quality.csv").round(4).to_string(index=False))
    print(drift.drop(columns=["family", "n"]).round(4).to_string(index=False))
    print(pres.drop(columns=["family"]).round(4).to_string(index=False))
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
