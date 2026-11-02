"""Stage 1b: follow-up analysis of Stage 1 tables (CPU, seconds, no images needed).

A) Normaliser check: % change relative to face width (cancels the shared IPD denominator).
B) Expression adjustment for mouth width (regress delta on smile/jaw-open blendshape change).
C) Realistic noise: inflate same-photo SEM by k = SNR(intercanthal), an anatomical control
   that these procedures rarely change; report % of pairs changing beyond that MDC, up vs down.
   Under pure noise, up% ~ down% ~ 2.5%; a clear asymmetry is evidence of a directional change.

Writes outputs/stage1/followup_*.csv.  Run: python -m medsg.stages.stage1b_analysis
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import get_config
from ..metrics2d import METRICS_2D
from ..stats import bootstrap_ci

SMILE = ["bs_mouthSmileLeft", "bs_mouthSmileRight"]


def main() -> None:
    cfg = get_config()
    d = cfg.out_dir / "stage1"
    pr = pd.read_csv(d / "pairs.csv", index_col=0)
    pr = pr[pr.include_pose]
    img = pd.read_csv(d / "images.csv")
    pre = img[img.role == "pre"].set_index("pid").loc[pr.index]
    post = img[img.role == "post"].set_index("pid").loc[pr.index]
    summ = pd.read_csv(d / "summary.csv").set_index("metric")

    # A) normaliser
    rows = []
    for m in METRICS_2D:
        if m == "face_width":
            continue
        vs_ipd = 100 * (np.log(post[m]) - np.log(pre[m]))
        vs_fw = vs_ipd - 100 * (np.log(post.face_width) - np.log(pre.face_width))
        e, lo, hi = bootstrap_ci(vs_fw.to_numpy(), n_boot=cfg.n_boot, seed=cfg.seed)
        rows.append({"metric": m, "pct_vs_ipd": vs_ipd.mean(), "pct_vs_face_width": e, "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(rows).to_csv(d / "followup_normaliser.csv", index=False)

    # B) expression-adjusted mouth width
    X = np.c_[np.ones(len(pr)), post[SMILE].mean(axis=1) - pre[SMILE].mean(axis=1), post.bs_jawOpen - pre.bs_jawOpen]
    y = pr.d_mouth_width.to_numpy()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rng = np.random.default_rng(cfg.seed)
    boots = [np.linalg.lstsq(X[i], y[i], rcond=None)[0][0]
             for i in (rng.integers(0, len(y), len(y)) for _ in range(2000))]
    pd.DataFrame([{"metric": "mouth_width", "raw_mean_delta": y.mean(), "adjusted_delta": beta[0],
                   "ci_lo": np.quantile(boots, 0.025), "ci_hi": np.quantile(boots, 0.975),
                   "smile_coef": beta[1], "jaw_coef": beta[2], "r2_expression": r2}]
                 ).to_csv(d / "followup_expression.csv", index=False)

    # C) realistic noise via anatomical control
    k = {"2D": summ.loc["intercanthal", "snr"], "3D": summ.loc["intercanthal_3d", "snr"]}
    rows = []
    for m, s in summ.iterrows():
        kk = k[s.family]
        mdc = s.mdc95 * kk
        dm = pr[f"d_{m}"]
        rows.append({"metric": m, "family": s.family, "exploratory": s.exploratory, "k": kk,
                     "snr_vs_rephoto": s.snr / kk, "mdc_real": mdc,
                     "pct_up": 100 * (dm > mdc).mean(), "pct_down": 100 * (dm < -mdc).mean()})
    out = pd.DataFrame(rows)
    out.to_csv(d / "followup_realistic_noise.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
