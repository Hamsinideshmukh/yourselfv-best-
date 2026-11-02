"""Results figures (static PNG for README/report): E3 controllability, E7 deformation priors, Stage 7 encoders.

Palette: dataviz reference categorical slots 1-4 (validated, light mode). Slots 3-4 are below 3:1 contrast, so
every series also has a distinct marker shape, a legend, and a printed value; the CSVs are the table view.
Writes outputs/figures/*.png. Uses only aggregate numbers (no images, no HDA-derived pictures).
"""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARK = ["o", "s", "D", "^"]
LABEL = {"brow_height": "Brow height", "palpebral_height": "Eye opening", "jaw_width": "Jaw width",
         "mouth_width": "Mouth width", "nose_width": "Alar-junction width", "lower_face_height": "Lower face height"}


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)


def dot_ci(ax, rows, methods, names, value, ci_col=None, xlim=(-0.03, 1.08), fmt="{:.2f}", label_first_only=True):
    n = len(methods)
    for j, (meth, nm) in enumerate(zip(methods, names)):
        for i, (_, r) in enumerate(rows.iterrows()):
            sub = r[meth]
            y = i + (j - (n - 1) / 2) * 0.22
            if ci_col:
                lo, hi = sub[ci_col]
                ax.plot([lo, hi], [y, y], color=SERIES[j], lw=2, solid_capstyle="round")
            ax.plot(sub[value], y, MARK[j], ms=8, color=SERIES[j], mec=SURF, mew=2, label=nm if i == 0 else None)
            if j == 0 or not label_first_only:
                right = max(sub[value], sub[ci_col][1]) if ci_col else sub[value]
                ax.text(right + 0.015 * (xlim[1] - xlim[0]), y, fmt.format(sub[value]), va="center",
                        fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABEL[t] for t in rows.index], color=INK)
    ax.set_xlim(*xlim)
    ax.invert_yaxis()


def fig_e3():
    t = pd.read_csv(ROOT / "outputs/stage5/analysis_test_editor.csv")
    t = t[t.request_kind == "clinical_2mdc"]
    order = ["brow_height", "palpebral_height", "jaw_width", "mouth_width", "nose_width"]
    rows = {}
    for tg in order:
        rows[tg] = {}
        for m in ("closed", "open", "b4"):
            r = t[(t.target == tg) & (t.method == m)].iloc[0]
            lo, hi = [float(x) for x in r.success_ci.strip("[]").split(",")]
            rows[tg][m] = {"v": r.success, "ci": (lo, hi)}
    df = pd.DataFrame(rows).T
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURF)
    _style(ax)
    dot_ci(ax, df, ["closed", "open", "b4"], ["Closed loop (re-measure and correct)", "Open loop (one step)",
                                              "Global direction (baseline B4)"], "v", "ci")
    ax.set_xlabel("Edits that hit the request within noise, with no side effects, identity kept  (share, 95% CI)",
                  color=INK2, fontsize=9)
    ax.set_title("E3 controllability on 440 held-out faces (pre-registered; requests = 2 noise floors)",
                 loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.02), fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig_e3_controllability.png", dpi=200, facecolor=SURF)


def fig_e7():
    s = pd.read_csv(ROOT / "outputs/stage5/e7_summary.csv")
    order = ["nose_width", "mouth_width", "brow_height", "palpebral_height", "jaw_width", "lower_face_height"]
    pri = ["GAN_POP", "HDA_EMP", "GAUSS_MP", "GAN_PERS"]
    names = ["GAN population pattern (used by the warp mode)", "Pattern from real surgery photos (dev)",
             "Hand-made smooth warp", "GAN pattern from the face's own code"]
    rows = {tg: {p: {"v": s[(s.target == tg) & (s.prior == p)].mean_cos.iloc[0]} for p in pri} for tg in order}
    df = pd.DataFrame(rows).T
    n = s[s.prior == "GAN_POP"].set_index("target").n_pairs
    fig, ax = plt.subplots(figsize=(8, 5.2), facecolor=SURF)
    _style(ax)
    dot_ci(ax, df, pri, names, "v", None, xlim=(0, 0.62))
    ax.set_yticklabels([f"{LABEL[t]} (n={n[t]})" for t in order], color=INK)
    ax.set_xlabel("Shape agreement with each person's real post-op change (mean cosine; affine motion removed)",
                  color=INK2, fontsize=9)
    ax.set_title("E7: which deformation prior matches real surgery? (held-out test pairs)", loc="left",
                 color=INK, fontsize=11)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=2, fontsize=8.5, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig_e7_priors.png", dpi=200, facecolor=SURF)


def fig_s7_encoders():
    t = pd.read_csv(ROOT / "outputs/stage7/analysis_editor_real.csv").set_index("target")
    order = ["brow_height", "palpebral_height", "jaw_width", "mouth_width", "nose_width"]
    enc = ["hyperstyle", "e4e", "e4e_hda"]
    rows = {tg: {e: {"v": t.loc[tg, e], "ci": (t.loc[tg, f"{e}_lo"], t.loc[tg, f"{e}_hi"])} for e in enc} for tg in order}
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURF)
    _style(ax)
    dot_ci(ax, pd.DataFrame(rows).T, enc, ["HyperStyle (adopted for brow, mouth)", "e4e (official)",
                                          "e4e fine-tuned on HDA (rejected: identity worse)"], "v", "ci")
    ax.set_xlabel("Hit the request, no side effects, AND still look like the real person (share, 95% CI; defined post hoc)",
                  color=INK2, fontsize=9)
    ax.set_title("Stage 7: which inversion feeds the editor? (440 held-out faces)",
                 loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.02), fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig_s7_encoders.png", dpi=200, facecolor=SURF)


def fig_s7_speed():
    p = pd.read_csv(ROOT / "outputs/stage7/analysis_pti.csv")
    p = p[p.method.isin(["e4e (no tuning)", "HyperStyle", "PTI 100 steps from e4e", "PTI 300 steps from e4e"])]
    tau = float(pd.read_csv(ROOT / "outputs/stage5/analysis_identity.csv").tau_arc_dev_p05.iloc[0])
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=SURF)
    _style(ax)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.axhline(tau, color=INK2, lw=1, ls="--")
    ax.text(0.11, tau + 0.008, "real pre- vs post-op identity, 5th percentile (threshold)", color=INK2, fontsize=8)
    for _, r in p.iterrows():
        ax.plot(r.seconds, r.arc_median, "o", ms=9, color=SERIES[0], mec=SURF, mew=2)
        ax.text(r.seconds * 1.15, r.arc_median - 0.012, f"{r.method.replace(' from e4e', '')}  {r.arc_median:.2f}",
                fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.08, 400)
    ax.set_ylim(0.5, 0.92)
    ax.set_xlabel("Seconds per photo on a T4 GPU (log scale)", color=INK2, fontsize=9)
    ax.set_ylabel("ArcFace identity vs the real photo (median)", color=INK2, fontsize=9)
    ax.set_title("Identity vs speed (40 held-out faces)", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_s7_identity_speed.png", dpi=200, facecolor=SURF)


def fig_s8_detectors():
    r = pd.read_csv(ROOT / "outputs/stage8/analysis_detector_ratio_exploratory.csv")
    import numpy as np
    d = pd.read_csv(ROOT / "outputs/stage8/s1_arms.csv")
    rows = [("brow_height", "REF", "Brow height (offered in GAN mode)"), ("nose_width", "A", "Nose (alar-junction width)"),
            ("palpebral_height", "A", "Eye opening"), ("palpebral_height", "C", "Eye opening, fine-layer direction")]
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=SURF)
    _style(ax)
    for i, (t, arm, lab) in enumerate(rows):
        x = r[(r.target == t) & (r.arm == arm)].iloc[0]
        lo, hi = [float(v) for v in x.expected_range.strip("[]").split(",")]
        ax.add_patch(plt.Rectangle((lo, i - 0.22), min(hi, 1.5) - lo, 0.44, color=GRID, lw=0))
        if hi > 1.5:
            ax.text(1.48, i, f"to {hi:.2f} →", ha="right", va="center", fontsize=8, color=INK2)
        ax.plot(x.observed_fan_over_mp, i, MARK[0], ms=9, color=SERIES[0], mec=SURF, mew=2)
        g = d[(d.target == t) & (d.arm == arm)]
        right = float((np.sign(g.filter(like=f"d3_{t}_3d_d").iloc[:, 0]) == (-1 if t == "nose_width" else 1)).mean())
        ax.text(1.03, i, f"3D: {right:.0%} right direction", va="center", fontsize=9, color=INK,
                transform=ax.get_yaxis_transform())
        ax.text(x.observed_fan_over_mp, i - 0.3, f"{x.observed_fan_over_mp:.2f}", ha="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab for *_, lab in rows], color=INK)
    ax.set_xlim(0, 1.5)
    ax.invert_yaxis()
    ax.set_xlabel("FAN change ÷ MediaPipe change (same units).  Grey = what a REAL shape change gives (calibrated)",
                  color=INK2, fontsize=8.5)
    ax.set_title("Stage 8: do GAN edits look like real anatomy to independent judges? (440 faces)", loc="left",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.subplots_adjust(right=0.80)
    fig.savefig(OUT / "fig_s8_detectors.png", dpi=200, facecolor=SURF)


if __name__ == "__main__":
    fig_e3()
    fig_e7()
    fig_s7_encoders()
    fig_s7_speed()
    fig_s8_detectors()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
