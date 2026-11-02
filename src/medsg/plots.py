"""Static matplotlib figures (light theme tokens from the dataviz reference palette)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # fixed categorical order: 2D, 3D, extra


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def forest(summary: pd.DataFrame, path: Path, title: str) -> None:
    """Mean change in MDC units with 95% CI per metric; +-1 MDC band shaded.

    `summary` needs columns: metric, family (2D/3D), z_mean, z_lo, z_hi.
    """
    import matplotlib.pyplot as plt
    s = summary.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(s) + 1.4), facecolor=SURFACE)
    _style(ax)
    ax.axvspan(-1, 1, color=GRID, alpha=0.6, lw=0)
    ax.axvline(0, color=AXIS, lw=1)
    y = np.arange(len(s))[::-1]
    for fam, col in zip(("2D", "3D"), SERIES):
        m = s["family"] == fam
        ax.errorbar(s.loc[m, "z_mean"], y[m.values],
                    xerr=[s.loc[m, "z_mean"] - s.loc[m, "z_lo"], s.loc[m, "z_hi"] - s.loc[m, "z_mean"]],
                    fmt="o", ms=6, color=col, ecolor=col, elinewidth=2, capsize=0, label=f"{fam} measurement")
    ax.set_yticks(y, s["metric"], color=INK2, fontsize=9)
    ax.set_xlabel("mean pre→post change, in units of MDC95 (shaded: within noise)", color=INK2)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def snr_bars(snr: pd.DataFrame, path: Path) -> None:
    """Signal-to-noise per metric: SD of real pre→post change / SD expected from noise alone."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(snr) + 1.4), facecolor=SURFACE)
    _style(ax)
    y = np.arange(len(snr))[::-1]
    cols = [SERIES[0] if f == "2D" else SERIES[1] for f in snr["family"]]
    ax.barh(y, snr["snr"], height=0.6, color=cols)
    ax.axvline(1, color=INK2, lw=1, ls="--")
    ax.text(1.02, y.max() + 0.6, "noise only = 1", color=INK2, fontsize=8, va="bottom")
    ax.set_yticks(y, snr["metric"], color=INK2, fontsize=9)
    ax.set_xlabel("SD(real change) / SD(change expected from measurement noise)", color=INK2)
    ax.set_title("Is there real pre→post signal above measurement noise?", color=INK, loc="left", fontsize=11)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SERIES[0], label="2D"), Patch(color=SERIES[1], label="3D")],
              frameon=False, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
