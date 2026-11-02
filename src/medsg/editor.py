"""GAN-mode editor for the app: frozen Stage-4b directions + the Stage-5 closed loop (same secant rule, same
stopping tolerance), packaged without the evaluation machinery. Stage 5: brow, mouth and jaw edits confirmed by an
independent landmarker (nose, eye opening not). Stage 7 (HyperStyle inversion): brow and mouth pass the pre-set
editability margin, jaw does not -> GAN mode offers brow and mouth (docs/RESULTS.md).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from . import editspec as E
from .measure import MCOLS

ASSET = Path(__file__).resolve().parent / "assets" / "editor_v1.npz"
GROUPS = {0: (0, 4), 1: (4, 8), 2: (8, 18)}        # coarse, middle, fine W+ layers (= stage3_synth.GROUPS)
GAN_TARGETS = ["brow_height", "mouth_width"]
U_CAP = 23.0


class Directions:
    def __init__(self, asset: Path = ASSET, n_layers: int = 18):
        z = np.load(asset)
        self.targets = [str(t) for t in z["targets"]]
        self.mdc = dict(zip([str(m) for m in z["mdc_metrics"]], z["mdc"].astype(float)))
        w_std = z["w_std"].astype(np.float64)
        self.D, self.a_cap = {}, {}
        for t, g, wd in zip(self.targets, z["groups"], z["w_delta"]):
            D = np.zeros((n_layers, 512), np.float32)
            a, b = GROUPS[int(g)]
            D[a:b] = wd
            self.D[t] = D
            self.a_cap[t] = U_CAP / float(np.linalg.norm(wd / w_std))


def secant_edit(render_measure: Callable[[float], np.ndarray | None], r: float, tol: float, a_cap: float):
    """Find amplitude a with measured change(a) ~= r. render_measure(a) returns the measured target change (or None).
    Returns (records [(a, achieved)], status). Same rule as stage5_overnight._closed_loop."""
    a_prev, ach_prev, a, retried, recs = 0.0, 0.0, r, False, []
    for it in range(E.LOOP_MAX_ITERS):
        t = render_measure(a)
        if t is None:
            return recs, "measure_fail"
        recs.append((a, t))
        if abs(t - r) <= tol:
            return recs, "converged"
        slope = (t - ach_prev) / (a - a_prev)
        if not np.isfinite(slope) or slope <= 0:
            if it == 0 and not retried:
                retried, a = True, 2 * a
                continue
            return recs, "non_monotone"
        a_prev, ach_prev = a, t
        a = a + (r - t) / slope
        if abs(a) > a_cap:
            a = float(np.sign(a) * a_cap)
            if abs(a_prev) >= a_cap:
                return recs, "range_limit"
    return recs, "not_converged"


def target_index(t: str) -> int:
    return MCOLS.index(t)
