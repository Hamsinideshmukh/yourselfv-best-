"""Distil the GAN population deformation patterns (E7 winner) into src/medsg/assets/patterns_v1.npz.

Pattern for metric t = how every MediaPipe landmark moves (iris frame: unit IPD) per +1 IPD-unit change of t,
estimated over the Stage-3 synthetic within-face perturbations (all layer groups), adjusting for expression and
pose changes (Δsmile, ΔjawOpen, Δyaw, Δpitch). Metrics: the 10 frontal ratios + 5 clinical (computed from the
stored landmarks). Also stores the mean iris-frame face (for affine removal) and the sample count.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from medsg.metrics2d import CLINICAL_2D, METRICS_2D, compute_clinical_2d, compute_metrics_2d  # noqa: E402
from medsg.stages.stage3b_analysis import load_chunks  # noqa: E402
from medsg.stages.stage5c_e7 import _nuis_from_bs, frame, pattern  # noqa: E402

d3 = ROOT / "outputs/stage3"
g, loc = load_chunks(d3, "global"), load_chunks(d3, "local")
ok = loc["ok"] & g["ok"][loc["base_idx"]]
b = loc["base_idx"][ok]
Lb, Ll = g["lm"][b].astype(np.float64) * 512, loc["lm"][ok].astype(np.float64) * 512
D = (frame(Ll) - frame(Lb)).reshape(ok.sum(), -1)
N = _nuis_from_bs(loc["bs"][ok], loc["pose"][ok]) - _nuis_from_bs(g["bs"][b], g["pose"][b])
names = list(METRICS_2D) + list(CLINICAL_2D)


def mets(L):
    return np.array([[*compute_metrics_2d(np.c_[x, np.zeros(478)]).values(), *compute_clinical_2d(np.c_[x, np.zeros(478)]).values()]
                     for x in L])
dM = mets(Ll) - mets(Lb)
P = np.stack([pattern(D, dM[:, k], N) for k in range(len(names))])
np.savez_compressed(ROOT / "src/medsg/assets/patterns_v1.npz", names=np.array(names), P=P.astype(np.float32),
                    mean_face=frame(Lb).mean(0).astype(np.float32), n=ok.sum(),
                    source=np.array("Stage 3 synthetic local perturbations (6,000 bases, all groups); E7 2026-11-01"))
print("patterns", P.shape, "n", ok.sum(), "| |P| per metric:", dict(zip(names, np.linalg.norm(P, axis=1).round(2))))
