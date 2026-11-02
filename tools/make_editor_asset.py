"""Freeze the Stage-4b editor (directions fitted on synthetic + DEV real codes only) into src/medsg/assets/editor_v1.npz.

Shipped inside the Kaggle cell so the overnight run uses exactly these, pre-declared, numbers:
directions (w_delta per +1 IPD-unit of the target, applied to every layer of its group), B4 global-ridge directions,
w_std, synthetic MDC95 (Stage 4a noise floor) and the dev-mean E3 requests (Stage 4b envelope.json).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
d = np.load(ROOT / "outputs/stage4b/directions.npz")
assert str(d["mode"]) == "full", "directions must come from Stage 4b full mode"
nf = pd.read_csv(ROOT / "outputs/stage4a/noise_synth.csv").set_index("metric")
env = json.load(open(ROOT / "outputs/stage4b/envelope.json"))
m10 = list(d["b4_metrics"])
np.savez_compressed(
    ROOT / "src/medsg/assets/editor_v1.npz",
    targets=d["targets"], groups=d["groups"], w_delta=d["w_delta"].astype(np.float32), kappa=d["kappa"],
    b4_metrics=d["b4_metrics"], b4_w_delta=d["b4_w_delta"].astype(np.float32), w_std=d["w_std"].astype(np.float32),
    mdc_metrics=np.array(m10), mdc=nf.loc[m10, "mdc95"].to_numpy(np.float64),
    e3_targets=np.array(list(env["e3_requests"])), e3_requests=np.array(list(env["e3_requests"].values())),
    source=np.array("stage4b full 2026-10-31; noise_synth Stage 4a; envelope dev n=%d" % env["n_pairs"]))
print("wrote editor_v1.npz", (ROOT / "src/medsg/assets/editor_v1.npz").stat().st_size, "bytes")
