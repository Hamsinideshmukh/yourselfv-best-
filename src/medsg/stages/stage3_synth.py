"""Stage 3: synthetic latent <-> anatomy dataset (GPU synthesis, CPU measurement).

Two sample sets from the frozen e4e/StyleGAN2 generator (truncation psi):
  global : N_GLOBAL random faces, w ~ mapping(z). Learns *across-face* linear maps
           (ridge: w -> each measurement), the data behind "latent dimension <-> feature".
  local  : for N_BASES of those faces, perturb ONE layer group (coarse 0-3, middle 4-7,
           fine 8-17) by delta = SIGMA * std(w) * eps. Learns *within-face* effects
           (finite-difference Jacobians) and which layers control which anatomy.

Each sample stores: w, perturbation (group, delta), 2D+3D metrics, pose, blendshapes,
3DMM shape coeffs, 478 MediaPipe landmarks (normalised). Chunked + resumable.
Outputs: outputs/stage3/chunks/{global,local}_NNNN.npz, outputs/stage3/meta.json
"""
from __future__ import annotations

import json
import time

import cv2
import numpy as np
import torch

from ..config import get_config
from ..data import seed_everything
from ..face3d import METRICS_3D
from ..metrics2d import METRICS_2D
from .stage1_measure import BLENDSHAPES, measure_full
from .stage2_invert import load_models
from ..gan import to_uint8
from ..landmarks import FaceNotFound

N_GLOBAL, N_BASES, CHUNK, BATCH = 30_000, 6_000, 1_000, 8
PSI, SIGMA, MEAS_RES = 0.7, 0.5, 512
GROUPS = {0: (0, 4), 1: (4, 8), 2: (8, 18)}       # coarse, middle, fine W+ layers
METRICS = list(METRICS_2D) + list(METRICS_3D)
POSE = ["yaw", "pitch", "roll", "yaw_3d", "pitch_3d", "roll_3d"]


def _measure_batch(imgs: np.ndarray, lmk, f3d) -> dict[str, np.ndarray]:
    n = len(imgs)
    res = {"ok": np.zeros(n, bool), "metrics": np.full((n, len(METRICS)), np.nan, np.float32),
           "pose": np.full((n, len(POSE)), np.nan, np.float32),
           "bs": np.full((n, len(BLENDSHAPES)), np.nan, np.float32),
           "alpha_shp": np.full((n, 40), np.nan, np.float32),
           "lm": np.full((n, 478, 2), np.nan, np.float16)}
    for i, rgb in enumerate(imgs):
        small = cv2.resize(rgb, (MEAS_RES, MEAS_RES), interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
        try:
            row, a_shp, _, obs = measure_full(bgr, lmk, f3d)
        except (FaceNotFound, ValueError):
            continue
        res["ok"][i] = True
        res["metrics"][i] = [row[m] for m in METRICS]
        res["pose"][i] = [row[k] for k in POSE]
        res["bs"][i] = [row[f"bs_{k}"] for k in BLENDSHAPES]
        res["alpha_shp"][i] = a_shp
        res["lm"][i] = (obs.lm_px[:, :2] / MEAS_RES).astype(np.float16)
    return res


def _run_chunk(path, ws_plus, meta_arrays, gan, lmk, f3d):
    """Synthesise + measure one chunk of W+ codes (n, 18, 512) and save it."""
    parts = []
    for b in range(0, len(ws_plus), BATCH):
        with torch.no_grad():
            imgs = to_uint8(gan.synth(ws_plus[b:b + BATCH]))
        parts.append(_measure_batch(imgs, lmk, f3d))
    res = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez_compressed(path, **res, **meta_arrays)
    return res["ok"].mean()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 3 needs a GPU")
    device = "cuda"
    cfg = get_config()
    seed_everything(cfg.seed)
    out = cfg.out_dir / "stage3"
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    gan, lmk, f3d, _, _ = load_models(cfg, device)
    L = gan.n_layers

    gen = torch.Generator().manual_seed(cfg.seed)
    w_all = gan.sample_w(N_GLOBAL, PSI, gen).cpu()                       # (N, 512), deterministic
    w_std = gan.sample_w(10_000, 1.0, torch.Generator().manual_seed(cfg.seed + 1)).std(0).cpu()
    json.dump({"N_GLOBAL": N_GLOBAL, "N_BASES": N_BASES, "PSI": PSI, "SIGMA": SIGMA, "GROUPS": GROUPS,
               "METRICS": METRICS, "POSE": POSE, "BLENDSHAPES": BLENDSHAPES, "MEAS_RES": MEAS_RES,
               "seed": cfg.seed}, open(out / "meta.json", "w"), indent=1)
    np.save(out / "w_std.npy", w_std.numpy())

    t0 = time.time()
    for c in range(0, N_GLOBAL, CHUNK):
        path = out / "chunks" / f"global_{c // CHUNK:04d}.npz"
        if path.exists():
            continue
        w = w_all[c:c + CHUNK]
        rate = _run_chunk(path, w.unsqueeze(1).repeat(1, L, 1).to(device),
                          {"idx": np.arange(c, c + len(w)), "w": w.numpy()}, gan, lmk, f3d)
        print(f"  global {c + len(w)}/{N_GLOBAL}  ok={rate:.2%}  {time.time() - t0:.0f}s", flush=True)

    rng = np.random.default_rng(cfg.seed + 2)
    for c in range(0, N_BASES, CHUNK):
        path = out / "chunks" / f"local_{c // CHUNK:04d}.npz"
        base = np.arange(c, min(c + CHUNK, N_BASES))
        group = rng.integers(0, 3, len(base))
        delta = (SIGMA * w_std.numpy() * rng.standard_normal((len(base), w_std.shape[0]))).astype(np.float32)
        if path.exists():
            continue
        wp = w_all[base].unsqueeze(1).repeat(1, L, 1).clone()
        for i, g in enumerate(group):
            a, b = GROUPS[int(g)]
            wp[i, a:b] += torch.from_numpy(delta[i])
        rate = _run_chunk(path, wp.to(device), {"base_idx": base, "group": group, "delta": delta},
                          gan, lmk, f3d)
        print(f"  local {base[-1] + 1}/{N_BASES}  ok={rate:.2%}  {time.time() - t0:.0f}s", flush=True)
    lmk.close()


if __name__ == "__main__":
    main()
