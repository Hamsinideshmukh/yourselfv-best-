"""Stage 2: inversion benchmark on HDA (GPU).

For every HDA image: FFHQ-align (MediaPipe) -> e4e -> StyleGAN2 reconstruction, then
measure identity (FaceNet cosine), LPIPS, and every Stage-1 anatomical metric on both the
aligned real photo and its reconstruction. PTI on a fixed dev subset (pre and post).

Key questions answered (analysed locally against Stage-1 noise thresholds):
  1. Does reconstruction alone move anatomy beyond re-photography noise? (drift vs MDC)
  2. Does e4e's latent space carry the real surgical change? (Δ on reconstructions vs Δ real)
  3. How much does PTI fix both?

Outputs (outputs/stage2/): images.csv, pti.csv, private/latents.npz (HDA codes; never commit),
private/examples_*.png.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import pandas as pd
import torch

from ..align import align_ffhq
from ..config import get_config
from ..data import list_pairs, make_split, seed_everything
from ..face3d import Face3D
from ..gan import E4EGAN, from_uint8, install_e4e_package, pti_tune, to_uint8
from ..identity import FaceNetID
from ..landmarks import FaceNotFound, Landmarker, ensure_task_file
from .stage1_measure import measure, read_bgr

N_PTI_PAIRS = 24
PTI_STEPS = 300


def load_models(cfg, device):
    install_e4e_package(cfg.repos_dir / "encoder4editing", cfg.out_dir.parent / "src_ext")
    import lpips
    gan = E4EGAN(cfg.models_dir / "e4e_ffhq_encode.pt", device)
    lmk = Landmarker(ensure_task_file(cfg.models_dir))
    f3d = Face3D(cfg.repos_dir / "3DDFA_V2")
    return gan, lmk, f3d, FaceNetID(device), lpips.LPIPS(net="alex", verbose=False).to(device).eval()


def _metrics_of(rgb: np.ndarray, lmk, f3d, prefix: str) -> dict:
    m, _, _ = measure(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), lmk, f3d)
    return {f"{prefix}{k}": v for k, v in m.items()}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 2 needs a GPU (Kaggle: Settings -> Accelerator -> GPU T4)")
    device = "cuda"
    cfg = get_config()
    seed_everything(cfg.seed)
    out = cfg.out_dir / "stage2"
    (out / "private").mkdir(parents=True, exist_ok=True)
    pairs = list_pairs(cfg)
    split = make_split(pairs, cfg, cfg.out_dir / "splits.json")
    gan, lmk, f3d, idnet, lp = load_models(cfg, device)

    if (out / "images.csv").exists() and (out / "private" / "latents.npz").exists():
        df = pd.read_csv(out / "images.csv")       # e4e pass already finished: resume at PTI
        codes = dict(np.load(out / "private" / "latents.npz"))
        print(f"stage2: reusing e4e results ({len(df)} images, {len(codes)} codes)", flush=True)
    else:
        df, codes = _e4e_pass(pairs, split, gan, lmk, f3d, idnet, lp, out, device)
    _pti_pass(pairs, split, df, codes, gan, lmk, f3d, idnet, lp, out, device)
    lmk.close()


def _e4e_pass(pairs, split, gan, lmk, f3d, idnet, lp, out, device):
    """Invert every HDA image with e4e; write images.csv and private/latents.npz."""
    rows, codes = [], {}
    t0 = time.time()
    for i, p in enumerate(pairs):
        for role, path in (("pre", p.pre), ("post", p.post)):
            r = {"pid": p.pid, "role": role, "split": "dev" if p.pid in split["dev"] else "test", "ok": False, "error": ""}
            try:
                rgb = cv2.cvtColor(read_bgr(path), cv2.COLOR_BGR2RGB)
                aligned, _ = align_ffhq(rgb, lmk(rgb).lm_px, 1024)
                x = from_uint8(aligned, 1024, device)
                code = gan.encode(torch.nn.functional.interpolate(x, size=(256, 256), mode="area"))
                with torch.no_grad():
                    rec = gan.synth(code)
                    r["id_cos"] = float(idnet.cosine(rec, x))
                    r["lpips"] = float(lp(torch.nn.functional.interpolate(rec, size=256, mode="area"),
                                          torch.nn.functional.interpolate(x, size=256, mode="area")))
                rec_rgb = to_uint8(rec)[0]
                r.update(_metrics_of(aligned, lmk, f3d, "real_"))
                r.update(_metrics_of(rec_rgb, lmk, f3d, "e4e_"))
                codes[f"{p.pid}_{role}"] = code[0].cpu().numpy().astype(np.float16)
                r["ok"] = True
                if i < 4:
                    cv2.imwrite(str(out / "private" / f"examples_{p.pid}_{role}.png"),
                                cv2.cvtColor(np.hstack([aligned, rec_rgb]), cv2.COLOR_RGB2BGR)[::2, ::2])
            except (FaceNotFound, ValueError) as e:
                r["error"] = f"{type(e).__name__}: {e}"
            rows.append(r)
        if i % 50 == 0:
            print(f"  stage2 {i}/{len(pairs)} pairs, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "images.csv", index=False)
    np.savez_compressed(out / "private" / "latents.npz", **codes)
    n_fail = int((~df.ok).sum())
    print(f"stage2 e4e: {len(df)} images, {n_fail} failures; id_cos median {df.id_cos.median():.3f}, "
          f"LPIPS median {df.lpips.median():.3f}")
    if n_fail > 0.05 * len(df):
        raise RuntimeError("too many stage-2 failures; inspect images.csv")
    return df, codes


def _pti_pass(pairs, split, df, codes, gan, lmk, f3d, idnet, lp, out, device):
    """PTI on a fixed dev subset (both pre and post of each pair); write pti.csv."""
    dev_ids = [pid for pid in split["dev"] if df[(df.pid == pid)].ok.all()][:N_PTI_PAIRS]
    pti_rows = []
    for j, pid in enumerate(dev_ids):
        p = next(q for q in pairs if q.pid == pid)
        for role, path in (("pre", p.pre), ("post", p.post)):
            rgb = cv2.cvtColor(read_bgr(path), cv2.COLOR_BGR2RGB)
            aligned, _ = align_ffhq(rgb, lmk(rgb).lm_px, 1024)
            x = from_uint8(aligned, 1024, device)
            code = torch.from_numpy(codes[f"{pid}_{role}"].astype(np.float32)).unsqueeze(0).to(device)
            t1 = time.time()
            G = pti_tune(gan, code, x, lp, steps=PTI_STEPS)
            with torch.no_grad():
                rec = gan.synth(code, G)
                r = {"pid": pid, "role": role, "pti_seconds": time.time() - t1,
                     "id_cos": float(idnet.cosine(rec, x)),
                     "lpips": float(lp(torch.nn.functional.interpolate(rec, size=256, mode="area"),
                                       torch.nn.functional.interpolate(x, size=256, mode="area")))}
            try:
                r.update(_metrics_of(to_uint8(rec)[0], lmk, f3d, "pti_"))
                r["ok"] = True
            except (FaceNotFound, ValueError) as e:
                r.update(ok=False, error=str(e))
            pti_rows.append(r)
            del G
            torch.cuda.empty_cache()
        print(f"  PTI {j + 1}/{len(dev_ids)} pairs, last {pti_rows[-1]['pti_seconds']:.0f}s/img", flush=True)
    pti = pd.DataFrame(pti_rows)
    pti.to_csv(out / "pti.csv", index=False)
    print(f"stage2 PTI: n={len(pti)}, id_cos median {pti.id_cos.median():.3f}, LPIPS median {pti.lpips.median():.3f}")


if __name__ == "__main__":
    main()
