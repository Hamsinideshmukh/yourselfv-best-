"""Domain fine-tuning of the e4e encoder on HDA dev photos (generator frozen).

Why: e4e was trained on FFHQ; HDA photos are clinical (older faces, flat light, plain backgrounds). On HDA test
faces its rebuilds score ArcFace 0.55 vs the photo, below the real pre-vs-post threshold (Stage 5).
What stays fixed: the StyleGAN2 generator, so the latent space, the learned anatomy directions and every
noise floor stay valid; only the photo -> code map adapts.
Objective = e4e's own published losses (Tov et al. 2021; values read from the official checkpoint's opts):
  1.0 * pixel L2  +  0.8 * LPIPS(alex)  +  0.1 * identity  +  2e-4 * sum_i ||w_i - w_0||_2  (editability term).
Differences, stated: e4e's latent discriminator is not in the released checkpoint, so it is not used; identity
uses FaceNet (VGGFace2) because the judge is ArcFace (insightface) and must stay independent; BatchNorm statistics
are frozen (batches of 2 are too small to re-estimate them). Early stopping on held-out dev subjects, never test.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

L2_W, LPIPS_W, ID_W, DELTA_W = 1.0, 0.8, 0.1, 2e-4      # e4e checkpoint opts: l2, lpips, id, delta_norm lambdas


def _facenet_embed(net, img256: torch.Tensor) -> torch.Tensor:
    """FaceNet embedding with gradients (same crop/normalisation as identity.FaceNetID)."""
    crop = F.interpolate(img256[:, :, 51:204, 51:204], size=(160, 160), mode="area")
    return F.normalize(net(crop * (127.5 / 128.0)), dim=1)


def e4e_loss(gan, enc, x256: torch.Tensor, lpips_fn, facenet) -> tuple[torch.Tensor, dict]:
    """e4e objective for a batch of aligned 256 px photos in [-1, 1]."""
    codes = enc(x256) + gan.latent_avg.unsqueeze(0)
    rec = F.interpolate(gan.synth(codes), size=(256, 256), mode="area")
    l2 = F.mse_loss(rec, x256)
    lp = lpips_fn(rec, x256).mean()
    idl = (1 - (_facenet_embed(facenet, rec) * _facenet_embed(facenet, x256).detach()).sum(1)).mean()
    delta = (codes[:, 1:] - codes[:, :1]).norm(dim=2).sum(1).mean()
    loss = L2_W * l2 + LPIPS_W * lp + ID_W * idl + DELTA_W * delta
    return loss, {"l2": l2.item(), "lpips": lp.item(), "id": idl.item(), "delta": delta.item()}


def finetune_e4e(gan, train: np.ndarray, val: np.ndarray, lpips_fn, facenet, steps: int = 500, lr: float = 2e-5,
                 micro: int = 2, accum: int = 4, eval_every: int = 50, patience: int = 4, seed: int = 0,
                 deadline: float = float("inf")) -> tuple[dict, list[dict]]:
    """Fine-tune a copy of gan.encoder. train/val: (N, 256, 256, 3) uint8 aligned photos.

    Effective batch = micro * accum (8, as in e4e). Returns (best encoder state_dict, log rows)."""
    assert train.ndim == 4 and train.shape[1:] == (256, 256, 3), train.shape
    dev = gan.device
    enc = copy.deepcopy(gan.encoder).to(dev)
    enc.eval()                                           # frozen BatchNorm statistics; weights still train
    for p in enc.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    to_t = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(dev)  # noqa: E731

    def val_loss() -> float:
        tot = []
        with torch.no_grad():
            for b in range(0, len(val), 4):
                tot.append(e4e_loss(gan, enc, to_t(val[b:b + 4]), lpips_fn, facenet)[0].item())
        return float(np.mean(tot))

    log, best, best_state, bad = [], val_loss(), copy.deepcopy(enc.state_dict()), 0
    log.append({"step": 0, "val_loss": best, "seconds": 0.0})
    t0 = time.time()
    with torch.enable_grad():
        for step in range(1, steps + 1):
            opt.zero_grad(set_to_none=True)
            parts = {}
            for _ in range(accum):
                idx = rng.choice(len(train), micro, replace=False)
                x = train[idx]
                flip = rng.random(micro) < 0.5                   # horizontal flip augmentation
                x = np.where(flip[:, None, None, None], x[:, :, ::-1], x)
                loss, parts = e4e_loss(gan, enc, to_t(np.ascontiguousarray(x)), lpips_fn, facenet)
                (loss / accum).backward()
            opt.step()
            if step % eval_every == 0 or step == steps:
                v = val_loss()
                log.append({"step": step, "val_loss": v, "seconds": time.time() - t0, **parts})
                print(f"  finetune step {step}: val {v:.4f} (best {best:.4f})", flush=True)
                if v < best:
                    best, best_state, bad = v, copy.deepcopy(enc.state_dict()), 0
                else:
                    bad += 1
                if bad >= patience or time.time() > deadline:
                    break
    return best_state, log
