"""HyperStyle (Alaluf et al., CVPR 2022): fast per-photo generator refinement, compared against e4e and PTI.

How it works, plainly: an e4e encoder gives a code w (in W, the editable latent space). A second network (a
hypernetwork) looks at the photo and the current rebuild and predicts small per-photo *offsets to the generator's
convolution weights*, refined over 5 passes. That is what PTI does with ~300 optimisation steps (~60 s); HyperStyle
predicts it in one forward pass (~1 s). Codes stay in W, so our anatomy directions apply unchanged; the offsets play
the role of PTI's tuned generator. Reported identity (CurricularFace, CelebA-HQ): e4e 0.50, HyperStyle 0.76, PTI 0.85.
Code: github.com/yuval-alaluf/hyperstyle (MIT). Weights: official Google Drive links in its README.
"""
from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path

import torch

from .gan import N_LAYERS, install_vendor


class HyperStyleGAN:
    """Loads HyperStyle; `invert` returns (codes (B,18,512), weight offsets); `for_face` gives a renderer."""

    def __init__(self, repo: Path, ckpt: Path, w_encoder_ckpt: Path, dst_root: Path, device: str = "cuda"):
        install_vendor(repo, {"models": "hs_models", "configs": "hs_configs", "utils": "hs_utils"}, dst_root,
                       "hs_models/stylegan2/op")
        for f in (Path(dst_root) / "hs_models").rglob("*.py"):   # upstream hard-codes .cuda() (crashes on CPU);
            t = f.read_text()                                    # those modules are moved by .to(device) below and
            if ".cuda()" in t:                                   # their random init is overwritten by the checkpoint
                f.write_text(t.replace(".cuda()", ""))
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")   # upstream checkpoints store argparse opts
        from hs_models.hyperstyle import HyperStyle
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        opts = dict(ck["opts"])
        del ck
        opts.update(checkpoint_path=str(ckpt), load_w_encoder=True, w_encoder_checkpoint_path=str(w_encoder_ckpt),
                    device=device)
        self.net = HyperStyle(Namespace(**opts)).eval().to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device, self.n_layers, self.n_iters = device, N_LAYERS, 5

    @torch.no_grad()
    def invert(self, x256: torch.Tensor):
        """(1,3,256,256) aligned photo in [-1,1] -> (codes (1,18,512), weight offsets). 5 refinement passes."""
        assert x256.shape[1:] == (3, 256, 256), x256.shape
        y_hat = codes = deltas = None
        for _ in range(self.n_iters):
            y_hat, _, deltas, codes, _ = self.net(x256.to(self.device), y_hat=y_hat, codes=codes,
                                                  weights_deltas=deltas, return_latents=True, resize=True,
                                                  randomize_noise=False, return_weight_deltas_and_codes=True)
        assert codes.shape[1:] == (self.n_layers, 512), codes.shape
        return codes, deltas

    def for_face(self, deltas) -> "_FaceRenderer":
        return _FaceRenderer(self, deltas)


class _FaceRenderer:
    """Same interface as E4EGAN.synth (used by stage5 render / closed loop), with one face's weight offsets."""

    def __init__(self, hs: HyperStyleGAN, deltas):
        self.hs, self.deltas, self.device, self.n_layers = hs, deltas, hs.device, hs.n_layers

    def synth(self, codes: torch.Tensor, G=None) -> torch.Tensor:
        assert codes.ndim == 3 and codes.shape[1:] == (self.n_layers, 512), codes.shape
        out = []
        for c in codes:                                  # offsets are per face (batch 1); loop over code variants
            img, _ = self.hs.net.decoder([c[None].to(self.device)], weights_deltas=self.deltas, input_is_latent=True,
                                         randomize_noise=False)
            out.append(img)
        return torch.cat(out)


def decoder_tensors(ckpt: Path) -> dict:
    """Generator ("decoder.*") tensors of an e4e or HyperStyle checkpoint (safe load: tensors + argparse opts only)."""
    import argparse
    torch.serialization.add_safe_globals([argparse.Namespace])
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]
    return {k[len("decoder."):]: v for k, v in sd.items() if k.startswith("decoder.")}


def same_generator(ckpt_a: Path, ckpt_b: Path) -> float:
    """Max |difference| between two checkpoints' generators (0.0 = same latent space). Raises if layouts differ.
    (Stage 7 compared live modules instead and wrongly reported a layout mismatch; checked 2026-11-02 on the files:
    HyperStyle's and e4e's generators are identical, 171 tensors, max difference 0.0.)"""
    a, b = decoder_tensors(ckpt_a), decoder_tensors(ckpt_b)
    if set(a) != set(b) or any(a[k].shape != b[k].shape for k in a):
        raise ValueError("generators have different layouts")
    return max(float((a[k].float() - b[k].float()).abs().max()) for k in a)
