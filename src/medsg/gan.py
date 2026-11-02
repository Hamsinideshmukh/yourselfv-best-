"""Frozen StyleGAN2 (FFHQ 1024) + e4e encoder, from the official e4e checkpoint.

Engineering notes
- e4e ships rosinality's StyleGAN2, whose two custom CUDA ops (fused_leaky_relu,
  upfirdn2d) are JIT-compiled at import and break on new torch/CUDA. We replace
  them with their exact pure-PyTorch reference implementations (same maths).
- e4e and 3DDFA-V2 both have a top-level package called `models`. We copy e4e's
  package as `e4e_models` with rewritten imports so both can live in one process.
- The e4e checkpoint contains the decoder (generator) it was trained with, so
  inversion and synthesis use exactly matched weights.
"""
from __future__ import annotations

import copy
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

N_LAYERS = 18      # W+ layers at 1024 px
W_DIM = 512

OP_SHIM = '''"""Pure-PyTorch replacements for rosinality's CUDA ops (reference implementations)."""
import torch
import torch.nn.functional as F
from torch import nn


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    rest = [1] * (input.ndim - bias.ndim - 1)
    return F.leaky_relu(input + bias.view(1, bias.shape[0], *rest), negative_slope) * scale


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    return upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)
    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape
    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)
    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[:, max(-pad_y0, 0): out.shape[1] - max(-pad_y1, 0),
              max(-pad_x0, 0): out.shape[2] - max(-pad_x1, 0), :]
    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
                      in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1)
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
    return out.view(-1, channel, out_h, out_w)
'''


def install_vendor(repo: Path, rename: dict[str, str], dst_root: Path, op_dir: str) -> None:
    """Copy packages from a cloned repo to dst_root under new names (avoids clashes such as several top-level
    `models` packages), rewrite their imports, and swap the CUDA ops in `op_dir` for the pure-PyTorch shim."""
    for old, new in rename.items():
        src, dst = Path(repo) / old, Path(dst_root) / new
        if not src.exists():
            raise FileNotFoundError(f"{src} not found (clone the repo first)")
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.cu", "*.cpp", "__pycache__"))
    for new in rename.values():
        for f in (Path(dst_root) / new).rglob("*.py"):
            t = f.read_text()
            for old, nn in rename.items():
                t = re.sub(rf"\b(from|import) {old}\b", rf"\1 {nn}", t)
            f.write_text(t)
        for d in [Path(dst_root) / new, *[q for q in (Path(dst_root) / new).rglob("*") if q.is_dir()]]:
            (d / "__init__.py").touch(exist_ok=True)
    opdir = Path(dst_root) / op_dir
    for f in opdir.glob("*.py"):
        f.unlink()
    (opdir / "__init__.py").write_text(OP_SHIM)
    if str(dst_root) not in sys.path:
        sys.path.insert(0, str(dst_root))


def install_e4e_package(e4e_repo: Path, dst_root: Path) -> None:
    """e4e's `models` package -> dst_root/e4e_models (imports rewritten, CUDA-free ops)."""
    install_vendor(e4e_repo, {"models": "e4e_models"}, dst_root, "e4e_models/stylegan2/op")


class E4EGAN:
    """e4e encoder + matched StyleGAN2 generator. Images are float tensors in [-1, 1], NCHW."""

    def __init__(self, ckpt_path: Path | None, device: str, size: int = 1024):
        from e4e_models.encoders.psp_encoders import Encoder4Editing
        from e4e_models.stylegan2.model import Generator
        self.device, self.size = device, size
        self.n_layers = 2 * int(np.log2(size)) - 2
        self.encoder = Encoder4Editing(50, "ir_se", SimpleNamespace(stylegan_size=size)).eval()
        self.G = Generator(size, W_DIM, 8, channel_multiplier=2).eval()
        if ckpt_path is not None:
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = ck["state_dict"]
            self.encoder.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("encoder.")}, strict=True)
            self.G.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("decoder.")}, strict=True)
            avg = ck["latent_avg"]
        else:                                   # random weights: CPU unit tests only
            with torch.no_grad():
                avg = self.G.mean_latent(256)
        avg = avg.float().reshape(-1, W_DIM)
        self.latent_avg = (avg if avg.shape[0] == self.n_layers else avg[:1].repeat(self.n_layers, 1)).to(device)
        assert self.latent_avg.shape == (self.n_layers, W_DIM), self.latent_avg.shape
        self.encoder.to(device)
        self.G.to(device)
        for p in [*self.encoder.parameters(), *self.G.parameters()]:
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, x256: torch.Tensor) -> torch.Tensor:
        """(B,3,256,256) in [-1,1] -> W+ codes (B, n_layers, 512)."""
        assert x256.ndim == 4 and x256.shape[1:] == (3, 256, 256), x256.shape
        codes = self.encoder(x256.to(self.device)) + self.latent_avg.unsqueeze(0)
        assert codes.shape[1:] == (self.n_layers, W_DIM), codes.shape
        return codes

    def synth(self, codes: torch.Tensor, G=None) -> torch.Tensor:
        """W+ codes -> images (B,3,size,size) in [-1,1], fixed noise (deterministic)."""
        assert codes.ndim == 3 and codes.shape[1:] == (self.n_layers, W_DIM), codes.shape
        img, _ = (G or self.G)([codes.to(self.device)], input_is_latent=True, randomize_noise=False)
        return img

    @torch.no_grad()
    def sample_w(self, n: int, psi: float, gen: torch.Generator) -> torch.Tensor:
        """Sample W (n,512) through the mapping network with truncation psi towards the mean w."""
        z = torch.randn(n, W_DIM, generator=gen).to(self.device)
        w = self.G.style(z)
        return self.latent_avg[0] + psi * (w - self.latent_avg[0])


def to_uint8(img: torch.Tensor) -> np.ndarray:
    """(B,3,H,W) [-1,1] -> (B,H,W,3) uint8 RGB."""
    return ((img.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def from_uint8(rgb: np.ndarray, size: int, device: str) -> torch.Tensor:
    """(H,W,3) uint8 RGB -> (1,3,size,size) [-1,1] tensor (area-resized)."""
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1
    if t.shape[-1] != size:
        t = F.interpolate(t, size=(size, size), mode="area")
    return t.to(device)


def pti_tune(gan: E4EGAN, pivot: torch.Tensor, target: torch.Tensor, lpips_fn,
             steps: int = 300, lr: float = 3e-4, loss_res: int = 256) -> torch.nn.Module:
    """Pivotal Tuning Inversion (Roich et al., 2022): fine-tune a copy of G so G'(pivot) matches target.

    pivot: (1, L, 512) e4e code (kept fixed). target: (1,3,S,S) in [-1,1].
    Loss = L2 + LPIPS at loss_res. Returns the tuned generator copy.
    """
    G = copy.deepcopy(gan.G).train()
    for p in G.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(G.parameters(), lr=lr)
    tgt = F.interpolate(target.to(gan.device), size=(loss_res, loss_res), mode="area")
    with torch.enable_grad():                   # robust to libraries that disable grad globally
        for _ in range(steps):
            out = F.interpolate(gan.synth(pivot, G), size=(loss_res, loss_res), mode="area")
            loss = F.mse_loss(out, tgt) + lpips_fn(out, tgt).mean()
            assert loss.requires_grad, "PTI loss has no grad: gradient tracking is off"
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    G.eval()
    for p in G.parameters():
        p.requires_grad_(False)
    return G
