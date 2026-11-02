"""Face-identity similarity with FaceNet (InceptionResnetV1, VGGFace2 weights; Schroff 2015 / Cao 2018).

Used as the identity metric because its weights are on GitHub releases (reliable on
Kaggle). Note: the HyperStyle benchmark numbers (e4e 0.50, PTI 0.85) use CurricularFace,
so absolute values are NOT comparable to those; only our own baselines are.
Input: FFHQ-aligned crops; the face region is the central ~60% of an FFHQ crop.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class FaceNetID:
    def __init__(self, device: str):
        from facenet_pytorch import InceptionResnetV1
        self.net = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        self.device = device

    @torch.no_grad()
    def embed(self, img: torch.Tensor) -> torch.Tensor:
        """(B,3,S,S) FFHQ-aligned in [-1,1] -> L2-normalised (B,512) embeddings."""
        s = img.shape[-1]
        a, b = int(0.20 * s), int(0.80 * s)
        crop = img[:, :, a:b, a:b]
        crop = F.interpolate(crop.to(self.device), size=(160, 160), mode="area")
        # facenet fixed_image_standardization: (x*255 - 127.5) / 128  ==  x_in_[-1,1] * 127.5/128
        e = self.net(crop * (127.5 / 128.0))
        return F.normalize(e, dim=1)

    def cosine(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (self.embed(a) * self.embed(b)).sum(1)
