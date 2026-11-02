"""CPU smoke test for stage5_overnight: runs every component end to end on 2 faces with stand-in models.

Stand-ins (smoke only; the real run loads the real models and has no access to these):
- generator: a real aligned HDA photo whose nose region bulges sideways by an amount set by the middle-layer
  latent (torch grid_sample, so PTI gradients flow); the closed loop therefore has a real, measurable target.
- ArcFace: normalised pixels; FAN: MediaPipe points mapped to 68-point indices; LPIPS: mean squared error.
Run:  MEDSG_DATA_DIR=... uv run --no-project --offline --python 3.12 --index https://download.pytorch.org/whl/cpu
      --index-strategy unsafe-best-match --with torch==2.15.0+cpu --with numpy --with pandas --with scipy
      --with matplotlib --with opencv-python-headless --with mediapipe python tools/smoke_stage5.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

MP68 = [234, 93, 132, 58, 172, 136, 150, 149, 152, 378, 379, 365, 397, 288, 361, 323, 454,
        70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 168, 197, 5, 4, 98, 97, 2, 326, 327,
        33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
        61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, 78, 82, 13, 312, 308, 317, 14, 87]


class StandInG(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(3))


class StandInGAN:
    def __init__(self, base_rgb: np.ndarray, S: int = 256):
        self.device, self.size, self.n_layers = "cpu", S, 18
        t = torch.from_numpy(cv2.resize(base_rgb, (S, S), interpolation=cv2.INTER_AREA)).permute(2, 0, 1).float()
        self.base = (t / 127.5 - 1).unsqueeze(0)
        self.G = StandInG()
        g = torch.Generator().manual_seed(0)
        self.v = torch.randn(512, generator=g) / 512 ** 0.5
        self.latent_avg = torch.zeros(18, 512)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, S), torch.linspace(-1, 1, S), indexing="ij")
        self.xx, self.yy = xx, yy
        self.bump = torch.exp(-((xx ** 2) + (yy - 0.25) ** 2) / (2 * 0.12 ** 2))   # nose region of an FFHQ crop

    def encode(self, x256):
        assert x256.shape[1:] == (3, 256, 256)
        return self.latent_avg.unsqueeze(0).repeat(len(x256), 1, 1) + 0.01

    def synth(self, codes, G=None):
        assert codes.ndim == 3 and codes.shape[1:] == (18, 512), codes.shape
        s = torch.tanh(codes[:, 4:8, :].float() @ self.v).mean(1) * 0.15       # sideways bulge strength
        dx = s[:, None, None] * self.xx * self.bump                              # move content outward
        grid = torch.stack([self.xx - dx, self.yy.expand_as(dx)], -1)
        img = F.grid_sample(self.base.expand(len(codes), -1, -1, -1), grid, align_corners=True,
                            padding_mode="border")
        return img + (G or self.G).bias.view(1, 3, 1, 1)


class StandInArc:
    def embed(self, rgb, lm):
        e = cv2.resize(rgb, (16, 16)).astype(np.float64).ravel() - 127.5
        return e / np.linalg.norm(e)


class StandInFAN:
    def __init__(self, lmk):
        self.lmk = lmk

    def landmarks(self, rgb, box):
        return self.lmk(np.ascontiguousarray(rgb)).lm_px[MP68, :2].astype(np.float64)


def main() -> None:
    from medsg.align import align_ffhq
    from medsg.landmarks import Landmarker
    from medsg.stages import stage5_overnight as S5
    tmp = Path(tempfile.mkdtemp(prefix="smoke5_", dir=os.environ.get("SMOKE_DIR")))
    (tmp / "outputs").mkdir()
    (tmp / "models").mkdir()
    shutil.copy(ROOT / "outputs/splits.json", tmp / "outputs/splits.json")
    os.symlink(ROOT / "models/face_landmarker.task", tmp / "models/face_landmarker.task")
    os.environ["MEDSG_ROOT"] = str(tmp)
    data = Path(os.environ["MEDSG_DATA_DIR"])
    lmk = Landmarker(ROOT / "models/face_landmarker.task")
    rgb = cv2.cvtColor(cv2.imread(str(data / "01_b.jpg")), cv2.COLOR_BGR2RGB)
    aligned, _ = align_ffhq(rgb, lmk(rgb).lm_px, 1024)
    gan = StandInGAN(aligned)

    def fake_lpips(a, b):
        return ((a - b) ** 2).mean(dim=(1, 2, 3), keepdim=True)
    S5.DEVICE = "cpu"
    S5.load_kit = lambda cfg, device: S5.Kit(gan, lmk, fake_lpips, StandInArc(), StandInFAN(lmk),
                                            {"smoke": "stand-in models"})
    S5.N_PHANTOM, S5.N_PTI = 2, 1
    S5.GROUP_K = {1: 3, 0: 1, 2: 1}
    import medsg.gan as G
    real_pti = G.pti_tune
    G.pti_tune = lambda gan_, pivot, target, lp, **kw: real_pti(gan_, pivot, target, lp, steps=5)
    S5.main(["--limit", "2", "--hours", "2"])
    out = tmp / "outputs" / "stage5"
    import json
    import pandas as pd
    st = json.loads((out / "status.json").read_text())
    print({k: v.get("status") if isinstance(v, dict) else v for k, v in st.items()})
    assert all(st[c]["status"] == "done" for c in ("c1", "c2", "c3", "c4", "c5")), st
    c4 = pd.concat([pd.read_csv(f) for f in out.glob("c4_edits_*.csv")])
    fin = c4[(c4.method == "closed") & (c4.final == True)]  # noqa: E712
    print(fin[["pid", "split", "target", "request_kind", "request", "iter", "a", "status"]].to_string(index=False))
    nw = fin[fin.target == "nose_width"]
    print("nose_width requested vs achieved:", list(zip(nw.request.round(4), nw.d_nose_width.round(4))))
    print("C2 rows", len(pd.read_csv(out / "c2_phantom.csv")), "| C5 rows", len(pd.read_csv(out / "c5_pti.csv")))
    print("zips:", [p.name for p in (tmp / "outputs").glob("*.zip")], "| tmp:", tmp)


if __name__ == "__main__":
    main()
