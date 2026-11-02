"""CPU smoke test for stage7_final: every component end to end on 2 faces with stand-in models (as smoke_stage5).

Extra stand-ins: a tiny trainable encoder (so K1's fine-tuning loop runs and its gradients flow), FaceNet = small
conv net, HyperStyle = the stand-in generator with a per-face colour offset, Inception = fixed random projection.
Run with the same uv command as tools/smoke_stage5.py (MEDSG_DATA_DIR set).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from smoke_stage5 import StandInArc, StandInFAN, StandInGAN  # noqa: E402


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3 * 8 * 8, 512)

    def forward(self, x):
        z = torch.nn.functional.adaptive_avg_pool2d(x, 8).flatten(1)
        return self.lin(z)[:, None].repeat(1, 18, 1) * 0.1


class StandInGAN7(StandInGAN):
    def __init__(self, base):
        super().__init__(base)
        self.encoder = TinyEncoder()

    def encode(self, x256):
        with torch.no_grad():
            return self.encoder(x256) + self.latent_avg[None]

    def sample_w(self, n, psi, gen):
        return torch.randn(n, 512, generator=gen) * 0.1


class StandInHS:
    def __init__(self, gan):
        self.gan, self.device, self.n_layers = gan, "cpu", 18

    def invert(self, x256):
        return self.gan.encode(x256), torch.full((1, 3, 1, 1), 0.02)

    def for_face(self, deltas):
        gan = self.gan
        return SimpleNamespace(device="cpu", n_layers=18, synth=lambda c, G=None: gan.synth(c) + deltas)


def main() -> None:
    from medsg.align import align_ffhq
    from medsg.landmarks import Landmarker
    import medsg.hyperstyle as H
    import medsg.identity as I
    from medsg.stages import stage5_overnight as S5
    from medsg.stages import stage7_final as S7
    tmp = Path(tempfile.mkdtemp(prefix="smoke7_", dir=os.environ.get("SMOKE_DIR")))
    (tmp / "outputs").mkdir()
    (tmp / "models").mkdir()
    shutil.copy(ROOT / "outputs/splits.json", tmp / "outputs/splits.json")
    os.symlink(ROOT / "models/face_landmarker.task", tmp / "models/face_landmarker.task")
    os.environ["MEDSG_ROOT"] = str(tmp)
    data = Path(os.environ["MEDSG_DATA_DIR"])
    lmk = Landmarker(ROOT / "models/face_landmarker.task")
    rgb = cv2.cvtColor(cv2.imread(str(data / "01_b.jpg")), cv2.COLOR_BGR2RGB)
    gan = StandInGAN7(align_ffhq(rgb, lmk(rgb).lm_px, 1024)[0])

    def fake_lpips(a, b):
        return ((a - b) ** 2).mean(dim=(1, 2, 3), keepdim=True)
    S7.DEVICE = S5.DEVICE = "cpu"
    S5.load_kit = lambda cfg, device: S5.Kit(gan, lmk, fake_lpips, StandInArc(), StandInFAN(lmk), {"smoke": "stand-ins"})
    S7.load_hyperstyle = lambda cfg: StandInHS(gan)
    H.same_generator = lambda a, b: 0.0
    I.FaceNetID = lambda device: SimpleNamespace(net=torch.nn.Sequential(torch.nn.Conv2d(3, 4, 5, 4),
                                                                         torch.nn.Flatten(), torch.nn.Linear(4 * 39 * 39, 512)))
    import medsg.face3d as F3

    class FakeFace3D:                                   # 3D stand-in: "shape" = landmark spread statistics
        def __init__(self, repo, gpu=False):
            pass

        def fit(self, bgr, lm_px):
            return SimpleNamespace(alpha_shp=np.asarray(lm_px)[:, :2].std(0))

        def canonical_metrics(self, a):
            return {"nose_width_3d": float(a[0]), "jaw_width_3d": float(a[1])}
    F3.Face3D = FakeFace3D
    import medsg.depth as DP

    class FakeDepth:                                    # depth stand-in: blurred brightness
        def __init__(self, device):
            pass

        def __call__(self, rgb):
            return cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 5)
    DP.DepthNet = FakeDepth
    P = np.random.default_rng(0).standard_normal((3 * 16 * 16, 16)).astype(np.float32)

    class FakeInception:
        def __init__(self, device):
            pass

        def __call__(self, rgb):
            return (cv2.resize(rgb, (16, 16)).astype(np.float32).reshape(-1) / 255) @ P
    S7.Inception = FakeInception
    S7.N_PTI, S7.N_EXAMPLES = 1, 3
    import medsg.gan as G
    real_pti = G.pti_tune
    G.pti_tune = lambda gan_, pivot, target, lp, steps=300, **kw: real_pti(gan_, pivot, target, lp, steps=3)
    try:
        S7.main(["--limit", "2", "--hours", "8"])
    except SystemExit as e:
        print("SystemExit:", e)
    out = tmp / "outputs" / "stage7"
    st = json.loads((out / "status.json").read_text())
    print({k: v.get("status") if isinstance(v, dict) else v for k, v in st.items()})
    for k, v in st.items():
        if isinstance(v, dict) and v.get("status") == "failed":
            print(k, v["traceback"])
    print(pd.read_csv(out / "k1_finetune_log.csv").round(4).to_string(index=False))
    print(pd.read_csv(out / "k2_inversion.csv")[["pid", "encoder", "ok", "arc_cos_real_rec"]].to_string(index=False))
    e = pd.concat([pd.read_csv(f) for f in out.glob("k3_*_edits_*.csv")])
    print("K3 final rows per encoder file:", {f.name: len(pd.read_csv(f)) for f in out.glob("k3_*_edits_*.csv")})
    print(pd.read_csv(out / "k5_kid.csv").round(3).to_string(index=False))
    print(pd.read_csv(out / "k4_pti.csv").round(3).to_string(index=False))
    print(pd.read_csv(out / "k6_app_timing.csv").round(2).to_string(index=False))
    k7 = pd.read_csv(out / "k7_warp_judges.csv")
    print(k7.filter(regex="pid|target|exact|true_fan_d_nose|meas_fan_nose|meas_d3_nose").round(4).to_string(index=False))
    print(pd.read_csv(out / "k7_judge_noise.csv").round(4).to_string(index=False))
    print(k7.filter(regex="target|depth_").round(3).to_string(index=False))
    print(pd.read_csv(out / "k5_gan_depth.csv").filter(regex="encoder|target|d3_nose|depth_loc|depth_in").round(3).head(12).to_string(index=False))
    print("zips:", [p.name for p in (tmp / "outputs").glob("*.zip")], "| tmp:", tmp, "| k3 rows", len(e))
    assert all(st[c]["status"] == "done" for c in S7.DONE), st


if __name__ == "__main__":
    main()
