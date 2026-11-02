"""CPU smoke test for stage8_consensus with the stand-in models of smoke_stage5/7 (same uv command)."""
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

from smoke_stage5 import StandInArc, StandInFAN  # noqa: E402
from smoke_stage7 import StandInGAN7, StandInHS  # noqa: E402


def main() -> None:
    from medsg.align import align_ffhq
    from medsg.landmarks import Landmarker
    import medsg.depth as DP
    import medsg.face3d as F3
    from medsg.stages import stage5_overnight as S5
    from medsg.stages import stage7_final as S7
    from medsg.stages import stage8_consensus as S8
    tmp = Path(tempfile.mkdtemp(prefix="smoke8_", dir=os.environ.get("SMOKE_DIR")))
    (tmp / "outputs").mkdir()
    (tmp / "models").mkdir()
    shutil.copy(ROOT / "outputs/splits.json", tmp / "outputs/splits.json")
    os.symlink(ROOT / "models/face_landmarker.task", tmp / "models/face_landmarker.task")
    os.environ["MEDSG_ROOT"] = str(tmp)
    lmk = Landmarker(ROOT / "models/face_landmarker.task")
    rgb = cv2.cvtColor(cv2.imread(str(Path(os.environ["MEDSG_DATA_DIR"]) / "01_b.jpg")), cv2.COLOR_BGR2RGB)
    gan = StandInGAN7(align_ffhq(rgb, lmk(rgb).lm_px, 1024)[0])
    S5.DEVICE = S7.DEVICE = S8.DEVICE = "cpu"
    S5.load_kit = lambda cfg, device: S5.Kit(gan, lmk, None, StandInArc(), StandInFAN(lmk), {"smoke": "stand-ins"})
    S7.load_hyperstyle = lambda cfg: StandInHS(gan)

    class FakeFace3D:
        def __init__(self, repo, gpu=False):
            pass

        def fit(self, bgr, lm_px):
            return SimpleNamespace(alpha_shp=np.asarray(lm_px)[:, :2].std(0))

        def canonical_metrics(self, a):
            return {"nose_width_3d": float(a[0]), "palpebral_height_3d": float(a[1]), "brow_height_3d": float(a[1])}

    class FakeDepth:
        def __init__(self, device):
            pass

        def __call__(self, rgb):
            return cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 5)
    F3.Face3D, DP.DepthNet = FakeFace3D, FakeDepth
    S8.main(["--limit", "2"])
    out = tmp / "outputs" / "stage8"
    st = json.loads((out / "status.json").read_text())
    print({k: v.get("status") if isinstance(v, dict) else v for k, v in st.items()})
    d = pd.read_csv(out / "s1_arms.csv")
    print(d[["pid", "target", "arm", "status", "renders", "a", "z_mp", "z_fan", "steered", "latent_norm",
             "depth_locality", "arc_edit_real"]].round(4).to_string(index=False))
    print(pd.read_csv(out / "s2_app_hyperstyle.csv").round(3).to_string(index=False))
    print(len(pd.read_csv(out / "s1_judge_noise_rows.csv")), "noise rows | tmp", tmp)
    assert all(st[c]["status"] == "done" for c in S8.DONE), st


if __name__ == "__main__":
    main()
