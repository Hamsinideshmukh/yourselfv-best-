"""Synthetic example faces for the public app (StyleGAN2-FFHQ samples, psi 0.7: not real people).

Candidates are sampled with fixed seeds and filtered to consultation-like photos (frontal, neutral: MediaPipe pose and
smile/jaw-open blendshapes). The final set is chosen by eye from the contact sheet for ADULT faces of varied sex and
skin tone (a cosmetic-surgery demo must not show children); the chosen seeds are listed in KEEP so the set is
reproducible. Run on CPU (~10 min): writes outputs/figures/example_candidates.jpg and app/examples/*.jpg.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from medsg.gan import E4EGAN, install_e4e_package, to_uint8  # noqa: E402
from medsg.landmarks import Landmarker  # noqa: E402
from medsg.measure import BLENDSHAPES, measure  # noqa: E402

N, PSI = 120, 0.7
KEEP = [19, 27, 43, 60, 76, 93]   # chosen 2026-11-02 from the contact sheet: adults, varied sex and skin tone, no glasses


def main() -> None:
    install_e4e_package(ROOT / "repos/encoder4editing", ROOT / "build/src_ext")
    gan = E4EGAN(ROOT / "models/e4e_ffhq_encode.pt", "cpu")
    lmk = Landmarker(ROOT / "models/face_landmarker.task")
    tiles, imgs = [], {}
    for seed in (KEEP or range(N)):          # once KEEP is set, only the chosen seeds are rendered
        w = gan.sample_w(1, PSI, torch.Generator().manual_seed(seed))
        with torch.no_grad():
            img = to_uint8(gan.synth(w[:, None].repeat(1, gan.n_layers, 1)))[0]
        m = measure(img, lmk)
        if m is None:
            continue
        bs = dict(zip(BLENDSHAPES, m["bs"]))
        ok = abs(m["pose"][0]) < 12 and abs(m["pose"][1]) < 15 and bs["mouthSmileLeft"] + bs["mouthSmileRight"] < 1.0 \
            and bs["jawOpen"] < 0.25
        if not ok:
            continue
        imgs[seed] = img
        t = cv2.resize(img, (192, 192), interpolation=cv2.INTER_AREA)
        cv2.putText(t, str(seed), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        tiles.append(t)
        print(f"  seed {seed} ok", flush=True)
    rows = [np.hstack(tiles[i:i + 8] + [np.zeros_like(tiles[0])] * (8 - len(tiles[i:i + 8]))) for i in range(0, len(tiles), 8)]
    if not KEEP:
        cv2.imwrite(str(ROOT / "outputs/figures/example_candidates.jpg"), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    out = ROOT / "app/examples"
    out.mkdir(exist_ok=True)
    for k, seed in enumerate(KEEP):
        cv2.imwrite(str(out / f"synthetic_adult_{k + 1}.jpg"), cv2.cvtColor(imgs[seed], cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    lmk.close()
    print("candidates:", sorted(imgs), "| kept:", KEEP)


if __name__ == "__main__":
    main()
