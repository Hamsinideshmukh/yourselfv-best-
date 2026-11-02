"""Public demo figure: warp mode on SYNTHETIC adult faces (app/examples; GAN-generated, not real people; no HDA image).

Rows: three synthetic faces. Columns: original, then one clinically sized request each (alar-junction width -3.5 mm,
medial brow +3 mm, jaw width -8 mm, approx. mm via IPD 63.36 mm), each with a zoom on the operated region and the
main landmark motion drawn x5. Uses the app's own code path (transfer.edit_multi with the app's couplings).
CPU, ~1 min. Writes outputs/figures/fig_demo_warp.jpg.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from medsg.landmarks import Landmarker  # noqa: E402
from medsg.transfer import COUPLED, IPD_MM, Transfer, warp_photo  # noqa: E402

FACES = ["synthetic_adult_2.jpg", "synthetic_adult_4.jpg", "synthetic_adult_5.jpg"]
REQ = [("nose_width", -3.5, "Nose (alar-junction width) -3.5 mm", [129, 358, 4, 2], 0.75),
       ("brow_medial", 3.0, "Brow (medial) +3 mm", [107, 336, 105, 334], 0.9),
       ("jaw_width", -8.0, "Jaw width -8 mm", [172, 397, 152, 2], 1.45)]      # last: zoom radius in IPD
APP_COUPLED = {**COUPLED, "jaw_width": [], "face_width": []}       # = app/app.py
S = 300                                                           # tile size


def zoom(img, L, idx, ipd, k):
    c = L[idx].mean(0)
    r = k * ipd
    x0, y0 = np.clip((c - r).astype(int), 0, None)
    return cv2.resize(img[y0:int(c[1] + r), x0:int(c[0] + r)], (S, S), interpolation=cv2.INTER_CUBIC)


def label(t, text):
    t = t.copy()
    cv2.rectangle(t, (0, 0), (S, 26), (252, 252, 251), -1)
    cv2.putText(t, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (11, 11, 11), 1, cv2.LINE_AA)
    return t


def main() -> None:
    lmk, T = Landmarker(ROOT / "models/face_landmarker.task"), Transfer()
    rows = []
    for f in FACES:
        rgb = cv2.cvtColor(cv2.imread(str(ROOT / "app/examples" / f)), cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (768, 768), interpolation=cv2.INTER_AREA)
        L = lmk(rgb).lm_px[:, :2].astype(np.float64)
        ipd = float(np.linalg.norm(L[468] - L[473]))
        tiles = [label(cv2.resize(rgb, (S, S), interpolation=cv2.INTER_AREA), "Original (synthetic face)")]
        for key, mm, text, idx, k in REQ:
            e = T.edit_multi(L, {key: mm / IPD_MM}, couplings=APP_COUPLED)
            assert abs((e.after[key] - e.before[key]) * IPD_MM - mm) < 0.01, "exact change must equal the request"
            out = warp_photo(rgb, L, e.field_px)
            zb, za = zoom(rgb, L, idx, ipd, k), zoom(out, L, idx, ipd, k)
            pair = np.hstack([label(zb, "before (zoom)"), label(za, text)])
            tiles.append(cv2.resize(pair, (2 * S, S)))
        rows.append(np.hstack(tiles))
    lmk.close()
    fig = np.vstack(rows)
    out = ROOT / "outputs/figures/fig_demo_warp.jpg"
    cv2.imwrite(str(out), cv2.cvtColor(fig, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote", out, fig.shape)


if __name__ == "__main__":
    main()
