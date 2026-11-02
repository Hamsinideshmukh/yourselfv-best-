"""Does the photo check (medsg.quality) catch photos on which MediaPipe's landmarks drift?

60 HDA TEST pre-op photos (raw, as a user would upload, max side 1280 like the app), each clean and degraded:
blur, heavy JPEG, low resolution, dark exposure, part of the face covered. For each image:
  check score / level (exactly the app's code path: MediaPipe -> FAN in MediaPipe's box -> PhotoCheck);
  MediaPipe drift = mean distance of all 478 MediaPipe points from their clean-photo positions (IPD units),
  i.e. how far the degradation moved the landmarks every edit is placed on.
Writes outputs/quality/validation_rows.csv, validation_summary.csv (numbers only; images are never saved).
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from medsg.dedup import EXCLUDE  # noqa: E402
from medsg.judges import FAN68, face_box  # noqa: E402
from medsg.landmarks import FaceNotFound, Landmarker  # noqa: E402
from medsg.quality import PhotoCheck  # noqa: E402

N, MAX_SIDE, SEED = 60, 1280, 7


def degrade(rgb: np.ndarray, kind: str, L: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    ipd = float(np.linalg.norm(L[468] - L[473]))
    if kind == "clean":
        return rgb
    if kind.startswith("blur"):
        return cv2.GaussianBlur(rgb, (0, 0), float(kind[4:]) * ipd / 100)      # sigma in % of IPD
    if kind == "jpeg5":
        ok, b = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 5])
        return cv2.cvtColor(cv2.imdecode(b, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if kind == "lowres":                                                         # face ~24 px between pupils
        s = 24 / ipd
        small = cv2.resize(rgb, (max(8, round(w * s)), max(8, round(h * s))), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "dark":
        return (255 * (rgb / 255.0) ** 3.0).astype(np.uint8)
    if kind in ("cover_mouth", "cover_eye"):
        out = rgb.copy()
        c = L[[61, 291, 17]].mean(0) if kind == "cover_mouth" else L[[33, 133]].mean(0)
        r = int(0.45 * ipd)
        cv2.rectangle(out, tuple((c - r).astype(int)), tuple((c + r).astype(int)), (128, 128, 128), -1)
        return out
    raise ValueError(kind)


KINDS = ["clean", "blur3", "blur6", "jpeg5", "lowres", "dark", "cover_mouth", "cover_eye"]


def main() -> None:
    data = Path(os.environ["MEDSG_DATA_DIR"])
    split = json.load(open(ROOT / "outputs/splits.json"))
    test = [p for p in split["test"] if p not in EXCLUDE]
    rng = np.random.default_rng(SEED)
    sel = [test[i] for i in sorted(rng.choice(len(test), N, replace=False))]
    lmk = Landmarker(ROOT / "models/face_landmarker.task")
    check = PhotoCheck(FAN68("cpu", face_detector="blazeface"))
    rows, offs = [], {}
    for n, pid in enumerate(sel):
        rgb = cv2.cvtColor(cv2.imread(str(data / f"{pid}_b.jpg")), cv2.COLOR_BGR2RGB)
        s = MAX_SIDE / max(rgb.shape[:2])
        if s < 1:
            rgb = cv2.resize(rgb, (round(rgb.shape[1] * s), round(rgb.shape[0] * s)), interpolation=cv2.INTER_AREA)
        try:
            Lc = lmk(rgb).lm_px[:, :2].astype(np.float64)
        except FaceNotFound:
            continue
        ipd = float(np.linalg.norm(Lc[468] - Lc[473]))
        for kind in KINDS:
            img = np.ascontiguousarray(degrade(rgb, kind, Lc))
            r = {"pid": pid, "kind": kind}
            try:
                L = lmk(img).lm_px[:, :2].astype(np.float64)
            except FaceNotFound:
                rows.append({**r, "mp_found": False})
                continue
            r.update(mp_found=True, mp_drift_ipd=float(np.linalg.norm(L - Lc, axis=1).mean() / ipd))
            try:
                q = check(img, np.c_[L, np.zeros(len(L))], face_box(L))
                r.update(score=q["score"], level=q["level"], regions="|".join(q["regions"]))
                offs[f"{pid}_{kind}"] = q["offsets"]
            except ValueError as e:
                r.update(level=f"error: {e}")
            rows.append(r)
        print(f"  {n + 1}/{len(sel)}", flush=True)
    lmk.close()
    out = ROOT / "outputs/quality"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "validation_rows.csv", index=False)
    np.savez_compressed(out / "private_validation_offsets.npz", **offs)      # HDA-derived: never commit
    ok = df[df.mp_found == True].dropna(subset=["score"])  # noqa: E712
    summ = []
    for kind in KINDS:
        s = ok[ok.kind == kind]
        summ.append({"kind": kind, "n": len(s), "mp_not_found": int((df[df.kind == kind].mp_found == False).sum()),  # noqa: E712
                     "median_mp_drift_pct_ipd": 100 * s.mp_drift_ipd.median(),
                     "median_check_score_pct_ipd": 100 * s.score.median(),
                     "rate_caution_or_warn": float((s.level != "ok").mean()), "rate_warn": float((s.level == "warn").mean())})
    S = pd.DataFrame(summ)
    deg = ok[ok.kind != "clean"]
    big = deg.mp_drift_ipd > 0.02                                    # landmarks moved > 2 % IPD (~1.3 mm)
    extra = {"spearman_score_vs_mp_drift (degraded)": deg.score.corr(deg.mp_drift_ipd, method="spearman"),
             "flag_rate_when_mp_drift_gt_2pct": float((deg[big].level != "ok").mean()) if big.any() else np.nan,
             "flag_rate_when_mp_drift_le_2pct": float((deg[~big].level != "ok").mean()) if (~big).any() else np.nan,
             "n_drift_gt_2pct": int(big.sum())}
    S.to_csv(out / "validation_summary.csv", index=False)
    (out / "validation_extra.json").write_text(json.dumps(extra, indent=1))
    pd.set_option("display.width", 200)
    print(S.round(3).to_string(index=False))
    print(json.dumps(extra, indent=1))


if __name__ == "__main__":
    main()
