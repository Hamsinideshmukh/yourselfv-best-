"""Stage 1: measurement foundation on all HDA pairs (CPU only).

Produces, under outputs/stage1/:
  images.csv          per-image pose, expression blendshapes, 2D + 3D metrics, failures
  shape_params.npz    3DDFA shape/expression coefficients per image
  noise_floor.csv     SEM and MDC95 per metric from harmless re-captures of the same photo
  pairs.csv           per-pair deltas and inclusion flags
  summary.csv         per-metric population change with bootstrap CI, Wilcoxon + Holm, SNR
  checks.csv          pose agreement (MediaPipe vs 3DDFA), pose contamination 2D vs 3D
  fig_forest.png, fig_snr.png, private/fig_depth_*.png (contains faces; never commit)
  splits.json, run_meta.json

Run:  python -m medsg.stages.stage1_measure   (with src/ on PYTHONPATH)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import stats as sps

from ..config import get_config
from ..data import list_pairs, make_split, seed_everything
from ..face3d import METRICS_3D, Face3D
from ..landmarks import FaceNotFound, Landmarker, ensure_task_file
from ..metrics2d import METRICS_2D, compute_metrics_2d
from ..plots import forest, snr_bars
from ..runmeta import write_run_meta
from ..stats import bootstrap_ci, holm, mdc95, sem_within

BLENDSHAPES = ["jawOpen", "mouthSmileLeft", "mouthSmileRight", "eyeBlinkLeft", "eyeBlinkRight",
               "eyeSquintLeft", "eyeSquintRight", "browInnerUp", "browDownLeft", "browDownRight"]
M2D, M3D = list(METRICS_2D), list(METRICS_3D)
# 2D metric -> its 3D counterpart, for convergent-validity checks
PAIRED = {m: f"{m}_3d" for m in M2D if f"{m}_3d" in METRICS_3D}


def read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read {path}")
    return img


def measure(bgr: np.ndarray, lmk: Landmarker, f3d: Face3D) -> tuple[dict, np.ndarray, np.ndarray]:
    """All measurements for one image. Raises FaceNotFound / ValueError on failure."""
    row, a_shp, a_exp, _ = measure_full(bgr, lmk, f3d)
    return row, a_shp, a_exp


def measure_full(bgr: np.ndarray, lmk: Landmarker, f3d: Face3D):
    """Like measure(), also returning the MediaPipe observation (landmarks)."""
    obs = lmk(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    fit = f3d.fit(bgr, obs.lm_px)
    row = {"n_faces": obs.n_faces, "h": obs.img_hw[0], "w": obs.img_hw[1],
           "ipd_px": float(np.linalg.norm(obs.lm_px[468, :2] - obs.lm_px[473, :2])),
           "yaw": obs.yaw, "pitch": obs.pitch, "roll": obs.roll,
           "yaw_3d": fit.yaw, "pitch_3d": fit.pitch, "roll_3d": fit.roll}
    row.update({f"bs_{k}": obs.blendshapes.get(k, np.nan) for k in BLENDSHAPES})
    row.update(compute_metrics_2d(obs.lm_px))
    row.update(f3d.canonical_metrics(fit.alpha_shp))
    return row, fit.alpha_shp, fit.alpha_exp, obs


def perturb(bgr: np.ndarray, kind: str) -> np.ndarray:
    """Harmless re-captures of the same photo for the noise floor."""
    h, w = bgr.shape[:2]
    if kind == "orig":
        return bgr
    if kind == "flip":
        return cv2.flip(bgr, 1)
    if kind == "shift":
        M = np.float32([[1, 0, 0.02 * w], [0, 1, 0.02 * h]])
        return cv2.warpAffine(bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if kind == "rotate3":
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 3, 1.0)
        return cv2.warpAffine(bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if kind == "downscale":
        return cv2.resize(bgr, (int(w * 0.6), int(h * 0.6)), interpolation=cv2.INTER_AREA)
    if kind == "jpeg70":
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if kind == "gamma":
        lut = (255 * (np.arange(256) / 255) ** 0.8).astype(np.uint8)
        return cv2.LUT(bgr, lut)
    raise ValueError(kind)


PERTURBATIONS = ["orig", "flip", "shift", "rotate3", "downscale", "jpeg70", "gamma"]


def main(limit: int | None = None) -> None:
    cfg = get_config()
    seed_everything(cfg.seed)
    out = cfg.out_dir / "stage1"
    (out / "private").mkdir(parents=True, exist_ok=True)
    task = ensure_task_file(cfg.models_dir)
    repo = cfg.repos_dir / "3DDFA_V2"
    lmk, f3d = Landmarker(task), Face3D(repo)

    pairs = list_pairs(cfg)
    print(f"{len(pairs)} pairs (pre={cfg.pre_suffix}, post={cfg.post_suffix})")
    split = make_split(pairs, cfg, cfg.out_dir / "splits.json")
    if limit:
        pairs = pairs[:limit]

    # ---- 1. per-image measurement -------------------------------------------------
    rows, shp, exp = [], {}, {}
    t0 = time.time()
    for i, p in enumerate(pairs):
        for role, path in (("pre", p.pre), ("post", p.post)):
            r = {"pid": p.pid, "role": role, "file": path.name, "ok": False, "error": ""}
            try:
                m, a_s, a_e = measure(read_bgr(path), lmk, f3d)
                r.update(m, ok=True)
                shp[f"{p.pid}{role}"], exp[f"{p.pid}{role}"] = a_s, a_e
            except (FaceNotFound, ValueError) as e:
                r["error"] = f"{type(e).__name__}: {e}"
            rows.append(r)
        if i % 50 == 0:
            print(f"  {i}/{len(pairs)} pairs, {time.time() - t0:.0f}s")
    img = pd.DataFrame(rows)
    img.to_csv(out / "images.csv", index=False)
    np.savez(out / "shape_params.npz", **{f"shp_{k}": v for k, v in shp.items()},
             **{f"exp_{k}": v for k, v in exp.items()})
    n_fail = int((~img["ok"]).sum())
    print(f"measured {len(img)} images, {n_fail} failures")
    if n_fail > 0.05 * len(img):
        raise RuntimeError(f"{n_fail} measurement failures (>5%); inspect images.csv before continuing")

    # ---- 2. noise floor (dev split, pre-op images only) ------------------------------
    rng = np.random.default_rng(cfg.seed)
    dev = [p for p in pairs if p.pid in set(split["dev"])]
    sample = rng.choice(len(dev), size=min(cfg.noise_n_images, len(dev)), replace=False)
    reps = []
    for j in sample:
        bgr = read_bgr(dev[j].pre)
        try:
            vals = [measure(perturb(bgr, k), lmk, f3d)[0] for k in PERTURBATIONS]
        except (FaceNotFound, ValueError):
            continue
        reps.append(vals)
    print(f"noise floor from {len(reps)} images x {len(PERTURBATIONS)} perturbations")
    noise = []
    for m in M2D + M3D:
        arr = np.array([[v[m] for v in vals] for vals in reps])
        sem = sem_within(arr)
        noise.append({"metric": m, "family": "2D" if m in METRICS_2D else "3D", "sem": sem,
                      "mdc95": mdc95(sem), "n_images": len(reps)})
    noise = pd.DataFrame(noise).set_index("metric")
    noise.to_csv(out / "noise_floor.csv")

    # ---- 3. pairs ---------------------------------------------------------------------
    ok = img[img["ok"]]
    pre = ok[ok.role == "pre"].set_index("pid")
    post = ok[ok.role == "post"].set_index("pid")
    ids = pre.index.intersection(post.index)
    pr = pd.DataFrame(index=ids)
    pr["split"] = ["dev" if i in set(split["dev"]) else "test" for i in ids]
    for c in ["yaw", "pitch", "roll", "yaw_3d", "pitch_3d"]:
        pr[f"d_{c}"] = post.loc[ids, c] - pre.loc[ids, c]
    pr["include_pose"] = (pr.d_yaw.abs() <= cfg.max_abs_dyaw_deg) & (pr.d_pitch.abs() <= cfg.max_abs_dpitch_deg)
    pr["multi_face"] = (pre.loc[ids, "n_faces"] > 1) | (post.loc[ids, "n_faces"] > 1)
    for m in M2D + M3D:
        pr[f"pre_{m}"] = pre.loc[ids, m]
        pr[f"d_{m}"] = post.loc[ids, m] - pre.loc[ids, m]
    pr.to_csv(out / "pairs.csv")
    inc = pr[pr.include_pose]
    print(f"{len(pr)} measurable pairs, {len(inc)} pass the pre-declared pose filter")

    # ---- 4. population summary --------------------------------------------------------
    summ = []
    for m in M2D + M3D:
        d = inc[f"d_{m}"].to_numpy()
        mdc = noise.loc[m, "mdc95"]
        est, lo, hi = bootstrap_ci(d, n_boot=cfg.n_boot, seed=cfg.seed)
        summ.append({
            "metric": m, "family": "2D" if m in METRICS_2D else "3D",
            "region": (METRICS_2D.get(m) or METRICS_3D[m])[0],
            "exploratory": bool(METRICS_3D[m][2]) if m in METRICS_3D else False,
            "n": len(d), "pre_mean": inc[f"pre_{m}"].mean(),
            "delta_mean": est, "ci_lo": lo, "ci_hi": hi, "delta_median": float(np.median(d)),
            "wilcoxon_p": float(sps.wilcoxon(d).pvalue), "mdc95": mdc,
            "frac_up_gt_mdc": float((d > mdc).mean()), "frac_down_gt_mdc": float((d < -mdc).mean()),
            "snr": float(np.std(d, ddof=1) / (np.sqrt(2) * noise.loc[m, "sem"])),
            "z_mean": est / mdc, "z_lo": lo / mdc, "z_hi": hi / mdc,
        })
    summ = pd.DataFrame(summ)
    summ["holm_p"] = holm(summ.wilcoxon_p.to_numpy())
    summ.to_csv(out / "summary.csv", index=False)

    # ---- 5. checks ----------------------------------------------------------------------
    checks = []
    for a, b in (("yaw", "yaw_3d"), ("pitch", "pitch_3d"), ("roll", "roll_3d")):
        r = sps.pearsonr(ok[a], ok[b])
        checks.append({"check": f"pose agreement {a}: MediaPipe vs 3DDFA (pearson r)", "value": r.statistic,
                       "n": len(ok)})
    dpose = np.hypot(pr.d_yaw, pr.d_pitch)
    for m2, m3 in PAIRED.items():
        for m in (m2, m3):
            rho = sps.spearmanr(pr[f"d_{m}"].abs(), dpose).statistic
            checks.append({"check": f"pose contamination: spearman(|d {m}|, |d pose|)", "value": rho, "n": len(pr)})
        rho = sps.spearmanr(inc[f"d_{m2}"], inc[f"d_{m3}"]).statistic
        checks.append({"check": f"convergent validity: spearman(d {m2}, d {m3})", "value": rho, "n": len(inc)})
    pd.DataFrame(checks).to_csv(out / "checks.csv", index=False)

    # ---- 6. figures ---------------------------------------------------------------------
    clin = summ[~summ.exploratory]
    forest(clin, out / "fig_forest.png",
           f"HDA pre→post change (n={len(inc)} pose-matched pairs, 95% CI)")
    snr_bars(clin[["metric", "family", "snr"]], out / "fig_snr.png")
    _depth_examples(pairs[:2], lmk, f3d, out / "private")

    write_run_meta(out, cfg.as_dict(),
                   {"face_landmarker.task": task, "mb1_120x120.pth": repo / "weights/mb1_120x120.pth",
                    "bfm_noneck_v3.pkl": repo / "configs/bfm_noneck_v3.pkl"},
                   {"stage": "stage1_measure", "n_pairs": len(pairs), "n_failures": n_fail,
                    "noise_perturbations": PERTURBATIONS})
    print(summ[["metric", "n", "delta_mean", "ci_lo", "ci_hi", "mdc95", "snr", "holm_p"]].round(4).to_string())
    print(pd.DataFrame(checks).round(3).to_string())
    print(img.loc[~img.ok, ["file", "error"]].head(20).to_string())
    _bundle_results(out, cfg.out_dir / "stage1_results.zip")
    lmk.close()


def _bundle_results(out: Path, zip_path: Path) -> None:
    """Zip shareable results (tables, charts, metadata). Excludes private/ (faces) and per-image arrays."""
    import zipfile
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.iterdir()):
            if f.is_file() and f.suffix in {".csv", ".json", ".png"}:
                z.write(f, f"stage1/{f.name}")
        z.write(out.parent / "splits.json", "splits.json")
    print(f"results bundle: {zip_path}")


def shade_depth(depth: np.ndarray) -> np.ndarray:
    """Lambertian shading of a depth map (light from upper-left) so 3D shape is visible; NaN stays NaN."""
    face = np.isfinite(depth)
    z = np.where(face, depth, np.nanmedian(depth))
    z = (z - np.percentile(z[face], 1)) / (np.ptp(np.percentile(z[face], [1, 99])) + 1e-9)
    z = cv2.GaussianBlur(z.astype(np.float32), (0, 0), 1.5)
    gy, gx = np.gradient(z * 0.15 * max(depth.shape))
    n = np.dstack([-gx, -gy, np.ones_like(z)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    light = np.array([-0.4, -0.5, 0.77]) / np.linalg.norm([-0.4, -0.5, 0.77])
    out = np.clip(n @ light, 0, 1)
    out[~face] = np.nan
    return out


def _depth_examples(pairs, lmk, f3d, out: Path) -> None:
    """Photo | 3D depth for a few pairs. Contains faces: written to private/ only."""
    import matplotlib.pyplot as plt
    for p in pairs:
        fig, axs = plt.subplots(1, 4, figsize=(12, 3.4))
        for k, (role, path) in enumerate((("pre", p.pre), ("post", p.post))):
            bgr = read_bgr(path)
            obs = lmk(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            fit = f3d.fit(bgr, obs.lm_px)
            dm = f3d.depth_map(fit, bgr.shape[:2])
            axs[2 * k].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            axs[2 * k + 1].imshow(shade_depth(dm), cmap="Greys_r", vmin=0, vmax=1)
            axs[2 * k].set_title(f"{p.pid} {role}", fontsize=9)
            axs[2 * k + 1].set_title(f"3DMM depth (yaw {fit.yaw:+.0f}°)", fontsize=9)
        for a in axs:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(out / f"fig_depth_{p.pid}.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N pairs (smoke test)")
    main(ap.parse_args().limit)
