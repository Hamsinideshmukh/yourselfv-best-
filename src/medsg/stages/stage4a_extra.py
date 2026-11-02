"""Stage 4a: identity yardstick + extra GPU data for the latent <-> anatomy maps (one Kaggle run).

A) Identity yardstick (all 662 pairs). FaceNet cosine of
     real pre vs real post      (same person, different photo, surgery in between),
     rebuilt pre vs rebuilt post (does e4e keep that relationship?),
     every pre-op vs every other pair's pre-op (different people).
   This is the scale for Stage 2's cos(x, G(E(x))) = 0.876. The highest different-ID scores also flag
   possible repeated subjects (the HDA paper reports 638 subjects for 662 pairs).
B) HDA-like synthetic bases. Stage 3 synthetic faces smile and turn far more than HDA photos
   (smile 0.55 vs 0.10, yaw SD 10.8 vs 1.6 deg), so bases are kept only if |yaw| <= MAX_YAW and
   mean smile <= MAX_SMILE (HDA: mean smile 0.10, SD 0.19; yaw SD 1.6 deg).
C) Synthetic noise floor: N_NOISE bases x the 7 Stage-1 perturbations -> SEM per metric on G outputs.
D) Linearity set: N_LIN bases x 3 layer groups x one random direction at several signed sizes,
   so we can check whether a metric changes in proportion to the step (straight line) or not.
E) Real-code local set: e4e codes of all dev pre-op photos x 30 W+ perturbations
   (10 per layer group, sigma cycling 0.25 / 0.5 / 1.0 x std(w)) + the unperturbed rebuild.

Outputs (outputs/stage4a/): identity_pairs.csv, identity_summary.csv, identity_top_impostors.csv,
bases.npz, noise_synth.csv, linearity.npz, real_local.npz (measurements + deltas only);
private/: HDA embeddings and e4e codes (never commit).
"""
from __future__ import annotations

import json
import time
import zipfile

import cv2
import numpy as np
import pandas as pd
import torch

from ..align import align_ffhq
from ..config import get_config
from ..data import list_pairs, make_split, seed_everything
from ..gan import from_uint8, to_uint8
from ..landmarks import FaceNotFound
from ..runmeta import write_run_meta
from ..stats import mdc95, sem_within
from .stage1_measure import PERTURBATIONS, measure, perturb, read_bgr
from .stage2_invert import load_models
from .stage3_synth import BATCH, GROUPS, MEAS_RES, METRICS, PSI, _measure_batch

MAX_YAW, MAX_SMILE = 5.0, 0.30          # pre-declared HDA-like filter (see B above)
N_NOISE, N_LIN, MAX_CANDIDATES = 200, 300, 12_000
LIN_SIGMAS = (-0.5, 0.25, 0.5, 1.0)
REAL_PER_GROUP, REAL_SIGMAS = 10, (0.25, 0.5, 1.0)
_DEVICE = "cuda"                       # overridden only by the CPU smoke test
SMILE_COLS = (1, 2)                    # indices of mouthSmileLeft/Right in stage-1 BLENDSHAPES


def identity_pass(pairs, split, gan, lmk, idnet, device) -> tuple[pd.DataFrame, dict, dict]:
    """FaceNet embeddings of aligned real photos and their e4e rebuilds; returns per-pair table, embeddings, codes."""
    emb, codes = {}, {}
    for i, p in enumerate(pairs):
        for role, path in (("pre", p.pre), ("post", p.post)):
            rgb = cv2.cvtColor(read_bgr(path), cv2.COLOR_BGR2RGB)
            aligned, _ = align_ffhq(rgb, lmk(rgb).lm_px, 1024)
            x = from_uint8(aligned, 1024, device)
            code = gan.encode(torch.nn.functional.interpolate(x, size=(256, 256), mode="area"))
            with torch.no_grad():
                rec = gan.synth(code)
                emb[(p.pid, role, "real")] = idnet.embed(x)[0].cpu().numpy()
                emb[(p.pid, role, "rec")] = idnet.embed(rec)[0].cpu().numpy()
            codes[(p.pid, role)] = code[0].cpu().numpy()
        if i % 100 == 0:
            print(f"  identity {i}/{len(pairs)}", flush=True)
    rows = []
    for p in pairs:
        e = {k: emb[(p.pid, *k)] for k in [("pre", "real"), ("post", "real"), ("pre", "rec"), ("post", "rec")]}
        rows.append({"pid": p.pid, "split": "dev" if p.pid in split["dev"] else "test",
                     "cos_real_pre_post": float(e["pre", "real"] @ e["post", "real"]),
                     "cos_rec_pre_post": float(e["pre", "rec"] @ e["post", "rec"]),
                     "cos_pre_real_rec": float(e["pre", "real"] @ e["pre", "rec"]),
                     "cos_post_real_rec": float(e["post", "real"] @ e["post", "rec"])})
    return pd.DataFrame(rows), emb, codes


def identity_summary(df: pd.DataFrame, emb: dict, pids: list[str], out) -> None:
    """Genuine vs different-person distributions; top different-ID similarities (possible repeat subjects)."""
    E = np.stack([emb[(p, "pre", "real")] for p in pids])
    Epost = np.stack([emb[(p, "post", "real")] for p in pids])
    S = np.maximum(E @ E.T, E @ Epost.T)                      # best of pre-pre and pre-post across IDs
    iu = np.triu_indices(len(pids), 1)
    imp = (E @ E.T)[iu]
    q = [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    rows = [{"quantity": c, "n": len(df), "median": df[c].median(),
             **{f"p{int(100 * k):02d}": df[c].quantile(k) for k in q}} for c in df.columns if c.startswith("cos_")]
    rows.append({"quantity": "cos_different_person_pre_pre", "n": len(imp), "median": float(np.median(imp)),
                 **{f"p{int(100 * k):02d}": float(np.quantile(imp, k)) for k in q}})
    pd.DataFrame(rows).to_csv(out / "identity_summary.csv", index=False)
    S2 = np.where(np.eye(len(pids), dtype=bool), -1, np.maximum(S, S.T))
    order = np.argsort(-S2[iu])[:40]
    pd.DataFrame({"pid_a": [pids[iu[0][k]] for k in order], "pid_b": [pids[iu[1][k]] for k in order],
                  "max_cos": S2[iu][order]}).to_csv(out / "identity_top_impostors.csv", index=False)


def render_measure(gan, wplus: torch.Tensor, lmk, f3d, keep_images: bool = False):
    """Synthesise (n, L, 512) W+ codes in batches and measure them (Stage-3 measurement at MEAS_RES)."""
    assert wplus.ndim == 3 and wplus.shape[1:] == (gan.n_layers, 512), wplus.shape
    parts, imgs = [], []
    for b in range(0, len(wplus), BATCH):
        with torch.no_grad():
            im = to_uint8(gan.synth(wplus[b:b + BATCH].to(gan.device)))
        parts.append(_measure_batch(im, lmk, f3d))
        if keep_images:
            imgs += [cv2.resize(x, (MEAS_RES, MEAS_RES), interpolation=cv2.INTER_AREA) for x in im]
    res = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return (res, imgs) if keep_images else res


def select_bases(gan, lmk, f3d, n_needed: int, seed: int):
    """Sample synthetic W until n_needed faces pass the HDA-like filter. Returns (w (n,512), measurements, images)."""
    gen = torch.Generator().manual_seed(seed)
    kept_w, kept_res, kept_img, n_seen = [], [], [], 0
    while sum(len(w) for w in kept_w) < n_needed:
        if n_seen >= MAX_CANDIDATES:
            raise RuntimeError(f"only {sum(len(w) for w in kept_w)} HDA-like faces in {n_seen} candidates")
        w = gan.sample_w(200, PSI, gen)
        res, imgs = render_measure(gan, w.unsqueeze(1).repeat(1, gan.n_layers, 1), lmk, f3d, keep_images=True)
        smile = res["bs"][:, list(SMILE_COLS)].mean(1)
        keep = res["ok"] & (np.abs(res["pose"][:, 0]) <= MAX_YAW) & (smile <= MAX_SMILE)
        kept_w.append(w.cpu().numpy()[keep])
        kept_res.append({k: v[keep] for k, v in res.items()})
        kept_img += [im for im, k in zip(imgs, keep) if k]
        n_seen += len(w)
        print(f"  bases: {sum(len(x) for x in kept_w)}/{n_needed} kept of {n_seen} candidates", flush=True)
    W = np.concatenate(kept_w)[:n_needed]
    R = {k: np.concatenate([r[k] for r in kept_res])[:n_needed] for k in kept_res[0]}
    return W, R, kept_img[:n_needed], n_seen


def noise_floor(imgs, lmk, f3d) -> pd.DataFrame:
    """SEM / MDC95 per metric from the Stage-1 perturbations applied to synthetic 512 px faces."""
    vals = []
    for rgb in imgs:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        try:
            vals.append([[measure(perturb(bgr, k), lmk, f3d)[0][m] for m in METRICS] for k in PERTURBATIONS])
        except (FaceNotFound, ValueError):                 # counted: the run fails if > 10% are lost
            continue
    v = np.asarray(vals)                                   # (n_faces, n_perturb, n_metrics)
    if len(v) < 0.9 * len(imgs):
        raise RuntimeError(f"noise floor: only {len(v)}/{len(imgs)} faces measured under all perturbations")
    sem = [sem_within(v[:, :, j]) for j in range(len(METRICS))]
    return pd.DataFrame({"metric": METRICS, "sem": sem, "mdc95": [mdc95(s) for s in sem], "n_images": len(v)})


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 4a needs a GPU (Kaggle: Settings -> Accelerator -> GPU T4)")
    device = _DEVICE
    cfg = get_config()
    seed_everything(cfg.seed)
    out = cfg.out_dir / "stage4a"
    (out / "private").mkdir(parents=True, exist_ok=True)
    pairs = list_pairs(cfg)
    split = make_split(pairs, cfg, cfg.out_dir / "splits.json")
    gan, lmk, f3d, idnet, _ = load_models(cfg, device)
    L = gan.n_layers
    t0 = time.time()

    # A) identity yardstick (+ e4e codes for E)
    codes_path = out / "private" / "e4e_codes.npz"
    if not (out / "identity_pairs.csv").exists() or not codes_path.exists():
        df, emb, codes = identity_pass(pairs, split, gan, lmk, idnet, device)
        df.to_csv(out / "identity_pairs.csv", index=False)
        identity_summary(df, emb, [p.pid for p in pairs], out)
        np.savez_compressed(out / "private" / "facenet_embeddings.npz", **{"_".join(k): v for k, v in emb.items()})
        np.savez_compressed(codes_path, **{f"{pid}_{r}": c.astype(np.float32) for (pid, r), c in codes.items()})
        print(pd.read_csv(out / "identity_summary.csv").round(3).to_string(index=False), flush=True)
    codes = dict(np.load(codes_path))
    print(f"== A identity done {time.time() - t0:.0f}s", flush=True)

    # B-D) HDA-like synthetic bases, noise floor, linearity
    w_std = gan.sample_w(10_000, 1.0, torch.Generator().manual_seed(cfg.seed + 1)).std(0).cpu().numpy()
    if not (out / "linearity.npz").exists():
        W, R, imgs, n_seen = select_bases(gan, lmk, f3d, N_NOISE + N_LIN, cfg.seed + 10)
        np.savez_compressed(out / "bases.npz", w=W, n_candidates=n_seen, **R)
        nf = noise_floor(imgs[:N_NOISE], lmk, f3d)
        nf.to_csv(out / "noise_synth.csv", index=False)
        print(nf.round(4).to_string(index=False), flush=True)
        print(f"== C noise floor done {time.time() - t0:.0f}s", flush=True)
        rng = np.random.default_rng(cfg.seed + 11)
        lin_base = np.arange(N_NOISE, N_NOISE + N_LIN)
        recs = []
        for bi in lin_base:
            for g, (a, b) in GROUPS.items():
                eps = (w_std * rng.standard_normal(512)).astype(np.float32)
                for s in LIN_SIGMAS:
                    recs.append((bi, g, s, eps))
        wp = torch.from_numpy(W[[r[0] for r in recs]]).unsqueeze(1).repeat(1, L, 1)
        for i, (_, g, s, eps) in enumerate(recs):
            a, b = GROUPS[g]
            wp[i, a:b] += torch.from_numpy(s * eps)
        res = render_measure(gan, wp, lmk, f3d)
        np.savez_compressed(out / "linearity.npz", base=np.array([r[0] for r in recs]),
                            group=np.array([r[1] for r in recs]), sigma=np.array([r[2] for r in recs], np.float32),
                            eps=np.stack([r[3] for r in recs]), **res)
        print(f"== D linearity done ({len(recs)} renders, ok {res['ok'].mean():.1%}) {time.time() - t0:.0f}s", flush=True)

    # E) real-code local set on dev pre-op codes
    if not (out / "real_local.npz").exists():
        dev = [pid for pid in split["dev"] if f"{pid}_pre" in codes]
        base_codes = np.stack([codes[f"{pid}_pre"] for pid in dev])          # (n, L, 512)
        assert base_codes.shape[1:] == (L, 512), base_codes.shape
        rng = np.random.default_rng(cfg.seed + 12)
        rows, wp = [], []
        for k, pid in enumerate(dev):
            wp.append(base_codes[k]); rows.append((k, -1, 0.0, np.zeros(512, np.float32)))
            for g, (a, b) in GROUPS.items():
                for j in range(REAL_PER_GROUP):
                    s = REAL_SIGMAS[j % len(REAL_SIGMAS)]
                    d = (s * w_std * rng.standard_normal(512)).astype(np.float32)
                    c = base_codes[k].copy()
                    c[a:b] += d
                    wp.append(c); rows.append((k, g, s, d))
        res = render_measure(gan, torch.from_numpy(np.stack(wp)), lmk, f3d)
        np.savez_compressed(out / "real_local.npz", base=np.array([r[0] for r in rows]),
                            pid=np.array([dev[r[0]] for r in rows]), group=np.array([r[1] for r in rows]),
                            sigma=np.array([r[2] for r in rows], np.float32),
                            delta=np.stack([r[3] for r in rows]), **{k: v for k, v in res.items() if k != "lm"})
        print(f"== E real-code local done ({len(rows)} renders, ok {res['ok'].mean():.1%}) {time.time() - t0:.0f}s",
              flush=True)
    json.dump({"MAX_YAW": MAX_YAW, "MAX_SMILE": MAX_SMILE, "N_NOISE": N_NOISE, "N_LIN": N_LIN,
               "LIN_SIGMAS": LIN_SIGMAS, "REAL_PER_GROUP": REAL_PER_GROUP, "REAL_SIGMAS": REAL_SIGMAS,
               "PSI": PSI, "GROUPS": GROUPS, "METRICS": METRICS, "seed": cfg.seed},
              open(out / "meta.json", "w"), indent=1)
    lmk.close()
    write_run_meta(out, cfg.as_dict(), {"e4e_ffhq_encode.pt": cfg.models_dir / "e4e_ffhq_encode.pt",
                                        "face_landmarker.task": cfg.models_dir / "face_landmarker.task"},
                   {"stage": "stage4a_extra", "seconds": time.time() - t0})
    _zip([*out.glob("*.csv"), *out.glob("*.json"), *out.glob("*.npz"),
          *cfg.out_dir.glob("splits.json")],
         cfg.out_dir, cfg.out_dir / "stage4a_results.zip")
    _zip(list((out / "private").glob("*")), cfg.out_dir, cfg.out_dir / "stage4a_private_hda.zip")


def _zip(paths, root, dst) -> None:
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as z:
        for f in paths:
            z.write(f, f.relative_to(root))
    print(f"wrote {dst} ({dst.stat().st_size / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
