"""Stage 8 (last GPU run): consensus steering — an edit counts only when MediaPipe AND FAN both see it — judged by depth.

Question. GAN nose and eye-opening edits pass MediaPipe but are not confirmed by FAN (Stage 5) and, for eye opening,
not by the 3D face model either (Stage 7: right direction 47 %). Are they too small for independent detectors, or
wrong in kind? Pre-registered in outputs/eval_plan.json "stage8" before the run.

Arms, on HyperStyle inversions of every included TEST pre-op photo (the GAN-mode inversion since Stage 7):
  A  MediaPipe stopping  (= the E3 editor): stop when the MediaPipe change reaches the request (+-2 MDC);
  B  consensus stopping, same direction: steer on the smaller of the two changes, each in its own noise-floor
     units, min(z_MediaPipe, z_FAN) — both detectors must see the request;
  C  (eye opening only) consensus stopping with the fine-layer (8-17) direction (assets/candidates_v1.npz),
     i.e. steering may also choose HOW to move, not only how far;
  REF brow height, arm A: an edit already confirmed by FAN, as the benchmark for "what a real edit looks like".
Judges (neither arm steers on them): 3DDFA-V2 canonical shape (pose and expression removed), Depth Anything V2
locality, ArcFace vs the real photo; MediaPipe-protected leakage (E3 criterion ii). FAN is a judge for arm A only.
"Further vs better": 3D change per unit of latent movement is reported, so a larger edit is not mistaken for a
more anatomical one. Also: the app's HyperStyle GAN path timed on synthetic faces.
Outputs: outputs/stage8/*.csv|json; private/* (HDA-derived). Zips: stage8_results.zip, stage8_private_hda.zip.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..config import get_config
from ..data import list_pairs, make_split, seed_everything
from ..editor import Directions, secant_edit
from ..judges import FAN_METRICS, face_box, metrics_fan
from ..landmarks import FaceNotFound
from ..measure import MCOLS, MEAS_RES, _shift, measure, measure_avg
from ..runmeta import write_run_meta
from . import stage5_overnight as S5
from . import stage7_final as S7
from .stage1_measure import perturb
from .stage3_synth import GROUPS

DEVICE = "cuda"
# Noise floors (MDC95) of the two detectors, each in its own units. MediaPipe: editor asset (Stage 1). FAN: Stage 7
# K7, 7 re-captures x 40 aligned 512 px test photos (outputs/stage7/k7_judge_noise.csv).
FAN_MDC = {"nose_width": 0.0360, "palpebral_height": 0.0239, "brow_height": 0.0261}
SIGN = {"nose_width": -1, "palpebral_height": 1, "brow_height": 1}        # clinical request directions (E3)
REQ_MDC = 2.0
ASSET_C = Path(__file__).resolve().parents[1] / "assets" / "candidates_v1.npz"
N_NOISE = 40


def fan_avg(kit, img: np.ndarray, L: np.ndarray) -> dict | None:
    """FAN metrics averaged over the same 4 harmless variants as measure_avg (original, mirror, +-1 % shift)."""
    W = img.shape[1]
    x1, y1, x2, y2 = face_box(L)
    variants = [(img, [x1, y1, x2, y2]), (cv2.flip(img, 1), [W - 1 - x2, y1, W - 1 - x1, y2]),
                (_shift(img, 0.01), [x1 + 0.01 * W, y1, x2 + 0.01 * W, y2]),
                (_shift(img, -0.01), [x1 - 0.01 * W, y1, x2 - 0.01 * W, y2])]
    vals = []
    for v, box in variants:
        try:
            vals.append(metrics_fan(kit.fan.landmarks(np.ascontiguousarray(v), np.array(box))))
        except ValueError:
            pass
    if len(vals) < 3:
        return None
    return {k: float(np.mean([x[k] for x in vals])) for k in FAN_METRICS}


def run_arm(kit, rend, w0, D, a_cap, t, steer: str, base: dict, mdc_mp: float):
    """Closed loop on one face; steer = "mp" or "consensus". Returns (final record, final image).
    Units: directions give +1 metric unit (IPD) per unit amplitude, so the loop works in MediaPipe metric units:
    request = 2 MDC_MediaPipe, tolerance 0.5 MDC (as E3); the consensus value is converted back the same way."""
    r, tol, ti, cache = SIGN[t] * REQ_MDC * mdc_mp, 0.5 * mdc_mp, MCOLS.index(t), {}

    def render_measure(a: float):
        img = S5.render(rend, (w0 + a * D)[None])[0]
        m, first = measure_avg(img, kit.lmk)
        if m is None:
            return None
        z_mp = (m[ti] - base["m"][ti]) / mdc_mp
        fa = fan_avg(kit, img, first["lm"] * MEAS_RES)                 # always measured (arm A: as a judge)
        z_fan = (fa[t] - base["fan"][t]) / FAN_MDC[t] if fa is not None else np.nan
        cache[a] = (img, m, first, z_mp, z_fan)
        if steer == "mp":
            return z_mp * mdc_mp
        if not np.isfinite(z_fan):
            return None
        return SIGN[t] * min(SIGN[t] * z_mp, SIGN[t] * z_fan) * mdc_mp     # the less convinced detector

    recs, status = secant_edit(render_measure, r, tol, a_cap)
    if not recs:
        return {"status": status, "renders": 0}, None
    a = recs[-1][0]
    img, m, first, z_mp, z_fan = cache[a]
    return {"status": status, "renders": len(recs), "a": a, "z_mp": z_mp, "z_fan": z_fan, "steered": recs[-1][1],
            **{f"d_{k}": v for k, v in zip(MCOLS, m - base["m"])},
            **{f"d_{k}": v for k, v in zip(("yaw", "pitch", "roll"), first["pose"] - base["first"]["pose"])},
            **{f"d_bs_{b}": v for b, v in zip(S5.BLENDSHAPES, first["bs"] - base["first"]["bs"])}}, (img, first)


def s1_arms(cfg, kit, pairs, split, out: Path, limit) -> int:
    from ..depth import depth_change
    hs = S7.load_hyperstyle(cfg)
    f3d, dnet, notes = S7.depth_judges(cfg)
    (out / "s1_notes.json").write_text(json.dumps(notes or "all depth judges available"))
    if f3d is None or dnet is None:
        raise RuntimeError(f"Stage 8 needs both depth judges: {notes}")
    dirs, cand = Directions(), np.load(ASSET_C)
    Dc = np.zeros((18, 512), np.float32)
    a, b = GROUPS[int(cand["groups"][0])]
    Dc[a:b] = cand["w_delta"][0]
    a_cap_c = 23.0 / float(np.linalg.norm(cand["w_delta"][0] / np.load(S5.ASSET)["w_std"]))
    arms = [("nose_width", "A", "mp", dirs.D["nose_width"], dirs.a_cap["nose_width"]),
            ("nose_width", "B", "consensus", dirs.D["nose_width"], dirs.a_cap["nose_width"]),
            ("palpebral_height", "A", "mp", dirs.D["palpebral_height"], dirs.a_cap["palpebral_height"]),
            ("palpebral_height", "B", "consensus", dirs.D["palpebral_height"], dirs.a_cap["palpebral_height"]),
            ("palpebral_height", "C", "consensus", Dc, a_cap_c),
            ("brow_height", "REF", "mp", dirs.D["brow_height"], dirs.a_cap["brow_height"])]
    w_std = np.load(S5.ASSET)["w_std"].astype(np.float64)
    rows, noise = [], []
    faces = S7.test_faces(pairs, split)[:limit]
    for i, p in enumerate(faces):
        try:
            al = S7.aligned_photo(kit, p.pre)
        except (FaceNotFound, IOError):
            continue
        real = S7.small(al, MEAS_RES)
        mr = measure(real, kit.lmk)
        if mr is None:
            continue
        e_real = S5.arc_embed(kit, real, mr["lm"])
        x = F.interpolate(torch.from_numpy(al).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(DEVICE),
                          size=(256, 256), mode="area")
        code, deltas = hs.invert(x)
        rend, w0 = hs.for_face(deltas), code[0].cpu().numpy().astype(np.float32)
        img0 = S5.render(rend, w0[None])[0]
        m0, f0 = measure_avg(img0, kit.lmk)
        if m0 is None:
            continue
        L0 = f0["lm"] * MEAS_RES
        fa0 = fan_avg(kit, img0, L0)
        if fa0 is None:
            continue
        base = {"m": m0, "first": f0, "fan": fa0}
        s0, d0 = S7.shape3d(f3d, img0, L0), dnet(img0)
        e0 = S5.arc_embed(kit, img0, f0["lm"])
        if i < N_NOISE:                                  # judges' noise on this kind of image (HyperStyle rebuilds)
            for kind in ("jpeg70", "gamma", "downscale"):
                v = cv2.cvtColor(perturb(cv2.cvtColor(img0, cv2.COLOR_RGB2BGR), kind), cv2.COLOR_BGR2RGB)
                v = np.ascontiguousarray(S7.small(v, MEAS_RES))
                mv = measure(v, kit.lmk)
                if mv is None:
                    continue
                rec = {"pid": p.pid, "kind": kind, **{f"{k}": val for k, val in S7.shape3d(f3d, v, mv["lm"] * MEAS_RES).items()}}
                dv = dnet(v)
                for t in SIGN:
                    rec.update({f"{t}_{k}": val for k, val in depth_change(d0, dv, L0, t).items()})
                noise.append(rec)
        for t, arm, steer, D, a_cap in arms:
            t0 = time.time()
            rec, res = run_arm(kit, rend, w0, D, a_cap, t, steer, base, dirs.mdc[t])
            rec.update({"pid": p.pid, "target": t, "arm": arm, "steer": steer, "seconds": time.time() - t0,
                        # whitened step size per edited layer (the unit of U_CAP = 23, the Stage 4 sampling range)
                        "latent_norm": float(abs(rec.get("a", np.nan)) * np.linalg.norm(D[np.abs(D).sum(1) > 0][0] / w_std))})
            if res is not None:
                img, first = res
                s1 = S7.shape3d(f3d, img, first["lm"] * MEAS_RES)
                rec.update({f"{q}_d": s1[q] - s0[q] for q in s1 if q in s0})
                rec.update(depth_change(d0, dnet(img), L0, t))
                e = S5.arc_embed(kit, img, first["lm"])
                rec.update(arc_edit_real=float(e @ e_real), arc_edit_base=float(e @ e0),
                           arc_base_real=float(e0 @ e_real))
            rows.append(rec)
        if i % 20 == 0:
            print(f"  S1 {i}/{len(faces)}", flush=True)
            pd.DataFrame(rows).to_csv(out / "s1_rows_partial.csv", index=False)
    pd.DataFrame(noise).to_csv(out / "s1_judge_noise_rows.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "s1_arms.csv", index=False)
    return len(rows)


def s2_app(cfg, kit, out: Path, limit) -> int:
    """App GAN path (HyperStyle; brow and mouth; MediaPipe stopping) on synthetic faces: does it run, how fast."""
    from ..align import align_ffhq
    hs, dirs = S7.load_hyperstyle(cfg), Directions()
    gen = torch.Generator().manual_seed(cfg.seed + 8)
    cands = []
    for _ in range(5 if limit else 6):
        w = kit.gan.sample_w(8, 0.7, gen)
        codes = w[:, None].repeat(1, kit.gan.n_layers, 1).cpu().numpy()
        for c, img in zip(codes, S5.render(kit.gan, codes)):
            m = measure(img, kit.lmk)
            if m is not None:
                bs = dict(zip(S5.BLENDSHAPES, m["bs"]))
                cands.append((abs(m["pose"][0]) + abs(m["pose"][1]) + 20 * (bs["mouthSmileLeft"] + bs["mouthSmileRight"]), c))
    cands.sort(key=lambda q: q[0])
    rows = []
    for j, (_, c) in enumerate(cands[:3]):
        with torch.no_grad():
            img = S5.to_uint8(kit.gan.synth(torch.from_numpy(c[None]).to(DEVICE)))[0]
        t0 = time.time()
        al = align_ffhq(img, kit.lmk(img).lm_px, 1024)[0]
        x = F.interpolate(torch.from_numpy(al).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(DEVICE),
                          size=(256, 256), mode="area")
        code, deltas = hs.invert(x)
        rend, w0 = hs.for_face(deltas), code[0].cpu().numpy()
        t_inv = time.time() - t0
        real = S7.small(al, MEAS_RES)
        mr = measure(real, kit.lmk)
        er = S5.arc_embed(kit, real, mr["lm"])
        m0, _ = measure_avg(S5.render(rend, w0[None])[0], kit.lmk)
        for t in ("brow_height", "mouth_width"):
            t1, cache = time.time(), {}

            def rm(a: float):
                cache[a] = S5.render(rend, (w0 + a * dirs.D[t])[None])[0]
                m, _ = measure_avg(cache[a], kit.lmk)
                return None if m is None else float((m - m0)[MCOLS.index(t)])
            recs, status = secant_edit(rm, 2 * dirs.mdc[t], 0.5 * dirs.mdc[t], dirs.a_cap[t])
            fin = cache[recs[-1][0]]
            mf = measure(fin, kit.lmk)
            rows.append({"face": j + 1, "target": t, "seconds_inversion": t_inv, "seconds_edit": time.time() - t1,
                         "renders": len(recs), "status": status,
                         "arc_edit_vs_input": float(S5.arc_embed(kit, fin, mf["lm"]) @ er) if mf else np.nan})
    pd.DataFrame(rows).to_csv(out / "s2_app_hyperstyle.csv", index=False)
    return len(rows)


DONE = {"s1": "s1_arms.csv", "s2": "s2_app_hyperstyle.csv"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(DONE))
    ap.add_argument("--limit", type=int, default=None, help="smoke test only")
    a = ap.parse_args(argv)
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Stage 8 needs a GPU (Kaggle: Settings -> Accelerator -> GPU T4)")
    t_start = time.time()
    cfg = get_config()
    seed_everything(cfg.seed)
    out = cfg.out_dir / "stage8"
    (out / "private").mkdir(parents=True, exist_ok=True)
    pairs = list_pairs(cfg)
    split = make_split(pairs, cfg, cfg.out_dir / "splits.json")
    kit = S5.load_kit(cfg, DEVICE)
    if kit.arc is None or kit.fan is None:
        raise RuntimeError(f"Stage 8 needs ArcFace and FAN: {kit.notes}")
    status = {"judges": kit.notes or "all available"}
    steps = {"s1": lambda: s1_arms(cfg, kit, pairs, split, out, a.limit), "s2": lambda: s2_app(cfg, kit, out, a.limit)}
    for name in [s for s in DONE if s in a.only]:
        if (out / DONE[name]).exists():
            continue
        t0 = time.time()
        print(f"== {name} start", flush=True)
        try:
            status[name] = {"status": "done", "n": steps[name](), "seconds": round(time.time() - t0)}
        except Exception as e:                    # noqa: BLE001  recorded, never replaced
            status[name] = {"status": "failed", "error": repr(e), "traceback": traceback.format_exc()[-3000:]}
            print(status[name]["traceback"], flush=True)
        (out / "status.json").write_text(json.dumps(status, indent=1, default=str))
        S5._zip([*out.glob("*.csv"), *out.glob("*.json")], cfg.out_dir, cfg.out_dir / "stage8_results.zip")
    kit.lmk.close()
    write_run_meta(out, cfg.as_dict(), {k: cfg.models_dir / k for k in ("hyperstyle_ffhq.pt", "faces_w_encoder.pt",
                                                                          "w600k_r50.onnx") if (cfg.models_dir / k).exists()},
                   {"stage": "stage8_consensus", "seconds": time.time() - t_start, "status": status})
    S5._zip([*out.glob("*.csv"), *out.glob("*.json")], cfg.out_dir, cfg.out_dir / "stage8_results.zip")
    failed = [k for k, v in status.items() if isinstance(v, dict) and v.get("status") == "failed"]
    if failed:
        raise SystemExit(f"components failed: {failed} (see status.json)")


if __name__ == "__main__":
    main()
