"""Stage 5 (overnight GPU run): independent judges, measurement calibration, per-face latent data, the editor.

One Kaggle T4 session, five resumable components (each skipped if its outputs exist; a failed component is
recorded in status.json and never replaced by substitute output):

C1 inversion + independent identity. Every HDA photo: MediaPipe -> FFHQ align -> e4e -> G. MediaPipe + FAN
   landmarks and ArcFace (buffalo_l) embeddings of the aligned photo and its rebuild, LPIPS. Gives the ArcFace
   identity yardstick (D8), FAN-measured inversion drift (D7), and the e4e codes for C3-C5.
C2 measurement calibration (A2). Known-deformation phantoms on 40 dev pre-op photos (8 metrics x 3 widths x
   +-0.02 IPD), measured by MediaPipe AND FAN: is the under-reporting (F6) MediaPipe-specific?
C3 per-face latent perturbations (P5, E7). For every included pre-op code: antithetic pairs w +- 0.5*std(w)*eps
   in one layer group (middle 16, coarse 8, fine 8), MediaPipe landmarks of every render. Gives each face's own
   GAN deformation pattern (personalised prior, E7) and antithetic dev data for better Jacobians.
C4 editor (Stage 5 / E3, GAN-rendered mode). Frozen Stage-4b directions (assets/editor_v1.npz, fitted on
   synthetic + DEV codes only). Per face and request: closed loop (secant on the measured target, k = 4 averaged
   measurements, <= 5 renders), its iteration 0 = open loop, and B4 (global ridge, open loop). Requests:
   clinically signed 2 x MDC (nose narrower, brow up, eye opening up, jaw narrower, mouth wider) for dev-100 and
   all test faces; dev-mean E3 requests on dev faces only. Judges on the final edit: ArcFace, FAN, LPIPS.
C5 PTI final renders (time permitting): 60 test faces, PTI-tuned G, rebuild + the five closed-loop clinical edits.

Excluded pairs (duplicates / dev-test same subject, medsg.dedup) are skipped everywhere except C1 (measured,
flagged). Test-split results are computed blind; they are analysed only after the dev analysis and the frozen
evaluation plan.
Outputs: outputs/stage5/*.csv|json (tables), outputs/stage5/private/* (HDA-derived codes, landmarks,
embeddings, images: never commit). Zips: outputs/stage5_results.zip, outputs/stage5_private_hda.zip.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .. import editspec as E
from ..align import align_ffhq
from ..config import get_config
from ..data import list_pairs, make_split, seed_everything
from ..dedup import EXCLUDE
from ..deform import PHANTOM_SPECS, moved_points, phantom_controls, warp_gaussian
from ..gan import from_uint8, to_uint8
from ..judges import FAN_METRICS, face_box, metrics_fan
from ..landmarks import FaceNotFound
from ..metrics2d import compute_clinical_2d, compute_metrics_2d
from ..runmeta import write_run_meta
from .stage1_measure import BLENDSHAPES, read_bgr
from .stage3_synth import GROUPS

from ..measure import C5, M10, MCOLS, MEAS_RES, _shift, measure, measure_avg  # noqa: E402,F401 (re-exported)
from ..measure import BLENDSHAPES as _BS_MEASURE  # noqa: E402
assert _BS_MEASURE == BLENDSHAPES, "measure.BLENDSHAPES must match the Stage 1 order"
BATCH = 8
SIGMA, GROUP_K = 0.5, {1: 24, 0: 8, 2: 8}         # C3: antithetic pairs per layer group (middle, coarse, fine)
CLIN_REQ_MDC = {"nose_width": -2.0, "brow_height": 2.0, "palpebral_height": 2.0, "jaw_width": -2.0,
                "mouth_width": 2.0}               # C4: clinically signed requests, in MDC units
N_DEV_EDIT, N_PHANTOM, N_PTI, N_IMG_SAVE = 10_000, 40, 60, 24   # N_DEV_EDIT large = every included dev face
PH_SIGMAS, PH_AMPS = (0.1, 0.2, 0.35), (-0.02, 0.02)
U_CAP = 23.0                                      # max whitened step norm = 95th pct sampled in Stage 4 (no extrapolation)
DEVICE = "cuda"                                   # the CPU smoke test (tools/smoke_stage5.py) overrides this
ASSET = Path(__file__).resolve().parents[1] / "assets" / "editor_v1.npz"


# ----------------------------------------------------------------------------------------------- kit
class Kit:
    """Models used by the run. Optional judges are None when unavailable (recorded in status, never faked)."""

    def __init__(self, gan, lmk, lpips_fn, arc, fan, notes: dict):
        self.gan, self.lmk, self.lpips, self.arc, self.fan, self.notes = gan, lmk, lpips_fn, arc, fan, notes


def load_kit(cfg, device: str) -> Kit:
    from ..gan import E4EGAN, install_e4e_package
    from ..landmarks import Landmarker, ensure_task_file
    install_e4e_package(cfg.repos_dir / "encoder4editing", cfg.out_dir.parent / "src_ext")
    gan = E4EGAN(cfg.models_dir / "e4e_ffhq_encode.pt", device)
    lmk = Landmarker(ensure_task_file(cfg.models_dir))
    notes, lp, arc, fan = {}, None, None, None
    try:
        import lpips
        lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    except Exception as e:                        # noqa: BLE001  (recorded, not hidden)
        notes["lpips"] = f"unavailable: {e!r}"
    try:
        from ..judges import ArcFace
        arc = ArcFace(cfg.models_dir / "w600k_r50.onnx")
    except Exception as e:                        # noqa: BLE001
        notes["arcface"] = f"unavailable: {e!r}"
    try:
        from ..judges import FAN68
        fan = FAN68(device)
    except Exception as e:                        # noqa: BLE001
        notes["fan"] = f"unavailable: {e!r}"
    return Kit(gan, lmk, lp, arc, fan, notes)


# ---------------------------------------------------------------------------------------- measuring
def render(gan, codes: np.ndarray) -> np.ndarray:
    """(n, L, 512) float codes -> (n, 512, 512, 3) uint8 renders (synthesised at 1024, area-downsampled)."""
    out = []
    for b in range(0, len(codes), BATCH):
        with torch.no_grad():
            im = gan.synth(torch.from_numpy(np.ascontiguousarray(codes[b:b + BATCH], np.float32)).to(gan.device))
            im = F.interpolate(im, size=(MEAS_RES, MEAS_RES), mode="area")
        out.append(to_uint8(im))
    return np.concatenate(out)


def lpips256(kit: Kit, a: np.ndarray, b: np.ndarray) -> float:
    if kit.lpips is None:
        return np.nan
    t = [F.interpolate(from_uint8(x, MEAS_RES, kit.gan.device), size=(256, 256), mode="area") for x in (a, b)]
    with torch.no_grad():
        return float(kit.lpips(t[0], t[1]).item())


def arc_embed(kit: Kit, rgb512: np.ndarray, lm_norm: np.ndarray) -> np.ndarray | None:
    if kit.arc is None:
        return None
    return kit.arc.embed(rgb512, lm_norm.astype(np.float64) * MEAS_RES)


def fan_metrics(kit: Kit, rgb: np.ndarray, box: np.ndarray) -> tuple[dict, np.ndarray | None]:
    if kit.fan is None:
        return {k: np.nan for k in FAN_METRICS}, None
    try:
        p = kit.fan.landmarks(rgb, box)
        return metrics_fan(p), p
    except ValueError:
        return {k: np.nan for k in FAN_METRICS}, None


# ------------------------------------------------------------------------------------------ C1
def c1_inversion(cfg, kit: Kit, pairs, split, out: Path, limit: int | None) -> int:
    rows, codes, priv = [], {}, {}
    for i, p in enumerate(pairs[:limit]):
        for role, path in (("pre", p.pre), ("post", p.post)):
            r = {"pid": p.pid, "role": role, "split": "dev" if p.pid in split["dev"] else "test",
                 "excluded": p.pid in EXCLUDE, "ok": False, "error": ""}
            try:
                rgb = cv2.cvtColor(read_bgr(path), cv2.COLOR_BGR2RGB)
                aligned, _ = align_ffhq(rgb, kit.lmk(rgb).lm_px, 1024)
                x = from_uint8(aligned, 1024, kit.gan.device)
                code = kit.gan.encode(F.interpolate(x, size=(256, 256), mode="area"))
                rec = render(kit.gan, code.cpu().numpy())[0]
                real = cv2.resize(aligned, (MEAS_RES, MEAS_RES), interpolation=cv2.INTER_AREA)
                mr, mc = measure(real, kit.lmk), measure(rec, kit.lmk)
                if mr is None or mc is None:
                    raise FaceNotFound("no face on aligned photo or rebuild")
                er, ec = arc_embed(kit, real, mr["lm"]), arc_embed(kit, rec, mc["lm"])
                fr, lr = fan_metrics(kit, real, face_box(mr["lm"] * MEAS_RES))
                fc, lc = fan_metrics(kit, rec, face_box(mc["lm"] * MEAS_RES))
                r.update(ok=True, arc_cos_real_rec=float(er @ ec) if er is not None else np.nan,
                         lpips_real_rec=lpips256(kit, real, rec),
                         **{f"mp_real_{k}": v for k, v in zip(MCOLS, mr["m"])},
                         **{f"mp_rec_{k}": v for k, v in zip(MCOLS, mc["m"])},
                         **{f"fan_real_{k}": v for k, v in fr.items()}, **{f"fan_rec_{k}": v for k, v in fc.items()},
                         **{f"real_{k}": v for k, v in zip(("yaw", "pitch", "roll"), mr["pose"])})
                key = f"{p.pid}_{role}"
                codes[key] = code[0].cpu().numpy().astype(np.float32)
                priv[f"lm_real_{key}"], priv[f"lm_rec_{key}"] = mr["lm"], mc["lm"]
                if er is not None:
                    priv[f"arc_real_{key}"], priv[f"arc_rec_{key}"] = er.astype(np.float32), ec.astype(np.float32)
                if lr is not None:
                    priv[f"fan_real_{key}"] = lr.astype(np.float32)
                if lc is not None:
                    priv[f"fan_rec_{key}"] = lc.astype(np.float32)
            except (FaceNotFound, ValueError, IOError) as e:
                r["error"] = repr(e)
            rows.append(r)
        if i % 50 == 0:
            print(f"  C1 {i}/{len(pairs[:limit])}", flush=True)
    np.savez_compressed(out / "private" / "c1_codes.npz", **codes)
    np.savez_compressed(out / "private" / "c1_landmarks_embeddings.npz", **priv)
    pd.DataFrame(rows).to_csv(out / "c1_images.csv", index=False)      # written last = completion marker
    return len(rows)


# ------------------------------------------------------------------------------------------ C2
def c2_phantom(cfg, kit: Kit, pairs, split, out: Path, limit: int | None) -> int:
    rng = np.random.default_rng(cfg.seed + 50)
    dev = [p for p in pairs if p.pid in split["dev"] and p.pid not in EXCLUDE]
    sel = [dev[i] for i in rng.choice(len(dev), min(N_PHANTOM, len(dev)), replace=False)][:limit]
    rows = []
    for n, p in enumerate(sel):
        rgb = cv2.cvtColor(read_bgr(p.pre), cv2.COLOR_BGR2RGB)
        try:
            lm0 = kit.lmk(rgb).lm_px[:, :2].astype(np.float64)
        except FaceNotFound:
            continue
        box = face_box(lm0)
        f0 = kit.fan.landmarks(rgb, box) if kit.fan is not None else None
        base = {**compute_metrics_2d(lm0), **compute_clinical_2d(lm0)}
        fb = metrics_fan(f0) if f0 is not None else {}
        ipd = float(np.linalg.norm(lm0[468] - lm0[473]))
        for met in PHANTOM_SPECS:
            for s in PH_SIGMAS:
                for amp in PH_AMPS:
                    ctrl, vecs = phantom_controls(lm0, met, amp)
                    img = warp_gaussian(rgb, ctrl, vecs, s * ipd)
                    lt = moved_points(lm0, ctrl, vecs, s * ipd)
                    true_mp = {**compute_metrics_2d(lt), **compute_clinical_2d(lt)}[met] - base[met]
                    try:
                        l1 = kit.lmk(img).lm_px[:, :2]
                        meas_mp = {**compute_metrics_2d(l1), **compute_clinical_2d(l1)}[met] - base[met]
                    except (FaceNotFound, ValueError):
                        meas_mp = np.nan
                    true_fan = meas_fan = np.nan
                    if f0 is not None and met in FAN_METRICS:
                        true_fan = metrics_fan(moved_points(f0, ctrl, vecs, s * ipd))[met] - fb[met]
                        try:
                            meas_fan = metrics_fan(kit.fan.landmarks(img, box))[met] - fb[met]
                        except ValueError:
                            pass
                    rows.append({"pid": p.pid, "metric": met, "sigma_ipd": s, "amp_ipd": amp, "true_mp": true_mp,
                                 "meas_mp": meas_mp, "true_fan": true_fan, "meas_fan": meas_fan})
        print(f"  C2 {n + 1}/{len(sel)}", flush=True)
    pd.DataFrame(rows).to_csv(out / "c2_phantom.csv", index=False)
    return len(rows)


# ------------------------------------------------------------------------------------------ C3
def _faces(pairs, split, which: str, codes: dict) -> list[str]:
    return [p.pid for p in pairs if p.pid in split[which] and p.pid not in EXCLUDE and f"{p.pid}_pre" in codes]


def c3_perturb(cfg, kit: Kit, pairs, split, out: Path, limit: int | None, w_std: np.ndarray) -> int:
    codes = dict(np.load(out / "private" / "c1_codes.npz"))
    faces = _faces(pairs, split, "dev", codes) + _faces(pairs, split, "test", codes)
    faces = faces[:limit]
    L = kit.gan.n_layers
    chunk, n_done = 32, 0
    for c0 in range(0, len(faces), chunk):
        path = out / "private" / f"c3_chunk_{c0 // chunk:03d}.npz"
        if path.exists():
            n_done += len(faces[c0:c0 + chunk])
            continue
        meta, eps_all, res = [], [], []
        for pid in faces[c0:c0 + chunk]:                 # render + measure per face (bounded memory)
            w0 = codes[f"{pid}_pre"]
            assert w0.shape == (L, 512), w0.shape
            rng = np.random.default_rng([cfg.seed, int(pid), 3])
            ws = [w0]
            meta.append((pid, -1, -1, 0))
            for g, K in GROUP_K.items():
                a, b = GROUPS[g]
                for k in range(K):
                    eps = (rng.standard_normal(512) * w_std).astype(np.float32)
                    eps_all.append(eps.astype(np.float16))
                    for sign in (1, -1):
                        w = w0.copy()
                        w[a:b] += sign * SIGMA * eps
                        meta.append((pid, g, k, sign))
                        ws.append(w)
            res += [measure(im, kit.lmk) for im in render(kit.gan, np.stack(ws))]
        ok = np.array([r is not None for r in res])
        nan = lambda shp: np.full(shp, np.nan, np.float32)   # noqa: E731
        np.savez_compressed(
            path, pid=np.array([m[0] for m in meta]), group=np.array([m[1] for m in meta], np.int8),
            k=np.array([m[2] for m in meta], np.int16), sign=np.array([m[3] for m in meta], np.int8), ok=ok,
            lm=np.stack([r["lm"] if r else nan((478, 2)) for r in res]),
            pose=np.stack([r["pose"] if r else nan(3) for r in res]),
            bs=np.stack([r["bs"] if r else nan(len(BLENDSHAPES)) for r in res]),
            m=np.stack([r["m"] if r else nan(len(MCOLS)) for r in res]),
            eps=np.stack(eps_all), sigma=SIGMA)
        n_done += len(faces[c0:c0 + chunk])
        print(f"  C3 {n_done}/{len(faces)} faces (ok {ok.mean():.1%})", flush=True)
    (out / "c3_done.json").write_text(json.dumps({"faces": faces, "group_k": GROUP_K, "sigma": SIGMA,
                                                   "groups": GROUPS, "metrics": MCOLS, "blendshapes": BLENDSHAPES}))
    return n_done


# ------------------------------------------------------------------------------------------ C4
def _closed_loop(kit, w0, D, a_cap, r, tol, m0, ti, gan=None):
    """Secant search on the measured target (editspec D6 settings). Yields per-iteration records."""
    gan = gan or kit.gan
    a_prev, ach_prev, a, retried = 0.0, 0.0, r, False
    for it in range(E.LOOP_MAX_ITERS):
        img = render(gan, (w0 + a * D)[None])[0]
        mavg, first = measure_avg(img, kit.lmk)
        if mavg is None:
            yield it, a, None, None, img, "measure_fail"
            return
        ach = mavg - m0
        yield it, a, mavg, first, img, None
        t = ach[ti]
        if abs(t - r) <= tol:
            return "converged"
        slope = (t - ach_prev) / (a - a_prev)
        if not np.isfinite(slope) or slope <= 0:
            if it == 0 and not retried:           # pre-declared: one doubled step in case noise flipped the sign
                retried, a = True, 2 * a
                continue
            return "non_monotone"
        a_prev, ach_prev = a, t
        a = a + (r - t) / slope
        if abs(a) > a_cap:
            a = float(np.sign(a) * a_cap)
            if abs(a_prev) >= a_cap:
                return "range_limit"
    return "not_converged"


def c4_edits(cfg, kit: Kit, pairs, split, out: Path, limit: int | None, asset, codes: dict | None = None,
             gan_for=None, prefix: str = "c4", test_only: bool = False, with_b4: bool = True,
             emb: dict | None = None) -> int:
    """E3 editor run. Stage 7 reuses it unchanged for other encoders: `codes` (pid_pre -> W+), `gan_for(pid)` ->
    renderer for that face (HyperStyle: per-face weight offsets), `prefix` for the output files."""
    codes = codes if codes is not None else dict(np.load(out / "private" / "c1_codes.npz"))
    emb = emb if emb is not None else dict(np.load(out / "private" / "c1_landmarks_embeddings.npz"))
    rng = np.random.default_rng(cfg.seed + 60)
    dev = _faces(pairs, split, "dev", codes)
    dev = [dev[i] for i in sorted(rng.choice(len(dev), min(N_DEV_EDIT, len(dev)), replace=False))]
    test = _faces(pairs, split, "test", codes)
    faces = ([] if test_only else [(p, "dev") for p in dev[:limit]]) + [(p, "test") for p in test[:limit]]
    tgt = [str(t) for t in asset["targets"]]
    mdc = dict(zip([str(m) for m in asset["mdc_metrics"]], asset["mdc"]))
    b4 = dict(zip([str(m) for m in asset["b4_metrics"]], asset["b4_w_delta"]))
    w_std = asset["w_std"].astype(np.float64)
    L = kit.gan.n_layers
    dirs = {}
    for t, g, wd in zip(tgt, asset["groups"], asset["w_delta"]):
        D = np.zeros((L, 512), np.float32)
        a, b = GROUPS[int(g)]
        D[a:b] = wd
        dirs[t] = (D, U_CAP / float(np.linalg.norm(wd / w_std)))      # direction, max |a| inside sampled range
    B4 = {t: np.repeat(v[None], L, 0).astype(np.float32) for t, v in b4.items()}
    chunk = 20
    img_dir = out / "private" / f"{prefix}_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    n_saved = 0
    for c0 in range(0, len(faces), chunk):
        path = out / f"{prefix}_edits_{c0 // chunk:03d}.csv"
        if path.exists():
            continue
        rows, lms = [], {}
        for pid, which in faces[c0:c0 + chunk]:
            w0 = codes[f"{pid}_pre"]
            rend = gan_for(pid) if gan_for else kit.gan
            base = render(rend, w0[None])[0]
            m0, f0 = measure_avg(base, kit.lmk)
            if m0 is None:
                rows.append({"pid": pid, "split": which, "status": "base_measure_fail"})
                continue
            e_base = arc_embed(kit, base, f0["lm"])
            fb, _ = fan_metrics(kit, base, face_box(f0["lm"] * MEAS_RES))
            e_real = emb.get(f"arc_real_{pid}_pre")
            lms[f"{pid}_base"] = f0["lm"]
            reqs = [(t, "clinical_2mdc", s * mdc[t]) for t, s in CLIN_REQ_MDC.items()]
            if which == "dev":
                reqs += [(str(t), "dev_mean", float(r)) for t, r in zip(asset["e3_targets"], asset["e3_requests"])]
            save_img = which == "test" and n_saved < N_IMG_SAVE
            for t, kind, r in reqs:
                D, a_cap = dirs[t]
                common = {"pid": pid, "split": which, "target": t, "request_kind": kind, "request": r,
                          "mdc_target": mdc[t]}
                loop = _closed_loop(kit, w0, D, a_cap, r, E.LOOP_TARGET_TOL_MDC * mdc[t], m0, MCOLS.index(t), rend)
                last, status = None, None
                while True:
                    try:
                        it, a, mavg, first, img, err = next(loop)
                    except StopIteration as stop:
                        status = stop.value
                        break
                    if err:
                        status = err
                        break
                    last = (it, a, mavg, first, img)
                    rows.append({**common, "method": "closed", "iter": it, "a": a,
                                 **{f"d_{k}": v for k, v in zip(MCOLS, mavg - m0)},
                                 **{f"d_{k}": v for k, v in zip(("yaw", "pitch", "roll"), first["pose"] - f0["pose"])},
                                 **{f"d_bs_{b}": v for b, v in zip(BLENDSHAPES, first["bs"] - f0["bs"])}})
                    lms[f"{pid}_{t}_{kind}_closed_{it}"] = first["lm"]
                if last is not None:                              # judges on the final iterate
                    it, a, mavg, first, img = last
                    e = arc_embed(kit, img, first["lm"])
                    fe, _ = fan_metrics(kit, img, face_box(first["lm"] * MEAS_RES))
                    rows[-1].update({"final": True, "status": status,
                                     "arc_cos_edit_base": float(e @ e_base) if e is not None else np.nan,
                                     "arc_cos_edit_real": float(e @ e_real) if (e is not None and e_real is not None) else np.nan,
                                     "lpips_edit_base": lpips256(kit, img, base),
                                     **{f"fan_d_{k}": fe[k] - fb[k] for k in FAN_METRICS}})
                    if save_img and t in ("nose_width", "brow_height") and kind == "clinical_2mdc":
                        cv2.imwrite(str(img_dir / f"{pid}_{t}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                else:
                    rows.append({**common, "method": "closed", "status": status, "final": True})
                if with_b4 and t in B4:                           # B4: global ridge, open loop, all layers
                    img = render(rend, (w0 + r * B4[t])[None])[0]
                    mavg, first = measure_avg(img, kit.lmk)
                    rec = {**common, "method": "b4", "iter": 0, "a": r, "final": True,
                           "status": "ok" if mavg is not None else "measure_fail"}
                    if mavg is not None:
                        e = arc_embed(kit, img, first["lm"])
                        rec.update({**{f"d_{k}": v for k, v in zip(MCOLS, mavg - m0)},
                                    **{f"d_{k}": v for k, v in zip(("yaw", "pitch", "roll"), first["pose"] - f0["pose"])},
                                    **{f"d_bs_{b}": v for b, v in zip(BLENDSHAPES, first["bs"] - f0["bs"])},
                                    "arc_cos_edit_base": float(e @ e_base) if e is not None else np.nan})
                    rows.append(rec)
            if save_img:
                cv2.imwrite(str(img_dir / f"{pid}_base.jpg"), cv2.cvtColor(base, cv2.COLOR_RGB2BGR))
                n_saved += 1
        np.savez_compressed(out / "private" / f"{prefix}_landmarks_{c0 // chunk:03d}.npz", **lms)
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  {prefix} {min(c0 + chunk, len(faces))}/{len(faces)} faces", flush=True)
    return len(faces)


# ------------------------------------------------------------------------------------------ C5
def c5_pti(cfg, kit: Kit, pairs, split, out: Path, limit: int | None, asset, deadline: float) -> int:
    from ..gan import pti_tune
    if kit.lpips is None:
        raise RuntimeError("PTI needs LPIPS, which is unavailable")
    codes = dict(np.load(out / "private" / "c1_codes.npz"))
    edits = pd.concat([pd.read_csv(f, dtype={"pid": str}) for f in sorted(out.glob("c4_edits_*.csv"))])
    fin = edits[(edits.method == "closed") & (edits.get("final") == True) & (edits.request_kind == "clinical_2mdc")]  # noqa: E712
    rng = np.random.default_rng(cfg.seed + 70)
    test = _faces(pairs, split, "test", codes)
    sel = [test[i] for i in sorted(rng.choice(len(test), min(N_PTI, len(test)), replace=False))][:limit]
    by_pid = {p.pid: p for p in pairs}
    tgt = [str(t) for t in asset["targets"]]
    (out / "private" / "c5_images").mkdir(parents=True, exist_ok=True)
    rows = []
    for n, pid in enumerate(sel):
        if time.time() > deadline - 20 * 60:
            print("  C5 stopping: time budget", flush=True)
            break
        rgb = cv2.cvtColor(read_bgr(by_pid[pid].pre), cv2.COLOR_BGR2RGB)
        try:
            aligned, _ = align_ffhq(rgb, kit.lmk(rgb).lm_px, 1024)
        except FaceNotFound:
            continue
        w0 = codes[f"{pid}_pre"]
        t0 = time.time()
        Gp = pti_tune(kit.gan, torch.from_numpy(w0[None]).to(kit.gan.device), from_uint8(aligned, 1024, kit.gan.device),
                      kit.lpips)
        secs = time.time() - t0
        real = cv2.resize(aligned, (MEAS_RES, MEAS_RES), interpolation=cv2.INTER_AREA)
        mr = measure(real, kit.lmk)
        e_real = arc_embed(kit, real, mr["lm"]) if mr else None
        variants = [("rebuild", 0.0, None)]
        for t in CLIN_REQ_MDC:
            f = fin[(fin.pid == pid) & (fin.target == t)]
            if len(f) and np.isfinite(f.a.iloc[0]):
                variants.append((t, float(f.a.iloc[0]), t))
        m_base = None
        for name, a, t in variants:
            w = w0.copy()
            if t is not None:
                gi = int(asset["groups"][tgt.index(t)])
                s, e_ = GROUPS[gi]
                w[s:e_] += a * asset["w_delta"][tgt.index(t)]
            with torch.no_grad():
                im = F.interpolate(kit.gan.synth(torch.from_numpy(w[None]).to(kit.gan.device), Gp),
                                   size=(MEAS_RES, MEAS_RES), mode="area")
            img = to_uint8(im)[0]
            mavg, first = measure_avg(img, kit.lmk)
            e = arc_embed(kit, img, first["lm"]) if first else None
            if name == "rebuild":
                m_base = mavg
            rows.append({"pid": pid, "variant": name, "a": a, "pti_seconds": secs,
                         "arc_cos_real": float(e @ e_real) if (e is not None and e_real is not None) else np.nan,
                         "lpips_real": lpips256(kit, img, real),
                         **({f"d_{k}": v for k, v in zip(MCOLS, mavg - m_base)} if (mavg is not None and m_base is not None) else {})})
            cv2.imwrite(str(out / "private" / "c5_images" / f"{pid}_{name}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        del Gp
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"  C5 {n + 1}/{len(sel)} ({secs:.0f}s PTI)", flush=True)
    pd.DataFrame(rows).to_csv(out / "c5_pti.csv", index=False)
    return len(rows)


# ------------------------------------------------------------------------------------------ main
def _zip(paths, root: Path, dst: Path) -> None:
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as z:
        for f in paths:
            if f.is_file():
                z.write(f, f.relative_to(root))
    print(f"wrote {dst} ({dst.stat().st_size / 1e6:.0f} MB)", flush=True)


def package(cfg, out: Path) -> None:
    _zip([*out.glob("*.csv"), *out.glob("*.json"), cfg.out_dir / "splits.json"], cfg.out_dir,
         cfg.out_dir / "stage5_results.zip")
    _zip(sorted((out / "private").rglob("*")), cfg.out_dir, cfg.out_dir / "stage5_private_hda.zip")


DONE_MARK = {"c1": "c1_images.csv", "c2": "c2_phantom.csv", "c3": "c3_done.json", "c4": "c4_done.json",
             "c5": "c5_pti.csv"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["c1", "c2", "c3", "c4", "c5"])
    ap.add_argument("--hours", type=float, default=11.0, help="stop starting new work after this many hours")
    ap.add_argument("--limit", type=int, default=None, help="smoke test only: first N faces/pairs")
    a = ap.parse_args(argv)
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Stage 5 needs a GPU (Kaggle: Settings -> Accelerator -> GPU T4)")
    t_start = time.time()
    deadline = t_start + a.hours * 3600
    cfg = get_config()
    seed_everything(cfg.seed)
    out = cfg.out_dir / "stage5"
    (out / "private").mkdir(parents=True, exist_ok=True)
    pairs = list_pairs(cfg)
    split = make_split(pairs, cfg, cfg.out_dir / "splits.json")
    asset = dict(np.load(ASSET))
    assert asset["w_delta"].shape[1] == 512 and asset["w_std"].shape == (512,), "editor asset malformed"
    kit = load_kit(cfg, DEVICE)
    status_path = out / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    status["judges"] = kit.notes or "all available"
    print("judges:", status["judges"], flush=True)
    steps = {"c1": lambda: c1_inversion(cfg, kit, pairs, split, out, a.limit),
             "c2": lambda: c2_phantom(cfg, kit, pairs, split, out, a.limit),
             "c3": lambda: c3_perturb(cfg, kit, pairs, split, out, a.limit, asset["w_std"].astype(np.float32)),
             "c4": lambda: c4_edits(cfg, kit, pairs, split, out, a.limit, asset),
             "c5": lambda: c5_pti(cfg, kit, pairs, split, out, a.limit, asset, deadline)}
    needs = {"c3": "c1", "c4": "c1", "c5": "c4"}
    # Order: C1 (needed by all), C2 (short), C4 (the editor, highest value), C3 (long), C5 (optional)
    for name in [s for s in ("c1", "c2", "c4", "c3", "c5") if s in a.only]:
        if (out / DONE_MARK[name]).exists():
            print(f"== {name}: already done, skipping", flush=True)
            continue
        if name in needs and not (out / DONE_MARK[needs[name]]).exists():
            status[name] = {"status": "skipped", "reason": f"needs {needs[name]}"}
            continue
        if time.time() > deadline - (90 * 60 if name == "c5" else 0):
            status[name] = {"status": "skipped", "reason": "time budget"}
            continue
        t0 = time.time()
        print(f"== {name} start ({(t0 - t_start) / 60:.0f} min elapsed)", flush=True)
        try:
            n = steps[name]()
            if name == "c4":
                (out / "c4_done.json").write_text(json.dumps({"faces": n}))
            status[name] = {"status": "done", "n": n, "seconds": round(time.time() - t0)}
        except Exception as e:                    # noqa: BLE001  recorded; dependent steps are skipped
            status[name] = {"status": "failed", "error": repr(e), "traceback": traceback.format_exc()[-3000:],
                            "seconds": round(time.time() - t0)}
            print(status[name]["traceback"], flush=True)
        status_path.write_text(json.dumps(status, indent=1, default=str))
        print(f"== {name}: {status[name]['status']} ({(time.time() - t0) / 60:.0f} min)", flush=True)
        package(cfg, out)                          # partial results survive a crash or timeout
    kit.lmk.close()
    weights = {"e4e_ffhq_encode.pt": cfg.models_dir / "e4e_ffhq_encode.pt",
               "face_landmarker.task": cfg.models_dir / "face_landmarker.task",
               "w600k_r50.onnx": cfg.models_dir / "w600k_r50.onnx", "editor_v1.npz": ASSET}
    from importlib import metadata as _md
    extra_pk = {}
    for name in ("face-alignment", "onnxruntime", "lpips"):
        try:
            extra_pk[name] = _md.version(name)
        except _md.PackageNotFoundError:
            extra_pk[name] = "not installed"
    write_run_meta(out, cfg.as_dict(), {k: v for k, v in weights.items() if Path(v).exists()},
                   {"stage": "stage5_overnight", "seconds": time.time() - t_start, "status": status,
                    "extra_packages": extra_pk})
    package(cfg, out)
    failed = [k for k, v in status.items() if isinstance(v, dict) and v.get("status") == "failed"]
    if failed:
        raise SystemExit(f"components failed: {failed} (see status.json); finished ones are in the zips")


if __name__ == "__main__":
    main()
