"""Stage 7 (final GPU run): which inversion should feed the editor? Fine-tune e4e on HDA, compare with HyperStyle.

Pre-registered in outputs/eval_plan.json ("stage7"). One Kaggle T4 session, resumable components (skipped when
their outputs exist; a failure is recorded in status.json, never replaced by substitute output):

K1 fine-tune  e4e encoder on HDA DEV photos (pre + post, included pairs), generator frozen, e4e's own losses
              (medsg.finetune). Early stopping on held-out dev subjects. Test photos are never seen.
K2 inversion  every included TEST pre-op photo through: e4e (official), e4e-HDA (K1), HyperStyle (official).
              ArcFace(real, rebuild), LPIPS, MediaPipe + FAN landmark drift, seconds; Inception features (KID).
K6 examples   synthetic StyleGAN2 faces (not real people) for the public demo + GPU timing of the app's GAN path.
K3 editor     the pre-registered E3 editor (stage5 c4_edits, unchanged) on TEST faces for each encoder:
              5 clinical requests (+-2 MDC), closed loop, judges. Same session for all encoders = fair comparison.
K5 realism    Inception features of every final closed-loop edit -> KID vs real photos (with K2's rebuilds); the
              same edits judged in depth: 3DDFA canonical shape change and Depth Anything V2 depth-change locality.
K7 warp judge warp mode (A1) on all test faces, judged by FAN, the 3D face model and a dense depth net: does an independent
              detector see the exact change, and does it confirm nothing else moved? (+ Inception features)
K4 PTI speed  40 test faces: PTI 300 steps from e4e (Stage 5 setting) vs 100 steps from e4e / e4e-HDA pivots.
Outputs: outputs/stage7/*.csv|json (tables), outputs/stage7/private/* (HDA-derived: codes, features, images,
fine-tuned weights; never commit, never publish). Zips: stage7_results.zip, stage7_private_hda.zip.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..align import align_ffhq
from ..config import get_config
from ..data import list_pairs, make_split, seed_everything
from ..dedup import EXCLUDE
from ..judges import face_box
from ..landmarks import FaceNotFound
from ..measure import MCOLS, MEAS_RES, measure, measure_avg
from ..runmeta import write_run_meta
from . import stage5_overnight as S5
from .stage1_measure import read_bgr

DEVICE = "cuda"
ENCODERS = ["e4e", "e4e_hda", "hyperstyle"]
N_PTI, N_SYNTH, N_EXAMPLES = 40, 400, 8
VAL_FRAC = 0.15                       # share of dev subjects held out for early stopping


# ------------------------------------------------------------------------------------------ helpers
def aligned_photo(kit, path: Path) -> np.ndarray:
    """Raw HDA photo -> FFHQ-aligned 1024 px RGB (same as Stage 5 C1)."""
    rgb = cv2.cvtColor(read_bgr(path), cv2.COLOR_BGR2RGB)
    return align_ffhq(rgb, kit.lmk(rgb).lm_px, 1024)[0]


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def small(img: np.ndarray, s: int) -> np.ndarray:
    return cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)


class Inception:
    """torchvision Inception-v3 (ImageNet) pool features, 2048-d, for KID. Same weights are used locally for warps."""

    def __init__(self, device: str):
        from torchvision.models import Inception_V3_Weights, inception_v3
        self.net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).eval().to(device)
        self.net.fc = torch.nn.Identity()
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255).to(self.device)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        return self.net((x - self.mean) / self.std)[0].cpu().numpy().astype(np.float32)


def kid(a: np.ndarray, b: np.ndarray, n_sub: int = 100, size: int = 100, seed: int = 0) -> tuple[float, float]:
    """Kernel Inception Distance (Binkowski et al. 2018): unbiased MMD^2, cubic polynomial kernel; mean, sd x 1e3."""
    rng = np.random.default_rng(seed)
    m = min(size, len(a), len(b))
    d = a.shape[1]
    vals = []
    for _ in range(n_sub):
        x, y = a[rng.choice(len(a), m, replace=False)], b[rng.choice(len(b), m, replace=False)]
        kxx, kyy, kxy = [(u @ v.T / d + 1) ** 3 for u, v in ((x, x), (y, y), (x, y))]
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                    - 2 * kxy.mean())
    return 1e3 * float(np.mean(vals)), 1e3 * float(np.std(vals))


def depth_judges(cfg) -> tuple:
    """(3DDFA-V2 face model, Depth Anything V2) or None for each, with a note; absence is recorded, never faked."""
    notes, f3d, dnet = {}, None, None
    try:
        from ..face3d import Face3D
        f3d = Face3D(cfg.repos_dir / "3DDFA_V2")
    except Exception as e:                        # noqa: BLE001
        notes["3ddfa"] = f"unavailable: {e!r}"
    try:
        from ..depth import DepthNet
        dnet = DepthNet(DEVICE)
    except Exception as e:                        # noqa: BLE001
        notes["depth_anything"] = f"unavailable: {e!r}"
    return f3d, dnet, notes


def shape3d(f3d, rgb: np.ndarray, lm_px: np.ndarray) -> dict:
    """Pose- and expression-free 3D shape metrics (3DDFA canonical mesh); {} if the fit fails."""
    if f3d is None:
        return {}
    try:
        fit = f3d.fit(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), lm_px)
    except (ValueError, IndexError):
        return {}
    return {f"d3_{k}": v for k, v in f3d.canonical_metrics(fit.alpha_shp).items()}


def test_faces(pairs, split) -> list:
    return [p for p in pairs if p.pid in split["test"] and p.pid not in EXCLUDE]


def load_hyperstyle(cfg):
    from ..hyperstyle import HyperStyleGAN
    return HyperStyleGAN(cfg.repos_dir / "hyperstyle", cfg.models_dir / "hyperstyle_ffhq.pt",
                         cfg.models_dir / "faces_w_encoder.pt", cfg.out_dir.parent / "src_ext", DEVICE)


def hda_encoder(kit, out: Path):
    """The K1 fine-tuned encoder, or None if K1 did not finish (its rows are then simply absent)."""
    if not (out / "private" / "e4e_hda_encoder.pt").exists():
        return None
    enc = copy.deepcopy(kit.gan.encoder)
    enc.load_state_dict(torch.load(out / "private" / "e4e_hda_encoder.pt", map_location=DEVICE))
    return enc.eval()


# ------------------------------------------------------------------------------------------ K1
def k1_finetune(cfg, kit, pairs, split, out: Path, limit, deadline: float) -> int:
    from ..finetune import finetune_e4e
    from ..identity import FaceNetID
    dev = [p for p in pairs if p.pid in split["dev"] and p.pid not in EXCLUDE][:limit]
    rng = np.random.default_rng(cfg.seed + 80)
    val_ids = set(rng.choice([p.pid for p in dev], max(1, round(VAL_FRAC * len(dev))), replace=False))
    tr, va = [], []
    for p in dev:
        for path in (p.pre, p.post):
            try:
                (va if p.pid in val_ids else tr).append(small(aligned_photo(kit, path), 256))
            except (FaceNotFound, IOError):
                pass
    print(f"  K1: {len(tr)} train / {len(va)} val photos (dev only)", flush=True)
    facenet = FaceNetID(DEVICE).net
    for q in facenet.parameters():
        q.requires_grad_(False)
    kw = dict(steps=20 if limit else 500, eval_every=10 if limit else 50, seed=cfg.seed, deadline=deadline - 7 * 3600)
    try:
        state, log = finetune_e4e(kit.gan, np.stack(tr), np.stack(va), kit.lpips, facenet, **kw)
    except torch.cuda.OutOfMemoryError:                  # same effective batch (8), smaller pieces
        torch.cuda.empty_cache()
        print("  K1: out of GPU memory at micro-batch 2; retrying with 1 x 8", flush=True)
        state, log = finetune_e4e(kit.gan, np.stack(tr), np.stack(va), kit.lpips, facenet, micro=1, accum=8, **kw)
    torch.save(state, out / "private" / "e4e_hda_encoder.pt")
    pd.DataFrame(log).assign(n_train=len(tr), n_val=len(va)).to_csv(out / "k1_finetune_log.csv", index=False)
    return len(tr)


# ------------------------------------------------------------------------------------------ K2
def k2_inversion(cfg, kit, pairs, split, out: Path, limit) -> int:
    inc, enc_hda = Inception(DEVICE), hda_encoder(kit, out)
    hs, notes = None, {}
    try:
        from ..hyperstyle import same_generator
        hs = load_hyperstyle(cfg)
        notes["hyperstyle_vs_e4e_generator_max_abs_diff"] = same_generator(cfg.models_dir / "hyperstyle_ffhq.pt",
                                                                           cfg.models_dir / "e4e_ffhq_encode.pt")
    except Exception as e:                        # noqa: BLE001  recorded, HyperStyle rows simply absent
        notes["hyperstyle"] = f"unavailable: {e!r}"
    print("  K2 notes:", notes, flush=True)
    rows, codes, feats, emb = [], {k: {} for k in ENCODERS}, {}, {}
    for i, p in enumerate(test_faces(pairs, split)[:limit]):
        try:
            al = aligned_photo(kit, p.pre)
        except (FaceNotFound, IOError):
            rows.append({"pid": p.pid, "encoder": "all", "ok": False})
            continue
        real = small(al, MEAS_RES)
        mr = measure(real, kit.lmk)
        if mr is None:
            continue
        er = S5.arc_embed(kit, real, mr["lm"])
        emb[f"arc_real_{p.pid}_pre"] = er
        fr, _ = S5.fan_metrics(kit, real, face_box(mr["lm"] * MEAS_RES))
        feats[f"real_{p.pid}"] = inc(real)
        x256 = F.interpolate(torch.from_numpy(al).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(DEVICE),
                             size=(256, 256), mode="area")
        for name in ENCODERS:
            if (name == "hyperstyle" and hs is None) or (name == "e4e_hda" and enc_hda is None):
                continue
            _sync()
            t0 = time.time()
            with torch.no_grad():
                if name == "e4e":
                    code, rend = kit.gan.encode(x256), kit.gan
                elif name == "e4e_hda":
                    code, rend = enc_hda(x256) + kit.gan.latent_avg[None], kit.gan
                else:
                    code, deltas = hs.invert(x256)
                    rend = hs.for_face(deltas)
                rec = S5.render(rend, code.cpu().numpy())[0]
            _sync()
            secs = time.time() - t0
            mc = measure(rec, kit.lmk)
            r = {"pid": p.pid, "encoder": name, "ok": mc is not None, "seconds": secs,
                 "lpips_real_rec": S5.lpips256(kit, real, rec)}
            if mc is not None:
                ec = S5.arc_embed(kit, rec, mc["lm"])
                fc, _ = S5.fan_metrics(kit, rec, face_box(mc["lm"] * MEAS_RES))
                r.update(arc_cos_real_rec=float(er @ ec),
                         **{f"mp_drift_{k}": c - b for k, c, b in zip(MCOLS, mc["m"], mr["m"])},
                         **{f"fan_drift_{k}": fc[k] - fr[k] for k in fr})
            rows.append(r)
            codes[name][f"{p.pid}_pre"] = code[0].cpu().numpy().astype(np.float32)
            feats[f"{name}_rec_{p.pid}"] = inc(rec)
        if i % 50 == 0:
            print(f"  K2 {i} faces", flush=True)
    for name, c in codes.items():
        if c:
            np.savez_compressed(out / "private" / f"k2_codes_{name}.npz", **c)
    np.savez_compressed(out / "private" / "k2_arc_real.npz", **emb)
    np.savez_compressed(out / "private" / "k2_features.npz", **feats)
    (out / "k2_notes.json").write_text(json.dumps(notes, indent=1))
    pd.DataFrame(rows).to_csv(out / "k2_inversion.csv", index=False)
    return len(rows)


# ------------------------------------------------------------------------------------------ K3
def _renderer_factory(cfg, kit, pairs, name: str):
    """gan_for(pid) for c4_edits: e4e codes render with G; HyperStyle re-predicts that face's weight offsets."""
    if name != "hyperstyle":
        return None, None
    hs, by_pid = load_hyperstyle(cfg), {p.pid: p for p in pairs}

    def gan_for(pid: str):
        al = aligned_photo(kit, by_pid[pid].pre)
        x = F.interpolate(torch.from_numpy(al).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(DEVICE),
                          size=(256, 256), mode="area")
        return hs.for_face(hs.invert(x)[1])
    return gan_for, hs


def k3_editor(cfg, kit, pairs, split, out: Path, limit, asset, name: str) -> int:
    f = out / "private" / f"k2_codes_{name}.npz"
    if not f.exists():
        raise FileNotFoundError(f"no K2 codes for {name}")
    gan_for, _ = _renderer_factory(cfg, kit, pairs, name)
    emb = dict(np.load(out / "private" / "k2_arc_real.npz"))
    return S5.c4_edits(cfg, kit, pairs, split, out, limit, asset, codes=dict(np.load(f)), gan_for=gan_for,
                       prefix=f"k3_{name}", test_only=True, with_b4=False, emb=emb)


# ------------------------------------------------------------------------------------------ K5
def k5_realism(cfg, kit, pairs, split, out: Path, limit, asset) -> int:
    """KID of every final closed-loop edit, and the two depth judges on each edit vs its own rebuild."""
    from ..depth import REGION, depth_change
    inc = Inception(DEVICE)
    f3d, dnet, notes = depth_judges(cfg)
    (out / "k5_notes.json").write_text(json.dumps(notes or "all depth judges available", indent=1))
    drows = []
    feats = dict(np.load(out / "private" / "k2_features.npz"))
    tgt = [str(t) for t in asset["targets"]]
    from .stage3_synth import GROUPS
    new = {}
    for name in ENCODERS:
        files = sorted(out.glob(f"k3_{name}_edits_*.csv"))
        cf = out / "private" / f"k2_codes_{name}.npz"
        if not files or not cf.exists():
            continue
        codes = dict(np.load(cf))
        ed = pd.concat([pd.read_csv(x, dtype={"pid": str}) for x in files])
        fin = ed[(ed.method == "closed") & (ed.get("final") == True) & ed.a.notna()]  # noqa: E712
        gan_for, _ = _renderer_factory(cfg, kit, pairs, name)
        for pid, g in list(fin.groupby("pid"))[:limit]:
            rend = gan_for(pid) if gan_for else kit.gan
            base = S5.render(rend, codes[f"{pid}_pre"][None])[0]
            mb = measure(base, kit.lmk)
            if mb is None:
                continue
            Lb = mb["lm"] * MEAS_RES
            s0, d0 = shape3d(f3d, base, Lb), (dnet(base) if dnet else None)
            for _, r in g.iterrows():
                w = codes[f"{pid}_pre"].copy()
                k = tgt.index(r.target)
                s, e = GROUPS[int(asset["groups"][k])]
                w[s:e] += float(r.a) * asset["w_delta"][k]
                img = S5.render(rend, w[None])[0]
                new[f"{name}_edit_{r.target}_{pid}"] = inc(img)
                me = measure(img, kit.lmk)
                rec = {"encoder": name, "pid": pid, "target": r.target, "a": float(r.a)}
                if me is not None:
                    s1 = shape3d(f3d, img, me["lm"] * MEAS_RES)
                    rec.update({f"{q}_d": s1[q] - s0[q] for q in s1 if q in s0})
                if dnet is not None and r.target in REGION:
                    rec.update(depth_change(d0, dnet(img), Lb, r.target))
                drows.append(rec)
        print(f"  K5 {name} done", flush=True)
    feats.update(new)
    if (out / "private" / "k7_features.npz").exists():           # warp-mode edits (K7), same Inception
        feats.update(dict(np.load(out / "private" / "k7_features.npz")))
    np.savez_compressed(out / "private" / "k5_features.npz", **feats)
    real = np.stack([v for k, v in feats.items() if k.startswith("real_")])
    rows = []
    for setname in sorted({k.rsplit("_", 1)[0] for k in feats if not k.startswith("real_")}):
        X = np.stack([v for k, v in feats.items() if k.rsplit("_", 1)[0] == setname])
        m, sd = kid(real, X)
        rows.append({"set": setname, "n": len(X), "kid_x1e3": m, "kid_sd_x1e3": sd})
    half = len(real) // 2                                  # floor: two halves of the real photos
    m, sd = kid(real[:half], real[half:])
    rows.append({"set": "real_half_vs_half (floor)", "n": half, "kid_x1e3": m, "kid_sd_x1e3": sd})
    pd.DataFrame(drows).to_csv(out / "k5_gan_depth.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "k5_kid.csv", index=False)
    return len(rows)


# ------------------------------------------------------------------------------------------ K4
def k4_pti(cfg, kit, pairs, split, out: Path, limit, deadline: float) -> int:
    from ..gan import pti_tune
    enc_hda = hda_encoder(kit, out)
    faces = test_faces(pairs, split)
    rng = np.random.default_rng(cfg.seed + 90)
    sel = [faces[i] for i in sorted(rng.choice(len(faces), min(N_PTI, len(faces)), replace=False))][:limit]
    emb = dict(np.load(out / "private" / "k2_arc_real.npz"))
    rows = []
    for n, p in enumerate(sel):
        if time.time() > deadline - 15 * 60:
            break
        try:
            al = aligned_photo(kit, p.pre)
        except (FaceNotFound, IOError):
            continue
        x = torch.from_numpy(al).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(DEVICE)
        x256 = F.interpolate(x, size=(256, 256), mode="area")
        real = small(al, MEAS_RES)
        er = emb.get(f"arc_real_{p.pid}_pre")
        with torch.no_grad():
            pivots = {"e4e": kit.gan.encode(x256)}
            if enc_hda is not None:
                pivots["e4e_hda"] = enc_hda(x256) + kit.gan.latent_avg[None]
        for pivot, steps in [(q, n) for q, n in (("e4e", 300), ("e4e", 100), ("e4e_hda", 100)) if q in pivots]:
            _sync()
            t0 = time.time()
            G = pti_tune(kit.gan, pivots[pivot], x, kit.lpips, steps=steps)
            _sync()
            secs = time.time() - t0
            with torch.no_grad():
                rec = S5.to_uint8(F.interpolate(kit.gan.synth(pivots[pivot], G), size=(MEAS_RES,) * 2, mode="area"))[0]
            mc = measure(rec, kit.lmk)
            ec = S5.arc_embed(kit, rec, mc["lm"]) if mc is not None else None
            rows.append({"pid": p.pid, "pivot": pivot, "steps": steps, "seconds": secs,
                         "arc_cos_real_rec": float(er @ ec) if (er is not None and ec is not None) else np.nan,
                         "lpips_real_rec": S5.lpips256(kit, real, rec)})
            del G
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"  K4 {n + 1}/{len(sel)}", flush=True)
    pd.DataFrame(rows).to_csv(out / "k4_pti.csv", index=False)
    return len(rows)


# ------------------------------------------------------------------------------------------ K6
def k6_synthetic(cfg, kit, out: Path, limit, asset) -> int:
    """Synthetic demo faces (StyleGAN2 samples, psi 0.7) that look like consultation photos: frontal, neutral."""
    from ..editor import Directions, secant_edit, target_index
    from ..gan import pti_tune
    pub = out / "public_examples"
    pub.mkdir(exist_ok=True)
    gen = torch.Generator().manual_seed(cfg.seed + 7)
    cands = []
    for b in range(0, (40 if limit else N_SYNTH), 8):
        w = kit.gan.sample_w(8, 0.7, gen)
        codes = w[:, None].repeat(1, kit.gan.n_layers, 1).cpu().numpy()
        for c, img in zip(codes, S5.render(kit.gan, codes)):
            m = measure(img, kit.lmk)
            if m is None:
                continue
            bs = dict(zip(S5.BLENDSHAPES, m["bs"]))
            score = (abs(m["pose"][0]) + abs(m["pose"][1]) + 20 * (bs["mouthSmileLeft"] + bs["mouthSmileRight"])
                     + 20 * bs["jawOpen"])
            cands.append((score, c))
    cands.sort(key=lambda t: t[0])
    picked = cands[:N_EXAMPLES]
    for j, (sc, c) in enumerate(picked):
        with torch.no_grad():
            img = S5.to_uint8(kit.gan.synth(torch.from_numpy(c[None]).to(DEVICE)))[0]
        cv2.imwrite(str(pub / f"synthetic_{j + 1:02d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    # GPU timing of the app's GAN path (same calls as app/_gan_core) on 3 synthetic faces
    D, rows = Directions(), []
    for j in range(min(3, len(picked))):
        img = cv2.cvtColor(cv2.imread(str(pub / f"synthetic_{j + 1:02d}.jpg")), cv2.COLOR_BGR2RGB)
        al = align_ffhq(img, kit.lmk(img).lm_px, 1024)[0]
        x = torch.from_numpy(al).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(DEVICE)
        for steps in (300, 100):
            t0 = time.time()
            code = kit.gan.encode(F.interpolate(x, size=(256, 256), mode="area"))
            G = pti_tune(kit.gan, code, x, kit.lpips, steps=steps)
            t_pti = time.time() - t0

            def render(a: float) -> np.ndarray:
                w = code + a * torch.from_numpy(D.D["brow_height"]).to(DEVICE)[None]
                with torch.no_grad():
                    return S5.to_uint8(F.interpolate(kit.gan.synth(w, G), size=(512, 512), mode="area"))[0]
            m0, _ = measure_avg(render(0.0), kit.lmk)

            def rm(a: float):
                m, _ = measure_avg(render(a), kit.lmk)
                return None if m is None else float((m - m0)[target_index("brow_height")])
            r = 2 * D.mdc["brow_height"]
            recs, status = secant_edit(rm, r, 0.5 * D.mdc["brow_height"], D.a_cap["brow_height"])
            rows.append({"face": j + 1, "pti_steps": steps, "seconds_encode_pti": t_pti,
                         "seconds_total": time.time() - t0, "renders": len(recs), "status": status})
    pd.DataFrame(rows).to_csv(out / "k6_app_timing.csv", index=False)
    return len(picked)


# ------------------------------------------------------------------------------------------ K7
WARP_REQ_MM = {"nose_width": -3.5, "brow_height": 3.0, "mrd1": 0.8, "jaw_width": -8.0}   # = eval_plan A1 requests
N_NOISE = 40


def _sample(D: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Bilinear sample of a dense (H, W, 2) field at (n, 2) pixel points."""
    from scipy.ndimage import map_coordinates
    return np.stack([map_coordinates(D[..., c], [pts[:, 1], pts[:, 0]], order=1, mode="nearest") for c in (0, 1)], 1)


def k7_warp_judges(cfg, kit, pairs, split, out: Path, limit) -> int:
    """Warp mode (A1) judged by detectors it was never fitted to: FAN (2D) and 3DDFA-V2 (3D shape, depth).

    FAN 'true' change = FAN's own points moved by the exact dense field; 'measured' = FAN re-run on the warped photo.
    3DDFA: canonical shape-only mesh (pose and expression removed), measured before/after. Noise floors (MDC95)
    of both judges from 7 same-photo perturbations on N_NOISE faces, as Stage 1."""
    from ..depth import depth_change
    from ..judges import FAN_METRICS, metrics_fan
    from ..transfer import IPD_MM, Transfer, dense_field, warp_photo
    from .stage1_measure import PERTURBATIONS, perturb
    T, inc = Transfer(), Inception(DEVICE)
    f3d, dnet, notes = depth_judges(cfg)
    (out / "k7_notes.json").write_text(json.dumps(notes or "all depth judges available", indent=1))

    def judge(img: np.ndarray, lm_px: np.ndarray) -> dict:
        fp = kit.fan.landmarks(img, face_box(lm_px))
        return {**{f"fan_{k}": v for k, v in metrics_fan(fp).items()}, **shape3d(f3d, img, lm_px)}, fp

    rows, noise, feats = [], [], {}
    for i, p in enumerate(test_faces(pairs, split)[:limit]):
        try:
            al = small(aligned_photo(kit, p.pre), MEAS_RES)
        except (FaceNotFound, IOError):
            continue
        m0, f0 = measure_avg(al, kit.lmk)
        if m0 is None:
            continue
        L = f0["lm"].astype(np.float64) * MEAS_RES
        try:
            j0, fp0 = judge(al, L)
        except ValueError:
            continue
        d0 = dnet(al) if dnet else None
        if i < N_NOISE:                                 # judges' own noise floor on harmless re-captures
            for kind in PERTURBATIONS:
                img = cv2.cvtColor(perturb(cv2.cvtColor(al, cv2.COLOR_RGB2BGR), kind), cv2.COLOR_BGR2RGB)
                img = small(img, MEAS_RES) if img.shape[0] != MEAS_RES else img
                mp = measure(img, kit.lmk)
                if mp is None:
                    continue
                img = np.ascontiguousarray(img)
                try:
                    rec = {"pid": p.pid, "kind": kind, **judge(img, mp["lm"] * MEAS_RES)[0]}
                except ValueError:
                    continue
                if dnet is not None and kind in ("jpeg70", "gamma", "downscale"):   # same pixels, photometric noise
                    dd = dnet(img)
                    for t in WARP_REQ_MM:
                        rec.update({f"noise_{t}_{k}": v for k, v in depth_change(d0, dd, L, t).items()})
                noise.append(rec)
        for t, mm in WARP_REQ_MM.items():
            e = T.edit(L, t, mm / IPD_MM)
            w = warp_photo(al, L, e.field_px)
            D = dense_field(L, e.field_px, al.shape[:2])
            true_fan = metrics_fan(fp0 + _sample(D, fp0))
            m1, f1 = measure_avg(w, kit.lmk)
            rec = {"pid": p.pid, "target": t, "request_mm": mm, "exact_achieved": e.after[t] - e.before[t],
                   **{f"true_fan_d_{k}": true_fan[k] - j0[f"fan_{k}"] for k in FAN_METRICS}}
            if m1 is not None:
                try:
                    j1, _ = judge(w, f1["lm"].astype(np.float64) * MEAS_RES)
                    rec.update({f"meas_{k}_d": j1[k] - j0[k] for k in j0 if k in j1}, ok=True)
                except ValueError:
                    rec["ok"] = False
            if d0 is not None:
                rec.update(depth_change(d0, dnet(w), L, t))
            rows.append(rec)
            feats[f"warp_edit_{t}_{p.pid}"] = inc(w)
        if i % 50 == 0:
            print(f"  K7 {i} faces", flush=True)
    np.savez_compressed(out / "private" / "k7_features.npz", **feats)
    N = pd.DataFrame(noise)
    N.to_csv(out / "k7_judge_noise_rows.csv", index=False)
    if len(N):
        from ..stats import mdc95, sem_within
        cols = [c for c in N.columns if c.startswith(("fan_", "d3_"))]
        pd.DataFrame([{"metric": c, "mdc95": mdc95(sem_within(N.pivot(index="pid", columns="kind", values=c).to_numpy())), "n_faces": N.pid.nunique()}
                      for c in cols]).to_csv(out / "k7_judge_noise.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "k7_warp_judges.csv", index=False)
    return len(rows)


# ------------------------------------------------------------------------------------------ main
DONE = {"k1": "k1_finetune_log.csv", "k2": "k2_inversion.csv", "k6": "k6_app_timing.csv",
        **{f"k3_{n}": f"k3_{n}_done.json" for n in ENCODERS}, "k7": "k7_warp_judges.csv", "k5": "k5_kid.csv",
        "k4": "k4_pti.csv"}
NEEDS = {**{f"k3_{n}": "k2" for n in ENCODERS}, "k5": "k2", "k4": "k2"}   # K2 runs without K1 (e4e_hda absent)


def package(cfg, out: Path) -> None:
    S5._zip([*out.glob("*.csv"), *out.glob("*.json"), *(out / "public_examples").glob("*.jpg")], cfg.out_dir,
            cfg.out_dir / "stage7_results.zip")
    S5._zip([f for f in sorted((out / "private").rglob("*")) if f.suffix != ".pt"], cfg.out_dir,
            cfg.out_dir / "stage7_private_hda.zip")          # fine-tuned weights stay on Kaggle unless asked for


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(DONE))
    ap.add_argument("--hours", type=float, default=11.0)
    ap.add_argument("--limit", type=int, default=None, help="smoke test only")
    a = ap.parse_args(argv)
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Stage 7 needs a GPU (Kaggle: Settings -> Accelerator -> GPU T4)")
    t_start = time.time()
    deadline = t_start + a.hours * 3600
    cfg = get_config()
    seed_everything(cfg.seed)
    out = cfg.out_dir / "stage7"
    (out / "private").mkdir(parents=True, exist_ok=True)
    pairs = list_pairs(cfg)
    split = make_split(pairs, cfg, cfg.out_dir / "splits.json")
    asset = dict(np.load(S5.ASSET))
    kit = S5.load_kit(cfg, DEVICE)
    if kit.arc is None or kit.lpips is None:
        raise RuntimeError(f"Stage 7 needs ArcFace and LPIPS: {kit.notes}")
    status_path = out / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    status["judges"] = kit.notes or "all available"
    steps = {"k1": lambda: k1_finetune(cfg, kit, pairs, split, out, a.limit, deadline),
             "k2": lambda: k2_inversion(cfg, kit, pairs, split, out, a.limit),
             "k6": lambda: k6_synthetic(cfg, kit, out, a.limit, asset),
             **{f"k3_{n}": (lambda n=n: k3_editor(cfg, kit, pairs, split, out, a.limit, asset, n)) for n in ENCODERS},
             "k7": lambda: k7_warp_judges(cfg, kit, pairs, split, out, a.limit),
             "k5": lambda: k5_realism(cfg, kit, pairs, split, out, a.limit, asset),
             "k4": lambda: k4_pti(cfg, kit, pairs, split, out, a.limit, deadline)}
    for name in [s for s in DONE if s in a.only]:
        if (out / DONE[name]).exists():
            print(f"== {name}: already done, skipping", flush=True)
            continue
        if name in NEEDS and not (out / DONE[NEEDS[name]]).exists():
            status[name] = {"status": "skipped", "reason": f"needs {NEEDS[name]}"}
            continue
        if time.time() > deadline:
            status[name] = {"status": "skipped", "reason": "time budget"}
            continue
        t0 = time.time()
        print(f"== {name} start ({(t0 - t_start) / 60:.0f} min elapsed)", flush=True)
        try:
            n = steps[name]()
            if name.startswith("k3_"):
                (out / DONE[name]).write_text(json.dumps({"faces": n}))
            status[name] = {"status": "done", "n": n, "seconds": round(time.time() - t0)}
        except Exception as e:                    # noqa: BLE001  recorded; dependants are skipped
            status[name] = {"status": "failed", "error": repr(e), "traceback": traceback.format_exc()[-3000:],
                            "seconds": round(time.time() - t0)}
            print(status[name]["traceback"], flush=True)
        status_path.write_text(json.dumps(status, indent=1, default=str))
        print(f"== {name}: {status[name]['status']} ({(time.time() - t0) / 60:.0f} min)", flush=True)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        package(cfg, out)
    kit.lmk.close()
    weights = {k: cfg.models_dir / k for k in ("e4e_ffhq_encode.pt", "face_landmarker.task", "w600k_r50.onnx",
                                               "hyperstyle_ffhq.pt", "faces_w_encoder.pt")}
    src = {k: (cfg.models_dir / f).read_text().strip() for k, f in
           (("e4e", "e4e_source.txt"), ("hyperstyle", "hyperstyle_source.txt")) if (cfg.models_dir / f).exists()}
    write_run_meta(out, cfg.as_dict(), {k: v for k, v in weights.items() if Path(v).exists()},
                   {"stage": "stage7_final", "seconds": time.time() - t_start, "status": status, "sources": src})
    package(cfg, out)
    failed = [k for k, v in status.items() if isinstance(v, dict) and v.get("status") == "failed"]
    if failed:
        raise SystemExit(f"components failed: {failed} (see status.json); finished ones are in the zips")


if __name__ == "__main__":
    main()
