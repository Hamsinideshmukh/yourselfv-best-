"""Yourself (Best) — facial-surgery goal visualisation (research demo, not a medical device).

Two modes:
1. Warp mode (always on, CPU): the person's own photo, moved by an exactly computed landmark field whose shape is
   the GAN's learned deformation pattern (best or tied prior on held-out real surgery, E7). Every requested change is
   met exactly on the face's own landmarks; other measured features are held fixed; the field is downloadable as a
   reproducible "recipe" in approx. mm.
2. GAN mode (on by default; ~1 s on a GPU, about 1-3 min on a free 2-core CPU Space): HyperStyle inversion + closed-loop latent edit with the
   evaluated Stage-4b directions. Offered only for brow height and mouth width (docs/RESULTS.md).
Privacy: photos are processed in memory; Gradio's temporary upload cache is deleted every few minutes; nothing is
logged or stored by this app. Every result is a simulation, not a prediction.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd

from medsg.judges import face_box
from medsg.landmarks import FaceNotFound, Landmarker, ensure_task_file
from medsg.transfer import COUPLED, IPD_MM, NAMES, Transfer, metric_values, warp_photo

HERE = Path(__file__).resolve().parent
MODELS = Path(os.environ.get("MEDSG_MODELS", HERE / "models"))
MAX_SIDE = 1280
LMK = Landmarker(ensure_task_file(MODELS))
TRANSFER = Transfer()
try:                                    # photo check: a second, independent landmark detector (FAN) on the input
    from medsg.judges import FAN68
    from medsg.quality import PhotoCheck
    CHECK, CHECK_NOTE = PhotoCheck(FAN68("cpu", face_detector="blazeface")), ""
except Exception as err:                # noqa: BLE001  shown to the user, never replaced by a fake "OK"
    CHECK, CHECK_NOTE = None, f"*Photo check unavailable on this server ({type(err).__name__}).*"

DISCLAIMER = ("**Simulation, not a prediction.** For clinician-guided consultation with adults only. Sizes are approximate "
              "(mm via an average inter-pupil distance of 63.4 mm, ±6 % per person). Real outcomes depend on "
              "anatomy, technique and healing.")

# key, label, slider min, max (approx. mm), published reference (lo, hi) in mm or None, reference text
SLIDERS = [
    ("nose_width", "Nose: alar-junction width", -6.0, 3.0, (-4.4, 0.0),
     "alar base resection: −3.3 to −4.4 mm (n = 120; n = 25)"),
    ("brow_medial", "Brow height: medial", -3.0, 6.0, (0.0, 3.25), "endoscopic brow lift, medial: +3.25 mm (meta-analysis)"),
    ("brow_central", "Brow height: central", -3.0, 6.0, (0.0, 3.86), "endoscopic brow lift, central: +3.86 mm (meta-analysis)"),
    ("brow_lateral", "Brow height: lateral", -3.0, 6.0, (0.0, 4.35), "endoscopic brow lift, lateral: +4.35 mm (meta-analysis)"),
    ("mrd1", "Upper eyelid: MRD1 (iris centre to lid margin)", -1.5, 2.0, (0.0, 0.92),
     "upper blepharoplasty: MRD1 +0.7 to +0.9 mm"),
    ("jaw_width", "Jaw width (lower jaw)", -14.0, 4.0, (-13.0, 0.0), "mandibular angle reduction: −6 to −13 mm"),
    ("mouth_width", "Mouth width", -4.0, 4.0, None, "no surgical reference collected"),
    ("lower_face_height", "Lower face height (nose base to chin)", -6.0, 6.0, None, "no surgical reference collected"),
]
LABEL = {k: lab for k, lab, *_ in SLIDERS}
# App couplings: features allowed to move with a target. Stricter than the evaluated default (transfer.COUPLED):
# jaw-angle reduction should not move cheekbone width, so face_width is protected when editing the jaw.
APP_COUPLED = {**COUPLED, "jaw_width": [], "face_width": []}
FEATURE_NAME = {**{k: lab for k, lab, *_ in SLIDERS}, "brow_height": "Brow height (peak)", "palpebral_height":
                "Eye opening", "palpebral_width": "Eye width", "intercanthal": "Inner-eye-corner distance",
                "nose_length": "Nose length", "face_width": "Cheekbone width", "mrd2": "Lower lid (MRD2)"}
GAN_LABEL = {"brow_height": "Brow height", "mouth_width": "Mouth width", "jaw_width": "Jaw width (lower jaw)"}


def _photo_check(rgb: np.ndarray, L: np.ndarray) -> str:
    """Markdown line: do MediaPipe and FAN agree on this photo? (warns, never blocks)"""
    if CHECK is None:
        return CHECK_NOTE
    try:
        return CHECK(rgb, np.c_[L, np.zeros(len(L))], face_box(L))["text"]
    except ValueError as err:
        return f"**Photo check: ⚠ could not run** ({err})."


def _prepare(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if img is None:
        raise gr.Error("Please upload a photo or pick an example.")
    rgb = np.ascontiguousarray(img[..., :3]).astype(np.uint8)
    s = MAX_SIDE / max(rgb.shape[:2])
    if s < 1:
        rgb = cv2.resize(rgb, (round(rgb.shape[1] * s), round(rgb.shape[0] * s)), interpolation=cv2.INTER_AREA)
    try:
        obs = LMK(rgb)
    except FaceNotFound:
        raise gr.Error("No face found. Use a front-facing, well-lit photo with one face.")
    if obs.n_faces > 1:
        gr.Warning("More than one face found; editing the largest one.")
    if abs(obs.yaw) > 12 or abs(obs.pitch) > 15:
        gr.Warning(f"Head is turned (yaw {obs.yaw:.0f}°, pitch {obs.pitch:.0f}°). Frontal photos give valid measurements.")
    return rgb, obs.lm_px[:, :2].astype(np.float64)


def _flags(key: str, mm: float) -> str:
    ref = next(s for s in SLIDERS if s[0] == key)
    lo_hi, text = ref[4], ref[5]
    if lo_hi is None:
        return "no published reference"
    lo, hi = lo_hi
    if mm < lo - 1e-9 or mm > hi + 1e-9:
        return f"⚠ outside the reported clinical range ({text})"
    return f"within reported range ({text})"


def _arrows(rgb: np.ndarray, L: np.ndarray, field: np.ndarray, scale: float = 5.0) -> np.ndarray:
    out = rgb.copy()
    t = max(1, round(max(rgb.shape[:2]) / 700))
    norms = np.linalg.norm(field, axis=1)
    keep = norms >= 0.3 * norms.max() if norms.max() > 0 else np.zeros(len(L), bool)   # main movements only
    for p, d, k in zip(L, field, keep):
        if not k or np.linalg.norm(d) * scale < 1.0:
            continue
        a, b = tuple(np.round(p).astype(int)), tuple(np.round(p + scale * d).astype(int))
        cv2.arrowedLine(out, a, b, (255, 210, 0), t, cv2.LINE_AA, tipLength=0.3)
    return out


def run_warp(img, *mm_values):
    rgb, L = _prepare(img)
    req_mm = {k: float(v) for (k, *_), v in zip(SLIDERS, mm_values) if abs(float(v)) > 1e-9}
    try:
        e = TRANSFER.edit_multi(L, {k: v / IPD_MM for k, v in req_mm.items()}, couplings=APP_COUPLED)
    except ValueError as err:
        raise gr.Error(f"These requests cannot be set independently for this face: {err}")
    out = warp_photo(rgb, L, e.field_px)
    check = _photo_check(rgb, L)
    lines = ["| Feature | Requested (≈ mm) | Achieved, exact (≈ mm) | Re-measured by MediaPipe (≈ mm) | Reference |",
             "|---|---|---|---|---|"]
    try:
        meas = dict(zip(NAMES, metric_values(LMK(out).lm_px[:, :2]) - metric_values(L)))
    except (FaceNotFound, ValueError):
        meas = {}
    for k, mm in req_mm.items():
        ach = (e.after[k] - e.before[k]) * IPD_MM
        m = meas.get(k, np.nan) * IPD_MM
        lines.append(f"| {LABEL[k]} | {mm:+.2f} | {ach:+.2f} | {m:+.2f} | {_flags(k, mm)} |")
    coupled = sorted(set().union(*[APP_COUPLED.get(k, []) for k in req_mm]) - set(req_mm)) if req_mm else []
    protected = [n for n in NAMES if n not in req_mm and n not in coupled]
    report = "\n".join(lines) if req_mm else "_Move a slider to request a change. All sliders at 0 = the original photo._"
    if req_mm:
        worst = max(protected, key=lambda n: abs(e.after[n] - e.before[n]))
        report += (f"\n\n**Held fixed** ({len(protected)} other measured features, exact): largest change "
                   f"{abs(e.after[worst] - e.before[worst]) * IPD_MM:.2f} mm ({FEATURE_NAME.get(worst, worst)}).")
        if coupled:
            report += "\n\n**Moves together with the request** (anatomically linked, by design): " + ", ".join(
                f"{FEATURE_NAME.get(n, n)} {(e.after[n] - e.before[n]) * IPD_MM:+.2f} mm" for n in coupled) + "."
        report += ("\n\n*Re-measurement note:* landmark detectors under-report local shape changes (we measured "
                   "16–94 % depending on region and size), so the MediaPipe column is smaller than the exact change "
                   "by design. The exact column is what the image shows.")
    ipd_px = float(np.linalg.norm(L[468] - L[473]))
    recipe = pd.DataFrame({"landmark": np.arange(478), "x_px": L[:, 0], "y_px": L[:, 1],
                           "dx_px": e.field_px[:, 0], "dy_px": e.field_px[:, 1],
                           "dx_mm_approx": e.field_px[:, 0] * IPD_MM / ipd_px,
                           "dy_mm_approx": e.field_px[:, 1] * IPD_MM / ipd_px})
    meta = pd.DataFrame({"feature": list(e.before), "before_ipd_units": list(e.before.values()),
                         "after_ipd_units": [e.after[k] for k in e.before]})
    path = Path(tempfile.mkdtemp()) / "edit_recipe.csv"
    with open(path, "w") as f:
        f.write("# Yourself (Best) edit recipe: MediaPipe 478-landmark motion; mm approx (IPD 63.36 mm, ±6 %)\n")
        f.write("# requests_mm: " + ", ".join(f"{k}={v:+.2f}" for k, v in req_mm.items()) + "\n")
        recipe.to_csv(f, index=False)
        f.write("\n")
        meta.to_csv(f, index=False)
    return (rgb, out), _arrows(rgb, L, e.field_px), check + "\n\n" + report + "\n\n" + DISCLAIMER, str(path)


# ----------------------------------------------------------------------------- GAN mode (CPU or GPU)
# HyperStyle inversion (Stage 7: identity 0.79 vs e4e 0.54, 100 % above the real-surgery threshold, 0.8 s on a T4) +
# the evaluated closed-loop latent edit. Brow and mouth only: the features that passed Stage 7's pre-set editability
# margin with HyperStyle (jaw failed; eye opening and nose are not confirmed by independent judges) -> warp mode.
GAN_ENABLED = os.environ.get("MEDSG_GAN_MODE", "1") != "0"     # set MEDSG_GAN_MODE=0 for a warp-only server
# Official HyperStyle weights (README Google Drive links); any mirror must match these SHA-256 hashes.
HS_FILES = {"hyperstyle_ffhq.pt": "9d4c627724e9f6b7f87f13a03b71731a4bb537875271cb46662be08a9885c364",
            "faces_w_encoder.pt": "3cb757ae38268b08db8235b4a3d42c98c74428a817b619338848f277cd23e715"}
if GAN_ENABLED:
    import hashlib

    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download

    from medsg.align import align_ffhq
    from medsg.editor import GAN_TARGETS, Directions, secant_edit, target_index
    from medsg.gan import to_uint8
    from medsg.hyperstyle import HyperStyleGAN
    from medsg.measure import measure_avg
    try:
        import spaces
        gpu = spaces.GPU(duration=60)
    except ImportError:
        def gpu(f):
            return f

    def _weights(name: str) -> Path:
        """Local file (MEDSG_MODELS) or the Hugging Face mirror; refused unless its SHA-256 is the official one."""
        f = MODELS / name
        if not f.exists():
            f = Path(hf_hub_download("trysem/DualStyleGANplus", f"models/hyperstyle/{name}"))
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 24), b""):
                h.update(block)
        if h.hexdigest() != HS_FILES[name]:
            raise RuntimeError(f"{name}: SHA-256 differs from the official HyperStyle release; refusing to load")
        return f
    DEVICE = "cuda" if (os.environ.get("SPACES_ZERO_GPU") or torch.cuda.is_available()) else "cpu"
    HS = HyperStyleGAN(HERE / "vendor" / "hyperstyle", _weights("hyperstyle_ffhq.pt"), _weights("faces_w_encoder.pt"),
                       Path(tempfile.gettempdir()) / "medsg_ext", DEVICE)
    DIRS = Directions()

    @gpu
    def _gan_core(aligned: np.ndarray, target: str, steps_mdc: float):
        lmk = Landmarker(ensure_task_file(MODELS))          # own instance inside the GPU worker
        x = torch.from_numpy(aligned).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(DEVICE)
        code, deltas = HS.invert(F.interpolate(x, size=(256, 256), mode="area"))
        rend = HS.for_face(deltas)

        def render(a: float) -> np.ndarray:
            w = code + a * torch.from_numpy(DIRS.D[target]).to(DEVICE)[None]
            with torch.no_grad():
                return to_uint8(F.interpolate(rend.synth(w), size=(512, 512), mode="area"))[0]
        base = render(0.0)
        m0, _ = measure_avg(base, lmk)
        if m0 is None:
            lmk.close()
            return base, base, [], "measure_fail", np.nan
        r = steps_mdc * DIRS.mdc[target]
        cache = {}

        def rm(a: float):
            cache[a] = render(a)
            m, _ = measure_avg(cache[a], lmk)
            return None if m is None else float((m - m0)[target_index(target)])
        recs, status = secant_edit(rm, r, 0.5 * DIRS.mdc[target], DIRS.a_cap[target])
        lmk.close()
        final = cache[recs[-1][0]] if recs else base
        return base, final, recs, status, r

    def run_gan(img, target_label, steps_mdc):
        rgb, L = _prepare(img)
        check = _photo_check(rgb, L)
        target = {GAN_LABEL[t]: t for t in GAN_TARGETS}[target_label]
        aligned, _ = align_ffhq(rgb, np.c_[L, np.zeros(len(L))], 1024)
        base, final, recs, status, r = _gan_core(aligned, target, float(steps_mdc))
        ach = recs[-1][1] if recs else np.nan
        rep = check + "\n\n" + (
            f"Request: {steps_mdc:+.1f} noise-floor units = {r:+.4f} IPD (MediaPipe-measured). "
            f"Achieved (MediaPipe, averaged): {ach:+.4f}. Loop: {status} after {len(recs)} render(s).\n\n"
            "Rendered with HyperStyle (in our held-out tests, edits kept identity vs the real photo: ArcFace median 0.78, "
            "100 % above the real-surgery threshold). GAN-mode sizes are in *measured* units (landmark detectors "
            "under-report true change); use the warp mode for exact-geometry requests in mm.\n\n" + DISCLAIMER)
        return (base, final), rep


# ----------------------------------------------------------------------------- UI
EXAMPLES = sorted(str(p) for p in (HERE / "examples").glob("*.jpg"))
with gr.Blocks(title="Yourself (Best) — surgical goal visualisation (research)", delete_cache=(180, 300)) as demo:
    gr.Markdown("# Yourself (Best)\nGoal visualisation for facial-surgery consultations — a research demo. "
                "Upload a front-facing photo (or use a synthetic example face), request changes in approximate "
                "millimetres, and get the edited photo plus an exact, reproducible recipe of what moved.\n\n"
                + DISCLAIMER + "\n\n*Privacy:* photos are processed in memory; temporary files are deleted "
                "within minutes; nothing is logged or kept.")
    with gr.Tab("Warp mode (exact geometry)"):
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(label="Photo (front-facing)", type="numpy", sources=["upload", "webcam"])
                if EXAMPLES:
                    gr.Examples(EXAMPLES, inputs=inp, label="Synthetic example faces (GAN-generated, not real people)")
                sliders = [gr.Slider(lo, hi, value=0.0, step=0.1, label=f"{lab} (≈ mm)")
                           for _, lab, lo, hi, *_ in SLIDERS]
                btn = gr.Button("Apply", variant="primary")
            with gr.Column(scale=1):
                out_cmp = gr.ImageSlider(label="Original ↔ edited (drag)", type="numpy")
                out_arrows = gr.Image(label="Main movements (arrows ×5 length; smaller motions hidden)", type="numpy")
                out_rep = gr.Markdown()
                out_file = gr.File(label="Edit recipe (CSV)")
        btn.click(run_warp, [inp, *sliders], [out_cmp, out_arrows, out_rep, out_file])
    if GAN_ENABLED:
        with gr.Tab("GAN mode (generator re-render; slow on CPU)"):
            with gr.Row():
                with gr.Column():
                    ginp = gr.Image(label="Photo (front-facing)", type="numpy", sources=["upload", "webcam"])
                    gt = gr.Dropdown([GAN_LABEL[t] for t in GAN_TARGETS], value=GAN_LABEL["brow_height"],
                                     label="Feature (those that passed the held-out tests; others: warp mode)")
                    gs = gr.Slider(-3, 3, value=2, step=0.5, label="Requested change (noise-floor units; ±2 was evaluated)")
                    gr.Markdown("*On a free CPU server one request takes about 1–3 minutes (inversion + up to "
                                "6 renders, each measured). Please wait; ~1 s on a GPU.*")
                    gbtn = gr.Button("Render", variant="primary")
                with gr.Column():
                    gout = gr.ImageSlider(label="GAN rebuild ↔ edited", type="numpy")
                    grep = gr.Markdown()
            gbtn.click(run_gan, [ginp, gt, gs], [gout, grep])
    else:
        gr.Markdown("*GAN mode is switched off on this server (MEDSG_GAN_MODE=0).*")

if __name__ == "__main__":
    demo.launch()
