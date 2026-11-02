---
title: Yourself (Best)
emoji: 🔬
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
license: other
short_description: Facial-surgery goal visualisation (research demo)
---

# Yourself (Best) — facial-surgery goal visualisation (research demo)

**Simulation, not a prediction. Not a medical device.** For clinician-guided consultation with adults, and research only.

**Warp mode (CPU):** request changes in approximate millimetres; the photo is moved by an exactly computed field
whose shape is a StyleGAN2-learned deformation pattern. On held-out real surgery pairs this pattern matched real
nose changes twice as well as a hand-made warp (shape agreement 0.48 vs 0.24). Requested changes are met exactly
on the face's own landmarks, other measured features stay fixed, and the field is downloadable as a CSV recipe.

**GAN mode (slow on this free CPU server: about 1–3 minutes per render request; ~1 s on a GPU):** HyperStyle inversion + closed-loop latent edits for brow height and mouth width. On 440
held-out faces, edits met the request within noise, without side effects and still looking like the real person
(ArcFace above the real pre-vs-post-surgery threshold; a composite defined after the run) for 88 % (brow) and 36 %
(mouth) of requests, vs 29 % and 7 %
with plain e4e inversion. Jaw, eye opening and nose are offered in warp mode only (they did not pass the GAN tests).

**Photo check:** before any edit, a second, independent landmark detector (FAN) checks the photo; if the two
detectors disagree more than on typical clinical photos, the app warns (6.7 % false alarms; 87–88 % of photos with a
covered eye or mouth flagged, in held-out tests).

**Privacy:** photos are processed in memory; Gradio's temporary files are deleted within minutes; nothing is logged.

**Licences:** StyleGAN2 weights (used by HyperStyle) are NVIDIA Source Code License-NC (non-commercial); HyperStyle
code MIT; MediaPipe Apache-2.0; face-alignment (FAN) BSD-3.
No photo from the research dataset (HDA, research-only) is included. Example faces are GAN-generated, not real people.
