# RESULTS: everything the project measured, on one page (as of 2026-11-03)

Plain-language summary of every measured result, Stages 1–9. Each line gives the number, the comparison
(baseline) and the sample size.

Rules behind every number: test rules written *before* each run (`outputs/eval_plan.json`); dev photos to build,
test photos only evaluated, never trained or tuned on (several stages evaluated on the same test set, each with rules fixed in advance); people split so nobody is in both; 95 % confidence intervals by
resampling people ("subject bootstrap"); Holm correction when many things are tested at once.

---

## Summary

1. **It reproduces a real surgery result.** Given the amount each patient really changed, the simulated face lands
   within 0.4–0.7 mm of the real post-op photo on the operated feature (real before-to-after changes 0.9–1.9 mm; n = 436).
2. **Its changes move the face like real surgery.** For the nose, the simulated motion pattern matches real surgery
   twice as well as a standard warp (0.48 vs 0.24).
3. **It keeps the person.** HyperStyle rebuilds look like the real person in 100 % of faces (e4e: 42 %); warp edits
   in 100 % (n = 440).
4. **It checks itself.** Every edit is judged by tools it was not built with (second landmark detector, identity
   model, 3D face model, depth network); edits that failed those checks were removed from the product.

---

## Stage by stage

### Stage 1: can faces be measured reliably? (662 real pairs, all photos)
- 1,324/1,324 photos measured, 0 failures; 655 pairs pass the head-pose filter (≤ 10° difference).
- Noise floors measured for every feature (how much a number wobbles on the same photo).
- Strongest real surgical change: brow height +0.019 IPD [0.016, 0.022] (≈ +4.8 %).
- Real changes are small: only 27.3 % of pairs show a brow rise beyond realistic noise.
- Post-op photos smile more (p < 1e-6); expression explains 53 % of the mouth-width change → expression is a
  confound, reported, never hidden.
- Side-profile measures (nose projection etc.) excluded: not valid from front photos.

### Stages 2–4: the GAN and its "latent space ↔ feature" map
- e4e inversion keeps identity by FaceNet (0.876) but **not** by the stricter ArcFace (0.55; only 42–48 % of
  rebuilds look like the person) → the two judges disagree, so e4e images are not shown to users.
- 36,000 synthetic faces measured: the GAN's code predicts features on unseen faces (R² 0.45–0.84; middle layers
  4–7 carry most anatomy).
- One edit direction per feature, built to leave the other features alone; open-loop edits miss their size
  (gain 0.26–0.86) → the reason for the closed loop.
- 15 groups of duplicate photos found and 25 pairs excluded, so test people never appear in training.

### Stage 5: the closed-loop GAN editor (440 test faces, pre-registered primary endpoint)
- Success rate (right size, no side effects, still the same person), closed loop vs open loop vs a simple baseline:
  brow **0.87** vs 0.48 vs 0.02; eye opening 0.69 vs 0.20; jaw 0.50 vs 0.29; mouth 0.23 vs 0.10; nose 0.12 vs 0.07.
  Closed loop wins on every feature (Holm p = 0.001). Dev and test agree within 0.07.
- Independent landmark detector (FAN) confirms brow, mouth and jaw edits as real; **nose and eye opening not**
  (they move MediaPipe's points more than the anatomy).
- Landmark detectors under-report real changes (they see 16–94 % depending on feature): measured on photos with a
  known, built-in deformation.
- E7: shape of the simulated change vs the real surgical change on held-out pairs: GAN pattern 0.48 vs standard
  warp 0.24 for the nose (+0.24 [0.19, 0.29]), as good as a pattern learned from real surgery photos → warp mode uses it.

### Stage 6: warp mode (440 test faces × 4 surgical requests, 0 failures)
- Requested size met exactly for all (nose −3.5, brow +3.0, eyelid MRD1 +0.8, jaw −8.0 mm approx.).
- Other features unchanged (≤ 0.001 noise floors).
- Still the same person: ArcFace 0.96–0.99, 100 % above the threshold, vs GAN edits 0.47–0.54 (13–36 %).

### Stage 7: a better encoder? fine-tune on our data? (440 test faces)
- Looks like the real person (ArcFace): e4e 0.543 (42 %); **e4e fine-tuned on HDA 0.528 (32 %) → rejected**
  (fine-tuning improved its own training loss by 14 % but lost identity on the independent judge);
  **HyperStyle 0.793 (100 %)**, 0.8 s per photo on a GPU.
- Realism (KID, lower = more real): warp ≈ 0 < HyperStyle 14 < fine-tuned e4e 24 < e4e 34.
- HyperStyle edits: brow and mouth pass; jaw fails the pre-set margin → GAN mode = HyperStyle for brow and mouth.
  Brow success vs the real photo 0.88 vs 0.29 with e4e (post-hoc composite, labelled as such).
- HyperStyle halves feature drift of the rebuild (eye opening 1.05 → 0.52, mouth 0.99 → 0.54 noise floors):
  the "anatomy-preserving encoder" goal met without training a new one.
- PTI: best identity (0.86) but 69 s per photo; HyperStyle is the practical choice.
- Warp mode judged by FAN: right direction brow 98 %, jaw 98 %, nose 74 %; side effects small (0.25–0.43 noise floors).
- 3D face model (pose and expression removed): warp edits in the right direction 74–99.5 %; GAN eye-opening edits
  45–48 % (chance) → second sign that GAN eye edits are not real lid anatomy.
- Depth network: depth change concentrated in the operated region (jaw ≈ 100 %, nose 84–97 % above noise).

### Stage 8: can two detectors together rescue GAN nose and eye edits? (440 test faces, 2,640 edits)
- Pre-registered answer: **no**. "Anatomically confirmed" rate: brow reference 0.23; nose ≤ 0.023; eye ≤ 0.016.
- Eye edits are wrong in kind (FAN/MediaPipe ratio 0.19 vs 0.45–0.49 expected for a real change; 3D at chance);
  nose edits partly real but entangled. Brow behaves like real anatomy on every judge (ratio 0.84, inside [0.68, 1.27]).
- This confirms the product split: GAN mode = brow + mouth; nose, eye, jaw = exact warp mode.

### Stage 9 (E4): simulated vs REAL post-op photos (436 test pairs, 0 failures)
- No change (baseline): average landmark error vs the real post-op face 0.0266 IPD (≈ 1.7 mm).
- **Average-change prediction** (no patient information): closer than no change, +0.0004 IPD [0.0001, 0.0007],
  p = 0.003, 56 % of pairs closer → passes the pre-set rule; small gain (≈ 1.5 %). Nose and mouth width clearly closer.
- **With the real change size:** typical error on the operated feature vs the real post-op photo: brow 0.4 mm
  (unchanged photo: 1.6 mm), nose 0.6 (0.9), jaw 0.6 (1.7), mouth 0.7 (1.9); 73–91 % of pairs closer; all p < 0.001.
- Whole-face error stays ≈ 1.6 mm even then: pose, expression and lighting differ between the two photos.
- So the simulator draws a given change accurately; an average change alone barely improves on no change.

### Photo check (app)
- Two independent detectors (MediaPipe vs FAN) must agree on the face before an edit.
- On held-out photos: false alarms on clean photos 6.7 %; covered eye or mouth caught 87–88 %; landmark drift
  > 2 % IPD caught 76 %.

### App speed
- Warp mode: ≈ 1.5 s for the edit itself; ≈ 2.5 s per request through the app (after a ≈ 1 min first-request warm-up), laptop CPU.
- GAN mode: ≈ 1 s on a GPU (T4); 74–184 s on 2 CPU threads.

---

## What is NOT claimed
- Per-patient prediction of how much a face will change.
- Millimetre accuracy: mm are approximate (±6 %), from an average eye distance.
- Side-profile measures from front photos.
- GAN nose and eye edits (failed independent checks; warp mode only).
