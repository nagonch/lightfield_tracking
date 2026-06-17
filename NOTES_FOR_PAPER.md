# ReLiFT-6DoF — methods & findings (for the ACCV paper / thesis)

This documents the **final** tracking pipeline (`main.py`) and the design decisions
behind it, with the rationale you'll want when writing the method section. Numbers
are GT-depth; synth depth is out of scope for now.

## Pipeline overview (per frame t > 0)

```
LF frame t ─► SurfaceLightField (per-point pc, normals, multi-view colours, SH)
           ─► Reflection separation ─► diffuse central view  +  env map  +  alpha
                                          │
  COARSE  (contribution 1): reflectivity-aware tracking
     reflectivity 1-alpha blends   LoFTR(diffuse_{t-1}, diffuse_t)   (alpha high)
                              with  colored-ICP(pc_{t-1}, pc_t)       (alpha low)
                                          │
  FINE   (contribution 2): diffuse photometric refinement  [alpha > 0.2]
     gsplat-render prev SLF's *diffuse* appearance at a candidate pose, align to
     the curr SLF's diffuse render; optimise 6-DoF (rotation + small translation)
                                          │
  FINAL  (explored): relight refinement  [alpha < 0.85, default OFF]
     same, but render alpha*diffuse + (1-alpha)*env[reflect]; the reflection moves
     with the pose. Loss landscape is correct but its accuracy floor loses to
     GT-depth ICP — see "Relighting" below.
```

The two stages of refinement are a **per-frame overlay**: they correct the
*reported* pose for frame t, but the **coarse pose is what is fed forward** as the
next frame's tracking anchor. This is essential (see Design decision 1).

## Contribution 1 — reflection separation + reflectivity-aware tracking

A surface light field (5×5 sub-aperture views back-projected to a shared point
cloud) is decomposed per frame into a **view-independent diffuse colour per point**
and a **shared reflected environment map**, by fitting
`obs[p,v] = alpha·diffuse[p] + (1-alpha)·env(reflect(view_v, normal_p))`
(`reflection_separation.py`). LoFTR then matches the *diffuse* views — robust where
matching the raw (reflection-corrupted) views fails — and at high reflectivity the
near-mirror diffuse is too weak for LoFTR so tracking falls back to geometry-only
ICP. The blend weight is the reflectivity `1-alpha`.

This is the big win over the LoFTR-on-raw-views baseline: at full reflectivity the
baseline collapses (cube_1.0 ~87° rotation) while separation+ICP stays ~few degrees.

## Contribution 2 — diffuse photometric refinement ("fine pose")

`src/photometric.py::refine_pose_photometric`. The prev-frame SLF is rendered with
gsplat at a candidate pose and aligned, in image space, to the current SLF's render
(`SurfaceLightField.render_relit`, fully differentiable w.r.t. gsplat means and the
analytic per-point colour, so the pose gradient is exact — no scatter aliasing, no
Lucas-Kanade detach). Loss = ROI-weighted photometric MSE + depth-anchoring L1.

Parameterisation (the honest one): `pose = [dR·R0 | t0 + dt]`, optimise the 6-D
rotation `dR` and translation `dt`, **report exactly what is optimised**. Coarse-to-
fine over render scales (0.5, 1.0).

Result: a small but **consistent** improvement on all metrics where there is a
diffuse signal (objects especially, where the coarse has more error to fix).

## Design decisions & rationale (these are the load-bearing ones)

1. **Refinement is an overlay; the coarse pose is fed forward, not the refined
   pose.** A per-frame photometric correction has a tiny systematic bias; recycling
   it as the next frame's anchor compounds geometrically (measured: 0.9°→6.7° over a
   single diffuse sequence). Feeding the *coarse* pose forward makes the refined
   trajectory provably no worse than the coarse backbone, and the per-frame ADD/Rot
   metric still sees the correction. (`main.track_sequence`)

2. **The diffuse stage is gated off on reflective frames (alpha ≤ 0.2).** A mirror
   has no view-independent diffuse signal, so its separated diffuse colour is
   meaningless and its loss basin is displaced 5–10° from GT (measured,
   `diag_relight.py`). Reflective frames are left to the coarse (ICP) pose.

3. **Splat footprint `scale_factor` 0.5 → 1.0.** Half-pixel isotropic Gaussians left
   inter-splat gaps → aliasing (the horizontal seams you saw) and a noisy loss. One-
   pixel footprints overlap: render mismatch at GT dropped |diff| 3.2→1.7 / 255 and
   the loss minimum now sits *exactly* at GT on all 6 DoF (`analyze_photometric.py`).

4. **Alpha is pinned to the known reflectivity, not estimated, for the main
   results.** The view-variance estimator (`estimate_alpha`) is reliable at low/mid
   reflectivity but unstable per-sequence at the mirror extreme (sticks at ~0.5 for
   some cube_1.0 seqs, env-contrast dependent); alpha≈0.5 blends 50% LoFTR into a
   near-mirror track and regresses the coarse (5.3°→14° agg). The estimator is kept
   (`main.PIN_ALPHA=False`) as an ablation. (For a "no known reflectivity" claim it
   needs a more robust estimator — future work.)

## Relighting ("final pose") — what we found

The relight idea is sound and we show it: the **relit loss landscape is correct** —
swept about GT it bottoms ≤1.5° from truth and drops the loss ~45% (`diag_relight.py`),
i.e. the moving reflection genuinely encodes pose. **But** on GT depth it is
net-negative and is **OFF by default**: its accuracy floor (env-map resolution +
PCA-normal + alpha error ≈ 1.5–3°) is *worse* than GT-depth ICP, and its convergence
basin is only ~±12°. So it nudges already-good coarse poses (cube_1.0 1.7–2.7°)
toward its bias and cannot reach far-off ones (tomato 40°). A noise-floor gate
(`relight_min_correction_deg`) keeps it harmless when enabled.

Honest framing for the paper: present relight's correct loss landscape as evidence
the cue is valid, and its limited benefit as an **appearance-model-accuracy
limitation** — it would help where the geometric coarse is poor (worse depth, RGB-
only, stronger symmetry). This is the natural future-work hook.

## Results (GT depth, split-average ADD-AUC↑ / Rot°↓; `compare_eval.py`)

Regenerated with the CURRENT code: coarse = separation+blend (contribution 1),
refined = + diffuse photometric refine (contribution 2). LoFTR = raw-view baseline.

CUBE (complete):
| split    | LoFTR ADD | coarse ADD | refined ADD | LoFTR Rot | coarse Rot | refined Rot |
|----------|-----------|------------|-------------|-----------|------------|-------------|
| cube_0.0 |   0.921   |   0.905    |   0.907     |   2.45    |   3.14     |   3.03      |
| cube_0.5 |   0.883   |   0.894    |   0.897     |   3.76    |   2.85     |   2.78      |
| cube_0.7 |   0.798   |   0.875    |   0.876     |   6.59    |   3.07     |   3.06      |
| cube_1.0 |   0.067   |   0.545    |   0.547     |  87.42    |  13.79     |  13.65      |

Reading: refined ≥ coarse on every split (both ADD and Rot — contribution 2 never
hurts and consistently helps). The pipeline crushes LoFTR as reflectivity rises
(cube_1.0: 0.067→0.547 ADD, 87°→13.7°; LoFTR collapses on mirrors, we hold ~14°).
Only on the pure-diffuse cube_0.0 are we just under LoFTR (0.907 vs 0.921) — running
reflection separation on an already-diffuse textured cube slightly softens the
texture LoFTR keys on. (OBJECTS table: regenerating; per-sequence: refine helps the
coarse on objects_0.0 — see exp_traj logs.)

## Tools (kept for further experiments)

- `exp_traj.py` — full-trajectory refine-OFF vs refine-ON with real ADD/Rot/ATE eval.
- `diag_relight.py` — relit-loss landscape sweep about GT (where does the basin sit?).
- `analyze_photometric.py` — per-DoF loss-landscape + source/target alignment render.
- `diag_diffuse.py` — per-iteration rotation-error curve (START_GT=1 = pure loss bias).
- `exp_strategy.py` / `exp_quick.py` — per-frame parameterisation/config sweeps.
- viser viewer (`PhotometricRefineViewer`, `main.ENABLE_VIS=True`) — live splat +
  source/target/loss frustums for eyeballing convergence.
