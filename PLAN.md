# ReLiFT-6DoF — Implementation Plan

## High-level goal

Robust 6-DoF tracking on reflective objects by exploiting the light field.
Analogy: BundleSDF but instead of a NeRF canonical model we accumulate a
**canonical surface light field** (point cloud + per-point intrinsic diffuse
color + environment map), and we use **relighting** to bridge the illumination
change between the canonical view and the current view during pose refinement.

## Key assumptions (to relax later)

- Reflectivity α is **known** (given per-sequence).
- Depth from GT or synthetic source (Depth-Anything-V3 later).
- Segmentation mask from GT (Cutie later).

## Algorithm per frame

```
Frame 0
  ├─ LF → SurfaceLF(pc, images)          # builds per-point diffuse color + env map
  ├─ diffuse_midview = rasterize_diffuse(slf)
  ├─ canonical_model ← from_frame0(slf, pose0=gt_pose[0])
  └─ est_poses[0] = gt_pose[0]

Frame t > 0
  ├─ LF → SurfaceLF(pc_t, images_t)      # per-frame reflection separation
  ├─ diffuse_midview_t = rasterize_diffuse(slf_t)
  │
  ├─ COARSE POSE (α-weighted mix):
  │    if α ≥ 0.1:
  │      T_loftr = loftr_coarse(diffuse_{t-1}, diffuse_t, depth_{t-1}, depth_t, K)
  │    if α < 0.9:
  │      T_geom  = icp_coarse(pc_{t-1}, pc_t, est_poses[-1])
  │    coarse_pose = weighted_merge(T_loftr, T_geom, α)   # or just pick best
  │
  ├─ REFINEMENT:
  │    image_canon, depth_canon, _ = canonical_model.rasterize(coarse_pose)
  │    # image_canon is re-lit with current env map
  │    refined_pose = gradient_refine(
  │        rendered=(image_canon, depth_canon),
  │        target=(diffuse_t, depth_t),
  │        init=coarse_pose)
  │
  ├─ FUSION:
  │    canonical_model.fuse_frame(slf_t, refined_pose)
  │    # ↳ transforms new points to object frame, voxel-merges
  │
  └─ est_poses[t] = refined_pose
```

## Module map

| File | Role |
|------|------|
| `src/dataset.py` | LFDataset (unchanged) |
| `surface_lf.py` | SurfaceLF — per-frame reflection separation + env map + SH fitting |
| `reflection_separation.py` | Optimisation core used by SurfaceLF (unchanged) |
| `loftr_wrapper.py` | LoFTR runner (unchanged) |
| `loftr_baseline.py` | Helpers: `_backproject`, `_procrustes`, `_ransac_relative_pose` (reused) |
| `icp.py` | ICP coarse pose (reused) |
| `loss.py` | `refine_pose` photometric refinement (reused) |
| **`diffuse_view.py`** | NEW — rasterize_diffuse: render diffuse-only image from SurfaceLF for LoFTR |
| **`canonical_model.py`** | NEW — CanonicalModel: accumulates point cloud + env map, rasterises with relighting |
| **`coarse_pose.py`** | NEW — LoFTR + ICP coarse pose with α-mixing |
| **`main.py`** | NEW — main tracking loop (replaces main_old.py) |

## CanonicalModel internals

Stores everything in **object frame** (defined by first-frame pose):

```python
class CanonicalModel:
    points_obj: Tensor  # [N, 3]
    normals_obj: Tensor # [N, 3]
    diffuse_colors: Tensor  # [N, 3]  (intrinsic, view-independent)
    pc_scales: Tensor   # [N, 3]
    environment_map: Tensor  # [H_env, W_env, 3]
    rig: SurfaceLFRig
    alpha: float        # separation_alpha = 1 - reflectivity
```

`rasterize(pose_t)`:
1. Transform points to camera frame: `p_cam = R_t @ p_obj + t_t`
2. Transform normals: `n_cam = R_t @ n_obj`
3. Sample env map using reflected dirs from LF rig cameras
4. Mix: `relit = α * diffuse + (1-α) * env_sample`
5. Fit SH → `batch_rasterize` with identity camera

`fuse_frame(slf_t, pose_t)`:
1. `new_pts_obj = inv(R_t) @ (points_cam_t - t_t)`
2. `new_norms_obj = inv(R_t) @ normals_cam_t`
3. Update env map: running exponential average
4. Concatenate + voxel_downsample to keep N bounded (~50k pts)

## Coarse pose α-mixing strategy

- α = 1.0 (pure diffuse): use LoFTR only
- α = 0.0 (pure reflective): use ICP only  
- 0 < α < 1: try LoFTR first; if it returns fewer than MIN_MATCHES inliers,
  fall back to ICP; otherwise blend

LoFTR → pose:
1. Run on (diffuse_{t-1}, diffuse_t)
2. Back-project matched pixels to 3D with depth (reuse `loftr_baseline._backproject`)
3. RANSAC Procrustes → T_rel (reuse `loftr_baseline._ransac_relative_pose`)

## Pose refinement on diffuse image

Same as existing `refine_pose()` in `loss.py`, but:
- `target_rgb` = `diffuse_t` (not raw middle view)
- rendered image = canonical model re-lit at candidate pose

## Environment map accumulation

Exponential moving average of env maps across all frames:
```python
env_map_new = (1 - β) * env_map_old + β * env_map_curr
```
where β = `env_fusion_alpha` (currently 0.6 in SurfaceLF).

## Voxel downsampling

Grid-based: hash each point to (ix, iy, iz) = floor(p / voxel_size).
Average colors + normals per voxel. Max canonical model size ~ 50k points.

## Files NOT changed

- `reflection_separation.py` — building block, keep as-is
- `surface_lf.py` — add `rasterize_diffuse()` method only
- `loftr_baseline.py` — helper functions reused directly
- `icp.py` — reused directly
- `loss.py` — `refine_pose` reused directly

## Status

- [ ] `diffuse_view.py`
- [ ] `canonical_model.py`
- [ ] `coarse_pose.py`
- [ ] `main.py`
