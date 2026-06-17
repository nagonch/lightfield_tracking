# Shading in ReLiFT-6DoF renders

## Summary (current)

The diffuse photometric stage stores **intrinsic albedo** and **re-applies SoftPhong
shading from the surface normals at each candidate pose** — reproducing the dataset
renderer (`ycbv-eoat-lf/render.py`).  This is `mode="diffuse_shaded"` in
`SurfaceLightField.render_relit` (`src/shading.py` holds the model); it is the default
diffuse stage (`RefineConfig.diffuse_mode`).  A rotation therefore re-shades the
surface instead of rigidly carrying a frozen shaded colour.

This replaces the earlier "no shading" bake.  The change was made because the bake is
physically wrong under rotation and was visibly so:

| test (objects bleach0)                         | baked `diffuse` | `diffuse_shaded` |
|------------------------------------------------|-----------------|------------------|
| prev@GT vs **actual** curr, 6.5° motion        | 7.64e-3 MSE     | 7.49e-3          |
| prev@GT vs **actual** curr, 45° motion         | 2.41e-2 MSE     | **8.96e-3**      |
| end-to-end refine rot-err (small-motion sweep) | 0.717° mean     | 0.628° mean      |

The benefit grows with rotation (the bake freezes frame-0 shading); near GT it is
neutral-to-slightly-better and **never worse**.  The un-shade → re-shade round-trips
exactly at the canonical pose (per-point MSE ~4e-15), so the central view is unchanged
and only a pose change alters the result.

## The shading model (matches the renderer)

PyTorch3D `SoftPhongShader` + one `PointLights` + default `Materials` (shininess 64):

```
observed = (ambient + diffuse·relu(n·l))·albedo + specular·relu(v·r)^shininess
  ambient=[.5,.5,.5]  diffuse=[.5,.4,.25]  specular=[.5,.45,.35]
  l = normalize(light_pos - point)   # point light, world [0,0.1,0]
  v = normalize(cam_pos - point)     # camera at origin
  r = -l + 2(n·l)n                   # gated by n·l>0
```

Recovery (un-shade) and re-shade are exact inverses at a fixed `(point, normal)`:

```
albedo   = (observed_raw - specular) / (ambient + diffuse·relu(n·l))
rendered = albedo·(ambient + diffuse·relu(n'·l)) + specular(n',v')   # at the new pose
```

Two correctness details that matter:
- **Colour space.** Shading was applied in the raw 8-bit (sRGB-encoded) space the PNG
  was written in.  So un-/re-shade in that space (`linear_to_srgb` ⇄ `srgb_to_linear`
  around the math); gsplat compositing and the photometric loss stay linear.
- **Normal orientation.** PCA normals carry an arbitrary sign and `n·l` is sign
  sensitive, so normals are oriented to face the camera (`n·(cam-point)>0`).  The
  specular reflection is sign-invariant, which is why the relit path tolerated
  unsigned normals.

## Frame

Points/normals are in the OpenCV central-camera frame (x-right, y-down, z-forward),
camera at the origin.  The renderer's world equals the central PyTorch3D view (identity
central camera); OpenCV = diag(-1,-1,1)·p3d, so the world light `[0,0.1,0]` maps to
`[0,-0.1,0]` here.

## Other render modes

| mode             | normals used for            | shading                         | use |
|------------------|-----------------------------|---------------------------------|-----|
| `diffuse_shaded` | re-applying SoftPhong shading + (orientation) | albedo·SoftPhong (re-shaded) | **diffuse refine (default)** |
| `diffuse`        | nothing                     | none (frozen shaded bake)       | legacy / ablation |
| `relit`          | reflection dir + diffuse shading | `α·(re-shaded diffuse) + (1-α)·env[reflect]` | reflective refine (default OFF) |
| `sh`             | view dir in object frame    | data-driven SH (implicit)       | legacy |

`relit` now re-shades its diffuse channel too (`render_relit(shade_diffuse=True)`,
the default), exactly reproducing the dataset's `α·(SoftPhong diffuse) + (1-α)·env`
linear blend.  On actual reflective data the re-shaded diffuse tracks a rotated frame
much better than the bake (cube_0.5 45° pair: on-data MSE 2.34e-2 → 1.22e-2), and the
relit loss bottoms at GT (≈9e-4) with a convex basin and 8°→0.000° convergence for
reflectivities 0.5/0.7.  Pass `shade_diffuse=False` for the legacy baked-diffuse relit.

## Splat footprint (related render fidelity fix)

Independent of shading, the isotropic Gaussian footprint was 1.0× the one-pixel depth
footprint, which over-blurs the render vs the (sharp) source image.  It is now 0.5×
(`SurfaceLightField.DEFAULT_FOOTPRINT_SCALE`): formation MSE roughly halves (cube
3.8e-2→1.9e-2, objects 1.0e-2→6.3e-3) with ≥98.6% pixel coverage.
