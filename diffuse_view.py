"""Render a diffuse-only (view-independent) middle view from a SurfaceLF.

The diffuse middle view is what LoFTR should receive: reflections are stripped,
only the intrinsic surface color remains.  When alpha (= 1 - reflectivity) is
close to zero the diffuse component is nearly absent and the returned image
will be dark/flat — the caller should fall back to geometry-based features.
"""

import torch
from sh_helpers import RGB2SH
from surface_lf import SurfaceLF, batch_rasterize


def rasterize_diffuse(surface_lf: SurfaceLF) -> torch.Tensor:
    """Return a [H, W, 3] float32 image of the diffuse-only middle view.

    Uses the DC (ℓ=0) spherical-harmonic term only so the colour is
    view-independent — exactly the per-point intrinsic diffuse colour from
    reflection separation.  Falls back to the mean-over-views colour when
    diffuse separation has not been run (use_environment_map=False).
    """
    if surface_lf.diffuse_color_per_point is not None:
        diffuse_rgb = surface_lf.diffuse_color_per_point.to(
            device=surface_lf.values["means"].device,
            dtype=surface_lf.values["means"].dtype,
        )
    else:
        # Fallback: mean of raw observed colors
        valid = surface_lf.valid.float()
        count = valid.sum(dim=0).clamp(min=1).unsqueeze(-1)
        diffuse_rgb = (surface_lf.colors * valid.unsqueeze(-1)).sum(dim=0) / count

    N = diffuse_rgb.shape[0]
    n_coeffs = (2 + 1) ** 2  # degree-2 SH → 9 coefficients

    harmonics_dc = torch.zeros(
        N, n_coeffs, 3,
        device=diffuse_rgb.device,
        dtype=diffuse_rgb.dtype,
    )
    harmonics_dc[:, 0, :] = RGB2SH(diffuse_rgb)

    image, _, _ = batch_rasterize(
        points=surface_lf.values["means"].float(),
        quats=surface_lf.values["rotations"].float(),
        scales=surface_lf.values["scales"].float(),
        opacities=surface_lf.values["opacities"].float(),
        colors=harmonics_dc.float(),
        poses=torch.eye(4).unsqueeze(0).to(surface_lf.values["means"].device),
        camera_matrix=surface_lf.K,
        height=surface_lf.H,
        width=surface_lf.W,
    )
    return torch.clamp(image, 0.0, 1.0)


def diffuse_midview_uint8(surface_lf: SurfaceLF) -> "np.ndarray":
    """Return [H, W, 3] uint8 numpy array suitable for LoFTR input."""
    import numpy as np
    img = rasterize_diffuse(surface_lf)
    return (img.cpu().numpy() * 255).astype("uint8")
