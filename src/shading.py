"""SoftPhong shading matching the dataset renderer (ycbv-eoat-lf/render.py).

Reproduces PyTorch3D SoftPhongShader: (ambient + diffuse·relu(n·l))·albedo + specular·relu(v·r)^64.
Constants are in config.yaml (shading section). Shading is computed in raw sRGB space
(as the original renderer did) and converted to/from linear for compositing.
unshade_to_albedo and shade_from_albedo are exact inverses at a fixed pose; only a
rotation changes the result, so the diffuse channel re-shades consistently with the renderer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from utils import linear_to_srgb, srgb_to_linear

LIGHT_POS_CAM = (0.0, -0.1, 0.0)
AMBIENT = (0.5, 0.5, 0.5)
DIFFUSE = (0.5, 0.4, 0.25)
SPECULAR = (0.5, 0.45, 0.35)
SHININESS = 64.0


def orient_to_camera(normals: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Flip normals so they face the camera (n·(cam-point) > 0, cam at origin).

    PCA normals carry an arbitrary sign; ``n·l`` is sign-sensitive, so the surface
    normal must point toward the viewer (the visible surface does).
    """
    facing = (normals * (-points)).sum(-1, keepdim=True)
    return torch.where(facing < 0, -normals, normals)


def shading_terms(
    points: torch.Tensor, normals: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-point SoftPhong (ambient+diffuse) factor and additive specular, raw space.

    points/normals : [N, 3] in the OpenCV camera frame (camera at origin).
    Returns (amb_diff [N,3], spec [N,3]).  Normals are oriented to the camera here.
    """
    device, dtype = points.device, points.dtype
    light = torch.tensor(LIGHT_POS_CAM, device=device, dtype=dtype)
    amb = torch.tensor(AMBIENT, device=device, dtype=dtype)
    dif = torch.tensor(DIFFUSE, device=device, dtype=dtype)
    spc = torch.tensor(SPECULAR, device=device, dtype=dtype)

    n = orient_to_camera(F.normalize(normals, dim=-1, eps=1e-6), points)
    l = F.normalize(light - points, dim=-1, eps=1e-6)
    v = F.normalize(-points, dim=-1, eps=1e-6)

    ndotl = (n * l).sum(-1, keepdim=True)
    amb_diff = amb + dif * F.relu(ndotl)
    refl = -l + 2.0 * ndotl * n
    vdotr = F.relu((v * refl).sum(-1, keepdim=True))
    spec = spc * vdotr.pow(SHININESS) * (ndotl > 0).to(dtype)
    return amb_diff, spec


def unshade_to_albedo(
    observed_linear: torch.Tensor, points: torch.Tensor, normals: torch.Tensor
) -> torch.Tensor:
    """Recover intrinsic albedo (raw space) from a shaded linear observation."""
    raw = linear_to_srgb(observed_linear.clamp(0.0, 1.0))
    amb_diff, spec = shading_terms(points, normals)
    return (raw - spec) / amb_diff.clamp(min=1e-4)


def shade_from_albedo(
    albedo_raw: torch.Tensor, points: torch.Tensor, normals: torch.Tensor
) -> torch.Tensor:
    """Re-apply SoftPhong shading to an albedo and return a LINEAR colour."""
    amb_diff, spec = shading_terms(points, normals)
    raw = (albedo_raw * amb_diff + spec).clamp(0.0, 1.0)
    return srgb_to_linear(raw)
