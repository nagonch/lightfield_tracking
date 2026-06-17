"""SoftPhong shading that reproduces the dataset renderer (ycbv-eoat-lf/render.py).

The cube_*/objects_* diffuse images were produced by PyTorch3D's ``SoftPhongShader``
with a single ``PointLights`` and the default ``Materials`` (shininess 64):

    observed = (ambient + diffuse·relu(n·l))·albedo + specular·relu(v·r)^shininess
        ambient  = [0.5, 0.5,  0.5 ]      l = normalize(light_pos - point)
        diffuse  = [0.5, 0.4,  0.25]      v = normalize(cam_pos  - point)   (cam at origin)
        specular = [0.5, 0.45, 0.35]      r = -l + 2(n·l)n   (gated by n·l>0)
        shininess = 64

The surface light field stores per-point *intrinsic albedo* (shading removed); the
renderer re-applies shading at the candidate pose, so a rotation re-shades the
surface exactly like the original renderer instead of rigidly transporting a frozen
shaded colour.

Colour space: the renderer applied shading in the raw 8-bit (sRGB-encoded) space the
PNG was written in, then we load it with ``srgb_to_linear``.  So un-/re-shading is
done in that raw space (``linear_to_srgb`` ⇄ ``srgb_to_linear`` around the math),
while gsplat compositing and the photometric loss stay linear.  ``shade_from_albedo``
is the exact inverse of ``unshade_to_albedo`` at a fixed (point, normal): the central
view round-trips bit-for-bit, and only a pose change alters the result.

Frame: points/normals are in the OpenCV central-camera frame (x-right, y-down,
z-forward), camera at the origin.  The renderer's world equals the central p3d view
(identity central camera); OpenCV = diag(-1,-1,1)·p3d, so the world light [0,0.1,0]
maps to [0,-0.1,0] here.
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
