import math
import torch
import torch.nn.functional as F

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def RGB2SH(rgb: torch.Tensor) -> torch.Tensor:
    # rgb: (..., 3) in [0,1]
    return (rgb - 0.5) / C0


def SH2RGB(sh: torch.Tensor) -> torch.Tensor:
    return sh * C0 + 0.5


def get_sh_bases_torch(view_dirs: torch.Tensor, max_degree: int) -> torch.Tensor:
    """
    view_dirs: [b, n, 3] (assumed unit, but we normalize anyway)
    returns:   [b, n, (max_degree+1)^2]
    Basis ordering matches eval_sh hardcoded coefficients:
      l=0: 0
      l=1: 1..3
      l=2: 4..8
      l=3: 9..15
      l=4: 16..24
    """
    assert 0 <= max_degree <= 4

    view_dirs = view_dirs / (view_dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12))
    x = view_dirs[..., 0]
    y = view_dirs[..., 1]
    z = view_dirs[..., 2]

    b, n, _ = view_dirs.shape
    n_coeffs = (max_degree + 1) ** 2
    bases = view_dirs.new_zeros((b, n, n_coeffs))

    # l=0
    bases[..., 0] = C0

    if max_degree >= 1:
        bases[..., 1] = -C1 * y
        bases[..., 2] = C1 * z
        bases[..., 3] = -C1 * x

    if max_degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        bases[..., 4] = C2[0] * xy
        bases[..., 5] = C2[1] * yz
        bases[..., 6] = C2[2] * (2.0 * zz - xx - yy)
        bases[..., 7] = C2[3] * xz
        bases[..., 8] = C2[4] * (xx - yy)

    if max_degree >= 3:
        xx, yy, zz = x * x, y * y, z * z
        bases[..., 9] = C3[0] * y * (3 * xx - yy)
        bases[..., 10] = C3[1] * (x * y) * z
        bases[..., 11] = C3[2] * y * (4 * zz - xx - yy)
        bases[..., 12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
        bases[..., 13] = C3[4] * x * (4 * zz - xx - yy)
        bases[..., 14] = C3[5] * z * (xx - yy)
        bases[..., 15] = C3[6] * x * (xx - 3 * yy)

    if max_degree >= 4:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        bases[..., 16] = C4[0] * xy * (xx - yy)
        bases[..., 17] = C4[1] * yz * (3 * xx - yy)
        bases[..., 18] = C4[2] * xy * (7 * zz - 1)
        bases[..., 19] = C4[3] * yz * (7 * zz - 3)
        bases[..., 20] = C4[4] * (zz * (35 * zz - 30) + 3)
        bases[..., 21] = C4[5] * xz * (7 * zz - 3)
        bases[..., 22] = C4[6] * (xx - yy) * (7 * zz - 1)
        bases[..., 23] = C4[7] * xz * (xx - 3 * yy)
        bases[..., 24] = C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

    return bases


def pad_views_with_extremes(
    colors_rgb: torch.Tensor,  # [b, n, 3]
    view_dirs: torch.Tensor,  # [b, n, 3]
    valid: torch.Tensor,  # [b, n]
):
    b, n, _ = colors_rgb.shape
    k = int(math.isqrt(b))
    if k * k != b:
        # Non-square view sets (e.g. the 17-view EPI cross) cannot be grid-padded.
        # Same spirit as the replicate-pad below: append extreme-direction
        # pseudo-views, each carrying the colors of the view leaning furthest
        # toward that extreme, duplicated so the extreme-to-data row ratio stays
        # comparable to the square-grid path (4k+4 : k^2 ~ 1:1).
        device, dtype = view_dirs.device, view_dirs.dtype

        def _norm(v):
            return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        card = torch.tensor(
            [[0, 1, 0], [0, -1, 0], [-1, 0, 0], [1, 0, 0]], device=device, dtype=dtype
        )
        diag = _norm(
            torch.tensor(
                [[-1, 1, 0], [1, 1, 0], [-1, -1, 0], [1, -1, 0]],
                device=device, dtype=dtype,
            )
        )
        extremes = torch.cat([card, diag], dim=0)  # [8, 3]
        mean_dirs = _norm(view_dirs.mean(dim=1))  # [b, 3] per-view mean direction
        nearest = torch.argmax(mean_dirs @ extremes.T, dim=0)  # [8]

        cols_add = colors_rgb[nearest].repeat_interleave(2, dim=0)  # [16, n, 3]
        dirs_add = extremes.repeat_interleave(2, dim=0)[:, None, :].expand(-1, n, -1)
        valid_add = torch.ones(16, n, device=device, dtype=torch.bool)
        return (
            torch.cat([colors_rgb, cols_add], dim=0),
            torch.cat([view_dirs, dirs_add], dim=0),
            torch.cat([valid.bool(), valid_add], dim=0),
        )

    # reshape view grid
    colors_g = colors_rgb.view(k, k, n, 3)
    dirs_g = view_dirs.view(k, k, n, 3)
    valid_g = valid.view(k, k, n).float()

    # pad by replicating nearest neighbour
    colors_p = F.pad(
        colors_g.permute(2, 3, 0, 1), (1, 1, 1, 1), mode="replicate"
    ).permute(2, 3, 0, 1)
    dirs_p = F.pad(dirs_g.permute(2, 3, 0, 1), (1, 1, 1, 1), mode="replicate").permute(
        2, 3, 0, 1
    )
    valid_p = (
        F.pad(valid_g.permute(2, 0, 1).unsqueeze(1), (1, 1, 1, 1), mode="replicate")
        .squeeze(1)
        .permute(1, 2, 0)
    )

    # extreme directions (x=left/right, y=up/down)
    device, dtype = view_dirs.device, view_dirs.dtype
    up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    dn = torch.tensor([0.0, -1.0, 0.0], device=device, dtype=dtype)
    lf = torch.tensor([-1.0, 0.0, 0.0], device=device, dtype=dtype)
    rt = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)

    def _norm(v):
        return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    # overwrite borders (broadcast over n)
    dirs_p[0, :, :, :] = up
    dirs_p[-1, :, :, :] = dn
    dirs_p[:, 0, :, :] = lf
    dirs_p[:, -1, :, :] = rt

    # corners
    dirs_p[0, 0, :, :] = _norm(up + lf)
    dirs_p[0, -1, :, :] = _norm(up + rt)
    dirs_p[-1, 0, :, :] = _norm(dn + lf)
    dirs_p[-1, -1, :, :] = _norm(dn + rt)

    # border all valid
    valid_p[0, :, :] = 1.0
    valid_p[-1, :, :] = 1.0
    valid_p[:, 0, :] = 1.0
    valid_p[:, -1, :] = 1.0

    # flatten back to [b_new, n, *]
    k2 = k + 2
    colors_out = colors_p.reshape(k2 * k2, n, 3)
    dirs_out = dirs_p.reshape(k2 * k2, n, 3)
    valid_out = valid_p.reshape(k2 * k2, n)

    return colors_out, dirs_out, valid_out.bool()


def fit_sh_coeffs_per_point(
    colors_rgb: torch.Tensor,  # [b, n, 3]
    view_dirs: torch.Tensor,  # [b, n, 3]
    valid: torch.Tensor,  # [b, n] (0/1 or bool)
    max_degree: int = 2,
    lambda_reg: float = 1e-5,
) -> torch.Tensor:
    """
    Returns:
      sh_coeffs: [n, (max_degree+1)^2, 3]
    """
    assert colors_rgb.ndim == 3 and colors_rgb.shape[-1] == 3
    assert view_dirs.shape == colors_rgb.shape
    assert valid.shape == colors_rgb.shape[:2]

    colors_rgb, view_dirs, valid = pad_views_with_extremes(colors_rgb, view_dirs, valid)

    b, n, _ = colors_rgb.shape
    n_coeffs = (max_degree + 1) ** 2

    # Design matrix
    A = get_sh_bases_torch(view_dirs, max_degree=max_degree)  # [b, n, n_coeffs]

    # Targets in SH space
    Y = RGB2SH(colors_rgb)  # [b, n, 3]

    # Mask invalid observations
    mask = valid.to(dtype=A.dtype).unsqueeze(-1)  # [b, n, 1]
    Aw = A * mask  # [b, n, n_coeffs]

    # Normal equations per point:
    # AtA: [n, n_coeffs, n_coeffs], AtY: [n, n_coeffs, 3]
    AtA = torch.einsum("bnc,bnd->ncd", Aw, A)
    AtY = torch.einsum("bnc,bnk->nck", Aw, Y)

    # Regularization (degree-weighted like your numpy version: exp(l))
    reg_vector = []
    for l in range(max_degree + 1):
        reg_vector.extend(
            [torch.exp(torch.tensor(float(l), device=A.device, dtype=A.dtype))]
            * (2 * l + 1)
        )
    reg_vector = torch.stack(reg_vector)  # [n_coeffs]

    AtA = AtA + lambda_reg * torch.diag(reg_vector).unsqueeze(0)  # broadcast over n

    # Solve
    coeffs = torch.linalg.solve(AtA, AtY)  # [n, n_coeffs, 3]

    # If a point has zero valid views, force zeros (otherwise regularizer returns ~0 but be explicit)
    valid_counts = valid.to(dtype=A.dtype).sum(dim=0)  # [n]
    coeffs = torch.where(
        valid_counts.view(n, 1, 1) > 0, coeffs, torch.zeros_like(coeffs)
    )
    coeffs *= 0.28209479177387814
    return coeffs


if __name__ == "__main__":
    points = torch.load("points.pt").float()  # unused for fitting
    colors = torch.load("colors.pt").float()  # -> [b, n, 3]
    view_dirs = torch.load("view_dirs.pt").float()  # -> [b, n, 3]
    valid = torch.load("valid.pt").float()  # -> [b, n]

    # Fit degree-2 SH per point, per RGB channel
    sh_coeffs = fit_sh_coeffs_per_point(
        colors_rgb=colors,
        view_dirs=view_dirs,
        valid=valid,
        max_degree=2,
        lambda_reg=1e-3,
    )
    opacities = torch.ones_like(points[:, :1])
    scales = torch.ones_like(points) * 1e-3

    means = points
    colors = sh_coeffs
    quats = torch.stack(
        [
            torch.tensor([1, 0, 0, 0]).cuda(),
        ]
        * means.shape[0]
    ).float()

    torch.save(
        {
            "means": means,
            "harmonics": colors,
            "rotations": quats,
            "scales": scales,
            "opacities": opacities,
        },
        "gaussians.pt",
    )
