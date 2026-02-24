import torch
from tqdm import tqdm

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


def rgb_to_sh_values(rgb_0_1: torch.Tensor) -> torch.Tensor:
    """RGB in [0,1] -> SH-space values (same mapping as your RGB2SH)."""
    return (rgb_0_1 - 0.5) / C0


def sh_to_rgb_values(sh_values: torch.Tensor) -> torch.Tensor:
    """SH-space values -> RGB in [0,1] (same mapping as your SH2RGB)."""
    return sh_values * C0 + 0.5


def real_sh_basis_from_dirs(dirs_unit: torch.Tensor, degree: int) -> torch.Tensor:
    """
    Build real SH basis values B for dirs, matching the coefficient ordering in eval_sh().
    Args:
        dirs_unit: [M, 3] unit directions
        degree: 0..4
    Returns:
        B: [M, K] where K=(degree+1)^2
    """
    assert 0 <= degree <= 4
    device = dirs_unit.device
    dtype = dirs_unit.dtype

    M = dirs_unit.shape[0]
    K = (degree + 1) ** 2
    B = torch.zeros((M, K), device=device, dtype=dtype)

    x = dirs_unit[:, 0:1]
    y = dirs_unit[:, 1:2]
    z = dirs_unit[:, 2:3]

    # l=0
    B[:, 0] = C0

    if degree >= 1:
        # coefficients 1..3 (match: -C1*y, +C1*z, -C1*x)
        B[:, 1] = (-C1 * y).squeeze(-1)
        B[:, 2] = (C1 * z).squeeze(-1)
        B[:, 3] = (-C1 * x).squeeze(-1)

    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z

        B[:, 4] = (C2[0] * xy).squeeze(-1)
        B[:, 5] = (C2[1] * yz).squeeze(-1)
        B[:, 6] = (C2[2] * (2.0 * zz - xx - yy)).squeeze(-1)
        B[:, 7] = (C2[3] * xz).squeeze(-1)
        B[:, 8] = (C2[4] * (xx - yy)).squeeze(-1)

    if degree >= 3:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z

        B[:, 9] = (C3[0] * y * (3 * xx - yy)).squeeze(-1)
        B[:, 10] = (C3[1] * xy * z).squeeze(-1)
        B[:, 11] = (C3[2] * y * (4 * zz - xx - yy)).squeeze(-1)
        B[:, 12] = (C3[3] * z * (2 * zz - 3 * xx - 3 * yy)).squeeze(-1)
        B[:, 13] = (C3[4] * x * (4 * zz - xx - yy)).squeeze(-1)
        B[:, 14] = (C3[5] * z * (xx - yy)).squeeze(-1)
        B[:, 15] = (C3[6] * x * (xx - 3 * yy)).squeeze(-1)

    if degree >= 4:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z

        B[:, 16] = (C4[0] * xy * (xx - yy)).squeeze(-1)
        B[:, 17] = (C4[1] * yz * (3 * xx - yy)).squeeze(-1)
        B[:, 18] = (C4[2] * xy * (7 * zz - 1)).squeeze(-1)
        B[:, 19] = (C4[3] * yz * (7 * zz - 3)).squeeze(-1)
        B[:, 20] = (C4[4] * (zz * (35 * zz - 30) + 3)).squeeze(-1)
        B[:, 21] = (C4[5] * xz * (7 * zz - 3)).squeeze(-1)
        B[:, 22] = (C4[6] * (xx - yy) * (7 * zz - 1)).squeeze(-1)
        B[:, 23] = (C4[7] * xz * (xx - 3 * yy)).squeeze(-1)
        B[:, 24] = (C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))).squeeze(-1)

    return B


def sh_ridge_reg_vector(degree: int, device, dtype) -> torch.Tensor:
    """
    Regularizer per coefficient, matching your idea: exp(l) for all m in each l.
    Length K=(degree+1)^2.
    """
    assert 0 <= degree <= 4
    reg_list = []
    for l in range(degree + 1):
        count = 2 * l + 1
        reg_list.extend(
            [torch.exp(torch.tensor(float(l), device=device, dtype=dtype))] * count
        )
    return torch.stack(reg_list, dim=0)  # [K]


def fit_sh_coeffs_per_point(
    colors_rgb_0_1: torch.Tensor,  # [N, P, 3]
    view_dirs_unit: torch.Tensor,  # [N, P, 3]
    valid: torch.Tensor,  # [N, P] bool
    degree: int = 3,
    lambda_reg: float = 1e-5,
) -> torch.Tensor:
    """
    Batched per-point ridge fit:
        Y[n,p,:] ~= B[n,p,:] @ coeffs[p,:,:]^T   (per channel)
    Returns:
        coeffs: [P, 3, K], K=(degree+1)^2
    """
    assert colors_rgb_0_1.ndim == 3 and view_dirs_unit.ndim == 3 and valid.ndim == 2
    N, P, C = colors_rgb_0_1.shape
    assert C == 3
    assert view_dirs_unit.shape == (N, P, 3)
    assert valid.shape == (N, P)

    device = colors_rgb_0_1.device
    dtype = colors_rgb_0_1.dtype
    K = (degree + 1) ** 2

    # Build everything for all (n,p)
    sh_targets = rgb_to_sh_values(colors_rgb_0_1.reshape(-1, 3)).reshape(
        N, P, 3
    )  # [N,P,3]
    B = real_sh_basis_from_dirs(view_dirs_unit.reshape(-1, 3), degree=degree).reshape(
        N, P, K
    )  # [N,P,K]

    # Weights: 1 for valid, 0 for invalid
    w = valid.to(dtype=dtype)  # [N,P]
    Bw = B * w[..., None]  # [N,P,K]
    Yw = sh_targets * w[..., None]  # [N,P,3]

    # Normal equations per point p:
    # BtB[p] = sum_n B[n,p]^T B[n,p]
    # BtY[p] = sum_n B[n,p]^T Y[n,p]
    BtB = torch.einsum("npk,npl->pkl", Bw, B)  # [P,K,K]
    BtY = torch.einsum("npk,npc->pkc", Bw, Yw)  # [P,K,3]

    # Ridge diag, broadcast over P
    reg_vec = sh_ridge_reg_vector(degree, device=device, dtype=dtype)  # [K]
    ridge = lambda_reg * torch.diag(reg_vec).unsqueeze(0)  # [1,K,K]
    lhs = BtB + ridge  # [P,K,K]

    # Solve batched systems
    coeffs_pk3 = torch.linalg.solve(lhs, BtY)  # [P,K,3]

    # If a point has no valid samples, we'd like coeffs=0 (your loop did that).
    # With only ridge and no data, solve gives 0 anyway because BtY is 0.
    # Still, keep it explicit in case of weird NaNs in inputs.
    has_obs = valid.sum(dim=0) > 0  # [P]
    coeffs_pk3 = torch.where(
        has_obs[:, None, None], coeffs_pk3, torch.zeros_like(coeffs_pk3)
    )

    return coeffs_pk3.permute(0, 2, 1).contiguous()  # [P,3,K]


if __name__ == "__main__":
    colors = torch.load("colors.pt").float()
    view_dirs = torch.load("view_dirs.pt").float()
    valid = torch.load("valid.pt").float()
    view_dirs /= view_dirs.norm(dim=-1, keepdim=True)
    sh_coeffs = fit_sh_coeffs_per_point(
        colors, view_dirs, valid, degree=3, lambda_reg=1e-5
    )
    print(sh_coeffs.shape)
