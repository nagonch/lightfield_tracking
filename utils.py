import torch
import torch.nn.functional as F
import torch


def srgb_to_linear(
    srgb: torch.Tensor,
    cutoff: float = 0.04045,
    linear_scale: float = 12.92,
    gamma_offset: float = 0.055,
    gamma_scale: float = 1.055,
    gamma_exponent: float = 2.4,
) -> torch.Tensor:
    """
    srgb: float tensor in [0,1]
    returns linear float tensor in [0,1]
    """
    below = srgb <= cutoff
    linear = torch.empty_like(srgb)
    linear[below] = srgb[below] / linear_scale
    linear[~below] = ((srgb[~below] + gamma_offset) / gamma_scale) ** gamma_exponent
    return linear


def linear_to_srgb(
    linear: torch.Tensor,
    cutoff: float = 0.0031308,
    linear_scale: float = 12.92,
    gamma_offset: float = 0.055,
    gamma_scale: float = 1.055,
    gamma_exponent: float = 2.4,
) -> torch.Tensor:
    """
    linear: float tensor in [0,1]
    returns srgb float tensor in [0,1]
    """
    below = linear <= cutoff
    srgb = torch.empty_like(linear)
    srgb[below] = linear[below] * linear_scale
    srgb[~below] = (
        gamma_scale * (linear[~below] ** (1.0 / gamma_exponent)) - gamma_offset
    )
    return srgb


def rhs_to_lhs_rel(rhs_rel: torch.Tensor, pose_abs: torch.Tensor) -> torch.Tensor:
    """
    Converts P1 = pose_abs @ rhs_rel  =>  P1 = lhs_rel @ pose_abs
    Formula: L = P @ R @ P^-1
    """
    inv_pose = torch.linalg.inv(pose_abs)
    return pose_abs @ rhs_rel @ inv_pose


def lhs_to_rhs_rel(lhs_rel: torch.Tensor, pose_abs: torch.Tensor) -> torch.Tensor:
    """
    Converts P1 = lhs_rel @ pose_abs  =>  P1 = pose_abs @ rhs_rel
    Formula: R = P^-1 @ L @ P
    """
    inv_pose = torch.linalg.inv(pose_abs)
    return inv_pose @ lhs_rel @ pose_abs


def matrix_to_axis_angle(R: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    R: [3,3] rotation matrix
    returns: [3] axis-angle (rotation vector)
    """
    trace = R.diagonal(offset=0, dim1=-1, dim2=-2).sum()
    cos_theta = (trace - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)

    # Small angle: log(R) ~ vee(R - R^T)/2
    if theta < 1e-4:
        w = 0.5 * torch.tensor(
            [
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1],
            ],
            device=R.device,
        )
        return w

    # General case
    omega_hat = (R - R.transpose(0, 1)) / (2.0 * torch.sin(theta) + eps)
    axis = torch.tensor(
        [omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]], device=R.device
    )
    return axis * theta


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    axis_angle: [3]
    returns: [3,3]
    """
    theta = torch.norm(axis_angle) + 1e-8
    axis = axis_angle / theta

    K = torch.tensor(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ],
        device=axis.device,
    )

    eye = torch.eye(3, device=axis.device)
    R = eye + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
    return R


def compose_pose(
    rotation_vec: torch.Tensor, translation_vec: torch.Tensor
) -> torch.Tensor:
    """
    rotation_vec: [3] axis-angle
    translation_vec: [3]
    returns 4x4 pose matrix
    """
    R = axis_angle_to_matrix(rotation_vec)
    T = torch.eye(4, device=rotation_vec.device)
    T[:3, :3] = R
    T[:3, 3] = translation_vec
    return T


if __name__ == "__main__":
    pass
