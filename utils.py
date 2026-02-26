import torch
import torch.nn.functional as F


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
