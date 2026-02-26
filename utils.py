import torch
import torch.nn.functional as F


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
