"""6-D rotation representation helpers (Zhou et al., CVPR 2019).

A continuous 6-D parameterisation of SO(3) used by the photometric pose
refinement: the optimizer carries a 6-D vector, which is mapped to a valid
rotation matrix via Gram-Schmidt on each step.
"""

from __future__ import annotations

import torch


def safe_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """6-D vector → rotation matrix via Gram-Schmidt (differentiable)."""
    inputs = d6.view(-1, 2, 3)
    a1 = inputs[:, 0]
    a2 = inputs[:, 1]

    b1 = safe_normalize(a1)
    proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = safe_normalize(a2 - proj)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack((b1, b2, b3), dim=-1).squeeze()


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Rotation matrix → 6-D vector (first two columns).

    Input: (3, 3) matrix. Output: (6,) vector.
    """
    return matrix[:3, :2].transpose(0, 1).reshape(-1)
