"""6-DoF pose tracking: LoFTR + ICP geodesically blended by reflectivity.

reflectivity=0 → LoFTR only; reflectivity=1 → geometry-only ICP; intermediate → blend.
Diffuse images feed both branches when separation is active.
"""

from __future__ import annotations

import numpy as np
from utils import linear_to_srgb
from loftr_baseline import (
    _backproject,
    _filter_depth_percentile,
    _ransac_relative_pose,
    _resize_for_loftr,
)
from loftr_wrapper import LoftrRunner, _RESIZE
from icp import (
    get_coarsest_pose,
    run_explorative_icp_with_centering,
)

from config import MIN_LOFTR_INLIERS as MIN_INLIERS


def loftr_relative_pose(
    rgb_prev: np.ndarray,
    rgb_curr: np.ndarray,
    depth_prev: np.ndarray,
    depth_curr: np.ndarray,
    mask_prev: np.ndarray,
    mask_curr: np.ndarray,
    K: np.ndarray,
    loftr: LoftrRunner,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray | None, int]:
    """Relative pose mapping prev-frame points into the curr-frame camera.

    Returns ``(T_rel [4,4], n_inliers)`` or ``(None, 0)`` on failure.
    """
    rng = rng or np.random.default_rng()

    depth_prev = _filter_depth_percentile(depth_prev, mask_prev)
    depth_curr = _filter_depth_percentile(depth_curr, mask_curr)

    small_prev, scale_prev = _resize_for_loftr(rgb_prev, _RESIZE)
    small_curr, scale_curr = _resize_for_loftr(rgb_curr, _RESIZE)

    corres = loftr.predict(small_prev[None], small_curr[None])[0]  # [M, 5]
    if len(corres) < 3:
        return None, 0

    uvs_prev = corres[:, :2] / scale_prev
    uvs_curr = corres[:, 2:4] / scale_curr

    pts_prev, ok_prev = _backproject(uvs_prev, depth_prev, K)
    pts_curr, ok_curr = _backproject(uvs_curr[ok_prev], depth_curr, K)
    pts_prev = pts_prev[ok_curr]
    if len(pts_prev) < 3:
        return None, 0

    T_rel, inliers = _ransac_relative_pose(pts_prev, pts_curr, rng=rng)
    return T_rel, int(inliers.sum()) if inliers is not None else 0


def _icp_abs_pose(
    pc_prev: np.ndarray,
    pc_curr: np.ndarray,
    color_prev: np.ndarray | None,
    color_curr: np.ndarray | None,
    abs_pose_prev: np.ndarray,
) -> np.ndarray:
    coarsest_pose, pc_prev_trans = get_coarsest_pose(pc_prev, pc_curr, abs_pose_prev)
    reg = run_explorative_icp_with_centering(
        source_points_xyz=pc_prev_trans,
        target_points_xyz=pc_curr,
        source_colors_rgb=color_prev,
        target_colors_rgb=color_curr,
        max_correspondence_distance=0.01,
    )
    return reg["transform_source_to_target"] @ coarsest_pose


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → unit quaternion [x, y, z, w]."""
    m = R
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        return np.array(
            [
                (m[2, 1] - m[1, 2]) * s,
                (m[0, 2] - m[2, 0]) * s,
                (m[1, 0] - m[0, 1]) * s,
                0.25 / s,
            ]
        )
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        return np.array(
            [
                0.25 * s,
                (m[0, 1] + m[1, 0]) / s,
                (m[0, 2] + m[2, 0]) / s,
                (m[2, 1] - m[1, 2]) / s,
            ]
        )
    if m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        return np.array(
            [
                (m[0, 1] + m[1, 0]) / s,
                0.25 * s,
                (m[1, 2] + m[2, 1]) / s,
                (m[0, 2] - m[2, 0]) / s,
            ]
        )
    s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
    return np.array(
        [
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
            (m[1, 0] - m[0, 1]) / s,
        ]
    )


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [x, y, z, w] → rotation matrix."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _slerp(q_a: np.ndarray, q_b: np.ndarray, t: float) -> np.ndarray:
    dot = np.clip(np.dot(q_a, q_b), -1.0, 1.0)
    if dot < 0:  # take shorter arc
        q_b, dot = -q_b, -dot
    if dot > 0.9995:
        return q_a + t * (q_b - q_a)
    theta = np.arccos(dot)
    return (np.sin((1 - t) * theta) * q_a + np.sin(t * theta) * q_b) / np.sin(theta)


def _blend_poses(T_a: np.ndarray, T_b: np.ndarray, w: float) -> np.ndarray:
    """Geodesic blend: w=0 → T_a, w=1 → T_b."""
    q_a = _mat_to_quat(T_a[:3, :3])
    q_b = _mat_to_quat(T_b[:3, :3])
    R_blend = _quat_to_mat(_slerp(q_a, q_b, w))
    t_blend = (1.0 - w) * T_a[:3, 3] + w * T_b[:3, 3]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_blend
    T[:3, 3] = t_blend
    return T


def track_pose(
    abs_pose_prev: np.ndarray,
    diffuse_prev: np.ndarray,
    diffuse_curr: np.ndarray,
    depth_prev: np.ndarray,
    depth_curr: np.ndarray,
    mask_prev: np.ndarray,
    mask_curr: np.ndarray,
    K: np.ndarray,
    loftr: LoftrRunner,
    alpha: float = 1.0,
    pc_prev: np.ndarray | None = None,
    pc_curr: np.ndarray | None = None,
    color_prev: np.ndarray | None = None,
    color_curr: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Absolute pose blended by reflectivity (= 1 - alpha).

    LoFTR and ICP each run where appropriate and fall back gracefully;
    returns prev pose if both fail.
    """
    reflectivity = 1.0 - alpha
    icp_has_inputs = pc_prev is not None and pc_curr is not None

    # ── LoFTR branch ──────────────────────────────────────────────────────────
    pose_loftr: np.ndarray | None = None
    if reflectivity < 1.0:
        T_rel, n_inliers = loftr_relative_pose(
            diffuse_prev,
            diffuse_curr,
            depth_prev,
            depth_curr,
            mask_prev,
            mask_curr,
            K,
            loftr,
            rng,
        )
        if T_rel is not None and n_inliers >= MIN_INLIERS:
            pose_loftr = T_rel @ abs_pose_prev

    # ── ICP branch ────────────────────────────────────────────────────────────
    pose_icp: np.ndarray | None = None
    if reflectivity > 0.0 and icp_has_inputs:
        # suppress colors at full reflectivity → geometry-only ICP
        icp_colors_prev = None if reflectivity == 1.0 else color_prev
        icp_colors_curr = None if reflectivity == 1.0 else color_curr
        try:
            pose_icp = _icp_abs_pose(
                pc_prev,
                pc_curr,
                icp_colors_prev,
                icp_colors_curr,
                abs_pose_prev,
            )
        except Exception:
            pose_icp = None

    # ── blend / fallback ──────────────────────────────────────────────────────
    if pose_loftr is not None and pose_icp is not None:
        return _blend_poses(pose_loftr, pose_icp, reflectivity)
    if pose_loftr is not None:
        return pose_loftr
    if pose_icp is not None:
        return pose_icp
    return abs_pose_prev.copy()
