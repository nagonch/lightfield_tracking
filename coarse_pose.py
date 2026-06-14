"""Coarse pose estimation: α-weighted mix of LoFTR (diffuse images) and ICP (geometry).

α = separation_alpha = 1 - reflectivity
  → 1.0 : purely diffuse surface  → LoFTR only
  → 0.0 : purely reflective        → ICP only
  → middle: try LoFTR, fall back to ICP if too few inliers

The returned pose is an ABSOLUTE pose matrix [4, 4] (same convention as
est_poses in main_old.py: object pose in camera frame at time t).
"""

import numpy as np
import cv2
from typing import Optional

from loftr_wrapper import LoftrRunner, _RESIZE
from loftr_baseline import (
    _backproject,
    _resize_for_loftr,
    _ransac_relative_pose,
    _filter_depth_percentile,
    DEPTH_ZNEAR,
    DEPTH_ZFAR,
)
from icp import (
    get_coarsest_pose,
    run_explorative_icp_with_centering,
    apply_transform_to_points,
)


# Minimum number of LoFTR inliers to trust the LoFTR estimate
_LOFTR_MIN_INLIERS = 12
_LOFTR_ALPHA_THRESHOLD = 0.05   # below this α → skip LoFTR entirely


def loftr_relative_pose(
    rgb_prev: np.ndarray,
    rgb_curr: np.ndarray,
    depth_prev: np.ndarray,
    depth_curr: np.ndarray,
    mask_prev: np.ndarray,
    mask_curr: np.ndarray,
    K: np.ndarray,
    loftr: LoftrRunner,
    rng: Optional[np.random.Generator] = None,
) -> tuple[Optional[np.ndarray], int]:
    """Run LoFTR and estimate relative pose via RANSAC Procrustes.

    rgb_*   : [H, W, 3] uint8
    depth_* : [H, W] float32, metres
    mask_*  : [H, W] bool

    Returns (T_prev_to_curr, n_inliers).  T is [4,4] float64 mapping
    3D points from prev camera frame into curr camera frame.
    Returns (None, 0) on failure.
    """
    if rng is None:
        rng = np.random.default_rng()

    depth_prev_f = _filter_depth_percentile(depth_prev, mask_prev)
    depth_curr_f = _filter_depth_percentile(depth_curr, mask_curr)

    small_prev, scale_prev = _resize_for_loftr(rgb_prev, _RESIZE)
    small_curr, scale_curr = _resize_for_loftr(rgb_curr, _RESIZE)

    corres_list = loftr.predict(small_prev[None], small_curr[None])
    corres = corres_list[0]   # [M, 5]: x0 y0 x1 y1 conf

    if len(corres) < 3:
        return None, 0

    uvs_prev = corres[:, :2] / scale_prev
    uvs_curr = corres[:, 2:4] / scale_curr

    pts3d_prev, valid_prev = _backproject(uvs_prev, depth_prev_f, K)
    uvs_curr_v = uvs_curr[valid_prev]
    pts3d_curr, valid_curr = _backproject(uvs_curr_v, depth_curr_f, K)
    pts3d_prev_v = pts3d_prev[valid_curr]

    if len(pts3d_prev_v) < 3:
        return None, 0

    T_rel, inlier_mask = _ransac_relative_pose(pts3d_prev_v, pts3d_curr, rng=rng)
    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    return T_rel, n_inliers


def icp_relative_pose(
    pc_prev: np.ndarray,
    pc_curr: np.ndarray,
    color_prev: np.ndarray,
    color_curr: np.ndarray,
    abs_pose_prev: np.ndarray,
) -> np.ndarray:
    """ICP-based coarse absolute pose at current frame.

    pc_*, color_* in camera frame (output of backproject filtered by mask).
    abs_pose_prev is est_poses[-1] (4×4).
    Returns new absolute pose [4,4] float64.
    """
    coarsest_pose, pc_prev_trans = get_coarsest_pose(pc_prev, pc_curr, abs_pose_prev)
    reg = run_explorative_icp_with_centering(
        source_points_xyz=pc_prev_trans,
        target_points_xyz=pc_curr,
        source_colors_rgb=color_prev,
        target_colors_rgb=color_curr,
        max_correspondence_distance=0.01,
    )
    coarse_pose = reg["transform_source_to_target"] @ coarsest_pose
    return coarse_pose


def mixed_coarse_pose(
    alpha: float,
    # LoFTR inputs
    diffuse_prev: np.ndarray,
    diffuse_curr: np.ndarray,
    depth_prev: np.ndarray,
    depth_curr: np.ndarray,
    mask_prev: np.ndarray,
    mask_curr: np.ndarray,
    K: np.ndarray,
    loftr: LoftrRunner,
    # ICP inputs
    pc_prev: np.ndarray,
    pc_curr: np.ndarray,
    color_prev: np.ndarray,
    color_curr: np.ndarray,
    abs_pose_prev: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Choose coarse pose via α-weighted strategy.

    Returns new absolute pose [4,4] float64.
    """
    loftr_pose, n_inliers = None, 0

    if alpha >= _LOFTR_ALPHA_THRESHOLD:
        T_rel, n_inliers = loftr_relative_pose(
            rgb_prev=diffuse_prev,
            rgb_curr=diffuse_curr,
            depth_prev=depth_prev,
            depth_curr=depth_curr,
            mask_prev=mask_prev,
            mask_curr=mask_curr,
            K=K,
            loftr=loftr,
            rng=rng,
        )
        if T_rel is not None and n_inliers >= _LOFTR_MIN_INLIERS:
            # T_rel maps camera pts from prev to curr.
            # New absolute pose = T_rel @ abs_pose_prev
            loftr_pose = T_rel @ abs_pose_prev

    use_loftr = (loftr_pose is not None) and (alpha >= 0.5 or n_inliers >= _LOFTR_MIN_INLIERS * 2)

    if use_loftr and alpha >= 0.9:
        return loftr_pose

    # Always compute ICP (needed as fallback or for mixing)
    try:
        geom_pose = icp_relative_pose(
            pc_prev=pc_prev,
            pc_curr=pc_curr,
            color_prev=color_prev,
            color_curr=color_curr,
            abs_pose_prev=abs_pose_prev,
        )
    except Exception:
        geom_pose = abs_pose_prev.copy()

    if not use_loftr:
        return geom_pose

    # Soft blend in SE(3): interpolate rotation via SLERP and translation linearly
    return _blend_poses(loftr_pose, geom_pose, alpha)


def _blend_poses(
    pose_a: np.ndarray,
    pose_b: np.ndarray,
    weight_a: float,
) -> np.ndarray:
    """Blend two 4×4 poses: weight_a for pose_a, (1-weight_a) for pose_b."""
    from scipy.spatial.transform import Rotation, Slerp
    w = float(np.clip(weight_a, 0.0, 1.0))

    R_a = Rotation.from_matrix(pose_a[:3, :3])
    R_b = Rotation.from_matrix(pose_b[:3, :3])
    slerp = Slerp([0.0, 1.0], Rotation.concatenate([R_b, R_a]))
    R_blend = slerp(w).as_matrix()

    t_blend = (1 - w) * pose_b[:3, 3] + w * pose_a[:3, 3]

    out = np.eye(4)
    out[:3, :3] = R_blend
    out[:3, 3] = t_blend
    return out
