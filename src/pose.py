"""Coarse pose from LoFTR correspondences on diffuse views, with ICP fallback.

Match two consecutive diffuse central views with LoFTR, back-project the matches
with depth, and solve a RANSAC Procrustes for the relative rigid transform.  The
new absolute pose is ``T_rel @ prev_pose``.  When alpha < _LOFTR_ALPHA_THRESHOLD
(reflectivity = 1) or LoFTR produces too few inliers, falls back to ICP.
"""

from __future__ import annotations

import numpy as np

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

MIN_INLIERS = 12
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


def _icp_pose(
    pc_prev: np.ndarray,
    pc_curr: np.ndarray,
    color_prev: np.ndarray,
    color_curr: np.ndarray,
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
    """New absolute pose for the current frame.

    When alpha >= _LOFTR_ALPHA_THRESHOLD, tries LoFTR first; falls back to ICP
    if LoFTR fails or produces too few inliers.  When alpha < threshold (i.e.
    reflectivity = 1), skips LoFTR entirely and goes straight to ICP.
    ICP inputs (pc_prev, pc_curr, color_prev, color_curr) are required for the
    ICP path; if absent, holds the previous pose on failure.
    """
    loftr_ok = False

    if alpha >= _LOFTR_ALPHA_THRESHOLD:
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
            return T_rel @ abs_pose_prev

    if pc_prev is not None and pc_curr is not None and color_prev is not None and color_curr is not None:
        try:
            return _icp_pose(pc_prev, pc_curr, color_prev, color_curr, abs_pose_prev)
        except Exception:
            pass

    return abs_pose_prev.copy()
