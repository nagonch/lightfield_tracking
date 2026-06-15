"""Coarse pose from LoFTR correspondences on diffuse views.

Match two consecutive diffuse central views with LoFTR, back-project the matches
with depth, and solve a RANSAC Procrustes for the relative rigid transform.  The
new absolute pose is ``T_rel @ prev_pose``.  If matching is unreliable the previous
pose is held.
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

MIN_INLIERS = 12


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
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """New absolute pose for the current frame; holds the previous on failure."""
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
    return abs_pose_prev.copy()
