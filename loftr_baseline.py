"""LoFTR-based 6-DoF coarse pose tracker baseline.

Implements the coarse-pose-initialisation step from BundleSDF (Sec. 3.1):
  1. Resize consecutive RGB frames to `resize` px (short edge).
  2. Run LoFTR to find 2-D correspondences.
  3. Back-project both sets of pixels into 3-D using the depth images.
  4. RANSAC over 3-point SVD (Procrustes) to estimate the relative pose.
  5. Accumulate relative poses into absolute poses, then rebase to GT frame 0.

All thresholds are taken from config_ycbvlf_prod.yml.
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loftr_wrapper import LoftrRunner, _RESIZE
from icp import rebase_poses


# ── depth / matching parameters from config_ycbvlf_prod.yml ──────────────────
DEPTH_ZFAR = 2.0          # depth_processing.zfar
DEPTH_ZNEAR = 0.01
DEPTH_PERCENTILE = 95     # depth_processing.percentile
INLIER_DIST = 0.05        # ransac.inlier_dist  (metres)
RANSAC_MAX_ITER = 5000    # ransac.max_iter
MIN_MATCH_AFTER_RANSAC = 3  # ransac.min_match_after_ransac


# ── geometry helpers ──────────────────────────────────────────────────────────

def _backproject(uvs: np.ndarray, depth: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Back-project 2-D pixel coordinates into camera-space 3-D points.

    Parameters
    ----------
    uvs   : (M, 2) float32 – pixel coordinates (x=col, y=row)
    depth : (H, W) float64 – depth in metres
    K     : (3, 3) – camera intrinsics

    Returns
    -------
    pts3d : (N, 3) – valid 3-D points (N ≤ M)
    mask  : (M,) bool – which input uvs are valid
    """
    us = np.round(uvs[:, 0]).astype(int)
    vs = np.round(uvs[:, 1]).astype(int)
    H, W = depth.shape
    in_bounds = (us >= 0) & (us < W) & (vs >= 0) & (vs < H)
    us_c = np.clip(us, 0, W - 1)
    vs_c = np.clip(vs, 0, H - 1)
    d = depth[vs_c, us_c]
    valid = in_bounds & (d > DEPTH_ZNEAR) & (d < DEPTH_ZFAR)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (us_c.astype(np.float64) - cx) * d / fx
    y = (vs_c.astype(np.float64) - cy) * d / fy
    pts3d = np.stack([x, y, d], axis=-1)
    return pts3d[valid], valid


def _procrustes(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """SVD-based least-squares rigid alignment (src → dst).

    Parameters
    ----------
    src, dst : (N, 3) with N ≥ 3

    Returns
    -------
    T : (4, 4) transform that maps src points to dst frame.
    """
    c_src = src.mean(0)
    c_dst = dst.mean(0)
    A = (src - c_src).T @ (dst - c_dst)
    U, _, Vt = np.linalg.svd(A)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = c_dst - R @ c_src
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _ransac_relative_pose(
    pts0: np.ndarray,
    pts1: np.ndarray,
    max_iter: int = RANSAC_MAX_ITER,
    inlier_dist: float = INLIER_DIST,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray | None, np.ndarray]:
    """RANSAC over 3-point Procrustes to find T such that T @ pts0 ≈ pts1.

    Returns
    -------
    T_best : (4, 4) or None if fewer than MIN_MATCH_AFTER_RANSAC inliers
    inlier_mask : (N,) bool
    """
    if rng is None:
        rng = np.random.default_rng()
    N = len(pts0)
    if N < 3:
        return None, np.zeros(N, dtype=bool)

    best_T = None
    best_inliers = np.zeros(N, dtype=bool)
    best_count = 0

    for _ in range(max_iter):
        idx = rng.choice(N, 3, replace=False)
        try:
            T = _procrustes(pts0[idx], pts1[idx])
        except np.linalg.LinAlgError:
            continue
        R, t = T[:3, :3], T[:3, 3]
        residuals = np.linalg.norm((pts0 @ R.T + t) - pts1, axis=1)
        inliers = residuals < inlier_dist
        count = inliers.sum()
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_T = T
            if best_count == N:
                break

    if best_count >= MIN_MATCH_AFTER_RANSAC and best_T is not None:
        # Refit on all inliers
        try:
            best_T = _procrustes(pts0[best_inliers], pts1[best_inliers])
        except np.linalg.LinAlgError:
            pass

    if best_count < MIN_MATCH_AFTER_RANSAC:
        return None, best_inliers

    return best_T, best_inliers


def _resize_for_loftr(rgb: np.ndarray, target: int = _RESIZE) -> tuple[np.ndarray, float]:
    """Resize so that the short edge == target; return image and scale factor."""
    H, W = rgb.shape[:2]
    scale = target / min(H, W)
    new_H, new_W = int(round(H * scale)), int(round(W * scale))
    # LoFTR requires dimensions divisible by 8
    new_H = (new_H // 8) * 8
    new_W = (new_W // 8) * 8
    resized = cv2.resize(rgb, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
    return resized, scale


def _filter_depth_percentile(depth: np.ndarray, mask: np.ndarray, pct: int = DEPTH_PERCENTILE) -> np.ndarray:
    """Zero out depth values above the pct-th percentile inside the mask."""
    depth = depth.copy()
    valid = (depth >= DEPTH_ZNEAR) & mask
    if valid.sum() > 0:
        thres = np.percentile(depth[valid], pct)
        depth[depth >= thres] = 0.0
    return depth


# ── main tracker class ────────────────────────────────────────────────────────

class LoftrBase:
    """Frame-to-frame pose tracker using LoFTR correspondences + RANSAC.

    Mirrors the coarse-pose-initialisation thread of BundleSDF
    (Wen et al., CVPR 2023, Sec. 3.1) without the pose-graph or NeRF threads.
    """

    def __init__(self, seq, resize: int = _RESIZE, rng_seed: int = 0):
        """
        Parameters
        ----------
        seq    : SpecTrackSequence (or any object with .K, .get_frame(), __len__)
        resize : short-edge target for LoFTR input
        """
        self.seq = seq
        self.resize = resize
        self.rng = np.random.default_rng(rng_seed)
        self.loftr = LoftrRunner()

    def _get_depth_filtered(self, idx: int):
        rgb, depth, mask = self.seq.get_frame(idx)
        depth = _filter_depth_percentile(depth, mask)
        return rgb, depth, mask

    def run_coarse(self) -> np.ndarray:
        """Accumulate frame-to-frame LoFTR poses into an absolute pose sequence.

        Returns
        -------
        poses : (N, 4, 4) float64
            Absolute poses in the coordinate frame of frame 0 (identity at t=0).
            Not rebased to GT — suitable as coarse initialisation for refinement.
        """
        n = len(self.seq)
        est_poses = [np.eye(4)]
        rgb_prev, depth_prev, _ = self._get_depth_filtered(0)

        for idx in tqdm(range(1, n), desc="  frames", leave=False):
            rgb_cur, depth_cur, _ = self._get_depth_filtered(idx)
            T_rel = self._estimate_relative_pose(
                rgb_prev, depth_prev,
                rgb_cur, depth_cur,
                self.seq.K,
            )
            if T_rel is None:
                T_rel = np.eye(4)
            est_poses.append(T_rel @ est_poses[-1])
            rgb_prev, depth_prev = rgb_cur, depth_cur

        return np.stack(est_poses)

    def run(self) -> np.ndarray:
        """Track the full sequence (baseline evaluation entry point).

        Returns
        -------
        poses : (N, 4, 4) float64
            Absolute object-to-camera poses rebased so poses[0] == GT poses[0].
        """
        n = len(self.seq)
        gt_poses = np.stack([self.seq.get_gt_pose(i) for i in range(n)])
        est_poses = self.run_coarse()
        return rebase_poses(gt_poses, est_poses)

    def _estimate_relative_pose(
        self,
        rgb0: np.ndarray, depth0: np.ndarray,
        rgb1: np.ndarray, depth1: np.ndarray,
        K: np.ndarray,
    ) -> np.ndarray | None:
        """Estimate T such that T maps 3-D points from frame0 into frame1."""
        small0, scale0 = _resize_for_loftr(rgb0, self.resize)
        small1, scale1 = _resize_for_loftr(rgb1, self.resize)

        # LoFTR expects (N, H, W, 3) uint8
        batch0 = small0[None]
        batch1 = small1[None]
        corres_list = self.loftr.predict(batch0, batch1)
        corres = corres_list[0]   # (M, 5): x0 y0 x1 y1 conf

        if len(corres) < 3:
            return None

        # Scale pixel coordinates back to original resolution
        uvs0 = corres[:, :2] / scale0
        uvs1 = corres[:, 2:4] / scale1

        pts0, valid0 = _backproject(uvs0, depth0, K)
        # Only keep correspondences valid in frame0; apply same mask to uvs1
        uvs1_valid = uvs1[valid0]
        pts1_valid, valid1 = _backproject(uvs1_valid, depth1, K)
        pts0_valid = pts0[valid1]

        if len(pts0_valid) < 3:
            return None

        T_rel, _ = _ransac_relative_pose(pts0_valid, pts1_valid, rng=self.rng)
        return T_rel
