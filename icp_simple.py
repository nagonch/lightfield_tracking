import os
import kiss_matcher
import numpy as np
import open3d as o3d
from src.utilities import Visualizer
import torch
from icp import rebase_poses, pose_errors
from scipy.spatial.transform import Rotation as R
import itertools
from src.dataset import LFDataset
from tqdm import tqdm
from src.utilities import backproject_depth_to_pointcloud


def rebase_poses(gt_poses, est_poses):
    pose_est_0 = est_poses[0]
    pose_gt_0 = gt_poses[0]
    est_to_gt = np.linalg.inv(pose_est_0) @ pose_gt_0
    est_poses = [p @ est_to_gt for p in est_poses]
    est_poses = np.stack(est_poses, axis=0)

    return est_poses


def rotation_angle_deg(R_err):
    trace = np.trace(R_err, axis1=-2, axis2=-1)
    cos_theta = np.clip((trace - 1) / 2, -1.0, 1.0)
    return np.arccos(cos_theta) * (180.0 / np.pi)


def pose_errors(gt_poses, est_poses):
    assert (
        gt_poses.shape == est_poses.shape
    ), f"GT poses shape {gt_poses.shape} does not match estimated poses shape {est_poses.shape}"
    assert gt_poses.shape[-2:] == (4, 4)

    N = gt_poses.shape[0]

    R_gt = gt_poses[:, :3, :3]
    t_gt = gt_poses[:, :3, 3]
    R_est = est_poses[:, :3, :3]
    t_est = est_poses[:, :3, 3]
    R_err_abs = R_est @ np.transpose(R_gt, (0, 2, 1))  # (N, 3, 3)
    rot_err_abs = rotation_angle_deg(R_err_abs)  # (N,)
    trans_err_abs = np.linalg.norm(t_est - t_gt, axis=1)  # (N,)
    ate_rmse = np.sqrt((trans_err_abs**2).mean())

    rel_rot_errs = []
    rel_trans_errs = []
    for i in range(N - 1):
        T_gt_rel = np.linalg.inv(gt_poses[i]) @ gt_poses[i + 1]
        T_est_rel = np.linalg.inv(est_poses[i]) @ est_poses[i + 1]

        R_gt_rel = T_gt_rel[:3, :3]
        R_est_rel = T_est_rel[:3, :3]
        t_gt_rel = T_gt_rel[:3, 3]
        t_est_rel = T_est_rel[:3, 3]

        R_err_rel = R_est_rel @ R_gt_rel.transpose(-1, -2)
        rel_rot_errs.append(rotation_angle_deg(R_err_rel))
        rel_trans_errs.append(np.linalg.norm(t_est_rel - t_gt_rel))

    rel_rot_errs = np.array(rel_rot_errs)
    rel_trans_errs = np.array(rel_trans_errs)
    return {
        "mean_abs_rot_deg": rot_err_abs.mean().item(),
        "mean_abs_trans": trans_err_abs.mean().item(),
        "mean_rel_rot_deg": rel_rot_errs.mean().item(),
        "mean_rel_trans": rel_trans_errs.mean().item(),
        "ate_rmse": ate_rmse.item(),
    }


def get_coarsest_pose(point_cloud_previous, point_cloud_current, pose_previous):
    previous_median = np.median(point_cloud_previous, axis=0)
    current_median = np.median(point_cloud_current, axis=0)
    translation_coarse = current_median - previous_median

    pose_coarse = pose_previous.copy()
    pose_coarse[:3, 3] += translation_coarse

    point_cloud_previous_aligned = point_cloud_previous + translation_coarse

    return pose_coarse, point_cloud_previous_aligned


def get_aligned_pc(pc, pose_rel):
    aligned_pc = (pose_rel[:3, :3] @ pc.T).T + pose_rel[:3, 3]
    return aligned_pc


def numpy_point_cloud_from_arrays(
    points_xyz: np.ndarray,
    colors_rgb: np.ndarray | None = None,
) -> o3d.geometry.PointCloud:
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"points_xyz must have shape [N, 3], got {points_xyz.shape}")

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))

    if colors_rgb is not None:
        if colors_rgb.shape != points_xyz.shape:
            raise ValueError(
                f"colors_rgb must match points_xyz shape {points_xyz.shape}, got {colors_rgb.shape}"
            )
        point_cloud.colors = o3d.utility.Vector3dVector(colors_rgb.astype(np.float64))

    return point_cloud


def make_transform(
    rotation_matrix: np.ndarray,
    translation_vector: np.ndarray,
) -> np.ndarray:
    transform_matrix = np.eye(4, dtype=np.float64)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation_vector
    return transform_matrix


def invert_transform(transform_matrix: np.ndarray) -> np.ndarray:
    rotation_matrix = transform_matrix[:3, :3]
    translation_vector = transform_matrix[:3, 3]

    transform_inverse = np.eye(4, dtype=np.float64)
    transform_inverse[:3, :3] = rotation_matrix.T
    transform_inverse[:3, 3] = -rotation_matrix.T @ translation_vector
    return transform_inverse


def build_rotation_hypotheses(
    angle_step_degrees: float = 3.0,
    max_angle_degrees: float = 9.0,
    include_full_axis_combinations: bool = True,
) -> list[np.ndarray]:
    """
    Build a small SO(3) hypothesis set around identity.

    Example with step=3, max=9:
        angles = [-9, -6, -3, 0, 3, 6, 9]

    Returns:
        List of [3, 3] rotation matrices.
    """
    sampled_angles_deg = np.arange(
        -max_angle_degrees,
        max_angle_degrees + 1e-9,
        angle_step_degrees,
        dtype=np.float64,
    )

    rotation_hypotheses = []

    # Identity first
    rotation_hypotheses.append(np.eye(3, dtype=np.float64))

    # Single-axis perturbations
    for roll_deg in sampled_angles_deg:
        if abs(roll_deg) > 1e-12:
            rotation_hypotheses.append(
                R.from_euler("x", roll_deg, degrees=True).as_matrix()
            )

    for pitch_deg in sampled_angles_deg:
        if abs(pitch_deg) > 1e-12:
            rotation_hypotheses.append(
                R.from_euler("y", pitch_deg, degrees=True).as_matrix()
            )

    for yaw_deg in sampled_angles_deg:
        if abs(yaw_deg) > 1e-12:
            rotation_hypotheses.append(
                R.from_euler("z", yaw_deg, degrees=True).as_matrix()
            )

    # Full combinations around identity
    if include_full_axis_combinations:
        for roll_deg, pitch_deg, yaw_deg in itertools.product(
            sampled_angles_deg, sampled_angles_deg, sampled_angles_deg
        ):
            if (
                abs(roll_deg) < 1e-12
                and abs(pitch_deg) < 1e-12
                and abs(yaw_deg) < 1e-12
            ):
                continue
            rotation_hypotheses.append(
                R.from_euler(
                    "xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True
                ).as_matrix()
            )

    # Deduplicate numerically
    deduplicated_rotations = []
    for candidate_rotation in rotation_hypotheses:
        is_duplicate = any(
            np.allclose(candidate_rotation, kept_rotation, atol=1e-10)
            for kept_rotation in deduplicated_rotations
        )
        if not is_duplicate:
            deduplicated_rotations.append(candidate_rotation)

    return deduplicated_rotations


def rank_icp_result(
    registration_result: o3d.pipelines.registration.RegistrationResult,
) -> tuple[float, float]:
    """
    Higher fitness is better, lower RMSE is better.
    We return a tuple suitable for lexicographic comparison.
    """
    return (registration_result.fitness, -registration_result.inlier_rmse)


def run_explorative_icp_with_centering(
    source_points_xyz: np.ndarray,
    target_points_xyz: np.ndarray,
    source_colors_rgb: np.ndarray | None = None,
    target_colors_rgb: np.ndarray | None = None,
    max_correspondence_distance: float = 0.05,
    coarse_iterations: int = 10,
    final_iterations: int = 50,
    angle_step_degrees: float = 4.5,
    max_angle_degrees: float = 9.0,
    include_full_axis_combinations: bool = False,
    coarse_voxel_size: float | None = None,
    final_voxel_size: float | None = None,
    top_k_finalists: int = 3,
) -> dict:
    source_points_xyz = np.asarray(source_points_xyz, dtype=np.float64)
    target_points_xyz = np.asarray(target_points_xyz, dtype=np.float64)

    if source_points_xyz.ndim != 2 or source_points_xyz.shape[1] != 3:
        raise ValueError(
            f"source_points_xyz must have shape [N, 3], got {source_points_xyz.shape}"
        )
    if target_points_xyz.ndim != 2 or target_points_xyz.shape[1] != 3:
        raise ValueError(
            f"target_points_xyz must have shape [N, 3], got {target_points_xyz.shape}"
        )

    # Center with NumPy once.
    source_centroid = source_points_xyz.mean(axis=0)
    target_centroid = target_points_xyz.mean(axis=0)

    centered_source_points_xyz = source_points_xyz - source_centroid
    centered_target_points_xyz = target_points_xyz - target_centroid

    # Build coarse clouds.
    coarse_source_point_cloud = numpy_point_cloud_from_arrays(
        centered_source_points_xyz, source_colors_rgb
    )
    coarse_target_point_cloud = numpy_point_cloud_from_arrays(
        centered_target_points_xyz, target_colors_rgb
    )

    if coarse_voxel_size is not None and coarse_voxel_size > 0.0:
        coarse_source_point_cloud = coarse_source_point_cloud.voxel_down_sample(
            coarse_voxel_size
        )
        coarse_target_point_cloud = coarse_target_point_cloud.voxel_down_sample(
            coarse_voxel_size
        )

    # Build final clouds separately so coarse downsampling does not affect final ICP.
    final_source_point_cloud = numpy_point_cloud_from_arrays(
        centered_source_points_xyz, source_colors_rgb
    )
    final_target_point_cloud = numpy_point_cloud_from_arrays(
        centered_target_points_xyz, target_colors_rgb
    )

    if final_voxel_size is not None and final_voxel_size > 0.0:
        final_source_point_cloud = final_source_point_cloud.voxel_down_sample(
            final_voxel_size
        )
        final_target_point_cloud = final_target_point_cloud.voxel_down_sample(
            final_voxel_size
        )

    rotation_hypotheses = build_rotation_hypotheses(
        angle_step_degrees=angle_step_degrees,
        max_angle_degrees=max_angle_degrees,
        include_full_axis_combinations=include_full_axis_combinations,
    )

    coarse_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-3,
        relative_rmse=1e-3,
        max_iteration=coarse_iterations,
    )
    final_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6,
        relative_rmse=1e-6,
        max_iteration=final_iterations,
    )
    estimation_method = (
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    coarse_candidates = []

    for initial_rotation_matrix in rotation_hypotheses:
        initial_transform = np.eye(4, dtype=np.float64)
        initial_transform[:3, :3] = initial_rotation_matrix

        coarse_icp_result = o3d.pipelines.registration.registration_icp(
            coarse_source_point_cloud,
            coarse_target_point_cloud,
            max_correspondence_distance,
            initial_transform,
            estimation_method,
            coarse_criteria,
        )

        coarse_candidates.append(
            (
                rank_icp_result(coarse_icp_result),
                initial_transform,
                coarse_icp_result.transformation,
            )
        )

    coarse_candidates.sort(key=lambda item: item[0], reverse=True)
    finalist_candidates = coarse_candidates[:top_k_finalists]

    best_final_result = None
    best_final_score = -np.inf
    best_initial_transform = None

    for _, initial_transform, finalist_seed_transform in finalist_candidates:
        final_icp_result = o3d.pipelines.registration.registration_icp(
            final_source_point_cloud,
            final_target_point_cloud,
            max_correspondence_distance,
            finalist_seed_transform,
            estimation_method,
            final_criteria,
        )

        final_score = rank_icp_result(final_icp_result)
        if final_score[0] > best_final_score:
            best_final_result = final_icp_result
            best_final_score = final_score[0]
            best_initial_transform = initial_transform

    centered_source_to_centered_target = best_final_result.transformation

    transform_source_to_target = np.eye(4, dtype=np.float64)
    transform_source_to_target[:3, :3] = centered_source_to_centered_target[:3, :3]
    transform_source_to_target[:3, 3] = (
        target_centroid
        + centered_source_to_centered_target[:3, 3]
        - centered_source_to_centered_target[:3, :3] @ source_centroid
    )

    return {
        "transform_source_to_target": transform_source_to_target,
        "fitness": best_final_result.fitness,
        "inlier_rmse": best_final_result.inlier_rmse,
        "icp_result": best_final_result,
        "num_rotation_hypotheses": len(rotation_hypotheses),
        "best_initial_transform": best_initial_transform,
    }


def apply_transform_to_points(
    points_xyz: np.ndarray, transform_matrix: np.ndarray
) -> np.ndarray:
    """
    Apply a 4x4 rigid transform to [N, 3] points.
    """
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    homogeneous_points = np.concatenate(
        [points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    transformed_points_h = (transform_matrix @ homogeneous_points.T).T
    return transformed_points_h[:, :3]


if __name__ == "__main__":
    pass
