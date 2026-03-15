import numpy as np
import open3d as o3d
from src.utilities import Visualizer
import torch
from icp import rebase_poses, pose_errors, icp_track
from scipy.spatial.transform import Rotation as R
import itertools
from src.dataset import LFDataset
from src.utilities import backproject_depth_to_pointcloud


def get_coarsest_pose(pc, pose_prev):
    pose_next = np.copy(pose_prev)
    pose_next[:3, 3] = np.median(pc, axis=0)
    return pose_next


def pc_coarsest_init(pc_prev, pc):
    pc_prev += np.median(pc, axis=0) - np.median(pc_prev, axis=0)
    return pc_prev


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
    coarse_iterations: int = 20,
    final_iterations: int = 80,
    angle_step_degrees: float = 3.0,
    max_angle_degrees: float = 9.0,
    include_full_axis_combinations: bool = True,
    voxel_size: float | None = None,
    top_k_finalists: int = 5,
) -> dict:
    """
    Multi-start ICP with centered point clouds and rotation hypothesis search.

    Strategy:
        1. Center source and target separately.
        2. Try many initial rotation hypotheses around identity.
        3. Run short ICP from each hypothesis.
        4. Keep the best few hypotheses.
        5. Re-run longer ICP from those finalists.
        6. Return the best transform in the original coordinate system.
    """
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

    source_point_cloud = numpy_point_cloud_from_arrays(
        source_points_xyz, source_colors_rgb
    )
    target_point_cloud = numpy_point_cloud_from_arrays(
        target_points_xyz, target_colors_rgb
    )

    if voxel_size is not None and voxel_size > 0.0:
        source_point_cloud = source_point_cloud.voxel_down_sample(voxel_size)
        target_point_cloud = target_point_cloud.voxel_down_sample(voxel_size)

    source_centroid = np.asarray(source_point_cloud.points).mean(axis=0)
    target_centroid = np.asarray(target_point_cloud.points).mean(axis=0)

    source_to_center_transform = make_transform(
        rotation_matrix=np.eye(3, dtype=np.float64),
        translation_vector=-source_centroid,
    )
    target_to_center_transform = make_transform(
        rotation_matrix=np.eye(3, dtype=np.float64),
        translation_vector=-target_centroid,
    )

    centered_source_point_cloud = source_point_cloud.transform(
        source_to_center_transform.copy()
    )
    centered_target_point_cloud = target_point_cloud.transform(
        target_to_center_transform.copy()
    )

    rotation_hypotheses = build_rotation_hypotheses(
        angle_step_degrees=angle_step_degrees,
        max_angle_degrees=max_angle_degrees,
        include_full_axis_combinations=include_full_axis_combinations,
    )

    coarse_candidates = []

    coarse_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=coarse_iterations
    )
    final_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=final_iterations
    )
    estimation_method = (
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    for initial_rotation_matrix in rotation_hypotheses:
        initial_transform = make_transform(
            rotation_matrix=initial_rotation_matrix,
            translation_vector=np.zeros(3, dtype=np.float64),
        )

        coarse_icp_result = o3d.pipelines.registration.registration_icp(
            centered_source_point_cloud,
            centered_target_point_cloud,
            max_correspondence_distance,
            initial_transform,
            estimation_method,
            coarse_criteria,
        )

        coarse_candidates.append(
            {
                "initial_transform": initial_transform,
                "coarse_result": coarse_icp_result,
                "score": rank_icp_result(coarse_icp_result),
            }
        )

    coarse_candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
    finalist_candidates = coarse_candidates[:top_k_finalists]

    best_final_result = None
    best_final_score = None
    best_initial_transform = None

    for finalist_candidate in finalist_candidates:
        finalist_seed_transform = finalist_candidate["coarse_result"].transformation

        final_icp_result = o3d.pipelines.registration.registration_icp(
            centered_source_point_cloud,
            centered_target_point_cloud,
            max_correspondence_distance,
            finalist_seed_transform,
            estimation_method,
            final_criteria,
        )

        final_score = rank_icp_result(final_icp_result)

        if best_final_result is None or final_score > best_final_score:
            best_final_result = final_icp_result
            best_final_score = final_score
            best_initial_transform = finalist_candidate["initial_transform"]

    centered_source_to_centered_target = best_final_result.transformation
    center_to_target_original_transform = invert_transform(target_to_center_transform)

    transform_source_to_target = (
        center_to_target_original_transform
        @ centered_source_to_centered_target
        @ source_to_center_transform
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
    paths = "/home/ngoncharov/cvpr2026/ycbv-eoat-lf/dataset_simple_box_reflective_full_0.0/bleach0"
    dataset = LFDataset(paths)
    s_size, t_size = dataset.metadata["n_views"]

    gt_poses = []
    est_poses = []
    v = Visualizer()
    pose_rel_prev = None
    for i, frame in enumerate(dataset):
        frame["pose"] = frame["object_pose"]
        mask = frame["masks"][s_size // 2, t_size // 2]
        img_central = frame["LF"][s_size // 2, t_size // 2]
        camera_matrix = frame["camera_matrix"]
        depth = frame["depth"]
        pc = backproject_depth_to_pointcloud(
            pixel_indices=None,
            depths=depth,
            camera_matrix=camera_matrix,
        )
        pc = pc[(mask > 0).reshape(-1)].cpu().numpy()
        color = img_central[mask > 0].reshape(-1, 3).cpu().numpy()
        if i == 0:
            pc_prev = pc
            color_prev = color
            est_poses.append(frame["pose"].cpu().numpy())
        else:
            coarsest_pose = get_coarsest_pose(pc, est_poses[-1])
            pc_prev_trans = pc_coarsest_init(pc_prev, pc)

            registration_result = run_explorative_icp_with_centering(
                source_points_xyz=pc_prev_trans,
                target_points_xyz=pc,
                source_colors_rgb=color_prev,
                target_colors_rgb=color,
                max_correspondence_distance=0.01,
            )

            coarse_pose = registration_result["transform_source_to_target"]
            pc_refined = apply_transform_to_points(pc_prev_trans, coarse_pose)
            print(pc_refined.shape, color_prev.shape)
            coarse_pose = coarse_pose @ coarsest_pose

            est_poses.append(coarse_pose)
            v.add_point_cloud(
                f"pc_coarse_{i}", pc_prev_trans, color_prev, point_size=1e-3
            )
            v.add_point_cloud(
                f"pc_aligned_{i}", pc_refined, color_prev, point_size=1e-3
            )
            v.add_point_cloud(f"pc_{i}", pc, color, point_size=1e-3)

            pc_prev = pc
            color_prev = color
    gt_poses = np.stack(gt_poses, axis=0)
    est_poses = np.stack(est_poses, axis=0)
    est_poses = rebase_poses(gt_poses, est_poses)
    print(pose_errors(gt_poses, est_poses))
    for i, (pose_est, pose_gt) in enumerate(zip(est_poses, gt_poses)):
        v.add_frame(f"{i}_est", pose_est, frames_scale=0.01)
        v.add_frame(f"{i}_gt", pose_gt, frames_scale=0.01, origin_color=(255, 255, 255))
    v.run()
