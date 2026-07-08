"""ReLiFT-6DoF — 6-DoF object tracking in light fields, robust to reflections.

Per frame: build a surface light field from the LF views → separate it into a
diffuse view + reflected environment map → coarse pose from LoFTR/ICP on the
diffuse view → photometric refinement against the relightable SLF.

Writes one (N, 4, 4) .npy trajectory of object-to-camera poses per sequence.
"""

import argparse
import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from config import (
    ALPHA_STABLE_TOL,
    LF_DEPTH_CFG,
    REFINE_CFG,
    REFINE_FEED_FORWARD,
    SEPARATION_ITERS,
)
from lf_depth import LFPlaneSweepDepth
from loftr_wrapper import LoftrRunner
from src.dataset import LFDataset
from src.photometric import PhotometricRefineViewer, refine_pose_photometric
from src.pose import DEPTH_ZFAR, DEPTH_ZNEAR, track_pose
from src.reflection import frame_diffuse

logging.basicConfig(
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

# Below this many object pixels in the central view the object is effectively
# lost: too few pixels for LoFTR/ICP to recover a non-degenerate pose. The
# remaining frames repeat the last tracked pose.
MIN_TRACK_PIXELS = 200


def _rot_diff_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = np.clip((np.trace(Ra @ Rb.T) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def _build_pc(depth: np.ndarray, mask: np.ndarray, rgb: np.ndarray, K: np.ndarray):
    """Backproject masked pixels to camera-space points + linear RGB colours."""
    ys, xs = np.where(mask & (depth > DEPTH_ZNEAR) & (depth < DEPTH_ZFAR))
    d = depth[ys, xs]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pts = np.stack([(xs - cx) * d / fx, (ys - cy) * d / fy, d], axis=1)
    return pts, rgb[ys, xs].astype(np.float32)


def track_sequence(
    seq_path: str,
    out_path: str,
    loftr: LoftrRunner,
    rng: np.random.Generator,
    depth_estimator: LFPlaneSweepDepth | None = None,
    viewer: PhotometricRefineViewer | None = None,
) -> None:
    """Track one sequence and save the estimated trajectory as (N, 4, 4) .npy."""
    dataset = LFDataset(
        seq_path, depth_source="lf" if depth_estimator is not None else "gt"
    )
    s_size, t_size = dataset.metadata["n_views"]

    est_poses: list[np.ndarray] = []
    coarse_poses: list[np.ndarray] = []
    # Independent pure-LoFTR/ICP backbone (never refined), used as the
    # feed-forward drift reference.
    backbone_poses: list[np.ndarray] = []
    prev = None  # (view, depth, mask, pc, color, env_map, slf, env_conf)
    prev_env = None
    prev_env_conf = None
    alpha_history: list[float] = []
    alpha_stable = False
    prev_alpha = None
    tracking_lost = False

    def _pad_with_last() -> None:
        est_poses.append(est_poses[-1])
        coarse_poses.append(coarse_poses[-1])
        backbone_poses.append(backbone_poses[-1])

    with tqdm(
        dataset, desc="  frames", unit="fr", leave=False, dynamic_ncols=True
    ) as bar:
        for i, frame in enumerate(bar):
            if tracking_lost:
                _pad_with_last()
                continue

            mask = frame["masks"][s_size // 2, t_size // 2]

            # The mask can lose the object (occlusion, out of frame). With too few
            # pixels there is nothing to track; stop and pad the remaining frames
            # with the last tracked pose. (Frame 0 seeds tracking, never gated.)
            n_mask_px = int(mask.sum().item())
            if i > 0 and n_mask_px < MIN_TRACK_PIXELS:
                logging.warning(
                    "lost object at frame %d (%d mask px < %d) — padding "
                    "remaining %d frames with the last tracked pose",
                    i,
                    n_mask_px,
                    MIN_TRACK_PIXELS,
                    len(dataset) - i,
                )
                tracking_lost = True
                _pad_with_last()
                continue

            if depth_estimator is not None:
                bar.set_postfix(fr=i, stage="depth")
                depth = depth_estimator.estimate(frame, mask=mask)["depth"]
            else:
                depth = frame["depth"]

            # Reflection separation. The env map and its confidence are withheld
            # until the alpha estimate stabilises: feeding an unreliable early env
            # map into separation would contaminate the accumulated map. Stability
            # is latched once reached.
            bar.set_postfix(fr=i, stage="separate")
            was_stable = alpha_stable
            view, prev_env, prev_env_conf, slf, alpha, alpha_history = frame_diffuse(
                frame=frame,
                mask=mask,
                depth=depth,
                s_size=s_size,
                t_size=t_size,
                iterations=SEPARATION_ITERS,
                previous_environment_map=prev_env if was_stable else None,
                previous_env_confidence=prev_env_conf if was_stable else None,
                alpha_stat_history=alpha_history,
            )
            if not alpha_stable:
                if prev_alpha is not None and abs(alpha - prev_alpha) <= ALPHA_STABLE_TOL:
                    alpha_stable = True
            prev_alpha = alpha
            env_curr = prev_env if alpha_stable else None
            env_conf_curr = prev_env_conf if alpha_stable else None

            depth_np = depth.cpu().numpy()
            mask_np = (mask > 0).cpu().numpy()
            K_np = frame["camera_matrix"].cpu().numpy().astype(np.float64)
            pc, color = _build_pc(depth_np, mask_np, view, K_np)

            if i == 0:
                # Tracking starts from the first-frame object pose.
                init_pose = frame["object_pose"].cpu().numpy().astype(np.float64)
                est_poses.append(init_pose)
                coarse_poses.append(init_pose)
                backbone_poses.append(init_pose)
            else:
                bar.set_postfix(fr=i, stage="track", alpha=f"{alpha:.2f}")
                coarse_pose = track_pose(
                    abs_pose_prev=coarse_poses[-1],
                    diffuse_prev=prev[0],
                    diffuse_curr=view,
                    depth_prev=prev[1],
                    depth_curr=depth_np,
                    mask_prev=prev[2],
                    mask_curr=mask_np,
                    K=K_np,
                    loftr=loftr,
                    alpha=alpha,
                    pc_prev=prev[3],
                    pc_curr=pc,
                    color_prev=prev[4],
                    color_curr=color,
                    rng=rng,
                )

                # A degenerate pose (all in-mask depth out of range → empty point
                # cloud) would crash the next frame's refinement; treat it as a
                # lost track and pad the rest.
                if not np.all(np.isfinite(coarse_pose)) or (
                    abs(np.linalg.det(coarse_pose[:3, :3])) < 1e-6
                ):
                    logging.warning(
                        "degenerate pose at frame %d — padding remaining "
                        "%d frames with the last tracked pose",
                        i,
                        len(dataset) - i,
                    )
                    tracking_lost = True
                    _pad_with_last()
                    continue

                # Propagate the independent backbone for the drift guard.
                rel = coarse_pose @ np.linalg.inv(coarse_poses[-1])
                backbone_poses.append(rel @ backbone_poses[-1])
                coarse_poses.append(coarse_pose)

                bar.set_postfix(fr=i, stage="refine", alpha=f"{alpha:.2f}")
                if viewer is not None:
                    viewer.reset_frame(i)
                refined_pose, _ = refine_pose_photometric(
                    slf_prev=prev[6],
                    slf_curr=slf,
                    env_map_prev=prev[5],
                    env_map_curr=env_curr,
                    env_conf_prev=prev[7],
                    env_conf_curr=env_conf_curr,
                    abs_pose_prev=coarse_poses[-2],
                    pose_coarse=coarse_pose,
                    alpha=alpha,
                    cfg=REFINE_CFG,
                    viewer=viewer,
                )
                est_poses.append(refined_pose)

                if REFINE_FEED_FORWARD:
                    # Drift guard: on diffuse frames (where the backbone is
                    # reliable) reset to it if the refined pose drifted too far.
                    if alpha >= REFINE_CFG.drift_reset_alpha_min:
                        ref = backbone_poses[-1]
                        d_deg = _rot_diff_deg(refined_pose[:3, :3], ref[:3, :3])
                        d_m = float(np.linalg.norm(refined_pose[:3, 3] - ref[:3, 3]))
                        if (
                            d_deg > REFINE_CFG.drift_reset_deg
                            or d_m > REFINE_CFG.drift_reset_trans
                        ):
                            refined_pose = ref.astype(np.float64).copy()
                            est_poses[-1] = refined_pose
                    coarse_poses[-1] = refined_pose

            prev = (view, depth_np, mask_np, pc, color, env_curr, slf, env_conf_curr)

    est = np.stack(est_poses)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.save(out_path, est)
    logging.info("%s → %s", est.shape, out_path)


def _is_sequence(path: str) -> bool:
    return (
        os.path.isfile(os.path.join(path, "camera_matrix.txt"))
        and os.path.isdir(os.path.join(path, "object_poses"))
        and any(d.startswith("LF_") for d in os.listdir(path))
    )


def find_sequences(data_root: str) -> list[tuple[str, str]]:
    """Return (sequence_path, sequence_name) pairs under ``data_root``.

    ``data_root`` may also point directly at a single sequence.
    """
    if _is_sequence(data_root):
        return [(data_root, os.path.basename(os.path.normpath(data_root)))]
    sequences = []
    for cur, dirs, _files in sorted(os.walk(data_root)):
        if _is_sequence(cur):
            sequences.append((cur, os.path.relpath(cur, data_root)))
            dirs[:] = []  # do not descend into frame folders
    return sequences


def main() -> None:
    parser = argparse.ArgumentParser(description="ReLiFT-6DoF tracker")
    parser.add_argument(
        "--data",
        required=True,
        help="Dataset root (or a single sequence directory).",
    )
    parser.add_argument(
        "--out", default="results", help="Output directory for .npy trajectories."
    )
    parser.add_argument(
        "--depth",
        choices=["lf", "gt"],
        default="lf",
        help="Depth source: 'lf' estimates depth from the light field (default), "
        "'gt' reads the dataset depth maps.",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Open a viser viewer (http://localhost:8080) showing the photometric "
        "refinement live.",
    )
    args = parser.parse_args()

    sequences = find_sequences(args.data)
    if not sequences:
        raise SystemExit(f"No sequences found under {args.data}")
    logging.info("%d sequence(s), depth=%s", len(sequences), args.depth)

    loftr = LoftrRunner()
    rng = np.random.default_rng(seed=42)
    depth_estimator = LFPlaneSweepDepth(LF_DEPTH_CFG) if args.depth == "lf" else None
    viewer = PhotometricRefineViewer(port=8080) if args.vis else None

    with tqdm(sequences, desc="sequences", unit="seq", dynamic_ncols=True) as bar:
        for seq_path, seq_name in bar:
            bar.set_postfix_str(seq_name)
            out_path = os.path.join(args.out, f"{seq_name}.npy")
            if os.path.exists(out_path):
                logging.info("%s: already tracked, skipping", seq_name)
                continue
            try:
                track_sequence(
                    seq_path=seq_path,
                    out_path=out_path,
                    loftr=loftr,
                    rng=rng,
                    depth_estimator=depth_estimator,
                    viewer=viewer,
                )
            except Exception:
                logging.exception("%s: FAILED", seq_name)


if __name__ == "__main__":
    main()
