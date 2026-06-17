"""ReLiFT-6DoF — production tracking pipeline.

Per frame:
  1. Load the light-field frame.
  2. Build a surface light field and run reflection separation → diffuse view
     (cached to disk; separation is slow).
  3. LoFTR-match consecutive diffuse views → relative pose → absolute pose.

Runs over every {depth source} × {split} × {reflectivity} × {sequence}, rebases
the estimated trajectory to the GT frame-0 pose, and saves it as <sequence>.npy.
"""

import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from icp import rebase_poses
from loftr_baseline import DEPTH_ZNEAR, DEPTH_ZFAR
from loftr_wrapper import LoftrRunner
from src.dataset import LFDataset
from src.photometric import (
    PhotometricRefineViewer,
    RefineConfig,
    refine_pose_photometric,
)
from src.pose import track_pose
from src.reflection import central_view, frame_diffuse
from src.surface_light_field import SurfaceLightField

logging.basicConfig(
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

# ── configuration ──────────────────────────────────────────────────────────────
DATASET_ROOT = "/home/ngoncharov/SpecTrack_dataset"
EXP_NAME = "results_ours_refined_"
CACHE_ROOT = "cache/diffuse"
SEPARATION_ITERS = 300
USE_REFLECTION_SEPARATION = True  # False → LoFTR on the raw central view
USE_PHOTOMETRIC_REFINE = True  # True → photometric pose refinement after coarse
ENABLE_VIS = True  # True → open viser viewer during refinement (manual inspection)
# Relighting only turns on once the estimated alpha has settled: two consecutive
# per-frame estimates within this tolerance ⇒ "stable". Until then we track as
# pure diffuse and discard the (still-unreliable) env map.
ALPHA_STABLE_TOL = 0.01
# Diffuse-fraction (alpha) source. The view-variance estimator is reliable at low/
# mid reflectivity but unstable per-sequence at the mirror extreme (sticks at ~0.5
# for some cube_1.0 seqs), which blends LoFTR into a near-mirror's tracking and
# regresses the coarse pose (cube_1.0 5.3°→14° agg). Pinning alpha to the dataset's
# known reflectivity (as the separation baseline does) keeps the coarse strong;
# the estimator stays available (PIN_ALPHA=False) as an ablation. See [[alpha-estimation]].
PIN_ALPHA = True

# All photometric-refine hyperparameters live here (see RefineConfig). The
# refinement is a per-frame OVERLAY (not fed forward), so free translation is safe
# and lets rotation stay well-conditioned. Two stages: diffuse (fine) then relight
# (final, reflective frames only).
REFINE_CFG = RefineConfig()


def _rot_err_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    R = Ra @ Rb.T
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def _pose_err(pose: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """(rotation error °, translation error m) of an absolute pose vs GT."""
    return _rot_err_deg(pose[:3, :3], gt[:3, :3]), float(
        np.linalg.norm(pose[:3, 3] - gt[:3, 3])
    )


def _build_pc(depth: np.ndarray, mask: np.ndarray, rgb: np.ndarray, K: np.ndarray):
    """Backproject all masked pixels into camera-space points + linear RGB colors."""
    H, W = depth.shape
    ys, xs = np.where(mask & (depth > DEPTH_ZNEAR) & (depth < DEPTH_ZFAR))
    d = depth[ys, xs]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pts = np.stack([(xs - cx) * d / fx, (ys - cy) * d / fy, d], axis=1)
    colors = rgb[ys, xs].astype(np.float32)  # already linear [0, 1]
    return pts, colors


DEPTH_SOURCES = ["gt"]  # "gt" | "synth"
SPLIT_PREFIXES = ["cube", "objects"]
REFLECTIVITIES = ["0.0", "0.5", "0.7", "1.0"]


def track_sequence(
    seq_path: str,
    results_dir: str,
    cache_dir: str,
    sequence_name: str,
    alpha: float | None,
    depth_source: str,
    loftr: LoftrRunner,
    rng: np.random.Generator,
    separate: bool,
    refine: bool = False,
    viewer: PhotometricRefineViewer | None = None,
    max_frames: int | None = None,
    refine_cfg: RefineConfig | None = None,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    dataset = LFDataset(seq_path, depth_source=depth_source)
    s_size, t_size = dataset.metadata["n_views"]

    gt_poses: list[np.ndarray] = []
    est_poses: list[np.ndarray] = []  # reported trajectory (refined when refine=True)
    coarse_poses: list[np.ndarray] = []  # tracking backbone, FED FORWARD (drift-safe)
    coarse_errs: list[tuple[float, float]] = []  # (rot°, trans m) per frame
    refined_errs: list[tuple[float, float]] = []
    prev = None  # (view, depth, mask, pc, color, env_map, slf)
    prev_env = None  # accumulated env map warm-start for separation
    alpha_history: list[float] = []  # per-frame alpha stats, carried forward
    alpha_stable = False  # latched: True once two alphas in a row agree
    prev_alpha_i: float | None = None  # last frame's raw alpha estimate

    with tqdm(
        dataset, desc="  frames", unit="fr", leave=False, dynamic_ncols=True
    ) as bar:
        for i, frame in enumerate(bar):
            if max_frames is not None and i >= max_frames:
                break
            gt_poses.append(frame["object_pose"].cpu().numpy())

            depth = frame["depth"]
            mask = frame["masks"][s_size // 2, t_size // 2]

            if separate:
                bar.set_postfix(fr=i, stage="separate")
                # Alpha is trusted once it settles: two consecutive estimates within
                # ALPHA_STABLE_TOL (a pinned alpha is trusted immediately). Until
                # then, withhold the env map as a warm-start too, so a frame fit
                # against an unreliable alpha never contaminates the map that
                # accumulation resumes from once alpha does stabilize. Latched: once
                # stable we keep relighting and accumulating the env map.
                was_stable = alpha_stable  # stability state entering this frame
                view, prev_env, slf, alpha_i, alpha_history = frame_diffuse(
                    frame=frame,
                    mask=mask,
                    depth=depth,
                    alpha=alpha,  # None → estimate on the fly from the SLF
                    s_size=s_size,
                    t_size=t_size,
                    cache_path=os.path.join(cache_dir, f"diffuse_{i:04d}.png"),
                    iterations=SEPARATION_ITERS,
                    verbose=True,
                    previous_environment_map=prev_env if was_stable else None,
                    alpha_stat_history=alpha_history,
                )
                if not alpha_stable:
                    if alpha is not None or (
                        prev_alpha_i is not None
                        and abs(alpha_i - prev_alpha_i) <= ALPHA_STABLE_TOL
                    ):
                        alpha_stable = True
                prev_alpha_i = alpha_i
                # Tracking ALWAYS uses the current alpha estimate: as soon as
                # reflectivity is suspected, track_pose blends in geometry-only ICP
                # (a near-mirror's separated diffuse is too weak for LoFTR). The
                # alpha-stability gate governs only the env-map-dependent RELIGHTING:
                # we withhold the env map until alpha settles so a noisy early map
                # can't mislead the refinement. (Previously alpha_eff was forced to
                # 1.0 until stable → pure LoFTR on a mirror → the early trajectory
                # drifted and the feed-forward backbone inherited it.)
                alpha_eff = alpha_i
                env_curr = prev_env if alpha_stable else None
            else:
                view = central_view(frame, s_size, t_size)
                env_curr = None
                slf = SurfaceLightField.from_frame(frame, mask, depth, s_size, t_size)
                # No separation → treat as fully diffuse (LoFTR-only) unless pinned.
                alpha_i = 1.0 if alpha is None else alpha
                alpha_eff = alpha_i
                alpha_stable = True
            depth_np = depth.cpu().numpy()
            mask_np = (mask > 0).cpu().numpy()
            K_np = frame["camera_matrix"].cpu().numpy().astype(np.float64)
            pc, color = _build_pc(depth_np, mask_np, view, K_np)

            if i == 0:
                est_poses.append(gt_poses[0])
                coarse_poses.append(gt_poses[0])
            else:
                bar.set_postfix(fr=i, stage="loftr")
                # Tracking backbone is fed forward from the COARSE pose, never the
                # refined one: a per-frame photometric correction has a small
                # systematic bias that compounds geometrically if recycled as the
                # next frame's anchor (observed: 0.9°→6.7° over a diffuse seq).
                # Refinement is therefore an overlay on the *reported* pose only.
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
                    alpha=alpha_eff,
                    pc_prev=prev[3],
                    pc_curr=pc,
                    color_prev=prev[4],
                    color_curr=color,
                    rng=rng,
                )

                coarse_poses.append(coarse_pose)

                if refine:
                    bar.set_postfix(fr=i, stage="photometric")
                    if viewer is not None:
                        viewer.reset_frame(i)
                    # Anchor the refinement at the prev backbone pose (coarse_poses[-2]),
                    # which is where slf_prev's geometry actually lives — NOT the
                    # reported/refined prev pose.
                    refined_pose, _ = refine_pose_photometric(
                        slf_prev=prev[6],
                        slf_curr=slf,
                        env_map_prev=prev[5],
                        env_map_curr=env_curr,
                        abs_pose_prev=coarse_poses[-2],
                        pose_coarse=coarse_pose,
                        alpha=alpha_eff,
                        cfg=refine_cfg,
                        viewer=viewer,
                        gt_pose_curr=gt_poses[i],
                    )
                    est_poses.append(refined_pose)

                    # Feed the relit pose forward into the backbone (drift-correcting
                    # on reflective frames); diffuse refine stays an overlay only.
                    relight_ran = (
                        refine_cfg.relight
                        and prev[5] is not None
                        and alpha_eff < refine_cfg.relight_alpha_max
                    )
                    if relight_ran and refine_cfg.relight_feed_forward:
                        coarse_poses[-1] = refined_pose

                    # ── per-frame metrics: does refine beat the LoFTR coarse? ──
                    cr, ct = _pose_err(coarse_pose, gt_poses[i])
                    rr, rt = _pose_err(refined_pose, gt_poses[i])
                    coarse_errs.append((cr, ct))
                    refined_errs.append((rr, rt))
                    mc = np.mean(coarse_errs, axis=0)
                    mr = np.mean(refined_errs, axis=0)
                    flag = ("↑rot" if rr > cr + 1e-6 else "") + (
                        " ↑trans" if rt > ct + 1e-6 else ""
                    )
                    bar.write(
                        f"  f{i:3d}: loftr {cr:5.2f}° {ct * 1000:6.2f}mm "
                        f"→ refined {rr:5.2f}° {rt * 1000:6.2f}mm  "
                        f"| mean loftr {mc[0]:.2f}°/{mc[1] * 1000:.1f}mm "
                        f"refined {mr[0]:.2f}°/{mr[1] * 1000:.1f}mm  {flag}"
                    )
                else:
                    est_poses.append(coarse_pose)

            prev = (view, depth_np, mask_np, pc, color, env_curr, slf)

    if refined_errs:
        c = np.array(coarse_errs)
        r = np.array(refined_errs)
        win_r = float((r[:, 0] < c[:, 0]).mean() * 100)
        win_t = float((r[:, 1] < c[:, 1]).mean() * 100)
        logging.info(
            "%s: refine vs loftr — rot %.2f°→%.2f° (better %.0f%%)  "
            "trans %.2f→%.2fmm (better %.0f%%)",
            sequence_name,
            c[:, 0].mean(),
            r[:, 0].mean(),
            win_r,
            c[:, 1].mean() * 1000,
            r[:, 1].mean() * 1000,
            win_t,
        )

    est = rebase_poses(np.stack(gt_poses), np.stack(est_poses))
    out_path = os.path.join(results_dir, f"{sequence_name}.npy")
    np.save(out_path, est)
    logging.info("%s: %s → %s", sequence_name, est.shape, out_path)
    return coarse_errs, refined_errs


def build_work_list() -> list[dict]:
    work = []
    for depth_source in DEPTH_SOURCES:
        for split_prefix in SPLIT_PREFIXES:
            for reflectivity in REFLECTIVITIES:
                split_dir = f"{DATASET_ROOT}/{split_prefix}_{reflectivity}"
                if not os.path.isdir(split_dir):
                    logging.warning("Split not found, skipping: %s", split_dir)
                    continue
                tag = f"{depth_source}/{split_prefix}_{reflectivity}"
                for sequence_name in sorted(os.listdir(split_dir)):
                    seq_path = os.path.join(split_dir, sequence_name)
                    if os.path.isdir(seq_path):
                        work.append(
                            {
                                "depth_source": depth_source,
                                "reflectivity": reflectivity,
                                "sequence_name": sequence_name,
                                "seq_path": seq_path,
                                "results_dir": f"{EXP_NAME}/{tag}",
                                "cache_dir": f"{CACHE_ROOT}/{tag}/{sequence_name}",
                                "tag": f"{tag}/{sequence_name}",
                            }
                        )
    return work


def main() -> None:
    loftr = LoftrRunner()
    rng = np.random.default_rng(seed=42)
    work = build_work_list()

    viewer = None
    if USE_PHOTOMETRIC_REFINE and ENABLE_VIS:
        viewer = PhotometricRefineViewer(port=8081)

    with tqdm(work, desc="sequences", unit="seq", dynamic_ncols=True) as bar:
        for item in bar:
            bar.set_postfix_str(item["tag"])
            os.makedirs(item["results_dir"], exist_ok=True)

            out_path = os.path.join(item["results_dir"], f"{item['sequence_name']}.npy")
            if os.path.exists(out_path):
                logging.info("%s: already done, skipping", item["tag"])
                continue

            try:
                track_sequence(
                    seq_path=item["seq_path"],
                    results_dir=item["results_dir"],
                    cache_dir=item["cache_dir"],
                    sequence_name=item["sequence_name"],
                    alpha=(1.0 - float(item["reflectivity"])) if PIN_ALPHA else None,
                    depth_source=item["depth_source"],
                    loftr=loftr,
                    rng=rng,
                    separate=USE_REFLECTION_SEPARATION,
                    refine=USE_PHOTOMETRIC_REFINE,
                    viewer=viewer,
                    refine_cfg=REFINE_CFG,
                )
            except Exception:
                logging.exception("%s: FAILED", item["tag"])

    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
