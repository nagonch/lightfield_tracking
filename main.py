"""ReLiFT-6DoF — 6-DoF reflective-object tracking pipeline.

Per frame: load the LF frame → build SLF → reflection separation → diffuse view →
LoFTR coarse pose → photometric refinement.  Runs over every split × reflectivity ×
sequence, rebases the estimated trajectory to the GT frame-0 pose, and saves as .npy.
"""

import argparse
import logging
import os
from dataclasses import replace

import numpy as np
import torch
from tqdm import tqdm

from config import (
    DATASET_ROOT,
    CACHE_ROOT,
    SEPARATION_ITERS,
    USE_REFLECTION_SEPARATION,
    ENABLE_VIS,
    ALPHA_STABLE_TOL,
    PIN_ALPHA,
    REFINE_FEED_FORWARD,
    DEPTH_SOURCES,
    SPLIT_PREFIXES,
    REFLECTIVITIES,
    REFINE_CFG,
)
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


def _rot_err_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = np.clip((np.trace(Ra @ Rb.T) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def _pose_err(pose: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """(rotation error °, translation error m) vs GT."""
    return _rot_err_deg(pose[:3, :3], gt[:3, :3]), float(
        np.linalg.norm(pose[:3, 3] - gt[:3, 3])
    )


def _build_pc(depth: np.ndarray, mask: np.ndarray, rgb: np.ndarray, K: np.ndarray):
    """Backproject masked pixels to camera-space points + linear RGB colours."""
    H, W = depth.shape
    ys, xs = np.where(mask & (depth > DEPTH_ZNEAR) & (depth < DEPTH_ZFAR))
    d = depth[ys, xs]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pts = np.stack([(xs - cx) * d / fx, (ys - cy) * d / fy, d], axis=1)
    return pts, rgb[ys, xs].astype(np.float32)


def _load_env_linear(path: str) -> torch.Tensor:
    """Load an sRGB env map as linearised [H, W, 3] float32 on CUDA."""
    from PIL import Image
    from utils import srgb_to_linear

    raw = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return srgb_to_linear(torch.from_numpy(raw)).cuda().float()


def _gt_diffuse_points(gt_ds, idx: int, mask_bool: torch.Tensor) -> torch.Tensor:
    """Central view of the diffuse-only (0.0) render at the masked surface points."""
    s, t = gt_ds.metadata["n_views"]
    central = gt_ds[idx]["LF"][s // 2, t // 2].cuda().float()
    return central.reshape(-1, 3)[mask_bool.reshape(-1)].clone()


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
    gt_refine: bool = False,
    gt0_seq_path: str | None = None,
    reflectivity: float = 0.0,
    gt_env: torch.Tensor | None = None,
    feed_forward: bool = False,
    collect: dict | None = None,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    dataset = LFDataset(seq_path, depth_source=depth_source)
    s_size, t_size = dataset.metadata["n_views"]

    # GT-refine: the fully-diffuse (0.0) dataset provides ground-truth per-point
    # diffuse colours to replace the separated estimate in the photometric refine.
    gt_ds0 = LFDataset(gt0_seq_path, depth_source="gt") if gt_refine else None
    gt_alpha = 1.0 - reflectivity

    gt_poses: list[np.ndarray] = []
    est_poses: list[np.ndarray] = []
    coarse_poses: list[np.ndarray] = []
    # Independent pure-LoFTR backbone (never refined) used as a drift reference.
    loftr_ref_poses: list[np.ndarray] = []
    coarse_errs: list[tuple[float, float]] = []
    refined_errs: list[tuple[float, float]] = []
    prev = None  # (view, depth, mask, pc, color, env_map, slf, env_conf)
    prev_env = None
    prev_env_conf = None
    alpha_history: list[float] = []
    alpha_stable = False
    prev_alpha_i: float | None = None

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
                # Env map and env conf are withheld until alpha stabilises: feeding
                # an unreliable early env map into separation would contaminate the
                # accumulated map once alpha does converge. Alpha stability is latched.
                was_stable = alpha_stable
                view, prev_env, prev_env_conf, slf, alpha_i, alpha_history = (
                    frame_diffuse(
                        frame=frame,
                        mask=mask,
                        depth=depth,
                        alpha=alpha,
                        s_size=s_size,
                        t_size=t_size,
                        cache_path=os.path.join(cache_dir, f"diffuse_{i:04d}.png"),
                        iterations=SEPARATION_ITERS,
                        verbose=True,
                        previous_environment_map=prev_env if was_stable else None,
                        previous_env_confidence=prev_env_conf if was_stable else None,
                        alpha_stat_history=alpha_history,
                    )
                )
                if not alpha_stable:
                    if alpha is not None or (
                        prev_alpha_i is not None
                        and abs(alpha_i - prev_alpha_i) <= ALPHA_STABLE_TOL
                    ):
                        alpha_stable = True
                prev_alpha_i = alpha_i
                alpha_eff = alpha_i
                env_curr = prev_env if alpha_stable else None
                env_conf_curr = prev_env_conf if alpha_stable else None
            else:
                view = central_view(frame, s_size, t_size)
                env_curr = None
                env_conf_curr = None
                slf = SurfaceLightField.from_frame(frame, mask, depth, s_size, t_size)
                alpha_i = 1.0 if alpha is None else alpha
                alpha_eff = alpha_i
                alpha_stable = True

            if gt_refine:
                slf.diffuse_colors = _gt_diffuse_points(gt_ds0, i, mask > 0)
                env_curr = gt_env
                env_conf_curr = None

            depth_np = depth.cpu().numpy()
            mask_np = (mask > 0).cpu().numpy()
            K_np = frame["camera_matrix"].cpu().numpy().astype(np.float64)
            pc, color = _build_pc(depth_np, mask_np, view, K_np)

            if i == 0:
                est_poses.append(gt_poses[0])
                coarse_poses.append(gt_poses[0])
                loftr_ref_poses.append(gt_poses[0])
            else:
                bar.set_postfix(fr=i, stage="loftr")
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

                # Propagate the independent LoFTR reference for the drift guard.
                rel = coarse_pose @ np.linalg.inv(coarse_poses[-1])
                loftr_ref_poses.append(rel @ loftr_ref_poses[-1])
                coarse_poses.append(coarse_pose)

                if refine:
                    bar.set_postfix(fr=i, stage="photometric")
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
                        alpha=(gt_alpha if gt_refine else alpha_eff),
                        cfg=refine_cfg,
                        viewer=viewer,
                        gt_pose_curr=gt_poses[i],
                    )
                    est_poses.append(refined_pose)

                    relight_ran = (
                        refine_cfg.relight
                        and prev[5] is not None
                        and alpha_eff < refine_cfg.relight_alpha_max
                    )
                    drift_reset = False
                    if feed_forward or (
                        relight_ran and refine_cfg.relight_feed_forward
                    ):
                        # Drift guard: only on diffuse frames where LoFTR is reliable.
                        alpha_guard = gt_alpha if gt_refine else alpha_eff
                        if alpha_guard >= refine_cfg.drift_reset_alpha_min:
                            lref = loftr_ref_poses[-1]
                            d_deg = _rot_err_deg(refined_pose[:3, :3], lref[:3, :3])
                            d_mm = (
                                float(np.linalg.norm(refined_pose[:3, 3] - lref[:3, 3]))
                                * 1000
                            )
                            if (
                                d_deg > refine_cfg.drift_reset_deg
                                or d_mm > refine_cfg.drift_reset_trans * 1000
                            ):
                                refined_pose = lref.astype(np.float64).copy()
                                est_poses[-1] = refined_pose
                                drift_reset = True
                        coarse_poses[-1] = refined_pose

                    cr, ct = _pose_err(coarse_pose, gt_poses[i])
                    rr, rt = _pose_err(refined_pose, gt_poses[i])
                    if collect is not None:
                        collect.setdefault("frames", []).append(
                            {
                                "i": i,
                                "coarse_rot": cr,
                                "coarse_mm": ct * 1000,
                                "refined_rot": rr,
                                "refined_mm": rt * 1000,
                                "corr_deg": _rot_err_deg(
                                    refined_pose[:3, :3], coarse_pose[:3, :3]
                                ),
                                "corr_mm": float(
                                    np.linalg.norm(
                                        refined_pose[:3, 3] - coarse_pose[:3, 3]
                                    )
                                )
                                * 1000,
                                "drift_reset": drift_reset,
                            }
                        )
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

            prev = (view, depth_np, mask_np, pc, color, env_curr, slf, env_conf_curr)

    if refined_errs:
        c = np.array(coarse_errs)
        r = np.array(refined_errs)
        logging.info(
            "%s: refine vs loftr — rot %.2f°→%.2f° (better %.0f%%)  "
            "trans %.2f→%.2fmm (better %.0f%%)",
            sequence_name,
            c[:, 0].mean(),
            r[:, 0].mean(),
            float((r[:, 0] < c[:, 0]).mean() * 100),
            c[:, 1].mean() * 1000,
            r[:, 1].mean() * 1000,
            float((r[:, 1] < c[:, 1]).mean() * 100),
        )

    est = rebase_poses(np.stack(gt_poses), np.stack(est_poses))
    out_path = os.path.join(results_dir, f"{sequence_name}.npy")
    np.save(out_path, est)
    logging.info("%s: %s → %s", sequence_name, est.shape, out_path)
    if collect is not None:
        collect["gt_poses"] = np.stack(gt_poses)
        collect["est_poses"] = np.stack(est_poses)
        collect["coarse_poses"] = np.stack(coarse_poses)
    return coarse_errs, refined_errs


def build_work_list(exp_name: str, depth_sources: list[str]) -> list[dict]:
    work = []
    for depth_source in depth_sources:
        for split_prefix in SPLIT_PREFIXES:
            for reflectivity in REFLECTIVITIES:
                split_dir = f"{DATASET_ROOT}/{split_prefix}_{reflectivity}"
                if not os.path.isdir(split_dir):
                    logging.warning("Split not found, skipping: %s", split_dir)
                    continue
                tag = f"{depth_source}/{split_prefix}_{reflectivity}"
                for sequence_name in sorted(os.listdir(split_dir)):
                    seq_path = os.path.join(split_dir, sequence_name)
                    if not os.path.isdir(seq_path) or sequence_name == "models":
                        continue
                    if depth_source == "lf" and not os.path.isdir(
                        os.path.join(seq_path, "depth_lf")
                    ):
                        logging.warning(
                            "depth_lf missing for %s — run `./run_lf_depth.sh write` first; skipping",
                            f"{tag}/{sequence_name}",
                        )
                        continue
                    work.append(
                        {
                            "depth_source": depth_source,
                            "reflectivity": reflectivity,
                            "sequence_name": sequence_name,
                            "seq_path": seq_path,
                            "gt0_seq_path": f"{DATASET_ROOT}/{split_prefix}_0.0/{sequence_name}",
                            "results_dir": f"{exp_name}/{tag}",
                            "cache_dir": f"{CACHE_ROOT}/{tag}/{sequence_name}",
                            "tag": f"{tag}/{sequence_name}",
                        }
                    )
    return work


def main() -> None:
    parser = argparse.ArgumentParser(description="ReLiFT-6DoF tracking runner")
    parser.add_argument(
        "--refine", action="store_true", help="Enable photometric refinement"
    )
    parser.add_argument(
        "--gt",
        action="store_true",
        help="Use GT appearance in refinement (GT-refine ablation)",
    )
    parser.add_argument(
        "--depth",
        default=None,
        help="comma-separated depth sources: gt | synth | lf  (overrides config.yaml). "
        "'lf' requires running `./run_lf_depth.sh write` first.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Override the computed experiment name (output directory).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop each sequence after this many frames (useful for smoke tests).",
    )
    args = parser.parse_args()

    depth_sources = (
        [d for d in args.depth.split(",") if d] if args.depth else list(DEPTH_SOURCES)
    )
    use_refine = args.refine
    use_gt = args.refine and args.gt

    if not use_refine:
        exp_name = "ablation_loftr"
    elif use_gt:
        exp_name = "ablation_refine_gt"
    else:
        exp_name = "ablation_refine_est"
    if args.depth == "lf":
        exp_name += "_depth-" + "-".join(depth_sources)
    if args.name:
        exp_name = args.name
    pin_alpha = False if use_gt else PIN_ALPHA
    refine_cfg = REFINE_CFG

    logging.info("refine=%s  gt=%s  → %s", use_refine, use_gt, exp_name)

    loftr = LoftrRunner()
    rng = np.random.default_rng(seed=42)
    work = build_work_list(exp_name, depth_sources)

    gt_env = (
        _load_env_linear("/home/ngoncharov/cvpr2026/ycbv-eoat-lf/env2.jpg")
        if use_gt
        else None
    )

    viewer = None
    if use_refine and ENABLE_VIS:
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
                    alpha=(1.0 - float(item["reflectivity"])) if pin_alpha else None,
                    depth_source=item["depth_source"],
                    loftr=loftr,
                    rng=rng,
                    separate=USE_REFLECTION_SEPARATION,
                    refine=use_refine,
                    viewer=viewer,
                    max_frames=args.max_frames,
                    refine_cfg=refine_cfg,
                    gt_refine=use_gt,
                    gt0_seq_path=item.get("gt0_seq_path"),
                    reflectivity=float(item["reflectivity"]),
                    gt_env=gt_env,
                    feed_forward=REFINE_FEED_FORWARD,
                )
            except Exception:
                logging.exception("%s: FAILED", item["tag"])

    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
