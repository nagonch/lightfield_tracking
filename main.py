"""lightfield_tracking — 6-DoF reflective-object tracking pipeline.

Per frame: load the LF frame → build SLF → reflection separation → diffuse view →
LoFTR coarse pose → photometric refinement.  Runs over every split × reflectivity ×
sequence, rebases the estimated trajectory to the GT frame-0 pose, and saves as .npy.
"""

import argparse
import logging
import os
import time
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
    LF_DEPTH_CFG,
)
from icp import rebase_poses
from lf_depth import LFPlaneSweepDepth
from loftr_baseline import DEPTH_ZNEAR, DEPTH_ZFAR
from loftr_wrapper import LoftrRunner
from src.dataset import LFDataset
from src.photometric import (
    PhotometricRefineViewer,
    RefineConfig,
    refine_pose_photometric,
)
import config as _cfgmod
from src.pose import track_pose, loftr_relative_pose
from src.reflection import central_view, frame_diffuse
from src.surface_light_field import SurfaceLightField
from utils import linear_to_srgb

logging.basicConfig(
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)


# Below this many object pixels in the central view, segmentation has effectively
# lost the object: too few pixels for LoFTR/ICP to recover a non-degenerate pose
# (ICP on an empty point cloud yields a singular transform). We can't track an
# object we can't see, so we stop and pad the rest of the trajectory.
MIN_TRACK_PIXELS = 200

# When set, overrides the alpha used for the LoFTR/ICP blend in the coarse
# tracker (not the separation/refine alpha). Lets a pinned separation alpha
# remove reflections from the LoFTR input without dragging ICP into the blend.
BLEND_ALPHA: float | None = None


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
    no_cache_separation: bool = False,
    depth_estimator: LFPlaneSweepDepth | None = None,
    gt_masks: bool = False,
    measure_fps: bool = False,
    fps_samples: list[float] | None = None,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    # Live LF depth: compute plane-sweep depth per frame from the loaded LF instead
    # of reading pre-written depth_lf/. The dataset still needs a real depth folder
    # to index frames, so carry "synth" (always present) and override per frame.
    live_lf_depth = depth_estimator is not None and depth_source == "lf"
    ds_depth_source = "synth" if live_lf_depth else depth_source
    dataset = LFDataset(seq_path, depth_source=ds_depth_source)
    s_size, t_size = dataset.metadata["n_views"]

    # Object mask: by default segment the central view with the
    # GroundingDINO + SAM2 + Cutie segmentor, prompted on the object name (the
    # string used to locate the mesh). GT masks are used only with gt_masks=True.
    # Cutie tracks temporally, so the segmentor is single-use per sequence.
    segmentor = None
    if not gt_masks:
        from segmentor import Segmentor

        # Mesh names use underscores (e.g. "bleach_cleanser"); GroundingDINO detects
        # the natural-language form far more reliably ("bleach cleanser").
        prompt = dataset.object_name.replace("_", " ")
        segmentor = Segmentor(prompt=prompt)

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
    # Keyframe anchoring: hold one keyframe (frame 0 first — its pose is exact)
    # and additionally match/refine against it, so error accumulates per
    # keyframe hop instead of per frame. Promoted on viewpoint change or after
    # repeated match failures.
    use_kf = _cfgmod.TRACK_KEYFRAME
    kf: dict | None = None
    kf_missed = 0
    prev_env = None
    prev_env_conf = None
    alpha_history: list[float] = []
    alpha_stable = False
    prev_alpha_i: float | None = None
    alpha_veto_checked = False
    # pnp_refine "auto": off until this sequence's alpha veto proves the
    # features are albedo texture (reprojection only trustworthy then).
    if _cfgmod.PNP_MODE == "auto":
        _cfgmod.PNP_REFINE = False
    # Set once the object is lost; the remaining frames repeat the last good pose.
    tracking_lost = False

    def _pad_with_last() -> None:
        """Object lost: repeat the last tracked pose for this frame."""
        est_poses.append(est_poses[-1])
        coarse_poses.append(coarse_poses[-1])
        loftr_ref_poses.append(loftr_ref_poses[-1])

    # Honest per-frame timing (frame 0 has no tracking and absorbs CUDA warmup, so
    # it is excluded). Models are already loaded before the loop, so this measures
    # steady-state pipeline throughput, not setup cost.
    frame_times: list[float] = []
    _t0 = 0.0

    with tqdm(
        dataset, desc="  frames", unit="fr", leave=False, dynamic_ncols=True
    ) as bar:
        for i, frame in enumerate(bar):
            if max_frames is not None and i >= max_frames:
                break
            if measure_fps and i >= 1:
                torch.cuda.synchronize()
                _t0 = time.perf_counter()
            gt_poses.append(frame["object_pose"].cpu().numpy())

            # Once tracking is lost, skip all per-frame work and keep padding so the
            # saved trajectory still covers every GT frame.
            if tracking_lost:
                _pad_with_last()
                continue

            if gt_masks:
                mask = frame["masks"][s_size // 2, t_size // 2]
            else:
                bar.set_postfix(fr=i, stage="segment")
                central_srgb = linear_to_srgb(
                    frame["LF"][s_size // 2, t_size // 2].clamp(0.0, 1.0)
                )
                mask = segmentor(central_srgb).bool()

            # Segmentation can lose the object (occlusion, out of frame). With too
            # few pixels there is nothing to track, and ICP/LoFTR would return a
            # singular pose that later crashes the refine inverse. Stop here and pad
            # the remaining frames with the last good pose. (Frame 0 is seeded from
            # GT, so it is never gated.)
            n_mask_px = int(mask.sum().item())
            if i > 0 and n_mask_px < MIN_TRACK_PIXELS:
                logging.warning(
                    "%s: lost object at frame %d (%d mask px < %d) — padding "
                    "remaining %d frames with last tracked pose",
                    sequence_name,
                    i,
                    n_mask_px,
                    MIN_TRACK_PIXELS,
                    len(dataset) - i,
                )
                tracking_lost = True
                _pad_with_last()
                continue
            # Live plane-sweep depth (GPU tensor) replaces the disk depth for
            # depth=lf when caching is disabled; otherwise use the loaded depth.
            if live_lf_depth:
                bar.set_postfix(fr=i, stage="lf-depth")
                depth = depth_estimator.estimate(frame, mask=mask)["depth"]
            else:
                depth = frame["depth"]

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
                        cache_path=(
                            None
                            if no_cache_separation
                            else os.path.join(cache_dir, f"diffuse_{i:04d}.png")
                        ),
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

                # Alpha veto: once the estimate stabilises low, probe whether the
                # reflection model actually explains cross-view variance. If it
                # explains almost none, the low alpha is texture-fooled — pin a
                # near-diffuse alpha for the rest of the sequence instead of
                # subtracting a phantom reflection from the LoFTR input.
                if (
                    _cfgmod.ALPHA_VETO_ENABLED
                    and alpha is None
                    and alpha_stable
                    and not alpha_veto_checked
                ):
                    alpha_veto_checked = True
                    if alpha_i < _cfgmod.ALPHA_VETO_EST_MAX:
                        from reflection_separation import reflection_explained_ratio

                        ratio = reflection_explained_ratio(
                            slf,
                            probe_alpha=_cfgmod.ALPHA_VETO_PROBE,
                            iterations=SEPARATION_ITERS,
                        )
                        if ratio > _cfgmod.ALPHA_VETO_RATIO:
                            alpha = _cfgmod.ALPHA_VETO_CLAMP
                            if _cfgmod.PNP_MODE == "auto":
                                _cfgmod.PNP_REFINE = True
                            logging.info(
                                "%s: alpha veto — est %.2f but reflection explains "
                                "only %.1f%% of view variance (ratio %.3f) → pinned "
                                "alpha %.2f",
                                sequence_name,
                                alpha_i,
                                100 * (1 - ratio),
                                ratio,
                                alpha,
                            )
                        else:
                            logging.info(
                                "%s: alpha veto probe ratio %.3f — keeping "
                                "estimated alpha %.2f",
                                sequence_name,
                                ratio,
                                alpha_i,
                            )
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
                    alpha=(BLEND_ALPHA if BLEND_ALPHA is not None else alpha_eff),
                    pc_prev=prev[3],
                    pc_curr=pc,
                    color_prev=prev[4],
                    color_curr=color,
                    rng=rng,
                )

                # The mask can survive the pixel gate yet still yield a degenerate
                # pose (e.g. all in-mask depth out of range → empty point cloud, so
                # ICP returns a singular transform). A non-finite or singular pose
                # would crash the next frame's refine inverse, so treat it as a lost
                # track and pad the rest.
                if not np.all(np.isfinite(coarse_pose)) or (
                    abs(np.linalg.det(coarse_pose[:3, :3])) < 1e-6
                ):
                    logging.warning(
                        "%s: degenerate pose at frame %d — padding remaining "
                        "%d frames with last tracked pose",
                        sequence_name,
                        i,
                        len(dataset) - i,
                    )
                    tracking_lost = True
                    _pad_with_last()
                    continue

                # Keyframe match: LoFTR against the held keyframe view gives a
                # pose whose error is relative to the keyframe, not the drifted
                # previous frame. Inlier-gated, plus a gross-failure gate against
                # the frame-to-frame estimate (catches symmetry flips).
                kf_used = False
                kf_diag = ""
                if use_kf and kf is not None:
                    T_rel_kf, n_inl_kf = loftr_relative_pose(
                        kf["view"],
                        view,
                        kf["depth"],
                        depth_np,
                        kf["mask"],
                        mask_np,
                        K_np,
                        loftr,
                        rng,
                    )
                    if T_rel_kf is not None and n_inl_kf >= _cfgmod.KF_MIN_INLIERS:
                        kf_abs = (T_rel_kf @ kf["pose"]).astype(np.float64)
                        d_deg = _rot_err_deg(kf_abs[:3, :3], coarse_pose[:3, :3])
                        d_m = float(
                            np.linalg.norm(kf_abs[:3, 3] - coarse_pose[:3, 3])
                        )
                        kf_diag = (
                            f" [kfm n={n_inl_kf} d={d_deg:.1f}°/{d_m * 1000:.0f}mm]"
                        )
                        if (
                            np.all(np.isfinite(kf_abs))
                            and d_deg <= _cfgmod.KF_GROSS_DEG
                            and d_m <= _cfgmod.KF_GROSS_TRANS
                        ):
                            coarse_pose = kf_abs
                            kf_used = True
                    else:
                        kf_diag = f" [kfm n={n_inl_kf} fail]"
                    kf_missed = 0 if kf_used else kf_missed + 1

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
                        slf_kf=(kf["slf"] if use_kf and kf is not None else None),
                        abs_pose_kf=(
                            kf["pose"] if use_kf and kf is not None else None
                        ),
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
                    kf_tag = (" [kf]" if kf_used else "") + kf_diag if use_kf else ""
                    bar.write(
                        f"  f{i:3d}: loftr {cr:5.2f}° {ct * 1000:6.2f}mm "
                        f"→ refined {rr:5.2f}° {rt * 1000:6.2f}mm  "
                        f"| mean loftr {mc[0]:.2f}°/{mc[1] * 1000:.1f}mm "
                        f"refined {mr[0]:.2f}°/{mr[1] * 1000:.1f}mm  {flag}{kf_tag}"
                    )
                else:
                    est_poses.append(coarse_pose)

            prev = (view, depth_np, mask_np, pc, color, env_curr, slf, env_conf_curr)

            # Keyframe promotion (after this frame's refine so it used the old
            # keyframe): promote on viewpoint change beyond the caps, or after
            # repeated match failures (overlap gone). The snapshot pose is this
            # frame's best estimate, so keyframe-chain error grows only per hop.
            if use_kf:
                cur_est = est_poses[-1]
                if kf is None:
                    promote = True  # frame 0: pose is exact GT
                else:
                    d_deg = _rot_err_deg(cur_est[:3, :3], kf["pose"][:3, :3])
                    d_m = float(np.linalg.norm(cur_est[:3, 3] - kf["pose"][:3, 3]))
                    promote = (
                        d_deg > _cfgmod.KF_MAX_DEG
                        or d_m > _cfgmod.KF_MAX_TRANS
                        or kf_missed >= 3
                    )
                if promote:
                    kf = {
                        "idx": i,
                        "pose": cur_est.astype(np.float64).copy(),
                        "view": view,
                        "depth": depth_np,
                        "mask": mask_np,
                        "slf": slf,
                    }
                    kf_missed = 0
                    if i > 0:
                        bar.write(f"  [keyframe promoted at f{i}]")

            if measure_fps and i >= 1:
                torch.cuda.synchronize()
                frame_times.append(time.perf_counter() - _t0)

    if measure_fps and frame_times:
        tot = sum(frame_times)
        n = len(frame_times)
        fps = n / tot if tot > 0 else 0.0
        logging.info(
            "%s: FPS %.2f  (%d frames, %.1f ms/frame mean)",
            sequence_name,
            fps,
            n,
            1000.0 * tot / n,
        )
        if fps_samples is not None:
            fps_samples.extend(frame_times)

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


def build_work_list(
    exp_name: str, depth_sources: list[str], no_cache_depth: bool = False
) -> list[dict]:
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
                    if (
                        depth_source == "lf"
                        and not no_cache_depth
                        and not os.path.isdir(os.path.join(seq_path, "depth_lf"))
                    ):
                        logging.warning(
                            "depth_lf missing for %s — run `./run_lf_depth.sh write` first "
                            "(or pass --no-cache-depth to compute it live); skipping",
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
    parser = argparse.ArgumentParser(description="lightfield_tracking")
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
    parser.add_argument(
        "--no-cache-separation",
        action="store_true",
        help="Recompute reflection separation every frame instead of reading the "
        "diffuse cache (for honest timing / fresh experiments).",
    )
    parser.add_argument(
        "--no-cache-depth",
        action="store_true",
        help="Compute LF plane-sweep depth live in the main loop (depth=lf) instead "
        "of reading pre-written depth_lf/. No-op for depth=gt|synth.",
    )
    parser.add_argument(
        "--gt-masks",
        action="store_true",
        help="Use ground-truth object masks from the dataset instead of segmenting "
        "the central view with GroundingDINO+SAM2+Cutie (prompted on the object name).",
    )
    parser.add_argument(
        "--no-separation",
        action="store_true",
        help="Disable reflection separation (overrides config). LoFTR then tracks the "
        "raw central view instead of the separated diffuse view — the loftr-only "
        "fallback ablation.",
    )
    parser.add_argument(
        "--fps",
        action="store_true",
        help="Measure steady-state per-frame throughput (excludes frame 0 / model "
        "load) and log per-sequence + overall FPS. Pair with --no-cache-separation "
        "(and --no-cache-depth for depth=lf) for an honest end-to-end number.",
    )
    args = parser.parse_args()

    depth_sources = (
        [d for d in args.depth.split(",") if d] if args.depth else list(DEPTH_SOURCES)
    )
    use_refine = args.refine
    use_gt = args.refine and args.gt
    separate = USE_REFLECTION_SEPARATION and not args.no_separation

    if not separate:
        exp_name = "ablation_no_separation"
    elif not use_refine:
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

    logging.info(
        "refine=%s  gt=%s  separation=%s  → %s",
        use_refine,
        use_gt,
        separate,
        exp_name,
    )
    logging.info("masks: %s", "GT (dataset)" if args.gt_masks else "segmentor")
    logging.info(
        "cache: separation=%s  lf_depth=%s",
        "off (live)" if args.no_cache_separation else "on",
        "off (live)" if args.no_cache_depth else "on",
    )

    loftr = LoftrRunner()
    rng = np.random.default_rng(seed=42)
    # Live LF plane-sweep depth estimator (only when needed); reused across sequences.
    depth_estimator = (
        LFPlaneSweepDepth(LF_DEPTH_CFG)
        if args.no_cache_depth and "lf" in depth_sources
        else None
    )
    work = build_work_list(exp_name, depth_sources, no_cache_depth=args.no_cache_depth)

    gt_env = (
        _load_env_linear("/home/ngoncharov/cvpr2026/ycbv-eoat-lf/env2.jpg")
        if use_gt
        else None
    )

    viewer = None
    if use_refine and ENABLE_VIS:
        viewer = PhotometricRefineViewer(port=8081)

    # Pooled per-frame times across all sequences for the overall FPS report.
    fps_samples: list[float] | None = [] if args.fps else None

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
                    separate=separate,
                    refine=use_refine,
                    viewer=viewer,
                    max_frames=args.max_frames,
                    refine_cfg=refine_cfg,
                    gt_refine=use_gt,
                    gt0_seq_path=item.get("gt0_seq_path"),
                    reflectivity=float(item["reflectivity"]),
                    gt_env=gt_env,
                    feed_forward=REFINE_FEED_FORWARD,
                    no_cache_separation=args.no_cache_separation,
                    depth_estimator=depth_estimator,
                    gt_masks=args.gt_masks,
                    measure_fps=args.fps,
                    fps_samples=fps_samples,
                )
            except Exception:
                logging.exception("%s: FAILED", item["tag"])

    if fps_samples:
        tot = sum(fps_samples)
        n = len(fps_samples)
        logging.info(
            "OVERALL FPS %.2f  (%d frames, %.1f ms/frame mean) over %s",
            n / tot if tot > 0 else 0.0,
            n,
            1000.0 * tot / n,
            exp_name,
        )

    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
