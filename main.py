"""ReLiFT-6DoF — production tracking pipeline.

Per frame:
  1. Load the light-field frame.
  2. Build a surface light field and run reflection separation → diffuse view
     (cached to disk; separation is slow).
  3. LoFTR-match consecutive diffuse views → relative pose → absolute pose.

Runs over every {depth source} × {split} × {reflectivity} × {sequence}, rebases
the estimated trajectory to the GT frame-0 pose, and saves it as <sequence>.npy.
"""

import argparse
import logging
import os
from dataclasses import replace

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
EXP_NAME = "results_est_refine_experiment"
CACHE_ROOT = "cache/diffuse"
SEPARATION_ITERS = 300
USE_REFLECTION_SEPARATION = True  # False → LoFTR on the raw central view
USE_PHOTOMETRIC_REFINE = True  # True → photometric pose refinement after coarse
ENABLE_VIS = False  # True → open viser viewer during refinement (manual inspection)
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
PIN_ALPHA = False

# ── GT-refine experiment: THE single toggle for ground-truth dependence ──────────
# True  → feed the photometric refinement GROUND TRUTH appearance to measure the
#         optimiser's ceiling: per-point diffuse from the fully-diffuse (0.0) render,
#         reflection from the GT env map (env2.jpg), and GT alpha (= 1 - reflectivity).
#         gt_refine=True in track_sequence gates EVERY GT injection. The coarse stage
#         is untouched — reflection separation still decomposes the SLF alpha-free and
#         LoFTR matches the separated diffuse, so the coarse pose must fight the
#         reflection; the question is how much GT-appearance refinement beats it.
# False → production: the separator's own diffuse/env map + alpha (estimated, or
#         pinned to the dataset reflectivity via PIN_ALPHA). No ground truth anywhere.
#
# NEXT STEP — remove GT dependence: set this False. The optimiser config below
# (REFINE_CFG / REFINE_FEED_FORWARD) is GT-independent and stays put; only the
# appearance source swaps decomposed-for-GT. The separated diffuse/env path then
# needs its own validation (it is noisier than GT — expect to retune the relight
# gate and the drift thresholds for it).
GT_REFINE_EXPERIMENT = True
GT_ENV_PATH = "/home/ngoncharov/cvpr2026/ycbv-eoat-lf/env2.jpg"
GT_REFINE_SAVE_VIS = False  # save per-frame [coarse|refined|target|err] relit triplets

# ── Pose-refinement optimiser config (GT-INDEPENDENT) ────────────────────────────
# The verified-good refinement settings — the landed "2.0", see
# [[feedforward-and-rollback]]. Feed the refined pose FORWARD into the tracking
# backbone (not a per-frame overlay): big win — cube_1.0 ATE 30.8→17.6mm, rot
# 8.4→4.7° vs LoFTR; cube_0.0 9.8→8.3mm. The alpha-gated drift guard
# (RefineConfig.drift_reset_*) keeps an independent pure-LoFTR reference and resets
# the backbone to it when a long sequence accumulates too much drift — on DIFFUSE
# frames only, so the reflective feed-forward gains are untouched. Two refine
# stages: diffuse (fine) then relight (final, reflective frames only).
REFINE_FEED_FORWARD = True
REFINE_CFG = RefineConfig(
    relight=True,
    drift_reset_deg=10.0,  # feed-forward drift guard (diffuse only, see alpha gate)
    drift_reset_trans=0.015,
)

# GT-only overrides: the experiment knows the true alpha and has an accurate env map.
if GT_REFINE_EXPERIMENT:
    EXP_NAME = "results_gt_refine_experiment_again"
    PIN_ALPHA = False  # the separator must ESTIMATE alpha — LoFTR fights reflection
    # The GT env map is accurate, so keep every valid relight correction (the
    # noise-floor revert gate that protects the *separated* env is not wanted here).
    REFINE_CFG = replace(REFINE_CFG, relight_min_correction_deg=0.0)


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


# ── GT-refine helpers ────────────────────────────────────────────────────────────


def _load_env_linear(path: str) -> torch.Tensor:
    """env2.jpg → [H, W, 3] LINEAR (srgb_to_linear of the raw jpg, the same space the
    cube_1.0 reflection lives in once the dataset loader linearises it)."""
    from PIL import Image
    from utils import srgb_to_linear

    raw = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return srgb_to_linear(torch.from_numpy(raw)).cuda().float()


def _gt_diffuse_points(gt_ds, idx: int, mask_bool: torch.Tensor) -> torch.Tensor:
    """Per-point GT diffuse (fully-diffuse 0.0 render) at the SLF's masked points.

    Frame-aligned across reflectivities (identical depth/masks), so the central view
    of the 0.0 split sampled at the row-major masked pixels is the GT diffuse colour
    for each surface point — no reflection separation needed.
    """
    s, t = gt_ds.metadata["n_views"]
    central = gt_ds[idx]["LF"][s // 2, t // 2].cuda().float()  # [H, W, 3] linear
    return central.reshape(-1, 3)[mask_bool.reshape(-1)].clone()


@torch.no_grad()
def _save_refine_vis(
    save_dir,
    i,
    slf_prev,
    slf_curr,
    env,
    alpha,
    abs_pose_prev,
    coarse_pose,
    refined_pose,
    gt_pose,
):
    """Save a [src@coarse | src@refined | target | |err| ] relit triplet for one frame."""
    from PIL import Image, ImageDraw

    from src.photometric import _to_display_u8

    os.makedirs(save_dir, exist_ok=True)
    inv_prev = np.linalg.inv(abs_pose_prev)

    def _render(pose):
        T = torch.from_numpy(pose @ inv_prev).float().cuda()
        img, _, _ = slf_prev.render_relit(
            rel_pose=T, env_map=env, alpha=alpha, mode="relit"
        )
        return img

    src_c, src_r = _render(coarse_pose), _render(refined_pose)
    tgt, _, _ = slf_curr.render_relit(
        rel_pose=None, env_map=env, alpha=alpha, mode="relit"
    )
    err = (src_r - tgt).abs()
    cols = [
        _to_display_u8(src_c),
        _to_display_u8(src_r),
        _to_display_u8(tgt),
        _to_display_u8((err * 6).clamp(0, 1)),
    ]
    H = cols[0].shape[0]
    gap = np.full((H, 5, 3), 255, np.uint8)
    strip = np.concatenate([cols[0], gap, cols[1], gap, cols[2], gap, cols[3]], axis=1)
    cr = _rot_err_deg(coarse_pose[:3, :3], gt_pose[:3, :3])
    rr = _rot_err_deg(refined_pose[:3, :3], gt_pose[:3, :3])
    im = Image.fromarray(strip)
    ImageDraw.Draw(im).text(
        (4, 4),
        f"f{i}  coarse {cr:.2f}deg -> refined {rr:.2f}deg   [coarse | refined | target | err]",
        fill=(255, 255, 0),
    )
    im.save(os.path.join(save_dir, f"frame_{i:04d}.png"))


DEPTH_SOURCES = ["gt"]  # "gt" | "synth"
SPLIT_PREFIXES = ["cube", "objects"]
# LoFTR must fight the reflection → reflective splits only for the GT-refine study.
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
    gt_refine: bool = False,
    gt0_seq_path: str | None = None,
    reflectivity: float = 0.0,
    gt_env: torch.Tensor | None = None,
    feed_forward: bool = False,
    collect: dict | None = None,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    dataset = LFDataset(seq_path, depth_source=depth_source)
    s_size, t_size = dataset.metadata["n_views"]

    # GT-refine: fully-diffuse (0.0) render gives GT per-point diffuse; GT alpha is the
    # known diffuse fraction. The separator still runs alpha-free for the coarse stage.
    gt_ds0 = LFDataset(gt0_seq_path, depth_source=depth_source) if gt_refine else None
    gt_alpha = 1.0 - reflectivity
    vis_dir = os.path.join(results_dir, "vis", sequence_name)

    gt_poses: list[np.ndarray] = []
    est_poses: list[np.ndarray] = []  # reported trajectory (refined when refine=True)
    coarse_poses: list[np.ndarray] = []  # tracking backbone (refined when feed_forward)
    # Independent pure-LoFTR backbone (never refined) used only as a drift reference:
    # if the fed-forward pose diverges from it by more than the cfg drift caps, reset.
    loftr_ref_poses: list[np.ndarray] = []
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

            # GT-refine override: replace the *decomposed* SLF appearance/env that feed
            # the photometric refinement with ground truth — GT per-point diffuse from
            # the 0.0 render and the GT env map. The coarse stage above (LoFTR on the
            # separated diffuse, estimated alpha) is untouched.
            if gt_refine:
                slf.diffuse_colors = _gt_diffuse_points(gt_ds0, i, mask > 0)
                env_curr = gt_env

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
                # The backbone anchor is coarse_poses[-1] — the refined pose when
                # feed_forward (so the next LoFTR match / SLF geometry builds on the
                # improvement), else the pure LoFTR pose. The inter-frame LoFTR motion
                # is anchor-independent, so we also propagate an independent pure-LoFTR
                # reference (loftr_ref_poses) for the drift guard below.
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

                # Propagate the independent pure-LoFTR reference with this frame's
                # inter-frame motion (rel = coarse_pose · anchor⁻¹, anchor-independent).
                rel = coarse_pose @ np.linalg.inv(coarse_poses[-1])
                loftr_ref_poses.append(rel @ loftr_ref_poses[-1])
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
                        alpha=(gt_alpha if gt_refine else alpha_eff),
                        cfg=refine_cfg,
                        viewer=viewer,
                        gt_pose_curr=gt_poses[i],
                    )
                    est_poses.append(refined_pose)

                    if gt_refine and GT_REFINE_SAVE_VIS and prev[5] is not None:
                        _save_refine_vis(
                            vis_dir,
                            i,
                            prev[6],
                            slf,
                            gt_env,
                            gt_alpha,
                            coarse_poses[-2],
                            coarse_pose,
                            refined_pose,
                            gt_poses[i],
                        )

                    # Feed the refined pose forward into the backbone so the next
                    # LoFTR match (and SLF geometry anchor) builds on the improved
                    # pose. DRIFT GUARD: feeding forward lets a small systematic
                    # per-frame bias accumulate over a long sequence (cube
                    # tomato_soup_can_yalehand0, 108 frames: 7.7→29mm). So if the
                    # fed-forward pose has wandered too far from the INDEPENDENT
                    # pure-LoFTR reference, reset both the reported pose and the
                    # backbone to that reference and move on — bounding the drift.
                    relight_ran = (
                        refine_cfg.relight
                        and prev[5] is not None
                        and alpha_eff < refine_cfg.relight_alpha_max
                    )
                    drift_reset = False
                    if feed_forward or (
                        relight_ran and refine_cfg.relight_feed_forward
                    ):
                        # The guard falls back to the pure-LoFTR pose, so only apply it
                        # where LoFTR is trustworthy (diffuse / high alpha). On reflective
                        # frames LoFTR is what relight is beating — falling back reverts
                        # the gains, so the guard is disabled there.
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

                    # ── per-frame metrics: does refine beat the LoFTR coarse? ──
                    cr, ct = _pose_err(coarse_pose, gt_poses[i])
                    rr, rt = _pose_err(refined_pose, gt_poses[i])
                    if collect is not None:
                        corr_deg = _rot_err_deg(
                            refined_pose[:3, :3], coarse_pose[:3, :3]
                        )
                        corr_mm = (
                            float(
                                np.linalg.norm(refined_pose[:3, 3] - coarse_pose[:3, 3])
                            )
                            * 1000
                        )
                        collect.setdefault("frames", []).append(
                            {
                                "i": i,
                                "coarse_rot": cr,
                                "coarse_mm": ct * 1000,
                                "refined_rot": rr,
                                "refined_mm": rt * 1000,
                                "corr_deg": corr_deg,
                                "corr_mm": corr_mm,
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
    if collect is not None:
        collect["gt_poses"] = np.stack(gt_poses)
        collect["est_poses"] = np.stack(est_poses)
        collect["coarse_poses"] = np.stack(coarse_poses)
    return coarse_errs, refined_errs


def build_work_list(exp_name: str) -> list[dict]:
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
                                # fully-diffuse (0.0) version of the same sequence —
                                # the GT diffuse appearance source for the experiment.
                                "gt0_seq_path": f"{DATASET_ROOT}/{split_prefix}_0.0/{sequence_name}",
                                "results_dir": f"{exp_name}/{tag}",
                                "cache_dir": f"{CACHE_ROOT}/{tag}/{sequence_name}",
                                "tag": f"{tag}/{sequence_name}",
                            }
                        )
    return work


def main() -> None:
    parser = argparse.ArgumentParser(description="ReLiFT-6DoF ablation runner")
    parser.add_argument("--refine", action="store_true", help="Enable photometric refinement")
    parser.add_argument("--gt", action="store_true", help="Use GT appearance in refinement (GT-refine experiment)")
    args = parser.parse_args()

    use_refine = args.refine
    use_gt = args.refine and args.gt

    if not use_refine:
        exp_name = "ablation_loftr"
    elif use_gt:
        exp_name = "ablation_refine_gt"
    else:
        exp_name = "ablation_refine_est"

    pin_alpha = PIN_ALPHA
    refine_cfg = REFINE_CFG
    if use_gt:
        pin_alpha = False
        refine_cfg = replace(REFINE_CFG, relight_min_correction_deg=0.0)

    logging.info("Ablation: refine=%s  gt=%s  → %s", use_refine, use_gt, exp_name)

    loftr = LoftrRunner()
    rng = np.random.default_rng(seed=42)
    work = build_work_list(exp_name)

    gt_env = _load_env_linear(GT_ENV_PATH) if use_gt else None

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
