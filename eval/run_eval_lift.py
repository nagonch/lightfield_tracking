#!/usr/bin/env python3
"""
Evaluate 6DoF pose tracking results on the real LiFT dataset.

Usage
-----
    python3 eval/run_eval_lift.py <results_folder> [--dataset-root PATH]
        [--output-dir PATH] [--qual-dir PATH | --no-qual] [--gifs]

Results folder layout expected (written by main_lift.py)
--------------------------------------------------------
    <results_folder>/{gt,lf}/<seq_name>.npy

Each .npy is an (N, 4, 4) array of estimated object-to-camera poses in the
OpenCV camera convention (real LiFT poses need no axis flip), ordered to match
the sequence's object_poses/ files and rebased to the GT frame-0 pose.

Metrics
-------
The real dataset ships no object meshes, so ADD / ADD-S use a *pseudo-model*:
the frame-0 central-view depth backprojected inside the GT mask and moved into
the object frame via the GT frame-0 pose. That yields the visible-surface
point set every method is scored against identically — standard practice for
model-free tracking benchmarks. ATE-RMSE and mean rotation error need no model.

Outputs
-------
    <output_dir>/metrics.json    – per-sequence + per-depth-mode averages
    <output_dir>/table.tex       – LaTeX table (copy-paste ready)
    <output_dir>/summary.txt     – plain-text table
    <output_dir>/error_over_time.{png,txt}
    <qual_dir>/<depth>/<seq>/frame_XXXX.png  – GT (dashed) vs est (solid) axes
    <qual_dir>/<depth>/<seq>.gif             – with --gifs
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_eval import (  # noqa: E402  (shared with the synthetic eval)
    _EST_COLORS,
    _GT_COLORS,
    _TimeSeriesAccumulator,
    _draw_axes,
    _fmt,
    build_error_over_time_plot,
    build_error_over_time_txt,
    eval_sequence,
    per_frame_pose_errors,
)

# Absolute (not ~-based): the container runs with HOME=/root but /home mounted.
DEFAULT_DATASET_ROOT = "/home/ngoncharov/cvpr2026/datasets/LiFT_dataset"
MODEL_SAMPLE_PTS = 2000
# RealSense depth bleeds background values across mask borders (<1% of pixels
# but ~0.8 m away). Keep only depths this close to the in-mask median when
# building the pseudo-model.
MODEL_DEPTH_BAND = 0.3
# Same working range as the tracker (loftr_baseline.DEPTH_ZNEAR/ZFAR), kept
# inline so the eval stays a lightweight numpy-only script.
DEPTH_ZNEAR, DEPTH_ZFAR = 0.01, 2.0

METRIC_KEYS = ["add_auc", "adds_auc", "ate_rmse", "mean_abs_rot_deg"]
COL_NAMES = ["ADD↑ AUC", "ADD-S↑ AUC", "ATE↓ (m)", "Rot↓ (°)"]


# ── dataset access ────────────────────────────────────────────────────────────


def central_view_index(seq_dir: str) -> int:
    with open(os.path.join(seq_dir, "metadata.json")) as f:
        S, T = json.load(f)["n_views"]
    return (S // 2) * T + T // 2


def frame_ids(seq_dir: str) -> list[str]:
    """Frame ids from the LF_XXXX dirs — depth/ and the LF views share this
    numbering, while object_poses/ names can differ in zero-padding
    (teabox_translation_prod uses 000.txt against LF_0000). Poses are read
    positionally via sorted() instead."""
    return [
        d[3:] for d in sorted(os.listdir(seq_dir)) if d.startswith("LF_")
    ]


def load_gt_poses(seq_dir: str) -> np.ndarray:
    """GT object-to-central-camera poses. Real LiFT poses are already OpenCV,
    matching the loader (LFDataset uses an identity flip in LiFT mode)."""
    cam = np.loadtxt(
        os.path.join(seq_dir, "camera_poses", f"{central_view_index(seq_dir):04d}.txt")
    )
    inv_cam = np.linalg.inv(cam)
    pose_dir = os.path.join(seq_dir, "object_poses")
    poses = []
    for pf in sorted(os.listdir(pose_dir)):
        world_obj = np.loadtxt(os.path.join(pose_dir, pf))
        poses.append(inv_cam @ world_obj)
    return np.stack(poses)


def pseudo_model_pts(seq_dir: str, gt0: np.ndarray) -> np.ndarray:
    """Frame-0 masked central-view depth, backprojected and moved into the
    object frame — the mesh-free stand-in for ADD / ADD-S model points."""
    fid0 = frame_ids(seq_dir)[0]
    cv = central_view_index(seq_dir)
    K = np.loadtxt(os.path.join(seq_dir, "camera_matrix.txt"))
    depth = (
        np.array(Image.open(os.path.join(seq_dir, "depth", f"{fid0}.png"))).astype(
            np.float64
        )
        / 1000.0
    )
    mask = (
        np.array(Image.open(os.path.join(seq_dir, f"LF_{fid0}", "masks", f"{cv:04d}.png")))
        > 0
    )
    valid = mask & (depth > DEPTH_ZNEAR) & (depth < DEPTH_ZFAR)
    med = np.median(depth[valid])
    ys, xs = np.where(valid & (np.abs(depth - med) <= MODEL_DEPTH_BAND))
    d = depth[ys, xs]
    pts_cam = np.stack(
        [(xs - K[0, 2]) * d / K[0, 0], (ys - K[1, 2]) * d / K[1, 1], d], axis=1
    )
    inv_gt0 = np.linalg.inv(gt0)
    pts_obj = pts_cam @ inv_gt0[:3, :3].T + inv_gt0[:3, 3]
    if len(pts_obj) > MODEL_SAMPLE_PTS:
        rng = np.random.default_rng(0)
        pts_obj = pts_obj[rng.choice(len(pts_obj), MODEL_SAMPLE_PTS, replace=False)]
    return pts_obj


# ── qualitative overlays ──────────────────────────────────────────────────────


def visualize_sequence(
    est_poses: np.ndarray,
    gt_poses: np.ndarray,
    seq_dir: str,
    out_dir: str,
    axis_len: float,
    make_gif: bool,
):
    os.makedirs(out_dir, exist_ok=True)
    cv = central_view_index(seq_dir)
    K = np.loadtxt(os.path.join(seq_dir, "camera_matrix.txt"))
    frames = []
    for i, (fid, est, gt) in enumerate(zip(frame_ids(seq_dir), est_poses, gt_poses)):
        img_path = os.path.join(seq_dir, f"LF_{fid}", f"{cv:04d}.png")
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        _draw_axes(draw, K, gt, _GT_COLORS, axis_len, width=2, dashed=True)
        _draw_axes(draw, K, est, _EST_COLORS, axis_len, width=3, dashed=False)
        img.save(os.path.join(out_dir, f"frame_{i:04d}.png"))
        if make_gif:
            frames.append(img.resize((img.width // 2, img.height // 2)))
    if make_gif and frames:
        frames[0].save(
            out_dir.rstrip("/") + ".gif",
            save_all=True,
            append_images=frames[1:],
            duration=350,
            loop=0,
        )


# ── tables ────────────────────────────────────────────────────────────────────


def _avg(seqs: dict) -> dict:
    return {
        k: float(np.mean([m[k] for m in seqs.values()])) if seqs else None
        for k in METRIC_KEYS
    }


def build_summary_txt(all_metrics: dict) -> str:
    fmt = dict(zip(COL_NAMES, ["{:.4f}".format] * 3 + ["{:.2f}".format]))
    parts = []
    for depth_mode, seqs in all_metrics.items():
        rows = [[m[k] for k in METRIC_KEYS] for m in seqs.values()]
        idx = list(seqs.keys())
        avg = _avg(seqs)
        rows.append([avg[k] for k in METRIC_KEYS])
        idx.append("── Average ──")
        df = pd.DataFrame(rows, index=idx, columns=COL_NAMES)
        body = df.to_string(formatters=fmt)
        bar = "─" * max(len(l) for l in body.splitlines())
        parts.append(f"LiFT real ({depth_mode} depth)\n{bar}\n{body}\n")
    return "\n".join(parts)


def build_latex_table(all_metrics: dict) -> str:
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"\textbf{Depth} & \textbf{Sequence} & \textbf{ADD\,↑} & \textbf{ADD-S\,↑} "
        r"& \textbf{ATE\,↓} & \textbf{Rot.\,↓} \\",
        r" & & \multicolumn{2}{c}{\footnotesize(AUC, 0–1)} & \footnotesize(m) "
        r"& \footnotesize(\textdegree) \\",
        r"\midrule",
    ]
    for depth_mode, seqs in all_metrics.items():
        for seq, m in seqs.items():
            lines.append(
                f"  {depth_mode} & {seq.replace('_', chr(92) + '_')} & "
                f"{_fmt(m['add_auc'])} & {_fmt(m['adds_auc'])} & "
                f"{_fmt(m['ate_rmse'])} & {_fmt(m['mean_abs_rot_deg'], 2)} \\\\"
            )
        avg = _avg(seqs)
        lines += [
            r"  \cmidrule{2-6}",
            f"  & \\textbf{{Avg.~({depth_mode})}} & "
            f"{_fmt(avg['add_auc'])} & {_fmt(avg['adds_auc'])} & "
            f"{_fmt(avg['ate_rmse'])} & {_fmt(avg['mean_abs_rot_deg'], 2)} \\\\",
            r"  \midrule",
        ]
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────


def run(
    results_root: str,
    dataset_root: str,
    output_dir: str,
    qual_dir: str | None,
    axis_len: float,
    gifs: bool,
):
    os.makedirs(output_dir, exist_ok=True)
    entries = []
    for depth_mode in ("gt", "lf", "synth"):
        mode_dir = os.path.join(results_root, depth_mode)
        if os.path.isdir(mode_dir):
            for fname in sorted(os.listdir(mode_dir)):
                if fname.endswith(".npy"):
                    entries.append((depth_mode, fname[:-4]))
    if not entries:
        print(f"No {{gt,lf}}/<seq>.npy files under {results_root}", file=sys.stderr)
        sys.exit(1)

    all_metrics: dict = {}
    model_cache: dict = {}
    time_accum = _TimeSeriesAccumulator()

    for depth_mode, seq in tqdm(entries, desc="Evaluating"):
        seq_dir = os.path.join(dataset_root, seq)
        if not os.path.isdir(seq_dir):
            tqdm.write(f"  SKIP {depth_mode}/{seq}: dataset dir missing")
            continue
        est_poses = np.load(os.path.join(results_root, depth_mode, f"{seq}.npy"))
        gt_poses = load_gt_poses(seq_dir)
        if len(est_poses) != len(gt_poses):
            tqdm.write(
                f"  WARN {depth_mode}/{seq}: est {len(est_poses)} vs "
                f"gt {len(gt_poses)} frames — truncating to min"
            )
            n = min(len(est_poses), len(gt_poses))
            est_poses, gt_poses = est_poses[:n], gt_poses[:n]

        if seq not in model_cache:
            model_cache[seq] = pseudo_model_pts(seq_dir, gt_poses[0])
        metrics = eval_sequence(est_poses, gt_poses, model_cache[seq])
        rot_errs, trans_errs, valid = per_frame_pose_errors(est_poses, gt_poses)
        time_accum.add(rot_errs, trans_errs, valid)
        if metrics["n_invalid"]:
            tqdm.write(
                f"  WARN {depth_mode}/{seq}: {metrics['n_invalid']}/"
                f"{metrics['n_frames']} non-finite est poses (counted as failures)"
            )
        all_metrics.setdefault(depth_mode, {})[seq] = metrics

        if qual_dir is not None:
            visualize_sequence(
                est_poses,
                gt_poses,
                seq_dir,
                os.path.join(qual_dir, depth_mode, seq),
                axis_len,
                gifs,
            )

    result_data = {
        "per_sequence": all_metrics,
        "averages": {dm: _avg(seqs) for dm, seqs in all_metrics.items()},
        "model_points": "pseudo (frame-0 masked depth in object frame)",
    }
    json_path = os.path.join(output_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(result_data, f, indent=2)
    with open(os.path.join(output_dir, "table.tex"), "w") as f:
        f.write(build_latex_table(all_metrics))
    summary = build_summary_txt(all_metrics)
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(summary)
    build_error_over_time_plot(
        time_accum, os.path.join(output_dir, "error_over_time.png")
    )
    with open(os.path.join(output_dir, "error_over_time.txt"), "w") as f:
        f.write(build_error_over_time_txt(time_accum))

    print(summary)
    print(f"Metrics  → {json_path}")
    print(f"Outputs  → {output_dir}")
    if qual_dir is not None:
        print(f"Qual viz → {qual_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate 6DoF tracking results on the real LiFT dataset."
    )
    parser.add_argument("results_folder", help="Folder with {gt,lf}/<seq>.npy layout")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="default: eval/results/<method_name>/",
    )
    parser.add_argument(
        "--qual-dir",
        default=None,
        help="default: eval/results_qual/<method_name>/; --no-qual to skip",
    )
    parser.add_argument("--no-qual", action="store_true")
    parser.add_argument(
        "--gifs", action="store_true", help="Also write a half-res GIF per sequence"
    )
    parser.add_argument("--axis-len", type=float, default=0.05)
    args = parser.parse_args()

    method_name = os.path.basename(os.path.normpath(args.results_folder))
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(eval_dir, "results", method_name)
    qual_dir = (
        None
        if args.no_qual
        else args.qual_dir or os.path.join(eval_dir, "results_qual", method_name)
    )
    run(
        args.results_folder,
        args.dataset_root,
        output_dir,
        qual_dir,
        args.axis_len,
        args.gifs,
    )
