#!/usr/bin/env python3
"""
Evaluate 6DoF pose tracking results on the SpecTrack dataset.

Usage
-----
    python3 eval/run_eval.py <results_folder> [--dataset-root PATH] [--output-dir PATH]

Results folder layout expected
-------------------------------
    <results_folder>/{gt,synth}/{split}/{seq_name}.npy

Each .npy is an (N, 4, 4) array of estimated object-to-camera poses in the
OpenCV camera convention, ordered to match the GT frames produced by the
SpecTrack dataset loader (central LF view, poses converted to OpenCV).

Outputs
-------
    <output_dir>/metrics.json   – full per-sequence and block-average metrics
    <output_dir>/table.tex      – LaTeX longtable (4 blocks, copy-paste ready)
    <qual_dir>/{gt,synth}/{split}/{seq}/frame_XXXX.png  – coordinate-frame overlays
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from tqdm import tqdm

# ── dataset constants ────────────────────────────────────────────────────────
CENTRAL_VIEW = 12  # flat index in 5×5 LF grid (row 2, col 2)
_TO_OPENCV = np.array(
    [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64
)
DEFAULT_DATASET_ROOT = "/home/ngoncharov/SpecTrack_dataset"
MESH_SAMPLE_PTS = 2000  # max model points used for ADD / ADD-S


# ── geometry helpers ─────────────────────────────────────────────────────────


def rotation_angle_deg(R_err: np.ndarray) -> np.ndarray:
    """Rotation angle (degrees) for a batch (..., 3, 3) of rotation matrices."""
    trace = np.trace(R_err, axis1=-2, axis2=-1)
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def _transform_pts(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4×4 rigid transform to (N, 3) points."""
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def _add_err(est: np.ndarray, gt: np.ndarray, model_pts: np.ndarray) -> float:
    return float(
        np.linalg.norm(
            _transform_pts(model_pts, est) - _transform_pts(model_pts, gt), axis=-1
        ).mean()
    )


def _adds_err(est: np.ndarray, gt: np.ndarray, model_pts: np.ndarray) -> float:
    pred_pts = _transform_pts(model_pts, est)
    gt_pts = _transform_pts(model_pts, gt)
    tree = cKDTree(pred_pts)
    nn_dists, _ = tree.query(gt_pts, k=1, workers=-1)
    return float(nn_dists.mean())


def _auc_under_accuracy_curve(errors: np.ndarray, threshold_max: float = 0.1) -> float:
    """Fraction-correct AUC: x-axis normalised to [0,1], y-axis = P(err < t).
    Matches the convention in eval/eval.py — higher is better, range [0, 1]."""
    thresholds = np.linspace(0, threshold_max, 100)
    accuracies = [(errors < t).mean() for t in thresholds]
    return float(np.trapz(accuracies, np.linspace(0, 1, 100)))


# ── dataset loading ───────────────────────────────────────────────────────────


def load_gt_poses(seq_dir: str) -> np.ndarray:
    """Load all GT object-to-camera poses for a sequence (OpenCV convention)."""
    cam_pose = np.loadtxt(
        os.path.join(seq_dir, "camera_poses", f"{CENTRAL_VIEW:04d}.txt")
    )
    inv_cam = np.linalg.inv(cam_pose)
    pose_files = sorted(os.listdir(os.path.join(seq_dir, "object_poses")))
    poses = []
    for pf in pose_files:
        world_obj = np.loadtxt(os.path.join(seq_dir, "object_poses", pf))
        poses.append(inv_cam @ world_obj @ _TO_OPENCV)
    return np.stack(poses)  # (N, 4, 4)


def get_object_name(dataset_root: str, split_name: str, seq_name: str) -> str:
    if split_name.startswith("cube_"):
        return "cube"
    mesh_root = os.path.join(dataset_root, "object_meshes")
    candidates = [
        m for m in os.listdir(mesh_root) if os.path.isdir(os.path.join(mesh_root, m))
    ]
    return max(candidates, key=lambda m: len(os.path.commonprefix([seq_name, m])))


def load_mesh_pts(dataset_root: str, split_name: str, seq_name: str) -> np.ndarray:
    obj_name = get_object_name(dataset_root, split_name, seq_name)
    obj_path = os.path.join(
        dataset_root, "object_meshes", obj_name, "textured_simple.obj"
    )
    mesh = trimesh.load(obj_path, force="mesh", process=False)
    pts = np.array(mesh.vertices, dtype=np.float64)
    if len(pts) > MESH_SAMPLE_PTS:
        rng = np.random.default_rng(0)
        pts = pts[rng.choice(len(pts), MESH_SAMPLE_PTS, replace=False)]
    return pts


# ── qualitative visualisation ────────────────────────────────────────────────

# EST axes: bright RGB.  GT axes: muted pastel RGB.
_EST_COLORS = {"x": (220, 30, 30), "y": (30, 200, 30), "z": (30, 80, 220)}
_GT_COLORS = {"x": (200, 130, 130), "y": (130, 200, 130), "z": (130, 130, 200)}


def load_camera_matrix(seq_dir: str) -> np.ndarray:
    return np.loadtxt(os.path.join(seq_dir, "camera_matrix.txt"))


def get_frame_image_paths(seq_dir: str) -> list:
    """Return central-view image paths ordered to match load_gt_poses."""
    pose_files = sorted(os.listdir(os.path.join(seq_dir, "object_poses")))
    paths = []
    for pf in pose_files:
        frame_id = pf[5:-4]  # "pose_0138.txt" → "0138"
        paths.append(
            os.path.join(seq_dir, f"LF_{frame_id}", "masks", f"{CENTRAL_VIEW:04d}.png")
        )
    return paths


def _project(pt3: np.ndarray, K: np.ndarray):
    """Project a single camera-space 3D point; returns (u, v) or None."""
    x, y, z = pt3
    if z <= 1e-6:
        return None
    return (K[0, 0] * x / z + K[0, 2], K[1, 1] * y / z + K[1, 2])


def _dashed_line(draw: ImageDraw.Draw, p1, p2, color, width=1, dash=8, gap=5):
    if p1 is None or p2 is None:
        return
    d = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if d < 1:
        return
    dx, dy = (p2[0] - p1[0]) / d, (p2[1] - p1[1]) / d
    t, on = 0.0, True
    while t < d:
        t_end = min(t + (dash if on else gap), d)
        if on:
            a = (p1[0] + t * dx, p1[1] + t * dy)
            b = (p1[0] + t_end * dx, p1[1] + t_end * dy)
            draw.line([a, b], fill=color, width=width)
        t, on = t_end, not on


def _draw_axes(
    draw: ImageDraw.Draw,
    K: np.ndarray,
    T: np.ndarray,
    colors: dict,
    axis_len: float,
    width: int,
    dashed: bool,
):
    origin = _project(T[:3, 3], K)
    if origin is None:
        return
    for name, col in colors.items():
        offset = {"x": [axis_len, 0, 0], "y": [0, axis_len, 0], "z": [0, 0, axis_len]}[
            name
        ]
        tip_3d = T[:3, :3] @ np.array(offset) + T[:3, 3]
        tip = _project(tip_3d, K)
        if tip is None:
            continue
        if dashed:
            _dashed_line(draw, origin, tip, col, width=width)
        else:
            draw.line([origin, tip], fill=col, width=width)
        r = width + 1
        draw.ellipse([tip[0] - r, tip[1] - r, tip[0] + r, tip[1] + r], fill=col)


def visualize_sequence(
    est_poses: np.ndarray,
    gt_poses: np.ndarray,
    img_paths: list,
    K: np.ndarray,
    qual_dir: str,
    depth_mode: str,
    split: str,
    seq_name: str,
    axis_len: float = 0.05,
):
    out_dir = os.path.join(qual_dir, depth_mode, split, seq_name)
    os.makedirs(out_dir, exist_ok=True)
    for i, (est, gt, img_path) in enumerate(zip(est_poses, gt_poses, img_paths)):
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        _draw_axes(draw, K, gt, _GT_COLORS, axis_len, width=2, dashed=True)
        _draw_axes(draw, K, est, _EST_COLORS, axis_len, width=3, dashed=False)
        img.save(os.path.join(out_dir, f"frame_{i:04d}.png"))


# ── per-sequence evaluation ───────────────────────────────────────────────────


def eval_sequence(
    est_poses: np.ndarray,
    gt_poses: np.ndarray,
    model_pts: np.ndarray,
) -> dict:
    assert (
        est_poses.shape == gt_poses.shape
    ), f"Shape mismatch: est {est_poses.shape} vs gt {gt_poses.shape}"
    N = len(est_poses)

    add_errs, adds_errs = [], []
    for est, gt in zip(est_poses, gt_poses):
        add_errs.append(_add_err(est, gt, model_pts))
        adds_errs.append(_adds_err(est, gt, model_pts))
    add_errs = np.array(add_errs)
    adds_errs = np.array(adds_errs)

    R_gt = gt_poses[:, :3, :3]
    t_gt = gt_poses[:, :3, 3]
    R_est = est_poses[:, :3, :3]
    t_est = est_poses[:, :3, 3]

    R_err = R_est @ np.transpose(R_gt, (0, 2, 1))
    rot_errs = rotation_angle_deg(R_err)
    trans_errs = np.linalg.norm(t_est - t_gt, axis=1)

    return {
        "add_auc": _auc_under_accuracy_curve(add_errs),
        "adds_auc": _auc_under_accuracy_curve(adds_errs),
        "ate_rmse": float(np.sqrt((trans_errs**2).mean())),
        "mean_abs_rot_deg": float(rot_errs.mean()),
        "n_frames": N,
    }


# ── table helpers ─────────────────────────────────────────────────────────────

BLOCK_ORDER = ["cube_gt", "cube_synth", "objects_gt", "objects_synth"]

BLOCK_DISPLAY = {
    "cube_gt": "Cube (GT depth)",
    "cube_synth": "Cube (Synth depth)",
    "objects_gt": "Objects (GT depth)",
    "objects_synth": "Objects (Synth depth)",
}

METRIC_KEYS = ["add_auc", "adds_auc", "ate_rmse", "mean_abs_rot_deg"]


def _fmt(v, decimals: int = 4) -> str:
    return "--" if v is None else f"{v:.{decimals}f}"


def block_for(depth_mode: str, split: str) -> str:
    if split.startswith("cube_"):
        return f"cube_{depth_mode}"
    if split.startswith("objects_"):
        return f"objects_{depth_mode}"
    return "other"


def _split_avg(seqs: dict) -> dict:
    """Average metrics over all sequences in a split."""
    vals = {k: [] for k in METRIC_KEYS}
    for m in seqs.values():
        for k in METRIC_KEYS:
            vals[k].append(m[k])
    return {k: float(np.mean(v)) if v else None for k, v in vals.items()}


def build_latex_table(all_metrics: dict) -> str:
    lines = [
        r"\begin{longtable}{llrrrr}",
        r"\toprule",
        r"\textbf{Split} & \textbf{Sequence} & \textbf{ADD\,↑} & \textbf{ADD-S\,↑} "
        r"& \textbf{ATE\,↓} & \textbf{Rot.\,↓} \\",
        r" & & \multicolumn{2}{c}{\footnotesize(AUC, 0–1)} & \footnotesize(m) & \footnotesize(\textdegree) \\",
        r"\midrule",
        r"\endhead",
    ]

    for blk in BLOCK_ORDER:
        if blk not in all_metrics:
            continue
        label = BLOCK_DISPLAY.get(blk, blk)
        lines += [
            r"\multicolumn{6}{l}{\textbf{" + label + r"}} \\",
            r"\midrule",
        ]
        splits_dict = all_metrics[blk]
        for split in sorted(splits_dict):
            for seq in sorted(splits_dict[split]):
                m = splits_dict[split][seq]
                seq_tex = seq.replace("_", r"\_")
                lines.append(
                    f"  {split} & {seq_tex} & "
                    f"{_fmt(m['add_auc'])} & {_fmt(m['adds_auc'])} & "
                    f"{_fmt(m['ate_rmse'])} & {_fmt(m['mean_abs_rot_deg'], 2)} \\\\"
                )
            avg = _split_avg(splits_dict[split])
            lines += [
                r"  \cmidrule{2-6}",
                f"  & \\textbf{{Avg.~{split}}} & "
                f"{_fmt(avg.get('add_auc'))} & {_fmt(avg.get('adds_auc'))} & "
                f"{_fmt(avg.get('ate_rmse'))} & {_fmt(avg.get('mean_abs_rot_deg'), 2)} \\\\",
                r"  \midrule",
            ]

    lines += [r"\bottomrule", r"\end{longtable}"]
    return "\n".join(lines)


def build_summary_txt(all_metrics: dict) -> str:
    col_names = ["ADD↑ AUC", "ADD-S↑ AUC", "ATE↓ (m)", "Rot↓ (°)"]
    mk = ["add_auc", "adds_auc", "ate_rmse", "mean_abs_rot_deg"]
    fmt = {
        "ADD↑ AUC": "{:.4f}".format,
        "ADD-S↑ AUC": "{:.4f}".format,
        "ATE↓ (m)": "{:.4f}".format,
        "Rot↓ (°)": "{:.2f}".format,
    }

    parts = []
    for blk in BLOCK_ORDER:
        if blk not in all_metrics:
            continue
        label = BLOCK_DISPLAY.get(blk, blk)
        splits_dict = all_metrics[blk]

        # Build one DataFrame per split, each followed by its average row
        block_lines = []
        header_printed = False
        for split in sorted(splits_dict):
            rows, idx = [], []
            for seq in sorted(splits_dict[split]):
                m = splits_dict[split][seq]
                rows.append([m[k] for k in mk])
                idx.append((split, seq))

            df = pd.DataFrame(
                rows,
                index=pd.MultiIndex.from_tuples(idx, names=["split", "sequence"]),
                columns=col_names,
            )
            avg = _split_avg(splits_dict[split])
            avg_df = pd.DataFrame(
                [[avg.get(k, float("nan")) for k in mk]],
                index=pd.MultiIndex.from_tuples(
                    [(split, "── Average ──")], names=["split", "sequence"]
                ),
                columns=col_names,
            )

            body = df.to_string(formatters=fmt, header=not header_printed)
            avg_line = avg_df.to_string(formatters=fmt, header=False)
            header_printed = True

            width = max(len(line) for line in body.splitlines())
            thin = "·" * width
            block_lines += [body, avg_line, thin]

        full_body = "\n".join(block_lines)
        width = max(len(line) for line in full_body.splitlines())
        bar = "─" * width
        parts.append(f"{label}\n{bar}\n{full_body}\n")

    return "\n".join(parts)


# ── main ──────────────────────────────────────────────────────────────────────


def collect_sequences(results_root: str) -> list:
    entries = []
    for depth_mode in ("gt", "synth"):
        mode_dir = os.path.join(results_root, depth_mode)
        if not os.path.isdir(mode_dir):
            continue
        for split in sorted(os.listdir(mode_dir)):
            split_dir = os.path.join(mode_dir, split)
            if not os.path.isdir(split_dir):
                continue
            for fname in sorted(os.listdir(split_dir)):
                if fname.endswith(".npy"):
                    entries.append((depth_mode, split, fname[:-4]))
    return entries


def run(
    results_root: str,
    dataset_root: str,
    output_dir: str,
    qual_dir: str | None = None,
    axis_len: float = 0.05,
):
    os.makedirs(output_dir, exist_ok=True)

    entries = collect_sequences(results_root)
    if not entries:
        print(f"No .npy files found under {results_root}", file=sys.stderr)
        sys.exit(1)

    all_metrics: dict = {}  # block → split → seq → metrics dict
    mesh_cache: dict = {}  # (split, seq) → model_pts

    for depth_mode, split, seq_name in tqdm(entries, desc="Evaluating"):
        if seq_name == "tomato_soup_can_yalehand0":
            continue
        npy_path = os.path.join(results_root, depth_mode, split, f"{seq_name}.npy")
        seq_dir = os.path.join(dataset_root, split, seq_name)
        if not os.path.isdir(seq_dir):
            tqdm.write(f"  SKIP {depth_mode}/{split}/{seq_name}: dataset dir missing")
            continue

        est_poses = np.load(npy_path)  # (N, 4, 4)
        gt_poses = load_gt_poses(seq_dir)  # (N, 4, 4)

        if len(est_poses) != len(gt_poses):
            tqdm.write(
                f"  WARN {depth_mode}/{split}/{seq_name}: "
                f"est {len(est_poses)} frames vs gt {len(gt_poses)} — truncating to min"
            )
            n = min(len(est_poses), len(gt_poses))
            est_poses = est_poses[:n]
            gt_poses = gt_poses[:n]

        key = (split, seq_name)
        if key not in mesh_cache:
            mesh_cache[key] = load_mesh_pts(dataset_root, split, seq_name)
        model_pts = mesh_cache[key]

        metrics = eval_sequence(est_poses, gt_poses, model_pts)
        blk = block_for(depth_mode, split)
        all_metrics.setdefault(blk, {}).setdefault(split, {})[seq_name] = metrics

        if qual_dir is not None:
            K = load_camera_matrix(seq_dir)
            img_paths = get_frame_image_paths(seq_dir)
            visualize_sequence(
                est_poses,
                gt_poses,
                img_paths,
                K,
                qual_dir,
                depth_mode,
                split,
                seq_name,
                axis_len=axis_len,
            )

    # ── split-level averages (4 blocks × 4 reflectivity levels = 16) ─────────
    split_avgs = {}
    for blk, splits_dict in all_metrics.items():
        split_avgs[blk] = {
            split: _split_avg(seqs) for split, seqs in splits_dict.items()
        }

    # ── save JSON ────────────────────────────────────────────────────────────
    result_data = {"per_sequence": all_metrics, "split_averages": split_avgs}
    json_path = os.path.join(output_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(result_data, f, indent=2)

    # ── save + print table ───────────────────────────────────────────────────
    tex = build_latex_table(all_metrics)
    tex_path = os.path.join(output_dir, "table.tex")
    with open(tex_path, "w") as f:
        f.write(tex)

    summary = build_summary_txt(all_metrics)
    txt_path = os.path.join(output_dir, "summary.txt")
    with open(txt_path, "w") as f:
        f.write(summary)

    print(summary)
    print(f"Metrics  → {json_path}")
    print(f"LaTeX    → {tex_path}")
    print(f"Summary  → {txt_path}")
    if qual_dir is not None:
        print(f"Qual viz → {qual_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate 6DoF tracking results on SpecTrack."
    )
    parser.add_argument(
        "results_folder",
        help="Folder with {gt,synth}/{split}/{seq}.npy layout",
    )
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help=f"SpecTrack dataset root (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write metrics.json and table.tex "
        "(default: eval/results/<method_name>/)",
    )
    parser.add_argument(
        "--qual-dir",
        default=None,
        help="Where to write coordinate-frame overlay images "
        "(default: eval/results_qual/<method_name>/; use --no-qual to skip)",
    )
    parser.add_argument(
        "--no-qual",
        action="store_true",
        help="Skip qualitative visualisation entirely.",
    )
    parser.add_argument(
        "--axis-len",
        type=float,
        default=0.05,
        help="Axis length in metres for the coordinate-frame overlay (default: 0.05).",
    )
    args = parser.parse_args()

    method_name = os.path.basename(os.path.normpath(args.results_folder))
    eval_dir = os.path.dirname(os.path.abspath(__file__))

    output_dir = args.output_dir or os.path.join(eval_dir, "results", method_name)

    if args.no_qual:
        qual_dir = None
    else:
        qual_dir = args.qual_dir or os.path.join(eval_dir, "results_qual", method_name)

    run(
        args.results_folder,
        args.dataset_root,
        output_dir,
        qual_dir=qual_dir,
        axis_len=args.axis_len,
    )
