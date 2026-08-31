#!/usr/bin/env python3
"""Evaluate 6DoF tracking results on the captured sequences (realsense + epi).

Reads (N,4,4) trajectories from baselines_captured/results_<method>/<capture>_<seq>.npy
(object-to-camera, OpenCV convention, rebased to GT frame 0) and scores them against
poses_object/ with the real corrected mustard mesh (captured/mustard_bottle_mesh) —
unlike the LiFT eval, no pseudo-model is needed. For the reflective sequences the
bare-bottle mesh is nominal (the physical object is foil-wrapped), but every method
is scored against the identical point set so the comparison stands.

Metrics: ADD AUC, ADD-S AUC (0..10 cm), ATE-RMSE, mean |rot| — via eval/run_eval.py.

Outputs
-------
    eval/results_captured/metrics.json / summary.txt / table.tex
    eval/qual_captured/<method>/<capture>_<seq>/frame_XXXX.png   GT dashed / est solid
    eval/qual_captured/<method>/<capture>_<seq>.gif              with --gifs

Run inside the lift6dof container:
    docker exec -w "$PWD" lift6dof python eval/run_eval_captured.py --gifs
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_eval import _EST_COLORS, _GT_COLORS, _draw_axes, _fmt, eval_sequence  # noqa: E402

CAPTURED_ROOT = "/home/ngoncharov/SpecTrack_dataset/captured"
MESH_PATH = os.path.join(CAPTURED_ROOT, "mustard_bottle_mesh", "textured_simple.obj")
RESULTS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "baselines_captured")
SEQUENCES = ["realsense_diffuse", "realsense_reflective", "epi_diffuse", "epi_reflective"]
MODEL_SAMPLE_PTS = 2000
AXIS_LEN = 0.08

METRIC_KEYS = ["add_auc", "adds_auc", "ate_rmse", "mean_abs_rot_deg"]
COL_NAMES = ["ADD↑ AUC", "ADD-S↑ AUC", "ATE↓ (m)", "Rot↓ (°)"]


def load_sequence_meta(tag: str):
    """K, per-frame image paths, GT poses, and an undistortion map (epi) for one tag."""
    capture, seq = tag.split("_", 1)
    seq_dir = os.path.join(CAPTURED_ROOT, capture, seq)
    K = np.loadtxt(os.path.join(seq_dir, "intrinsics.txt"))
    entries = sorted(os.listdir(os.path.join(seq_dir, "images")))
    stems = [e.replace(".png", "") for e in entries]
    img_paths = [
        os.path.join(seq_dir, "images", e, "cam_4_4.png")
        if os.path.isdir(os.path.join(seq_dir, "images", e))
        else os.path.join(seq_dir, "images", e)
        for e in entries
    ]
    gt = np.stack([
        np.loadtxt(os.path.join(seq_dir, "poses_object", s + ".txt")) for s in stems
    ])
    maps = None
    dpath = os.path.join(seq_dir, "distortion.txt")
    if os.path.exists(dpath):
        D = np.loadtxt(dpath)
        h, w = np.asarray(Image.open(img_paths[0])).shape[:2]
        maps = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_32FC1)
    return K, img_paths, gt, maps


def sample_model_points() -> np.ndarray:
    import trimesh

    mesh = trimesh.load(MESH_PATH, process=False)
    return np.asarray(mesh.sample(MODEL_SAMPLE_PTS))


def visualize(tag, est, gt, K, img_paths, maps, out_dir, make_gif):
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    for i, (e, g, p) in enumerate(zip(est, gt, img_paths)):
        img = Image.open(p).convert("RGB")
        if maps is not None:  # draw on the undistorted image (pinhole K)
            img = Image.fromarray(cv2.remap(np.asarray(img), maps[0], maps[1],
                                            cv2.INTER_LINEAR))
        draw = ImageDraw.Draw(img)
        _draw_axes(draw, K, g, _GT_COLORS, AXIS_LEN, width=2, dashed=True)
        _draw_axes(draw, K, e, _EST_COLORS, AXIS_LEN, width=3, dashed=False)
        img.save(os.path.join(out_dir, f"frame_{i:04d}.png"))
        if make_gif:
            frames.append(img.resize((img.width // 2, img.height // 2)))
    if make_gif and frames:
        frames[0].save(out_dir.rstrip("/") + ".gif", save_all=True,
                       append_images=frames[1:], duration=350, loop=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="loftr,icp,pnp")
    ap.add_argument("--output-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results_captured"))
    ap.add_argument("--qual-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "qual_captured"))
    ap.add_argument("--no-qual", action="store_true")
    ap.add_argument("--gifs", action="store_true")
    args = ap.parse_args()

    methods = [m for m in args.methods.split(",") if m]
    model_pts = sample_model_points()
    os.makedirs(args.output_dir, exist_ok=True)

    meta = {tag: load_sequence_meta(tag) for tag in SEQUENCES}
    rows = {}
    for method in methods:
        for tag in tqdm(SEQUENCES, desc=method, dynamic_ncols=True):
            res_path = os.path.join(RESULTS_ROOT, f"results_{method}", f"{tag}.npy")
            if not os.path.exists(res_path):
                tqdm.write(f"  {method}/{tag}: missing {res_path}")
                continue
            K, img_paths, gt, maps = meta[tag]
            est = np.load(res_path)
            m = eval_sequence(est, gt, model_pts)
            rows[(method, tag)] = m
            if not args.no_qual:
                visualize(tag, est, gt, K, img_paths, maps,
                          os.path.join(args.qual_dir, method, tag), args.gifs)

    # tables: one block per sequence, methods as rows
    records = []
    for (method, tag), m in rows.items():
        records.append({"method": method, "sequence": tag,
                        **{k: m[k] for k in METRIC_KEYS}})
    df = pd.DataFrame(records)
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({f"{m}/{t}": v for (m, t), v in rows.items()}, f, indent=2)

    lines = []
    for tag in SEQUENCES:
        sub = df[df.sequence == tag]
        if not len(sub):
            continue
        lines.append(f"== {tag} ==")
        lines.append(f"{'method':8s} " + " ".join(f"{c:>12s}" for c in COL_NAMES))
        for _, r in sub.iterrows():
            lines.append(f"{r.method:8s} " +
                         " ".join(f"{_fmt(r[k]):>12s}" for k in METRIC_KEYS))
        lines.append("")
    # cross-sequence averages per method
    lines.append("== average over all 4 sequences ==")
    lines.append(f"{'method':8s} " + " ".join(f"{c:>12s}" for c in COL_NAMES))
    for method in methods:
        sub = df[df.method == method]
        if not len(sub):
            continue
        lines.append(f"{method:8s} " +
                     " ".join(f"{_fmt(sub[k].mean()):>12s}" for k in METRIC_KEYS))
    summary = "\n".join(lines)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    df.to_latex(os.path.join(args.output_dir, "table.tex"), index=False,
                float_format="%.3f")
    print(summary)


if __name__ == "__main__":
    main()
