#!/usr/bin/env python3
"""Per-method tracking GIFs: one animated GIF per (method, depth, split, seq).

Each frame is the raw central RGB view at its original resolution with only two
things drawn on top, exactly as eval/plot_tracking_grid.py does per cell:

  * the projected mesh silhouette outline (gold "boundary") at that method's
    estimated pose, and
  * that pose's predicted coordinate frame.

No crop, no labels, no badges, no GT overlay — just the frame + border + frame,
stitched over every timestep of the sequence.

Two sets (see METHOD_SETS):
  baseline  — ours (ablation_full_gt_mask) vs bsdf/fp/icp/loftr/pnp,
              each at GT depth and estimated depth (lf for ours, synth for the
              baselines). LoFTR has no GT-depth run, so only its synth gif.
  ablation  — ours vs the ablated variants, LF depth (synth for the synth-depth
              ablation), no GT depth.

Self-contained: it encodes the GIFs itself with the static ffmpeg bundled in
imageio-ffmpeg (the container has no system ffmpeg), using the same two-pass
palettegen/paletteuse family as prod_dataset_new/gifs, and shrinks the palette
per sequence until each GIF is under 1 MB (resolution is never touched).

Runs inside the lift6dof container (CPU only, no GPU / SLF needed):
    python eval/make_tracking_gifs.py                    # everything
    python eval/make_tracking_gifs.py --set baseline
    python eval/make_tracking_gifs.py --set ablation --split objects_1.0
    python eval/make_tracking_gifs.py --split objects_1.0 --seq sugar_box1

Use eval/make_tracking_gifs.sh to run the full batch as your host user so the
output GIFs are not owned by root.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw

# keep matplotlib (imported transitively by plot_tracking_grid) from warning about
# a read-only default config dir when run as a non-root uid in the container
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig"))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_eval as R  # noqa: E402
# reuse the exact overlay used by the qualitative grid figure
from plot_tracking_grid import (  # noqa: E402
    CONTOUR_COLOR,
    CONTOUR_W,
    DATASET_ROOT,
    draw_frame_halo,
    load_mesh,
    rgb_path,
)

BASELINES_ROOT = os.path.join(os.path.dirname(R.__file__), "..", "baselines")
OUT_ROOT = os.path.join(HERE, "gifs")
FPS = 12
SKIP_SEQS = {"tomato_soup_can_yalehand0", "models"}  # excluded from eval (bad data)

# Silhouette is rasterised on a canvas padded by this many px so that when the
# object leaves the frame the outline is simply clipped to the view (parts off
# screen vanish) instead of the contour tracing the image border and collapsing.
CONTOUR_PAD = 512

# GIF encoding: bundled static ffmpeg (imageio-ffmpeg) + two-pass palette, same
# filter family as prod_dataset_new/gifs. Drop the palette size until the gif is
# under TARGET_BYTES so busy scenes stay small without touching resolution.
TARGET_BYTES = 1_000_000
COLOR_LADDER = [128, 64, 32]


def _ffmpeg_exe():
    import imageio_ffmpeg  # ships a static ffmpeg 7.x; the container has no system one
    return imageio_ffmpeg.get_ffmpeg_exe()

# ── the two sets: (results_folder, output_tag, [depth_subdirs]) ────────────────
# estimated-depth subdir is "lf" for our method and "synth" for the baselines.
BASELINE_METHODS = [
    ("ablation_full_gt_mask", "ours", ["gt", "lf"]),
    ("results_bsdf", "bsdf", ["gt", "synth"]),
    ("results_fp", "fp", ["gt", "synth"]),
    ("results_icp", "icp", ["gt", "synth"]),
    ("results_loftr", "loftr", ["synth"]),        # no GT-depth run
    ("results_pnp", "pnp", ["gt", "synth"]),
]

ABLATION_METHODS = [
    ("ablation_full_gt_mask", "full_gt_mask", ["lf"]),
    ("ablation_full", "full", ["lf"]),
    ("ablation_no_refine_gt_mask", "no_refine", ["lf"]),
    ("ablation_no_separation_gt_mask", "no_separation", ["lf"]),
    ("ablation_synth_depth_gt_mask", "synth_depth", ["synth"]),
]

METHOD_SETS = {"baseline": BASELINE_METHODS, "ablation": ABLATION_METHODS}


def frame_ids(seq_dir):
    """Frame-id strings, index-aligned to the estimated-pose .npy (as in eval)."""
    pf = sorted(os.listdir(os.path.join(seq_dir, "object_poses")))
    return [p[5:-4] for p in pf]


def draw_contour(arr, K, T, verts, faces):
    """Stroke the mesh silhouette outline at pose T onto arr (RGB).

    Same as eval/plot_tracking_grid.draw_contour but the silhouette is rasterised
    on a padded canvas, so when the object goes partly/fully out of frame the
    outline is clipped to the view instead of the contour running along the image
    border (which made it "collapse"). Off-screen parts simply don't get drawn.
    """
    Vc = (T[:3, :3] @ verts.T).T + T[:3, 3]
    z = Vc[:, 2]
    u = K[0, 0] * Vc[:, 0] / z + K[0, 2]
    v = K[1, 1] * Vc[:, 1] / z + K[1, 2]
    pix = np.stack([u, v], axis=1)

    good = (z[faces] > 1e-6).all(axis=1)          # drop faces crossing the camera
    tris = pix[faces[good]]
    if len(tris) == 0:
        return

    H, W = arr.shape[:2]
    P = CONTOUR_PAD
    sil = np.zeros((H + 2 * P, W + 2 * P), np.uint8)
    # fillPoly rasterises the true (shifted) polygon and clips to the canvas, so
    # verts beyond the padded border still produce the correct outline inside it;
    # a contour that lands on the padded border is outside the image and is
    # dropped by polylines below. Guard only against int overflow from huge coords.
    tp = np.clip(tris + P, -(2 ** 30), 2 ** 30).astype(np.int32)
    cv2.fillPoly(sil, tp, 255)
    contours, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c - P for c in contours]          # back to image coordinates
    # black halo then colour, so the outline reads on both the object and the floor
    cv2.polylines(arr, contours, True, (0, 0, 0), CONTOUR_W + 2, cv2.LINE_AA)
    cv2.polylines(arr, contours, True, CONTOUR_COLOR, CONTOUR_W, cv2.LINE_AA)


def render_frames(split, seq, npy_path):
    """Return the list of overlaid RGB frames (uint8, original resolution)."""
    seq_dir = os.path.join(DATASET_ROOT, split, seq)
    K = R.load_camera_matrix(seq_dir)
    verts, faces = load_mesh(split, seq)
    fids = frame_ids(seq_dir)
    poses = np.load(npy_path)
    n = min(len(poses), len(fids))

    frames = []
    for i in range(n):
        arr = np.array(Image.open(rgb_path(seq_dir, fids[i])).convert("RGB"))
        draw_contour(arr, K, poses[i], verts, faces)  # gold silhouette outline
        img = Image.fromarray(arr)
        draw_frame_halo(ImageDraw.Draw(img), K, poses[i])  # predicted coord frame
        frames.append(np.asarray(img))
    return frames


def _encode_gif(frame_dir, out, colors):
    """Two-pass palette GIF (dither off, per-region diff) at a given palette size."""
    vf = (
        f"split[s0][s1];[s0]palettegen=max_colors={colors}:stats_mode=diff[p];"
        f"[s1][p]paletteuse=dither=none:diff_mode=rectangle"
    )
    subprocess.run(
        [_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
         "-framerate", str(FPS), "-i", os.path.join(frame_dir, "%04d.png"),
         "-vf", vf, out],
        check=True,
    )


def emit(name, frames, out_dir, frames_out):
    """Write frame PNGs (if --frames-out) or encode the GIF under TARGET_BYTES."""
    if frames_out is not None:
        d = os.path.join(frames_out, name)
        os.makedirs(d, exist_ok=True)
        for i, f in enumerate(frames):
            Image.fromarray(f).save(os.path.join(d, f"{i:04d}.png"))
        return

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{name}.gif")
    tmp = tempfile.mkdtemp(prefix="gifframes_")
    try:
        for i, f in enumerate(frames):
            Image.fromarray(f).save(os.path.join(tmp, f"{i:04d}.png"))
        # drop the palette size until the GIF fits; keep the best size that fits
        # (or the smallest tried if a busy scene never gets under the target).
        for colors in COLOR_LADDER:
            _encode_gif(tmp, out, colors)
            if os.path.getsize(out) <= TARGET_BYTES:
                break
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run(set_name, split_filter=None, seq_filter=None, method_filter=None,
        frames_out=None):
    methods = METHOD_SETS[set_name]
    out_dir = os.path.join(OUT_ROOT, set_name)
    made = skipped = 0
    for folder, tag, depths in methods:
        if method_filter and tag != method_filter:
            continue
        for depth in depths:
            base = os.path.join(BASELINES_ROOT, folder, depth)
            if not os.path.isdir(base):
                continue
            for split in sorted(os.listdir(base)):
                if split_filter and split != split_filter:
                    continue
                split_dir = os.path.join(base, split)
                if not os.path.isdir(split_dir):
                    continue
                for fn in sorted(os.listdir(split_dir)):
                    if not fn.endswith(".npy"):
                        continue
                    seq = fn[:-4]
                    if seq in SKIP_SEQS or (seq_filter and seq != seq_filter):
                        continue
                    npy = os.path.join(split_dir, fn)
                    name = f"{split}_{seq}_{tag}_{depth}"
                    try:
                        frames = render_frames(split, seq, npy)
                        emit(name, frames, out_dir, frames_out)
                        made += 1
                        if frames_out is None:
                            mb = os.path.getsize(os.path.join(out_dir, f"{name}.gif")) / 1e6
                            print(f"[{set_name}] {name}  ({len(frames)} frames, {mb:.2f} MB)")
                        else:
                            print(f"[{set_name}] {name}  ({len(frames)} frames)")
                    except Exception as e:  # never let one bad seq kill the batch
                        skipped += 1
                        print(f"[{set_name}] SKIP {split}/{seq} {tag}/{depth}: {e}")
    dest = frames_out if frames_out is not None else out_dir
    print(f"[{set_name}] done — {made} emitted, {skipped} skipped → {dest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(METHOD_SETS) + ["all"], default="all")
    ap.add_argument("--split", default=None, help="e.g. objects_1.0")
    ap.add_argument("--seq", default=None, help="single sequence")
    ap.add_argument("--method", default=None, help="single output tag, e.g. ours / bsdf")
    ap.add_argument("--frames-out", default=None,
                    help="write per-gif frame PNGs under DIR/<name>/ instead of a GIF "
                         "(consumed by eval/make_tracking_gifs.sh)")
    args = ap.parse_args()

    sets = list(METHOD_SETS) if args.set == "all" else [args.set]
    for s in sets:
        run(s, args.split, args.seq, args.method, args.frames_out)
