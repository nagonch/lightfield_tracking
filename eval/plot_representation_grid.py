#!/usr/bin/env python3
"""Qualitative figure of our relightable surface-light-field representation.

For the top-K sequences of one reflectivity split we pick the single best-tracked
("perfect") frame and lay out, side by side, everything the tracker actually
relies on:

    input | depth | surface normals | albedo | shading | environment map | (conf)

The diffuse decomposition (albedo·shading) and the environment map are read from
the separation cache written by main.py; depth comes from the dataset; surface
normals are computed from depth; albedo/shading use the same SoftPhong model the
pipeline uses (src/shading.py). Everything runs on CPU — no SLF / GPU needed.

Run inside the lift6dof container:
    python eval/plot_representation_grid.py                 # objects_0.7 and objects_0.5
    python eval/plot_representation_grid.py --split objects_0.7
    python eval/plot_representation_grid.py --split objects_0.7 --seq sugar_box1 --frame 30
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# repo root on path: reuse run_eval helpers and src/shading + utils
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import run_eval as R  # noqa: E402
from src.shading import shading_terms, unshade_to_albedo  # noqa: E402
from utils import linear_to_srgb  # noqa: E402

# ── what to plot ──────────────────────────────────────────────────────────────
DATASET_ROOT = "/home/ngoncharov/SpecTrack_dataset"
CACHE_ROOT = os.path.join(os.path.dirname(HERE), "cache", "diffuse_corrected", "gt")
RESULTS_ROOT = os.path.join(
    os.path.dirname(HERE), "baselines", "ablation_full_gt_mask", "gt"
)
METRICS = os.path.join(HERE, "results", "ablation_full_gt_mask", "metrics.json")
OUT_DIR = os.path.join(HERE, "plots")

TOPK = 5
# Restrict the "perfect frame" search to the back of the sequence, where the
# accumulated environment map has converged (early frames have a sparse env map).
FRAME_MIN_FRAC = 0.15
MIN_MASK_PX = 1500

CROP_PAD = 0.28           # extra margin factor around the object bbox
CELL_PX = 360             # rendered object-cell resolution

COLS = [
    ("input", "Input"),
    ("depth", "Depth"),
    ("normals", "Surface normals"),
    ("albedo", "Albedo"),
    ("shading", "Shading"),
    ("env", "Environment map"),
    ("conf", "Env. confidence"),
]


# ── data loading ──────────────────────────────────────────────────────────────

def frame_ids(seq_dir):
    """Frame-id strings ordered to match the GT poses / separation cache index."""
    pf = sorted(os.listdir(os.path.join(seq_dir, "object_poses")))
    return [p[5:-4] for p in pf]


def load_depth_m(seq_dir, fid):
    """Dataset GT depth (metres); 16-bit PNG in mm."""
    d = np.array(Image.open(os.path.join(seq_dir, "depth", f"{fid}.png")))
    return d.astype(np.float32) / 1000.0


def load_mask(seq_dir, fid):
    p = os.path.join(seq_dir, f"LF_{fid}", "masks", f"{R.CENTRAL_VIEW:04d}.png")
    return np.array(Image.open(p).convert("L")) > 10


def load_input_rgb(seq_dir, fid):
    p = os.path.join(seq_dir, f"LF_{fid}", f"{R.CENTRAL_VIEW:04d}.png")
    return np.array(Image.open(p).convert("RGB"))


def cache_dir(split, seq):
    return os.path.join(CACHE_ROOT, split, seq)


def has_cache(split, seq, i):
    d = cache_dir(split, seq)
    return all(
        os.path.exists(os.path.join(d, f"diffuse_{i:04d}{s}.npy"))
        for s in ("", "_env", "_envconf")
    )


def separation_valid(split, seq, i, min_cov=1000, max_env_nan=0.25):
    """True if the cached separation is well-populated (some frames separate to a
    degenerate all-zero diffuse + mostly-NaN env map; those make ugly black panels)."""
    if not has_cache(split, seq, i):
        return False
    d = cache_dir(split, seq)
    diff = np.load(os.path.join(d, f"diffuse_{i:04d}.npy"))
    cov = int((np.isfinite(diff).all(-1) & (diff.sum(-1) > 1e-3)).sum())
    env = np.load(os.path.join(d, f"diffuse_{i:04d}_env.npy"))
    return cov >= min_cov and float(np.isnan(env).mean()) <= max_env_nan


# ── geometry: depth → camera-space points + normals ──────────────────────────

def backproject(depth, K):
    """Per-pixel camera-space points [H,W,3] from a depth map and intrinsics."""
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    X = (us - cx) * depth / fx
    Y = (vs - cy) * depth / fy
    return np.stack([X, Y, depth], axis=-1).astype(np.float32)


def normals_from_points(P, mask):
    """Camera-space surface normals [H,W,3] from a point map, oriented to camera."""
    dpdx = np.zeros_like(P)
    dpdy = np.zeros_like(P)
    dpdx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dpdy[1:-1, :] = P[2:, :] - P[:-2, :]
    n = np.cross(dpdx, dpdy)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.clip(norm, 1e-8, None)
    # camera at origin → surface normal should face the viewer (n·(-P) > 0)
    flip = np.sum(n * (-P), axis=-1, keepdims=True) < 0
    n = np.where(flip, -n, n)
    n[~mask] = 0.0
    return n.astype(np.float32)


# ── per-cell panel builders (all return RGB uint8, object-cropped where noted) ──

def crop_box(mask, pad=CROP_PAD):
    ys, xs = np.where(mask)
    cx = 0.5 * (xs.min() + xs.max())
    cy = 0.5 * (ys.min() + ys.max())
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + pad)
    half = side / 2
    H, W = mask.shape
    cx = float(np.clip(cx, half, W - half))
    cy = float(np.clip(cy, half, H - half))
    return int(cx - half), int(cy - half), int(cx + half), int(cy + half)


def _crop_resize(rgb_u8, box):
    return np.array(
        Image.fromarray(rgb_u8).crop(box).resize((CELL_PX, CELL_PX), Image.LANCZOS)
    )


def colormap_panel(values, mask, box, cmap, vmin, vmax, bg=255):
    norm = np.clip((values - vmin) / max(vmax - vmin, 1e-8), 0, 1)
    rgb = (matplotlib.colormaps[cmap](norm)[..., :3] * 255).astype(np.uint8)
    rgb[~mask] = bg
    return _crop_resize(rgb, box)


def build_object_panels(split, seq, seq_dir, fid, i, K):
    """All object-cropped panels for one frame (input, depth, normals, albedo, shading)."""
    mask = load_mask(seq_dir, fid)
    depth = load_depth_m(seq_dir, fid)
    box = crop_box(mask)
    valid = mask & (depth > 1e-4)

    # input
    inp = _crop_resize(load_input_rgb(seq_dir, fid), box)

    # depth (turbo over the object's depth range)
    dvals = depth[valid]
    dmin, dmax = np.percentile(dvals, 2), np.percentile(dvals, 98)
    depth_panel = colormap_panel(depth, valid, box, "turbo", dmin, dmax)

    # geometry → normals
    P = backproject(depth, K)
    N = normals_from_points(P, valid)
    nrgb = ((N * 0.5 + 0.5) * 255).astype(np.uint8)
    nrgb[~valid] = 255
    normals_panel = _crop_resize(nrgb, box)

    # diffuse decomposition (cache holds the separated diffuse image, linear)
    diffuse = np.nan_to_num(
        np.load(os.path.join(cache_dir(split, seq), f"diffuse_{i:04d}.npy"))
    )
    m = valid.reshape(-1)
    pts = torch.from_numpy(P.reshape(-1, 3)[m])
    nrm = torch.from_numpy(N.reshape(-1, 3)[m])
    diff_pts = torch.from_numpy(diffuse.reshape(-1, 3)[m].astype(np.float32))

    albedo_pts = np.nan_to_num(unshade_to_albedo(diff_pts, pts, nrm).clamp(0, 1).numpy())
    amb_diff, _ = shading_terms(pts, nrm)
    shade_pts = amb_diff.clamp(0, 1).mean(-1).numpy()  # grayscale shading factor

    albedo_img = np.full((*mask.shape, 3), 255, np.uint8)
    flat = albedo_img.reshape(-1, 3)
    flat[m] = (albedo_pts * 255).astype(np.uint8)
    albedo_panel = _crop_resize(albedo_img, box)

    shade_img = np.zeros(mask.shape, np.float32)
    shade_img.reshape(-1)[m] = shade_pts
    shading_panel = colormap_panel(shade_img, valid, box, "gray", 0.4, 1.0)

    return {
        "input": inp,
        "depth": depth_panel,
        "normals": normals_panel,
        "albedo": albedo_panel,
        "shading": shading_panel,
    }


def build_env_panels(split, seq, i):
    """Equirect environment map (sRGB) and its observation-confidence heatmap."""
    d = cache_dir(split, seq)
    env = np.nan_to_num(np.load(os.path.join(d, f"diffuse_{i:04d}_env.npy"))).clip(0, 1)
    env_srgb = (linear_to_srgb(torch.from_numpy(env)).clamp(0, 1).numpy() * 255).astype(
        np.uint8
    )
    conf = np.nan_to_num(np.load(os.path.join(d, f"diffuse_{i:04d}_envconf.npy")))
    conf_rgb = (matplotlib.colormaps["magma"](np.clip(conf, 0, 1))[..., :3] * 255).astype(
        np.uint8
    )
    return {"env": env_srgb, "conf": conf_rgb}


# ── frame selection ───────────────────────────────────────────────────────────

def pick_best_frame(split, seq, seq_dir, gt, est, mesh_pts):
    """Frame (back half) with the lowest ADD error and a clean, cached separation."""
    fids = frame_ids(seq_dir)
    n = len(fids)
    start = int(n * FRAME_MIN_FRAC)
    best, best_err = None, np.inf
    for i in range(start, n):
        if not separation_valid(split, seq, i):
            continue
        mask = load_mask(seq_dir, fids[i])
        if int(mask.sum()) < MIN_MASK_PX:
            continue
        err = R._add_err(est[i], gt[i], mesh_pts)
        if err < best_err:
            best, best_err = i, err
    return best, best_err


# ── plotting ──────────────────────────────────────────────────────────────────

def pretty_name(split, seq):
    obj = R.get_object_name(DATASET_ROOT, split, seq).replace("_", " ").title()
    return obj


def top_sequences(split):
    """Top-K sequences for this split by ADD-S AUC (the ablation_full_gt_mask run)."""
    import json

    m = json.load(open(METRICS))["per_sequence"]["objects_gt"][split]
    ranked = sorted(m.items(), key=lambda kv: kv[1]["adds_auc"], reverse=True)
    return [s for s, _ in ranked]


def build_row(split, seq):
    seq_dir = os.path.join(DATASET_ROOT, split, seq)
    K = R.load_camera_matrix(seq_dir)
    gt = R.load_gt_poses(seq_dir)
    est = np.load(os.path.join(RESULTS_ROOT, split, f"{seq}.npy"))
    mesh_pts = R.load_mesh_pts(DATASET_ROOT, split, seq)
    i, err = pick_best_frame(split, seq, seq_dir, gt, est, mesh_pts)
    if i is None:
        return None
    fid = frame_ids(seq_dir)[i]
    panels = build_object_panels(split, seq, seq_dir, fid, i, K)
    panels.update(build_env_panels(split, seq, i))
    return {"seq": seq, "frame": i, "fid": fid, "add_mm": err * 1000, "panels": panels}


def render_figure(split, rows, out_stem):
    keys = [k for k, _ in COLS]
    # square object cells (ratio 1) + wide 2:1 equirect cells (ratio 2)
    wratios = [2 if k in ("env", "conf") else 1 for k in keys]
    nrow, ncol = len(rows), len(keys)
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(sum(wratios) * 1.55, nrow * 1.55),
        gridspec_kw={"wspace": 0.04, "hspace": 0.04, "width_ratios": wratios},
        squeeze=False,
    )
    for r, row in enumerate(rows):
        for c, (key, header) in enumerate(COLS):
            ax = axes[r][c]
            ax.imshow(row["panels"][key], aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("white")
                s.set_linewidth(1.4)
            if r == 0:
                ax.set_title(header, fontsize=11, fontweight="bold", pad=6)
            if c == 0:
                ax.set_ylabel(
                    pretty_name(split, row["seq"]),
                    fontsize=11,
                    fontweight="bold",
                    labelpad=6,
                )
                ax.text(
                    0.03,
                    0.96,
                    f"{row['seq']}\n#{row['fid']}  ADD {row['add_mm']:.1f}mm",
                    transform=ax.transAxes,
                    fontsize=7,
                    color="white",
                    va="top",
                    ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.55),
                )

    refl = split.split("_")[-1]
    fig.suptitle(
        f"Relightable surface-light-field representation  (reflectivity {refl})",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.92, bottom=0.008)

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        out = os.path.join(OUT_DIR, f"{out_stem}.{ext}")
        fig.savefig(out, dpi=300)
        print("Saved →", out)
    plt.close(fig)


def run_split(split, seqs=None, by_type=False):
    # Walk the ADD-S-AUC ranking and keep the first TOPK sequences that yield a
    # valid "perfect" frame (a sequence with no clean separated frame is skipped
    # and the next-ranked one takes its place, so we always fill the grid).
    # by_type: at most one sequence per object type (4 distinct objects here).
    ranked = seqs or top_sequences(split)
    print(f"[{split}] ranking: {ranked}")
    rows = []
    seen_types = set()
    for seq in ranked:
        if len(rows) >= TOPK:
            break
        otype = R.get_object_name(DATASET_ROOT, split, seq)
        if by_type and otype in seen_types:
            continue
        row = build_row(split, seq)
        if row is None:
            print(f"  {seq}: no valid frame, skipping")
            continue
        seen_types.add(otype)
        print(f"  {seq}: frame #{row['fid']} (i={row['frame']}) ADD {row['add_mm']:.1f}mm")
        rows.append(row)
    if rows:
        render_figure(split, rows, f"representation_grid_{split}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None, help="e.g. objects_0.7 (default: 0.7 and 0.5)")
    ap.add_argument("--seq", default=None, help="single sequence (with --split)")
    ap.add_argument("--frame", type=int, default=None, help="force a frame index")
    ap.add_argument("--by-type", action="store_true",
                    help="at most one row per object type (4 distinct objects)")
    args = ap.parse_args()

    if args.split and args.seq:
        row = None
        seq_dir = os.path.join(DATASET_ROOT, args.split, args.seq)
        if args.frame is not None:
            # single forced frame
            K = R.load_camera_matrix(seq_dir)
            fid = frame_ids(seq_dir)[args.frame]
            gt = R.load_gt_poses(seq_dir)
            est = np.load(os.path.join(RESULTS_ROOT, args.split, f"{args.seq}.npy"))
            mesh_pts = R.load_mesh_pts(DATASET_ROOT, args.split, args.seq)
            panels = build_object_panels(args.split, args.seq, seq_dir, fid, args.frame, K)
            panels.update(build_env_panels(args.split, args.seq, args.frame))
            row = {
                "seq": args.seq,
                "frame": args.frame,
                "fid": fid,
                "add_mm": R._add_err(est[args.frame], gt[args.frame], mesh_pts) * 1000,
                "panels": panels,
            }
        else:
            row = build_row(args.split, args.seq)
        render_figure(args.split, [row], f"representation_grid_{args.split}_{args.seq}")
    else:
        for sp in ([args.split] if args.split else ["objects_0.7", "objects_0.5"]):
            run_split(sp, by_type=args.by_type)
