#!/usr/bin/env python3
"""First-frame qualitative grids for reflection separation — ALL sequences.

For frame 0 of every sequence, in each split × reflectivity, lay out five columns:
    Original (obs_r) | GT diffuse (obs_0) | Recon diffuse | GT env | Recon env

Produced for BOTH depth sources so they can be compared:
    eval/reflection_report/firstframe_gt/grid_<split>_<r>.png   (GT depth)
    eval/reflection_report/firstframe_lf/grid_<split>_<r>.png   (estimated depth)
Same filenames in the two folders → open side by side to compare.

Lightweight: only frame 0 of each sequence is read. Reuses helpers from
eval/reflection_report.py.
"""
import os
import argparse

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reflection_report import (
    CACHE, DATASET, SPLITS, REFLS, REFL_F, EXCLUDE_PREFIXES,
    srgb_to_linear, u8, env2_linear, obj_of, bbox_of, list_cache_frames,
)

OUTROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reflection_report")


def central_linear(split_prefix, refl, seq, fi=0):
    """Central sub-aperture of the {split}_{refl}/{seq} render at frame fi (linear)."""
    seqdir = os.path.join(DATASET, f"{split_prefix}_{refl}", seq)
    if not os.path.isdir(seqdir):
        return None
    frames = sorted(d for d in os.listdir(seqdir) if d.startswith("LF_"))
    if fi >= len(frames):
        return None
    fdir = os.path.join(seqdir, frames[fi])
    pngs = sorted(f for f in os.listdir(fdir) if f.endswith(".png"))
    if not pngs:
        return None
    n = int(round(len(pngs) ** 0.5))
    mid = min((n // 2) * n + (n // 2), len(pngs) - 1)
    img = np.asarray(Image.open(os.path.join(fdir, pngs[mid])).convert("RGB")).astype(np.float32)
    return srgb_to_linear(img / 255.0)


def first_frame_data(depth, split_prefix, refl, seq):
    cache_dir = os.path.join(CACHE, depth, f"{split_prefix}_{refl}", seq)
    diff_files = list_cache_frames(cache_dir)
    if not diff_files:
        return None
    stem = diff_files[0][:-4]
    try:
        recon_d = np.load(stem + ".npy")
        recon_env = np.load(stem + "_env.npy")
        conf = np.load(stem + "_envconf.npy")
    except Exception:
        return None
    original = central_linear(split_prefix, refl, seq, 0)
    gt_diffuse = central_linear(split_prefix, "0.0", seq, 0)
    valid = recon_d.sum(-1) > 1e-6
    if gt_diffuse is None or valid.sum() < 50:
        return None
    mse = ((u8(recon_d).astype(np.float32) - u8(gt_diffuse).astype(np.float32))[valid] ** 2).mean()
    psnr = float(-10 * np.log10(mse / (255.0 ** 2) + 1e-12))
    return {"original": original, "gt_diffuse": gt_diffuse, "recon_diffuse": recon_d,
            "recon_env": recon_env, "valid": valid, "cov": float((conf > 0.5).mean()),
            "psnr": psnr, "seq": seq}


def build_grid(depth, split_prefix, refl):
    root = os.path.join(CACHE, depth, f"{split_prefix}_{refl}")
    if not os.path.isdir(root):
        return
    rows = []
    for seq in sorted(os.listdir(root)):
        if seq.startswith(EXCLUDE_PREFIXES) or not os.path.isdir(os.path.join(root, seq)):
            continue
        d = first_frame_data(depth, split_prefix, refl, seq)
        if d is not None:
            rows.append(d)
    if not rows:
        return

    env2 = env2_linear()
    cols = [f"original (r={refl})", "GT diffuse", "recon diffuse",
            "GT env (env2)", "recon env"]
    fig, axes = plt.subplots(len(rows), 5, figsize=(15.5, 2.7 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    for ri, d in enumerate(rows):
        valid = d["valid"]
        bb = bbox_of(valid)
        orig = d["original"] if d["original"] is not None else np.zeros_like(d["recon_diffuse"])
        gt = d["gt_diffuse"] * valid[..., None]
        rec = d["recon_diffuse"].copy(); rec[~valid] = 0
        if bb:
            y0, y1, x0, x1 = bb
            orig_c, gt_c, rec_c = orig[y0:y1, x0:x1], gt[y0:y1, x0:x1], rec[y0:y1, x0:x1]
        else:
            orig_c, gt_c, rec_c = orig, gt, rec
        for ci, im in enumerate([orig_c, gt_c, rec_c, env2, d["recon_env"]]):
            axes[ri, ci].imshow(u8(im))
            axes[ri, ci].set_xticks([]); axes[ri, ci].set_yticks([])
            if ri == 0:
                axes[ri, ci].set_title(cols[ci], fontsize=10)
        axes[ri, 0].set_ylabel(f"{d['seq'][:18]}\nPSNR={d['psnr']:.1f}\ncov={d['cov']:.2f}",
                               fontsize=7.5)
    fig.suptitle(f"{split_prefix}  ·  r={refl}  (alpha={1-REFL_F[refl]:.1f})  ·  "
                 f"{depth.upper()} depth  ·  FIRST FRAME — all sequences",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    outdir = os.path.join(OUTROOT, f"firstframe_{depth}")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"grid_{split_prefix}_{refl}.png")
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"wrote {out}  ({len(rows)} rows)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="gt,lf",
                    help="comma-separated depth caches to render (default both)")
    args = ap.parse_args()
    for depth in [d for d in args.depths.split(",") if d]:
        for sp in SPLITS:
            for r in REFLS:
                build_grid(depth, sp, r)


if __name__ == "__main__":
    main()
