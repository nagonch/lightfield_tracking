#!/usr/bin/env python3
"""Paper hero figure for reflection separation (r=0.7, GT depth).

Design: lead with the diffuse-reflection *decomposition* (the strong, honest
result), demote the environment map to a chrome light-probe sphere shown only
for a sequence that actually covers the scene.

Layout (rows = best objects, by diffuse PSNR):
  Input (obs_r) | Diffuse (ours) | Diffuse (GT) | Reflection (ours) || Env probes
The env-probe column (GT ball over ours) is a single best-covered sequence.
"""
import os
import glob

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reflection_report import (
    CACHE, DATASET, srgb_to_linear, linear_to_srgb, u8, env2_linear, list_cache_frames,
)
from scipy.ndimage import gaussian_filter

REFL = "0.7"
DEPTH = "gt"
SPLIT = "objects"
N_OBJ = 3
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reflection_report",
                   f"hero_{SPLIT}_{REFL}_{DEPTH}.png")


def central_linear(split, refl, seq, fi):
    seqdir = os.path.join(DATASET, f"{split}_{refl}", seq)
    frames = sorted(d for d in os.listdir(seqdir) if d.startswith("LF_"))
    fi = min(fi, len(frames) - 1)
    fdir = os.path.join(seqdir, frames[fi])
    pngs = sorted(f for f in os.listdir(fdir) if f.endswith(".png"))
    n = int(round(len(pngs) ** 0.5)); mid = (n // 2) * n + (n // 2)
    img = np.asarray(Image.open(os.path.join(fdir, pngs[mid])).convert("RGB")).astype(np.float32)
    return srgb_to_linear(img / 255.0)


def square_bbox(mask, pad_frac=0.12):
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    half = max(y1 - y0, x1 - x0) / 2 * (1 + pad_frac)
    H, W = mask.shape
    y0 = int(max(cy - half, 0)); y1 = int(min(cy + half, H))
    x0 = int(max(cx - half, 0)); x1 = int(min(cx + half, W))
    return y0, y1, x0, x1


def mirror_ball(env, size=200, ours=False, blur=1.2, bg=1.0):
    env = np.nan_to_num(env, nan=0.0)
    if blur > 0:
        env = np.stack([gaussian_filter(env[..., c], blur, mode="wrap") for c in range(3)], -1)
    H, W = env.shape[:2]
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    x = (xs / (size - 1)) * 2 - 1; y = -((ys / (size - 1)) * 2 - 1)
    r2 = x * x + y * y; disk = r2 <= 1.0
    nz = np.sqrt(np.clip(1 - r2, 0, 1))
    Rx, Ry, Rz = 2 * nz * x, 2 * nz * y, 2 * nz * nz - 1
    if ours:   # flip_u=flip_v convention
        u = (0.5 - np.arctan2(Rx, Rz) / (2 * np.pi)) % 1.0
        v = np.clip(0.5 + np.arcsin(np.clip(Ry, -1, 1)) / np.pi, 0, 1)
    else:      # renderer convention
        u = (0.5 + np.arctan2(Rx, Rz) / (2 * np.pi)) % 1.0
        v = np.clip(0.5 - np.arcsin(np.clip(Ry, -1, 1)) / np.pi, 0, 1)
    px = np.clip((u * (W - 1)).astype(int), 0, W - 1)
    py = np.clip((v * (H - 1)).astype(int), 0, H - 1)
    out = np.ones((size, size, 3), np.float32) * bg
    out[disk] = env[py[disk], px[disk]]
    return out, disk


def seq_record(seq, alpha_default=0.3):
    cache_dir = os.path.join(CACHE, DEPTH, f"{SPLIT}_{REFL}", seq)
    fs = list_cache_frames(cache_dir)
    if not fs:
        return None
    stem = fs[-1][:-4]
    D = np.nan_to_num(np.load(stem + ".npy"))
    env = np.load(stem + "_env.npy")
    conf = np.load(stem + "_envconf.npy")
    alpha = float(np.load(stem + "_alpha.npy")) if os.path.exists(stem + "_alpha.npy") else alpha_default
    fi = len(fs) - 1
    inp = central_linear(SPLIT, REFL, seq, fi)
    gtD = central_linear(SPLIT, "0.0", seq, fi)
    valid = D.sum(-1) > 1e-6
    if valid.sum() < 50:
        return None
    mse = ((u8(D).astype(np.float32) - u8(gtD).astype(np.float32))[valid] ** 2).mean()
    psnr = float(-10 * np.log10(mse / 255.0 ** 2 + 1e-12))
    refl = np.clip((inp - alpha * D) / max(1 - alpha, 1e-3), 0, 1)
    env_qual = float(np.nan_to_num(env).max(-1).mean()) * float((conf > 0.5).mean())
    return dict(seq=seq, inp=inp, D=D, gtD=gtD, refl=refl, env=env, valid=valid,
                alpha=alpha, psnr=psnr, env_qual=env_qual)


def main():
    root = os.path.join(CACHE, DEPTH, f"{SPLIT}_{REFL}")
    recs = [r for r in (seq_record(s) for s in sorted(os.listdir(root))
                        if not s.startswith("tomato")
                        and os.path.isdir(os.path.join(root, s))) if r]
    diffuse_rows = sorted(recs, key=lambda r: -r["psnr"])[:N_OBJ]
    env_rec = max(recs, key=lambda r: r["env_qual"])
    print("diffuse rows:", [(r["seq"], round(r["psnr"], 1)) for r in diffuse_rows])
    print("env probe seq:", env_rec["seq"], "qual", round(env_rec["env_qual"], 3))

    env2 = env2_linear()
    col_titles = [f"Input  (r={REFL})", "Diffuse — ours", "Diffuse — GT", "Reflection — ours"]
    fig, axes = plt.subplots(N_OBJ, 5, figsize=(15, 3.1 * N_OBJ),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.05]))
    for ri, r in enumerate(diffuse_rows):
        y0, y1, x0, x1 = square_bbox(r["valid"])
        vm = r["valid"][y0:y1, x0:x1][..., None]
        panels = [r["inp"][y0:y1, x0:x1],
                  r["D"][y0:y1, x0:x1] * vm,
                  r["gtD"][y0:y1, x0:x1] * vm,
                  r["refl"][y0:y1, x0:x1] * vm]
        for ci, im in enumerate(panels):
            axes[ri, ci].imshow(u8(im))
            axes[ri, ci].set_xticks([]); axes[ri, ci].set_yticks([])
            if ri == 0:
                axes[ri, ci].set_title(col_titles[ci], fontsize=12)
        axes[ri, 1].text(0.03, 0.05, f"{r['psnr']:.1f} dB", transform=axes[ri, 1].transAxes,
                         color="w", fontsize=10, va="bottom", ha="left",
                         bbox=dict(boxstyle="round,pad=0.2", fc="k", ec="none", alpha=0.6))
        axes[ri, 0].set_ylabel(r["seq"].replace("_", " ")[:18], fontsize=10)

    # env-probe column (col 4): GT ball (row 0), ours ball (row 1), caption (row 2)
    gt_ball, disk = mirror_ball(env2, ours=False)
    our_ball, _ = mirror_ball(env_rec["env"], ours=True)
    for ax, ball, lab in [(axes[0, 4], gt_ball, "Illumination — GT"),
                          (axes[1, 4], our_ball, "Illumination — ours")]:
        ax.imshow(u8(ball)); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(lab, fontsize=12)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[2, 4].axis("off")
    axes[2, 4].text(0.5, 0.5, f"light probe\n(α={diffuse_rows[0]['alpha']:.2f},  "
                              f"env from\n{env_rec['seq'].replace('_',' ')[:16]})",
                    ha="center", va="center", fontsize=9, transform=axes[2, 4].transAxes)

    fig.suptitle(f"Reflection separation on YCB objects  (reflectivity r={REFL}, "
                 f"α={1-float(REFL):.1f};  {DEPTH.upper()} depth)\n"
                 f"a strongly-reflective input is decomposed into a clean diffuse "
                 f"albedo + the surrounding illumination", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT, dpi=130); plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
