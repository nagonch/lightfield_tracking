#!/usr/bin/env python3
"""Clean env-accumulation figure for cube/mustard_easy_00_02.

Top row    : RGB central view as the cube rotates (4 frames).
Bottom row : reconstructed env map at the same frames, THRESHOLDED by observation
             confidence — only the part of the env we have actually observed is
             shown (the rest is left blank).
Right      : GT env map (reference).

Estimated (lf) depth — GT depth loses cube tracking too early to fill the env.
"""
import os
import glob

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter

from reflection_report import CACHE, DATASET, srgb_to_linear, linear_to_srgb, u8, env2_linear

ENV_BLUR = 1.0          # recon env is intrinsically low-frequency; light blur drops splat dots
DEPTH = "lf"
REFL = "0.7"
SPLIT = "cube"
SEQ = "mustard_easy_00_02"
CONF_THR = 0.5          # show env only where confidence exceeds this
N_FRAMES = 4
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "reflection_report", "env_cube_mustard_easy.png")


def frame_files():
    d = os.path.join(CACHE, DEPTH, f"{SPLIT}_{REFL}", SEQ)
    fs = sorted(glob.glob(os.path.join(d, "diffuse_*.npy")))
    return [f for f in fs if not f.endswith(("_env.npy", "_envconf.npy", "_alpha.npy"))]


def load_env_conf(fi):
    f = frame_files()[fi]
    return (np.nan_to_num(np.load(f[:-4] + "_env.npy")),
            np.nan_to_num(np.load(f[:-4] + "_envconf.npy")))


def coverage_curve():
    return np.array([float((np.nan_to_num(np.load(f[:-4] + "_envconf.npy")) > CONF_THR).mean())
                     for f in frame_files()])


def pick_frames(k=N_FRAMES):
    """First frames reaching k evenly spaced coverage levels (monotonic growth;
    coverage isn't monotone because the env resets on alpha instability)."""
    cov = coverage_curve()
    run = np.maximum.accumulate(cov)
    idx = []
    for t in np.linspace(cov[0], run[-1], k):
        f = int(np.where(run >= t - 1e-9)[0][0])
        while f in idx and f < len(cov) - 1:
            f += 1
        idx.append(f)
    return sorted(set(idx)), cov


def dataset_frame_dir(fi):
    sd = os.path.join(DATASET, f"{SPLIT}_{REFL}", SEQ)
    frames = sorted(d for d in os.listdir(sd) if d.startswith("LF_"))
    return os.path.join(sd, frames[min(fi, len(frames) - 1)])


def central_view(fi):
    fdir = dataset_frame_dir(fi)
    pngs = sorted(f for f in os.listdir(fdir) if f.endswith(".png"))
    n = int(round(len(pngs) ** 0.5)); mid = (n // 2) * n + (n // 2)
    img = np.asarray(Image.open(os.path.join(fdir, pngs[mid])).convert("RGB")).astype(np.float32)
    return srgb_to_linear(img / 255.0)


def central_mask(fi):
    md = os.path.join(dataset_frame_dir(fi), "masks")
    if not os.path.isdir(md):
        return None
    pngs = sorted(f for f in os.listdir(md) if f.endswith(".png"))
    n = int(round(len(pngs) ** 0.5)); mid = (n // 2) * n + (n // 2)
    return np.asarray(Image.open(os.path.join(md, pngs[mid])).convert("L")) > 127


def rect_crop(img, mask, aspect=2.0, pad=0.18, fill=1.0):
    """A (W:H = aspect):1 rectangle that fully CONTAINS the object (the cube is
    never cropped) — wider than the object, padding the sides with image content
    where possible, else `fill`. aspect=2.0 matches the 512x256 env maps."""
    H, W = mask.shape[:2]
    ys, xs = np.where(mask)
    cy, cx = (ys.min() + ys.max()) / 2.0, (xs.min() + xs.max()) / 2.0
    oh = (ys.max() - ys.min() + 1) * (1 + pad)
    ow = (xs.max() - xs.min() + 1) * (1 + pad)
    bh = int(round(max(oh, ow / aspect)))
    bw = int(round(aspect * bh))
    y0 = int(round(cy - bh / 2.0)); x0 = int(round(cx - bw / 2.0))
    if bw <= W:                       # shift into frame so real content shows, not padding
        x0 = min(max(x0, 0), W - bw)
    if bh <= H:
        y0 = min(max(y0, 0), H - bh)
    out = np.full((bh, bw, img.shape[2]), fill, np.float32)
    sy0, sx0 = max(y0, 0), max(x0, 0)
    sy1, sx1 = min(y0 + bh, H), min(x0 + bw, W)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def thresholded_env(env, conf, thr=CONF_THR, bg=1.0):
    if ENV_BLUR > 0:
        env = np.stack([gaussian_filter(env[..., c], ENV_BLUR, mode="wrap") for c in range(3)], -1)
    out = np.full_like(env, bg)
    m = conf > thr
    out[m] = linear_to_srgb(np.clip(env, 0, 1))[m]
    return out


def main():
    fis, cov = pick_frames()
    env2 = env2_linear()
    fig = plt.figure(figsize=(14, 3.3))
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 0.12, 1.25],
                          height_ratios=[1, 1], wspace=0.03, hspace=0.05,
                          left=0.05, right=0.99, top=0.84, bottom=0.10)
    for ci, fi in enumerate(fis):
        rgb = central_view(fi); m = central_mask(fi)
        rgb_c = rect_crop(rgb, m) if m is not None and m.sum() > 50 else rgb
        ax_r = fig.add_subplot(gs[0, ci]); ax_r.imshow(u8(rgb_c))
        ax_r.set_title(f"frame {fi}", fontsize=11)
        env, conf = load_env_conf(fi)
        ax_e = fig.add_subplot(gs[1, ci]); ax_e.imshow(thresholded_env(env, conf))
        ax_e.text(0.5, -0.13, f"{cov[fi]*100:.0f}% observed", transform=ax_e.transAxes,
                  ha="center", va="top", fontsize=10)
        if ci == 0:
            ax_r.set_ylabel("middle view", fontsize=12)
            ax_e.set_ylabel("recon env", fontsize=12)
        for ax in (ax_r, ax_e):
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("0.8"); s.set_linewidth(0.6)
    ax_gt = fig.add_subplot(gs[:, 5])
    ax_gt.imshow(u8(env2)); ax_gt.set_title("GT env", fontsize=12)
    ax_gt.set_xticks([]); ax_gt.set_yticks([])
    for s in ax_gt.spines.values():
        s.set_edgecolor("0.55"); s.set_linewidth(0.9)
    fig.suptitle("Environment map reconstruction over time", fontsize=14)
    fig.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT[:-4] + ".pdf", facecolor="white", bbox_inches="tight")
    print("wrote", OUT, "and .pdf  | frames", fis, "cov", [round(float(cov[i]), 2) for i in fis])


if __name__ == "__main__":
    main()
