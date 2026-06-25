#!/usr/bin/env python3
"""Production qualitative grid for reflection separation (r=0.7).

One CVPR-style figure. Rows = one best cube sequence + one best sequence per YCB
object type (cracker_box, sugar_box, mustard, bleach). Columns:

  Input | GT diffuse | Ours diffuse (GT depth) | Ours diffuse (est. depth)
        ‖ GT env | Ours env (GT depth) | Ours env (est. depth)

Object panels are square crops (uniform aspect, no weird crops); env panels are
equirectangular. Per sequence a single frame is chosen that is healthy (mask not
collapsed) in BOTH depth caches and has the highest env coverage — the same
frame for GT/est depth so the comparison is fair. Saved as PNG and PDF.
"""
import os
import glob

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reflection_report import CACHE, DATASET, srgb_to_linear, u8, env2_linear
from scipy.ndimage import gaussian_filter

ENV_RECON_BLUR = 1.3   # recon env is intrinsically low-frequency; light blur drops splat dots

REFL = "0.7"
OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "reflection_report", f"grid_final_r{REFL}.png")
PROGRESS_PROBES = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
MIN_VALID = 1500

# (row label, split prefix, sequence-name filter)
GROUPS = [
    ("cube", "cube", lambda s: True),
    ("cracker box", "objects", lambda s: s.startswith("cracker_box")),
    ("sugar box", "objects", lambda s: s.startswith("sugar_box")),
    ("mustard", "objects", lambda s: s.startswith("mustard")),
    ("bleach", "objects", lambda s: s.startswith("bleach")),
]


def cache_frames(depth, split, seq):
    d = os.path.join(CACHE, depth, f"{split}_{REFL}", seq)
    fs = sorted(glob.glob(os.path.join(d, "diffuse_*.npy")))
    return [f for f in fs if not f.endswith(("_env.npy", "_envconf.npy", "_alpha.npy"))]


def coverage_at(depth, split, seq, fi):
    f = cache_frames(depth, split, seq)[fi]
    return float((np.load(f[:-4] + "_envconf.npy") > 0.5).mean())


def load_diffuse(depth, split, seq, fi):
    return np.nan_to_num(np.load(cache_frames(depth, split, seq)[fi]))


def load_env(depth, split, seq, fi):
    return np.nan_to_num(np.load(cache_frames(depth, split, seq)[fi][:-4] + "_env.npy"))


def dataset_frames(split, refl, seq):
    sd = os.path.join(DATASET, f"{split}_{refl}", seq)
    return sd, sorted(d for d in os.listdir(sd) if d.startswith("LF_"))


def central_index(fdir):
    pngs = sorted(f for f in os.listdir(fdir) if f.endswith(".png"))
    n = int(round(len(pngs) ** 0.5))
    return pngs, (n // 2) * n + (n // 2)


def central_linear(split, refl, seq, fi):
    sd, frames = dataset_frames(split, refl, seq)
    fdir = os.path.join(sd, frames[min(fi, len(frames) - 1)])
    pngs, mid = central_index(fdir)
    img = np.asarray(Image.open(os.path.join(fdir, pngs[mid])).convert("RGB")).astype(np.float32)
    return srgb_to_linear(img / 255.0)


def central_mask(split, refl, seq, fi):
    sd, frames = dataset_frames(split, refl, seq)
    fdir = os.path.join(sd, frames[min(fi, len(frames) - 1)], "masks")
    if not os.path.isdir(fdir):
        return None
    pngs, mid = central_index(fdir)
    return np.asarray(Image.open(os.path.join(fdir, pngs[mid])).convert("L")) > 127


def evaluate_seq(split, seq):
    """Probe frames healthy in BOTH depths. Pick the cleanest-diffuse frame
    (max PSNR vs GT) and, separately, the best-covered env frame (the env is an
    accumulated quantity). Returns a dict or None."""
    n = min(len(cache_frames("gt", split, seq)), len(cache_frames("lf", split, seq)))
    if n == 0:
        return None
    healthy = []
    for p in [0.0] + PROGRESS_PROBES:
        fi = min(int(p * (n - 1)), n - 1)
        Dg = load_diffuse("gt", split, seq, fi)
        vg = (Dg.sum(-1) > 1e-6)
        vl = (load_diffuse("lf", split, seq, fi).sum(-1) > 1e-6).sum()
        if vg.sum() >= MIN_VALID and vl >= MIN_VALID:
            gt = central_linear(split, "0.0", seq, fi)
            mse = ((u8(Dg).astype(np.float32) - u8(gt).astype(np.float32))[vg] ** 2).mean()
            psnr = float(-10 * np.log10(mse / 255.0 ** 2 + 1e-12))
            healthy.append((fi, coverage_at("gt", split, seq, fi), psnr))
    if not healthy:
        healthy = [(0, coverage_at("gt", split, seq, 0), 0.0)]
    diff_fi, _, max_psnr = max(healthy, key=lambda t: t[2])
    env_fi, max_cov, _ = max(healthy, key=lambda t: t[1])
    return dict(seq=seq, diff_fi=diff_fi, env_fi=env_fi, psnr=max_psnr, cov=max_cov,
                score=max_psnr / 20.0 + max_cov)


def square_bbox(mask, pad=0.18):
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
    half = max(y1 - y0, x1 - x0) * (1 + pad) / 2.0
    H, W = mask.shape
    half = min(half, cy + 0.5, H - cy - 0.5, cx + 0.5, W - cx - 0.5)  # keep in-frame, stay square
    y0 = int(round(cy - half)); x0 = int(round(cx - half))
    s = int(round(2 * half))
    return y0, y0 + s, x0, x0 + s


def build_row(label, split, filt):
    root = os.path.join(CACHE, "gt", f"{split}_{REFL}")
    seqs = [s for s in sorted(os.listdir(root))
            if filt(s) and not s.startswith("tomato")
            and os.path.isdir(os.path.join(root, s))]
    best = None
    for seq in seqs:
        ev = evaluate_seq(split, seq)
        if ev is not None and (best is None or ev["score"] > best["score"]):
            best = ev
    if best is None:
        return None
    seq, dfi, efi = best["seq"], best["diff_fi"], best["env_fi"]
    m = central_mask(split, REFL, seq, dfi)
    if m is None or m.sum() < 50:
        m = load_diffuse("gt", split, seq, dfi).sum(-1) > 1e-6
    y0, y1, x0, x1 = square_bbox(m)
    sl = (slice(y0, y1), slice(x0, x1))
    mc = m[sl][..., None]
    inp = central_linear(split, REFL, seq, dfi)[sl]
    gtd = central_linear(split, "0.0", seq, dfi)[sl] * mc
    rec_g = load_diffuse("gt", split, seq, dfi)[sl]
    rec_l = load_diffuse("lf", split, seq, dfi)[sl]
    def env(depth):
        e = load_env(depth, split, seq, efi)
        return np.stack([gaussian_filter(e[..., c], ENV_RECON_BLUR, mode="wrap")
                         for c in range(3)], -1) if ENV_RECON_BLUR > 0 else e
    print(f"  {label:12s} -> {seq:28s} diffuse f{dfi:<3d} env f{efi:<3d} "
          f"cov={best['cov']:.2f} psnr={best['psnr']:.1f}", flush=True)
    return dict(label=label, seq=seq,
                panels=[inp, gtd, rec_g, rec_l], envs=[env2_linear(), env("gt"), env("lf")])


def main():
    rows = [r for r in (build_row(*g) for g in GROUPS) if r]
    nrows = len(rows)
    obj_titles = ["Input", "GT diffuse", "Ours\n(GT depth)", "Ours\n(est. depth)"]
    env_titles = ["GT env", "Ours\n(GT depth)", "Ours\n(est. depth)"]
    # 4 square object cols | spacer | 3 equirect (2:1) env cols
    wr = [1, 1, 1, 1, 0.28, 2, 2, 2]
    fig, axes = plt.subplots(nrows, 8, figsize=(sum(wr) * 1.5, nrows * 1.5 + 0.7),
                             gridspec_kw=dict(width_ratios=wr, wspace=0.04, hspace=0.06))
    if nrows == 1:
        axes = axes[None, :]
    for ax in axes[:, 4]:
        ax.axis("off")
    for ri, r in enumerate(rows):
        cells = list(zip([0, 1, 2, 3], r["panels"])) + list(zip([5, 6, 7], r["envs"]))
        for ci, im in cells:
            ax = axes[ri, ci]
            ax.imshow(u8(im)); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("0.8"); s.set_linewidth(0.6)
            if ri == 0:
                ax.set_title((obj_titles + [None] + env_titles)[ci], fontsize=11, pad=4)
        axes[ri, 0].set_ylabel(r["label"], fontsize=12, rotation=90,
                               va="center", labelpad=8)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    # group headers, centred on the actual column blocks
    p = lambda i: axes[0, i].get_position()
    x_diff = (p(0).x0 + p(3).x1) / 2
    x_env = (p(5).x0 + p(7).x1) / 2
    y_hdr = max(p(0).y1, p(5).y1) + 0.045
    fig.text(x_diff, y_hdr, "Diffuse separation", ha="center", fontsize=14, weight="bold")
    fig.text(x_env, y_hdr, "Environment map", ha="center", fontsize=14, weight="bold")
    from matplotlib.lines import Line2D
    xdiv = (p(3).x1 + p(5).x0) / 2
    ybot = axes[-1, 0].get_position().y0
    fig.add_artist(Line2D([xdiv, xdiv], [ybot, y_hdr - 0.01], color="0.7", lw=1.0))
    fig.savefig(OUT_PNG, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT_PNG[:-4] + ".pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT_PNG, "and .pdf")


if __name__ == "__main__":
    main()
