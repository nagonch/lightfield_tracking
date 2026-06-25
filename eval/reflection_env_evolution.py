#!/usr/bin/env python3
"""Environment-map accumulation + observation-confidence visualization.

For cube and objects `mustard_easy_00_02`, show how the reflected environment map
and its per-pixel observation confidence build up as the object rotates, plus the
GT env and a "what we rely on" map (env weighted by confidence).

Uses estimated (lf) depth: it keeps tracking healthy for the whole sequence,
whereas GT depth collapses early on the cube (the env would never fill in).

Layout (4 rows x 5 cols):
  cube    · recon env :  env@4 coverage milestones        | GT env
  cube    · confidence:  confidence@same frames           | env x confidence
  objects · recon env :  env@4 coverage milestones        | GT env
  objects · confidence:  confidence@same frames           | env x confidence
"""
import os
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reflection_report import CACHE, srgb_to_linear, linear_to_srgb, u8, env2_linear

DEPTH = "lf"
REFL = "0.7"
SEQ = "mustard_easy_00_02"
SPLITS = ["cube", "objects"]
PROGRESS = [0.0, 0.33, 0.66, 1.0]
CMAP = "magma"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "reflection_report", "env_evolution_mustard_easy.png")


def frame_files(split):
    d = os.path.join(CACHE, DEPTH, f"{split}_{REFL}", SEQ)
    fs = sorted(glob.glob(os.path.join(d, "diffuse_*.npy")))
    return [f for f in fs if not f.endswith(("_env.npy", "_envconf.npy", "_alpha.npy"))]


def load(split, fi):
    f = frame_files(split)[fi]
    env = np.nan_to_num(np.load(f[:-4] + "_env.npy"))
    conf = np.nan_to_num(np.load(f[:-4] + "_envconf.npy"))
    return env, conf


def coverage_curve(split):
    return np.array([float((np.nan_to_num(np.load(f[:-4] + "_envconf.npy")) > 0.5).mean())
                     for f in frame_files(split)])


def pick_frames(split):
    """Evenly spaced temporal frames (same indices for both sequences) so the
    accumulation is shown over time; each is labelled with its actual coverage."""
    cov = coverage_curve(split)
    n = len(cov)
    idx = sorted({int(round(p * (n - 1))) for p in PROGRESS})
    return idx, cov


def reliance(env, conf, grey=0.5):
    """env where confident, fading to flat grey where unobserved (sRGB)."""
    s = linear_to_srgb(np.clip(env, 0, 1))
    c = conf[..., None]
    return np.clip(c * s + (1 - c) * grey, 0, 1)


def main():
    env2 = env2_linear()
    fig, axes = plt.subplots(4, 5, figsize=(16.5, 8.4),
                             gridspec_kw=dict(wspace=0.05, hspace=0.16))
    conf_im = None
    for si, split in enumerate(SPLITS):
        fis, cov = pick_frames(split)
        r_env, r_conf = 2 * si, 2 * si + 1
        env_last, conf_last = load(split, fis[-1])
        for ci, fi in enumerate(fis):
            env, conf = load(split, fi)
            axes[r_env, ci].imshow(u8(env))
            axes[r_env, ci].set_title(f"frame {fi}", fontsize=11)
            conf_im = axes[r_conf, ci].imshow(conf, cmap=CMAP, vmin=0, vmax=1)
            axes[r_conf, ci].text(0.5, -0.12, f"coverage {cov[fi]*100:.0f}%",
                                  transform=axes[r_conf, ci].transAxes, ha="center",
                                  va="top", fontsize=10)
        # reference column
        axes[r_env, 4].imshow(u8(env2)); axes[r_env, 4].set_title("GT env", fontsize=11)
        axes[r_conf, 4].imshow(reliance(env_last, conf_last))
        axes[r_conf, 4].set_title("env $\\times$ confidence\n(what we rely on)", fontsize=10)
        axes[r_env, 0].set_ylabel(f"{split}\nrecon env", fontsize=12)
        axes[r_conf, 0].set_ylabel(f"{split}\nconfidence", fontsize=12)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("0.8"); s.set_linewidth(0.6)
    # divider between the two reference panels and the evolution
    for r in range(4):
        axes[r, 4].set_facecolor("white")
    cbar = fig.colorbar(conf_im, ax=axes.ravel().tolist(), location="right",
                        shrink=0.45, pad=0.012, aspect=22)
    cbar.set_label("observation confidence", fontsize=11)
    fig.suptitle(f"Environment-map accumulation and observation confidence "
                 f"as the object rotates\n({SEQ},  r={REFL},  {DEPTH} depth)",
                 fontsize=14)
    fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT[:-4] + ".pdf", facecolor="white", bbox_inches="tight")
    print("wrote", OUT, "and .pdf")
    for split in SPLITS:
        fis, cov = pick_frames(split)
        print(f"  {split}: frames {fis}  cov {[round(float(cov[i]),2) for i in fis]}")


if __name__ == "__main__":
    main()
