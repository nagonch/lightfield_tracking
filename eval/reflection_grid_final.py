#!/usr/bin/env python3
"""Production qualitative grid for reflection separation (r=0.7).

One CVPR-style figure. Rows = one best cube sequence + one best sequence per YCB
object type (cracker_box, sugar_box, mustard, bleach). Columns:

  Input | GT diffuse | Ours diffuse (GT depth) | Ours diffuse (est. depth)
        ‖ GT env | Ours env (GT depth) | Ours env (est. depth)

Object panels are square crops (uniform aspect, no weird crops); env panels are
equirectangular. The diffuse columns share one frame (healthy in BOTH depth
caches, post alpha-stabilization, highest PSNR). The env is an *accumulated*
quantity, so each env column uses its own depth's best-covered healthy frame
(decoupled — lets a sequence whose GT cache collapses early still show a full LF
env, and vice-versa). Sequences are pinned to the best est-depth trackers. PNG+PDF.
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
THRESHOLD_ENV = True   # show recon env only where observed (confidence>thr); blank elsewhere
ENV_CONF_THR = 0.9
ALPHA_STABLE_TOL = 0.01  # alpha "stabilized" once consecutive estimates change less than this
MIN_STABLE_PROGRESS = 0.45  # also wait at least this far into the sequence (≈ frame 20 of ~40)

FIXED_FRAME = 20       # None → best-frame selection; int → every panel at this frame (clamped)

REFL = "0.7"
_suffix = f"_frame{FIXED_FRAME}" if FIXED_FRAME is not None else ("_envthresh" if THRESHOLD_ENV else "")
OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "reflection_report", f"grid_final_r{REFL}{_suffix}.png")
PROGRESS_PROBES = [0.05, 0.12, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.97]
MIN_VALID = 1500

# (row label, split prefix, sequence). Best-est sequences (LF-depth tracking, from
# eval/results/ablation_full_gt_mask/summary.txt, ranked by ADD-S AUC + rotation).
GROUPS_BEST = [
    ("cube", "cube", "cracker_box_reorient"),            # cube LF: best Rot 2.6°
    ("cracker box", "objects", "cracker_box_reorient"),  # 0.898 / 8.6° vs yalehand0 0.863 / 11.5°
    ("sugar box", "objects", "sugar_box_yalehand0"),     # 7.6° vs sugar_box1 19.5°
    ("mustard", "objects", "mustard_easy_00_02"),        # 0.919 / 5.5° vs mustard0 14.0°
    ("bleach", "objects", "bleach_hard_00_03_chaitanya"),  # 7.2° vs bleach0 19.0°
]
# Sequences whose GT *and* LF reflection caches both survive to FIXED_FRAME (=20):
# reorient is only 20 frames and sugar_box_yalehand0's GT cache dies at frame 10.
GROUPS_FRAME20 = [
    ("cube", "cube", "cracker_box_yalehand0"),
    ("cracker box", "objects", "cracker_box_yalehand0"),
    ("sugar box", "objects", "sugar_box1"),
    ("mustard", "objects", "mustard_easy_00_02"),
    ("bleach", "objects", "bleach_hard_00_03_chaitanya"),
]
GROUPS = GROUPS_FRAME20 if FIXED_FRAME is not None else GROUPS_BEST


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


def load_conf(depth, split, seq, fi):
    return np.nan_to_num(np.load(cache_frames(depth, split, seq)[fi][:-4] + "_envconf.npy"))


def stable_floor(split, seq):
    """First frame at/after which the alpha estimate has stabilized (two
    consecutive changes < tol), but at least MIN_STABLE_PROGRESS into the
    sequence — so we read the env/diffuse once the separation has settled."""
    fs = cache_frames("gt", split, seq)
    a = np.array([float(np.load(f[:-4] + "_alpha.npy")) if os.path.exists(f[:-4] + "_alpha.npy")
                  else np.nan for f in fs])
    fstab = len(a) // 3
    for i in range(2, len(a)):
        if abs(a[i] - a[i - 1]) < ALPHA_STABLE_TOL and abs(a[i - 1] - a[i - 2]) < ALPHA_STABLE_TOL:
            fstab = i
            break
    return max(fstab, int(MIN_STABLE_PROGRESS * (len(a) - 1)))


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
    floor = min(stable_floor(split, seq), n - 1)
    rows = []  # (fi, gt_ok, lf_ok, gt_cov, lf_cov, psnr)
    for p in PROGRESS_PROBES:
        fi = min(int(p * (n - 1)), n - 1)
        Dg = load_diffuse("gt", split, seq, fi)
        vg = Dg.sum(-1) > 1e-6
        gt_ok = vg.sum() >= MIN_VALID
        lf_ok = (load_diffuse("lf", split, seq, fi).sum(-1) > 1e-6).sum() >= MIN_VALID
        psnr = 0.0
        if gt_ok:
            gt = central_linear(split, "0.0", seq, fi)
            mse = ((u8(Dg).astype(np.float32) - u8(gt).astype(np.float32))[vg] ** 2).mean()
            psnr = float(-10 * np.log10(mse / 255.0 ** 2 + 1e-12))
        rows.append((fi, gt_ok, lf_ok, coverage_at("gt", split, seq, fi),
                     coverage_at("lf", split, seq, fi), psnr))

    def best(pool, key, default=0):
        late = [r for r in pool if r[0] >= floor] or pool
        return max(late, key=key)[0] if late else default

    # diffuse: a frame healthy in BOTH depths (shared pose), highest PSNR vs GT
    diff_fi = best([r for r in rows if r[1] and r[2]], lambda r: r[5])
    # env is accumulated → pick each depth's own best-covered healthy frame
    env_gt_fi = best([r for r in rows if r[1]], lambda r: r[3])
    env_lf_fi = best([r for r in rows if r[2]], lambda r: r[4])
    healthy_both = [r for r in rows if r[1] and r[2]] or rows
    max_psnr = max((r[5] for r in healthy_both), default=0.0)
    max_cov = max((r[3] for r in rows if r[1]), default=0.0)
    return dict(seq=seq, diff_fi=diff_fi, env_gt_fi=env_gt_fi, env_lf_fi=env_lf_fi,
                psnr=max_psnr, cov=max_cov, score=max_psnr / 20.0 + max_cov)


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


def build_row(label, split, seq):
    best = evaluate_seq(split, seq)
    if best is None:
        return None
    dfi, egt, elf = best["diff_fi"], best["env_gt_fi"], best["env_lf_fi"]
    if FIXED_FRAME is not None:
        n = min(len(cache_frames("gt", split, seq)), len(cache_frames("lf", split, seq)))
        dfi = egt = elf = min(FIXED_FRAME, n - 1)
        gv = int((load_diffuse("gt", split, seq, dfi).sum(-1) > 1e-6).sum())
        lv = int((load_diffuse("lf", split, seq, dfi).sum(-1) > 1e-6).sum())
        print(f"  [{label}] frame {dfi}/{n-1}  gt_valid={gv} lf_valid={lv}"
              f"{'  <-- GT COLLAPSED' if gv < MIN_VALID else ''}", flush=True)
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
    def env(depth, fi):
        e = load_env(depth, split, seq, fi)
        if ENV_RECON_BLUR > 0:
            e = np.stack([gaussian_filter(e[..., c], ENV_RECON_BLUR, mode="wrap")
                          for c in range(3)], -1)
        if THRESHOLD_ENV:
            # show env only where observed; blank (linear white) elsewhere
            out = np.ones_like(e)
            m = load_conf(depth, split, seq, fi) > ENV_CONF_THR
            out[m] = np.clip(e, 0, 1)[m]
            e = out
        return e
    print(f"  {label:12s} -> {seq:28s} diffuse f{dfi:<3d} env_gt f{egt:<3d} env_lf f{elf:<3d} "
          f"psnr={best['psnr']:.1f}", flush=True)
    return dict(label=label, seq=seq,
                panels=[inp, gtd, rec_g, rec_l],
                envs=[env2_linear(), env("gt", egt), env("lf", elf)])


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
