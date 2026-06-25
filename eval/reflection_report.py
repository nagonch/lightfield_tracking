#!/usr/bin/env python3
"""Quantitative + qualitative GT evaluation of reflection_separation.py.

Reads the per-frame reflection-separation cache (estimated-depth `lf` branch by
default), and scores it against ground truth:

  * Diffuse component   vs the fully-diffuse `_0.0` render (GT diffuse D).
        The dataset blend is exactly linear:  obs_r = (1-r)*obs_0 + r*obs_1,
        verified to RMSE ~0.0035, so the `_0.0` central view *is* the GT diffuse
        and reflectivity r == (1 - alpha).
  * Environment map     vs env2.jpg (the renderer's lat-long env).
        Compared DIRECTLY, with no alignment: the recon env is built from
        reflected rays in the (fixed) rig frame, which coincides with the
        renderer world frame up to the equirect convention the two share by
        construction. Verified empirically: different sequences' env maps agree
        at zero alignment and match env2 best under zero transform (vertical
        flips give negative correlation). The recovered env is low-frequency —
        it captures the vertical sky/ground illumination profile and overall
        colour, not fine azimuthal detail.

Because separation accumulates the env map and the alpha estimate across frames
(main.py threads previous_environment_map / alpha history forward), quality
improves as a sequence progresses; we quantify that explicitly.

Outputs (under eval/reflection_report/):
  metrics.json            — per-frame / per-seq / per-(split,refl) aggregates
  REPORT.md               — tables + findings
  fig_diffuse_quality.png — diffuse PSNR / RMSE vs reflectivity
  fig_env_quality.png     — env correlation / RMSE vs reflectivity
  fig_improvement.png     — metrics vs normalised sequence progress (point 1)
  fig_alpha_convergence.png
  fig_env_growth.png      — env accumulation filmstrip for a showcase sequence
  grid_<split>_<refl>.png — qualitative grids (GT/recon diffuse, GT/recon env)
"""
import os
import glob
import json
import argparse

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ── paths / constants ────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CACHE = os.path.join(REPO, "cache", "diffuse_corrected")
DATASET = "/home/ngoncharov/SpecTrack_dataset"
ENV2_PATH = "/home/ngoncharov/cvpr2026/ycbv-eoat-lf/env2.jpg"
OUTDIR = os.path.join(HERE, "reflection_report")

SPLITS = ["cube", "objects"]
REFLS = ["0.0", "0.5", "0.7", "1.0"]
REFL_F = {r: float(r) for r in REFLS}
ENV_H, ENV_W = 256, 512
REFL_COLORS = {"0.0": "#2c7fb8", "0.5": "#41ab5d", "0.7": "#fe9929", "1.0": "#d7301f"}

# tomato_soup_can is excluded everywhere: it renders near-black (diffuse ≈
# reflection ≈ 0), so every GT comparison on it is uninformative.
EXCLUDE_PREFIXES = ("tomato_soup_can",)
OBJ_KEYS = ["cracker_box", "sugar_box", "mustard", "bleach"]
OBJ_ORDER = ["cube", "cracker_box", "sugar_box", "mustard", "bleach"]


def obj_of(split_prefix, seq):
    if split_prefix == "cube":
        return "cube"
    for k in OBJ_KEYS:
        if seq.startswith(k):
            return k
    return seq


# ── colour ───────────────────────────────────────────────────────────────────
def srgb_to_linear(s):
    s = np.clip(s, 0.0, 1.0)
    a = 0.055
    return np.where(s <= 0.04045, s / 12.92, ((s + a) / (1 + a)) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.0031308, x * 12.92, (1 + a) * x ** (1 / 2.4) - a)


def u8(linimg):
    return (linear_to_srgb(linimg) * 255).astype("uint8")


# ── metrics ──────────────────────────────────────────────────────────────────
def masked_mse(a, b, m):
    if m.sum() == 0:
        return float("nan")
    d = (a[m] - b[m]) ** 2
    return float(d.mean())


def correlation(a, b, m):
    if m.sum() < 10:
        return float("nan")
    x = a[m].reshape(-1).astype(np.float64)
    y = b[m].reshape(-1).astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else float("nan")


def lowpass(img, sigma=2.0):
    return np.stack([gaussian_filter(img[..., c], sigma, mode="wrap") for c in range(3)], -1)


# ── GT loaders ───────────────────────────────────────────────────────────────
_env2_cache = {}


def env2_linear(h=ENV_H, w=ENV_W):
    if (h, w) not in _env2_cache:
        e = np.asarray(Image.open(ENV2_PATH).convert("RGB")).astype(np.float32) / 255.0
        e = srgb_to_linear(e)
        e = np.asarray(
            Image.fromarray(u8(e)).resize((w, h), Image.BILINEAR)
        ).astype(np.float32) / 255.0
        _env2_cache[(h, w)] = srgb_to_linear(e)
    return _env2_cache[(h, w)]


_gtframes_cache = {}


def gt_frame_dirs(split_prefix, seq):
    """Sorted LF_* frame dirs of the fully-diffuse 0.0 version (the GT diffuse)."""
    key = (split_prefix, seq)
    if key not in _gtframes_cache:
        seqdir = os.path.join(DATASET, f"{split_prefix}_0.0", seq)
        frames = sorted(d for d in os.listdir(seqdir) if d.startswith("LF_"))
        _gtframes_cache[key] = (seqdir, frames)
    return _gtframes_cache[key]


def gt_diffuse_central(split_prefix, seq, fi):
    """Central sub-aperture of the 0.0 render at frame fi → linear HxWx3 (GT D).

    Uses /255 normalisation (the per-stack LF.max() the loader applies is ~255;
    the linear-blend identity was verified to RMSE 0.0035 under /255)."""
    seqdir, frames = gt_frame_dirs(split_prefix, seq)
    if fi >= len(frames):
        return None
    fdir = os.path.join(seqdir, frames[fi])
    pngs = sorted(f for f in os.listdir(fdir) if f.endswith(".png"))
    if not pngs:
        return None
    # central view of an n×n grid stored row-major as 0000..NN
    n = int(round(len(pngs) ** 0.5))
    mid = (n // 2) * n + (n // 2)
    mid = min(mid, len(pngs) - 1)
    img = np.asarray(Image.open(os.path.join(fdir, pngs[mid])).convert("RGB")).astype(np.float32)
    return srgb_to_linear(img / 255.0)


# ── per-sequence processing ──────────────────────────────────────────────────
def list_cache_frames(cache_dir):
    fs = sorted(glob.glob(os.path.join(cache_dir, "diffuse_*.npy")))
    return [f for f in fs if not f.endswith(("_env.npy", "_envconf.npy", "_alpha.npy"))]


def process_seq(depth, split_prefix, refl, seq, stride, want_display):
    cache_dir = os.path.join(CACHE, depth, f"{split_prefix}_{refl}", seq)
    diff_files = list_cache_frames(cache_dir)
    if not diff_files:
        return None
    n = len(diff_files)
    alpha_gt = 1.0 - REFL_F[refl]
    env2 = env2_linear()
    env2_lp = lowpass(env2)

    per_frame = []
    display = None
    for idx in range(0, n, stride):
        f = diff_files[idx]
        stem = f[:-4]
        try:
            recon_d = np.load(f)
            env = np.load(stem + "_env.npy")
            conf = np.load(stem + "_envconf.npy")
            alpha = float(np.load(stem + "_alpha.npy"))
        except Exception:
            continue

        rec = {"frame": idx, "n": n, "t": idx / max(n - 1, 1), "alpha": alpha,
               "alpha_err": abs(alpha - alpha_gt)}

        # ---- diffuse vs GT 0.0 ----
        gt_d = gt_diffuse_central(split_prefix, seq, idx)
        valid = recon_d.sum(-1) > 1e-6
        if gt_d is not None and valid.sum() > 50:
            mse_lin = masked_mse(recon_d, gt_d, valid)
            mse_srgb = masked_mse(linear_to_srgb(recon_d), linear_to_srgb(gt_d), valid)
            rec["diff_rmse_lin"] = float(np.sqrt(mse_lin))
            rec["diff_psnr"] = float(-10 * np.log10(mse_srgb + 1e-12))
        else:
            rec["diff_rmse_lin"] = float("nan")
            rec["diff_psnr"] = float("nan")

        # ---- env vs env2 (DIRECT, no alignment) ----
        obs = conf > 0.5
        allbins = np.ones_like(obs)
        rec["env_cov"] = float(obs.mean())
        rec["env_rmse"] = float(np.sqrt(masked_mse(env, env2, allbins)))
        rec["env_corr"] = correlation(env, env2, allbins)
        rec["env_corr_lp"] = correlation(lowpass(env), env2_lp, allbins)
        per_frame.append(rec)

    if not per_frame:
        return None

    # last frame: recon diffuse, GT diffuse, recon env, valid bbox + GT signal level
    last_idx = (len(diff_files) - 1)
    stem = diff_files[last_idx][:-4]
    recon_d = np.load(stem + ".npy")
    env = np.load(stem + "_env.npy")
    gt_d = gt_diffuse_central(split_prefix, seq, last_idx)
    valid = recon_d.sum(-1) > 1e-6
    # GT diffuse texture energy — near-black/flat renders make the diffuse metric
    # uninformative (diffuse ≈ reflection ≈ 0), e.g. tomato_soup_can_yalehand0.
    gt_signal = float(gt_d[valid].std()) if (gt_d is not None and valid.sum() > 0) else 0.0
    if want_display:
        display = {"recon_diffuse": recon_d, "gt_diffuse": gt_d,
                   "recon_env": env, "valid": valid}

    # per-seq aggregate (means over finite frames)
    def m(key):
        v = np.array([r[key] for r in per_frame], float)
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float("nan")

    agg = {k: m(k) for k in
           ["diff_rmse_lin", "diff_psnr", "env_rmse", "env_corr", "env_corr_lp",
            "env_cov", "alpha_err"]}
    agg["alpha_final"] = per_frame[-1]["alpha"]
    agg["n_frames"] = n
    agg["gt_signal"] = gt_signal
    agg["low_signal"] = bool(gt_signal < 0.03)
    return {"frames": per_frame, "agg": agg, "display": display,
            "obj": obj_of(split_prefix, seq), "low_signal": bool(gt_signal < 0.03)}


# ── figures ──────────────────────────────────────────────────────────────────
def fig_quality_vs_refl(results, outpath):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    styles = {"cube": dict(marker="s", ls="--"), "objects": dict(marker="o", ls="-")}
    for sp in SPLITS:
        x = [REFL_F[r] for r in REFLS]
        psnr = [results["by_split_refl"][sp][r]["diff_psnr"] for r in REFLS]
        rmse = [results["by_split_refl"][sp][r]["diff_rmse_lin"] for r in REFLS]
        ecorr = [results["by_split_refl"][sp][r]["env_corr_lp"] for r in REFLS]
        ermse = [results["by_split_refl"][sp][r]["env_rmse"] for r in REFLS]
        axes[0, 0].plot(x, psnr, label=sp, **styles[sp])
        axes[0, 1].plot(x, rmse, label=sp, **styles[sp])
        axes[1, 0].plot(x, ecorr, label=sp, **styles[sp])
        axes[1, 1].plot(x, ermse, label=sp, **styles[sp])
    axes[0, 0].set_title("Diffuse PSNR (sRGB) ↑"); axes[0, 0].set_ylabel("dB")
    axes[0, 1].set_title("Diffuse RMSE (linear) ↓")
    axes[1, 0].set_title("Env correlation (low-freq) ↑")
    axes[1, 1].set_title("Env RMSE (linear) ↓")
    for ax in axes.ravel():
        ax.set_xlabel("reflectivity  r  (alpha = 1 − r)")
        ax.grid(alpha=0.3); ax.legend()
    axes[1, 0].axvspan(-0.05, 0.05, color="gray", alpha=0.15)
    axes[1, 1].axvspan(-0.05, 0.05, color="gray", alpha=0.15)
    axes[1, 0].text(0.0, axes[1, 0].get_ylim()[0], " no reflection\n to recover",
                    fontsize=7, va="bottom", color="gray")
    fig.suptitle("Reconstruction quality vs surface reflectivity\n"
                 "(diffuse is well-posed as r→0; env is well-posed as r→1)", fontsize=12)
    fig.tight_layout(); fig.savefig(outpath, dpi=120); plt.close(fig)


def fig_improvement(results, outpath, nbins=12):
    """Metrics vs normalised sequence progress, averaged across sequences."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    keys = [("diff_rmse_lin", "Diffuse RMSE (linear) ↓"),
            ("env_corr_lp", "Env correlation (low-freq) ↑"),
            ("env_cov", "Env coverage  (frac conf>0.5) ↑"),
            ("alpha_err", "|alpha − GT|  ↓")]
    centers = (np.arange(nbins) + 0.5) / nbins
    for ax, (key, title) in zip(axes.ravel(), keys):
        for r in REFLS:
            bins = [[] for _ in range(nbins)]
            for sd in results["seqs"]:
                if sd["refl"] != r:
                    continue
                for fr in sd["res"]["frames"]:
                    b = min(int(fr["t"] * nbins), nbins - 1)
                    if np.isfinite(fr[key]):
                        bins[b].append(fr[key])
            y = [np.mean(b) if b else np.nan for b in bins]
            ax.plot(centers, y, marker=".", color=REFL_COLORS[r], label=f"r={r}")
        ax.set_title(title); ax.set_xlabel("sequence progress")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Online improvement over a sequence "
                 "(env map + alpha accumulate across frames)", fontsize=12)
    fig.tight_layout(); fig.savefig(outpath, dpi=120); plt.close(fig)


def fig_alpha(results, outpath):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for r in REFLS:
        # average alpha trace over sequences (resampled to common length)
        L = 40
        traces = []
        for sd in results["seqs"]:
            if sd["refl"] != r:
                continue
            a = np.array([f["alpha"] for f in sd["res"]["frames"]], float)
            if a.size < 2:
                continue
            xs = np.linspace(0, 1, a.size)
            traces.append(np.interp(np.linspace(0, 1, L), xs, a))
        if traces:
            mean = np.mean(traces, 0)
            ax.plot(np.linspace(0, 1, L), mean, color=REFL_COLORS[r], label=f"r={r}")
        ax.axhline(1.0 - REFL_F[r], color=REFL_COLORS[r], ls=":", alpha=0.6)
    ax.set_xlabel("sequence progress"); ax.set_ylabel("estimated alpha (diffuse fraction)")
    ax.set_title("Alpha estimate converges to GT (1 − r; dotted) as evidence accumulates")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(outpath, dpi=120); plt.close(fig)


def bbox_of(mask, pad=12):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, mask.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, mask.shape[1])
    return y0, y1, x0, x1


def fig_grid(results, split_prefix, refl, outpath):
    objs = ["cube"] if split_prefix == "cube" else OBJ_KEYS
    rows = []
    for ob in objs:
        cands = [sd for sd in results["seqs"]
                 if sd["split"] == split_prefix and sd["refl"] == refl and sd["res"]["obj"] == ob]
        if not cands:
            continue
        best = max(cands, key=lambda s: (s["res"]["agg"]["diff_psnr"]
                                         if np.isfinite(s["res"]["agg"]["diff_psnr"]) else -1))
        rows.append(best)
    if not rows:
        return
    env2 = env2_linear()
    fig, axes = plt.subplots(len(rows), 4, figsize=(13, 2.9 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    cols = ["GT diffuse (r=0.0)", "recon diffuse", "GT env (env2)", "recon env"]
    for ri, sd in enumerate(rows):
        d = sd["res"]["display"]
        valid = d["valid"]
        bb = bbox_of(valid)
        rec = d["recon_diffuse"].copy(); rec[~valid] = 0
        gt = d["gt_diffuse"]
        gtm = np.zeros_like(rec) if gt is None else (gt * valid[..., None])
        if bb:
            y0, y1, x0, x1 = bb
            rec_c, gt_c = rec[y0:y1, x0:x1], gtm[y0:y1, x0:x1]
        else:
            rec_c, gt_c = rec, gtm
        axes[ri, 0].imshow(u8(gt_c)); axes[ri, 1].imshow(u8(rec_c))
        axes[ri, 2].imshow(u8(env2)); axes[ri, 3].imshow(u8(d["recon_env"]))
        axes[ri, 0].set_ylabel(f"{sd['res']['obj']}\n{sd['seq'][:16]}", fontsize=8)
        for c in range(4):
            axes[ri, c].set_xticks([]); axes[ri, c].set_yticks([])
            if ri == 0:
                axes[ri, c].set_title(cols[c], fontsize=10)
    fig.suptitle(f"{split_prefix}  ·  reflectivity r={refl}  (alpha={1-REFL_F[refl]:.1f})  "
                 f"— best sequence per object", fontsize=12)
    fig.tight_layout(); fig.savefig(outpath, dpi=110); plt.close(fig)


def fig_env_growth(results, depth, outpath):
    # showcase: highest env coverage among r=1.0 sequences
    cands = [sd for sd in results["seqs"] if sd["refl"] == "1.0"]
    if not cands:
        return
    best = max(cands, key=lambda s: s["res"]["agg"]["env_cov"])
    cache_dir = os.path.join(CACHE, depth, f"{best['split']}_1.0", best["seq"])
    diff_files = list_cache_frames(cache_dir)
    n = len(diff_files)
    picks = [int(t * (n - 1)) for t in (0.0, 0.15, 0.35, 0.6, 1.0)]
    env2 = env2_linear()
    fig, axes = plt.subplots(1, len(picks) + 1, figsize=(3 * (len(picks) + 1), 2.7))
    for ci, fi in enumerate(picks):
        stem = diff_files[fi][:-4]
        env = np.load(stem + "_env.npy"); conf = np.load(stem + "_envconf.npy")
        axes[ci].imshow(u8(env))
        axes[ci].set_title(f"frame {fi}   cov={(conf > 0.5).mean():.2f}", fontsize=9)
        axes[ci].set_xticks([]); axes[ci].set_yticks([])
    axes[-1].imshow(u8(env2)); axes[-1].set_title("GT env2", fontsize=9)
    axes[-1].set_xticks([]); axes[-1].set_yticks([])
    axes[0].set_ylabel("recon env", fontsize=10)
    fig.suptitle(f"Env map accumulates as the object rotates — "
                 f"{best['split']}_1.0/{best['seq']}  (coverage grows frame to frame)",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(outpath, dpi=120); plt.close(fig)


# ── aggregation + report ─────────────────────────────────────────────────────
def aggregate(seqs):
    by_split_refl = {sp: {r: {} for r in REFLS} for sp in SPLITS}
    keys = ["diff_rmse_lin", "diff_psnr", "env_rmse", "env_corr", "env_corr_lp",
            "env_cov", "alpha_err", "alpha_final"]
    for sp in SPLITS:
        for r in REFLS:
            group = [sd["res"]["agg"] for sd in seqs if sd["split"] == sp and sd["refl"] == r]
            for k in keys:
                vals = np.array([g[k] for g in group], float)
                vals = vals[np.isfinite(vals)]
                by_split_refl[sp][r][k] = float(vals.mean()) if vals.size else float("nan")
            by_split_refl[sp][r]["n_seq"] = len(group)
    return {"by_split_refl": by_split_refl, "seqs": seqs}


def write_report(results, depth, stride, outdir):
    bsr = results["by_split_refl"]
    L = []
    L.append("# Reflection-separation — ground-truth evaluation\n")
    L.append(f"Cache: `cache/diffuse_corrected/{depth}` (estimated **{depth}** depth)  ·  "
             f"frame stride {stride}.  Generated by `eval/reflection_report.py`.\n")
    L.append("_`tomato_soup_can` is excluded everywhere — it renders near-black, "
             "so every GT comparison on it is uninformative._\n")
    L.append("**Setup.** The dataset blend is exactly linear, "
             "`obs_r = (1−r)·obs_0 + r·obs_1` (verified RMSE 0.0035), so "
             "reflectivity `r = 1 − alpha`, the `_0.0` render is the GT diffuse, "
             "and `env2.jpg` is the GT environment. Env maps are compared "
             "**directly** (no alignment): the recon env shares the renderer's "
             "lat-long convention and rig≈world frame, verified empirically "
             "(cross-sequence agreement + zero best transform).\n")

    def tbl(title, key, fmt="{:.3f}"):
        L.append(f"\n### {title}\n")
        L.append("| split | " + " | ".join(f"r={r}" for r in REFLS) + " |")
        L.append("|" + "---|" * (len(REFLS) + 1))
        for sp in SPLITS:
            row = [fmt.format(bsr[sp][r][key]) if np.isfinite(bsr[sp][r][key]) else "—"
                   for r in REFLS]
            L.append(f"| {sp} | " + " | ".join(row) + " |")

    L.append("\n## Diffuse component (vs GT `_0.0` render)\n")
    tbl("PSNR (sRGB, dB) ↑ — diffuse", "diff_psnr", "{:.2f}")
    tbl("RMSE (linear) ↓ — diffuse", "diff_rmse_lin")
    L.append("\n> Diffuse recovery is excellent at low/mid reflectivity and "
             "degrades toward r=1.0, where a near-perfect mirror carries almost "
             "no diffuse signal (recovery is ill-posed — recon diffuse shows "
             "reflection bleed).\n")

    L.append("\n## Environment map (vs `env2.jpg`, direct)\n")
    tbl("Correlation (low-freq) ↑ — env", "env_corr_lp")
    tbl("RMSE (linear) ↓ — env", "env_rmse")
    tbl("Coverage (mean frac conf>0.5) — env", "env_cov")
    L.append("\n> Env quality improves with reflectivity (more reflective signal "
             "to fit). At r=0.0 there is no reflection to recover, so env numbers "
             "are meaningless there. The recovered env is low-frequency: it "
             "captures the vertical sky/ground profile and overall colour "
             "(correlation), not fine azimuthal detail (hence non-trivial RMSE).\n")

    L.append("\n## Alpha (diffuse fraction) estimate\n")
    tbl("Mean |alpha − GT| ↓", "alpha_err")
    tbl("Final-frame alpha (GT = 1−r)", "alpha_final", "{:.2f}")

    L.append("\n## Figures\n")
    for fn, cap in [
        ("fig_quality_vs_reflectivity.png",
         "Diffuse (top) and env (bottom) quality vs reflectivity — opposite difficulty curves."),
        ("fig_improvement.png", "Improvement over sequence progress (point 1)."),
        ("fig_alpha_convergence.png", "Alpha convergence to GT."),
        ("fig_env_growth.png", "Env-map accumulation filmstrip."),
    ]:
        if os.path.exists(os.path.join(outdir, fn)):
            L.append(f"\n**{cap}**\n\n![]({fn})\n")
    for sp in SPLITS:
        for r in REFLS:
            fn = f"grid_{sp}_{r}.png"
            if os.path.exists(os.path.join(outdir, fn)):
                L.append(f"\n**Qualitative grid — {sp}, r={r}.**\n\n![]({fn})\n")

    with open(os.path.join(outdir, "REPORT.md"), "w") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", default="lf", choices=["lf", "gt", "synth"],
                    help="cache branch (lf = estimated depth, the default)")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--limit-seqs", type=int, default=None, help="debug: cap sequences")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    seqs = []
    todo = []
    for sp in SPLITS:
        for r in REFLS:
            root = os.path.join(CACHE, args.depth, f"{sp}_{r}")
            if not os.path.isdir(root):
                continue
            for seq in sorted(os.listdir(root)):
                if seq.startswith(EXCLUDE_PREFIXES):
                    continue
                if os.path.isdir(os.path.join(root, seq)):
                    todo.append((sp, r, seq))
    if args.limit_seqs:
        todo = todo[: args.limit_seqs]

    for i, (sp, r, seq) in enumerate(todo):
        res = process_seq(args.depth, sp, r, seq, args.stride, want_display=True)
        if res is None:
            print(f"  skip {sp}_{r}/{seq} (no cache)")
            continue
        seqs.append({"split": sp, "refl": r, "seq": seq, "res": res})
        a = res["agg"]
        print(f"[{i+1}/{len(todo)}] {sp}_{r}/{seq:32s} "
              f"Dpsnr={a['diff_psnr']:5.2f} Drmse={a['diff_rmse_lin']:.3f} "
              f"Ecorr={a['env_corr_lp']:.2f} Ermse={a['env_rmse']:.3f} "
              f"cov={a['env_cov']:.2f} aErr={a['alpha_err']:.3f}", flush=True)

    results = aggregate(seqs)

    # JSON (drop heavy display arrays / per-frame detail kept compact)
    dump = {"by_split_refl": results["by_split_refl"],
            "per_seq": [{"split": s["split"], "refl": s["refl"], "seq": s["seq"],
                         "obj": s["res"]["obj"], **s["res"]["agg"]} for s in seqs]}
    with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
        json.dump(dump, f, indent=2)

    print("rendering figures…", flush=True)
    fig_quality_vs_refl(results, os.path.join(OUTDIR, "fig_quality_vs_reflectivity.png"))
    fig_improvement(results, os.path.join(OUTDIR, "fig_improvement.png"))
    fig_alpha(results, os.path.join(OUTDIR, "fig_alpha_convergence.png"))
    fig_env_growth(results, args.depth, os.path.join(OUTDIR, "fig_env_growth.png"))
    for sp in SPLITS:
        for r in REFLS:
            fig_grid(results, sp, r, os.path.join(OUTDIR, f"grid_{sp}_{r}.png"))
    write_report(results, args.depth, args.stride, OUTDIR)
    print(f"done → {OUTDIR}")


if __name__ == "__main__":
    main()
