"""RMSE and missing-pixel fraction: depth_synth vs depth_lf vs GT depth.

Two panels (Objects / Cube), each showing RMSE and missing-% vs reflectivity
for both depth sources.  Mirrors the style of reflectivity.py (seaborn lineplot,
shared legend, ggplot theme via set_style()).
"""

import os

import numpy as np
import seaborn as sns
import pandas as pd

from .common import REFL, SEQS8, OUTPUT_DIR, plt, save_fig

# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------
DATASET_ROOT = "/home/ngoncharov/SpecTrack_dataset"
CENTRAL_VIEW = 12
MAX_SYNTH_DEPTH = 2.0   # ray-escaped reflection artefacts → treat as missing

GEOMS   = [("objects", "Objects"), ("cube", "Cube")]
SOURCES = [("depth_synth", "Synth depth"), ("depth_lf", "LF depth")]

SOURCE_COLOR = {
    "Synth depth": "#4C72B0",  # blue
    "LF depth":    "#C44E52",  # red
}
SOURCE_DASH  = {"Synth depth": "",        "LF depth": (4, 1.5)}
SOURCE_MARK  = {"Synth depth": "o",       "LF depth": "s"}


# ---------------------------------------------------------------------------
# Per-frame metric
# ---------------------------------------------------------------------------
def _frame_metrics(seq_dir: str, frame_id: str, src_dir: str):
    gt_path   = os.path.join(seq_dir, "depth",   f"{frame_id}.png")
    src_path  = os.path.join(seq_dir, src_dir,   f"{frame_id}.png")
    mask_path = os.path.join(seq_dir, f"LF_{frame_id}", "masks",
                             f"{CENTRAL_VIEW:04d}.png")

    if not (os.path.exists(gt_path) and os.path.exists(src_path)
            and os.path.exists(mask_path)):
        return np.nan, np.nan

    from PIL import Image
    gt   = np.array(Image.open(gt_path),   dtype=np.float32) / 1000.0
    src  = np.array(Image.open(src_path),  dtype=np.float32) / 1000.0
    mask = np.array(Image.open(mask_path), dtype=np.uint8)

    obj   = mask > 127
    if not obj.any():
        return np.nan, np.nan

    valid = obj & (src > 0) & (src <= MAX_SYNTH_DEPTH)
    rmse  = (float(np.sqrt(((gt[valid] - src[valid]) ** 2).mean()))
             if valid.any() else np.nan)
    miss  = float((obj & ~valid).sum() / obj.sum())
    return rmse, miss


# ---------------------------------------------------------------------------
# Sequence-level stats (mean over frames)
# ---------------------------------------------------------------------------
def _seq_stats(geom: str, r: float, seq: str, src_dir: str):
    seq_dir = os.path.join(DATASET_ROOT, f"{geom}_{r}", seq)
    gt_frames  = {os.path.splitext(f)[0]
                  for f in os.listdir(os.path.join(seq_dir, "depth"))
                  if f.endswith(".png")}
    src_frames = {os.path.splitext(f)[0]
                  for f in os.listdir(os.path.join(seq_dir, src_dir))
                  if f.endswith(".png")}
    frames = sorted(gt_frames & src_frames)
    if not frames:
        return np.nan, np.nan
    rmses, misses = [], []
    for fid in frames:
        rm, ms = _frame_metrics(seq_dir, fid, src_dir)
        if np.isfinite(rm):
            rmses.append(rm)
        if np.isfinite(ms):
            misses.append(ms)
    return (float(np.mean(rmses)) if rmses else np.nan,
            float(np.mean(misses)) if misses else np.nan)


# ---------------------------------------------------------------------------
# Build tidy DataFrame for seaborn
# ---------------------------------------------------------------------------
def _build_df(geom: str):
    rows = []
    for r in REFL:
        for src_dir, src_label in SOURCES:
            rmse_vals, miss_vals = [], []
            for seq in SEQS8:
                seq_path = os.path.join(DATASET_ROOT, f"{geom}_{r}", seq)
                if not os.path.isdir(seq_path):
                    continue
                rm, ms = _seq_stats(geom, r, seq, src_dir)
                if np.isfinite(rm):
                    rmse_vals.append(rm)
                if np.isfinite(ms):
                    miss_vals.append(ms)
            rows.append({
                "Reflectivity": r,
                "Source": src_label,
                "RMSE": float(np.mean(rmse_vals)) if rmse_vals else np.nan,
                "Missing": float(np.mean(miss_vals)) * 100 if miss_vals else np.nan,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def depth_sources_figure(out_name: str = "depth_sources"):
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0), sharex=True)

    for row, (geom, geom_title) in enumerate(GEOMS):
        df = _build_df(geom)

        # --- RMSE panel ---
        ax_rmse = axes[row, 0]
        sns.lineplot(
            data=df, x="Reflectivity", y="RMSE",
            hue="Source", hue_order=[s for _, s in SOURCES],
            palette=SOURCE_COLOR,
            style="Source", dashes=SOURCE_DASH, markers=SOURCE_MARK,
            markersize=6.5, linewidth=2.0, errorbar=None,
            ax=ax_rmse, legend=(row == 0),
        )
        ax_rmse.set_yscale("log")
        ax_rmse.set_ylabel("RMSE [m]  ($\\downarrow$)")
        ax_rmse.set_xlabel("Reflectivity" if row == 1 else "")
        if row == 0:
            ax_rmse.set_title("Depth RMSE", pad=8)

        # --- Missing % panel ---
        ax_miss = axes[row, 1]
        sns.lineplot(
            data=df, x="Reflectivity", y="Missing",
            hue="Source", hue_order=[s for _, s in SOURCES],
            palette=SOURCE_COLOR,
            style="Source", dashes=SOURCE_DASH, markers=SOURCE_MARK,
            markersize=6.5, linewidth=2.0, errorbar=None,
            ax=ax_miss, legend=False,
        )
        ax_miss.set_ylabel("Missing pixels [%]  ($\\downarrow$)")
        ax_miss.set_xlabel("Reflectivity" if row == 1 else "")
        if row == 0:
            ax_miss.set_title("Missing pixels", pad=8)

        # Row label
        axes[row, 0].annotate(
            geom_title, xy=(-0.28, 0.5), xycoords="axes fraction", rotation=90,
            ha="center", va="center", fontsize=12.5, fontweight="bold",
            color="#333333",
        )

    for ax in axes.flat:
        ax.set_xticks(REFL)
        ax.set_xlim(-0.04, 1.04)
        ax.margins(y=0.08)

    # Shared legend (pull from RMSE panel row 0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if axes[0, 0].get_legend():
        axes[0, 0].get_legend().remove()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=len(SOURCES),
        frameon=True, framealpha=0.95, edgecolor="#cccccc",
        bbox_to_anchor=(0.5, -0.02), handlelength=2.4, columnspacing=1.8,
    )
    fig.suptitle("Depth quality: synth sensor vs LF plane-sweep",
                 fontsize=14.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0.04, 0.06, 1.0, 0.96))
    save_fig(fig, out_name)
