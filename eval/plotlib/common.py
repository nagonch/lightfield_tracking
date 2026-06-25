"""Shared style, palette, data loading and saving for the production plots.

All figure modules import from here so that styling stays consistent and new
plot types can be added as independent files (see ``build.py``).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Paths  (this file lives in eval/plotlib/, results + plots live in eval/)
# ---------------------------------------------------------------------------
EVAL_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = EVAL_DIR / "results"
OUTPUT_DIR = EVAL_DIR / "plots"

# ---------------------------------------------------------------------------
# Methods + colour-blind-safe palette (seaborn "deep")
# ---------------------------------------------------------------------------
FOLDER = {
    "PnP": "results_pnp",
    "ICP": "results_icp",
    "LoFTR": "results_loftr",
    "FP": "results_fp",
    "BundleSDF": "results_bsdf",
    "Ours": "ablation_full_gt_mask",
}
COLOR = {
    "FP": "#4C72B0",         # blue
    "BundleSDF": "#55A868",  # green
    "LoFTR": "#8172B3",      # purple
    "ICP": "#937860",        # brown
    "PnP": "#8C8C8C",        # grey
    "Ours": "#C44E52",       # red (emphasis)
}

REFL = [0.0, 0.5, 0.7, 1.0]
SEQS8 = [
    "bleach0", "bleach_hard_00_03_chaitanya",
    "cracker_box_reorient", "cracker_box_yalehand0",
    "mustard0", "mustard_easy_00_02",
    "sugar_box1", "sugar_box_yalehand0",
]


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def set_style():
    """Apply the matplotlib ``ggplot`` style (grey panel, white grid, sans-serif)."""
    plt.style.use("ggplot")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "legend.title_fontsize": 10,
    })


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
_cache = {}


def _per_sequence(folder):
    if folder not in _cache:
        with open(RESULTS_DIR / folder / "metrics.json") as f:
            _cache[folder] = json.load(f)["per_sequence"]
    return _cache[folder]


def series(label, geom, depth, metric):
    """Mean of `metric` over the common 8 sequences vs reflectivity (length-4)."""
    block = _per_sequence(FOLDER[label]).get(f"{geom}_{depth}", {})
    means = []
    for r in REFL:
        seqd = block.get(f"{geom}_{r}", {})
        vals = [seqd[s][metric] for s in SEQS8
                if s in seqd and np.isfinite(seqd[s][metric])]
        means.append(float(np.mean(vals)) if vals else np.nan)
    return np.array(means)


def long_frame(methods, geom, metric, depths):
    """Tidy DataFrame [Reflectivity, Method, Depth, value] for seaborn.

    `depths` is a list of (depth_key, depth_label) pairs.
    """
    rows = []
    for label in methods:
        for dkey, dlabel in depths:
            for r, y in zip(REFL, series(label, geom, dkey, metric)):
                rows.append({"Reflectivity": r, "Method": label,
                             "Depth": dlabel, "value": y})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def save_fig(fig, name):
    OUTPUT_DIR.mkdir(exist_ok=True)
    for suffix in (".pdf", ".png"):
        out = OUTPUT_DIR / f"{name}{suffix}"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved → {out}")
    plt.close(fig)
