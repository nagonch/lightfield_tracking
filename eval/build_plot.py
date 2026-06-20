#!/usr/bin/env python3
"""Build reflectivity vs. metric plots from evaluation results.

Usage (inside container via bash run_container.sh):
    python eval/build_plot.py
Output: eval/plots/reflectivity_plots.{pdf,png}
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR = SCRIPT_DIR / "plots"

# Ordered list of (folder_name, display_label)
BASELINES = [
    ("results_fp", "FP"),
    ("results_pnp", "PnP"),
    ("results_bsdf", "BSDF"),
    ("results_icp", "ICP"),
    ("results_loftr", "LoFTR"),
    ("ablation_full_gt_mask", "Ours"),
]

COLORS = ["#1f77b4", "#9467bd", "#2ca02c", "#d62728", "#ff7f0e", "#8c564b"]

REFLECTIVITIES = [0.0, 0.5, 0.7, 1.0]

METRICS = [
    ("add_auc", "ADD AUC ↑"),
    ("ate_rmse", "ATE RMSE [m] ↓"),
    ("mean_abs_rot_deg", "Rot. Error [°] ↓"),
]

# (variant_key, linestyle, linewidth, alpha, marker, fill_marker)
# Synth variants are primary (solid, full opacity); GT is supplementary (dashed, dim).
VARIANTS = [
    ("cube_synth", "solid", 0.9, 1.00, "o", True),
    ("objects_synth", "solid", 0.9, 1.00, "s", True),
    ("cube_gt", "dashed", 0.5, 0.40, "o", False),
    ("objects_gt", "dashed", 0.5, 0.40, "s", False),
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data() -> dict:
    """Return {display_label: split_averages_dict}."""
    data = {}
    for folder, label in BASELINES:
        fpath = RESULTS_DIR / folder / "metrics.json"
        if not fpath.exists():
            print(f"WARNING: {fpath} not found – skipping.")
            continue
        with open(fpath) as f:
            metrics = json.load(f)
        data[label] = metrics["split_averages"]
    return data


def get_series(split_averages: dict, variant: str, metric_key: str):
    """Return (x_reflectivities, y_values) for the given variant and metric."""
    variant_data = split_averages.get(variant, {})
    prefix = "cube" if variant.startswith("cube") else "objects"
    xs, ys = [], []
    for r in REFLECTIVITIES:
        key = f"{prefix}_{r}"
        if key in variant_data:
            xs.append(r)
            ys.append(variant_data[key][metric_key])
    return xs, ys


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = load_data()
    if not data:
        print("No data found – check RESULTS_DIR.")
        return

    baseline_labels = [label for _, label in BASELINES if label in data]
    color_map = {
        label: COLORS[i % len(COLORS)] for i, label in enumerate(baseline_labels)
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.subplots_adjust(wspace=0.35)

    for ax, (metric_key, metric_label) in zip(axes, METRICS):
        for label in baseline_labels:
            color = color_map[label]
            for var_key, ls, lw, alpha, marker, fill in VARIANTS:
                xs, ys = get_series(data[label], var_key, metric_key)
                if not xs:
                    continue
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    linestyle=ls,
                    linewidth=lw,
                    alpha=alpha,
                    marker=marker,
                    markersize=5 if fill else 4,
                    markerfacecolor=color if fill else "none",
                    markeredgecolor=color,
                    markeredgewidth=1.2,
                    zorder=3 if fill else 2,
                )

        if metric_key == "ate_rmse":
            ax.set_yscale("log")
        ax.set_xlabel("Reflectivity", fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(metric_label, fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks(REFLECTIVITIES)
        ax.set_xlim(-0.05, 1.05)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ---- Compact legend below all subplots ----
    # Section 1: one color patch per baseline
    legend_handles = [
        mpatches.Patch(color=color_map[label], label=label) for label in baseline_labels
    ]
    # Separator (invisible) to visually divide the two groups
    legend_handles.append(Line2D([], [], color="none", label=" "))
    # Section 2: four style entries (color-agnostic, gray)
    legend_handles += [
        Line2D(
            [0],
            [0],
            color="#444",
            linestyle="solid",
            linewidth=1.8,
            marker="o",
            markersize=6.5,
            label="cube – synth",
        ),
        Line2D(
            [0],
            [0],
            color="#444",
            linestyle="solid",
            linewidth=1.8,
            marker="s",
            markersize=6.5,
            label="obj – synth",
        ),
        Line2D(
            [0],
            [0],
            color="#888",
            linestyle="dashed",
            linewidth=1.0,
            alpha=0.7,
            marker="o",
            markersize=5,
            markerfacecolor="none",
            markeredgecolor="#888",
            label="cube – GT",
        ),
        Line2D(
            [0],
            [0],
            color="#888",
            linestyle="dashed",
            linewidth=1.0,
            alpha=0.7,
            marker="s",
            markersize=5,
            markerfacecolor="none",
            markeredgecolor="#888",
            label="obj – GT",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        fontsize=10,
        frameon=True,
        framealpha=0.9,
        edgecolor="#ccc",
        bbox_to_anchor=(0.5, -0.07),
    )

    fig.suptitle("Metric vs. Reflectivity", fontsize=14, fontweight="bold", y=1.02)

    for suffix in (".pdf", ".png"):
        out = OUTPUT_DIR / f"reflectivity_plots{suffix}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")


if __name__ == "__main__":
    main()
