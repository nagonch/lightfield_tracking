#!/usr/bin/env python3
"""
Consolidate error_over_time.txt curves from multiple eval/run_eval.py result
folders into one joint comparison plot.

Usage
-----
    python3 eval/plot_joint_error_over_time.py [names...] [--results-dir PATH] [--output PATH]

Each <name> is a subfolder of --results-dir (default: eval/results) that
contains an error_over_time.txt, i.e. the output of a prior run_eval.py call.
If no names are given, every such subfolder is used.

Output
------
    One PNG with two stacked plots (rotation error, translation error), each
    holding one curve per selected result folder.
"""

import argparse
import itertools
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(EVAL_DIR, "results")
DEFAULT_OUTPUT = os.path.join(DEFAULT_RESULTS_DIR, "joint_error_over_time.png")

# Fixed categorical hue order (dataviz palette), cycled with linestyles once
# more curves than hues are requested.
_HUES = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
_LINESTYLES = ["-", "--", "-.", ":"]


def discover_result_names(results_dir: str) -> list:
    names = []
    for name in sorted(os.listdir(results_dir)):
        if os.path.isfile(os.path.join(results_dir, name, "error_over_time.txt")):
            names.append(name)
    return names


def load_curve(results_dir: str, name: str):
    txt_path = os.path.join(results_dir, name, "error_over_time.txt")
    data = np.loadtxt(txt_path)  # (n_bins, 3): progress_pct, rot_deg, trans_m
    return data[:, 0], data[:, 1], data[:, 2]


def build_joint_plot(curves: list, out_path: str):
    """curves: list of (name, progress_pct, rot_err, trans_err)."""
    style_cycle = itertools.cycle(
        (color, ls) for ls in _LINESTYLES for color in _HUES
    )

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": "#0b0b0b",
            "text.color": "#0b0b0b",
            "xtick.color": "#52514e",
            "ytick.color": "#52514e",
        }
    )
    fig, (ax_rot, ax_trans) = plt.subplots(
        2, 1, figsize=(7, 7.5), sharex=True, facecolor="#fcfcfb"
    )

    for (name, x, rot, trans), (color, ls) in zip(curves, style_cycle):
        ax_rot.plot(x, rot, color=color, linestyle=ls, linewidth=2, label=name)
        ax_trans.plot(x, trans, color=color, linestyle=ls, linewidth=2, label=name)

    for ax, ylabel, title in (
        (ax_rot, "Rotation error (°)", "Relative rotation error over time"),
        (ax_trans, "Translation error (m)", "Relative translation error over time"),
    ):
        ax.set_facecolor("#fcfcfb")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11, loc="left", color="#0b0b0b")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.8)
        ax.set_ylim(bottom=0)

    ax_trans.set_xlabel("Trajectory progress (%)")
    ax_trans.set_xlim(0, 100)
    ax_trans.xaxis.set_major_formatter(lambda x, _pos: f"{x:.0f}%")

    ax_rot.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot error_over_time.txt curves from multiple result "
        "folders on one joint comparison plot."
    )
    parser.add_argument(
        "names",
        nargs="*",
        help="Subfolder names under --results-dir to include "
        "(default: all folders with an error_over_time.txt).",
    )
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    names = args.names or discover_result_names(args.results_dir)
    if not names:
        print(f"No error_over_time.txt found under {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    curves = []
    for name in names:
        txt_path = os.path.join(args.results_dir, name, "error_over_time.txt")
        if not os.path.isfile(txt_path):
            print(f"  SKIP {name}: no error_over_time.txt at {txt_path}", file=sys.stderr)
            continue
        x, rot, trans = load_curve(args.results_dir, name)
        curves.append((name, x, rot, trans))

    if not curves:
        print("Nothing to plot.", file=sys.stderr)
        sys.exit(1)

    build_joint_plot(curves, args.output)
    print(f"Plotted {len(curves)} result(s): {', '.join(n for n, *_ in curves)}")
    print(f"Joint plot → {args.output}")


if __name__ == "__main__":
    main()
