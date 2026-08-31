#!/usr/bin/env python3
"""Paper-style component ablation (Table 5 layout, no runtime) on the captured EPI
sequences, scored against the refined GT with the real mustard mesh.

Rows: Ours + segmentation | Ours | - photometric refinement | - reflection
separation | - light field depth (-> sensor). Inputs are the trajectories written
by main_lift.py into
    baselines_captured/ours_epi_final/lf/<seq>.npy        (Ours)
    ablation_epi/{segmentation,no_refine,no_separation,sensor_depth}/<depth>/<seq>.npy

Outputs eval/results_captured_refined/ablation_epi.{txt,tex}.

Run inside lift6dof:  docker exec -w "$PWD" lift6dof python eval/ablation_captured.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_eval import eval_sequence  # noqa: E402
from run_eval_captured import (  # noqa: E402
    RESULTS_ROOT, load_sequence_meta, rebase_to_gt, sample_model_points,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_captured_refined")
SEQS = ["epi_diffuse", "epi_reflective"]
ROWS = [  # (label, results dir, depth subdir, bold)
    ("Ours + segmentation", os.path.join(REPO, "ablation_epi", "segmentation"), "lf", False),
    ("Ours", os.path.join(RESULTS_ROOT, "ours_epi_final"), "lf", True),
    ("$-$ photometric refinement", os.path.join(REPO, "ablation_epi", "no_refine"), "lf", False),
    ("$-$ reflection separation", os.path.join(REPO, "ablation_epi", "no_separation"), "lf", False),
    ("$-$ light field depth ($\\rightarrow$ sensor)", os.path.join(REPO, "ablation_epi", "sensor_depth"), "gt", False),
]
KEYS = ["adds_auc", "add_auc", "ate_rmse", "mean_abs_rot_deg"]
HEAD = ["ADD-S ↑", "ADD ↑", "ATE [m] ↓", "ARE [°] ↓"]


def main():
    pts = sample_model_points()
    meta = {t: load_sequence_meta(t, "poses_object_refined") for t in SEQS}
    table = []
    for label, d, depth, bold in ROWS:
        per_seq = {}
        for t in SEQS:
            p = os.path.join(d, depth, f"{t}_prod.npy")
            if not os.path.exists(p):
                print(f"  missing {p}")
                continue
            K, _, gt, _, gt0 = meta[t]
            est = rebase_to_gt(np.load(p).astype(np.float64), gt0, gt[0])
            per_seq[t] = eval_sequence(est, gt, pts)
        if not per_seq:
            continue
        avg = {k: float(np.mean([m[k] for m in per_seq.values()])) for k in KEYS}
        table.append((label, bold, avg, per_seq))

    lines = ["== component ablation on the captured EPI sequences (refined GT), average over "
             "epi_diffuse + epi_reflective ==",
             f"{'Variant':44s} " + " ".join(f"{h:>10s}" for h in HEAD)]
    for label, bold, avg, _ in table:
        lines.append(f"{label.replace('$', '').replace(chr(92) + 'rightarrow', '->'):44s} "
                     f"{avg['adds_auc']:10.3f} {avg['add_auc']:10.3f} {avg['ate_rmse']:10.3f} {avg['mean_abs_rot_deg']:10.1f}")
    for t in SEQS:
        lines += ["", f"-- {t} --", f"{'Variant':44s} " + " ".join(f"{h:>10s}" for h in HEAD)]
        for label, bold, _, per_seq in table:
            if t in per_seq:
                m = per_seq[t]
                lines.append(f"{label.replace('$', '').replace(chr(92) + 'rightarrow', '->'):44s} "
                             f"{m['adds_auc']:10.3f} {m['add_auc']:10.3f} {m['ate_rmse']:10.3f} {m['mean_abs_rot_deg']:10.1f}")
    txt = "\n".join(lines)
    print(txt)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "ablation_epi.txt"), "w") as f:
        f.write(txt + "\n")
    tex = ["\\begin{tabular}{lcccc}", "\\toprule",
           "Variant & ADD-S $\\uparrow$ & ADD $\\uparrow$ & ATE\\,[m] $\\downarrow$ & ARE\\,[$^\\circ$] $\\downarrow$ \\\\", "\\midrule"]
    for label, bold, avg, _ in table:
        cells = [f"{avg['adds_auc']:.3f}", f"{avg['add_auc']:.3f}", f"{avg['ate_rmse']:.3f}", f"{avg['mean_abs_rot_deg']:.1f}"]
        if bold:
            label = f"\\textbf{{{label}}}"
            cells = [f"\\textbf{{{c}}}" for c in cells]
        tex.append(f"{label} & " + " & ".join(cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(OUT_DIR, "ablation_epi.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")
    print(f"-> {OUT_DIR}/ablation_epi.{{txt,tex}}")


if __name__ == "__main__":
    main()
