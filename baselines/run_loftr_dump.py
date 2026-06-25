#!/usr/bin/env python3
"""Dump LoFTR-baseline per-frame poses to .npy (so they can be overlaid / re-eval'd).

The LoFTR tracker poses were never persisted — only its metrics survived — so the
qualitative overlays need them regenerated. This runs the existing LoftrBase
tracker (loftr_baseline.py) on a chosen set of sequences with *synth* depth and
saves to baselines/results_loftr/synth/<split>/<seq>.npy, matching the layout the
other baselines use.

GPU job (LoFTR runs on CUDA). Run inside the lift6dof container:
    python baselines/run_loftr_dump.py            # the 5 figure sequences
    python baselines/run_loftr_dump.py --all      # every cube_/objects_ sequence
"""

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)   # so `import icp`, `import pnp` resolve to baselines/
sys.path.insert(0, _ROOT)   # so `import loftr_baseline`, `import loftr_wrapper` resolve

from pnp import SpecTrackSequence  # noqa: E402  (rgb/depth/mask + GT pose loader)
import loftr_baseline  # noqa: E402
from loftr_baseline import LoftrBase  # noqa: E402

# Load the LoFTR model once and reuse it across every sequence (otherwise each
# LoftrBase() reloads the checkpoint onto the GPU).
_SHARED_RUNNER = loftr_baseline.LoftrRunner()
loftr_baseline.LoftrRunner = lambda *a, **k: _SHARED_RUNNER

DATASET_ROOT = "/home/ngoncharov/SpecTrack_dataset"
RESULTS_DIR = os.path.join(_HERE, "results_loftr")
DEPTH = "synth"

# the five sequences used in the qualitative tracking grids
TARGETS = [
    ("objects_1.0", "sugar_box1"),
    ("objects_1.0", "sugar_box_yalehand0"),
    ("objects_0.7", "sugar_box1"),
    ("objects_1.0", "bleach_hard_00_03_chaitanya"),
    ("cube_1.0", "sugar_box1"),
]


def all_targets():
    splits = sorted(
        d for d in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, d))
        and (d.startswith("cube_") or d.startswith("objects_"))
    )
    out = []
    for split in splits:
        sd = os.path.join(DATASET_ROOT, split)
        for seq in sorted(os.listdir(sd)):
            if os.path.isdir(os.path.join(sd, seq)) and seq != "models":
                out.append((split, seq))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="run every sequence (not just the 5)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    targets = all_targets() if args.all else TARGETS

    for split, seq in tqdm(targets, desc="LoFTR"):
        out_dir = os.path.join(RESULTS_DIR, DEPTH, split)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{seq}.npy")
        if os.path.exists(out_path) and not args.overwrite:
            tqdm.write(f"  {split}/{seq}: exists, skip")
            continue

        seq_dir = os.path.join(DATASET_ROOT, split, seq)
        s = SpecTrackSequence(seq_dir, depth_mode=DEPTH)
        tracker = LoftrBase(s)
        poses = tracker.run()                       # (N,4,4), rebased to GT frame 0
        np.save(out_path, poses)
        tqdm.write(f"  {split}/{seq}: {poses.shape} → {out_path}")


if __name__ == "__main__":
    main()
