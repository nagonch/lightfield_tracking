#!/usr/bin/env python3
"""Predicted (GroundingDINO+SAM2) object mask for frame 0 of a real LiFT
sequence, cached to eval/plots/segmasks/<seq>.png. Used by
plot_tracking_grid_lift.py to build the *displayed* pseudo-model silhouette
from a predicted mask instead of the ground-truth one.

Run one sequence per process (SAM2/Cutie hold Hydra global state):
    python eval/segmask_frame0.py shiny_box_tilt_prod
"""

import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from segmentor import Segmentor  # noqa: E402

DATASET_ROOT = "/home/ngoncharov/cvpr2026/datasets/LiFT_dataset"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", "segmasks")


def main(seq: str) -> None:
    seq_dir = os.path.join(DATASET_ROOT, seq)
    with open(os.path.join(seq_dir, "metadata.json")) as f:
        S, T = json.load(f)["n_views"]
    cv = (S // 2) * T + T // 2
    fid = sorted(d for d in os.listdir(seq_dir) if d.startswith("LF_"))[0][3:]
    prompt = (
        sys.argv[2]
        if len(sys.argv) > 2
        else open(os.path.join(seq_dir, "gdino_prompt.txt")).read().strip()
    )
    img = np.array(
        Image.open(os.path.join(seq_dir, f"LF_{fid}", f"{cv:04d}.png")).convert("RGB")
    ).astype(np.float32) / 255.0

    seg = Segmentor(prompt=prompt)
    mask = seg(torch.from_numpy(img).cuda()).bool().cpu().numpy()

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{seq}.png")
    Image.fromarray((mask * 255).astype(np.uint8)).save(out)
    print(f"{seq}: prompt={prompt!r}  mask px={int(mask.sum())}  → {out}")


if __name__ == "__main__":
    main(sys.argv[1])
