"""Segment the object in every frame of the captured sequences (RealSense + EPI).

Uses the repo Segmentor (GroundingDINO box on frame 0 -> SAM2 mask -> Cutie temporal
tracking) on the RGB frames — for the EPI cross capture, on the center view
(images/XXXXXX/cam_4_4.png) only.  Writes per-frame binary masks to
<seq>/masks/XXXXXX.png (uint8, 0/255) and a side-by-side check image
[ rgb | rgb*mask ] to <seq>/masks_vis/XXXXXX.jpg.

Run inside the lift6dof container:
    docker exec -e CUDA_VISIBLE_DEVICES=0 -w "$PWD" lift6dof python segment_realsense_capture.py
Optionally pass dataset roots to (re)run a subset:
    ... python segment_realsense_capture.py epi
"""

import os
import sys

import numpy as np
import torch
from PIL import Image

from segmentor import Segmentor

CAPTURED = "/home/ngoncharov/SpecTrack_dataset/captured"
PROMPTS = {
    "diffuse": "yellow mustard bottle",
    "reflective": "object wrapped in aluminum foil",
}


def frame_image_path(seq_dir: str, entry: str) -> str:
    """RealSense: images/XXXXXX.png; EPI cross: images/XXXXXX/cam_4_4.png."""
    p = os.path.join(seq_dir, "images", entry)
    return os.path.join(p, "cam_4_4.png") if os.path.isdir(p) else p


def run_sequence(seq_dir: str, prompt: str):
    entries = sorted(os.listdir(os.path.join(seq_dir, "images")))
    mask_dir = os.path.join(seq_dir, "masks")
    vis_dir = os.path.join(seq_dir, "masks_vis")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    segmentor = Segmentor(prompt=prompt)
    print(f"[{seq_dir}] prompt='{prompt}' frames={len(entries)}")

    for e in entries:
        stem = e.replace(".png", "")
        rgb = np.asarray(Image.open(frame_image_path(seq_dir, e)))
        t = torch.from_numpy(rgb).cuda().float() / 255.0
        mask = (segmentor(t) == 1).cpu().numpy()
        Image.fromarray((mask * 255).astype(np.uint8)).save(
            os.path.join(mask_dir, f"{stem}.png"))

        vis = np.concatenate([rgb, rgb * mask[..., None]], axis=1)
        Image.fromarray(vis[::2, ::2]).save(
            os.path.join(vis_dir, f"{stem}.jpg"), quality=85)
        print(f"  {stem}: {mask.sum()} px", flush=True)


if __name__ == "__main__":
    roots = sys.argv[1:] or ["realsense", "epi"]
    for root in roots:
        for seq, prompt in PROMPTS.items():
            run_sequence(os.path.join(CAPTURED, root, seq), prompt)
    print("done")
