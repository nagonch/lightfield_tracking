"""Plot the four YCB objects used in main.py's dataset (all except tomato soup can).

For each object, take the central view of the first frame (LF_0000) from one
sequence in the fully-diffuse objects_0.0 split, crop to the object's mask
bounding box, and lay them out in a single row.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ROOT = "/home/ngoncharov/SpecTrack_dataset/objects_0.0"
CENTRAL = "0012.png"  # 5x5 light field → central view is sorted index 12
PAD = 12  # px of context around the mask bbox

# (display name, sequence) — one sequence per object, tomato soup can excluded.
OBJECTS = [
    ("Bleach cleanser", "bleach0"),
    ("Cracker box", "cracker_box_reorient"),
    ("Mustard bottle", "mustard0"),
    ("Sugar box", "sugar_box1"),
]


def cropped_central(seq: str) -> np.ndarray:
    base = f"{ROOT}/{seq}/LF_0000"
    rgb = np.asarray(Image.open(f"{base}/{CENTRAL}").convert("RGB"))
    mask = np.asarray(Image.open(f"{base}/masks/{CENTRAL}").convert("L")) > 127
    ys, xs = np.where(mask)
    y0, y1 = max(ys.min() - PAD, 0), min(ys.max() + PAD + 1, rgb.shape[0])
    x0, x1 = max(xs.min() - PAD, 0), min(xs.max() + PAD + 1, rgb.shape[1])
    return rgb[y0:y1, x0:x1]


fig, axes = plt.subplots(1, len(OBJECTS), figsize=(4 * len(OBJECTS), 4))
for ax, (name, seq) in zip(axes, OBJECTS):
    ax.imshow(cropped_central(seq))
    ax.set_title(name, fontsize=14)
    ax.axis("off")

fig.tight_layout()
fig.savefig("dataset_objects.pdf", bbox_inches="tight")
fig.savefig("dataset_objects.png", bbox_inches="tight", dpi=150)
print("wrote dataset_objects.pdf / dataset_objects.png")
