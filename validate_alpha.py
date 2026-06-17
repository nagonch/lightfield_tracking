"""Validate the production estimate_alpha() (reflection_separation.py) end to end:
build SLFs exactly as src/reflection.py packs them ([P, M, *]) and accumulate the
per-frame stat across a sequence, reporting the final alpha vs ground truth
(alpha = 1 - reflectivity) on diverse geometry.
"""
import os
import numpy as np

from src.dataset import LFDataset
from src.surface_light_field import SurfaceLightField
from reflection_separation import estimate_alpha

DATASET_ROOT = "/home/ngoncharov/SpecTrack_dataset"
SEQUENCES = ["cracker_box_yalehand0", "mustard0", "tomato_soup_can_yalehand0", "bleach0"]
REFLS = ["0.0", "0.5", "0.7", "1.0"]
SPLITS = ["cube", "objects"]
FRAMES = list(range(0, 60, 6))


def seq_alpha(seq_path):
    ds = LFDataset(seq_path, depth_source="gt")
    s_size, t_size = ds.metadata["n_views"]
    hist = None
    alpha = None
    for fi in FRAMES:
        if fi >= len(ds):
            break
        frame = ds[fi]
        mask = frame["masks"][s_size // 2, t_size // 2]
        slf = SurfaceLightField.from_frame(frame, mask, frame["depth"], s_size, t_size)
        colors = slf.colors.permute(1, 0, 2).float()     # [P, M, 3]
        view_dirs = slf.view_dirs.permute(1, 0, 2).float()
        valid = slf.valid.permute(1, 0)
        normals = slf.normals.float()
        alpha, hist = estimate_alpha(colors, valid, view_dirs, normals, hist)
    return alpha


def main():
    print(f"{'split':8s} {'a_true':>6s} {'a_hat mean':>10s} {'sd':>5s} {'MAE':>5s}")
    all_ae = []
    for split in SPLITS:
        for r in REFLS:
            a_true = 1.0 - float(r)
            ah = []
            for seq in SEQUENCES:
                sp = os.path.join(DATASET_ROOT, f"{split}_{r}", seq)
                if os.path.isdir(sp):
                    ah.append(seq_alpha(sp))
            ae = [abs(a - a_true) for a in ah]
            all_ae += ae
            print(f"{split:8s} {a_true:6.2f} {np.mean(ah):10.3f} {np.std(ah):5.2f} "
                  f"{np.mean(ae):5.3f}")
        print()
    print(f"overall MAE: {np.mean(all_ae):.3f}")


if __name__ == "__main__":
    main()
