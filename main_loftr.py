import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main_prod import SpecTrackSequence
from loftr_baseline import LoftrBase


DATASET_ROOT = "/home/ngoncharov/SpecTrack_dataset"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_loftr")


if __name__ == "__main__":
    splits = sorted(
        d
        for d in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, d))
        and (d.startswith("cube_") or d.startswith("objects_"))
    )

    for depth_mode in ("gt", "synth"):
        for split in splits:
            split_dir = os.path.join(DATASET_ROOT, split)
            out_dir = os.path.join(RESULTS_DIR, depth_mode, split)
            os.makedirs(out_dir, exist_ok=True)

            sequences = sorted(
                s
                for s in os.listdir(split_dir)
                if os.path.isdir(os.path.join(split_dir, s)) and s != "models"
            )

            for seq_name in tqdm(sequences, desc=f"{depth_mode}/{split}"):
                out_path = os.path.join(out_dir, f"{seq_name}.npy")
                if os.path.exists(out_path):
                    tqdm.write(f"  {depth_mode}/{split}/{seq_name}: already done, skipping")
                    continue

                try:
                    seq = SpecTrackSequence(os.path.join(split_dir, seq_name), depth_mode=depth_mode)
                    tracker = LoftrBase(seq)
                    poses = tracker.run()
                    np.save(out_path, poses)
                    tqdm.write(f"  {depth_mode}/{split}/{seq_name}: {poses.shape} → {out_path}")
                except Exception as e:
                    tqdm.write(f"  {depth_mode}/{split}/{seq_name}: FAILED — {e}")
