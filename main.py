"""ReLiFT-6DoF — production tracking pipeline.

Per frame:
  1. Load the light-field frame.
  2. Build a surface light field and run reflection separation → diffuse view
     (cached to disk; separation is slow).
  3. LoFTR-match consecutive diffuse views → relative pose → absolute pose.

Runs over every {depth source} × {split} × {reflectivity} × {sequence}, rebases
the estimated trajectory to the GT frame-0 pose, and saves it as <sequence>.npy.
"""

import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from icp import rebase_poses
from loftr_wrapper import LoftrRunner
from src.dataset import LFDataset
from src.pose import track_pose
from src.reflection import central_view, frame_diffuse

logging.basicConfig(
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

# ── configuration ──────────────────────────────────────────────────────────────
DATASET_ROOT = "/home/ngoncharov/SpecTrack_dataset"
EXP_NAME = "results_loftr"
CACHE_ROOT = "cache/diffuse"
SEPARATION_ITERS = 200
USE_REFLECTION_SEPARATION = False  # False → LoFTR on the raw central view

DEPTH_SOURCES = ["gt"]  # "gt" | "synth"
SPLIT_PREFIXES = ["cube", "objects"]
REFLECTIVITIES = ["0.0", "0.5", "0.7", "1.0"]  # "0.0" | "0.5" | "0.7" | "1.0"


def track_sequence(
    seq_path: str,
    results_dir: str,
    cache_dir: str,
    sequence_name: str,
    alpha: float,
    depth_source: str,
    loftr: LoftrRunner,
    rng: np.random.Generator,
    separate: bool,
) -> None:
    dataset = LFDataset(seq_path, depth_source=depth_source)
    s_size, t_size = dataset.metadata["n_views"]

    gt_poses: list[np.ndarray] = []
    est_poses: list[np.ndarray] = []
    prev = None  # (view, depth, mask)

    with tqdm(
        dataset, desc="  frames", unit="fr", leave=False, dynamic_ncols=True
    ) as bar:
        for i, frame in enumerate(bar):
            gt_poses.append(frame["object_pose"].cpu().numpy())

            depth = frame["depth"]
            mask = frame["masks"][s_size // 2, t_size // 2]

            if separate:
                bar.set_postfix(fr=i, stage="separate")
                view = frame_diffuse(
                    frame=frame,
                    mask=mask,
                    depth=depth,
                    alpha=alpha,
                    s_size=s_size,
                    t_size=t_size,
                    cache_path=os.path.join(cache_dir, f"diffuse_{i:04d}.png"),
                    iterations=SEPARATION_ITERS,
                    verbose=True,
                )
            else:
                view = central_view(frame, s_size, t_size)

            depth_np = depth.cpu().numpy()
            mask_np = (mask > 0).cpu().numpy()
            K_np = frame["camera_matrix"].cpu().numpy().astype(np.float64)

            if i == 0:
                est_poses.append(gt_poses[0])
            else:
                bar.set_postfix(fr=i, stage="loftr")
                est_poses.append(
                    track_pose(
                        abs_pose_prev=est_poses[-1],
                        diffuse_prev=prev[0],
                        diffuse_curr=view,
                        depth_prev=prev[1],
                        depth_curr=depth_np,
                        mask_prev=prev[2],
                        mask_curr=mask_np,
                        K=K_np,
                        loftr=loftr,
                        rng=rng,
                    )
                )

            prev = (view, depth_np, mask_np)

    est = rebase_poses(np.stack(gt_poses), np.stack(est_poses))
    out_path = os.path.join(results_dir, f"{sequence_name}.npy")
    np.save(out_path, est)
    logging.info("%s: %s → %s", sequence_name, est.shape, out_path)


def build_work_list() -> list[dict]:
    work = []
    for depth_source in DEPTH_SOURCES:
        for split_prefix in SPLIT_PREFIXES:
            for reflectivity in REFLECTIVITIES:
                split_dir = f"{DATASET_ROOT}/{split_prefix}_{reflectivity}"
                if not os.path.isdir(split_dir):
                    logging.warning("Split not found, skipping: %s", split_dir)
                    continue
                tag = f"{depth_source}/{split_prefix}_{reflectivity}"
                for sequence_name in sorted(os.listdir(split_dir)):
                    seq_path = os.path.join(split_dir, sequence_name)
                    if os.path.isdir(seq_path):
                        work.append(
                            {
                                "depth_source": depth_source,
                                "reflectivity": reflectivity,
                                "sequence_name": sequence_name,
                                "seq_path": seq_path,
                                "results_dir": f"{EXP_NAME}/{tag}",
                                "cache_dir": f"{CACHE_ROOT}/{tag}/{sequence_name}",
                                "tag": f"{tag}/{sequence_name}",
                            }
                        )
    return work


def main() -> None:
    loftr = LoftrRunner()
    rng = np.random.default_rng(seed=42)
    work = build_work_list()

    with tqdm(work, desc="sequences", unit="seq", dynamic_ncols=True) as bar:
        for item in bar:
            bar.set_postfix_str(item["tag"])
            os.makedirs(item["results_dir"], exist_ok=True)

            out_path = os.path.join(item["results_dir"], f"{item['sequence_name']}.npy")
            if os.path.exists(out_path):
                logging.info("%s: already done, skipping", item["tag"])
                continue

            try:
                track_sequence(
                    seq_path=item["seq_path"],
                    results_dir=item["results_dir"],
                    cache_dir=item["cache_dir"],
                    sequence_name=item["sequence_name"],
                    alpha=1.0 - float(item["reflectivity"]),
                    depth_source=item["depth_source"],
                    loftr=loftr,
                    rng=rng,
                    separate=USE_REFLECTION_SEPARATION,
                )
            except Exception:
                logging.exception("%s: FAILED", item["tag"])


if __name__ == "__main__":
    main()
