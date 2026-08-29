#!/usr/bin/env python3
"""Classical baselines on the real LiFT dataset (car_* sequences excluded).

Replicates the synthetic baselines/ suite for the real captures. The tracker
implementations are imported unmodified from baselines/ and loftr_baseline.py;
only the sequence loader is swapped for the real 9×9 layout:

    loftr : LoFTR + 3-pt RANSAC (BundleSDF coarse-pose thread)
    icp   : frame-to-frame colored ICP (Park et al. 2017, Open3D GPU)
    pnp   : SIFT + EPnP-RANSAC + LM refinement

FoundationPose / BundleSDF need external repos + meshes, so they are not
reproducible here (results_fp / results_bsdf remain synthetic-only).

Results (matches eval/run_eval_lift.py layout):
    baselines_real/results_<method>/<depth>/<seq>.npy

Run inside the lift6dof container:
    python baselines_real/run_baselines_lift.py                 # all three, gt depth
    python baselines_real/run_baselines_lift.py --methods icp --depth lf
Evaluate with ./run_eval_baselines_real.sh (inside the container).
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)


def _load_baseline_module(name: str):
    """Load baselines/<name>.py by path. A plain `import icp` is ambiguous:
    the repo root also has an icp.py (used by loftr_baseline), and whichever
    lands in sys.modules first wins."""
    spec = importlib.util.spec_from_file_location(
        f"baseline_{name}", os.path.join(_ROOT, "baselines", f"{name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

LIFT_ROOT = "/home/ngoncharov/cvpr2026/datasets/LiFT_dataset"


class LiFTSequence:
    """Central-view RGB, depth, mask, and GT object poses for a real sequence.

    Drop-in replacement for the baselines' SpecTrackSequence: 9×9 grid →
    central view flat index 40; poses are already OpenCV (no axis flip);
    depth_mode "gt" → depth/ (RealSense), "lf" → depth_lf/ (plane-sweep).
    """

    DEPTH_DIRS = {"gt": "depth", "lf": "depth_lf"}

    def __init__(self, seq_dir: str, depth_mode: str = "gt"):
        assert depth_mode in self.DEPTH_DIRS, f"bad depth_mode {depth_mode!r}"
        self.seq_dir = seq_dir
        self.depth_dir = self.DEPTH_DIRS[depth_mode]
        self.K = np.loadtxt(os.path.join(seq_dir, "camera_matrix.txt"))

        with open(os.path.join(seq_dir, "metadata.json")) as f:
            S, T = json.load(f)["n_views"]
        self.central_view = (S // 2) * T + T // 2

        self.frames = sorted(d for d in os.listdir(seq_dir) if d.startswith("LF_"))
        self.obj_pose_files = sorted(os.listdir(os.path.join(seq_dir, "object_poses")))

        cam_pose = np.loadtxt(
            os.path.join(seq_dir, "camera_poses", f"{self.central_view:04d}.txt")
        )
        self._inv_cam_pose = np.linalg.inv(cam_pose)

    def __len__(self) -> int:
        return len(self.frames)

    def get_gt_pose(self, idx: int) -> np.ndarray:
        world_obj = np.loadtxt(
            os.path.join(self.seq_dir, "object_poses", self.obj_pose_files[idx])
        )
        return self._inv_cam_pose @ world_obj

    def get_frame(self, idx: int):
        """(rgb uint8 HxWx3, depth float64 metres HxW, mask bool HxW)."""
        frame_dir = os.path.join(self.seq_dir, self.frames[idx])
        vname = f"{self.central_view:04d}.png"
        rgb = np.array(Image.open(os.path.join(frame_dir, vname)))
        depth_fname = self.frames[idx][3:] + ".png"  # "LF_0007" → "0007.png"
        depth = (
            np.array(
                Image.open(os.path.join(self.seq_dir, self.depth_dir, depth_fname))
            ).astype(np.float64)
            / 1000.0
        )
        mask = np.array(Image.open(os.path.join(frame_dir, "masks", vname))) > 0
        return rgb, depth, mask


def list_sequences(include: str | None) -> list[str]:
    seqs = [
        d
        for d in sorted(os.listdir(LIFT_ROOT))
        if d.endswith("_prod")
        and not d.startswith("car_")
        and os.path.isdir(os.path.join(LIFT_ROOT, d))
    ]
    if include:
        keys = [k for k in include.split(",") if k]
        seqs = [s for s in seqs if any(k in s for k in keys)]
    return seqs


def make_tracker(method: str, seq: LiFTSequence):
    if method == "loftr":
        from loftr_baseline import LoftrBase

        return LoftrBase(seq)
    if method == "icp":
        return _load_baseline_module("icp").ColoredICPTracker(seq)
    if method == "pnp":
        return _load_baseline_module("pnp").PnPTracker(seq)
    raise ValueError(method)


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-LiFT classical baselines")
    ap.add_argument("--methods", default="loftr,icp,pnp")
    ap.add_argument("--depth", default="gt", choices=["gt", "lf"])
    ap.add_argument("--seqs", default=None, help="comma-separated substrings")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    methods = [m for m in args.methods.split(",") if m]
    seqs = list_sequences(args.seqs)
    if not seqs:
        raise SystemExit("no sequences matched")
    print(f"methods={methods}  depth={args.depth}  sequences={seqs}", flush=True)

    if "loftr" in methods:
        # Load the LoFTR checkpoint once and reuse it across sequences.
        import loftr_baseline

        shared = loftr_baseline.LoftrRunner()
        loftr_baseline.LoftrRunner = lambda *a, **k: shared

    for method in methods:
        out_dir = os.path.join(_HERE, f"results_{method}", args.depth)
        os.makedirs(out_dir, exist_ok=True)
        for seq_name in tqdm(seqs, desc=method, dynamic_ncols=True):
            out_path = os.path.join(out_dir, f"{seq_name}.npy")
            if os.path.exists(out_path) and not args.overwrite:
                tqdm.write(f"  {method}/{seq_name}: already done, skipping")
                continue
            seq = LiFTSequence(os.path.join(LIFT_ROOT, seq_name), args.depth)
            try:
                poses = make_tracker(method, seq).run()
            except Exception as e:
                tqdm.write(f"  {method}/{seq_name}: FAILED ({e})")
                continue
            np.save(out_path, poses)
            tqdm.write(f"  {method}/{seq_name}: {poses.shape} → {out_path}")

    print("done.", flush=True)


if __name__ == "__main__":
    main()
