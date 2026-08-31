#!/usr/bin/env python3
"""Classical baselines on the captured sequences (realsense + epi light field).

Runs the repo's existing trackers, unmodified, over the four captured sequences
(realsense/{diffuse,reflective} = real RealSense RGB-D; epi/{diffuse,reflective} =
light-field cross: center view + the simulated-RealSense stereo depth):

    loftr : LoFTR + 3-pt RANSAC (BundleSDF coarse-pose thread)
    icp   : frame-to-frame colored ICP (Park et al. 2017, Open3D GPU)
    pnp   : SIFT + EPnP-RANSAC + LM refinement

Only the sequence loader differs: GT object poses come from poses_object/
(flange poses corrected by the fitted mount transform, see ee_T_obj.txt), masks
from masks/ (GroundingDINO+SAM2+Cutie), and the epi center view is undistorted
to a pinhole model on load (the trackers assume pinhole K).

Results:  baselines_captured/results_<method>/<capture>_<seq>.npy
Evaluate: python eval/run_eval_captured.py --gifs

Run inside the lift6dof container:
    docker exec -e CUDA_VISIBLE_DEVICES=0 -w "$PWD" lift6dof \
        python baselines_captured/run_baselines_captured.py
"""

import argparse
import importlib.util
import os
import sys

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

CAPTURED_ROOT = "/home/ngoncharov/SpecTrack_dataset/captured"
SEQUENCES = [
    ("realsense", "diffuse"),
    ("realsense", "reflective"),
    ("epi", "diffuse"),
    ("epi", "reflective"),
]


def _load_baseline_module(name: str):
    """Load baselines/<name>.py by path (a plain `import icp` is ambiguous:
    the repo root also has an icp.py used by loftr_baseline)."""
    spec = importlib.util.spec_from_file_location(
        f"baseline_{name}", os.path.join(_ROOT, "baselines", f"{name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CapturedSequence:
    """RGB, depth, mask, and GT object poses for one captured sequence.

    Drop-in for the baselines' SpecTrackSequence. GT pose = poses_object/
    (OpenCV camera convention, no axis flip). Depth is uint16 mm. The epi
    center view (images/XXXXXX/cam_4_4.png) carries lens distortion, so RGB
    is undistorted (linear) and depth/mask are undistorted nearest-neighbour
    onto the same pinhole K the trackers use.
    """

    def __init__(self, seq_dir: str):
        self.seq_dir = seq_dir
        self.K = np.loadtxt(os.path.join(seq_dir, "intrinsics.txt"))

        entries = sorted(os.listdir(os.path.join(seq_dir, "images")))
        self.stems = [e.replace(".png", "") for e in entries]
        self._img_paths = [
            os.path.join(seq_dir, "images", e, "cam_4_4.png")
            if os.path.isdir(os.path.join(seq_dir, "images", e))
            else os.path.join(seq_dir, "images", e)
            for e in entries
        ]

        self._maps = None
        dpath = os.path.join(seq_dir, "distortion.txt")
        if os.path.exists(dpath):
            D = np.loadtxt(dpath)
            h, w = np.asarray(Image.open(self._img_paths[0])).shape[:2]
            self._maps = cv2.initUndistortRectifyMap(
                self.K, D, None, self.K, (w, h), cv2.CV_32FC1)

    def __len__(self) -> int:
        return len(self.stems)

    def get_gt_pose(self, idx: int) -> np.ndarray:
        return np.loadtxt(
            os.path.join(self.seq_dir, "poses_object", self.stems[idx] + ".txt"))

    def get_frame(self, idx: int):
        """(rgb uint8 HxWx3, depth float64 metres HxW, mask bool HxW)."""
        s = self.stems[idx]
        rgb = np.asarray(Image.open(self._img_paths[idx]))
        depth = np.asarray(
            Image.open(os.path.join(self.seq_dir, "depth", s + ".png"))
        ).astype(np.float64) / 1000.0
        mask = np.asarray(Image.open(os.path.join(self.seq_dir, "masks", s + ".png"))) > 127
        if self._maps is not None:
            mx, my = self._maps
            rgb = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR)
            depth = cv2.remap(depth, mx, my, cv2.INTER_NEAREST)
            mask = cv2.remap(mask.astype(np.uint8), mx, my, cv2.INTER_NEAREST) > 0
        return rgb, depth, mask


def make_tracker(method: str, seq: CapturedSequence):
    if method == "loftr":
        from loftr_baseline import LoftrBase

        return LoftrBase(seq)
    if method == "icp":
        return _load_baseline_module("icp").ColoredICPTracker(seq)
    if method == "pnp":
        return _load_baseline_module("pnp").PnPTracker(seq)
    raise ValueError(method)


def main() -> None:
    ap = argparse.ArgumentParser(description="Captured-dataset classical baselines")
    ap.add_argument("--methods", default="loftr,icp,pnp")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--time-only", action="store_true",
                    help="Profile wall time per frame (whole tracker.run / frames) and write "
                         "timing_<method>.json instead of saving poses (results stay untouched)")
    ap.add_argument("--seqs", default=None, help="comma-separated <capture>_<seq> tags")
    args = ap.parse_args()

    methods = [m for m in args.methods.split(",") if m]
    seqs = SEQUENCES
    if args.seqs:
        want = set(args.seqs.split(","))
        seqs = [(c, s) for c, s in SEQUENCES if f"{c}_{s}" in want]
    print(f"methods={methods}  sequences={seqs}", flush=True)

    if "loftr" in methods:
        import loftr_baseline

        shared = loftr_baseline.LoftrRunner()
        loftr_baseline.LoftrRunner = lambda *a, **k: shared

    import json
    import time

    for method in methods:
        out_dir = os.path.join(_HERE, f"results_{method}")
        os.makedirs(out_dir, exist_ok=True)
        timing = {}
        for capture, seq_name in tqdm(seqs, desc=method, dynamic_ncols=True):
            tag = f"{capture}_{seq_name}"
            out_path = os.path.join(out_dir, f"{tag}.npy")
            if os.path.exists(out_path) and not args.overwrite and not args.time_only:
                tqdm.write(f"  {method}/{tag}: already done, skipping")
                continue
            seq = CapturedSequence(os.path.join(CAPTURED_ROOT, capture, seq_name))
            try:
                tracker = make_tracker(method, seq)
                t0 = time.perf_counter()
                poses = tracker.run()
                dt = time.perf_counter() - t0
            except Exception as e:
                tqdm.write(f"  {method}/{tag}: FAILED ({e})")
                continue
            if args.time_only:
                # frame 0 is the GT seed; the remaining frames are tracked
                timing[tag] = {"ms_per_frame": 1000.0 * dt / max(len(seq) - 1, 1),
                               "frames": len(seq) - 1}
                tqdm.write(f"  {method}/{tag}: {timing[tag]['ms_per_frame']:.1f} ms/frame")
                continue
            np.save(out_path, poses)
            tqdm.write(f"  {method}/{tag}: {poses.shape} -> {out_path}")
        if args.time_only and timing:
            with open(os.path.join(_HERE, f"timing_{method}.json"), "w") as f:
                json.dump(timing, f, indent=2)

    print("done.", flush=True)


if __name__ == "__main__":
    main()
