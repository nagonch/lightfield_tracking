"""FPS + GPU benchmark for the colored-ICP baseline (baselines/icp.py).

Single object sequence at the highest reflectivity. Per-frame timing excludes
frame 0 and the raw image load (to match main.track_sequence, which times the
tracking compute, not disk IO). Open3D CUDA ops are forced to complete each
frame via a device synchronise so the timing reflects real GPU work.

Run inside the container:
    python bench/fps_icp.py                 # objects_1.0/bleach0, gt depth
    python bench/fps_icp.py --seq mustard0 --depth synth
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import open3d as o3d

from baselines.icp import DATASET_ROOT, ColoredICPTracker, SpecTrackSequence
from bench.gpu_monitor import GpuMonitor


def _sync() -> None:
    try:
        o3d.core.cuda.synchronize()
    except (AttributeError, RuntimeError):
        pass


def timed_run(tracker: ColoredICPTracker) -> list[float]:
    """Mirror ColoredICPTracker.run() but record per-frame compute time."""
    seq = tracker.seq
    n = len(seq)
    est_poses: list[np.ndarray] = []
    prev_pcd = None
    times: list[float] = []

    for idx in range(n):
        rgb, depth, mask = seq.get_frame(idx)  # IO excluded from timing
        if idx >= 1:
            _sync()
            t0 = time.perf_counter()

        cur_pcd = tracker._build_pcd(rgb, depth, mask)
        if idx == 0:
            current_pose = np.eye(4)
        else:
            if idx == 1 or est_poses[-2] is None:
                init = np.eye(4)
            else:
                init = est_poses[-1] @ np.linalg.inv(est_poses[-2])
            if cur_pcd is None or prev_pcd is None:
                current_pose = est_poses[-1].copy()
            else:
                T_rel = tracker._register(prev_pcd, cur_pcd, init)
                current_pose = T_rel @ est_poses[-1]

        if idx >= 1:
            _sync()
            times.append(time.perf_counter() - t0)

        est_poses.append(current_pose)
        prev_pcd = cur_pcd

    return times


def main() -> None:
    ap = argparse.ArgumentParser(description="Colored-ICP FPS/GPU benchmark")
    ap.add_argument("--split", default="objects")
    ap.add_argument("--refl", default="1.0", help="highest reflectivity (default 1.0)")
    ap.add_argument("--seq", default="bleach0")
    ap.add_argument("--depth", default="gt", choices=["gt", "synth"])
    args = ap.parse_args()

    seq_dir = os.path.join(DATASET_ROOT, f"{args.split}_{args.refl}", args.seq)
    if not os.path.isdir(seq_dir):
        sys.exit(f"sequence not found: {seq_dir}")
    print(f"icp | {args.split}_{args.refl}/{args.seq} | depth={args.depth}")

    seq = SpecTrackSequence(seq_dir, depth_mode=args.depth)
    tracker = ColoredICPTracker(seq)

    # Warm up CUDA / build the kernels once so frame-1 timing is steady-state.
    rgb, depth, mask = seq.get_frame(0)
    _ = tracker._build_pcd(rgb, depth, mask)
    _sync()

    with GpuMonitor() as mon:
        times = timed_run(tracker)

    if times:
        tot = sum(times)
        n = len(times)
        print(
            f"\n── FPS [icp] ──\n  {n / tot:6.2f} FPS   "
            f"({n} timed frames, {1000.0 * tot / n:.1f} ms/frame mean)"
        )
    mon.report("icp")


if __name__ == "__main__":
    main()
