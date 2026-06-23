"""FPS + GPU benchmark for the SIFT+PnP baseline (baselines/pnp.py).

Single object sequence at the highest reflectivity. Per-frame timing excludes
frame 0 and the raw image load, matching the other benchmarks. This pipeline
(SIFT detection, FLANN matching, solvePnPRansac) runs on the CPU via OpenCV, so
the GPU numbers will be near-idle — that is the honest result and is reported as
such for comparison against the GPU-bound method and ICP.

Run inside the container:
    python bench/fps_pnp.py                 # objects_1.0/bleach0, gt depth
    python bench/fps_pnp.py --seq mustard0 --depth synth
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from baselines.pnp import DATASET_ROOT, PnPTracker, SpecTrackSequence
from bench.gpu_monitor import GpuMonitor


def timed_run(tracker: PnPTracker) -> list[float]:
    """Mirror PnPTracker.run() but record per-frame compute time."""
    seq = tracker.seq
    n = len(seq)
    est_poses: list[np.ndarray] = []
    prev_pts3d = prev_desc = None
    times: list[float] = []

    for idx in range(n):
        rgb, depth, mask = seq.get_frame(idx)  # IO excluded from timing
        if idx >= 1:
            t0 = time.perf_counter()

        pts3d, kp_pts, desc = tracker._extract(rgb, depth, mask)
        if idx == 0:
            current_pose = np.eye(4)
        else:
            init = (
                est_poses[-1] @ np.linalg.inv(est_poses[-2])
                if len(est_poses) >= 2
                else np.eye(4)
            )
            if pts3d is None or prev_pts3d is None:
                current_pose = init @ est_poses[-1]
            else:
                T_rel = tracker._register(prev_pts3d, prev_desc, kp_pts, desc)
                current_pose = (
                    (T_rel @ est_poses[-1]) if T_rel is not None else (init @ est_poses[-1])
                )

        if idx >= 1:
            times.append(time.perf_counter() - t0)

        est_poses.append(current_pose)
        if pts3d is not None:
            prev_pts3d, prev_desc = pts3d, desc

    return times


def main() -> None:
    ap = argparse.ArgumentParser(description="SIFT+PnP FPS/GPU benchmark")
    ap.add_argument("--split", default="objects")
    ap.add_argument("--refl", default="1.0", help="highest reflectivity (default 1.0)")
    ap.add_argument("--seq", default="bleach0")
    ap.add_argument("--depth", default="gt", choices=["gt", "synth"])
    args = ap.parse_args()

    seq_dir = os.path.join(DATASET_ROOT, f"{args.split}_{args.refl}", args.seq)
    if not os.path.isdir(seq_dir):
        sys.exit(f"sequence not found: {seq_dir}")
    print(f"pnp | {args.split}_{args.refl}/{args.seq} | depth={args.depth}")

    seq = SpecTrackSequence(seq_dir, depth_mode=args.depth)
    tracker = PnPTracker(seq)

    # Warm up SIFT/OpenCV allocations once so frame-1 timing is steady-state.
    rgb, depth, mask = seq.get_frame(0)
    _ = tracker._extract(rgb, depth, mask)

    with GpuMonitor() as mon:
        times = timed_run(tracker)

    lines = [f"SIFT+PnP (baseline, CPU) | {args.split}_{args.refl}/{args.seq} | depth={args.depth}"]
    if times:
        tot = sum(times)
        n = len(times)
        lines += [
            "── FPS [pnp] ──",
            f"  {n / tot:6.2f} FPS   "
            f"({n} timed frames, {1000.0 * tot / n:.1f} ms/frame mean)",
        ]
    lines.append(mon.format_report("pnp"))

    report = "\n".join(lines)
    out_txt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fps_pnp.txt")
    with open(out_txt, "w") as f:
        f.write(report + "\n")
    print("\n" + report + f"\n\nsaved -> {out_txt}")


if __name__ == "__main__":
    main()
