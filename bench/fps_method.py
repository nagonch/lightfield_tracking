"""FPS + GPU benchmark for ReLiFT-6DoF (our method) on a single sequence.

Runs the full pipeline (reflection separation → diffuse view → LoFTR coarse pose
→ photometric refinement) on one object sequence at the highest reflectivity and
reports steady-state FPS plus GPU memory / utilisation / power.

Timing reuses main.track_sequence's own per-frame measurement (`measure_fps`),
which excludes frame 0 (CUDA warm-up) and the model load, and synchronises CUDA
around each frame. Separation caching is disabled for an honest end-to-end number.

Run inside the container (see run_container.sh):
    python bench/fps_method.py                      # objects_1.0/bleach0
    python bench/fps_method.py --seq mustard0
    python bench/fps_method.py --gt-masks           # skip the segmentor
    python bench/fps_method.py --depth lf --no-cache-depth
"""

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from bench.gpu_monitor import GpuMonitor
from config import (
    DATASET_ROOT,
    DEPTH_SOURCES,
    LF_DEPTH_CFG,
    PIN_ALPHA,
    REFINE_CFG,
    REFINE_FEED_FORWARD,
    USE_REFLECTION_SEPARATION,
)
from lf_depth import LFPlaneSweepDepth
from loftr_wrapper import LoftrRunner
from main import track_sequence


def main() -> None:
    ap = argparse.ArgumentParser(description="ReLiFT-6DoF FPS/GPU benchmark")
    ap.add_argument("--split", default="objects", help="split prefix (default: objects)")
    ap.add_argument("--refl", default="1.0", help="reflectivity (default: 1.0 = highest)")
    ap.add_argument("--seq", default="bleach0", help="sequence name (default: bleach0)")
    ap.add_argument("--depth", default=None, help="depth source: gt|synth|lf (default: config)")
    ap.add_argument("--no-cache-depth", action="store_true", help="compute lf depth live")
    ap.add_argument(
        "--cache-separation",
        action="store_true",
        help="read the diffuse cache instead of recomputing (faster but not honest)",
    )
    ap.add_argument("--gt-masks", action="store_true", help="use GT masks (skip segmentor)")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    depth_source = args.depth or DEPTH_SOURCES[0]
    seq_path = os.path.join(DATASET_ROOT, f"{args.split}_{args.refl}", args.seq)
    gt0_seq_path = os.path.join(DATASET_ROOT, f"{args.split}_0.0", args.seq)
    if not os.path.isdir(seq_path):
        sys.exit(f"sequence not found: {seq_path}")

    print(
        f"method | {args.split}_{args.refl}/{args.seq} | depth={depth_source} "
        f"| masks={'GT' if args.gt_masks else 'segmentor'} "
        f"| separation_cache={'on' if args.cache_separation else 'off'}"
    )

    # Models load before timing starts (not counted toward FPS).
    loftr = LoftrRunner()
    rng = np.random.default_rng(seed=42)
    live_lf = args.no_cache_depth and depth_source == "lf"
    depth_estimator = LFPlaneSweepDepth(LF_DEPTH_CFG) if live_lf else None

    pin_alpha = PIN_ALPHA
    alpha = (1.0 - float(args.refl)) if pin_alpha else None

    tmp = tempfile.mkdtemp(prefix="fps_method_")
    fps_samples: list[float] = []

    torch.cuda.reset_peak_memory_stats()
    with GpuMonitor() as mon:
        track_sequence(
            seq_path=seq_path,
            results_dir=tmp,
            cache_dir=tmp,
            sequence_name=args.seq,
            alpha=alpha,
            depth_source=depth_source,
            loftr=loftr,
            rng=rng,
            separate=USE_REFLECTION_SEPARATION,
            refine=True,
            max_frames=args.max_frames,
            refine_cfg=REFINE_CFG,
            gt0_seq_path=gt0_seq_path,
            reflectivity=float(args.refl),
            feed_forward=REFINE_FEED_FORWARD,
            no_cache_separation=not args.cache_separation,
            depth_estimator=depth_estimator,
            gt_masks=args.gt_masks,
            measure_fps=True,
            fps_samples=fps_samples,
        )

    # ── LoFTR-only baseline (same sequence, no separation, no refinement) ──
    tmp_loftr = tempfile.mkdtemp(prefix="fps_loftr_")
    fps_samples_loftr: list[float] = []

    torch.cuda.reset_peak_memory_stats()
    with GpuMonitor() as mon_loftr:
        track_sequence(
            seq_path=seq_path,
            results_dir=tmp_loftr,
            cache_dir=tmp_loftr,
            sequence_name=args.seq,
            alpha=alpha,
            depth_source=depth_source,
            loftr=loftr,
            rng=rng,
            separate=False,
            refine=False,
            max_frames=args.max_frames,
            gt0_seq_path=gt0_seq_path,
            reflectivity=float(args.refl),
            depth_estimator=depth_estimator,
            gt_masks=args.gt_masks,
            measure_fps=True,
            fps_samples=fps_samples_loftr,
        )

    tag = f"{args.split}_{args.refl}/{args.seq}"
    lines = [f"ReLiFT-6DoF (method) | {tag} | depth={depth_source}"]
    if fps_samples:
        tot = sum(fps_samples)
        n = len(fps_samples)
        lines += [
            "── FPS [method] ──",
            f"  {n / tot:6.2f} FPS   "
            f"({n} timed frames, {1000.0 * tot / n:.1f} ms/frame mean)",
        ]
    lines.append(mon.format_report("method", torch_module=torch))

    lines += ["", f"LoFTR baseline | {tag} | depth={depth_source}"]
    if fps_samples_loftr:
        tot_l = sum(fps_samples_loftr)
        n_l = len(fps_samples_loftr)
        lines += [
            "── FPS [loftr] ──",
            f"  {n_l / tot_l:6.2f} FPS   "
            f"({n_l} timed frames, {1000.0 * tot_l / n_l:.1f} ms/frame mean)",
        ]
    lines.append(mon_loftr.format_report("loftr", torch_module=torch))

    report = "\n".join(lines)
    out_txt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fps_method.txt")
    with open(out_txt, "w") as f:
        f.write(report + "\n")
    print("\n" + report + f"\n\nsaved -> {out_txt}")


if __name__ == "__main__":
    main()
