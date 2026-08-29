"""main_lift.py — benchmark the pipeline on the real LiFT dataset.

Runs the exact per-frame method from main.py (track_sequence is imported, not
modified) on the real 9×9 light-field captures in
~/cvpr2026/datasets/LiFT_dataset. This driver only swaps in a real-data work
list (flat sequences, no reflectivity splits, car_* excluded), and exposes the
tuning surface: LF thinning (--view-stride / --scale), alpha pinning
(--alpha), and per-run hyperparameter overrides (--set refine.KEY=VAL etc.)
without editing config.yaml.

Typical runs (inside the lift6dof container, or via ./run_lift.sh):

    # LoFTR-only backbone, GT (RealSense) depth, GT masks
    python main_lift.py --gt-masks

    # Full method with photometric refinement
    python main_lift.py --refine --gt-masks

    # Segmentor masks (gdino_prompt.txt drives GroundingDINO)
    python main_lift.py --refine

    # Pre-write LF plane-sweep depth, then track on it
    python main_lift.py --write-lf-depth
    python main_lift.py --refine --gt-masks --depth lf

    # Hyperparameter probe on one sequence, isolated output dir
    python main_lift.py --refine --gt-masks --seqs jug_tilt \
        --set refine.lambda_depth=1.0 --set refine.num_iters=100 \
        --no-cache-separation --name sweeps/lift_ld1.0

Results land in <exp>/<depth>/<seq>.npy (rebased to GT frame 0) plus a
run_config.json snapshot of every knob; evaluate with eval/run_eval_lift.py.
"""

import argparse
import json
import logging
import os
from dataclasses import asdict, replace

# The real dataset carries its own tuned hyperparameters: config.py deep-merges
# config_lift.yaml over config.yaml. Must be set before config/main are
# imported (they read the config at import time).
os.environ.setdefault("CONFIG_OVERRIDES", "config_lift.yaml")

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

import config as config_mod
import main as pipeline  # reuse track_sequence + its module-level knobs
from config import (
    CACHE_ROOT,
    ENABLE_VIS,
    LF_DEPTH_CFG,
    REFINE_CFG,
    REFINE_FEED_FORWARD,
    USE_REFLECTION_SEPARATION,
)
from lf_depth import LFPlaneSweepDepth
from loftr_wrapper import LoftrRunner
from src.dataset import LFDataset, set_lift_options
from src.photometric import PhotometricRefineViewer

# Absolute (not ~-based): the container runs with HOME=/root but /home mounted.
LIFT_ROOT = "/home/ngoncharov/cvpr2026/datasets/LiFT_dataset"

logging.basicConfig(
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)


def list_sequences(root: str, include: str | None) -> list[str]:
    """All benchmark sequences: top-level *_prod dirs, car_* excluded."""
    seqs = [
        d
        for d in sorted(os.listdir(root))
        if d.endswith("_prod")
        and not d.startswith("car_")
        and os.path.isdir(os.path.join(root, d))
    ]
    if include:
        keys = [k for k in include.split(",") if k]
        seqs = [s for s in seqs if any(k in s for k in keys)]
    return seqs


def apply_overrides(sets: list[str], refine_cfg, lf_cfg):
    """--set KEY=VAL overrides. refine.*/lf_depth.* rebuild the dataclass;
    a few pipeline globals are patched on the imported main module."""
    for kv in sets:
        key, _, raw = kv.partition("=")
        if not _:
            raise SystemExit(f"--set expects KEY=VAL, got: {kv}")
        val = yaml.safe_load(raw)
        if key.startswith("refine."):
            refine_cfg = replace(refine_cfg, **{key[len("refine."):]: val})
        elif key.startswith("lf_depth."):
            lf_cfg = replace(lf_cfg, **{key[len("lf_depth."):]: val})
        elif key == "tracking.loftr_resize":
            config_mod.LOFTR_RESIZE = int(val)
        elif key == "tracking.ransac_inlier_dist":
            config_mod.RANSAC_INLIER_DIST = float(val)
        elif key == "separation_iters":
            pipeline.SEPARATION_ITERS = int(val)
        elif key == "min_track_pixels":
            pipeline.MIN_TRACK_PIXELS = int(val)
        elif key == "alpha_stable_tol":
            pipeline.ALPHA_STABLE_TOL = float(val)
        else:
            raise SystemExit(f"unknown --set key: {key}")
    return refine_cfg, lf_cfg


def write_lf_depth(
    seqs: list[str], root: str, cfg, overwrite: bool = False
) -> None:
    """Plane-sweep depth for every frame → <seq>/depth_lf/*.png (uint16 mm).

    Always saved at the native 1280×720 so depth_lf/ mirrors depth/ regardless
    of the --scale the estimator ran at (the loader rescales on read).
    """
    estimator = LFPlaneSweepDepth(cfg)
    for seq in seqs:
        seq_path = os.path.join(root, seq)
        ds = LFDataset(seq_path, depth_source="gt")
        s_size, t_size = ds.metadata["n_views"]
        out_dir = os.path.join(seq_path, "depth_lf")
        os.makedirs(out_dir, exist_ok=True)
        native = Image.open(os.path.join(ds.depth_dir, ds.depth_fnames[0])).size
        for i in tqdm(range(len(ds)), desc=seq, unit="fr", dynamic_ncols=True):
            out_path = os.path.join(out_dir, ds.depth_fnames[i])
            if os.path.exists(out_path) and not overwrite:
                continue
            frame = ds[i]
            mask = (
                frame["masks"][s_size // 2, t_size // 2]
                if frame["masks"] is not None
                else None
            )
            depth = estimator.estimate(frame, mask=mask)["depth"]
            if depth.shape[::-1] != native:
                depth = torch.nn.functional.interpolate(
                    depth[None, None], size=(native[1], native[0]), mode="nearest"
                )[0, 0]
            d16 = (depth.clamp(0, 65.0) * 1000.0).round().to(torch.int32)
            Image.fromarray(d16.cpu().numpy().astype(np.uint16)).save(out_path)
        logging.info("%s: depth_lf written → %s", seq, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="ReLiFT-6DoF on the real LiFT dataset")
    parser.add_argument("--root", default=LIFT_ROOT, help="LiFT dataset root")
    parser.add_argument(
        "--refine", action="store_true", help="Enable photometric refinement"
    )
    parser.add_argument(
        "--depth",
        default="gt",
        help="comma-separated depth sources: gt (RealSense depth/) | lf "
        "(depth_lf/, pre-written with --write-lf-depth, or live via --no-cache-depth)",
    )
    parser.add_argument(
        "--seqs",
        default=None,
        help="comma-separated substrings selecting sequences (default: all non-car)",
    )
    parser.add_argument("--name", default=None, help="Override experiment name/dir")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Pin the diffuse fraction alpha instead of estimating it from the SLF",
    )
    parser.add_argument(
        "--view-stride",
        type=int,
        default=2,
        help="Keep every Nth LF row/col, centred on the central view "
        "(default 2 → 9×9 becomes 5×5 over the full 40 mm baseline)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Image downscale factor applied to LF/depth/masks/K (default 1.0 = 720p)",
    )
    parser.add_argument("--gt-masks", action="store_true", help="Dataset masks, not segmentor")
    parser.add_argument("--no-separation", action="store_true")
    parser.add_argument("--no-cache-separation", action="store_true")
    parser.add_argument(
        "--no-cache-depth",
        action="store_true",
        help="depth=lf: compute plane-sweep depth live instead of reading depth_lf/",
    )
    parser.add_argument("--fps", action="store_true")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="Hyperparameter override, repeatable. Namespaces: refine.* "
        "(RefineConfig), lf_depth.* (LFDepthConfig), plus separation_iters, "
        "min_track_pixels, alpha_stable_tol.",
    )
    parser.add_argument(
        "--write-lf-depth",
        action="store_true",
        help="Only write depth_lf/*.png for the selected sequences, then exit",
    )
    parser.add_argument(
        "--overwrite-lf-depth",
        action="store_true",
        help="With --write-lf-depth: regenerate existing files",
    )
    args = parser.parse_args()

    set_lift_options(view_stride=args.view_stride, image_scale=args.scale)
    refine_cfg, lf_cfg = apply_overrides(args.set, REFINE_CFG, LF_DEPTH_CFG)
    seqs = list_sequences(args.root, args.seqs)
    if not seqs:
        raise SystemExit(f"No sequences matched under {args.root}")
    logging.info("sequences (%d): %s", len(seqs), ", ".join(seqs))

    if args.write_lf_depth:
        write_lf_depth(seqs, args.root, lf_cfg, overwrite=args.overwrite_lf_depth)
        return

    depth_sources = [d for d in args.depth.split(",") if d]
    separate = USE_REFLECTION_SEPARATION and not args.no_separation
    if not separate:
        exp_name = "lift_no_separation"
    elif not args.refine:
        exp_name = "lift_loftr"
    else:
        exp_name = "lift_refine_est"
    if args.name:
        exp_name = args.name

    logging.info(
        "refine=%s  separation=%s  depth=%s  stride=%d  scale=%.2f  alpha=%s  → %s",
        args.refine,
        separate,
        depth_sources,
        args.view_stride,
        args.scale,
        "est" if args.alpha is None else args.alpha,
        exp_name,
    )
    logging.info("masks: %s", "GT (dataset)" if args.gt_masks else "segmentor")

    # Work list: <exp>/<depth>/<seq>.npy — flat, no reflectivity splits.
    work = []
    for depth_source in depth_sources:
        for seq in seqs:
            seq_path = os.path.join(args.root, seq)
            if (
                depth_source == "lf"
                and not args.no_cache_depth
                and not os.path.isdir(os.path.join(seq_path, "depth_lf"))
            ):
                logging.warning(
                    "depth_lf missing for %s — run `python main_lift.py "
                    "--write-lf-depth` first (or pass --no-cache-depth); skipping",
                    seq,
                )
                continue
            work.append(
                {
                    "depth_source": depth_source,
                    "sequence_name": seq,
                    "seq_path": seq_path,
                    "results_dir": os.path.join(exp_name, depth_source),
                    # Separation output depends on both the depth source and the
                    # mask source, so the cache is keyed by both.
                    "cache_dir": os.path.join(
                        f"{CACHE_ROOT}_lift",
                        f"{depth_source}_{'gtmask' if args.gt_masks else 'segmask'}",
                        seq,
                    ),
                    "tag": f"{depth_source}/{seq}",
                }
            )

    # Snapshot every knob for reproducible tuning runs.
    os.makedirs(exp_name, exist_ok=True)
    with open(os.path.join(exp_name, "run_config.json"), "w") as f:
        json.dump(
            {
                "args": vars(args),
                "refine_cfg": asdict(refine_cfg),
                "lf_depth_cfg": asdict(lf_cfg),
                "separation_iters": pipeline.SEPARATION_ITERS,
                "min_track_pixels": pipeline.MIN_TRACK_PIXELS,
                "loftr_resize": config_mod.LOFTR_RESIZE,
                "config_overrides": os.environ.get("CONFIG_OVERRIDES"),
            },
            f,
            indent=2,
        )

    loftr = LoftrRunner()
    rng = np.random.default_rng(seed=42)
    depth_estimator = (
        LFPlaneSweepDepth(lf_cfg)
        if args.no_cache_depth and "lf" in depth_sources
        else None
    )
    viewer = PhotometricRefineViewer(port=8081) if args.refine and ENABLE_VIS else None
    fps_samples: list[float] | None = [] if args.fps else None

    with tqdm(work, desc="sequences", unit="seq", dynamic_ncols=True) as bar:
        for item in bar:
            bar.set_postfix_str(item["tag"])
            os.makedirs(item["results_dir"], exist_ok=True)
            out_path = os.path.join(
                item["results_dir"], f"{item['sequence_name']}.npy"
            )
            if os.path.exists(out_path):
                logging.info("%s: already done, skipping", item["tag"])
                continue
            try:
                pipeline.track_sequence(
                    seq_path=item["seq_path"],
                    results_dir=item["results_dir"],
                    cache_dir=item["cache_dir"],
                    sequence_name=item["sequence_name"],
                    alpha=args.alpha,
                    depth_source=item["depth_source"],
                    loftr=loftr,
                    rng=rng,
                    separate=separate,
                    refine=args.refine,
                    viewer=viewer,
                    max_frames=args.max_frames,
                    refine_cfg=refine_cfg,
                    gt_refine=False,
                    gt0_seq_path=None,
                    reflectivity=0.0,
                    gt_env=None,
                    feed_forward=REFINE_FEED_FORWARD,
                    no_cache_separation=args.no_cache_separation,
                    depth_estimator=depth_estimator,
                    gt_masks=args.gt_masks,
                    measure_fps=args.fps,
                    fps_samples=fps_samples,
                )
            except Exception:
                logging.exception("%s: FAILED", item["tag"])

    if fps_samples:
        tot, n = sum(fps_samples), len(fps_samples)
        logging.info(
            "OVERALL FPS %.2f  (%d frames, %.1f ms/frame mean) over %s",
            n / tot if tot > 0 else 0.0,
            n,
            1000.0 * tot / n,
            exp_name,
        )
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
