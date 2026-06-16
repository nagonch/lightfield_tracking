"""Quick photometric-refine hyperparameter sweep vs LoFTR.

Runs a handful of frames over a few diverse sequences (caches make this fast),
viser OFF, for each named RefineConfig variant, then prints an aggregate table:
mean rotation / translation error of LoFTR coarse vs refined, plus the fraction
of frames where refine beats LoFTR.

Usage:
    python exp_quick.py                  # run all CONFIGS
    python exp_quick.py baseline lr_t1   # run only these configs
    MAX_FRAMES=15 python exp_quick.py    # fewer frames per sequence
"""

import dataclasses
import logging
import os
import sys

import numpy as np

from loftr_wrapper import LoftrRunner
from main import DATASET_ROOT, track_sequence
from src.photometric import RefineConfig

logging.basicConfig(
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.WARNING,  # quiet; we print our own table
)

# (split, reflectivity, sequence) — diffuse case (reflectivity 0.0, alpha 1.0)
SEQUENCES = [
    ("cube", "0.0", "cracker_box_yalehand0"),
    ("cube", "0.0", "tomato_soup_can_yalehand0"),
    ("cube", "0.0", "mustard0"),
    ("cube", "0.0", "sugar_box_yalehand0"),
]

MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "20"))

# ── config variants to compare ───────────────────────────────────────────────
import dataclasses as _dc


def _cfg(**kw) -> RefineConfig:
    return _dc.replace(RefineConfig(), **kw)


CONFIGS: dict[str, RefineConfig] = {
    # no-op refine == pure-LoFTR trajectory (the true reference to beat)
    "noop": _cfg(lr_rot=0.0, lr_trans=0.0),
    # current defaults (3-level pyramid 0.25/0.5/1.0, finest-level cosine decay)
    "baseline": _cfg(),
    # pyramid choices: drop the harmful coarsest 0.25 level
    "two_level": _cfg(scales=(0.5, 1.0), blur_sigmas=(2.0, 1.5)),
    "one_level": _cfg(scales=(1.0,), blur_sigmas=(1.5,)),
    # two-level + gentler LR (smoother settle)
    "two_lvl_lr2e3": _cfg(scales=(0.5, 1.0), blur_sigmas=(2.0, 1.5), lr_rot=2e-3),
}


def run_config(name: str, cfg: RefineConfig, loftr: LoftrRunner) -> None:
    rng = np.random.default_rng(seed=42)
    os.makedirs("exp_quick_results", exist_ok=True)
    all_c: list[tuple[float, float]] = []
    all_r: list[tuple[float, float]] = []
    per_seq: list[str] = []

    for split, refl, seq in SEQUENCES:
        seq_path = f"{DATASET_ROOT}/{split}_{refl}/{seq}"
        if not os.path.isdir(seq_path):
            logging.warning("missing %s", seq_path)
            continue
        cache_dir = f"cache/diffuse/gt/{split}_{refl}/{seq}"
        c, r = track_sequence(
            seq_path=seq_path,
            results_dir="exp_quick_results",
            cache_dir=cache_dir,
            sequence_name=seq,
            alpha=1.0 - float(refl),
            depth_source="gt",
            loftr=loftr,
            rng=rng,
            separate=True,
            refine=True,
            viewer=None,
            max_frames=MAX_FRAMES,
            refine_cfg=cfg,
        )
        all_c += c
        all_r += r
        if r:
            ca, ra = np.array(c), np.array(r)
            per_seq.append(
                f"    {seq:30s} loftr {ca[:,0].mean():5.2f}°/{ca[:,1].mean()*1000:5.2f}mm"
                f"  refined {ra[:,0].mean():5.2f}°/{ra[:,1].mean()*1000:5.2f}mm"
            )

    c, r = np.array(all_c), np.array(all_r)
    win_r = (r[:, 0] < c[:, 0]).mean() * 100
    win_t = (r[:, 1] < c[:, 1]).mean() * 100
    print(f"\n### CONFIG: {name}")
    for line in per_seq:
        print(line)
    print(
        f"  >> AGG  loftr {c[:,0].mean():5.2f}°/{c[:,1].mean()*1000:5.2f}mm"
        f"   refined {r[:,0].mean():5.2f}°/{r[:,1].mean()*1000:5.2f}mm"
        f"   | rot wins {win_r:4.0f}%  trans wins {win_t:4.0f}%",
        flush=True,
    )


def main() -> None:
    only = set(sys.argv[1:])
    loftr = LoftrRunner()
    for name, cfg in CONFIGS.items():
        if only and name not in only:
            continue
        print(f"\n{'='*90}\n{name}: {dataclasses.asdict(cfg)}\n{'='*90}", flush=True)
        run_config(name, cfg, loftr)


if __name__ == "__main__":
    main()
