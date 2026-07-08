"""Load config.yaml and expose typed configs used across the pipeline."""

import os

import yaml

from lf_depth import LFDepthConfig
from src.photometric import RefineConfig

_here = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_here, "config.yaml")) as _f:
    _C = yaml.safe_load(_f)

SEPARATION_ITERS: int = _C["separation_iters"]
ALPHA_STABLE_TOL: float = _C["alpha_stable_tol"]
REFINE_FEED_FORWARD: bool = _C["refine_feed_forward"]

REFINE_CFG = RefineConfig(**_C["refine"])
LF_DEPTH_CFG = LFDepthConfig(**_C["lf_depth"])

MIN_LOFTR_INLIERS: int = _C["tracking"]["min_loftr_inliers"]
