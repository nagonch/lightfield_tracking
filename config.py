"""Load config.yaml and expose typed configs used across the pipeline."""
import os
import yaml
from src.photometric import RefineConfig
from lf_depth import LFDepthConfig

_here = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_here, "config.yaml")) as _f:
    _C = yaml.safe_load(_f)

# ── paths / pipeline flags ────────────────────────────────────────────────────
DATASET_ROOT: str = _C["dataset_root"]
CACHE_ROOT: str = _C["cache_root"]
SEPARATION_ITERS: int = _C["separation_iters"]
USE_REFLECTION_SEPARATION: bool = _C["use_reflection_separation"]
ENABLE_VIS: bool = _C["enable_vis"]
ALPHA_STABLE_TOL: float = _C["alpha_stable_tol"]
PIN_ALPHA: bool = _C["pin_alpha"]
REFINE_FEED_FORWARD: bool = _C["refine_feed_forward"]
DEPTH_SOURCES: list[str] = list(_C["depth_sources"])
SPLIT_PREFIXES: list[str] = list(_C["split_prefixes"])
REFLECTIVITIES: list[str] = [str(r) for r in _C["reflectivities"]]

# ── photometric refinement ────────────────────────────────────────────────────
_r = _C["refine"]
REFINE_CFG = RefineConfig(
    num_iters=_r["num_iters"],
    lr_rot=_r["lr_rot"],
    lr_trans=_r["lr_trans"],
    cosine_decay=_r["cosine_decay"],
    lambda_depth=_r["lambda_depth"],
    diffuse_mode=_r["diffuse_mode"],
    diffuse_alpha_min=_r["diffuse_alpha_min"],
    scales=tuple(_r["scales"]),
    accept_on_loss=_r["accept_on_loss"],
    max_correction_deg=_r["max_correction_deg"],
    max_correction_trans=_r["max_correction_trans"],
    patience_loss=_r["patience_loss"],
    min_rel_improve=_r["min_rel_improve"],
    update_every=_r["update_every"],
    drift_reset_deg=_r["drift_reset_deg"],
    drift_reset_trans=_r["drift_reset_trans"],
    drift_reset_alpha_min=_r["drift_reset_alpha_min"],
    relight=_r["relight"],
    relight_alpha_max=_r["relight_alpha_max"],
    lr_rot_relight=_r["lr_rot_relight"],
    lr_trans_relight=_r["lr_trans_relight"],
    relight_min_correction_deg=_r["relight_min_correction_deg"],
    relight_feed_forward=_r["relight_feed_forward"],
    relight_conf_floor=_r["relight_conf_floor"],
)

# ── LF plane-sweep depth ──────────────────────────────────────────────────────
_d = _C["lf_depth"]
LF_DEPTH_CFG = LFDepthConfig(
    z_min=_d["z_min"],
    z_max=_d["z_max"],
    n_planes=_d["n_planes"],
    patch=_d["patch"],
    grad_weight=_d["grad_weight"],
    robust_frac=_d["robust_frac"],
    robust_min_views=_d["robust_min_views"],
    tex_scale=_d["tex_scale"],
    conf_gamma=_d["conf_gamma"],
    smooth_lambda=_d["smooth_lambda"],
    edge_sigma=_d["edge_sigma"],
    edge_min_w=_d["edge_min_w"],
    cg_iters=_d["cg_iters"],
    mirror_reject=_d["mirror_reject"],
    mirror_near_pct=_d["mirror_near_pct"],
    mirror_margin=_d["mirror_margin"],
    mirror_prior_w=_d["mirror_prior_w"],
)

# ── surface light field ────────────────────────────────────────────────────────
SLF_FOOTPRINT_SCALE: float = _C["slf"]["footprint_scale"]
SLF_SH_DEGREE: int = _C["slf"]["sh_degree"]

# ── pose tracking ─────────────────────────────────────────────────────────────
MIN_LOFTR_INLIERS: int = _C["tracking"]["min_loftr_inliers"]
