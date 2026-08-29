"""Load config.yaml and expose typed configs used across the pipeline.

If the CONFIG_OVERRIDES env var names a second yaml (e.g. config_lift.yaml,
set by main_lift.py), its keys are deep-merged over config.yaml — so each
dataset can carry its own tuned hyperparameters without forking the base
config. main.py never sets it, keeping the synthetic pipeline untouched.
"""
import os
import yaml
from src.photometric import RefineConfig
from lf_depth import LFDepthConfig

_here = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_here, "config.yaml")) as _f:
    _C = yaml.safe_load(_f)


def _deep_merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


_overrides = os.environ.get("CONFIG_OVERRIDES")
if _overrides:
    _ov_path = (
        _overrides if os.path.isabs(_overrides) else os.path.join(_here, _overrides)
    )
    if os.path.exists(_ov_path):
        with open(_ov_path) as _f:
            _deep_merge(_C, yaml.safe_load(_f) or {})

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
    refine_alpha_max=float(_r.get("refine_alpha_max", 1.01)),
    scales=tuple(_r["scales"]),
    accept_on_loss=_r["accept_on_loss"],
    max_correction_deg=_r["max_correction_deg"],
    max_correction_trans=_r["max_correction_trans"],
    patience_loss=_r["patience_loss"],
    min_rel_improve=_r["min_rel_improve"],
    update_every=_r["update_every"],
    photo_loss=str(_r.get("photo_loss", "mse")),
    anchor=str(_r.get("anchor", "prev")),
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
# Short-edge target for LoFTR input in the tracking backbone. 400 matches the
# historical loftr_wrapper._RESIZE (synthetic 640x480); the real 1280x720
# capture needs more to keep the object at a useful scale.
LOFTR_RESIZE: int = int(_C["tracking"].get("loftr_resize", 400))
# 3D-3D RANSAC inlier threshold (metres) for the LoFTR relative pose. 0.05
# matches the historical loftr_baseline.INLIER_DIST (BundleSDF's value, sized
# for noisy synthetic depth); clean LF plane-sweep depth supports much tighter.
RANSAC_INLIER_DIST: float = float(_C["tracking"].get("ransac_inlier_dist", 0.05))
# Alpha veto: override a low estimated alpha with a near-diffuse clamp when a
# probe separation shows the reflection model explains almost none of the
# cross-view variance (texture-fooled estimator on real captures).
_av = _C.get("alpha_veto", {})
ALPHA_VETO_ENABLED: bool = bool(_av.get("enabled", False))
ALPHA_VETO_RATIO: float = float(_av.get("ratio_min", 0.90))
ALPHA_VETO_EST_MAX: float = float(_av.get("est_alpha_max", 0.85))
ALPHA_VETO_CLAMP: float = float(_av.get("clamp_alpha", 0.95))
ALPHA_VETO_PROBE: float = float(_av.get("probe_alpha", 0.8))

# 2D-3D PnP polish of the LoFTR relative pose (see src/pose.py).
# true/false, or "auto": enabled per sequence only when the alpha veto fires
# (features proven to be albedo texture, so reprojection is trustworthy).
_pnp = _C["tracking"].get("pnp_refine", False)
PNP_MODE: str = str(_pnp).lower()
PNP_REFINE: bool = _pnp is True
# Keyframe-anchored coarse tracking (see config.yaml tracking section).
TRACK_KEYFRAME: bool = bool(_C["tracking"].get("keyframe", False))
KF_MAX_DEG: float = float(_C["tracking"].get("kf_max_deg", 12.0))
KF_MAX_TRANS: float = float(_C["tracking"].get("kf_max_trans", 0.08))
KF_MIN_INLIERS: int = int(_C["tracking"].get("kf_min_inliers", 60))
KF_GROSS_DEG: float = float(_C["tracking"].get("kf_gross_deg", 30.0))
KF_GROSS_TRANS: float = float(_C["tracking"].get("kf_gross_trans", 0.10))
