"""Light-field plane-sweep depth estimation for the ReLiFT-6DoF tracker.

A feed-forward (non-trainable) classical depth estimator tailored to *this*
dataset's light field, designed to replace the noisy RealSense-style stereo
depth (``depth_synth``, produced by ``ycbv-eoat-lf/realsense.py``) with a dense,
smooth, hole-free metric depth map.

Why a plane sweep (and why it is cheap here)
--------------------------------------------
The capture is an ideal rectified light field: a 5x5 grid of pinhole views that
are **coplanar and share a single rotation** (verified: ``max rot diff == 0``),
spaced 5 mm apart (2 cm aperture), all looking down +z.  For such a rig, warping
an off-center view ``(s,t)`` onto the central view under a fronto-parallel depth
plane is a *global image translation* — no per-pixel homography:

    a world point at depth ``Z`` (inverse depth ``rho = 1/Z``) seen by the
    central camera at pixel ``(u, v)`` appears in view ``(s,t)`` (relative
    centre ``t_rel = (tx, ty, tz≈0)``) at

        u' = u - fx * tx * rho ,   v' = v - fy * ty * rho .

So the whole plane sweep is: for each inverse-depth hypothesis ``rho``, shift
every view by ``(-fx*tx*rho, -fy*ty*rho)`` px, measure photo-consistency against
the centre, and aggregate.  This is the epipolar-geometry / EPI cue (all views
agree along the correct slope) realised as a shift-and-compare cost volume, and
it runs in a couple of seconds on a GPU.

Robustness to reflections
-------------------------
A mirror cube violates photo-consistency: the reflection is view-dependent, so a
naive multi-view mean is dominated by the (disagreeing) specular views — exactly
why ``depth_synth`` collapses on ``cube_1.0`` (21% coverage, ~1 m error).  We
aggregate the per-view costs with a **robust truncated mean** (average of the
best-agreeing fraction of views), which rejects the specular/occluded minority
and locks onto the consistent diffuse structure.

Pipeline (all GPU, all ground-truth-free)
-----------------------------------------
1. Per-view features: sRGB luminance + gradient magnitude (gradient matching is
   robust to residual cross-view shading differences; sRGB tames bright specular
   highlights).
2. Cost volume over ``n_planes`` inverse-depth hypotheses: global-shift warp,
   patch-aggregated abs-diff, robust cross-view truncated mean.
3. Winner-take-all + parabolic sub-pixel interpolation -> metric depth.  No scale
   alignment is ever done: the baselines are calibrated, so the depth is metric.
4. Confidence from peak sharpness + match cost.
5. Confidence-weighted, edge-aware least-squares smoothing (conjugate gradient on
   inverse depth, guided by central-view edges) -> dense, smooth, hole-free.

Nothing here reads the ground-truth depth; ``--mode estimate`` additionally
*scores* the result against GT and against ``depth_synth`` so we can show it wins.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F

from src.dataset import LFDataset
from utils import linear_to_srgb


# ──────────────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class LFDepthConfig:
    # Inverse-depth sweep range (metres).  Fixed constants (GT-free): objects live
    # at ~0.5-1.0 m, background out to ~7.5 m.  Uniform-in-inverse-depth sampling
    # concentrates planes on the near (object) range where disparity — and hence
    # matchability — is largest.
    z_min: float = 0.30
    z_max: float = 6.0
    n_planes: int = 192

    # Matching cost.  Gradient matching is weighted heavily over raw intensity:
    # it is invariant to the residual cross-view shading differences these glossy
    # surfaces show, which intensity matching would (wrongly) penalise.
    patch: int = 7            # spatial aggregation window (odd)
    grad_weight: float = 8.0  # weight of gradient-difference vs intensity-difference
    robust_frac: float = 0.6  # fraction of best-agreeing views kept in the mean
    robust_min_views: int = 4

    # Confidence (scale-free, robust to n_planes): a cost *margin* (how much the
    # best plane beats the typical plane) gated by a *texture* term (matching is
    # only trustworthy where the reference has gradient).  Low confidence regions
    # are filled by the edge-aware smoothing from their confident neighbours.
    tex_scale: float = 0.05   # reference gradient magnitude that maps to texture=1
    conf_gamma: float = 1.0   # raises data confidence to this power before smoothing

    # Edge-aware least-squares smoothing (on inverse depth).
    smooth_lambda: float = 4.0   # smoothness vs data balance
    edge_sigma: float = 0.06     # guide-gradient scale (sRGB luminance in [0,1])
    edge_min_w: float = 0.02     # floor on edge weight (keeps the system connected)
    cg_iters: int = 160          # CG iterations (mirror fill over big holes needs ~150)

    # Mirror handling (requires an object mask).  A flat-faced mirror produces a
    # multiview-consistent *virtual* image behind the surface, so the plane sweep
    # confidently matches the reflected environment at far depth.  But the object
    # physically occludes its background, so its true surface is the NEAR depth
    # mode inside the mask: we reject masked matches farther than the near surface
    # by a generous object-size margin and let the membrane fill from the near
    # (higher-confidence) anchors + an image-edge-free interior.  No-op on diffuse/
    # textured objects (their masked depth is already a single near mode).
    mirror_reject: bool = True
    mirror_near_pct: float = 0.10  # quantile of masked depth taken as the near surface
    mirror_margin: float = 0.30    # metres kept beyond the near surface (object extent)
    # The rejected region is filled as a harmonic membrane anchored by the real
    # near-surface pixels around it (and initialised at the near surface so CG
    # converges from the right basin).  A nonzero prior would instead flatten the
    # fill toward a single near plane and wash out the true relief, so keep it 0.
    mirror_prior_w: float = 0.0    # soft data weight pulling the fill toward the near surface


# ──────────────────────────────────────────────────────────────────────────────
#  Small image ops
# ──────────────────────────────────────────────────────────────────────────────
def _srgb_luma(lf_linear: torch.Tensor) -> torch.Tensor:
    """[..., H, W, 3] linear RGB -> [..., H, W] sRGB luminance in [0, 1]."""
    srgb = linear_to_srgb(lf_linear.clamp(0, 1))
    r, g, b = srgb[..., 0], srgb[..., 1], srgb[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).clamp(0, 1)


def _grad_mag(gray: torch.Tensor) -> torch.Tensor:
    """Sobel gradient magnitude of [N, H, W] -> [N, H, W]."""
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=gray.dtype, device=gray.device
    ).view(1, 1, 3, 3) / 8.0
    ky = kx.transpose(2, 3)
    g = gray.unsqueeze(1)
    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)[:, 0]


def _box(x: torch.Tensor, k: int) -> torch.Tensor:
    """Box/patch average over the last two dims, window k (odd), same size."""
    if k <= 1:
        return x
    nd = x.dim()
    x4 = x.view(-1, 1, x.shape[-2], x.shape[-1])
    out = F.avg_pool2d(x4, k, stride=1, padding=k // 2)
    return out.view(*x.shape[:-2], out.shape[-2], out.shape[-1]) if nd > 2 else out[:, 0]


# ──────────────────────────────────────────────────────────────────────────────
#  Plane-sweep depth estimator
# ──────────────────────────────────────────────────────────────────────────────
class LFPlaneSweepDepth:
    def __init__(self, cfg: LFDepthConfig | None = None, device: str = "cuda"):
        self.cfg = cfg or LFDepthConfig()
        self.device = torch.device(device)

    # ---- geometry -----------------------------------------------------------
    @staticmethod
    def _view_offsets(frame: dict):
        """Per-view (tx, ty) baseline in metres relative to the central view, plus
        the central flat index.  Uses ``camera_poses_rel`` (view->central), whose
        translation column is the relative camera centre (rotation is identity)."""
        S, T = frame["LF"].shape[0], frame["LF"].shape[1]
        cmid = (S // 2) * T + (T // 2)
        rel = frame["camera_poses_rel"].reshape(S * T, 4, 4)
        centres = rel[:, :3, 3]  # [N, 3] = (tx, ty, tz) of each view in central frame
        return centres, cmid

    # ---- core ---------------------------------------------------------------
    @torch.no_grad()
    def estimate(self, frame: dict, mask: torch.Tensor | None = None) -> dict:
        """Estimate central-view depth from the light field.

        ``mask`` (central-view object mask) is optional and used only for mirror
        handling (the occlusion constraint); the photo-consistency core ignores it.
        """
        cfg = self.cfg
        if mask is not None:
            mask = mask.to(self.device).bool()
        LF = frame["LF"].to(self.device)            # [S, T, H, W, 3] linear
        S, T, H, W, _ = LF.shape
        N = S * T
        K = frame["camera_matrix"].to(self.device)
        fx, fy = float(K[0, 0]), float(K[1, 1])

        centres, cmid = self._view_offsets(frame)
        centres = centres.to(self.device)

        # Per-view features: [N, 2, H, W] = (luma, grad-magnitude).
        luma = _srgb_luma(LF.reshape(N, H, W, 3))   # [N, H, W]
        grad = _grad_mag(luma)
        feat = torch.stack([luma, cfg.grad_weight * grad], dim=1)  # [N, 2, H, W]

        # Reference (central) vs the other 24 source views.
        src_idx = [i for i in range(N) if i != cmid]
        src_feat = feat[src_idx]                     # [V, 2, H, W]
        ref_feat = feat[cmid : cmid + 1]             # [1, 2, H, W]
        tx = centres[src_idx, 0]                     # [V]
        ty = centres[src_idx, 1]
        V = len(src_idx)

        # Base sampling grid (normalised, align_corners=True) + a validity channel
        # appended to the features so out-of-frame samples are detected per warp.
        ys, xs = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        base_gx = 2.0 * xs / (W - 1) - 1.0
        base_gy = 2.0 * ys / (H - 1) - 1.0
        src_aug = torch.cat([src_feat, torch.ones(V, 1, H, W, device=self.device)], 1)

        # Uniform-in-inverse-depth planes == uniform disparity == uniform matching
        # resolution, which concentrates planes on the near (object) range.
        rhos = torch.linspace(
            1.0 / cfg.z_max, 1.0 / cfg.z_min, cfg.n_planes, device=self.device
        )                                          # ascending inverse depth (far -> near)
        ref_l, ref_g = ref_feat[0, 0], ref_feat[0, 1]

        P = cfg.n_planes
        cost_vol = torch.empty(P, H, W, device=self.device)
        keep = max(cfg.robust_min_views, int(round(cfg.robust_frac * V)))
        rank = torch.arange(V, device=self.device).view(V, 1, 1)

        for p in range(P):
            rho = rhos[p]
            # Per-view normalised pixel offset for this plane (global translation).
            ox = (-fx * tx * rho) * (2.0 / (W - 1))   # [V]
            oy = (-fy * ty * rho) * (2.0 / (H - 1))
            gx = base_gx.unsqueeze(0) + ox.view(V, 1, 1)
            gy = base_gy.unsqueeze(0) + oy.view(V, 1, 1)
            grid = torch.stack([gx, gy], dim=-1)      # [V, H, W, 2]
            warp = F.grid_sample(
                src_aug, grid, mode="bilinear", padding_mode="zeros", align_corners=True
            )                                          # [V, 3, H, W]
            valid = warp[:, 2] > 0.999
            cost_v = (warp[:, 0] - ref_l).abs() + (warp[:, 1] - ref_g).abs()  # [V, H, W]
            cost_v = _box(cost_v, cfg.patch)
            cost_v = torch.where(valid, cost_v, cost_v.new_full((), 1e6))
            # Robust aggregation: mean of the ``keep`` best-agreeing views.
            sc, _ = torch.sort(cost_v, dim=0)
            m = (rank < keep).float()
            cost_vol[p] = (sc * m).sum(0) / m.sum(0)

        return self._finalise(cost_vol, rhos, ref_l, mask)

    def _finalise(self, cost_vol, rhos, ref_luma, mask):
        cfg = self.cfg
        P = cost_vol.shape[0]
        minc, arg = cost_vol.min(0)                  # [H, W]

        # Parabolic sub-pixel interpolation in plane index.
        am1 = (arg - 1).clamp(0, P - 1)
        ap1 = (arg + 1).clamp(0, P - 1)
        c0 = cost_vol.gather(0, am1[None])[0]
        c1 = minc
        c2 = cost_vol.gather(0, ap1[None])[0]
        denom = (c0 - 2 * c1 + c2)
        delta = torch.where(denom.abs() > 1e-9, 0.5 * (c0 - c2) / denom, torch.zeros_like(denom))
        delta = delta.clamp(-1, 1)

        rho_at = rhos[arg]
        rho_lo = rhos[(arg - 1).clamp(0, P - 1)]
        rho_hi = rhos[(arg + 1).clamp(0, P - 1)]
        # local spacing on each side (sweep is non-uniform in inverse depth)
        step = torch.where(delta >= 0, rho_hi - rho_at, rho_at - rho_lo)
        rho_data = (rho_at + delta * step).clamp(rhos.min(), rhos.max())

        # Confidence = cost margin x texture.  Margin: how much the winning plane
        # beats the typical (median) plane — scale-free and robust to n_planes,
        # high only where the cost curve has a clear minimum.  Texture: matching is
        # meaningless without reference gradient, so gate by it (this also makes the
        # smoothing fill flat regions from their textured boundaries).
        cmed = cost_vol.median(0).values
        margin = ((cmed - minc) / (cmed + 1e-6)).clamp(0, 1)
        ref_grad = _grad_mag(ref_luma[None])[0]
        tex = (ref_grad / cfg.tex_scale).clamp(0, 1)
        conf = (margin * tex).clamp(0, 1) ** cfg.conf_gamma

        # Mirror handling: inside the object mask, reject matches that fall beyond
        # the near surface (the reflected virtual image), then fill smoothly.  The
        # rejected pixels drop their (wrong, far) data term (conf -> mirror_prior_w,
        # 0 by default) and are *seeded* at the near surface so the conjugate-gradient
        # membrane — anchored by the surrounding real near-surface matches — converges
        # to the surface instead of getting stuck near the far reflection.
        fill_region = None
        rho_solve, conf_solve = rho_data, conf
        depth_raw_m = 1.0 / rho_data.clamp_min(1e-6)
        if mask is not None and cfg.mirror_reject and mask.any():
            md = depth_raw_m[mask]
            z_near = torch.quantile(md, cfg.mirror_near_pct)
            fill_region = mask & (depth_raw_m > (z_near + cfg.mirror_margin))
            rho_solve = torch.where(fill_region, (1.0 / z_near), rho_data)
            conf_solve = torch.where(
                fill_region, conf.new_full((), cfg.mirror_prior_w), conf
            )

        # Edge-aware least-squares smoothing on inverse depth.
        rho_smooth = self._edge_aware_solve(
            rho_solve, conf_solve, ref_luma, mask, fill_region
        )
        depth = (1.0 / rho_smooth.clamp_min(1e-6)).clamp(cfg.z_min, cfg.z_max)
        depth_raw = (1.0 / rho_data.clamp_min(1e-6)).clamp(cfg.z_min, cfg.z_max)
        return {
            "depth": depth,          # [H, W] dense smooth metric depth (metres)
            "depth_raw": depth_raw,  # [H, W] winner-take-all depth before smoothing
            "conf": conf,            # [H, W] in [0, 1]
        }

    def _edge_aware_solve(self, rho_data, conf, guide, mask=None, fill_region=None):
        """Minimise  sum c (x - rho)^2  +  lam sum w_ij (x_i - x_j)^2  over x,
        with per-edge weights ``w`` from central-view luminance gradients.  Solved
        with conjugate gradient (matrix-free).  Fills holes (c≈0) by diffusion from
        confident neighbours and smooths the surface.

        With ``mask``/``fill_region`` set (mirror handling), edges are overridden:
        the object/background boundary blocks diffusion (so the filled surface does
        not bleed into the far background), while edges *inside* the rejected mirror
        region are made fully smooth (so the membrane interpolates across the
        reflection's texture instead of getting stuck on its false edges)."""
        cfg = self.cfg
        g = guide
        wh = torch.exp(-(g[:, 1:] - g[:, :-1]).abs() / cfg.edge_sigma).clamp_min(cfg.edge_min_w)
        wv = torch.exp(-(g[1:, :] - g[:-1, :]).abs() / cfg.edge_sigma).clamp_min(cfg.edge_min_w)
        if fill_region is not None:
            fh = fill_region[:, 1:] | fill_region[:, :-1]
            fv = fill_region[1:, :] | fill_region[:-1, :]
            wh = torch.where(fh, torch.ones_like(wh), wh)
            wv = torch.where(fv, torch.ones_like(wv), wv)
        if mask is not None:
            bh = mask[:, 1:] ^ mask[:, :-1]   # object/background boundary edges
            bv = mask[1:, :] ^ mask[:-1, :]
            wh = torch.where(bh, wh.new_full((), cfg.edge_min_w), wh)
            wv = torch.where(bv, wv.new_full((), cfg.edge_min_w), wv)
        lam = cfg.smooth_lambda
        c = (conf ** 1.0)

        def laplacian(x):
            out = torch.zeros_like(x)
            dh = (x[:, 1:] - x[:, :-1]) * wh
            out[:, :-1] -= dh
            out[:, 1:] += dh
            dv = (x[1:, :] - x[:-1, :]) * wv
            out[:-1, :] -= dv
            out[1:, :] += dv
            return out

        def amul(x):
            return c * x + lam * laplacian(x)

        b = c * rho_data
        x = rho_data.clone()
        r = b - amul(x)
        p = r.clone()
        rs = (r * r).sum()
        for _ in range(cfg.cg_iters):
            Ap = amul(p)
            alpha = rs / (p * Ap).sum().clamp_min(1e-20)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = (r * r).sum()
            if rs_new < 1e-12:
                break
            p = r + (rs_new / rs) * p
            rs = rs_new
        return x


# ──────────────────────────────────────────────────────────────────────────────
#  Evaluation helpers (score vs GT and vs depth_synth — GT used only for scoring)
# ──────────────────────────────────────────────────────────────────────────────
def _central_mask(frame: dict) -> torch.Tensor:
    LF = frame["LF"]
    S, T, H, W = LF.shape[0], LF.shape[1], LF.shape[2], LF.shape[3]
    cmid = (S // 2) * T + (T // 2)
    masks = frame["masks"]
    if masks is None:
        return torch.ones(H, W, dtype=torch.bool, device=LF.device)
    return masks.reshape(-1, H, W)[cmid].bool()


def _metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict:
    """Depth error stats over ``valid`` pixels (all in metres -> reported in mm)."""
    v = valid & (gt > 1e-3) & (pred > 1e-3)
    n = int(v.sum())
    if n < 10:
        return {"n": n, "mae_mm": float("nan"), "rmse_mm": float("nan"),
                "med_mm": float("nan"), "d5": float("nan"), "d10": float("nan")}
    e = np.abs(pred[v] - gt[v])
    return {
        "n": n,
        "mae_mm": float(e.mean() * 1000),
        "rmse_mm": float(np.sqrt((e ** 2).mean()) * 1000),
        "med_mm": float(np.median(e) * 1000),
        "d5": float((e < 0.005).mean()),
        "d10": float((e < 0.010).mean()),
    }


def run_sequence(estimator: LFPlaneSweepDepth, seq_path: str, out_path: str,
                 limit: int | None = None) -> dict:
    """Estimate LF depth for a whole sequence, cache to ``out_path`` and score it
    against GT and against ``depth_synth`` on the object mask + full frame."""
    from PIL import Image
    ds_gt = LFDataset(seq_path, depth_source="gt")
    # Read synth depth straight off disk (same sorted-filename indexing the dataset
    # uses) to avoid re-decoding all 25 light-field views just to fetch its depth.
    sy_dir = os.path.join(seq_path, "depth_synth")
    sy_fnames = sorted(os.listdir(sy_dir))

    def _synth_depth(i):
        return np.array(Image.open(os.path.join(sy_dir, sy_fnames[i]))).astype(np.float32) / 1000.0

    n = len(ds_gt) if limit is None else min(limit, len(ds_gt))

    depth_l, conf_l, gt_l, sy_l, mask_l, rgb_l = [], [], [], [], [], []
    K_np = None
    obj_ours, obj_synth, full_ours, h2h_ours = [], [], [], []
    cov_ours, cov_synth = [], []
    for idx in range(n):
        frame = ds_gt[idx]
        H, W = frame["LF"].shape[2], frame["LF"].shape[3]
        if K_np is None:
            K_np = frame["camera_matrix"][:3, :3].cpu().numpy()
        mask = _central_mask(frame)
        out = estimator.estimate(frame, mask=mask)
        depth = out["depth"]
        gt = frame["depth"]

        d_np = depth.cpu().numpy()
        gt_np = gt.cpu().numpy()
        sy_np = _synth_depth(idx)
        m_np = mask.cpu().numpy()

        obj_ours.append(_metrics(d_np, gt_np, m_np))
        obj_synth.append(_metrics(sy_np, gt_np, m_np))
        full_ours.append(_metrics(d_np, gt_np, np.ones_like(m_np)))
        # Head-to-head: ours scored on exactly the pixels where synth is valid, so
        # the comparison is on equal footing (synth is sparse; we are dense).
        h2h_ours.append(_metrics(d_np, gt_np, m_np & (sy_np > 1e-3)))
        cov_ours.append(float(((d_np > 1e-3) & m_np).sum() / max(m_np.sum(), 1)))
        cov_synth.append(float(((sy_np > 1e-3) & m_np).sum() / max(m_np.sum(), 1)))

        S, T = frame["LF"].shape[0], frame["LF"].shape[1]
        cmid = (S // 2) * T + (T // 2)
        depth_l.append(depth.cpu().numpy().astype(np.float16))
        conf_l.append(out["conf"].cpu().numpy().astype(np.float16))
        gt_l.append(gt_np.astype(np.float16))
        sy_l.append(sy_np.astype(np.float16))
        mask_l.append(m_np)
        rgb_l.append(
            (linear_to_srgb(frame["LF"].reshape(-1, H, W, 3)[cmid]).clamp(0, 1) * 255)
            .to(torch.uint8).cpu().numpy()
        )

    def _mean(ms, key):
        vals = [m[key] for m in ms if np.isfinite(m[key])]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "seq": seq_path, "frames": n,
        "ours_obj_mae_mm": _mean(obj_ours, "mae_mm"),
        "ours_obj_med_mm": _mean(obj_ours, "med_mm"),
        "ours_obj_d10": _mean(obj_ours, "d10"),
        "ours_obj_cov": float(np.mean(cov_ours)),
        "ours_h2h_mae_mm": _mean(h2h_ours, "mae_mm"),  # ours on synth-valid pixels
        "synth_obj_mae_mm": _mean(obj_synth, "mae_mm"),
        "synth_obj_med_mm": _mean(obj_synth, "med_mm"),
        "synth_obj_cov": float(np.mean(cov_synth)),
        "ours_full_mae_mm": _mean(full_ours, "mae_mm"),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        depth=np.stack(depth_l), conf=np.stack(conf_l),
        gt=np.stack(gt_l), synth=np.stack(sy_l),
        mask=np.stack(mask_l), rgb=np.stack(rgb_l), K=K_np,
        obj_metrics=json.dumps(obj_ours), synth_metrics=json.dumps(obj_synth),
        summary=json.dumps(summary), cfg=json.dumps(asdict(estimator.cfg)),
    )
    return summary


def write_sequence_depth(estimator: LFPlaneSweepDepth, seq_path: str,
                         limit: int | None = None) -> int:
    """Estimate LF depth for every frame and write it into ``<seq>/depth_lf`` as
    uint16 millimetres, mirroring the ``depth``/``depth_synth`` layout (same
    filenames).  ``main.py`` then consumes it via ``LFDataset(..., depth_source='lf')``."""
    from PIL import Image
    ds = LFDataset(seq_path, depth_source="gt")
    out_dir = os.path.join(seq_path, "depth_lf")
    os.makedirs(out_dir, exist_ok=True)
    n = len(ds) if limit is None else min(limit, len(ds))
    for idx in range(n):
        frame = ds[idx]
        mask = _central_mask(frame)
        depth = estimator.estimate(frame, mask=mask)["depth"]
        mm = (depth.clamp(0, 65.535).cpu().numpy() * 1000.0).round().astype(np.uint16)
        Image.fromarray(mm).save(os.path.join(out_dir, ds.depth_fnames[idx]))
    return n


def _iter_sequences(dataset_root, splits, refls, seq_filter=None):
    for split in splits:
        for refl in refls:
            split_dir = os.path.join(dataset_root, f"{split}_{refl}")
            if not os.path.isdir(split_dir):
                continue
            for seq in sorted(os.listdir(split_dir)):
                seq_path = os.path.join(split_dir, seq)
                if not os.path.isdir(seq_path) or seq == "models":
                    continue
                if seq_filter and seq not in seq_filter:
                    continue
                yield split, refl, seq, seq_path


# ──────────────────────────────────────────────────────────────────────────────
#  Viser visualisation (GT vs ours vs synth point clouds)
# ──────────────────────────────────────────────────────────────────────────────
def _turbo(x: np.ndarray) -> np.ndarray:
    try:
        import matplotlib.cm as cm
        return (cm.get_cmap("turbo")(np.clip(x, 0, 1))[..., :3] * 255).astype(np.uint8)
    except Exception:
        x = np.clip(x, 0, 1)[..., None]
        c = np.array([0.0, 0.0, 1.0]) * (1 - x) + np.array([1.0, 0.0, 0.0]) * x
        return (c * 255).astype(np.uint8)


def _backproject(depth, mask, K):
    H, W = depth.shape
    ys, xs = np.where(mask & (depth > 1e-3))
    z = depth[ys, xs].astype(np.float64)
    x = (xs - K[0, 2]) / K[0, 0] * z
    y = (ys - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], 1).astype(np.float32), ys, xs


def visualize(results_dir: str, err_cap_mm: float = 20.0, port: int = 8080):
    import viser
    files = sorted(os.path.join(r, f) for r, _, fs in os.walk(results_dir)
                   for f in fs if f.endswith(".npz"))
    if not files:
        raise FileNotFoundError(f"No .npz under {results_dir}")
    labels = [os.path.relpath(f, results_dir)[:-4] for f in files]
    cache: dict = {}

    def load(label):
        if label not in cache:
            d = np.load(files[labels.index(label)], allow_pickle=True)
            cache[label] = {k: d[k] for k in d.files}
        return cache[label]

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("-y")
    obj_dd = server.gui.add_dropdown("Object", labels, initial_value=labels[0])
    d0 = load(labels[0])
    frame_sl = server.gui.add_slider("Frame", 0, len(d0["gt"]) - 1, 1, 0)
    show_gt = server.gui.add_checkbox("GT (green)", True)
    show_ours = server.gui.add_checkbox("Ours", True)
    show_synth = server.gui.add_checkbox("Synth (red)", False)
    color_err = server.gui.add_checkbox("Colour ours by error", True)
    obj_only = server.gui.add_checkbox("Object mask only", True)
    pt = server.gui.add_slider("Point size", 0.0005, 0.01, 0.0005, 0.002)
    info = server.gui.add_text("info", initial_value="")

    def render():
        d = load(obj_dd.value)
        f = min(int(frame_sl.value), len(d["gt"]) - 1)
        K = d["K"]
        mask = d["mask"][f] if obj_only.value else np.ones_like(d["mask"][f])
        gt = d["gt"][f].astype(np.float32)
        ours = d["depth"][f].astype(np.float32)
        synth = d["synth"][f].astype(np.float32)
        rgb = d["rgb"][f]
        server.scene.reset()
        if show_gt.value:
            gp, gy, gx = _backproject(gt, mask, K)
            if len(gp):
                server.scene.add_point_cloud("/gt", gp,
                    np.tile(np.array([[40, 200, 80]], np.uint8), (len(gp), 1)),
                    point_size=float(pt.value))
        if show_ours.value:
            op, oy, ox = _backproject(ours, mask, K)
            if len(op):
                col = _turbo(np.abs(ours[oy, ox] - gt[oy, ox]) * 1000 / err_cap_mm) \
                    if color_err.value else rgb[oy, ox]
                server.scene.add_point_cloud("/ours", op, col, point_size=float(pt.value))
        if show_synth.value:
            sp, sy, sx = _backproject(synth, mask, K)
            if len(sp):
                server.scene.add_point_cloud("/synth", sp,
                    np.tile(np.array([[220, 60, 60]], np.uint8), (len(sp), 1)),
                    point_size=float(pt.value))
        s = json.loads(str(d["summary"]))
        info.value = (f"obj MAE  ours {s['ours_obj_mae_mm']:.1f}mm (cov {s['ours_obj_cov']*100:.0f}%)"
                      f"  vs synth {s['synth_obj_mae_mm']:.1f}mm (cov {s['synth_obj_cov']*100:.0f}%)")

    @obj_dd.on_update
    def _(_):
        d = load(obj_dd.value)
        frame_sl.max = len(d["gt"]) - 1
        frame_sl.value = 0
        render()

    for ctrl in (frame_sl, show_gt, show_ours, show_synth, color_err, obj_only, pt):
        ctrl.on_update(lambda _: render())
    render()
    print(f"viser at http://localhost:{port}  ({len(labels)} objects)")
    while True:
        time.sleep(1.0)


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Light-field plane-sweep depth estimation")
    ap.add_argument("--mode", choices=["estimate", "vis", "write"], default="estimate",
                    help="estimate: score vs GT/synth into --out .npz; "
                         "write: save depth_lf/*.png into each sequence for main.py; "
                         "vis: viser comparison")
    ap.add_argument("--dataset-root", default="/home/ngoncharov/SpecTrack_dataset")
    ap.add_argument("--out", default="eval/depth_lf")
    ap.add_argument("--splits", default="cube,objects")
    ap.add_argument("--refls", default="0.0,0.5,0.7,1.0")
    ap.add_argument("--seqs", default="")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.mode == "vis":
        visualize(args.out, port=args.port)
        return

    splits = [s for s in args.splits.split(",") if s]
    refls = [r for r in args.refls.split(",") if r]
    seq_filter = set(s for s in args.seqs.split(",") if s) or None

    estimator = LFPlaneSweepDepth()

    if args.mode == "write":
        for split, refl, seq, seq_path in _iter_sequences(args.dataset_root, splits, refls, seq_filter):
            out_dir = os.path.join(seq_path, "depth_lf")
            # Resumable: skip only a COMPLETE depth_lf (file count matches the GT depth
            # folder), so a sequence interrupted mid-write is regenerated, not left
            # partial (a partial folder would mis-index in LFDataset).
            n_expected = len(os.listdir(os.path.join(seq_path, "depth")))
            if (args.limit is None and os.path.isdir(out_dir)
                    and len(os.listdir(out_dir)) >= n_expected and not args.overwrite):
                print(f"skip (complete) {split}_{refl}/{seq}", flush=True)
                continue
            t = time.time()
            nf = write_sequence_depth(estimator, seq_path, limit=args.limit)
            print(f"wrote {split}_{refl}/{seq}: {nf}f depth_lf ({time.time()-t:.0f}s)", flush=True)
        return

    summaries = []
    for split, refl, seq, seq_path in _iter_sequences(args.dataset_root, splits, refls, seq_filter):
        out_path = os.path.join(args.out, f"{split}_{refl}", f"{seq}.npz")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"skip (cached) {split}_{refl}/{seq}")
            continue
        t = time.time()
        try:
            s = run_sequence(estimator, seq_path, out_path, limit=args.limit)
        except RuntimeError as e:
            torch.cuda.empty_cache()
            print(f"FAILED {split}_{refl}/{seq}: {str(e)[:140]}")
            continue
        s.update(split=split, refl=refl, seq=seq)
        summaries.append(s)
        win = "WIN " if (np.isnan(s["synth_obj_mae_mm"]) or s["ours_obj_mae_mm"] <= s["synth_obj_mae_mm"]) else "lose"
        print(f"[{win}] {split}_{refl}/{seq:28s} ours {s['ours_obj_mae_mm']:6.1f}mm "
              f"(h2h {s['ours_h2h_mae_mm']:5.1f}, cov {s['ours_obj_cov']*100:3.0f}%)  vs "
              f"synth {s['synth_obj_mae_mm']:7.1f}mm (cov {s['synth_obj_cov']*100:3.0f}%)"
              f"   ({time.time()-t:.0f}s, {s['frames']}f)")

    if summaries:
        om = [s["ours_obj_mae_mm"] for s in summaries if np.isfinite(s["ours_obj_mae_mm"])]
        sm = [s["synth_obj_mae_mm"] for s in summaries if np.isfinite(s["synth_obj_mae_mm"])]
        wins = sum(1 for s in summaries
                   if np.isfinite(s["ours_obj_mae_mm"]) and
                   (np.isnan(s["synth_obj_mae_mm"]) or s["ours_obj_mae_mm"] <= s["synth_obj_mae_mm"]))
        print(f"\n=== {len(summaries)} seqs | ours obj MAE {np.mean(om):.1f}mm "
              f"vs synth {np.mean(sm):.1f}mm | ours wins {wins}/{len(summaries)} ===")
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(summaries, f, indent=2)


if __name__ == "__main__":
    main()
