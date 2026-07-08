import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Equirectangular environment-map helpers.
# ---------------------------------------------------------------------------
def dirs_to_equirect_uv(dirs, env_h, env_w, flip_u=True, flip_v=True):
    dirs = F.normalize(dirs, dim=-1)
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]

    lon = torch.atan2(x, z)
    lat = torch.asin(torch.clamp(y, -1.0, 1.0))

    u = (lon / (2.0 * torch.pi) + 0.5) * (env_w - 1)
    if flip_u:
        u = (env_w - 1) - u
    if flip_v:
        v = (0.5 + lat / torch.pi) * (env_h - 1)
    else:
        v = (0.5 - lat / torch.pi) * (env_h - 1)
    return u, v


def sample_environment_map(env_map_hwc, dirs, flip_u=True, flip_v=True):
    """Differentiable bilinear sampling of an equirect env map (horizontal wrap)."""
    env_h, env_w = env_map_hwc.shape[:2]
    out_shape = dirs.shape[:-1]
    dirs_flat = torch.nan_to_num(dirs.reshape(-1, 3))

    u, v = dirs_to_equirect_uv(dirs_flat, env_h, env_w, flip_u=flip_u, flip_v=flip_v)

    u0 = torch.floor(u).long()
    v0 = torch.floor(v).long()
    u1 = u0 + 1
    v1 = v0 + 1

    u0w = torch.remainder(u0, env_w)
    u1w = torch.remainder(u1, env_w)
    v0c = torch.clamp(v0, 0, env_h - 1)
    v1c = torch.clamp(v1, 0, env_h - 1)

    du = (u - u0.to(u.dtype)).unsqueeze(-1)
    dv = (v - v0.to(v.dtype)).unsqueeze(-1)
    w00 = (1.0 - du) * (1.0 - dv)
    w10 = du * (1.0 - dv)
    w01 = (1.0 - du) * dv
    w11 = du * dv

    sampled = (
        w00 * env_map_hwc[v0c, u0w]
        + w10 * env_map_hwc[v0c, u1w]
        + w01 * env_map_hwc[v1c, u0w]
        + w11 * env_map_hwc[v1c, u1w]
    )
    return sampled.view(*out_shape, 3)


def splat_environment_map(
    dirs, colors, weights, env_h, env_w, flip_u=True, flip_v=True, eps=1e-8,
    return_weight=False,
):
    """Nearest-bin weighted accumulation, used to seed the env-map parameter.

    With ``return_weight`` also returns the per-bin accumulated weight
    ``accum_w`` ([env_h*env_w]) — a raw measure of how much each env-map pixel
    was actually observed by reflected rays.
    """
    dirs = dirs.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    weights = weights.reshape(-1).to(colors.dtype)

    u, v = dirs_to_equirect_uv(dirs, env_h, env_w, flip_u=flip_u, flip_v=flip_v)
    u_idx = torch.clamp(u.round().long(), 0, env_w - 1)
    v_idx = torch.clamp(v.round().long(), 0, env_h - 1)
    lin = v_idx * env_w + u_idx

    accum_rgb = torch.zeros(
        (env_h * env_w, 3), device=colors.device, dtype=colors.dtype
    )
    accum_w = torch.zeros((env_h * env_w,), device=colors.device, dtype=colors.dtype)
    accum_rgb.index_add_(0, lin, colors * weights.unsqueeze(-1))
    accum_w.index_add_(0, lin, weights)

    env = (accum_rgb / (accum_w.unsqueeze(-1) + eps)).reshape(env_h, env_w, 3)
    valid = (accum_w > eps).reshape(env_h, env_w)
    if return_weight:
        return env, valid, accum_w
    return env, valid


class DiffuseEnvModel(nn.Module):
    """Decomposes a surface light field into a per-point diffuse color and a
    shared reflected environment map.

    reconstruction(p, m) = alpha * diffuse(p) + (1 - alpha) * env(reflect(p, m))

    Works per surface point (P points x M views) rather than on the dense image
    grid, so it only touches observed samples.
    """

    def __init__(self, alpha, diffuse_logits_init, env_init, flip_u=True, flip_v=True):
        super().__init__()
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        self.flip_u = flip_u
        self.flip_v = flip_v
        self.diffuse_logits = nn.Parameter(diffuse_logits_init)
        self.environment_map = nn.Parameter(env_init)

    def forward(self, reflected_dirs):
        # reflected_dirs: [P, M, 3]
        diffuse = torch.sigmoid(self.diffuse_logits)  # [P, 3]
        reflected = sample_environment_map(
            self.environment_map, reflected_dirs, self.flip_u, self.flip_v
        )  # [P, M, 3]
        reconstruction = (
            self.alpha * diffuse[:, None, :] + (1.0 - self.alpha) * reflected
        )
        return reconstruction, diffuse, reflected


# ---------------------------------------------------------------------------
# Diffuse-fraction (alpha) estimation from the surface light field.
#
# Appearance model: obs[p, v] = alpha * D[p] + (1 - alpha) * R[p, v], with the
# diffuse D view-independent and the reflection R view-dependent, so the
# per-point view-std of luminance is std_v(obs) = (1 - alpha) * std_v(R).
# Across the sub-aperture views the reflected direction sweep is driven by the
# LF baseline, not the surface geometry, so std_v(R) is nearly
# geometry-independent and calibrates to reflectivity.
#
# Aggregation: within a frame, a LOW percentile (q=0.30) of the
# frontal^2-weighted per-point std isolates the pervasive reflective signal
# from the noisy diffuse floor (shading, normal error, grazing-angle Fresnel).
# Across frames, a low-contrast pose under-reads reflectivity but never
# over-reads; alpha is a material constant, so per-frame stats accumulate and
# are read at an UPPER percentile (q=0.80).
#
# Calibration: (1 - alpha) = (stat - FLOOR) / SLOPE (least-squares fit). A
# different environment contrast rescales SLOPE — a low-contrast environment
# makes a reflective surface look diffuse, an inherent single-view ambiguity.
# ---------------------------------------------------------------------------
_LUM_WEIGHTS = (0.299, 0.587, 0.114)

ALPHA_MIN_VALID_VIEWS = 6  # require this many views per point to use it
ALPHA_MIN_POINTS = 50  # below this, decline to estimate
ALPHA_FRONTAL_POW = 2.0  # frontal^pow * viewcount weighting
ALPHA_WITHIN_Q = 0.30  # within-frame percentile of per-point std
ALPHA_ACROSS_Q = 0.80  # cross-frame accumulation percentile
ALPHA_ACCUM_MIN_FRAMES = 4  # use accumulation calibration past this many frames
# stat = FLOOR + SLOPE * (1 - alpha)
ALPHA_CAL_SINGLE = (0.0015, 0.0196)  # one-frame estimate
ALPHA_CAL_ACCUM = (0.0016, 0.0237)  # p80 over accumulated frames

# No real surface is a perfect mirror or a perfect diffuser: even "fully
# diffuse" materials keep a faint ambient env contribution, and mirror-like
# materials retain some intrinsic albedo. Clamp alpha away from the unphysical
# 0/1 endpoints so separate_reflection always has some reflective weight (and
# therefore always fits an env map) and some diffuse weight.
ALPHA_CLAMP_MIN = 0.03
ALPHA_CLAMP_MAX = 0.97


def _weighted_quantile(values, weights, q):
    """Weighted quantile of 1-D ``values`` (linear interpolation). Returns float."""
    if values.numel() == 1:
        return float(values.reshape(()))
    v, order = torch.sort(values)
    w = weights[order].clamp(min=0)
    cw = torch.cumsum(w, 0) - 0.5 * w
    cw = cw / w.sum().clamp(min=1e-8)
    q_t = torch.as_tensor(q, device=v.device, dtype=v.dtype)
    hi = torch.searchsorted(cw, q_t).clamp(1, v.numel() - 1)
    lo = hi - 1
    t = ((q_t - cw[lo]) / (cw[hi] - cw[lo]).clamp(min=1e-8)).clamp(0.0, 1.0)
    return float(v[lo] + t * (v[hi] - v[lo]))


def estimate_alpha_stat(colors, valid, view_dirs, normals):
    """Per-frame robust view-variance statistic (low percentile of per-point
    frontal-weighted luminance std). Larger => more reflective.

    colors    : [P, M, 3] linear surface light field
    valid     : [P, M] per-observation validity
    view_dirs : [P, M, 3] unit camera->point directions (for frontalness)
    normals   : [P, 3]   surface normals (for frontalness); ``None`` => unweighted

    Returns a float, or ``None`` when too few points are observed.
    """
    device = colors.device
    valid = valid.to(device).float()
    vc = valid.sum(1)  # [P]
    keep = vc >= ALPHA_MIN_VALID_VIEWS
    if int(keep.sum()) < ALPHA_MIN_POINTS:
        return None
    colors = colors[keep]
    valid = valid[keep]
    vc = vc[keep].clamp(min=1.0)

    lw = torch.tensor(_LUM_WEIGHTS, device=device, dtype=colors.dtype)
    lum = (colors * lw).sum(-1)  # [P, M]
    mean_l = (lum * valid).sum(1) / vc
    std_l = (((lum - mean_l[:, None]) ** 2 * valid).sum(1) / vc).clamp(min=0).sqrt()

    if normals is not None and view_dirs is not None:
        nrm = normals.to(device).float()[keep]
        vd = view_dirs.to(device).float()[keep]
        frontal = ((vd * nrm[:, None, :]).sum(-1).abs() * valid).sum(1) / vc
        weight = frontal.clamp(0.0, 1.0) ** ALPHA_FRONTAL_POW * vc
    else:
        weight = vc
    return _weighted_quantile(std_l, weight, ALPHA_WITHIN_Q)


def _stat_to_alpha(stat, cal):
    floor, slope = cal
    raw = 1.0 - (stat - floor) / slope
    return float(min(max(raw, ALPHA_CLAMP_MIN), ALPHA_CLAMP_MAX))


def estimate_alpha(colors, valid, view_dirs, normals, stat_history=None):
    """Estimate the diffuse fraction ``alpha`` in [0, 1] from one frame's SLF,
    accumulating evidence across frames.

    Pass the returned ``history`` back in on the next frame to refine the
    estimate (alpha is a material constant). With no/short history a
    single-frame calibration is used; once ``ALPHA_ACCUM_MIN_FRAMES`` stats are
    gathered the cross-frame upper-percentile calibration takes over.

    Returns ``(alpha, history)`` where ``history`` is the updated list of
    per-frame stats.
    """
    stat = estimate_alpha_stat(colors, valid, view_dirs, normals)
    history = list(stat_history) if stat_history else []
    if stat is not None:
        history.append(stat)
    if not history:
        return (
            ALPHA_CLAMP_MAX,
            history,
        )  # no evidence -> assume near-diffuse (safe path)
    if len(history) >= ALPHA_ACCUM_MIN_FRAMES:
        agg = float(
            torch.quantile(
                torch.tensor(history, dtype=torch.float32), ALPHA_ACROSS_Q
            )
        )
        alpha = _stat_to_alpha(agg, ALPHA_CAL_ACCUM)
    else:
        alpha = _stat_to_alpha(history[-1], ALPHA_CAL_SINGLE)
    return alpha, history


def separate_reflection(
    colors,
    alpha=None,
    reflected_dirs=None,
    valid=None,
    view_dirs=None,
    normals=None,
    mask=None,
    alpha_stat_history=None,
    previous_environment_map=None,
    previous_env_confidence=None,
    env_h=256,
    env_w=512,
    flip_u=True,
    flip_v=True,
    lr=1e-2,
    env_lr=5e-2,
    weight_env_tv=1e-2,
    weight_env_range=1e-1,
    weight_env_prior=1e-1,
    tv_observed_floor=0.1,
    iterations=300,
    eps=1e-8,
    verbose=False,
):
    """Jointly fit a per-point diffuse color and a reflected environment map,
    then decorrelate the diffuse from the back-projected reflection.

    Operates per surface point (P points observed across M views).

    Args:
        colors:         [P, M, 3] observed surface light field (linear).
        alpha:          scalar diffuse fraction in [0, 1]. ``None`` => estimate it
            from the light field via ``estimate_alpha`` (accumulating across frames
            through ``alpha_stat_history``).
        reflected_dirs: [P, M, 3] world-space reflected ray directions.
        valid:          [P, M] per-observation validity (defaults to all).
        view_dirs:      [P, M, 3] optional, for grazing-angle confidence and (when
            ``alpha is None``) alpha estimation.
        normals:        [P, 3] optional, for grazing-angle confidence and alpha
            estimation.
        alpha_stat_history: list of per-frame alpha stats from previous frames;
            only used when ``alpha is None`` to refine the estimate over time.
        mask:           [H, W] bool, foreground mask of the central view. When
            provided, the final decorrelation step is run in image space to strip
            residual reflection structure from the diffuse (see
            ``refine_diffuse_decorrelate``). Requires the ordering of P points to
            match ``image[mask]``.
        previous_environment_map: [env_h, env_w, 3] accumulated env map to warm-start
            and softly anchor the optimization (enables multi-frame accumulation).
        previous_env_confidence: [env_h, env_w] accumulated observation confidence
            from previous frames; OR-combined with this frame's so coverage grows as
            the object rotates. Thread forward alongside ``previous_environment_map``.
        env_h, env_w:   env-map resolution, fixed regardless of alpha.

    Returns:
        diffuse_point:   [P, 3]
        environment_map: [env_h, env_w, 3]
        reflective_point:[P, M, 3] env contribution sampled per observation.
        env_confidence:  [env_h, env_w] in [0, 1] — per-pixel observation confidence
            (high where reflected rays landed, low for interpolated fill). Use it to
            downweight unobserved reflections in the relight loss, and thread it
            forward as ``previous_env_confidence``.
    """
    p, m, _ = colors.shape
    device = "cuda"

    colors = colors.to(device)
    reflected_dirs = reflected_dirs.to(device)

    if valid is None:
        valid = torch.ones((p, m), device=device)
    valid = valid.to(device).float()

    # Determine alpha from the surface light field when not supplied.
    if alpha is None:
        alpha, _ = estimate_alpha(
            colors, valid, view_dirs, normals, stat_history=alpha_stat_history
        )
        if verbose:
            print(f"[separate_reflection] estimated alpha = {alpha:.3f}")
    alpha = float(min(max(float(alpha), ALPHA_CLAMP_MIN), ALPHA_CLAMP_MAX))
    alpha_t = torch.tensor(alpha, device=device)

    # Per-observation confidence: down-weight grazing angles (unreliable reflect dirs).
    weight = valid
    if view_dirs is not None and normals is not None:
        view_dirs = view_dirs.to(device)
        normals_ext = normals.to(device)[:, None, :].expand(-1, m, -1)
        cos = (view_dirs * normals_ext).sum(dim=-1).abs()
        weight = valid * cos.clamp(min=0.0)

    # ---- Initialize diffuse from per-point valid mean ----
    obs_sum = (colors * valid[..., None]).sum(dim=1)
    obs_cnt = valid.sum(dim=1).clamp(min=1.0)[..., None]
    diffuse0 = (obs_sum / obs_cnt).clamp(1e-4, 1.0 - 1e-4)
    diffuse_logits_init = torch.log(diffuse0 / (1.0 - diffuse0))

    # ---- Initialize env map: warm-start from previous, fill gaps with a splat of
    #      the residual reflective estimate (obs - alpha*diffuse) / (1 - alpha) ----
    denom = max(1.0 - float(alpha), eps)
    reflective_est = ((colors - alpha_t * diffuse0[:, None, :]) / denom).clamp(0.0, 1.0)
    env_splat, _, accum_w = splat_environment_map(
        reflected_dirs, reflective_est, weight, env_h, env_w, flip_u, flip_v, eps,
        return_weight=True,
    )

    # ---- Observation confidence: where reflected rays actually landed this frame,
    #      accumulated across frames via a probabilistic OR. ----
    accum_w_2d = accum_w.reshape(env_h, env_w)
    pos = accum_w_2d > 0
    # Soft "observed vs interpolated" mask. The per-bin splat weight is heavy-tailed
    # (front-facing dense regions dwarf sparse grazing hits), so normalise by a LOW
    # robust reference (a fraction of the median observed weight): any bin with a
    # meaningful number of reflected-ray hits saturates to ~1, only truly-unobserved
    # bins stay ~0.
    if bool(pos.any()):
        ref = (0.3 * torch.quantile(accum_w_2d[pos].float(), 0.5)).clamp(min=eps)
    else:
        ref = accum_w_2d.new_tensor(1.0)
    this_conf = 1.0 - torch.exp(-accum_w_2d / ref)  # [H, W] in [0,1]
    if previous_env_confidence is not None:
        prev_conf = previous_env_confidence.to(device).float()
        env_confidence = 1.0 - (1.0 - prev_conf) * (1.0 - this_conf)
    else:
        env_confidence = this_conf

    prev_valid = None
    if previous_environment_map is not None:
        prev = previous_environment_map.to(device).float()
        prev_valid = prev.sum(dim=-1) > eps
        env_init = torch.where(prev_valid[..., None], prev, env_splat)
        prev_anchor = prev
    else:
        env_init = env_splat
        prev_anchor = None

    model = DiffuseEnvModel(
        alpha=float(alpha),
        diffuse_logits_init=diffuse_logits_init,
        env_init=env_init,
        flip_u=flip_u,
        flip_v=flip_v,
    ).to(device)
    optimizer = optim.Adam(
        [
            {"params": [model.diffuse_logits], "lr": lr},
            {"params": [model.environment_map], "lr": env_lr},
        ]
    )

    target = colors
    w_ext = weight[..., None]
    w_norm = w_ext.sum().clamp(min=1.0)

    for _ in tqdm(range(iterations), disable=not verbose):
        optimizer.zero_grad()
        reconstruction, _, _ = model(reflected_dirs)

        loss_recon = (((reconstruction - target) ** 2) * w_ext).sum() / w_norm

        env = model.environment_map
        # Confidence-weighted TV: interpolation (smoothness) only governs UNobserved
        # pixels. Observed pixels keep a small floor of smoothing but are otherwise
        # driven by the reconstruction term, so real reflection structure is not
        # blurred away.
        w_tv = tv_observed_floor + (1.0 - tv_observed_floor) * (1.0 - env_confidence)
        du = (env - torch.roll(env, shifts=1, dims=1)).abs().mean(-1)  # [H, W]
        dv = (env[1:, :, :] - env[:-1, :, :]).abs().mean(-1)  # [H-1, W]
        tv_u = (du * w_tv).mean()
        tv_v = (dv * w_tv[1:, :]).mean()
        loss_env_tv = tv_u + tv_v

        loss_env_range = (F.relu(-env) + F.relu(env - 1.0)).mean()

        loss = (
            loss_recon + weight_env_tv * loss_env_tv + weight_env_range * loss_env_range
        )

        if prev_anchor is not None and weight_env_prior > 0:
            loss_prior = (
                ((env - prev_anchor) ** 2) * prev_valid[..., None]
            ).sum() / prev_valid.sum().clamp(min=1.0)
            loss = loss + weight_env_prior * loss_prior

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        _, diffuse_point, reflective_point = model(reflected_dirs)
        environment_map = model.environment_map.detach().clamp(0.0, 1.0)
        if mask is not None:
            mask_d = mask.to(device)
            H, W = mask.shape
            mid = reflective_point.shape[1] // 2
            diffuse_img = diffuse_point.new_zeros(H, W, 3)
            diffuse_img[mask_d] = diffuse_point
            refl_img = diffuse_point.new_zeros(H, W, 3)
            refl_img[mask_d] = reflective_point[:, mid, :].to(refl_img.dtype)
            diffuse_img = refine_diffuse_decorrelate(diffuse_img, refl_img, mask_d)
            diffuse_point = diffuse_img[mask_d]
    return diffuse_point, environment_map, reflective_point, env_confidence.detach()


# ---------------------------------------------------------------------------
# Diffuse refinement: decorrelate the diffuse from the back-projected reflection.
#
# The joint fit can leave residual reflection structure baked into the per-point
# diffuse colour. ``reflective_point`` returned by ``separate_reflection`` is the
# environment map already projected back onto the surface, so a central-view
# reflection image is available for free. Where the diffuse image varies
# linearly with that reflection inside a local window, that fluctuation is
# leaked reflection and is subtracted; the local mean (true albedo) is
# preserved. Everything is box-filtered (O(N), separable).
# ---------------------------------------------------------------------------
def _box_sum(x, radius):
    """Separable box *sum* (not mean) over [B, C, H, W]; pads with zeros."""
    c = x.shape[1]
    k = 2 * radius + 1
    wx = x.new_ones(c, 1, 1, k)
    wy = x.new_ones(c, 1, k, 1)
    x = F.conv2d(x, wx, padding=(0, radius), groups=c)
    x = F.conv2d(x, wy, padding=(radius, 0), groups=c)
    return x


def _smoothstep(lo, hi, x):
    t = ((x - lo) / (hi - lo + 1e-8)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _decorrelate_single(D, R, mf, cnt_fn, radius, corr_lo, corr_hi, strength, eps):
    """One local guided-linear decorrelation pass at a single window radius.

    D, R: [1, 3, H, W] (linear). mf: [1, 1, H, W] mask. Returns refined [1,3,H,W].
    Within each window the diffuse fluctuation explained linearly by R is the
    leaked reflection; subtract it, gated by the local |correlation|. The local
    mean (albedo) is preserved, so a constant reflection bias stays folded into
    the albedo while only structure is removed.
    """
    cnt = cnt_fn(radius)

    def bmean(x):  # mask-aware moving average at this radius
        return _box_sum(x * mf, radius) / cnt

    mean_D = bmean(D)
    mean_R = bmean(R)
    var_R = (bmean(R * R) - mean_R**2).clamp(min=0.0)
    var_D = (bmean(D * D) - mean_D**2).clamp(min=0.0)
    cov = bmean(D * R) - mean_D * mean_R

    a = cov / (var_R + eps)
    b = mean_D - a * mean_R
    # Guided-filter averaging of the linear coefficients for spatial coherence.
    q = bmean(a) * R + bmean(b)  # diffuse explained linearly by reflection

    corr = cov / torch.sqrt(var_D * var_R + eps)
    gate = strength * _smoothstep(corr_lo, corr_hi, corr.abs())

    leaked = q - mean_D  # R-correlated fluctuation (albedo mean removed)
    return (D - gate * leaked).clamp(min=0.0)


def refine_diffuse_decorrelate(
    diffuse_img,
    reflection_img,
    mask,
    radii=None,
    corr_lo=0.5,
    corr_hi=0.85,
    strength=1.0,
    global_corr_skip=0.25,
    eps=1e-6,
):
    """Strip residual reflection structure from an estimated diffuse view.

    Multi-scale local guided-linear decorrelation, run in *linear* radiance.
    Each pass removes the diffuse fluctuation that varies linearly with the
    back-projected reflection at one window scale, gated by the local
    correlation strength so only highly reflection-correlated structure is
    touched (the local mean / albedo is preserved). The fine->coarse cascade
    catches both high-frequency env detail and low-frequency reflection
    gradients. All box filters (O(N), separable); a cheap global-correlation
    early-out keeps it free when nothing leaks.

    Args:
        diffuse_img:    [H, W, 3] estimated diffuse, linear.
        reflection_img: [H, W, 3] reflection projected onto the surface at this
            view (``reflective_point[:, M//2]`` scattered to the grid), linear.
        mask:           [H, W] foreground mask.
        radii:          window radii for the cascade; defaults to fractions of the
            image size (~2%, 6%, 15% of min(H, W)).
        corr_lo/corr_hi: gate ramp on |local correlation| -> removal weight.
        strength:       scale in [0, 1] on the removed component.
        global_corr_skip: skip a pass when no channel's global |corr| exceeds this.

    Returns:
        [H, W, 3] refined diffuse, linear.
    """
    device = diffuse_img.device
    m = mask > 0
    H, W = mask.shape
    if m.sum() == 0:
        return diffuse_img

    if radii is None:
        s = min(H, W)
        radii = sorted({max(4, int(round(f * s))) for f in (0.02, 0.06, 0.15)})
    radii = [r for r in radii if r < min(H, W) // 2 - 1] or [max(4, min(H, W) // 4)]

    mf = m.float().to(device)[None, None]  # [1, 1, H, W]
    D = diffuse_img.permute(2, 0, 1)[None].to(device)  # [1, 3, H, W]
    R = reflection_img.permute(2, 0, 1)[None].to(device)
    R_flat = reflection_img[m]
    rc = R_flat - R_flat.mean(0, keepdim=True)
    rc_norm = rc.norm(dim=0)

    def cnt_fn(radius):
        return _box_sum(mf, radius).clamp(min=1.0)

    cur = D
    for radius in radii:
        # Cheap global early-out for this pass: nothing correlated -> skip.
        dm = cur[0].permute(1, 2, 0)[m]
        dc = dm - dm.mean(0, keepdim=True)
        gcorr = (dc * rc).sum(0).abs() / (dc.norm(dim=0) * rc_norm + eps)
        if float(gcorr.max()) < global_corr_skip:
            continue
        cur = _decorrelate_single(
            cur, R, mf, cnt_fn, radius, corr_lo, corr_hi, strength, eps
        )

    refined = torch.where(mf > 0, cur, D)
    return refined[0].permute(1, 2, 0)
