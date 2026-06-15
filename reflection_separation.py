import torch
import numpy as np
from PIL import Image
from src.dataset import LFDataset
from src.utilities import backproject_depth_to_pointcloud
from surface_lf import SurfaceLF, SurfaceLFRig
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from utils import srgb_to_linear, linear_to_srgb
from tqdm import tqdm


def build_surface_lf(frame, s_size, t_size):
    mask = frame["masks"][s_size // 2, t_size // 2]
    depth = frame["depth"]
    camera_matrix = frame["camera_matrix"]

    pc, pc_scales = backproject_depth_to_pointcloud(
        pixel_indices=None,
        depths=depth,
        camera_matrix=camera_matrix,
        return_scales=True,
    )
    pc = pc[(mask > 0).reshape(-1)]
    pc_scales = pc_scales[(mask > 0).reshape(-1)]

    surface_lf_rig = SurfaceLFRig.build(
        K=frame["camera_matrix"],
        poses_4x4=frame["camera_poses_rel"].reshape(-1, 4, 4),
        image_size_hw=(
            frame["LF"].shape[2],
            frame["LF"].shape[3],
        ),
    )

    surface_lf = SurfaceLF(
        surface_lf_rig,
        pc,
        frame["LF"]
        .reshape(-1, frame["LF"].shape[2], frame["LF"].shape[3], 3)
        .permute(0, 3, 1, 2),
        pc_scales,
        previous_environment_map=None,
        # We only need geometry (colors / view_dirs / normals / valid) out of the
        # rig here; the naive path avoids invoking the (now changed) reflection
        # separation internally.
        use_naive_relight=True,
    )
    return surface_lf


# ---------------------------------------------------------------------------
# Equirectangular environment-map helpers.
# Conventions mirror SurfaceLF._dirs_to_equirect_uv* so the two stay compatible
# when this module is integrated into surface_lf.py.
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
    dirs, colors, weights, env_h, env_w, flip_u=True, flip_v=True, eps=1e-8
):
    """Nearest-bin weighted accumulation, used to seed the env-map parameter."""
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


def separate_reflection(
    colors,
    alpha,
    reflected_dirs,
    valid=None,
    view_dirs=None,
    normals=None,
    mask=None,
    previous_environment_map=None,
    env_h=256,
    env_w=512,
    flip_u=True,
    flip_v=True,
    lr=1e-2,
    env_lr=5e-2,
    weight_env_tv=1e-2,
    weight_env_range=1e-1,
    weight_env_prior=1e-1,
    iterations=300,
    eps=1e-8,
    verbose=False,
):
    """Jointly fit a per-point diffuse color and a reflected environment map,
    then decorrelate the diffuse from the back-projected reflection.

    Operates per surface point (P points observed across M views).

    Args:
        colors:         [P, M, 3] observed surface light field (linear).
        alpha:          scalar diffuse fraction in [0, 1].
        reflected_dirs: [P, M, 3] world-space reflected ray directions.
        valid:          [P, M] per-observation validity (defaults to all).
        view_dirs:      [P, M, 3] optional, for grazing-angle confidence.
        normals:        [P, 3] optional, for grazing-angle confidence.
        mask:           [H, W] bool, foreground mask of the central view. When
            provided, the final decorrelation step is run in image space to strip
            residual reflection structure from the diffuse (see
            ``refine_diffuse_decorrelate``). Requires the ordering of P points to
            match ``image[mask]`` (the convention used by both callers).
        previous_environment_map: [env_h, env_w, 3] accumulated env map to warm-start
            and softly anchor the optimization (enables multi-frame accumulation).

    Returns:
        diffuse_point:   [P, 3]
        environment_map: [env_h, env_w, 3]
        reflective_point:[P, M, 3] env contribution sampled per observation.
    """
    p, m, _ = colors.shape
    device = "cuda"

    colors = colors.to(device)
    reflected_dirs = reflected_dirs.to(device)
    alpha_t = torch.tensor(float(alpha), device=device)

    if valid is None:
        valid = torch.ones((p, m), device=device)
    valid = valid.to(device).float()

    # Pure-diffuse shortcut: env plays no role.
    if torch.allclose(alpha_t, torch.ones_like(alpha_t)):
        diffuse = colors[:, m // 2, :]
        env = torch.zeros((env_h, env_w, 3), device=device)
        reflective = torch.zeros_like(colors)
        return diffuse, env, reflective

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
    env_splat, _ = splat_environment_map(
        reflected_dirs, reflective_est, weight, env_h, env_w, flip_u, flip_v, eps
    )

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
        tv_u = (env - torch.roll(env, shifts=1, dims=1)).abs().mean()  # wrap in u
        tv_v = (env[1:, :, :] - env[:-1, :, :]).abs().mean()
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
    return diffuse_point, environment_map, reflective_point


# ---------------------------------------------------------------------------
# Diffuse refinement: decorrelate the diffuse from the back-projected reflection.
#
# The joint fit can leave residual reflection structure baked into the per-point
# diffuse colour. ``reflective_point`` returned by ``separate_reflection`` is the
# environment map already projected back onto the surface (via normals -> reflect
# dirs -> equirect sample), so a central-view reflection image is available for
# free. Where the diffuse image varies *linearly with* that reflection inside a
# local window, that fluctuation is leaked reflection and is subtracted; the
# local mean (true albedo) is preserved. Everything is box-filtered (O(N),
# separable), so this is a sub-millisecond postprocess.
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
    the albedo (physically indistinguishable) while only structure is removed.
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

    Multi-scale local guided-linear decorrelation, run in *linear* radiance. Each
    pass removes the diffuse fluctuation that varies linearly with the
    back-projected reflection at one window scale, gated by the local correlation
    strength so only highly reflection-correlated structure is touched (the local
    mean / albedo is preserved). The fine->coarse cascade catches both
    high-frequency env detail and low-frequency reflection gradients without
    having to know the structure scale in advance. All box filters (O(N),
    separable); a cheap global-correlation early-out keeps it free when nothing
    leaks (e.g. the ``alpha == 1`` pure-diffuse case where the reflection is zero).

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


if __name__ == "__main__":
    sequence_name = "bleach0"

    MIDDLE_REFLECTIVITY = 0.7
    ALPHA = 1 - MIDDLE_REFLECTIVITY
    N_ITERS = 300

    out_folder = f"vis_env_{MIDDLE_REFLECTIVITY}"
    os.makedirs(out_folder, exist_ok=True)

    path_middle = f"/home/ngoncharov/SpecTrack_dataset/objects_{MIDDLE_REFLECTIVITY}/{sequence_name}"
    dataset = LFDataset(path_middle)
    s_size, t_size = dataset.metadata["n_views"]

    previous_environment_map = None

    for i in range(len(dataset)):
        frame = dataset[i]
        mask = frame["masks"][s_size // 2, t_size // 2]

        # Middle-view image for reference output.
        lf = frame["LF"][s_size // 2, t_size // 2].clone()
        lf[mask == 0] = 0

        surface_lf = build_surface_lf(frame, s_size, t_size)

        # Geometry from the rig is per-point: [M, P, *] with M = s_size * t_size.
        # Transpose to [P, M, *] for the point-based separation.
        colors_pt = surface_lf.colors.permute(1, 0, 2)  # [P, M, 3]
        view_dirs_pt = surface_lf.view_dirs.permute(1, 0, 2)  # [P, M, 3]
        valid_pt = surface_lf.valid.permute(1, 0)  # [P, M]
        normals_pt = surface_lf.surface_normals.float()  # [P, 3]

        normals_rep = normals_pt[:, None, :].expand_as(view_dirs_pt)
        reflected_pt = (
            view_dirs_pt
            - 2.0 * (view_dirs_pt * normals_rep).sum(dim=-1, keepdim=True) * normals_rep
        )
        reflected_pt = F.normalize(reflected_pt, dim=-1)  # [P, M, 3]

        mask_bool = mask > 0
        diffuse_point, environment_map, _ = separate_reflection(
            colors=srgb_to_linear(colors_pt),
            alpha=ALPHA,
            reflected_dirs=reflected_pt,
            valid=valid_pt,
            view_dirs=view_dirs_pt,
            normals=normals_pt,
            mask=mask_bool,
            previous_environment_map=previous_environment_map,
            iterations=N_ITERS,
            verbose=True,
        )
        previous_environment_map = environment_map.detach()

        # Scatter the per-point diffuse (already decorrelated) to the image grid.
        diffuse_img = torch.zeros((*mask.shape, 3), device=diffuse_point.device)
        diffuse_img[mask_bool] = diffuse_point

        # ---- Save: original middle view, rendered diffuse, environment map ----
        diffuse_vis = (
            linear_to_srgb(diffuse_img).clamp(0, 1).cpu().numpy() * 255
        ).astype(np.uint8)
        env_vis = (
            linear_to_srgb(environment_map).clamp(0, 1).cpu().numpy() * 255
        ).astype(np.uint8)
        orig_vis = (lf.cpu().numpy() * 255).astype(np.uint8)

        idx = str(i).zfill(4)
        Image.fromarray(orig_vis).save(f"{out_folder}/orig_{idx}.png")
        Image.fromarray(diffuse_vis).save(f"{out_folder}/diffuse_{idx}.png")
        Image.fromarray(env_vis).save(f"{out_folder}/env_{idx}.png")
