"""Photometric pose refinement: transform + relight the previous SLF, match current SLF render.

Per-step:
  1. Rotate/translate previous camera-space points with the candidate pose.
  2. Relight each point: alpha * diffuse + (1-alpha) * env_map[reflected_dir].
  3. Scatter-render the relighted points to an image; apply Gaussian blur.
     Compare against a pre-rendered, equally-blurred render of the *current* SLF.
     Both sides have identical splatting aliasing so the MSE is physically meaningful.
  4. Minimise a composite loss:
       photometric MSE   (primary, in linear space, on blurred images)
     + depth anchoring   (L1 between rendered z and target depth, small weight)
     + mask alignment    (penalise points projecting outside the target mask)
     + pose magnitude    (angular + translation regularisation from the coarse init)

All photometric quantities are kept in linear light throughout; only the viser
visualisation converts to sRGB for display.  Loss / translation-error /
rotation-error curves are rendered with matplotlib and shown as camera-frustum
images in the 3-D scene.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from loss import rotation_6d_to_matrix, matrix_to_rotation_6d
from src.surface_light_field import _estimate_normals
from utils import linear_to_srgb

# ── env-map sampling ────────────────────────────────────────────────────────────


def _sample_env_map(env_map_hwc: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Bilinear equirect lookup (flip_u=True, flip_v=True).

    env_map_hwc : [H, W, 3]  linear float [0, 1]
    dirs        : [N, 3]     unit directions
    returns     : [N, 3]
    """
    dirs = F.normalize(dirs.float(), dim=-1, eps=1e-8)
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    env_h, env_w = env_map_hwc.shape[:2]

    lon = torch.atan2(x, z)
    lat = torch.asin(torch.clamp(y, -1.0 + 1e-6, 1.0 - 1e-6))
    u_px = (lon / (2.0 * torch.pi) + 0.5) * (env_w - 1)
    u_px = (env_w - 1) - u_px
    v_px = (0.5 + lat / torch.pi) * (env_h - 1)

    u_n = u_px / (env_w - 1) * 2.0 - 1.0
    v_n = v_px / (env_h - 1) * 2.0 - 1.0
    grid = torch.stack([u_n, v_n], dim=-1).reshape(1, 1, -1, 2)

    img = env_map_hwc.permute(2, 0, 1).unsqueeze(0).float()
    out = F.grid_sample(
        img, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return out.squeeze(0).squeeze(1).T  # [N, 3]


# ── relighting ──────────────────────────────────────────────────────────────────


def _relight(
    diffuse: torch.Tensor,  # [N, 3] linear float [0, 1]
    normals: torch.Tensor,  # [N, 3] unit
    view_dirs: torch.Tensor,  # [N, 3] camera→point unit
    env_map: torch.Tensor | None,
    alpha: float,
) -> torch.Tensor:
    """Alpha * diffuse + (1-alpha) * env-map specular (all linear)."""
    if env_map is None or alpha >= 0.999:
        return diffuse.float()
    n = F.normalize(normals.float(), dim=-1, eps=1e-8)
    v = F.normalize(view_dirs.float(), dim=-1, eps=1e-8)
    reflected = F.normalize(
        v - 2.0 * (v * n).sum(-1, keepdim=True) * n, dim=-1, eps=1e-8
    )
    specular = _sample_env_map(env_map, reflected)
    return (alpha * diffuse.float() + (1.0 - alpha) * specular).clamp(0.0, 1.0)


# ── geometry helpers ────────────────────────────────────────────────────────────


def _transform_points(pts: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    pts_h = torch.cat([pts, pts.new_ones(len(pts), 1)], dim=-1)
    return (T.float() @ pts_h.T).T[:, :3]


def _project_uv_norm(
    pts: torch.Tensor, K: torch.Tensor, H: int, W: int
) -> torch.Tensor:
    z = pts[:, 2].clamp(min=1e-6)
    u = pts[:, 0] / z * K[0, 0] + K[0, 2]
    v = pts[:, 1] / z * K[1, 1] + K[1, 2]
    return torch.stack([u / (W - 1) * 2.0 - 1.0, v / (H - 1) * 2.0 - 1.0], dim=-1)


def _valid_mask(pts: torch.Tensor, K: torch.Tensor, H: int, W: int) -> torch.Tensor:
    uv_n = _project_uv_norm(pts, K, H, W)
    return (pts[:, 2] > 1e-3) & (uv_n[:, 0].abs() <= 1.0) & (uv_n[:, 1].abs() <= 1.0)


# ── silhouette distance field (mask loss) ────────────────────────────────────────


def _mask_distance_field(mask: torch.Tensor, device: str) -> torch.Tensor:
    """Euclidean distance (in pixels) from each pixel to the mask silhouette.

    0 inside the mask, growing outside.  Used as a smooth, far-reaching pull so
    the mask loss has a real gradient everywhere (a raw binary mask only has a
    1-px-wide gradient band, so it looks flat around the optimum).
    """
    m = mask.detach().cpu().numpy().astype(bool)
    dt = distance_transform_edt(~m)  # distance to nearest inside-pixel
    return torch.from_numpy(dt).float().to(device)


def _uv_to_px(uv_n: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Normalised [-1, 1] grid coords → pixel coords [u_px, v_px]."""
    u_px = (uv_n[:, 0] + 1.0) * 0.5 * (W - 1)
    v_px = (uv_n[:, 1] + 1.0) * 0.5 * (H - 1)
    return torch.stack([u_px, v_px], dim=-1)


# ── anti-aliasing: Gaussian blur ────────────────────────────────────────────────


def _gaussian_blur(img_hwc: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    """Isotropic Gaussian blur on [H, W, C] float tensor."""
    kernel_size = max(3, 2 * int(3.0 * sigma + 0.5) + 1)  # covers 3σ, always odd
    C = img_hwc.shape[2]
    k = (
        torch.arange(kernel_size, dtype=torch.float32, device=img_hwc.device)
        - kernel_size // 2
    )
    k1d = torch.exp(-(k**2) / (2.0 * sigma**2))
    k1d = k1d / k1d.sum()
    kernel = (k1d[:, None] * k1d[None, :])[None, None].expand(C, 1, -1, -1)
    nchw = img_hwc.permute(2, 0, 1).unsqueeze(0)
    return (
        F.conv2d(nchw, kernel, padding=kernel_size // 2, groups=C)
        .squeeze(0)
        .permute(1, 2, 0)
    )


# ── scatter helpers ─────────────────────────────────────────────────────────────


@torch.no_grad()
def _scatter_to_tensor(
    uv_n: torch.Tensor, colors: torch.Tensor, H: int, W: int
) -> torch.Tensor:
    """Scatter point colours → [H, W, 3] float32 (non-differentiable)."""
    u_px = ((uv_n[:, 0] + 1.0) * 0.5 * (W - 1)).long().clamp(0, W - 1)
    v_px = ((uv_n[:, 1] + 1.0) * 0.5 * (H - 1)).long().clamp(0, H - 1)
    flat = (v_px * W + u_px).unsqueeze(-1)
    img = torch.zeros(H * W, 3, device=colors.device)
    cnt = torch.zeros(H * W, 1, device=colors.device)
    img.scatter_add_(0, flat.expand(-1, 3), colors.float().clamp(0, 1))
    cnt.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
    return (img / cnt.clamp(min=1.0)).reshape(H, W, 3)


def _scatter_blur_nchw(
    uv_n: torch.Tensor, colors: torch.Tensor, H: int, W: int, sigma: float = 2.0
) -> torch.Tensor:
    """Scatter → Gaussian blur → [1, 3, H, W] (no grad through scatter, used as grid_sample input)."""
    with torch.no_grad():
        scattered = _scatter_to_tensor(uv_n, colors, H, W)
        blurred = _gaussian_blur(scattered, sigma=sigma).clamp(0.0, 1.0)
    return blurred.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]


@torch.no_grad()
def _to_display_u8(img_hwc_linear: torch.Tensor) -> np.ndarray:
    """Linear [H, W, 3] float → sRGB uint8 for display."""
    srgb = linear_to_srgb(img_hwc_linear.float().clamp(0, 1))
    return (srgb.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


# ── matplotlib curve → numpy image ─────────────────────────────────────────────


def _curve_image(
    ys: list[float],
    title: str,
    ylabel: str,
    color: str,
    W: int = 320,
    H: int = 160,
) -> np.ndarray:
    """Render a 1-D curve to an RGB uint8 numpy image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi = 80
    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
    xs = np.arange(len(ys))
    ax.plot(xs, ys, color=color, linewidth=1.2)
    ax.set_title(title, fontsize=7, pad=2)
    ax.set_ylabel(ylabel, fontsize=6)
    ax.set_xlabel("iter", fontsize=6)
    ax.tick_params(labelsize=5)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    from PIL import Image as PILImage

    return np.array(PILImage.open(buf).convert("RGB"))


# ── viser visualizer ────────────────────────────────────────────────────────────


class PhotometricRefineViewer:
    """Lightweight viser viewer for photometric refinement.

    Displays in the 3-D scene:
      • relighted point cloud
      • rendered / target / loss images as camera frustums
      • environment map frustum
      • loss, translation-error, and rotation-error curves rendered as frustum images
    """

    def __init__(self, port: int = 8081):
        import viser

        self._viser = viser
        self.server = viser.ViserServer(port=port, verbose=False)
        with self.server.gui.add_folder("Photometric Refinement"):
            self._iter_h = self.server.gui.add_number(
                "Iteration", initial_value=0, disabled=True
            )
            self._loss_h = self.server.gui.add_number(
                "Loss", initial_value=0.0, disabled=True
            )

        self._loss_hist: list[float] = []
        self._trans_hist: list[float] = []
        self._rot_hist: list[float] = []

    def reset_frame(self, _frame_idx: int) -> None:
        self._iter_h.value = 0
        self._loss_h.value = 0.0
        self._loss_hist = []
        self._trans_hist = []
        self._rot_hist = []

    @torch.no_grad()
    def update(
        self,
        iteration: int,
        loss: float,
        pts_curr: torch.Tensor,  # [N, 3] in current cam space
        colors_relit: torch.Tensor,  # [N, 3] linear float [0, 1]
        src_blurred: torch.Tensor,  # [H, W, 3] linear blurred source render
        tgt_rendered: torch.Tensor,  # [H, W, 3] linear blurred target render
        env_map: torch.Tensor | None,
        gt_pose_curr: np.ndarray | None = None,
        pose_curr_np: np.ndarray | None = None,
    ) -> None:
        self._iter_h.value = int(iteration)
        self._loss_h.value = float(loss)

        H, W = tgt_rendered.shape[:2]

        # 3D point cloud (sRGB for display)
        colors_srgb = linear_to_srgb(colors_relit.float().clamp(0, 1))
        self.server.scene.add_point_cloud(
            name="photometric/slf",
            points=pts_curr.float().cpu().numpy(),
            colors=colors_srgb.cpu().numpy(),
            point_size=0.002,
        )

        # Camera-frustum images: rendered, target, loss (all converted to sRGB uint8)
        rendered_u8 = _to_display_u8(src_blurred)
        target_u8 = _to_display_u8(tgt_rendered)
        loss_gray = (
            np.abs(rendered_u8.astype(np.float32) - target_u8.astype(np.float32))
            .mean(axis=-1)
            .clip(0, 255)
            .astype(np.uint8)
        )
        loss_img = np.repeat(loss_gray[:, :, np.newaxis], 3, axis=-1)

        fov = float(2.0 * np.arctan2(W / 2.0, float(max(W, 1))))
        aspect = W / max(H, 1)
        for name, img, x_off in [
            ("photometric/rendered", rendered_u8, -0.10),
            ("photometric/target", target_u8, 0.10),
            ("photometric/loss_img", loss_img, 0.0),
        ]:
            self.server.scene.add_camera_frustum(
                name=name,
                fov=fov,
                aspect=aspect,
                scale=0.03,
                image=img,
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(x_off, -0.02, 0.02),
            )

        if env_map is not None:
            env_np = env_map.detach().float()
            if env_np.ndim == 3 and env_np.shape[0] == 3:
                env_np = env_np.permute(1, 2, 0)
            env_u8 = _to_display_u8(env_np)
            eH, eW = env_u8.shape[:2]
            self.server.scene.add_camera_frustum(
                name="photometric/env_map",
                fov=float(2.0 * np.arctan2(eW / 2.0, float(max(eW, 1)))),
                aspect=eW / max(eH, 1),
                scale=0.03,
                image=env_u8,
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(0.28, -0.02, 0.02),
            )

        # ── live curve images ──────────────────────────────────────────────────
        self._loss_hist.append(float(loss))
        curve_aspect = 320 / 160

        if len(self._loss_hist) >= 2:
            loss_img_curve = _curve_image(
                self._loss_hist, "Photometric Loss", "loss", "red"
            )
            self.server.scene.add_camera_frustum(
                name="photometric/curve_loss",
                fov=float(2.0 * np.arctan2(320 / 2.0, 320.0)),
                aspect=curve_aspect,
                scale=0.03,
                image=loss_img_curve,
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(-0.15, -0.10, 0.02),
            )

        if gt_pose_curr is not None and pose_curr_np is not None:
            t_err = float(np.linalg.norm(pose_curr_np[:3, 3] - gt_pose_curr[:3, 3]))
            self._trans_hist.append(t_err)

            R_diff = pose_curr_np[:3, :3] @ gt_pose_curr[:3, :3].T
            cos_a = float(np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0))
            self._rot_hist.append(float(np.degrees(np.arccos(cos_a))))

            if len(self._trans_hist) >= 2:
                trans_img = _curve_image(
                    self._trans_hist, "Translation Error", "m", "blue"
                )
                self.server.scene.add_camera_frustum(
                    name="photometric/curve_trans",
                    fov=float(2.0 * np.arctan2(320 / 2.0, 320.0)),
                    aspect=curve_aspect,
                    scale=0.03,
                    image=trans_img,
                    wxyz=(1.0, 0.0, 0.0, 0.0),
                    position=(0.0, -0.10, 0.02),
                )
                rot_img = _curve_image(self._rot_hist, "Rotation Error", "deg", "green")
                self.server.scene.add_camera_frustum(
                    name="photometric/curve_rot",
                    fov=float(2.0 * np.arctan2(320 / 2.0, 320.0)),
                    aspect=curve_aspect,
                    scale=0.03,
                    image=rot_img,
                    wxyz=(1.0, 0.0, 0.0, 0.0),
                    position=(0.15, -0.10, 0.02),
                )

    def close(self) -> None:
        try:
            self.server.stop()
        except Exception:
            pass


# ── shared loss forward ──────────────────────────────────────────────────────────


@dataclass
class PhotometricContext:
    """Pose-independent inputs for one photometric loss evaluation.

    Everything here is fixed for a given (prev, curr) frame pair; only the
    candidate ``pose_curr`` varies between evaluations.  Built once by
    :func:`refine_pose_photometric` (and by the landscape analyzer) so the
    optimizer and any offline analysis evaluate *exactly* the same loss.
    """

    pts_v: torch.Tensor  # [M, 3] visible prev-frame points, prev cam space
    dif_v: torch.Tensor  # [M, 3] linear diffuse colours
    normals_v: torch.Tensor  # [M, 3] prev-cam-space normals
    env_map_prev: torch.Tensor | None
    alpha: float
    inv_pose_prev: torch.Tensor  # [4, 4] camera_prev → object (prev)
    K_t: torch.Tensor  # [3, 3]
    H: int
    W: int
    tgt_nchw: torch.Tensor  # [1, 3, H, W] blurred target render (curr SLF)
    depth_t: torch.Tensor | None  # [1, 1, H, W]
    mask_dt_t: torch.Tensor | None  # [1, 1, H, W] silhouette distance field (px)
    mask_fg_px: torch.Tensor | None  # [P, 2] subsampled target-fg pixel coords
    blur_sigma: float  # Gaussian sigma used for both source and target scatter renders
    R0_T: torch.Tensor  # [3, 3] anchor rotation^T for pose reg
    t0: torch.Tensor  # [3] anchor translation for pose reg
    lambda_depth: float
    lambda_mask: float
    lambda_rot: float
    lambda_trans: float


def photometric_forward(
    ctx: PhotometricContext, pose_curr: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Evaluate the composite photometric loss at one candidate pose.

    Returns ``(loss, components, aux)`` where ``components`` holds each
    additive term (``photo``/``depth``/``mask``/``rot``/``trans``) *before*
    its lambda weight, and ``aux`` carries intermediates needed for display
    (``pts_curr_t``, ``colors_relit``, ``src_nchw``).
    """
    T_rel = pose_curr @ ctx.inv_pose_prev

    pts_curr_t = _transform_points(ctx.pts_v, T_rel)
    normals_cur = (T_rel[:3, :3] @ ctx.normals_v.T).T
    view_dirs = F.normalize(pts_curr_t, dim=-1, eps=1e-8)
    colors_relit = _relight(
        ctx.dif_v, normals_cur, view_dirs, ctx.env_map_prev, ctx.alpha
    )

    uv_n = _project_uv_norm(pts_curr_t, ctx.K_t, ctx.H, ctx.W)  # grad through pose
    grid = uv_n.reshape(1, 1, -1, 2)

    # ── photometric residual (Lucas-Kanade style, both sides dealiased) ────────
    # Scatter+blur the source with the same sigma as the target so both images
    # are in the same spatial-frequency domain (equal aliasing cancels).
    # sampled_src is detached so gradient flows only through the target's image
    # gradient — the correct LK descent direction.
    src_nchw = _scatter_blur_nchw(uv_n, colors_relit, ctx.H, ctx.W, sigma=ctx.blur_sigma)

    def _sample(nchw: torch.Tensor) -> torch.Tensor:
        return (
            F.grid_sample(nchw, grid, mode="bilinear", padding_mode="border", align_corners=True)
            .squeeze(0).squeeze(1).T
        )

    sampled_src = _sample(src_nchw).detach()
    sampled_tgt = _sample(ctx.tgt_nchw)

    in_front = (pts_curr_t[:, 2] > 1e-3).float().detach()
    n_valid = in_front.sum().clamp(min=1.0)

    photo = ((sampled_src - sampled_tgt) ** 2 * in_front.unsqueeze(-1)).sum() / n_valid
    components: dict[str, torch.Tensor] = {"photo": photo}
    loss = photo

    if ctx.depth_t is not None and ctx.lambda_depth > 0.0:
        sampled_depth = F.grid_sample(
            ctx.depth_t, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).reshape(-1)
        depth_valid = in_front * (sampled_depth > 1e-3).float().detach()
        n_depth = depth_valid.sum().clamp(min=1.0)
        depth_loss = (
            (pts_curr_t[:, 2] - sampled_depth).abs() * depth_valid
        ).sum() / n_depth
        components["depth"] = depth_loss
        loss = loss + ctx.lambda_depth * depth_loss

    if ctx.mask_dt_t is not None and ctx.lambda_mask > 0.0:
        # ── symmetric silhouette chamfer (distance-transform based) ─────────────
        # model→data: pull each source point toward the target silhouette by
        # sampling the target distance field (smooth, far-reaching gradient).
        sampled_dt = F.grid_sample(
            ctx.mask_dt_t,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).reshape(-1)
        mask_a = (sampled_dt * in_front).sum() / n_valid

        # data→model: penalise target-silhouette pixels not covered by any source
        # point (prevents the model shrinking / drifting off the object).
        src_px = _uv_to_px(uv_n, ctx.H, ctx.W)
        keep = in_front > 0
        if ctx.mask_fg_px is not None and keep.any():
            d = torch.cdist(ctx.mask_fg_px, src_px[keep])  # [P, n_keep]
            mask_b = d.min(dim=1).values.mean()
        else:
            mask_b = src_px.new_zeros(())

        mask_loss = (mask_a + mask_b) / float(ctx.W)  # normalise → scale-invariant
        components["mask"] = mask_loss
        loss = loss + ctx.lambda_mask * mask_loss

    R = pose_curr[:3, :3]
    R_diff = R @ ctx.R0_T
    cos_angle = ((R_diff[0, 0] + R_diff[1, 1] + R_diff[2, 2]) - 1.0) / 2.0
    rot_reg = 1.0 - cos_angle.clamp(-1.0, 1.0)
    trans_reg = ((pose_curr[:3, 3] - ctx.t0) ** 2).sum()
    components["rot"] = rot_reg
    components["trans"] = trans_reg
    loss = loss + ctx.lambda_rot * rot_reg + ctx.lambda_trans * trans_reg

    aux = {
        "pts_curr_t": pts_curr_t,
        "colors_relit": colors_relit,
        "src_nchw": src_nchw,
    }
    return loss, components, aux


def build_photometric_context(
    points_prev: np.ndarray,
    diffuse_prev: np.ndarray,
    env_map_prev: torch.Tensor | None,
    points_curr: np.ndarray,
    diffuse_curr: np.ndarray,
    env_map_curr: torch.Tensor | None,
    K: np.ndarray,
    abs_pose_prev: np.ndarray,
    pose_coarse: np.ndarray,
    alpha: float,
    depth_curr: np.ndarray | None = None,
    mask_curr: np.ndarray | None = None,
    lambda_depth: float = 0.1,
    lambda_mask: float = 0.05,
    lambda_rot: float = 0.5,
    lambda_trans: float = 0.1,
    scale: float = 1.0,
    blur_sigma: float = 2.0,
    mask_fg_max: int = 2000,
    device: str = "cuda",
) -> tuple[PhotometricContext, torch.Tensor]:
    """Assemble a :class:`PhotometricContext` plus the [H,W,3] blurred target.

    The visible source-point set is frozen at ``pose_coarse`` (matching the
    optimizer); pose regularisation is anchored at ``pose_coarse``.  ``scale``
    < 1 builds a downsampled (coarse-to-fine) pyramid level: depth/mask are
    resized and ``K`` is rescaled accordingly.  Returns ``(ctx,
    tgt_rendered_hwc)``.  Raises ``ValueError`` if fewer than 10 source points
    are visible at the coarse pose.
    """
    if depth_curr is not None:
        H0, W0 = depth_curr.shape[:2]
    elif mask_curr is not None:
        H0, W0 = mask_curr.shape[:2]
    else:
        raise ValueError("depth_curr or mask_curr must be provided to determine H, W")

    H = max(1, round(H0 * scale))
    W = max(1, round(W0 * scale))
    sx, sy = W / W0, H / H0

    K_s = K.copy().astype(np.float64)
    K_s[0, 0] *= sx
    K_s[0, 2] *= sx
    K_s[1, 1] *= sy
    K_s[1, 2] *= sy

    pts = torch.from_numpy(points_prev).float().to(device)
    dif = torch.from_numpy(diffuse_prev).float().to(device)
    K_t = torch.from_numpy(K_s).float().to(device)

    def _resize(arr: np.ndarray, mode: str) -> torch.Tensor:
        t = torch.from_numpy(arr).float().to(device).unsqueeze(0).unsqueeze(0)
        if (H, W) != (H0, W0):
            t = F.interpolate(t, size=(H, W), mode=mode)
        return t

    depth_t = _resize(depth_curr, "nearest") if depth_curr is not None else None

    mask_dt_t: torch.Tensor | None = None
    mask_fg_px: torch.Tensor | None = None
    if mask_curr is not None:
        mask_s = _resize(mask_curr.astype(np.float32), "nearest").squeeze() > 0.5
        mask_dt_t = _mask_distance_field(mask_s, device).unsqueeze(0).unsqueeze(0)
        ys, xs = torch.where(mask_s)
        if len(xs) > 0:
            if len(xs) > mask_fg_max:
                sel = torch.randperm(len(xs), device=device)[:mask_fg_max]
                xs, ys = xs[sel], ys[sel]
            mask_fg_px = torch.stack([xs.float(), ys.float()], dim=-1)

    normals = _estimate_normals(pts)
    inv_pose_prev = torch.linalg.inv(torch.from_numpy(abs_pose_prev).float().to(device))

    with torch.no_grad():
        pts_c = torch.from_numpy(points_curr).float().to(device)
        col_c = torch.from_numpy(diffuse_curr).float().to(device)
        normals_c = _estimate_normals(pts_c)
        view_dirs_c = F.normalize(pts_c, dim=-1, eps=1e-8)
        colors_c_relit = _relight(col_c, normals_c, view_dirs_c, env_map_curr, alpha)

        uv_c = _project_uv_norm(pts_c, K_t, H, W)
        valid_c = (
            (pts_c[:, 2] > 1e-3) & (uv_c[:, 0].abs() <= 1.0) & (uv_c[:, 1].abs() <= 1.0)
        )
        tgt_scatter = _scatter_to_tensor(uv_c[valid_c], colors_c_relit[valid_c], H, W)
        tgt_rendered = _gaussian_blur(tgt_scatter, sigma=blur_sigma).clamp(0.0, 1.0)

    tgt_nchw = tgt_rendered.permute(2, 0, 1).unsqueeze(0)

    pose_init = torch.from_numpy(pose_coarse).float().to(device)
    with torch.no_grad():
        pts_coarse = _transform_points(pts, pose_init @ inv_pose_prev)
        valid_init = _valid_mask(pts_coarse, K_t, H, W)

    if valid_init.sum() < 10:
        raise ValueError("fewer than 10 source points visible at coarse pose")

    ctx = PhotometricContext(
        pts_v=pts[valid_init],
        dif_v=dif[valid_init],
        normals_v=normals[valid_init],
        env_map_prev=env_map_prev,
        alpha=alpha,
        inv_pose_prev=inv_pose_prev,
        K_t=K_t,
        H=H,
        W=W,
        tgt_nchw=tgt_nchw,
        depth_t=depth_t,
        mask_dt_t=mask_dt_t,
        mask_fg_px=mask_fg_px,
        blur_sigma=blur_sigma,
        R0_T=pose_init[:3, :3].T.detach(),
        t0=pose_init[:3, 3].detach(),
        lambda_depth=lambda_depth,
        lambda_mask=lambda_mask,
        lambda_rot=lambda_rot,
        lambda_trans=lambda_trans,
    )
    return ctx, tgt_rendered


# ── main refinement ─────────────────────────────────────────────────────────────


def refine_pose_photometric(
    points_prev: np.ndarray,  # [N, 3] camera-space points, prev frame
    diffuse_prev: np.ndarray,  # [N, 3] linear float [0, 1]
    env_map_prev: torch.Tensor | None,
    points_curr: np.ndarray,  # [M, 3] camera-space points, curr frame
    diffuse_curr: np.ndarray,  # [M, 3] linear float [0, 1]
    env_map_curr: torch.Tensor | None,
    K: np.ndarray,
    abs_pose_prev: np.ndarray,
    pose_coarse: np.ndarray,
    alpha: float,
    depth_curr: np.ndarray | None = None,
    mask_curr: np.ndarray | None = None,
    num_iters: int = 120,
    lr_rot: float = 5e-3,
    lr_trans: float = 0.0,  # rotation-only refine (translation kept at LoFTR coarse)
    lambda_depth: float = 0.1,
    lambda_mask: float = 0.05,
    lambda_rot: float = 0.0,
    lambda_trans: float = 0.0,
    scales: tuple[float, ...] = (0.25, 0.5, 1.0),
    blur_sigmas: tuple[float, ...] = (3.0, 2.0, 1.5),
    patience: int = 15,
    patience_loss: int = 10,
    min_rel_improve: float = 5e-3,
    tol_trans_m: float = 5e-5,
    tol_rot_deg: float = 0.02,
    viewer: PhotometricRefineViewer | None = None,
    update_every: int = 1,
    gt_pose_curr: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Gradient-based photometric pose refinement (coarse-to-fine).

    Both the source (prev SLF at candidate pose) and the target (curr SLF) are
    scatter-rendered and Gaussian-blurred before the MSE loss is computed, so
    splatting aliasing is symmetric and cancels out.

    Gradient path: the loss is computed via F.grid_sample on both blurred images
    with the *same* UV grid derived from the current pose; autograd differentiates
    through the grid argument (image-gradient × UV-jacobian), giving pose gradients
    even though the scatter operation itself is non-differentiable.

    ``scales`` runs an image pyramid coarse→fine: each level resizes the target /
    depth / mask (and rescales ``K``) and optimizes up to ``num_iters`` steps,
    carrying the pose forward.  A level stops early once the loss saturates (no
    relative improvement > ``min_rel_improve`` for ``patience`` steps), which
    advances to the finer level sooner.  Coarser levels have wider, smoother
    basins; finer levels sharpen.

    Parameterisation: the candidate pose is the coarse pose plus a rotation
    correction *about the object centroid* and a pure translation correction
    ``dt``.  Rotating about the centroid keeps rotation from injecting
    translation (the lever-arm coupling that otherwise drifts the centre), and
    makes ``lambda_trans * ‖dt‖²`` a clean penalty on the translation
    correction.  A cosine LR decay within each level removes the coarse-level
    overshoot.  Returns ``(refined_abs_pose [4,4] float64, loss_history)``.
    """
    device = "cuda"

    pose_coarse_t = torch.from_numpy(pose_coarse).float().to(device)
    R_coarse = pose_coarse_t[:3, :3].detach()
    t_coarse = pose_coarse_t[:3, 3].detach()

    # object centroid in current-camera space at the coarse pose = rotation pivot
    inv_prev = torch.linalg.inv(torch.from_numpy(abs_pose_prev).float().to(device))
    pts_all = torch.from_numpy(points_prev).float().to(device)
    T_rel0 = pose_coarse_t @ inv_prev
    centroid = ((T_rel0[:3, :3] @ pts_all.T).T + T_rel0[:3, 3]).mean(0).detach()

    eye3 = torch.eye(3, device=device)
    rot_6d = matrix_to_rotation_6d(eye3).clone().detach().requires_grad_(True)
    dt = torch.zeros(3, device=device, requires_grad=True)
    optimizer = torch.optim.Adam(
        [{"params": [rot_6d], "lr": lr_rot}, {"params": [dt], "lr": lr_trans}]
    )

    def _compose() -> torch.Tensor:
        dR = rotation_6d_to_matrix(rot_6d)
        pose = torch.eye(4, device=device)
        pose[:3, :3] = dR @ R_coarse
        pose[:3, 3] = dR @ (t_coarse - centroid) + centroid + dt
        return pose

    best_pose: np.ndarray = pose_coarse.copy()
    loss_history: list[float] = []

    for li, scale in enumerate(scales):
        blur_sigma = blur_sigmas[li] if li < len(blur_sigmas) else 2.0
        try:
            ctx, tgt_rendered = build_photometric_context(
                points_prev=points_prev,
                diffuse_prev=diffuse_prev,
                env_map_prev=env_map_prev,
                points_curr=points_curr,
                diffuse_curr=diffuse_curr,
                env_map_curr=env_map_curr,
                K=K,
                abs_pose_prev=abs_pose_prev,
                pose_coarse=pose_coarse,
                alpha=alpha,
                depth_curr=depth_curr,
                mask_curr=mask_curr,
                lambda_depth=lambda_depth,
                lambda_mask=lambda_mask,
                lambda_rot=0.0,  # reg applied below on the corrections directly
                lambda_trans=0.0,
                scale=scale,
                blur_sigma=blur_sigma,
                device=device,
            )
        except ValueError:
            continue

        # Two early-stop criteria — whichever fires first cuts to the next level:
        #   1. Loss plateau: no relative improvement > min_rel_improve for patience_loss steps.
        #   2. Pose convergence: both corrections stop moving (original criterion).
        stale = 0
        no_improve = 0
        best_level_loss = float("inf")
        prev_dt = dt.detach().clone()
        prev_r6 = rot_6d.detach().clone()
        tol_rot6 = float(np.radians(tol_rot_deg))
        for step in range(num_iters):
            optimizer.zero_grad()

            pose_curr = _compose()
            loss, _components, aux = photometric_forward(ctx, pose_curr)

            # penalise the corrections (anchored at the coarse pose)
            dR = pose_curr[:3, :3] @ R_coarse.T
            rot_corr = 1.0 - (((dR[0, 0] + dR[1, 1] + dR[2, 2]) - 1.0) / 2.0).clamp(
                -1.0, 1.0
            )
            loss = loss + lambda_rot * rot_corr + lambda_trans * (dt**2).sum()

            loss_val = loss.item()
            loss_history.append(loss_val)
            best_pose = pose_curr.detach().cpu().numpy().astype(np.float64)

            # loss-plateau check
            if loss_val < best_level_loss * (1.0 - min_rel_improve):
                best_level_loss = loss_val
                no_improve = 0
            else:
                no_improve += 1

            if viewer is not None and step % update_every == 0:
                with torch.no_grad():
                    src_blurred_hwc = aux["src_nchw"].squeeze(0).permute(1, 2, 0)
                    viewer.update(
                        iteration=len(loss_history),
                        loss=loss_val,
                        pts_curr=aux["pts_curr_t"].detach(),
                        colors_relit=aux["colors_relit"].detach(),
                        src_blurred=src_blurred_hwc,
                        tgt_rendered=tgt_rendered,
                        env_map=env_map_prev,
                        gt_pose_curr=gt_pose_curr,
                        pose_curr_np=pose_curr.detach().cpu().numpy(),
                    )

            loss.backward()
            optimizer.step()

            # pose-convergence early stop: both corrections must stop moving
            with torch.no_grad():
                d_trans = (dt - prev_dt).norm().item()
                d_rot6 = (rot_6d - prev_r6).norm().item()
                prev_dt = dt.detach().clone()
                prev_r6 = rot_6d.detach().clone()
            settled = d_trans < tol_trans_m and d_rot6 < tol_rot6
            stale = stale + 1 if settled else 0
            if stale >= patience or no_improve >= patience_loss:
                break

    return best_pose, loss_history
