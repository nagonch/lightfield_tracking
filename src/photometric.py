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

import numpy as np
import torch
import torch.nn.functional as F

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


# ── anti-aliasing: Gaussian blur ────────────────────────────────────────────────


def _gaussian_blur(
    img_hwc: torch.Tensor, kernel_size: int = 9, sigma: float = 2.0
) -> torch.Tensor:
    """Isotropic Gaussian blur on [H, W, C] float tensor."""
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
    uv_n: torch.Tensor, colors: torch.Tensor, H: int, W: int
) -> torch.Tensor:
    """Scatter → Gaussian blur → [1, 3, H, W] (no grad through scatter, used as grid_sample input)."""
    with torch.no_grad():
        scattered = _scatter_to_tensor(uv_n, colors, H, W)
        blurred = _gaussian_blur(scattered).clamp(0.0, 1.0)
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
    num_iters: int = 500,
    lr_rot: float = 1e-3,
    lr_trans: float = 0,
    lambda_depth: float = 0.1,
    lambda_mask: float = 0.05,
    lambda_rot: float = 0.5,
    lambda_trans: float = 0.1,
    viewer: PhotometricRefineViewer | None = None,
    update_every: int = 1,
    gt_pose_curr: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Gradient-based photometric pose refinement.

    Both the source (prev SLF at candidate pose) and the target (curr SLF) are
    scatter-rendered and Gaussian-blurred before the MSE loss is computed, so
    splatting aliasing is symmetric and cancels out.

    Gradient path: the loss is computed via F.grid_sample on both blurred images
    with the *same* UV grid derived from the current pose; autograd differentiates
    through the grid argument (image-gradient × UV-jacobian), giving pose gradients
    even though the scatter operation itself is non-differentiable.

    Returns ``(refined_abs_pose [4,4] float64, loss_history)``.
    """
    device = "cuda"

    if depth_curr is not None:
        H, W = depth_curr.shape[:2]
    elif mask_curr is not None:
        H, W = mask_curr.shape[:2]
    else:
        raise ValueError("depth_curr or mask_curr must be provided to determine H, W")

    pts = torch.from_numpy(points_prev).float().to(device)
    dif = torch.from_numpy(diffuse_prev).float().to(device)
    K_t = torch.from_numpy(K).float().to(device)

    depth_t = (
        torch.from_numpy(depth_curr).float().to(device).unsqueeze(0).unsqueeze(0)
        if depth_curr is not None
        else None
    )
    mask_t = (
        torch.from_numpy(mask_curr.astype(np.float32))
        .to(device)
        .unsqueeze(0)
        .unsqueeze(0)
        if mask_curr is not None
        else None
    )

    normals = _estimate_normals(pts)
    inv_pose_prev = torch.linalg.inv(torch.from_numpy(abs_pose_prev).float().to(device))

    # ── pre-render blurred target from current SLF (fixed, no grad needed) ────
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
        tgt_rendered = _gaussian_blur(tgt_scatter).clamp(0.0, 1.0)  # [H, W, 3]

    tgt_nchw = tgt_rendered.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    # ── fix visible-point set at the coarse pose ───────────────────────────────
    pose_init = torch.from_numpy(pose_coarse).float().to(device)
    with torch.no_grad():
        pts_coarse = _transform_points(pts, pose_init @ inv_pose_prev)
        valid_init = _valid_mask(pts_coarse, K_t, H, W)

    if valid_init.sum() < 10:
        return pose_coarse.copy(), []

    pts_v = pts[valid_init]
    dif_v = dif[valid_init]
    normals_v = normals[valid_init]

    R0_T = pose_init[:3, :3].T.detach()
    t0 = pose_init[:3, 3].detach()

    rot_6d = (
        matrix_to_rotation_6d(pose_init[:3, :3]).clone().detach().requires_grad_(True)
    )
    trans = pose_init[:3, 3].clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam(
        [{"params": [rot_6d], "lr": lr_rot}, {"params": [trans], "lr": lr_trans}]
    )

    best_loss: float = float("inf")
    best_pose: np.ndarray = pose_coarse.copy()
    loss_history: list[float] = []

    for step in range(num_iters):
        optimizer.zero_grad()

        R = rotation_6d_to_matrix(rot_6d)
        pose_curr = torch.eye(4, device=device)
        pose_curr[:3, :3] = R
        pose_curr[:3, 3] = trans
        T_rel = pose_curr @ inv_pose_prev

        pts_curr_t = _transform_points(pts_v, T_rel)
        normals_cur = (T_rel[:3, :3] @ normals_v.T).T
        view_dirs = F.normalize(pts_curr_t, dim=-1, eps=1e-8)
        colors_relit = _relight(dif_v, normals_cur, view_dirs, env_map_prev, alpha)

        uv_n = _project_uv_norm(
            pts_curr_t, K_t, H, W
        )  # [M, 2]  — has grad through pose
        grid = uv_n.reshape(1, 1, -1, 2)

        # ── symmetric anti-aliasing: scatter+blur source, compare to blurred target ──
        # Scatter and blur the source (no grad through scatter; grad flows through grid below)
        src_nchw = _scatter_blur_nchw(
            uv_n, colors_relit, H, W
        )  # [1,3,H,W], no requires_grad

        # Both grid_sample calls give ∂loss/∂grid → ∂grid/∂pose (image-gradient × UV-jacobian)
        sampled_src = (
            F.grid_sample(
                src_nchw,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            .squeeze(0)
            .squeeze(1)
            .T
        )  # [M, 3]
        sampled_tgt = (
            F.grid_sample(
                tgt_nchw,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            .squeeze(0)
            .squeeze(1)
            .T
        )  # [M, 3]

        in_front = (pts_curr_t[:, 2] > 1e-3).float().detach()
        n_valid = in_front.sum().clamp(min=1.0)

        # ── photometric MSE on blurred images ──────────────────────────────────
        loss = (
            (sampled_src - sampled_tgt) ** 2 * in_front.unsqueeze(-1)
        ).sum() / n_valid

        # ── depth anchoring ────────────────────────────────────────────────────
        if depth_t is not None and lambda_depth > 0.0:
            sampled_depth = F.grid_sample(
                depth_t, grid, mode="bilinear", padding_mode="zeros", align_corners=True
            ).reshape(-1)
            depth_valid = in_front * (sampled_depth > 1e-3).float().detach()
            n_depth = depth_valid.sum().clamp(min=1.0)
            depth_loss = (
                (pts_curr_t[:, 2] - sampled_depth).abs() * depth_valid
            ).sum() / n_depth
            loss = loss + lambda_depth * depth_loss

        # ── mask alignment ─────────────────────────────────────────────────────
        if mask_t is not None and lambda_mask > 0.0:
            sampled_mask = F.grid_sample(
                mask_t, grid, mode="bilinear", padding_mode="zeros", align_corners=True
            ).reshape(-1)
            loss = (
                loss + lambda_mask * ((1.0 - sampled_mask) * in_front).sum() / n_valid
            )

        # ── pose regularisation ────────────────────────────────────────────────
        R_diff = R @ R0_T
        cos_angle = ((R_diff[0, 0] + R_diff[1, 1] + R_diff[2, 2]) - 1.0) / 2.0
        rot_reg = 1.0 - cos_angle.clamp(-1.0, 1.0)
        trans_reg = ((trans - t0) ** 2).sum()
        loss = loss + lambda_rot * rot_reg + lambda_trans * trans_reg

        loss_val = loss.item()
        loss_history.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_pose = pose_curr.detach().cpu().numpy().astype(np.float64)

        if viewer is not None and step % update_every == 0:
            with torch.no_grad():
                # Recover blurred source as HWC for display
                src_blurred_hwc = src_nchw.squeeze(0).permute(1, 2, 0)
                viewer.update(
                    iteration=step,
                    loss=loss_val,
                    pts_curr=pts_curr_t.detach(),
                    colors_relit=colors_relit.detach(),
                    src_blurred=src_blurred_hwc,
                    tgt_rendered=tgt_rendered,
                    env_map=env_map_prev,
                    gt_pose_curr=gt_pose_curr,
                    pose_curr_np=pose_curr.detach().cpu().numpy(),
                )

        loss.backward()
        optimizer.step()

    return best_pose, loss_history
