"""Photometric pose refinement: transform + relight the previous SLF, match current view.

Per-step:
  1. Rotate/translate previous camera-space points with the candidate pose.
  2. Relight each point: alpha * diffuse + (1-alpha) * env_map[reflected_dir].
  3. Project transformed points and sample the current diffuse view at those positions.
  4. Minimise a composite loss:
       photometric MSE   (primary)
     + depth anchoring   (L1 between rendered z and target depth, small weight)
     + mask alignment    (penalise points projecting outside the target mask)
     + pose magnitude    (angular + translation regularisation from the coarse init)

Gradients flow through the perspective projection, so the pose parameters are
updated toward the photometrically consistent estimate.  The best-loss pose is
returned (not the final iterate) for robustness.

Refinement is intentionally small: the regularisation terms bound the rotation
to minor angular corrections from the coarse pose (mostly in-plane adjustments)
and keep translation perturbations sub-centimetre for typical scene scales.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from loss import rotation_6d_to_matrix, matrix_to_rotation_6d
from src.surface_light_field import _estimate_normals

# ── env-map sampling ────────────────────────────────────────────────────────────


def _sample_env_map(env_map_hwc: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Bilinear equirect lookup matching flip_u=True, flip_v=True convention.

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
    u_px = (env_w - 1) - u_px  # flip_u=True
    v_px = (0.5 + lat / torch.pi) * (env_h - 1)  # flip_v=True

    u_n = u_px / (env_w - 1) * 2.0 - 1.0
    v_n = v_px / (env_h - 1) * 2.0 - 1.0
    grid = torch.stack([u_n, v_n], dim=-1).reshape(1, 1, -1, 2)

    img = env_map_hwc.permute(2, 0, 1).unsqueeze(0).float()  # [1, 3, H, W]
    out = F.grid_sample(
        img, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return out.squeeze(0).squeeze(1).T  # [N, 3]


# ── relighting ──────────────────────────────────────────────────────────────────


def _relight(
    diffuse: torch.Tensor,  # [N, 3] float [0, 1]
    normals: torch.Tensor,  # [N, 3] unit
    view_dirs: torch.Tensor,  # [N, 3] camera→point unit
    env_map: torch.Tensor | None,  # [H, W, 3] linear or None
    alpha: float,
) -> torch.Tensor:  # [N, 3] float [0, 1]
    """Alpha * diffuse + (1-alpha) * env-map specular."""
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
    """[N, 3] × [4, 4] → [N, 3]."""
    pts_h = torch.cat([pts, pts.new_ones(len(pts), 1)], dim=-1)
    return (T.float() @ pts_h.T).T[:, :3]


def _project_uv_norm(
    pts: torch.Tensor, K: torch.Tensor, H: int, W: int
) -> torch.Tensor:
    """[N, 3] → normalised grid coords [N, 2] in [-1, 1] for F.grid_sample."""
    z = pts[:, 2].clamp(min=1e-6)
    u = pts[:, 0] / z * K[0, 0] + K[0, 2]
    v = pts[:, 1] / z * K[1, 1] + K[1, 2]
    return torch.stack([u / (W - 1) * 2.0 - 1.0, v / (H - 1) * 2.0 - 1.0], dim=-1)


def _valid_mask(pts: torch.Tensor, K: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Bool mask: in-front-of-camera and within image bounds."""
    uv_n = _project_uv_norm(pts, K, H, W)
    return (pts[:, 2] > 1e-3) & (uv_n[:, 0].abs() <= 1.0) & (uv_n[:, 1].abs() <= 1.0)


# ── display scatter helper ──────────────────────────────────────────────────────


@torch.no_grad()
def _scatter_to_image(
    uv_n: torch.Tensor, colors: torch.Tensor, H: int, W: int
) -> np.ndarray:
    """Scatter point colours into a display image (non-differentiable)."""
    u_px = ((uv_n[:, 0] + 1.0) * 0.5 * (W - 1)).long().clamp(0, W - 1)
    v_px = ((uv_n[:, 1] + 1.0) * 0.5 * (H - 1)).long().clamp(0, H - 1)
    flat = (v_px * W + u_px).unsqueeze(-1)  # [N, 1]
    img = torch.zeros(H * W, 3, device=colors.device)
    cnt = torch.zeros(H * W, 1, device=colors.device)
    img.scatter_add_(0, flat.expand(-1, 3), colors.float().clamp(0, 1))
    cnt.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
    img = img / cnt.clamp(min=1.0)
    return (img.reshape(H, W, 3).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


# ── viser visualizer ────────────────────────────────────────────────────────────


class PhotometricRefineViewer:
    """Lightweight viser viewer for photometric refinement.

    Shows: rotating relighted point cloud, rendered / target / loss images as
    camera frustums, the environment map, and live loss + iteration counters.
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

    def reset_frame(self, frame_idx: int) -> None:
        self._iter_h.value = 0
        self._loss_h.value = 0.0

    @torch.no_grad()
    def update(
        self,
        iteration: int,
        loss: float,
        pts_curr: torch.Tensor,  # [N, 3] in current cam space
        colors_relit: torch.Tensor,  # [N, 3] float [0, 1]
        uv_n: torch.Tensor,  # [N, 2] normalised grid coords
        target_img: np.ndarray,  # [H, W, 3] uint8
        env_map: torch.Tensor | None,
    ) -> None:
        self._iter_h.value = int(iteration)
        self._loss_h.value = float(loss)

        H, W = target_img.shape[:2]

        # 3D point cloud of the transformed (relighted) SLF
        self.server.scene.add_point_cloud(
            name="photometric/slf",
            points=pts_curr.float().cpu().numpy(),
            colors=colors_relit.float().clamp(0, 1).cpu().numpy(),
            point_size=0.002,
        )

        # Images as camera frustums
        rendered = _scatter_to_image(uv_n, colors_relit, H, W)
        loss_gray = (
            np.abs(rendered.astype(np.float32) - target_img.astype(np.float32))
            .mean(axis=-1)
            .clip(0, 255)
            .astype(np.uint8)
        )
        loss_img = np.repeat(loss_gray[:, :, np.newaxis], 3, axis=-1)  # B&W as RGB
        fov = float(2.0 * np.arctan2(W / 2.0, float(max(W, 1))))
        for name, img, x_off in [
            ("photometric/rendered", rendered, -0.10),
            ("photometric/target", target_img, 0.10),
            ("photometric/loss_img", loss_img, 0.0),
        ]:
            self.server.scene.add_camera_frustum(
                name=name,
                fov=fov,
                aspect=W / max(H, 1),
                scale=0.03,
                image=img,
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(x_off, -0.02, 0.02),
            )

        if env_map is not None:
            env_np = env_map.detach().float().cpu().numpy()
            if env_np.ndim == 3 and env_np.shape[0] == 3:
                env_np = env_np.transpose(1, 2, 0)
            env_u8 = (np.clip(env_np, 0, 1) * 255).astype(np.uint8)
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

    def close(self) -> None:
        try:
            self.server.stop()
        except Exception:
            pass


# ── main refinement ─────────────────────────────────────────────────────────────


def refine_pose_photometric(
    points_prev: np.ndarray,  # [N, 3] camera-space points, prev frame
    diffuse_prev: np.ndarray,  # [N, 3] float32 [0,1] sRGB diffuse colours
    env_map_prev: torch.Tensor | None,  # [H_e, W_e, 3] linear float, or None
    target_img: np.ndarray,  # [H, W, 3] float32 [0,1] raw central view
    K: np.ndarray,  # [3, 3] camera intrinsics
    abs_pose_prev: np.ndarray,  # [4, 4] absolute pose of prev frame
    pose_coarse: np.ndarray,  # [4, 4] coarse absolute pose (init)
    alpha: float,
    depth_curr: np.ndarray | None = None,  # [H, W] float32 depth of current frame
    mask_curr: np.ndarray | None = None,  # [H, W] bool object mask of current frame
    num_iters: int = 500,
    lr_rot: float = 1e-3,
    lr_trans: float = 1e-3,
    lambda_depth: float = 0.1,  # depth anchoring weight (secondary)
    lambda_mask: float = 0.05,  # mask alignment weight
    lambda_rot: float = 0.5,  # rotation magnitude regularisation
    lambda_trans: float = 0.1,  # translation magnitude regularisation
    viewer: PhotometricRefineViewer | None = None,
    update_every: int = 1,
) -> tuple[np.ndarray, list[float]]:
    """Gradient-based photometric pose refinement.

    Compares the relighted previous SLF (alpha*diffuse + (1-alpha)*specular) against
    the raw central view of the current frame.

    Loss = MSE_photo
         + lambda_depth  * L1_depth      (rendered z vs sampled target depth)
         + lambda_mask   * mask_penalty  (silhouette outside target mask)
         + lambda_rot    * rotation_reg  (angular distance from coarse init)
         + lambda_trans  * trans_reg     (L2 translation from coarse init)

    The depth and mask terms anchor the solution geometrically while keeping
    photometric error as the dominant signal.  The regularisation terms bound
    the refinement to small angular corrections from the coarse estimate.

    Returns ``(refined_abs_pose [4,4] float64, loss_history)``.  The best-loss
    iterate is returned, not the final one.
    """
    device = "cuda"

    pts = torch.from_numpy(points_prev).float().to(device)  # [N, 3]
    dif = torch.from_numpy(diffuse_prev).float().to(device)  # [N, 3]
    tgt = torch.from_numpy(target_img).float().to(device)  # [H, W, 3]
    K_t = torch.from_numpy(K).float().to(device)
    H, W = tgt.shape[:2]
    tgt_nchw = tgt.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    tgt_u8 = (target_img * 255).clip(0, 255).astype(np.uint8)

    # Optional geometric anchors
    depth_t = (
        torch.from_numpy(depth_curr).float().to(device).unsqueeze(0).unsqueeze(0)
        if depth_curr is not None
        else None
    )  # [1, 1, H, W]
    mask_t = (
        torch.from_numpy(mask_curr.astype(np.float32))
        .to(device)
        .unsqueeze(0)
        .unsqueeze(0)
        if mask_curr is not None
        else None
    )  # [1, 1, H, W]

    normals = _estimate_normals(pts)  # [N, 3]

    inv_pose_prev = torch.linalg.inv(torch.from_numpy(abs_pose_prev).float().to(device))

    # Fix the visible-point set at the coarse pose so gradient steps cannot
    # trivially reduce the loss by moving all points out of view.
    pose_init = torch.from_numpy(pose_coarse).float().to(device)
    with torch.no_grad():
        T_coarse = pose_init @ inv_pose_prev
        pts_coarse = _transform_points(pts, T_coarse)
        valid_init = _valid_mask(pts_coarse, K_t, H, W)

    if valid_init.sum() < 10:
        return pose_coarse.copy(), []

    pts_v = pts[valid_init]  # [M, 3]
    dif_v = dif[valid_init]  # [M, 3]
    normals_v = normals[valid_init]  # [M, 3]

    # Anchor values: coarse rotation and translation (fixed, no gradient).
    R0_T = pose_init[:3, :3].T.detach()  # R_init^T for angular distance
    t0 = pose_init[:3, 3].detach()  # init translation

    # Optimisable pose parameters (6D rotation + translation)
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

        # Transform points + normals into current camera space
        pts_curr = _transform_points(pts_v, T_rel)  # [M, 3]
        normals_cur = (T_rel[:3, :3] @ normals_v.T).T  # [M, 3]
        view_dirs = F.normalize(pts_curr, dim=-1, eps=1e-8)

        # Relight: alpha * diffuse + (1-alpha) * env_map specular
        colors_relit = _relight(dif_v, normals_cur, view_dirs, env_map_prev, alpha)

        # Project and sample target image at projected positions
        uv_n = _project_uv_norm(pts_curr, K_t, H, W)  # [M, 2]
        grid = uv_n.reshape(1, 1, -1, 2)  # [1, 1, M, 2]

        target_colors = (
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

        # In-front weight (detached to avoid degenerate collapse)
        in_front = (pts_curr[:, 2] > 1e-3).float().detach()  # [M]
        n_valid = in_front.sum().clamp(min=1.0)

        # ── photometric MSE (primary) ──────────────────────────────────────────
        loss = (
            (colors_relit - target_colors) ** 2 * in_front.unsqueeze(-1)
        ).sum() / n_valid

        # ── depth anchoring (L1, secondary) ───────────────────────────────────
        if depth_t is not None and lambda_depth > 0.0:
            sampled_depth = F.grid_sample(
                depth_t, grid, mode="bilinear", padding_mode="zeros", align_corners=True
            ).reshape(
                -1
            )  # [M]
            depth_valid = in_front * (sampled_depth > 1e-3).float().detach()
            n_depth = depth_valid.sum().clamp(min=1.0)
            depth_loss = (
                (pts_curr[:, 2] - sampled_depth).abs() * depth_valid
            ).sum() / n_depth
            loss = loss + lambda_depth * depth_loss

        # ── mask alignment ─────────────────────────────────────────────────────
        # Points should project inside the target silhouette.  We sample the
        # (soft) mask at each projected position and penalise values < 1.
        if mask_t is not None and lambda_mask > 0.0:
            sampled_mask = F.grid_sample(
                mask_t, grid, mode="bilinear", padding_mode="zeros", align_corners=True
            ).reshape(
                -1
            )  # [M]  ∈ [0, 1]
            mask_loss = ((1.0 - sampled_mask) * in_front).sum() / n_valid
            loss = loss + lambda_mask * mask_loss

        # ── pose magnitude regularisation ──────────────────────────────────────
        # Angular distance from the coarse init: (trace(R · R0^T) - 1) / 2 = cos(θ).
        # We penalise (1 - cos θ), which is 0 at zero rotation and ~1 at 90°.
        R_diff = R @ R0_T
        cos_angle = ((R_diff[0, 0] + R_diff[1, 1] + R_diff[2, 2]) - 1.0) / 2.0
        rot_reg = 1.0 - cos_angle.clamp(-1.0, 1.0)

        # Translation deviation from init (squared L2)
        trans_reg = ((trans - t0) ** 2).sum()

        loss = loss + lambda_rot * rot_reg + lambda_trans * trans_reg

        loss_val = loss.item()
        loss_history.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_pose = pose_curr.detach().cpu().numpy().astype(np.float64)

        if viewer is not None and step % update_every == 0:
            with torch.no_grad():
                viewer.update(
                    iteration=step,
                    loss=loss_val,
                    pts_curr=pts_curr.detach(),
                    colors_relit=colors_relit.detach(),
                    uv_n=uv_n.detach(),
                    target_img=tgt_u8,
                    env_map=env_map_prev,
                )

        loss.backward()
        optimizer.step()

    return best_pose, loss_history
