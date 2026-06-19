"""Photometric pose refinement via differentiable Gaussian-splat rendering.

Transforms prev-frame SLF Gaussians to the current camera under a candidate pose,
renders via gsplat (with optional relighting), and minimises a composite loss:
  photometric MSE (linear, object region) + L1 depth anchor + pose regularisation.
Pose gradient flows through Gaussian means (geometry) and analytically-evaluated
per-point colours (appearance).
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import torch

from src.rotation import rotation_6d_to_matrix, matrix_to_rotation_6d
from src.surface_light_field import SurfaceLightField
from utils import linear_to_srgb

# ── geometry helpers ────────────────────────────────────────────────────────────


def _rot_err_deg_np(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic rotation error (degrees) between two 3x3 matrices."""
    c = np.clip((np.trace(Ra @ Rb.T) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


# ── display helpers ──────────────────────────────────────────────────────────────


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
    """Viser viewer: rendered source/target/loss frustums + live loss/error curves.
    Optionally attaches an interactive gsplat viewer of the source SLF at the candidate pose.
    """

    def __init__(self, port: int = 8081, show_gaussians: bool = True):
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

        self._gs_viewer = None
        self._gs = None  # dict of live gaussian tensors (means/harmonics/...)
        if show_gaussians:
            self._init_gaussian_viewer()

    # ── interactive gsplat viewer (nerfview) ────────────────────────────────
    def _init_gaussian_viewer(self) -> None:
        import tempfile
        from pathlib import Path

        import torch as _torch
        from gsplat import rasterization

        from src.gaussian_splatting.gsplat_viewer import (
            GsplatViewer,
            GsplatRenderTabState,
        )

        device = _torch.device("cuda")

        @_torch.no_grad()
        def _render_fn(camera_state, render_tab_state):
            if render_tab_state.preview_render:
                W, H = render_tab_state.render_width, render_tab_state.render_height
            else:
                W, H = render_tab_state.viewer_width, render_tab_state.viewer_height
            gs = self._gs
            if gs is None:
                bg = np.array(render_tab_state.backgrounds, dtype=np.float32) / 255.0
                return np.tile(bg, (H, W, 1))

            c2w = _torch.from_numpy(camera_state.c2w).float().to(device)
            K = _torch.from_numpy(camera_state.get_K((W, H))).float().to(device)
            sh_degree = int(np.sqrt(gs["harmonics"].shape[-2]) - 1)
            colors, _alphas, info = rasterization(
                gs["means"],
                gs["quats"],
                gs["scales"],
                gs["opacities"],
                gs["harmonics"],
                c2w.inverse()[None],
                K[None],
                W,
                H,
                sh_degree=min(render_tab_state.max_sh_degree, sh_degree),
                near_plane=render_tab_state.near_plane,
                far_plane=render_tab_state.far_plane,
                radius_clip=render_tab_state.radius_clip,
                eps2d=render_tab_state.eps2d,
                backgrounds=_torch.tensor([render_tab_state.backgrounds], device=device)
                / 255.0,
                render_mode="RGB",
                rasterize_mode=render_tab_state.rasterize_mode,
                camera_model=render_tab_state.camera_model,
                packed=False,
            )
            render_tab_state.total_gs_count = gs["means"].shape[0]
            render_tab_state.rendered_gs_count = (
                (info["radii"] > 0).all(-1).sum().item()
            )
            img = _torch.clip(colors[0, ..., :3], 0.0, 1.0)
            return linear_to_srgb(img).cpu().numpy()

        self._GsplatRenderTabState = GsplatRenderTabState
        self._gs_viewer = GsplatViewer(
            server=self.server,
            render_fn=_render_fn,
            output_dir=Path(tempfile.mkdtemp(prefix="relift_gs_")),
            mode="rendering",
        )

    @torch.no_grad()
    def _set_gaussians(
        self,
        slf_prev: "SurfaceLightField",
        rel_pose: torch.Tensor,
    ) -> None:
        """Refresh the live gaussian set: source SLF only, at the candidate pose."""
        R, t = rel_pose[:3, :3], rel_pose[:3, 3]
        src_means = (R @ slf_prev.points.float().T).T + t  # prev → curr cam space
        # Isotropic (spherical) Gaussians → rotation leaves quats/scales unchanged.
        self._gs = {
            "means": src_means,
            "harmonics": slf_prev.harmonics.float(),
            "quats": slf_prev.quats.float(),
            "scales": slf_prev.scales.float(),
            "opacities": slf_prev.opacities.float(),
        }
        if self._gs_viewer is not None:
            self._gs_viewer.rerender(None)

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
        src_img: torch.Tensor,  # [H, W, 3] linear rendered source
        tgt_img: torch.Tensor,  # [H, W, 3] linear rendered target
        env_map: torch.Tensor | None,
        gt_pose_curr: np.ndarray | None = None,
        pose_curr_np: np.ndarray | None = None,
        slf_prev: "SurfaceLightField | None" = None,
        rel_pose: torch.Tensor | None = None,
    ) -> None:
        self._iter_h.value = int(iteration)
        self._loss_h.value = float(loss)

        if (
            self._gs_viewer is not None
            and slf_prev is not None
            and rel_pose is not None
        ):
            self._set_gaussians(slf_prev, rel_pose)

        H, W = tgt_img.shape[:2]

        rendered_u8 = _to_display_u8(src_img)
        target_u8 = _to_display_u8(tgt_img)
        loss_diff = torch.abs(src_img - tgt_img)  # ** 0.8
        loss_diff_normalized = loss_diff / (loss_diff.max() + 1e-8)
        loss_gray = _to_display_u8(
            loss_diff_normalized.mean(axis=-1, keepdim=False)
        ).astype(np.uint8)
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
        if self._gs_viewer is not None:
            try:
                self._gs_viewer.close()
            except Exception:
                pass
        try:
            self.server.stop()
        except Exception:
            pass


# ── shared loss forward ──────────────────────────────────────────────────────────


@dataclass
class PhotometricContext:
    """Fixed inputs for one photometric loss evaluation (one frame pair, one pyramid level).
    Only pose_curr varies; built once by build_photometric_context.
    """

    slf_prev: SurfaceLightField  # source Gaussians (prev camera space)
    env_map_prev: torch.Tensor | None
    # [H, W, 3] source env observation-confidence broadcast (None → uniform). Used
    # to downweight reflected-into-UNOBSERVED env pixels in the relit photo loss, so
    # only env signal that was actually observed drives the pose. See conf_floor.
    env_conf3_prev: (
        torch.Tensor | None
    )  # [H,W,3] env observation confidence; None → uniform
    conf_floor: float  # weight floor for fully-unobserved env pixels
    alpha: float
    mode: str  # render appearance model ("diffuse" | "relit" | "sh" | "auto")
    inv_pose_prev: torch.Tensor  # [4, 4] camera_prev → object (prev)
    scale: float  # pyramid render scale (≤ 1)
    H: int  # rendered height at this scale
    W: int  # rendered width at this scale
    tgt_img: torch.Tensor  # [H, W, 3] rendered target (curr SLF), detached
    tgt_depth: torch.Tensor  # [H, W] rendered target depth, detached
    roi: torch.Tensor  # [H, W] float weight for the photometric region
    R0_T: torch.Tensor  # [3, 3] anchor rotation^T for pose reg
    t0: torch.Tensor  # [3] anchor translation for pose reg
    lambda_depth: float
    lambda_rot: float
    lambda_trans: float


def photometric_forward(
    ctx: PhotometricContext, pose_curr: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Evaluate the composite photometric loss at one candidate pose.

    Returns ``(loss, components, aux)`` where ``components`` holds each additive
    term (``photo``/``depth``/``rot``/``trans``) *before* its lambda weight, and
    ``aux`` carries the rendered source image/depth/mask for display.
    """
    T_rel = pose_curr @ ctx.inv_pose_prev

    src_img, src_depth, src_mask = ctx.slf_prev.render_relit(
        rel_pose=T_rel,
        env_map=ctx.env_map_prev,
        alpha=ctx.alpha,
        scale=ctx.scale,
        mode=ctx.mode,
    )

    # Union of target region and source footprint: both silhouettes carry gradient.
    w = (ctx.roi + src_mask.float()).clamp(max=1.0)

    # Relit stage: downweight pixels whose reflected ray samples unobserved env.
    # Confidence rendered via the same equirect path, detached (stop-gradient mask).
    if ctx.env_conf3_prev is not None:
        with torch.no_grad():
            conf_img, _, _ = ctx.slf_prev.render_relit(
                rel_pose=T_rel,
                env_map=ctx.env_conf3_prev,
                alpha=0.0,
                scale=ctx.scale,
                mode="relit",
                shade_diffuse=False,
            )
        conf_w = ctx.conf_floor + (1.0 - ctx.conf_floor) * conf_img[..., 0]
        w = w * conf_w.detach()

    w_sum = w.sum().clamp(min=1.0)
    photo = (((src_img - ctx.tgt_img) ** 2).mean(-1) * w).sum() / w_sum

    components: dict[str, torch.Tensor] = {"photo": photo}
    loss = photo

    if ctx.lambda_depth > 0.0:
        both = (src_depth > 1e-3) & (ctx.tgt_depth > 1e-3)
        wd = both.float() * w
        nd = wd.sum().clamp(min=1.0)
        depth_loss = ((src_depth - ctx.tgt_depth).abs() * wd).sum() / nd
        components["depth"] = depth_loss
        loss = loss + ctx.lambda_depth * depth_loss

    R = pose_curr[:3, :3]
    R_diff = R @ ctx.R0_T
    cos_angle = ((R_diff[0, 0] + R_diff[1, 1] + R_diff[2, 2]) - 1.0) / 2.0
    rot_reg = 1.0 - cos_angle.clamp(-1.0, 1.0)
    trans_reg = ((pose_curr[:3, 3] - ctx.t0) ** 2).sum()
    components["rot"] = rot_reg
    components["trans"] = trans_reg
    loss = loss + ctx.lambda_rot * rot_reg + ctx.lambda_trans * trans_reg

    aux = {"src_img": src_img, "src_depth": src_depth, "src_mask": src_mask}
    return loss, components, aux


def build_photometric_context(
    slf_prev: SurfaceLightField,
    slf_curr: SurfaceLightField,
    env_map_prev: torch.Tensor | None,
    env_map_curr: torch.Tensor | None,
    abs_pose_prev: np.ndarray,
    pose_coarse: np.ndarray,
    alpha: float,
    lambda_depth: float = 0.1,
    lambda_rot: float = 0.0,
    lambda_trans: float = 0.0,
    scale: float = 1.0,
    device: str = "cuda",
    mode: str = "auto",
    env_conf_prev: torch.Tensor | None = None,
    conf_floor: float = 0.1,
) -> tuple[PhotometricContext, torch.Tensor]:
    """Build a PhotometricContext and render the target [H,W,3] once at ``scale``.
    Returns (ctx, tgt_img_hwc).
    """
    inv_pose_prev = torch.linalg.inv(torch.from_numpy(abs_pose_prev).float().to(device))

    with torch.no_grad():
        tgt_img, tgt_depth, tgt_mask = slf_curr.render_relit(
            rel_pose=None, env_map=env_map_curr, alpha=alpha, scale=scale, mode=mode
        )
        roi = tgt_mask.float()

    env_conf3_prev = None  # confidence masking only applies to the relit stage
    if mode == "relit" and env_conf_prev is not None and env_map_prev is not None:
        c = env_conf_prev.to(device=device, dtype=torch.float32)
        env_conf3_prev = c[..., None].expand(-1, -1, 3).contiguous()

    pose_init = torch.from_numpy(pose_coarse).float().to(device)
    ctx = PhotometricContext(
        slf_prev=slf_prev,
        env_map_prev=env_map_prev,
        env_conf3_prev=env_conf3_prev,
        conf_floor=conf_floor,
        alpha=alpha,
        mode=mode,
        inv_pose_prev=inv_pose_prev,
        scale=scale,
        H=tgt_img.shape[0],
        W=tgt_img.shape[1],
        tgt_img=tgt_img.detach(),
        tgt_depth=tgt_depth.detach(),
        roi=roi,
        R0_T=pose_init[:3, :3].T.detach(),
        t0=pose_init[:3, 3].detach(),
        lambda_depth=lambda_depth,
        lambda_rot=lambda_rot,
        lambda_trans=lambda_trans,
    )
    return ctx, tgt_img.detach()


# ── main refinement ─────────────────────────────────────────────────────────────


@dataclass
class RefineConfig:
    """Hyperparameters for two-stage photometric pose refinement.

    Stage 1 (diffuse): always run; matches separated diffuse appearance.
    Stage 2 (relight): reflective frames only; matches alpha·diffuse + (1-alpha)·env[reflect].
    All defaults mirror config.yaml; see config.yaml for rationale.
    """

    # optimisation (per pyramid level)
    num_iters: int = 60
    lr_rot: float = 5e-3
    lr_trans: float = (
        1e-4  # free dt avoids rotation absorbing translation error; depth anchor pins it
    )
    cosine_decay: bool = False
    lambda_depth: float = 3.0  # L1 depth anchor weight; dominant translation knob
    diffuse_mode: str = (
        "diffuse_shaded"  # re-shades at candidate pose; "diffuse" = frozen bake
    )
    diffuse_alpha_min: float = (
        0.2  # skip diffuse stage on frames more specular than this
    )
    scales: tuple[float, ...] = (1.0,)
    accept_on_loss: bool = True  # keep result only if it lowered the loss it optimised
    max_correction_deg: float = (
        1e9  # per-frame rollback cap; tighten to prevent basin drift
    )
    max_correction_trans: float = 1e9
    patience_loss: int = 10
    min_rel_improve: float = 5e-3
    update_every: int = 1

    # feed-forward drift guard
    drift_reset_deg: float = 1e9
    drift_reset_trans: float = 1e9
    drift_reset_alpha_min: float = (
        0.8  # guard only on diffuse frames where LoFTR is reliable
    )

    # relight stage
    relight: bool = False
    relight_alpha_max: float = 0.85
    lr_rot_relight: float = 3e-3
    lr_trans_relight: float = 2e-4
    relight_min_correction_deg: float = (
        3.0  # noise-floor gate; discard small corrections
    )
    relight_feed_forward: bool = False  # ablation lever; whole-pose FF is in main.py
    relight_conf_floor: float = 0.15  # weight floor for unobserved-env pixels


def _optimise_stage(
    slf_prev: SurfaceLightField,
    slf_curr: SurfaceLightField,
    env_prev: torch.Tensor | None,
    env_curr: torch.Tensor | None,
    abs_pose_prev: np.ndarray,
    pose_init: np.ndarray,
    alpha: float,
    mode: str,
    lr_rot: float,
    lr_trans: float,
    cfg: RefineConfig,
    *,
    stage: str,
    env_conf_prev: torch.Tensor | None = None,
    min_correction_deg: float = 0.0,
    viewer: PhotometricRefineViewer | None,
    gt_pose_curr: np.ndarray | None,
    diag: list[dict] | None,
    loss_history: list[float],
) -> np.ndarray:
    """Joint 6-DoF Adam optimisation under appearance ``mode``, coarse-to-fine.
    pose = [dR @ R0 | t0 + dt] with R0, t0 the stage init. Returns np.float64 pose.
    """
    device = "cuda"
    R0 = torch.from_numpy(pose_init[:3, :3]).float().to(device).detach()
    t0 = torch.from_numpy(pose_init[:3, 3]).float().to(device).detach()

    rot_6d = (
        matrix_to_rotation_6d(torch.eye(3, device=device))
        .clone()
        .detach()
        .requires_grad_(True)
    )
    dt = torch.zeros(3, device=device, requires_grad=True)
    optimizer = torch.optim.Adam(
        [{"params": [rot_6d], "lr": lr_rot}, {"params": [dt], "lr": lr_trans}]
    )
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    def _compose() -> torch.Tensor:
        dR = rotation_6d_to_matrix(rot_6d)
        pose = torch.eye(4, device=device)
        pose[:3, :3] = dR @ R0
        pose[:3, 3] = t0 + dt
        return pose

    best_pose = pose_init.copy()
    base_loss: float | None = None  # photometric loss at the stage init (scale 1.0)

    for li, scale in enumerate(cfg.scales):
        is_final_level = li == len(cfg.scales) - 1
        ctx, tgt_img = build_photometric_context(
            slf_prev=slf_prev,
            slf_curr=slf_curr,
            env_map_prev=env_prev,
            env_map_curr=env_curr,
            abs_pose_prev=abs_pose_prev,
            pose_coarse=pose_init,
            alpha=alpha,
            lambda_depth=cfg.lambda_depth,
            scale=scale,
            device=device,
            mode=mode,
            env_conf_prev=env_conf_prev,
            conf_floor=cfg.relight_conf_floor,
        )
        if is_final_level and base_loss is None:
            with torch.no_grad():
                base_loss = photometric_forward(
                    ctx, torch.from_numpy(pose_init).float().to(device)
                )[0].item()

        no_improve = 0
        best_level_loss = float("inf")
        for step in range(cfg.num_iters):
            optimizer.zero_grad()
            if cfg.cosine_decay and is_final_level and cfg.num_iters > 1:
                decay = 0.5 * (1.0 + np.cos(np.pi * step / (cfg.num_iters - 1)))
                for g, base in zip(optimizer.param_groups, base_lrs):
                    g["lr"] = base * float(decay)

            pose_curr = _compose()
            loss, _components, aux = photometric_forward(ctx, pose_curr)
            loss_val = loss.item()
            loss_history.append(loss_val)

            if not np.isfinite(loss_val):
                break  # NaN/Inf loss — keep best_pose from previous step

            pose_np = pose_curr.detach().cpu().numpy().astype(np.float64)

            if diag is not None and gt_pose_curr is not None:
                rot_e = _rot_err_deg_np(pose_np[:3, :3], gt_pose_curr[:3, :3])
                trans_e = float(np.linalg.norm(pose_np[:3, 3] - gt_pose_curr[:3, 3]))
                diag.append(
                    {
                        "stage": stage,
                        "level": li,
                        "scale": scale,
                        "step": step,
                        "lr": optimizer.param_groups[0]["lr"],
                        "loss": loss_val,
                        "rot_deg": rot_e,
                        "trans_mm": trans_e * 1000.0,
                    }
                )

            if loss_val < best_level_loss * (1.0 - cfg.min_rel_improve):
                best_level_loss = loss_val
                best_pose = pose_np
                no_improve = 0
            else:
                no_improve += 1

            if viewer is not None and step % cfg.update_every == 0:
                with torch.no_grad():
                    viewer.update(
                        iteration=len(loss_history),
                        loss=loss_val,
                        src_img=aux["src_img"].detach(),
                        tgt_img=tgt_img,
                        env_map=env_prev,
                        gt_pose_curr=gt_pose_curr,
                        pose_curr_np=best_pose,
                        slf_prev=slf_prev,
                        rel_pose=(pose_curr @ ctx.inv_pose_prev).detach(),
                    )

            loss.backward()
            optimizer.step()

            if (not is_final_level) and no_improve >= cfg.patience_loss:
                break

    # Noise-floor gate: discard a correction smaller than the stage's appearance
    # model can resolve (relight only) — it is within the model's own bias.
    if min_correction_deg > 0.0:
        if _rot_err_deg_np(best_pose[:3, :3], pose_init[:3, :3]) < min_correction_deg:
            return pose_init.copy()

    # Safety gate: only keep the result if it actually lowered the loss it optimised.
    if cfg.accept_on_loss and base_loss is not None:
        with torch.no_grad():
            ctx, _ = build_photometric_context(
                slf_prev=slf_prev,
                slf_curr=slf_curr,
                env_map_prev=env_prev,
                env_map_curr=env_curr,
                abs_pose_prev=abs_pose_prev,
                pose_coarse=pose_init,
                alpha=alpha,
                lambda_depth=cfg.lambda_depth,
                scale=1.0,
                device=device,
                mode=mode,
                env_conf_prev=env_conf_prev,
                conf_floor=cfg.relight_conf_floor,
            )
            final_loss = photometric_forward(
                ctx, torch.from_numpy(best_pose).float().to(device)
            )[0].item()
        if not np.isfinite(final_loss) or final_loss > base_loss:
            return pose_init.copy()
    return best_pose


def refine_pose_photometric(
    slf_prev: SurfaceLightField,
    slf_curr: SurfaceLightField,
    env_map_prev: torch.Tensor | None,
    env_map_curr: torch.Tensor | None,
    abs_pose_prev: np.ndarray,
    pose_coarse: np.ndarray,
    alpha: float,
    cfg: RefineConfig | None = None,
    viewer: PhotometricRefineViewer | None = None,
    gt_pose_curr: np.ndarray | None = None,
    diag: list[dict] | None = None,
    env_conf_prev: torch.Tensor | None = None,
    env_conf_curr: torch.Tensor | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Two-stage photometric pose refinement on top of the coarse (LoFTR/ICP) pose.

    Stage 1 (diffuse): matches separated diffuse appearance; skipped on reflective frames.
    Stage 2 (relight): matches alpha·diffuse + (1-alpha)·env[reflect]; reflective only.
    Each stage is gated to only keep results that lower the loss.
    Returns (refined_abs_pose [4,4] float64, loss_history).
    """
    cfg = cfg or RefineConfig()
    loss_history: list[float] = []

    pose = pose_coarse.astype(np.float64).copy()

    # Stage 1 — diffuse photometric (fine pose). Skipped on reflective frames,
    # whose diffuse channel carries no usable pose signal.
    if (cfg.lr_rot > 0.0 or cfg.lr_trans > 0.0) and alpha > cfg.diffuse_alpha_min:
        pose = _optimise_stage(
            slf_prev,
            slf_curr,
            None,
            None,
            abs_pose_prev,
            pose,
            alpha=1.0,
            mode=cfg.diffuse_mode,
            lr_rot=cfg.lr_rot,
            lr_trans=cfg.lr_trans,
            cfg=cfg,
            stage="diffuse",
            viewer=viewer,
            gt_pose_curr=gt_pose_curr,
            diag=diag,
            loss_history=loss_history,
        )

    # Stage 2 — relighting refinement (final pose), reflective frames only.
    if (
        cfg.relight
        and env_map_prev is not None
        and alpha < cfg.relight_alpha_max
        and cfg.lr_rot_relight > 0.0
    ):
        pose = _optimise_stage(
            slf_prev,
            slf_curr,
            env_map_prev,
            env_map_curr,
            abs_pose_prev,
            pose,
            alpha=alpha,
            mode="relit",
            lr_rot=cfg.lr_rot_relight,
            lr_trans=cfg.lr_trans_relight,
            cfg=cfg,
            stage="relit",
            env_conf_prev=env_conf_prev,
            min_correction_deg=cfg.relight_min_correction_deg,
            viewer=viewer,
            gt_pose_curr=gt_pose_curr,
            diag=diag,
            loss_history=loss_history,
        )

    # Drift rollback: a refinement that has moved too far from the coarse pose is
    # almost certainly a wrong-basin drift — discard it and keep the coarse pose.
    corr_deg = _rot_err_deg_np(pose[:3, :3], pose_coarse[:3, :3])
    corr_trans = float(np.linalg.norm(pose[:3, 3] - pose_coarse[:3, 3]))
    if corr_deg > cfg.max_correction_deg or corr_trans > cfg.max_correction_trans:
        return pose_coarse.astype(np.float64).copy(), loss_history

    return pose, loss_history
