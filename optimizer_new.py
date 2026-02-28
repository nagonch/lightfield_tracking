from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Literal

import numpy as np
import torch
from PIL import Image

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional runtime dependency
    plt = None


# ---------------------------
# Types
# ---------------------------

LossFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]
# loss_fn(pred_image, pred_depth, tgt_image, tgt_depth, mask) -> scalar loss


@dataclass
class OptimizeResult:
    pose_rel_optimized: torch.Tensor
    final_loss: float
    info: Dict[str, object]


@dataclass
class _ImageLogger:
    out_dir: Optional[str] = None
    show: bool = False
    every: int = 1
    counter: int = 0
    _fig: object = None
    _axes: object = None
    _images: object = None
    _show_disabled: bool = False

    def __post_init__(self):
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
            print(f"[optimize] logging images to: {self.out_dir}")
        self.every = max(1, int(self.every))

    @property
    def enabled(self) -> bool:
        return self.out_dir is not None or self.show

    def _save_rgb(self, image_hw3: torch.Tensor, path: str):
        image = image_hw3.detach().float()
        if image.ndim != 3:
            raise ValueError(f"Expected RGB image [H,W,3], got {tuple(image.shape)}")
        if image.shape[0] == 3 and image.shape[-1] != 3:
            image = image.permute(1, 2, 0).contiguous()
        if image.shape[-1] != 3:
            raise ValueError(f"Expected RGB last dim = 3, got {tuple(image.shape)}")
        image = torch.clamp(image, 0.0, 1.0)
        image_u8 = (image.cpu().numpy() * 255.0).astype(np.uint8)
        Image.fromarray(image_u8, mode="RGB").save(path)

    def _save_gray(self, image_hw: torch.Tensor, path: str):
        image = image_hw.detach().float()
        if image.ndim == 3:
            image = image.mean(dim=-1)
        image = image - image.min()
        image = image / (image.max() + 1e-8)
        image = torch.clamp(image, 0.0, 1.0)
        image_u8 = (image.cpu().numpy() * 255.0).astype(np.uint8)
        Image.fromarray(image_u8, mode="L").save(path)

    def _to_numpy_rgb(self, image_hw3: torch.Tensor) -> np.ndarray:
        image = image_hw3.detach().float()
        if image.shape[0] == 3 and image.shape[-1] != 3:
            image = image.permute(1, 2, 0).contiguous()
        image = torch.clamp(image, 0.0, 1.0)
        return image.cpu().numpy()

    def _to_numpy_gray(self, image_hw: torch.Tensor) -> np.ndarray:
        image = image_hw.detach().float()
        if image.ndim == 3:
            image = image.mean(dim=-1)
        image = image - image.min()
        image = image / (image.max() + 1e-8)
        image = torch.clamp(image, 0.0, 1.0)
        return image.cpu().numpy()

    def _show_live(self, rendered_rgb: np.ndarray, loss_map: np.ndarray, title: str):
        if not self.show or self._show_disabled:
            return
        if plt is None:
            self._show_disabled = True
            print("[optimize] matplotlib is unavailable; disabled live image window.")
            return
        try:
            if self._fig is None:
                plt.ion()
                self._fig, self._axes = plt.subplots(1, 2, figsize=(10, 4))
                self._images = [
                    self._axes[0].imshow(rendered_rgb),
                    self._axes[1].imshow(loss_map, cmap="inferno"),
                ]
                self._axes[0].set_title("Rendered")
                self._axes[1].set_title("Loss map")
                self._axes[0].axis("off")
                self._axes[1].axis("off")
            else:
                self._images[0].set_data(rendered_rgb)
                self._images[1].set_data(loss_map)
            self._fig.suptitle(title)
            self._fig.canvas.draw_idle()
            plt.pause(0.001)
        except Exception as exc:
            self._show_disabled = True
            print(f"[optimize] disabled live image window: {exc}")

    def log(
        self,
        *,
        stage: str,
        pred_image: torch.Tensor,
        pred_depth: torch.Tensor,
        tgt_image: torch.Tensor,
        tgt_depth: torch.Tensor,
        mask: torch.Tensor,
        loss_scalar: float,
    ):
        if not self.enabled:
            return

        step_id = self.counter
        self.counter += 1
        if step_id % self.every != 0:
            return

        h, w = pred_image.shape[0], pred_image.shape[1]
        mask_hw = _normalize_mask_hw(mask, h, w)

        photo_map = (pred_image - tgt_image).abs().mean(dim=-1)
        pred_depth_hw = pred_depth[..., 0] if pred_depth.ndim == 3 else pred_depth
        tgt_depth_hw = tgt_depth[..., 0] if tgt_depth.ndim == 3 else tgt_depth
        depth_map = (pred_depth_hw - tgt_depth_hw).abs()
        loss_map = (photo_map + 0.2 * depth_map) * mask_hw.float()

        rendered_np = self._to_numpy_rgb(pred_image)
        loss_map_np = self._to_numpy_gray(loss_map)
        self._show_live(rendered_np, loss_map_np, f"{stage} | loss={loss_scalar:.6f}")

        if self.out_dir is None:
            return

        prefix = f"{step_id:06d}_{stage}"
        self._save_rgb(pred_image, os.path.join(self.out_dir, f"{prefix}_rendered.png"))
        self._save_rgb(tgt_image, os.path.join(self.out_dir, f"{prefix}_tgt_image.png"))
        self._save_gray(loss_map, os.path.join(self.out_dir, f"{prefix}_loss.png"))


def _save_trajectory_plot(
    *,
    losses: list[float],
    rot_norms: list[float],
    trans_norms: list[float],
    title: str,
    out_path: Optional[str],
    show: bool,
) -> Optional[str]:
    if len(losses) == 0:
        return None
    if plt is None:
        return None

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    steps = np.arange(len(losses))

    axes[0].plot(steps, losses, color="#1f77b4")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, rot_norms, color="#d62728")
    axes[1].set_ylabel("||rot|| [rad]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, trans_norms, color="#2ca02c")
    axes[2].set_ylabel("||trans||")
    axes[2].set_xlabel("iteration")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()

    saved_path = None
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=140)
        saved_path = out_path

    if show:
        fig.canvas.draw_idle()
        plt.show(block=False)
        plt.pause(0.001)
    else:
        plt.close(fig)

    return saved_path


# ---------------------------
# Pose utilities (SE(3) from axis-angle + translation)
# ---------------------------


def _skew(vec3: torch.Tensor) -> torch.Tensor:
    """vec3: [3] -> [3,3] skew-symmetric matrix"""
    x, y, z = vec3[0], vec3[1], vec3[2]
    return torch.stack(
        [
            torch.stack([torch.zeros_like(x), -z, y]),
            torch.stack([z, torch.zeros_like(x), -x]),
            torch.stack([-y, x, torch.zeros_like(x)]),
        ],
        dim=0,
    )


def _so3_exp(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Exponential map for SO(3).
    axis_angle: [3]
    returns R: [3,3]
    """
    theta = torch.linalg.norm(axis_angle) + 1e-12
    axis = axis_angle / theta
    K = _skew(axis)
    I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)

    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    R = I + sin_t * K + (1.0 - cos_t) * (K @ K)
    return R


def se3_from_axis_angle_translation(
    axis_angle: torch.Tensor, translation: torch.Tensor
) -> torch.Tensor:
    """
    axis_angle: [3], translation: [3] -> T: [4,4]
    """
    R = _so3_exp(axis_angle)
    T = torch.eye(4, device=axis_angle.device, dtype=axis_angle.dtype)
    T[:3, :3] = R
    T[:3, 3] = translation
    return T


# ---------------------------
# Rendering hook
# ---------------------------


def _render(
    surface_lf_prev, pose_rel: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Required: pred_image, pred_depth = surface_lf_prev.render(pose_rel)
    """
    if not hasattr(surface_lf_prev, "render"):
        raise AttributeError(
            "surface_lf_prev must implement .render(pose_rel) -> (image, depth)"
        )
    return surface_lf_prev.render(pose_rel)


# ---------------------------
# Main API
# ---------------------------


def optimize(
    surface_lf_prev,
    pose_rel_init: torch.Tensor,
    image: torch.Tensor,
    depth: torch.Tensor,
    mask: torch.Tensor,
    *,
    loss_fn: LossFn,
    method: Literal["adam", "nuclear"] = "adam",
    # adam params
    steps: int = 60,
    lr_rot: float = 3e-3,
    lr_trans: float = 3e-3,
    grad_clip: float = 10.0,
    # nuclear multistart params (rotation-only sampling)
    num_samples: int = 96,
    max_angle_deg: float = 25.0,
    topk: int = 8,
    refine_topk: int = 3,
    refine_steps: int = 25,
    refine_lr_rot: float = 2e-2,
    refine_lr_trans: float = 0.0,
    log_dir: Optional[str] = None,
    log_every: int = 1,
    show_logs: bool = False,
) -> OptimizeResult:
    """
    Frame-to-frame relative pose optimization.

    Args:
      surface_lf_prev: previous-frame surface (must support .render(pose_rel))
      pose_rel_init: initial relative pose guess (4x4)
      image/depth/mask: target observations
      loss_fn: callable(pred_image, pred_depth, tgt_image, tgt_depth, mask)->scalar
      method:
        - "adam": gradient refinement from pose_rel_init
        - "nuclear": rotation multistart + optional local adam refine

    Returns:
      OptimizeResult with pose_rel_optimized (4x4) and final_loss.
    """
    logger = _ImageLogger(out_dir=log_dir, every=log_every, show=show_logs)

    if method == "adam":
        return _optimize_adam(
            surface_lf_prev,
            pose_rel_init,
            image,
            depth,
            mask,
            loss_fn=loss_fn,
            steps=steps,
            lr_rot=lr_rot,
            lr_trans=lr_trans,
            grad_clip=grad_clip,
            image_logger=logger,
        )

    if method == "nuclear":
        return _optimize_nuclear(
            surface_lf_prev,
            pose_rel_init,
            image,
            depth,
            mask,
            loss_fn=loss_fn,
            num_samples=num_samples,
            max_angle_deg=max_angle_deg,
            topk=topk,
            refine_topk=refine_topk,
            refine_steps=refine_steps,
            refine_lr_rot=refine_lr_rot,
            refine_lr_trans=refine_lr_trans,
            grad_clip=grad_clip,
            image_logger=logger,
        )

    raise ValueError(f"Unknown method: {method}")


# ---------------------------
# Adam (gradient) backend
# ---------------------------


def _optimize_adam(
    surface_lf_prev,
    pose_rel_init: torch.Tensor,
    image: torch.Tensor,
    depth: torch.Tensor,
    mask: torch.Tensor,
    *,
    loss_fn: LossFn,
    steps: int,
    lr_rot: float,
    lr_trans: float,
    grad_clip: float,
    image_logger: Optional[_ImageLogger] = None,
) -> OptimizeResult:
    device = pose_rel_init.device
    dtype = pose_rel_init.dtype

    pose_rel_anchor = pose_rel_init.detach()

    rot_param = torch.zeros(3, device=device, dtype=dtype, requires_grad=True)
    trans_param = torch.zeros(3, device=device, dtype=dtype, requires_grad=True)

    optimizer = torch.optim.Adam(
        [
            {"params": [rot_param], "lr": float(lr_rot)},
            {"params": [trans_param], "lr": float(lr_trans)},
        ]
    )

    best_pose = pose_rel_anchor.clone()
    best_loss = float("inf")
    losses: list[float] = []
    rot_norms: list[float] = []
    trans_norms: list[float] = []

    for step_idx in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)

        pose_delta = se3_from_axis_angle_translation(rot_param, trans_param)
        pose_rel = pose_rel_anchor @ pose_delta  # right-update, relative-only

        pred_image, pred_depth = _render(surface_lf_prev, pose_rel)
        loss_val = loss_fn(pred_image, pred_depth, image, depth, mask)
        losses.append(float(loss_val.detach().cpu()))
        rot_norms.append(float(torch.linalg.norm(rot_param.detach()).cpu()))
        trans_norms.append(float(torch.linalg.norm(trans_param.detach()).cpu()))

        loss_val.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([rot_param, trans_param], float(grad_clip))

        optimizer.step()

        loss_scalar = float(loss_val.detach().cpu())
        if image_logger is not None:
            image_logger.log(
                stage=f"adam_step{step_idx:04d}",
                pred_image=pred_image.detach(),
                pred_depth=pred_depth.detach(),
                tgt_image=image,
                tgt_depth=depth,
                mask=mask,
                loss_scalar=loss_scalar,
            )
        if loss_scalar < best_loss:
            best_loss = loss_scalar
            best_pose = pose_rel.detach().clone()

    curve_path = None
    if image_logger is not None and image_logger.out_dir is not None:
        curve_path = os.path.join(image_logger.out_dir, "adam_trajectory.png")
    curve_path = _save_trajectory_plot(
        losses=losses,
        rot_norms=rot_norms,
        trans_norms=trans_norms,
        title="Adam optimization trajectory",
        out_path=curve_path,
        show=bool(image_logger.show) if image_logger is not None else False,
    )

    return OptimizeResult(
        pose_rel_optimized=best_pose,
        final_loss=best_loss,
        info={
            "method": "adam",
            "steps": int(steps),
            "lr_rot": lr_rot,
            "lr_trans": lr_trans,
            "log_dir": image_logger.out_dir if image_logger is not None else None,
            "trajectory": {
                "loss": losses,
                "rot_norm": rot_norms,
                "trans_norm": trans_norms,
                "plot_path": curve_path,
            },
        },
    )


# ---------------------------
# "Nuclear" backend (multistart rotations)
# ---------------------------


@torch.no_grad()
def _sample_axis_angle_uniform(
    num_samples: int, max_angle_rad: float, device, dtype
) -> torch.Tensor:
    axis = torch.randn(num_samples, 3, device=device, dtype=dtype)
    axis = axis / (torch.linalg.norm(axis, dim=-1, keepdim=True) + 1e-9)
    angles = (
        2.0 * torch.rand(num_samples, device=device, dtype=dtype) - 1.0
    ) * max_angle_rad
    return axis * angles[:, None]  # [N,3]


def _optimize_nuclear(
    surface_lf_prev,
    pose_rel_init: torch.Tensor,
    image: torch.Tensor,
    depth: torch.Tensor,
    mask: torch.Tensor,
    *,
    loss_fn: LossFn,
    num_samples: int,
    max_angle_deg: float,
    topk: int,
    refine_topk: int,
    refine_steps: int,
    refine_lr_rot: float,
    refine_lr_trans: float,
    grad_clip: float,
    image_logger: Optional[_ImageLogger] = None,
) -> OptimizeResult:
    device = pose_rel_init.device
    dtype = pose_rel_init.dtype

    pose_center = pose_rel_init.detach()

    def eval_pose(pose_rel: torch.Tensor, stage: str) -> float:
        pred_image, pred_depth = _render(surface_lf_prev, pose_rel)
        loss_val = loss_fn(pred_image, pred_depth, image, depth, mask)
        loss_scalar = float(loss_val.detach().cpu())
        if image_logger is not None:
            image_logger.log(
                stage=stage,
                pred_image=pred_image.detach(),
                pred_depth=pred_depth.detach(),
                tgt_image=image,
                tgt_depth=depth,
                mask=mask,
                loss_scalar=loss_scalar,
            )
        return loss_scalar

    # 1) Coarse sampling around init (rotation-only)
    candidates: list[tuple[float, torch.Tensor]] = []
    candidates.append((eval_pose(pose_center, "nuclear_center"), pose_center))

    max_angle_rad = math.radians(float(max_angle_deg))
    axis_angles = _sample_axis_angle_uniform(
        int(num_samples), max_angle_rad, device, dtype
    )

    for k in range(int(num_samples)):
        pose_delta = se3_from_axis_angle_translation(
            axis_angles[k], torch.zeros(3, device=device, dtype=dtype)
        )
        pose_candidate = pose_center @ pose_delta
        candidates.append(
            (eval_pose(pose_candidate, f"nuclear_sample{k:04d}"), pose_candidate)
        )

    candidates.sort(key=lambda x: x[0])
    seeds = candidates[: max(1, int(topk))]

    best_loss, best_pose = seeds[0][0], seeds[0][1].detach().clone()
    refine_runs: list[Dict[str, object]] = []

    # 2) Optional local refine from best seeds (Adam)
    for seed_idx in range(min(int(refine_topk), len(seeds))):
        _, seed_pose = seeds[seed_idx]
        local = _optimize_local_around_seed(
            surface_lf_prev=surface_lf_prev,
            seed_pose=seed_pose.detach(),
            image=image,
            depth=depth,
            mask=mask,
            loss_fn=loss_fn,
            steps=refine_steps,
            lr_rot=refine_lr_rot,
            lr_trans=refine_lr_trans,
            grad_clip=grad_clip,
            image_logger=image_logger,
            stage_prefix=f"nuclear_refine_seed{seed_idx:02d}",
        )
        refine_runs.append(
            {
                "seed_idx": int(seed_idx),
                "final_loss": float(local.final_loss),
                "trajectory": local.info.get("trajectory"),
            }
        )
        if local.final_loss < best_loss:
            best_loss = local.final_loss
            best_pose = local.pose_rel_optimized.detach().clone()

    return OptimizeResult(
        pose_rel_optimized=best_pose,
        final_loss=best_loss,
        info={
            "method": "nuclear",
            "num_samples": int(num_samples),
            "max_angle_deg": float(max_angle_deg),
            "topk": int(topk),
            "refine_topk": int(refine_topk),
            "refine_steps": int(refine_steps),
            "log_dir": image_logger.out_dir if image_logger is not None else None,
            "refine_runs": refine_runs,
        },
    )


def _optimize_local_around_seed(
    *,
    surface_lf_prev,
    seed_pose: torch.Tensor,
    image: torch.Tensor,
    depth: torch.Tensor,
    mask: torch.Tensor,
    loss_fn: LossFn,
    steps: int,
    lr_rot: float,
    lr_trans: float,
    grad_clip: float,
    image_logger: Optional[_ImageLogger] = None,
    stage_prefix: str = "local_adam",
) -> OptimizeResult:
    device = seed_pose.device
    dtype = seed_pose.dtype

    rot_param = torch.zeros(3, device=device, dtype=dtype, requires_grad=True)
    trans_param = torch.zeros(3, device=device, dtype=dtype, requires_grad=True)

    param_groups = [{"params": [rot_param], "lr": float(lr_rot)}]
    if lr_trans and lr_trans > 0:
        param_groups.append({"params": [trans_param], "lr": float(lr_trans)})

    optimizer = torch.optim.Adam(param_groups)

    best_pose = seed_pose.detach().clone()
    best_loss = float("inf")
    losses: list[float] = []
    rot_norms: list[float] = []
    trans_norms: list[float] = []

    for step_idx in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)

        pose_delta = se3_from_axis_angle_translation(rot_param, trans_param)
        pose_rel = seed_pose @ pose_delta

        pred_image, pred_depth = _render(surface_lf_prev, pose_rel)
        loss_val = loss_fn(pred_image, pred_depth, image, depth, mask)
        losses.append(float(loss_val.detach().cpu()))
        rot_norms.append(float(torch.linalg.norm(rot_param.detach()).cpu()))
        trans_norms.append(float(torch.linalg.norm(trans_param.detach()).cpu()))

        loss_val.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([rot_param, trans_param], float(grad_clip))

        optimizer.step()

        loss_scalar = float(loss_val.detach().cpu())
        if image_logger is not None:
            image_logger.log(
                stage=f"{stage_prefix}_step{step_idx:04d}",
                pred_image=pred_image.detach(),
                pred_depth=pred_depth.detach(),
                tgt_image=image,
                tgt_depth=depth,
                mask=mask,
                loss_scalar=loss_scalar,
            )
        if loss_scalar < best_loss:
            best_loss = loss_scalar
            best_pose = pose_rel.detach().clone()

    curve_path = None
    if image_logger is not None and image_logger.out_dir is not None:
        curve_path = os.path.join(
            image_logger.out_dir, f"{stage_prefix}_trajectory.png"
        )
    curve_path = _save_trajectory_plot(
        losses=losses,
        rot_norms=rot_norms,
        trans_norms=trans_norms,
        title=f"{stage_prefix} trajectory",
        out_path=curve_path,
        show=bool(image_logger.show) if image_logger is not None else False,
    )

    return OptimizeResult(
        pose_rel_optimized=best_pose,
        final_loss=best_loss,
        info={
            "method": "local_adam",
            "steps": int(steps),
            "lr_rot": lr_rot,
            "lr_trans": lr_trans,
            "log_dir": image_logger.out_dir if image_logger is not None else None,
            "trajectory": {
                "loss": losses,
                "rot_norm": rot_norms,
                "trans_norm": trans_norms,
                "plot_path": curve_path,
            },
        },
    )


def _normalize_mask_hw(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Returns boolean mask with shape [H, W].
    Accepts mask shapes: [H,W], [H,W,1], [1,H,W], [H,W,C] (any C>1 treated as any-channel).
    """
    if mask.ndim == 2:
        mask_hw = mask
    elif mask.ndim == 3:
        if mask.shape[-1] == 1:  # [H,W,1]
            mask_hw = mask[..., 0]
        elif mask.shape[0] == 1:  # [1,H,W]
            mask_hw = mask[0]
        else:  # [H,W,C]
            # treat as "valid if any channel is valid"
            mask_hw = (
                mask.any(dim=-1)
                if mask.dtype == torch.bool
                else (mask > 0.5).any(dim=-1)
            )
    else:
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")

    if mask_hw.shape != (height, width):
        raise ValueError(
            f"Mask shape {tuple(mask_hw.shape)} doesn't match H,W={(height,width)}"
        )

    return mask_hw if mask_hw.dtype == torch.bool else (mask_hw > 0.5)


def masked_l1(
    pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    pred/tgt: [H,W] or [H,W,C]
    mask: flexible, normalized internally to [H,W]
    """
    if pred.shape != tgt.shape:
        raise ValueError(
            f"pred shape {tuple(pred.shape)} != tgt shape {tuple(tgt.shape)}"
        )

    h, w = pred.shape[0], pred.shape[1]
    mask_hw = _normalize_mask_hw(mask, h, w)

    diff = (pred - tgt).abs()
    if diff.ndim == 3:
        # broadcast [H,W] -> [H,W,1] for multiplication, but for boolean indexing expand to [H,W,C]
        mask_b = mask_hw.unsqueeze(-1).expand_as(diff)
        diff_masked = diff[mask_b]
    else:
        diff_masked = diff[mask_hw]
    return diff_masked.mean() if diff_masked.numel() > 0 else diff.sum() * 0.0


def loss_fn(pred_image, pred_depth, tgt_image, tgt_depth, mask):
    # photometric + depth, tune weights as needed
    photo = masked_l1(pred_image, tgt_image, mask)
    dep = masked_l1(pred_depth, tgt_depth, mask)
    return photo + 0.2 * dep


def summarize_tensor(name: str, tensor: torch.Tensor):
    print(
        f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device} "
        f"min={tensor.min().item():.4g} max={tensor.max().item():.4g}"
    )


if __name__ == "__main__":
    surf_debug = torch.load("surf_debug.pt", weights_only=False)

    pose_rel_init = torch.tensor(surf_debug["pose_rel"]).cuda()
    image_tgt = surf_debug["image"]
    depth_tgt = surf_debug["depth"]
    mask_tgt = surf_debug["mask"]
    surface_lf_prev = surf_debug["surface_lf"]

    print("Loaded surface_lf_prev:", surface_lf_prev)

    # Put everything on one device
    device = image_tgt.device
    pose_rel_init = pose_rel_init.to(device)
    depth_tgt = depth_tgt.to(device)
    mask_tgt = mask_tgt.to(device)

    summarize_tensor("pose_rel_init", pose_rel_init)
    summarize_tensor("image_tgt", image_tgt)
    summarize_tensor("depth_tgt", depth_tgt)
    summarize_tensor("mask_tgt", mask_tgt)

    # --- Baseline loss (init pose) ---
    with torch.no_grad():
        pred_image_init, pred_depth_init = surface_lf_prev.render(pose_rel_init)
        baseline_loss = float(
            loss_fn(
                pred_image_init, pred_depth_init, image_tgt, depth_tgt, mask_tgt
            ).cpu()
        )
    print(f"\nBaseline loss @ init pose_rel: {baseline_loss:.6f}")

    # --- ADAM refine ---
    result_adam = optimize(
        surface_lf_prev,
        pose_rel_init,
        image_tgt,
        depth_tgt,
        mask_tgt,
        loss_fn=loss_fn,
        method="adam",
        steps=100,
        lr_rot=1e-3,
        lr_trans=1e-3,
        grad_clip=10.0,
        log_dir="optimizer_logs/adam",
        log_every=1,
        show_logs=True,
    )
    print("\n[ADAM] final_loss:", result_adam.final_loss)
    print("[ADAM] info:", result_adam.info)

    with torch.no_grad():
        pred_image_adam, pred_depth_adam = surface_lf_prev.render(
            result_adam.pose_rel_optimized
        )
        adam_check_loss = float(
            loss_fn(
                pred_image_adam, pred_depth_adam, image_tgt, depth_tgt, mask_tgt
            ).cpu()
        )
    print(f"[ADAM] recomputed loss: {adam_check_loss:.6f}")

    # --- Nuclear multistart refine ---
    result_nuclear = optimize(
        surface_lf_prev,
        pose_rel_init,
        image_tgt,
        depth_tgt,
        mask_tgt,
        loss_fn=loss_fn,
        method="nuclear",
        num_samples=128,
        max_angle_deg=25.0,
        topk=10,
        refine_topk=3,
        refine_steps=30,
        refine_lr_rot=2e-2,
        refine_lr_trans=0.0,  # keep 0 if you want rotation-only refine
        grad_clip=10.0,
        log_dir="optimizer_logs/nuclear",
        log_every=1,
        show_logs=True,
    )
    print("\n[NUCLEAR] final_loss:", result_nuclear.final_loss)
    print("[NUCLEAR] info:", result_nuclear.info)

    with torch.no_grad():
        pred_image_nuc, pred_depth_nuc = surface_lf_prev.render(
            result_nuclear.pose_rel_optimized
        )
        nuclear_check_loss = float(
            loss_fn(
                pred_image_nuc, pred_depth_nuc, image_tgt, depth_tgt, mask_tgt
            ).cpu()
        )
    print(f"[NUCLEAR] recomputed loss: {nuclear_check_loss:.6f}")

    # --- Quick sanity verdict ---
    print("\nSanity:")
    print("  baseline:", baseline_loss)
    print("  adam    :", adam_check_loss)
    print("  nuclear :", nuclear_check_loss)
    if min(adam_check_loss, nuclear_check_loss) < baseline_loss:
        print("  OK: at least one optimizer improved the loss.")
    else:
        print(
            "  WARNING: no improvement — likely loss is wrong, render isn't differentiable, or pose convention mismatch."
        )
