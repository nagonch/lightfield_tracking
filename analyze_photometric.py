"""Photometric loss-landscape analyzer.

Feeds the *ground-truth* poses into the photometric loss and wiggles the
candidate current-frame pose around GT, one DoF at a time, to see whether the
loss actually has a clean minimum at the correct answer (and how it correlates
with pose error).

For a chosen sequence + frame pair (prev = i-1, curr = i):
  • abs_pose_prev  = GT object pose of frame i-1   (anchor, never perturbed)
  • pose_curr      = GT object pose of frame i      (the correct answer)
  • the visible source-point set is frozen at the GT current pose
  • pose regularisation is disabled (lambda_rot = lambda_trans = 0) so we see
    the *raw* photometric / depth / mask landscape, not a reg bowl.

We then perturb pose_curr independently along the 3 camera rotation axes
(rotating the object about its own centre) and the 3 camera translation axes,
and plot each loss term vs. the perturbation magnitude.  At perturbation = 0 the
candidate equals GT, so a well-behaved loss should bottom out there.

Outputs (PNG) go to ``analysis_out/``:
  • landscape_<seq>_f<idx>.png   — 2×3 grid of loss curves (rot row / trans row)
  • align_<seq>_f<idx>.png       — target vs source@GT vs |diff| sanity render

Run:  python analyze_photometric.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset import LFDataset
from src.photometric import (
    _to_display_u8,
    build_photometric_context,
    photometric_forward,
)
from src.surface_light_field import SurfaceLightField

# ── configuration ────────────────────────────────────────────────────────────
DATASET_ROOT = "/home/ngoncharov/SpecTrack_dataset"
SPLIT = "cube_0.0"
SEQUENCE = "cracker_box_yalehand0"  # one of the worse cube_0.0 sequences
FRAME_INDICES = [5, 15, 30]  # curr frame indices to analyze (prev = idx-1)
DEPTH_SOURCE = "gt"

ROT_MAX_DEG = 15.0
TRANS_MAX_M = 0.03
N_SAMPLES = 61

LAMBDA_DEPTH = 0.1

OUT_DIR = "analysis_out"

# cube_0.0 → reflectivity 0.0 → alpha = 1.0 (pure diffuse, env map unused)
ALPHA = 1.0


# ── SE(3) perturbation helpers ───────────────────────────────────────────────


def _rodrigues(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    a = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array(
        [[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]], dtype=np.float64
    )
    return np.eye(3) + np.sin(angle_rad) * K + (1.0 - np.cos(angle_rad)) * (K @ K)


def _perturb_rot(
    pose_gt: np.ndarray, axis: np.ndarray, angle_rad: float, center_cam: np.ndarray
) -> np.ndarray:
    """Rotate the object in place about ``center_cam`` (camera-space) by angle."""
    Ra = _rodrigues(axis, angle_rad)
    R, t = pose_gt[:3, :3], pose_gt[:3, 3]
    P = np.eye(4, dtype=np.float64)
    P[:3, :3] = Ra @ R
    P[:3, 3] = Ra @ (t - center_cam) + center_cam
    return P


def _perturb_trans(pose_gt: np.ndarray, axis: np.ndarray, delta: float) -> np.ndarray:
    P = pose_gt.copy().astype(np.float64)
    P[:3, 3] = P[:3, 3] + delta * axis / (np.linalg.norm(axis) + 1e-12)
    return P


@torch.no_grad()
def _eval(ctx, pose_np: np.ndarray) -> tuple[float, dict[str, float]]:
    pose_t = torch.from_numpy(pose_np).float().cuda()
    loss, comps, _ = photometric_forward(ctx, pose_t)
    return float(loss), {k: float(v) for k, v in comps.items()}


# ── per-frame analysis ───────────────────────────────────────────────────────


def _load_frame(dataset: LFDataset, idx: int, s_size: int, t_size: int):
    frame = dataset[idx]
    gt_pose = frame["object_pose"].cpu().numpy().astype(np.float64)
    slf = SurfaceLightField.from_frame(
        frame, frame["masks"][s_size // 2, t_size // 2], frame["depth"], s_size, t_size
    )
    return dict(gt_pose=gt_pose, slf=slf)


def _sweep(ctx, gt_pose, center_cam, mode: str, amax_native: float):
    """Return (xs_display, results) sweeping one axis-triplet around GT.

    ``amax_native`` is in the perturbation's native unit (deg for rot, metres
    for trans).  The returned ``xs_display`` is in the plotting unit (deg for
    rot, mm for trans) so the values are readable.
    """
    axes = np.eye(3)
    xs_native = np.linspace(-amax_native, amax_native, N_SAMPLES)
    xs_display = xs_native if mode == "rot" else xs_native * 1000.0
    results = {}
    for ai, axis in enumerate(axes):
        comp_curves: dict[str, list[float]] = {}
        total_curve: list[float] = []
        for x in xs_native:
            if mode == "rot":
                pose = _perturb_rot(gt_pose, axis, np.radians(x), center_cam)
            else:
                pose = _perturb_trans(gt_pose, axis, x)  # x in metres
            total, comps = _eval(ctx, pose)
            total_curve.append(total)
            for k, v in comps.items():
                comp_curves.setdefault(k, []).append(v)
        results[ai] = (comp_curves, total_curve)
    return xs_display, results


def analyze_frame(dataset, idx, s_size, t_size):
    prev = _load_frame(dataset, idx - 1, s_size, t_size)
    curr = _load_frame(dataset, idx, s_size, t_size)

    # Context: anchor at GT; disable reg to see the raw photometric landscape.
    ctx, tgt_rendered = build_photometric_context(
        slf_prev=prev["slf"],
        slf_curr=curr["slf"],
        env_map_prev=None,
        env_map_curr=None,
        abs_pose_prev=prev["gt_pose"],
        pose_coarse=curr["gt_pose"],  # anchor reg at the correct answer
        alpha=ALPHA,
        lambda_depth=LAMBDA_DEPTH,
        lambda_rot=0.0,
        lambda_trans=0.0,
    )

    # object centre in camera space at GT = centroid of transformed source points
    with torch.no_grad():
        T_rel = (
            torch.from_numpy(curr["gt_pose"]).float().cuda() @ ctx.inv_pose_prev
        )
        pts = ctx.slf_prev.points.float()
        pts_gt = (T_rel[:3, :3] @ pts.T).T + T_rel[:3, 3]
        center_cam = pts_gt.mean(0).cpu().numpy().astype(np.float64)

    rot_xs, rot_res = _sweep(ctx, curr["gt_pose"], center_cam, "rot", ROT_MAX_DEG)
    tr_xs, tr_res = _sweep(ctx, curr["gt_pose"], center_cam, "trans", TRANS_MAX_M)

    _print_summary(rot_xs, rot_res, "rot", "deg")
    _print_summary(tr_xs, tr_res, "trans", "mm")
    _plot_landscape(idx, rot_xs, rot_res, tr_xs, tr_res)
    _plot_alignment(idx, ctx, curr["gt_pose"], tgt_rendered)


def _print_summary(xs, res, mode, unit):
    """Per-axis: where the total bottoms out and how it compares to GT (x=0)."""
    names = ["X", "Y", "Z"]
    i0 = int(np.argmin(np.abs(xs)))  # index closest to 0
    for ai in range(3):
        comp_curves, total = res[ai]
        total = np.array(total)
        amin = xs[int(np.argmin(total))]
        loss0 = total[i0]
        lossmin = total.min()
        photo = np.array(comp_curves["photo"])
        ok = "OK " if abs(amin - xs[i0]) < (xs[1] - xs[0]) * 1.5 else "BAD"
        print(
            f"    [{ok}] {mode} {names[ai]}: argmin={amin:+7.2f}{unit}  "
            f"loss@GT={loss0:.5f}  loss@min={lossmin:.5f}  "
            f"photo@GT={photo[i0]:.5f}  photo_range=[{photo.min():.5f},{photo.max():.5f}]"
        )


# ── plotting ─────────────────────────────────────────────────────────────────


def _weighted(comp_curves: dict[str, list[float]]) -> dict[str, np.ndarray]:
    """Return weighted, plot-ready terms (reg excluded — weight 0)."""
    out = {"photo": np.array(comp_curves["photo"])}
    if "depth" in comp_curves:
        out["depth (×%.2g)" % LAMBDA_DEPTH] = LAMBDA_DEPTH * np.array(comp_curves["depth"])
    return out


def _plot_landscape(idx, rot_xs, rot_res, tr_xs, tr_res):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axis_names = ["X", "Y", "Z"]
    for col in range(3):
        # rotation row
        ax = axes[0, col]
        comp_curves, total = rot_res[col]
        for label, ys in _weighted(comp_curves).items():
            ax.plot(rot_xs, ys, label=label, linewidth=1.2)
        ax.plot(rot_xs, total, "k--", label="total", linewidth=1.5)
        ax.axvline(0.0, color="gray", linestyle=":", linewidth=0.8)
        argmin = rot_xs[int(np.argmin(total))]
        ax.set_title(f"Rotation about cam-{axis_names[col]}  (min@{argmin:+.1f}°)")
        ax.set_xlabel("perturbation [deg]")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=7)

        # translation row
        ax = axes[1, col]
        comp_curves, total = tr_res[col]
        for label, ys in _weighted(comp_curves).items():
            ax.plot(tr_xs, ys, label=label, linewidth=1.2)
        ax.plot(tr_xs, total, "k--", label="total", linewidth=1.5)
        ax.axvline(0.0, color="gray", linestyle=":", linewidth=0.8)
        argmin = tr_xs[int(np.argmin(total))]
        ax.set_title(f"Translation along cam-{axis_names[col]}  (min@{argmin:+.1f}mm)")
        ax.set_xlabel("perturbation [mm]")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=7)

    fig.suptitle(
        f"{SPLIT}/{SEQUENCE}  frame {idx}  (prev {idx - 1})  — loss wiggled around GT",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(OUT_DIR, f"landscape_{SEQUENCE}_f{idx:03d}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  saved {out}")


def _plot_alignment(idx, ctx, gt_pose, tgt_rendered):
    """Render source@GT vs target vs |diff| to confirm they align at GT."""
    with torch.no_grad():
        pose_t = torch.from_numpy(gt_pose).float().cuda()
        _, _, aux = photometric_forward(ctx, pose_t)
        src_hwc = aux["src_img"]

    src_u8 = _to_display_u8(src_hwc)
    tgt_u8 = _to_display_u8(tgt_rendered)
    diff = np.abs(src_u8.astype(np.float32) - tgt_u8.astype(np.float32)).mean(-1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(src_u8)
    axes[0].set_title("source @ GT (prev SLF reprojected)")
    axes[1].imshow(tgt_u8)
    axes[1].set_title("target (curr SLF)")
    im = axes[2].imshow(diff, cmap="magma")
    axes[2].set_title(f"|diff|  mean={diff.mean():.1f}")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{SPLIT}/{SEQUENCE}  frame {idx} — alignment at GT", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(OUT_DIR, f"align_{SEQUENCE}_f{idx:03d}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  saved {out}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    seq_path = os.path.join(DATASET_ROOT, SPLIT, SEQUENCE)
    dataset = LFDataset(seq_path, depth_source=DEPTH_SOURCE)
    s_size, t_size = dataset.metadata["n_views"]
    print(f"Analyzing {SPLIT}/{SEQUENCE}  ({len(dataset)} frames)")

    for idx in FRAME_INDICES:
        if idx < 1 or idx >= len(dataset):
            print(f"  skip frame {idx} (out of range)")
            continue
        print(f"frame {idx}:")
        analyze_frame(dataset, idx, s_size, t_size)


if __name__ == "__main__":
    main()
