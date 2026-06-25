#!/usr/bin/env python3
"""Qualitative "grid of objects" of our relightable surface-light-field representation,
reconstructed and rendered EXACTLY as main.py's photometric refinement does.

Per row (one sequence, its best-tracked frame) we show, all object-cropped:

    input | rendered (ours) | depth | surface normals | albedo | shading | reflective

Everything past "input" is produced from the real pipeline objects — the SLF built
by SurfaceLightField.from_frame on LF plane-sweep depth, the separation cache, and
SLF.render_relit (the same gsplat call photometric_forward uses):

  * rendered    = render_relit(env, alpha, mode="relit")  — α·diffuse_shaded + (1-α)·env[reflect]
  * reflective  = render_relit(env, alpha=0,  mode="relit") — the environment warped onto the object
  * albedo      = unshade_to_albedo(diffuse, points, normals), gsplat-rendered
  * shading     = SoftPhong ambient+diffuse factor, gsplat-rendered
  * normals     = camera-oriented surface normals, gsplat-rendered
  * depth       = the rendered (reconstructed) gsplat depth

Rows: top-5 object sequences (by ADD-S AUC) of one reflectivity split + the best
cube sequence. Uses LF-estimated depth (depth_lf / cache .../lf).  GPU required.

Run inside the lift6dof container (idle GPU):
    CUDA_VISIBLE_DEVICES=0 python eval/plot_representation_render_grid.py
    CUDA_VISIBLE_DEVICES=0 python eval/plot_representation_render_grid.py --refl 0.7
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import run_eval as R  # noqa: E402
from config import CACHE_ROOT, PIN_ALPHA  # noqa: E402
from src.dataset import LFDataset  # noqa: E402
from src.reflection import frame_diffuse  # noqa: E402
from src.shading import orient_to_camera, shading_terms  # noqa: E402
from src.surface_light_field import _gs_rasterize_direct  # noqa: E402
from utils import linear_to_srgb, srgb_to_linear  # noqa: E402

# ── what to plot ──────────────────────────────────────────────────────────────
DATASET_ROOT = "/home/ngoncharov/SpecTrack_dataset"
DEPTH = "lf"  # LF plane-sweep depth (depth_lf / cache .../lf / results .../lf)
CACHE_LF = os.path.join(ROOT, CACHE_ROOT, DEPTH)
RESULTS_LF = os.path.join(ROOT, "baselines", "ablation_full_gt_mask", DEPTH)
METRICS = os.path.join(HERE, "results", "ablation_full_gt_mask", "metrics.json")
OUT_DIR = os.path.join(HERE, "plots")

TOPK_OBJECTS = 5
FRAME_MIN_FRAC = 0.15
MIN_MASK_PX = 1500

# Cube row: the metric-best cube trajectory (mustard_easy) has very noisy LF depth;
# these trajectories reconstruct visibly cleaner. Override per reflectivity.
CUBE_SEQ = {"0.7": "sugar_box_yalehand0", "0.5": "sugar_box_yalehand0"}

CROP_PAD = 0.28
CELL_PX = 360
BG = 255  # white background outside the object

COLS = [
    ("input", "Input"),
    ("rendered", "Rendered"),
    ("photo_err", "Photometric loss"),
    ("depth", "Depth"),
    ("normals", "Surface normals"),
    ("diffuse", "Diffuse"),
    ("shading", "Shading"),
    ("reflective", "Reflection"),
]


# ── frame bookkeeping ─────────────────────────────────────────────────────────

def frame_ids(seq_dir):
    pf = sorted(os.listdir(os.path.join(seq_dir, "object_poses")))
    return [p[5:-4] for p in pf]


def cache_dir(split, seq):
    return os.path.join(CACHE_LF, split, seq)


def cache_base(split, seq, i):
    return os.path.join(cache_dir(split, seq), f"diffuse_{i:04d}")


def separation_valid(split, seq, i, min_cov=1000, max_env_nan=0.25):
    """The cached separation is well-populated (some frames separate to a
    degenerate all-zero diffuse + mostly-NaN env map)."""
    b = cache_base(split, seq, i)
    if not all(os.path.exists(b + s + ".npy") for s in ("", "_env", "_envconf")):
        return False
    diff = np.load(b + ".npy")
    cov = int((np.isfinite(diff).all(-1) & (diff.sum(-1) > 1e-3)).sum())
    env = np.load(b + "_env.npy")
    return cov >= min_cov and float(np.isnan(env).mean()) <= max_env_nan


def load_mask_px(seq_dir, fid):
    p = os.path.join(seq_dir, f"LF_{fid}", "masks", f"{R.CENTRAL_VIEW:04d}.png")
    return np.array(Image.open(p).convert("L")) > 10


def pick_best_frame(split, seq, seq_dir, gt, est, mesh_pts):
    """Back-portion frame with the lowest ADD error and a clean, cached separation."""
    fids = frame_ids(seq_dir)
    n = len(fids)
    best, best_err = None, np.inf
    for i in range(int(n * FRAME_MIN_FRAC), n):
        if not separation_valid(split, seq, i):
            continue
        if int(load_mask_px(seq_dir, fids[i]).sum()) < MIN_MASK_PX:
            continue
        err = R._add_err(est[i], gt[i], mesh_pts)
        if err < best_err:
            best, best_err = i, err
    return best, best_err


# ── rendering (exactly the pipeline objects) ──────────────────────────────────

def _render_colors(slf, colors):
    """gsplat rasterize the canonical SLF with per-point colours (same path as
    SLF.render_relit, just supplying our own colours for the intrinsic maps)."""
    img, _, mask = _gs_rasterize_direct(
        means=slf.points.float(),
        quats=slf.quats.float(),
        scales=slf.scales.float(),
        opacities=slf.opacities.float(),
        colors=colors.float(),
        K_mat=slf.K.float(),
        H=slf.H,
        W=slf.W,
    )
    return img, mask


def _to_u8(img_hwc, mask_hw, bg=BG, srgb=False):
    """[H,W,3] (+ object mask) → uint8 RGB with a flat background outside the object."""
    x = img_hwc.detach().float()
    if srgb:
        x = linear_to_srgb(x.clamp(0, 1))
    arr = (x.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    m = mask_hw.detach().cpu().numpy().astype(bool)
    arr[~m] = bg
    return arr


def build_render_panels(slf, env, alpha, input_lin):
    """All appearance/geometry panels for one frame, rendered from the SLF.

    input_lin : [H,W,3] linear observed central view, for the photometric residual.
    Returns (panels, mean_photo_loss).
    """
    env = torch.nan_to_num(env.float()).clamp(0, 1)
    pts, nrm = slf.points.float(), slf.normals.float()

    # full reconstruction: α·diffuse_shaded + (1-α)·env[reflect]  (the refine target)
    rendered, rdepth, rmask = slf.render_relit(
        rel_pose=None, env_map=env, alpha=alpha, mode="relit", shade_diffuse=True
    )

    # photometric residual vs the observed image, the same linear squared error
    # photometric_forward minimises: ((render - target)**2).mean(channels).
    sq_err = ((rendered - input_lin.to(rendered)) ** 2).mean(-1)  # [H,W]
    rmask_np = rmask.detach().cpu().numpy().astype(bool)
    err_np = sq_err.detach().cpu().numpy()
    mean_loss = float(err_np[rmask_np].mean()) if rmask_np.any() else 0.0
    vmax = float(np.percentile(err_np[rmask_np], 98)) if rmask_np.any() else 1.0
    en = np.clip(err_np / max(vmax, 1e-6), 0, 1)
    err_rgb = (matplotlib.colormaps["gray"](en)[..., :3] * 255).astype(np.uint8)
    err_rgb[~rmask_np] = BG
    # reflective component only: the environment warped onto the object
    reflective, _, refmask = slf.render_relit(
        rel_pose=None, env_map=env, alpha=0.0, mode="relit", shade_diffuse=False
    )

    # diffuse component (separated diffuse, shading kept on) — same colours the
    # diffuse refine stage matches; sRGB for display.
    diffuse_lin, _, dfmask = slf.render_relit(
        rel_pose=None, env_map=None, alpha=1.0, mode="diffuse"
    )

    amb_diff, _ = shading_terms(pts, nrm)
    shade = amb_diff.clamp(0, 1).mean(-1, keepdim=True).expand(-1, 3)
    shade_img, smask = _render_colors(slf, shade)

    ncol = (orient_to_camera(F.normalize(nrm, dim=-1), pts) * 0.5 + 0.5).clamp(0, 1)
    normals_img, nmask = _render_colors(slf, ncol)

    # depth → turbo over the object's range
    dmask = rmask.detach().cpu().numpy().astype(bool)
    dnp = rdepth.detach().cpu().numpy()
    if dmask.sum() > 0:
        lo, hi = np.percentile(dnp[dmask], 2), np.percentile(dnp[dmask], 98)
    else:
        lo, hi = 0.0, 1.0
    dn = np.clip((dnp - lo) / max(hi - lo, 1e-6), 0, 1)
    depth_rgb = (matplotlib.colormaps["turbo"](dn)[..., :3] * 255).astype(np.uint8)
    depth_rgb[~dmask] = BG

    panels = {
        "rendered": _to_u8(rendered, rmask, srgb=True),
        "photo_err": err_rgb,
        "reflective": _to_u8(reflective, refmask, srgb=True),
        "diffuse": _to_u8(diffuse_lin, dfmask, srgb=True),
        "shading": _to_u8(shade_img, smask),
        "normals": _to_u8(normals_img, nmask),
        "depth": depth_rgb,
    }
    return panels, mean_loss


# ── per-row assembly ──────────────────────────────────────────────────────────

def crop_box(mask):
    ys, xs = np.where(mask)
    cx, cy = 0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max())
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + CROP_PAD)
    half = side / 2
    H, W = mask.shape
    cx = float(np.clip(cx, half, W - half))
    cy = float(np.clip(cy, half, H - half))
    return int(cx - half), int(cy - half), int(cx + half), int(cy + half)


def crop_resize(rgb_u8, box):
    return np.array(
        Image.fromarray(rgb_u8).crop(box).resize((CELL_PX, CELL_PX), Image.LANCZOS)
    )


def build_row(split, seq):
    seq_dir = os.path.join(DATASET_ROOT, split, seq)
    K = R.load_camera_matrix(seq_dir)
    gt = R.load_gt_poses(seq_dir)
    est = np.load(os.path.join(RESULTS_LF, split, f"{seq}.npy"))
    mesh_pts = R.load_mesh_pts(DATASET_ROOT, split, seq)
    i, err = pick_best_frame(split, seq, seq_dir, gt, est, mesh_pts)
    if i is None:
        return None
    fid = frame_ids(seq_dir)[i]

    # build the SLF + separation exactly as main.py does (LF depth, GT mask, cache)
    ds = LFDataset(seq_dir, depth_source=DEPTH)
    s_size, t_size = ds.metadata["n_views"]
    frame = ds[i]
    mask = frame["masks"][s_size // 2, t_size // 2]
    depth = frame["depth"]
    alpha_in = (1.0 - float(split.split("_")[-1])) if PIN_ALPHA else None
    diffuse, env, _env_conf, slf, alpha, _ = frame_diffuse(
        frame=frame,
        mask=mask,
        depth=depth,
        alpha=alpha_in,
        s_size=s_size,
        t_size=t_size,
        cache_path=cache_base(split, seq, i) + ".png",
        verbose=False,
    )

    # observed central view (real image) → linear, for the photometric residual
    central = np.array(
        Image.open(
            os.path.join(seq_dir, f"LF_{fid}", f"{R.CENTRAL_VIEW:04d}.png")
        ).convert("RGB")
    )
    input_lin = srgb_to_linear(torch.from_numpy(central.astype(np.float32) / 255.0)).cuda()

    panels, photo_loss = build_render_panels(slf, env, float(alpha), input_lin)

    # crop everything to the object (GT mask bbox); input shown on white background
    mask_np = (mask > 0).cpu().numpy()
    box = crop_box(mask_np)
    central[~mask_np] = BG
    out = {"input": crop_resize(central, box)}
    for k, v in panels.items():
        out[k] = crop_resize(v, box)
    return {
        "seq": seq,
        "split": split,
        "frame": i,
        "fid": fid,
        "add_mm": err * 1000,
        "alpha": float(alpha),
        "photo_loss": photo_loss,
        "panels": out,
    }


# ── plotting ──────────────────────────────────────────────────────────────────

def pretty_name(split, seq):
    return R.get_object_name(DATASET_ROOT, split, seq).replace("_", " ").title()


def top_object_sequences(split):
    import json

    m = json.load(open(METRICS))["per_sequence"]["objects_gt"][split]
    return [s for s, _ in sorted(m.items(), key=lambda kv: kv[1]["adds_auc"], reverse=True)]


def best_cube_sequence(refl):
    if refl in CUBE_SEQ:
        return CUBE_SEQ[refl]
    import json

    m = json.load(open(METRICS))["per_sequence"]["cube_lf"][f"cube_{refl}"]
    return max(m.items(), key=lambda kv: kv[1]["adds_auc"])[0]


def render_figure(refl, rows):
    nrow, ncol = len(rows), len(COLS)
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(ncol * 1.7, nrow * 1.7),
        gridspec_kw={"wspace": 0.03, "hspace": 0.03},
        squeeze=False,
    )
    for r, row in enumerate(rows):
        for c, (key, header) in enumerate(COLS):
            ax = axes[r][c]
            ax.imshow(row["panels"][key])
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("0.85")
                s.set_linewidth(1.0)
            if r == 0:
                ax.set_title(header, fontsize=11, fontweight="bold", pad=6)
            if c == 0:
                ax.set_ylabel(
                    pretty_name(row["split"], row["seq"]),
                    fontsize=11,
                    fontweight="bold",
                    labelpad=6,
                )

    fig.suptitle(
        "Relightable Surface Light Field Representation",
        fontsize=15,
        fontweight="bold",
        y=0.997,
    )
    fig.subplots_adjust(left=0.065, right=0.997, top=0.93, bottom=0.006)
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = f"representation_render_grid_lf_{refl}"
    for ext in ("png", "pdf"):
        out = os.path.join(OUT_DIR, f"{stem}.{ext}")
        fig.savefig(out, dpi=300)
        print("Saved →", out)
    plt.close(fig)


def run_refl(refl):
    split = f"objects_{refl}"
    rows = []
    seen_types = set()
    for seq in top_object_sequences(split):
        if len(rows) >= TOPK_OBJECTS:
            break
        # one row per object type: keep the highest-ranked sequence of each type
        otype = R.get_object_name(DATASET_ROOT, split, seq)
        if otype in seen_types:
            continue
        row = build_row(split, seq)
        if row is None:
            print(f"  {split}/{seq}: no valid frame, skipping")
            continue
        print(f"  {split}/{seq}: #{row['fid']} (i={row['frame']}) "
              f"ADD {row['add_mm']:.1f}mm  alpha {row['alpha']:.2f}")
        seen_types.add(otype)
        rows.append(row)

    cube_split = f"cube_{refl}"
    cube_seq = best_cube_sequence(refl)
    cube_row = build_row(cube_split, cube_seq)
    if cube_row is not None:
        print(f"  {cube_split}/{cube_seq}: #{cube_row['fid']} (i={cube_row['frame']}) "
              f"ADD {cube_row['add_mm']:.1f}mm  alpha {cube_row['alpha']:.2f}")
        rows.append(cube_row)

    if rows:
        render_figure(refl, rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refl", default=None, help="0.5 or 0.7 (default: both)")
    ap.add_argument("--split", default=None, help="single split, e.g. objects_0.7")
    ap.add_argument("--seq", default=None, help="single sequence (with --split)")
    args = ap.parse_args()

    if args.split and args.seq:
        row = build_row(args.split, args.seq)
        if row is not None:
            render_figure(args.split.split("_")[-1], [row])
    else:
        for refl in [args.refl] if args.refl else ["0.7", "0.5"]:
            print(f"[reflectivity {refl}]")
            run_refl(refl)
