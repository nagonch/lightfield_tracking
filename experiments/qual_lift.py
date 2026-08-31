#!/usr/bin/env python3
"""Qualitative intermediates of the tuned pipeline on the real LiFT dataset.

Runs the separation front-end (tuned config_lift.yaml, LF depth, GT masks,
live — no cache) on every non-car sequence and saves, per frame, everything
the paper/supplementary can draw from:

    eval/qual_lift/<seq>/
      rgb/XXXX.png          central sub-aperture view (sRGB)
      depth/XXXX.png        LF plane-sweep depth, turbo colormap over in-mask range
      normals/XXXX.png      SLF per-point normals (camera space, n -> (n+1)/2 RGB)
      diffuse/XXXX.png      separated diffuse component (sRGB)
      reflection/XXXX.png   reflection residual (central - alpha*diffuse)/(1-alpha)
      env/XXXX.png          accumulated environment map (sRGB)
      env_conf/XXXX.png     env observation-confidence (gray)
      composite/XXXX.png    one row: rgb|depth|normals|diffuse|reflection|env (object crop)
      env_evolution.png     strip of the env map at ~6 evenly spaced frames
      alpha_curve.png       estimated diffuse fraction alpha over the sequence
      alpha.json            per-frame alpha values

Split across GPUs with --seqs, e.g.:
  GPU0: python experiments/qual_lift.py --seqs box_motion,jug_motion,jug_tilt,jug_translation
  GPU1: python experiments/qual_lift.py --seqs shiny_box,teabox

Captured EPI cross (LiFT-format copy from convert_epi_to_lift.py; 1x17 views, so
the stride must be 1 exactly as in the tracked baselines_captured/ours_epi run):
  python experiments/qual_lift.py --root ~/cvpr2026/datasets/EPI_LF_dataset \
      --view-stride 1 --out eval/qual_epi
"""

import argparse
import json
import os
import sys

os.environ.setdefault("CONFIG_OVERRIDES", "config_lift.yaml")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import config as _cfg
from config import ALPHA_STABLE_TOL, SEPARATION_ITERS
from src.dataset import LFDataset, set_lift_options
from src.reflection import frame_diffuse
from utils import linear_to_srgb

LIFT_ROOT = "/home/ngoncharov/cvpr2026/datasets/LiFT_dataset"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _srgb8(linear_hw3: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(linear_hw3)).clamp(0.0, 1.0)
    return (linear_to_srgb(t).numpy() * 255).astype(np.uint8)


def _save(path: str, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path)


def _depth_vis(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    d = depth.copy()
    dm = d[mask & (d > 0)]
    lo, hi = (np.percentile(dm, 2), np.percentile(dm, 98)) if len(dm) else (0, 1)
    norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = (cm.get_cmap("turbo")(norm)[..., :3] * 255).astype(np.uint8)
    rgb[~mask] = (rgb[~mask] * 0.25).astype(np.uint8)  # dim background
    return rgb


def _normals_vis(slf, H: int, W: int) -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.uint8)
    n = slf.normals.detach().cpu().numpy()
    # orient toward the camera (-z) for a consistent color scheme
    n = n * np.where(n[:, 2:3] > 0, -1.0, 1.0)
    rgb = ((n * 0.5 + 0.5) * 255).astype(np.uint8)
    m = slf.mask.cpu().numpy() > 0
    img[m] = rgb
    return img


def _env_vis(env: torch.Tensor | None, shape=(128, 256)) -> np.ndarray:
    if env is None:
        return np.zeros((*shape, 3), dtype=np.uint8)
    return _srgb8(env.detach().cpu().numpy().astype(np.float32))


def _crop_box(masks: list, H: int, W: int, pad_frac=0.3):
    x0, y0, x1, y1 = W, H, 0, 0
    for m in masks:
        ys, xs = np.where(m)
        if len(xs):
            x0, y0 = min(x0, xs.min()), min(y0, ys.min())
            x1, y1 = max(x1, xs.max()), max(y1, ys.max())
    pad = pad_frac * max(x1 - x0, y1 - y0)
    return (
        int(max(0, x0 - pad)),
        int(max(0, y0 - pad)),
        int(min(W, x1 + pad)),
        int(min(H, y1 + pad)),
    )


def process_sequence(seq: str, out_root: str, root: str = LIFT_ROOT) -> None:
    seq_path = os.path.join(root, seq)
    out = os.path.join(out_root, seq)
    subdirs = ["rgb", "depth", "normals", "diffuse", "reflection", "env", "env_conf", "composite"]
    for d in subdirs:
        os.makedirs(os.path.join(out, d), exist_ok=True)

    ds = LFDataset(seq_path, depth_source="lf")
    s_size, t_size = ds.metadata["n_views"]

    prev_env = prev_env_conf = None
    alpha_history: list[float] = []
    alpha_stable = False
    prev_alpha_i = None
    alpha_pinned = None  # set by the alpha veto (mirrors track_sequence)
    alpha_veto_checked = False
    alphas = []
    env_frames = []
    crop = None
    all_masks = []

    for i in range(len(ds)):
        frame = ds[i]
        mask = frame["masks"][s_size // 2, t_size // 2]
        depth = frame["depth"]
        mask_np = (mask > 0).cpu().numpy()
        all_masks.append(mask_np)

        was_stable = alpha_stable
        view, prev_env, prev_env_conf, slf, alpha_i, alpha_history = frame_diffuse(
            frame=frame,
            mask=mask,
            depth=depth,
            alpha=alpha_pinned,
            s_size=s_size,
            t_size=t_size,
            cache_path=None,
            iterations=SEPARATION_ITERS,
            verbose=False,
            previous_environment_map=prev_env if was_stable else None,
            previous_env_confidence=prev_env_conf if was_stable else None,
            alpha_stat_history=alpha_history,
        )
        if not alpha_stable and (
            prev_alpha_i is not None
            and abs(alpha_i - prev_alpha_i) <= ALPHA_STABLE_TOL
        ):
            alpha_stable = True
        prev_alpha_i = alpha_i

        # Alpha veto (mirrors main.py track_sequence): once the estimate
        # stabilises low, probe whether the reflection model explains any
        # cross-view variance; pin near-diffuse alpha if it does not.
        if (
            _cfg.ALPHA_VETO_ENABLED
            and alpha_pinned is None
            and alpha_stable
            and not alpha_veto_checked
        ):
            alpha_veto_checked = True
            if alpha_i < _cfg.ALPHA_VETO_EST_MAX:
                from reflection_separation import reflection_explained_ratio

                ratio = reflection_explained_ratio(
                    slf,
                    probe_alpha=_cfg.ALPHA_VETO_PROBE,
                    iterations=SEPARATION_ITERS,
                )
                if ratio > _cfg.ALPHA_VETO_RATIO:
                    alpha_pinned = _cfg.ALPHA_VETO_CLAMP
                    print(
                        f"  {seq}: alpha veto (est {alpha_i:.2f}, ratio "
                        f"{ratio:.3f}) → pinned {alpha_pinned}",
                        flush=True,
                    )

        alphas.append(float(alpha_i))

        central = frame["LF"][s_size // 2, t_size // 2].cpu().numpy().astype(np.float32)
        depth_np = depth.cpu().numpy()
        H, W = depth_np.shape

        rgb8 = _srgb8(central)
        diffuse8 = _srgb8(view)
        depth8 = _depth_vis(depth_np, mask_np)
        normals8 = _normals_vis(slf, H, W)
        # reflection residual under the model obs = a*diffuse + (1-a)*env
        resid = np.clip(central - alpha_i * view, 0.0, 1.0) / max(1.0 - alpha_i, 0.05)
        resid[~mask_np] = 0.0
        refl8 = _srgb8(np.clip(resid, 0.0, 1.0))
        env8 = _env_vis(prev_env)
        conf8 = (
            (prev_env_conf.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            if prev_env_conf is not None
            else np.zeros(env8.shape[:2], dtype=np.uint8)
        )

        _save(f"{out}/rgb/{i:04d}.png", rgb8)
        _save(f"{out}/diffuse/{i:04d}.png", diffuse8)
        _save(f"{out}/depth/{i:04d}.png", depth8)
        _save(f"{out}/normals/{i:04d}.png", normals8)
        _save(f"{out}/reflection/{i:04d}.png", refl8)
        _save(f"{out}/env/{i:04d}.png", env8)
        _save(f"{out}/env_conf/{i:04d}.png", conf8)
        env_frames.append(env8)
        print(f"  {seq} f{i:03d}  alpha={alpha_i:.3f}", flush=True)

    # composites (object crop) + env evolution + alpha curve
    crop = _crop_box(all_masks, H, W)
    n = len(ds)
    for i in range(n):
        panels = []
        for d in ["rgb", "depth", "normals", "diffuse", "reflection"]:
            img = Image.open(f"{out}/{d}/{i:04d}.png").crop(crop)
            panels.append(img)
        ph = panels[0].height
        env_img = Image.open(f"{out}/env/{i:04d}.png")
        env_img = env_img.resize((int(env_img.width * ph / env_img.height), ph))
        panels.append(env_img)
        total_w = sum(p.width for p in panels)
        row = Image.new("RGB", (total_w, ph), (0, 0, 0))
        x = 0
        for p in panels:
            row.paste(p, (x, 0))
            x += p.width
        row.save(f"{out}/composite/{i:04d}.png")

    pick = np.unique(np.linspace(0, n - 1, 6).astype(int))
    strip = np.concatenate([env_frames[j] for j in pick], axis=1)
    _save(f"{out}/env_evolution.png", strip)

    plt.figure(figsize=(5, 3))
    plt.plot(range(n), alphas, marker="o", ms=3)
    plt.xlabel("frame")
    plt.ylabel(r"estimated diffuse fraction $\alpha$")
    plt.title(seq)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out}/alpha_curve.png", dpi=150)
    plt.close()
    with open(f"{out}/alpha.json", "w") as f:
        json.dump({"alpha": alphas}, f, indent=2)
    print(f"[done] {seq} -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default=None, help="comma-separated substrings")
    ap.add_argument(
        "--out", default=os.path.join(REPO, "eval", "qual_lift"), help="output root"
    )
    ap.add_argument("--root", default=LIFT_ROOT, help="LiFT-format dataset root")
    ap.add_argument(
        "--view-stride",
        type=int,
        default=2,
        help="LF view thinning, must match the tracked run (2 for LiFT 9x9, "
        "1 for the 1x17 EPI cross)",
    )
    args = ap.parse_args()
    set_lift_options(view_stride=args.view_stride)

    seqs = [
        d
        for d in sorted(os.listdir(args.root))
        if d.endswith("_prod")
        and not d.startswith("car_")
        and os.path.isdir(os.path.join(args.root, d))
    ]
    if args.seqs:
        keys = [k for k in args.seqs.split(",") if k]
        seqs = [s for s in seqs if any(k in s for k in keys)]
    print(f"sequences: {seqs}", flush=True)
    for seq in seqs:
        process_sequence(seq, args.out, args.root)


if __name__ == "__main__":
    main()
