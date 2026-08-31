#!/usr/bin/env python3
"""Calibrate a brightness-invariant alpha statistic on the synthetic dataset.

Eq. (11) of the paper uses the ABSOLUTE per-point luminance spread std_j l(y_ij);
its calibration therefore scales with object brightness (a 2x brighter object
reads 2x more "reflective"). The relative spread std_j l / mean_j l is invariant
to albedo/exposure. This script computes both statistics on the synthetic
sequences with known alpha = 1 - r (split x r in {0, .5, .7, 1}), fits the linear
calibration (1 - alpha) = (stat - b) / s for the relative statistic (single-frame
and accumulated-p80, mirroring reflection_separation.py), reports the per-sequence
MAE, and finally prints the predicted alpha for the captured EPI sequences.

Run inside lift6dof:
    CUDA_VISIBLE_DEVICES=0 python experiments/calib_alpha_relative.py
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import reflection_separation as rs
from src.dataset import LFDataset, set_lift_options
from src.surface_light_field import SurfaceLightField

SYN_ROOT = "/home/ngoncharov/SpecTrack_dataset"
EPI_ROOTS = {
    "epi (shipped calib)": "/home/ngoncharov/cvpr2026/datasets/EPI_LF_dataset",
    "epi (self-calib)": "/home/ngoncharov/cvpr2026/datasets/EPI_LF_dataset_cal",
}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval", "results",
                   "alpha_relative_calibration.json")


def stats_for_slf(slf):
    """(absolute stat, relative stat) with the production weighting/percentile."""
    colors = slf.colors.float().permute(1, 0, 2)  # [P,M,3]
    valid = slf.valid.permute(1, 0).float()
    view_dirs = slf.view_dirs.permute(1, 0, 2).float()
    normals = slf.normals.float()
    vc = valid.sum(1)
    keep = vc >= rs.ALPHA_MIN_VALID_VIEWS
    if int(keep.sum()) < rs.ALPHA_MIN_POINTS:
        return None, None
    colors, valid, vc = colors[keep], valid[keep], vc[keep].clamp(min=1.0)
    lw = torch.tensor(rs._LUM_WEIGHTS, device=colors.device, dtype=colors.dtype)
    lum = (colors * lw).sum(-1)
    mean_l = (lum * valid).sum(1) / vc
    std_l = (((lum - mean_l[:, None]) ** 2 * valid).sum(1) / vc).clamp(min=0).sqrt()
    frontal = ((view_dirs[keep] * normals[keep][:, None, :]).sum(-1).abs() * valid).sum(1) / vc
    weight = frontal.clamp(0.0, 1.0) ** rs.ALPHA_FRONTAL_POW * vc
    abs_stat = rs._weighted_quantile(std_l, weight, rs.ALPHA_WITHIN_Q)
    rel_stat = rs._weighted_quantile(std_l / mean_l.clamp(min=1e-3), weight, rs.ALPHA_WITHIN_Q)
    return abs_stat, rel_stat


def sequence_stats(seq_path, frames, stride=None):
    if stride is not None:
        set_lift_options(view_stride=stride)
    ds = LFDataset(seq_path, depth_source="lf")
    S, T = ds.metadata["n_views"]
    out = []
    for i in frames:
        if i >= len(ds):
            break
        fr = ds[i]
        mask = fr["masks"][S // 2, T // 2]
        slf = SurfaceLightField.from_frame(fr, mask, fr["depth"], S, T)
        a, r = stats_for_slf(slf)
        if a is not None:
            out.append((a, r))
    return np.array(out)


def fit(x, y):
    """(1-alpha) = (stat - b)/s  <=>  stat = b + s*(1-alpha); lstsq over (1-alpha, stat)."""
    A = np.stack([np.ones_like(x), x], 1)
    b, s = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(b), float(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="0,5,10,15,20,25,30,35,40,45")
    ap.add_argument("--splits", default="cube,objects")
    args = ap.parse_args()
    frames = [int(f) for f in args.frames.split(",")]
    rows = []  # (split, seq, r, abs_p80, rel_p80, abs_single_mean, rel_single_mean)
    for split in args.splits.split(","):
        for r in (0.0, 0.5, 0.7, 1.0):
            root = os.path.join(SYN_ROOT, f"{split}_{r}")
            for seq in sorted(os.listdir(root)):
                p = os.path.join(root, seq)
                if not os.path.isdir(os.path.join(p, "depth_lf")):
                    continue
                st = sequence_stats(p, frames)
                if len(st) == 0:
                    continue
                p80 = np.quantile(st, rs.ALPHA_ACROSS_Q, axis=0)
                rows.append((split, seq, r, p80[0], p80[1], st[:, 0].mean(), st[:, 1].mean(), st))
                print(f"  {split:7s} r={r:.1f} {seq:28s} abs p80 {p80[0]:.4f}  rel p80 {p80[1]:.4f}", flush=True)
    rr = np.array([row[2] for row in rows])
    abs80 = np.array([row[3] for row in rows]); rel80 = np.array([row[4] for row in rows])
    # accumulated (p80) calibration
    b_rel, s_rel = fit(rr, rel80)
    b_abs, s_abs = fit(rr, abs80)
    # single-frame calibration over all frames
    fr_r = np.concatenate([[row[2]] * len(row[7]) for row in rows])
    fr_abs = np.concatenate([row[7][:, 0] for row in rows]); fr_rel = np.concatenate([row[7][:, 1] for row in rows])
    b_rel1, s_rel1 = fit(fr_r, fr_rel)

    def mae(stat, cal):
        alpha_hat = np.clip(1 - (stat - cal[0]) / cal[1], rs.ALPHA_CLAMP_MIN, rs.ALPHA_CLAMP_MAX)
        return np.abs(alpha_hat - (1 - rr)).mean(), alpha_hat

    m_abs_old, _ = mae(abs80, rs.ALPHA_CAL_ACCUM)
    m_abs_new, _ = mae(abs80, (b_abs, s_abs))
    m_rel, a_rel = mae(rel80, (b_rel, s_rel))
    print("\n== accumulated-p80 calibration (stat = b + s*(1-alpha)) ==")
    print(f" absolute, shipped   b={rs.ALPHA_CAL_ACCUM[0]:.4f} s={rs.ALPHA_CAL_ACCUM[1]:.4f}  per-seq MAE {m_abs_old:.3f}")
    print(f" absolute, refit     b={b_abs:.4f} s={s_abs:.4f}  per-seq MAE {m_abs_new:.3f}")
    print(f" RELATIVE, fit       b={b_rel:.4f} s={s_rel:.4f}  per-seq MAE {m_rel:.3f}")
    print(f" RELATIVE single-frame b={b_rel1:.4f} s={s_rel1:.4f}")
    print("\n per-sequence alpha_hat (relative, p80) vs truth:")
    for row, ah in zip(rows, a_rel):
        print(f"  {row[0]:7s} {row[1]:28s} true {1 - row[2]:.2f}  rel-hat {ah:.2f}   (abs stat {row[3]:.4f}, rel stat {row[4]:.4f})")
    for split in args.splits.split(","):
        sel = np.array([row[0] == split for row in rows])
        print(f"  {split}: MAE {np.abs(a_rel[sel] - (1 - rr[sel])).mean():.3f}")

    print("\n== captured EPI sequences: predicted alpha ==")
    epi = {}
    for label, root in EPI_ROOTS.items():
        for seq in ("epi_diffuse_prod", "epi_reflective_prod"):
            p = os.path.join(root, seq)
            if not os.path.isdir(p):
                continue
            st = sequence_stats(p, list(range(0, 47, 4)), stride=1)
            p80 = np.quantile(st, rs.ALPHA_ACROSS_Q, axis=0)
            a_abs = np.clip(1 - (p80[0] - rs.ALPHA_CAL_ACCUM[0]) / rs.ALPHA_CAL_ACCUM[1], 0.03, 0.97)
            a_relh = np.clip(1 - (p80[1] - b_rel) / s_rel, 0.03, 0.97)
            epi[f"{label}/{seq}"] = dict(abs_p80=float(p80[0]), rel_p80=float(p80[1]), alpha_abs=float(a_abs), alpha_rel=float(a_relh),
                                         rel_per_frame=st[:, 1].round(4).tolist())
            print(f"  {label:20s} {seq:22s} abs p80 {p80[0]:.4f} -> alpha {a_abs:.2f} | rel p80 {p80[1]:.4f} -> alpha {a_relh:.2f}   "
                  f"rel per-frame: {np.round(st[:, 1], 3)}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(dict(cal_rel_accum=(b_rel, s_rel), cal_rel_single=(b_rel1, s_rel1), cal_abs_refit=(b_abs, s_abs),
                       mae_rel=float(m_rel), mae_abs_shipped=float(m_abs_old),
                       rows=[dict(split=r[0], seq=r[1], r=r[2], abs_p80=float(r[3]), rel_p80=float(r[4])) for r in rows],
                       epi=epi), f, indent=1)
    print("->", OUT)


if __name__ == "__main__":
    main()
