"""Diagnostic v5: cache per-frame std_L distributions once, then sweep the
aggregation (within-frame percentile q_in, cross-frame percentile q_out,
frontal weighting) + refit linear calibration, all in-memory.

Run once to build the cache (slow), thereafter sweeps are instant.
"""

import os
import pickle
import numpy as np
import torch

from src.dataset import LFDataset
from src.surface_light_field import SurfaceLightField

DATASET_ROOT = "/home/ngoncharov/SpecTrack_dataset"
SEQUENCES = ["cracker_box_yalehand0", "mustard0", "tomato_soup_can_yalehand0",
             "bleach0", "sugar_box1", "mustard_easy_00_02", "sugar_box_yalehand0"]
REFLS = ["0.0", "0.5", "0.7", "1.0"]
SPLITS = ["cube", "objects"]
FRAMES = list(range(0, 60, 6))
MIN_VC = 6
NSUB = 8000  # subsample points per frame for caching
CACHE = "/tmp/alpha_cache.pkl"


def lum(c):
    return c[..., 0] * 0.299 + c[..., 1] * 0.587 + c[..., 2] * 0.114


def build_cache():
    records = []  # dict(split, alpha, seq, fi, std_L[np], frontal[np], vc[np])
    for split in SPLITS:
        for r in REFLS:
            alpha = 1.0 - float(r)
            for seq in SEQUENCES:
                seq_path = os.path.join(DATASET_ROOT, f"{split}_{r}", seq)
                if not os.path.isdir(seq_path):
                    continue
                try:
                    ds = LFDataset(seq_path, depth_source="gt")
                    s_size, t_size = ds.metadata["n_views"]
                    for fi in FRAMES:
                        if fi >= len(ds):
                            continue
                        frame = ds[fi]
                        mask = frame["masks"][s_size // 2, t_size // 2]
                        slf = SurfaceLightField.from_frame(
                            frame, mask, frame["depth"], s_size, t_size
                        )
                        colors = slf.colors.float()
                        valid = slf.valid.float()
                        vd = slf.view_dirs.float()
                        nrm = slf.normals.float()
                        vc = valid.sum(0)
                        keep = vc >= MIN_VC
                        colors, valid, vd = colors[:, keep], valid[:, keep], vd[:, keep]
                        nrm = nrm[keep]
                        vc = vc[keep].clamp(min=1.0)
                        n = int(keep.sum())
                        if n < 50:
                            continue
                        L = lum(colors)
                        mean_L = (L * valid).sum(0) / vc
                        std_L = ((((L - mean_L[None]) ** 2) * valid).sum(0) / vc).clamp(min=0).sqrt()
                        frontal = ((vd * nrm[None]).sum(-1).abs() * valid).sum(0) / vc
                        if n > NSUB:
                            sel = torch.randperm(n, device=std_L.device)[:NSUB]
                            std_L, frontal, vc = std_L[sel], frontal[sel], vc[sel]
                        records.append(dict(
                            split=split, alpha=alpha, seq=seq, fi=fi,
                            std=std_L.cpu().numpy().astype(np.float32),
                            frontal=frontal.cpu().numpy().astype(np.float32),
                            vc=vc.cpu().numpy().astype(np.float32),
                        ))
                except Exception as e:
                    print(f"  !! {split}_{r}/{seq}: {e}")
    with open(CACHE, "wb") as f:
        pickle.dump(records, f)
    print(f"cached {len(records)} frames -> {CACHE}")
    return records


def wpct(x, w, q):
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    cw = np.cumsum(ws) - 0.5 * ws
    cw /= max(ws.sum(), 1e-8)
    return float(np.interp(q, cw, xs))


def frame_stat(rec, q_in, frontal_pow):
    w = (np.clip(rec["frontal"], 0, 1) ** frontal_pow) * rec["vc"]
    return wpct(rec["std"], w, q_in)


def evaluate(records, q_in, q_out, frontal_pow, verbose=False):
    # group per (split, alpha, seq)
    groups = {}
    for rec in records:
        groups.setdefault((rec["split"], rec["alpha"], rec["seq"]), []).append(rec)
    per_key = {}
    for key, recs in groups.items():
        fs = np.array([frame_stat(r, q_in, frontal_pow) for r in recs])
        per_key[key] = float(np.percentile(fs, q_out))
    # fit calibration stat = FLOOR + SLOPE*(1-alpha)
    xs = np.array(list(per_key.values()))
    ys = np.array([1.0 - k[1] for k in per_key])
    A = np.vstack([np.ones_like(ys), ys]).T
    floor, slope = np.linalg.lstsq(A, xs, rcond=None)[0]

    def to_alpha(s):
        return 1.0 - min(max((s - floor) / slope, 0.0), 1.0)

    all_ae, lines = [], []
    for split in SPLITS:
        for r in REFLS:
            alpha = 1.0 - float(r)
            ah = [to_alpha(per_key[k]) for k in per_key if k[0] == split and k[1] == alpha]
            if not ah:
                continue
            ae = [abs(a - alpha) for a in ah]
            all_ae.extend(ae)
            lines.append(f"  {split:8s} {alpha:4.2f}  aHat {np.mean(ah):5.2f}"
                         f"±{np.std(ah):4.2f}  MAE {np.mean(ae):.3f}")
    mae = float(np.mean(all_ae))
    if verbose:
        print(f"floor={floor:.4f} slope={slope:.4f}")
        for ln in lines:
            print(ln)
    return mae, floor, slope


def main():
    if os.path.exists(CACHE):
        records = pickle.load(open(CACHE, "rb"))
        print(f"loaded cache: {len(records)} frames")
    else:
        records = build_cache()

    print("\n--- sweep (q_in, q_out, frontal_pow) ---")
    results = []
    for q_in in [0.3, 0.4, 0.5, 0.6]:
        for q_out in [70, 80, 90]:
            for fp in [0.0, 2.0]:
                mae, floor, slope = evaluate(records, q_in, q_out, fp)
                results.append((mae, q_in, q_out, fp, floor, slope))
    results.sort()
    for mae, q_in, q_out, fp, floor, slope in results[:12]:
        print(f"MAE {mae:.3f}  q_in={q_in} q_out={q_out} fp={fp}  "
              f"FLOOR={floor:.4f} SLOPE={slope:.4f}")

    best = results[0]
    print(f"\n=== BEST: q_in={best[1]} q_out={best[2]} fp={best[3]} ===")
    evaluate(records, best[1], best[2], best[3], verbose=True)


if __name__ == "__main__":
    main()
