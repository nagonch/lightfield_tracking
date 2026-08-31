#!/usr/bin/env python3
"""Per-frame runtime of every method on the captured EPI sequences (ms/frame).

Sources (all measured on this machine, one method per GPU at a time):
    baselines_captured/timing_{loftr,icp,pnp}.json   run_baselines_captured.py --time-only
    baselines_captured/results_fp/timing.json        FoundationPose/run_captured.py --time-only (track_one)
    baselines_captured/results_bsdf/timing.json      BundleSDF/run_captured.py --time-only (wall incl. NeRF thread)
    <ours log> 'FPS ... ms/frame mean' lines         main_lift.py --fps   (pass --ours-log)
Writes eval/results_captured_refined/timing_epi.txt.
"""

import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(HERE), "baselines_captured")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours-log", default=None)
    args = ap.parse_args()
    rows = {}
    for m in ("loftr", "icp", "pnp"):
        p = os.path.join(BASE, f"timing_{m}.json")
        if os.path.exists(p):
            rows[m] = {k: v["ms_per_frame"] for k, v in json.load(open(p)).items()}
    for m, sub in (("fp", "results_fp"), ("bsdf", "results_bsdf")):
        p = os.path.join(BASE, sub, "timing.json")
        if os.path.exists(p):
            rows[m] = {k: v["ms_per_frame"] for k, v in json.load(open(p)).items()}
    if args.ours_log and os.path.exists(args.ours_log):
        txt = open(args.ours_log, errors="ignore").read().replace("\r", "\n")
        ours = {}
        for seq, ms in re.findall(r"(epi_\w+?)_prod: FPS [\d.]+\s+\(\d+ frames, ([\d.]+) ms/frame", txt):
            ours[seq] = float(ms)
        if ours:
            rows["ours"] = ours
    seqs = ["epi_diffuse", "epi_reflective"]
    lines = ["== per-frame runtime on the captured EPI sequences [ms/frame] (FPS in parentheses) ==",
             f"{'method':8s} " + " ".join(f"{s:>22s}" for s in seqs) + f" {'mean':>22s}",
             "# fp: track_one only (registration excluded); bsdf: wall time incl. its NeRF thread; "
             "ours: steady-state per frame (frame 0 excluded); classic: tracker.run / frames"]
    for m in ("pnp", "icp", "loftr", "fp", "bsdf", "ours"):
        if m not in rows:
            continue
        vals = [rows[m].get(s) for s in seqs]
        cells = [f"{v:9.1f} ({1000 / v:5.2f} FPS)" if v else f"{'-':>22s}" for v in vals]
        got = [v for v in vals if v]
        mean = sum(got) / len(got)
        lines.append(f"{m:8s} " + " ".join(f"{c:>22s}" for c in cells) + f" {mean:9.1f} ({1000 / mean:5.2f} FPS)")
    txt = "\n".join(lines)
    print(txt)
    out = os.path.join(HERE, "results_captured_refined", "timing_epi.txt")
    with open(out, "w") as f:
        f.write(txt + "\n")
    print("->", out)


if __name__ == "__main__":
    main()
