"""Per-split comparison: LoFTR baseline vs our coarse (separation) vs refined.

Reads three metrics.json files and prints, per (block, split), the split-average
ADD-AUC and rotation for each method, with deltas — the table for the paper.
"""
import json
import sys

LOFTR = "eval/results/results_loftr/metrics.json"
COARSE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/eval_coarse/metrics.json"
REFINED = sys.argv[2] if len(sys.argv) > 2 else "/tmp/eval_refined/metrics.json"


def load(p):
    with open(p) as f:
        return json.load(f)["split_averages"]


def main():
    lo, co, re = load(LOFTR), load(COARSE), load(REFINED)
    blocks = ["cube_gt", "objects_gt"]
    print(f"{'split':16s} | {'ADD-AUC  loftr  coarse  refind':32s} | {'Rot°  loftr  coarse  refind':30s}")
    print("-" * 88)
    for blk in blocks:
        if blk not in re:
            continue
        for split in sorted(re[blk]):
            l = lo.get(blk, {}).get(split)
            c = co.get(blk, {}).get(split)
            r = re[blk][split]
            if c is None:
                continue
            la = l['add_auc'] if l else float('nan')
            lr = l['mean_abs_rot_deg'] if l else float('nan')
            # mark refined vs coarse and refined vs loftr
            add_mark = "✓" if r['add_auc'] >= c['add_auc'] - 1e-4 else "✗"
            rot_mark = "✓" if r['mean_abs_rot_deg'] <= c['mean_abs_rot_deg'] + 1e-2 else "✗"
            beat_loftr = "≫L" if (r['add_auc'] > la + 0.02) else ("·L" if r['add_auc'] >= la - 0.005 else "<L")
            print(f"{split:16s} | {la:6.3f} {c['add_auc']:7.3f} {r['add_auc']:7.3f} {add_mark} {beat_loftr:3s}"
                  f" | {lr:5.2f} {c['mean_abs_rot_deg']:7.2f} {r['mean_abs_rot_deg']:7.2f} {rot_mark}")
    # block averages
    print("-" * 88)
    for blk in blocks:
        if blk not in re:
            continue
        import numpy as np
        def avg(d, k):
            return np.mean([d[blk][s][k] for s in d[blk]]) if blk in d else float('nan')
        print(f"{blk+' AVG':16s} | {avg(lo,'add_auc'):6.3f} {avg(co,'add_auc'):7.3f} {avg(re,'add_auc'):7.3f}"
              f"    | {avg(lo,'mean_abs_rot_deg'):5.2f} {avg(co,'mean_abs_rot_deg'):7.2f} {avg(re,'mean_abs_rot_deg'):7.2f}")


if __name__ == "__main__":
    main()
