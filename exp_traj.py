"""Full-trajectory refine-OFF vs refine-ON comparison with real eval metrics.

Runs the actual pipeline (track_sequence) twice per sequence — coarse only, then
with two-stage photometric+relight refine — and reports the rebased-trajectory
ADD-AUC / Rot / ATE for each, so we see whether refinement helps the metric that
matters (not just per-frame). Uses the eval functions from eval/run_eval.py.

Usage:
    python exp_traj.py cube_0.0:bleach0,cracker_box_yalehand0 cube_1.0:bleach0
    python exp_traj.py objects_1.0:mustard0          # one spec = split:seq1,seq2
PIN_ALPHA=1 pins alpha to 1-reflectivity (isolates refinement from alpha est).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "eval"))
from eval.run_eval import eval_sequence, load_gt_poses, load_mesh_pts  # noqa: E402

from loftr_wrapper import LoftrRunner  # noqa: E402
from main import DATASET_ROOT, REFINE_CFG, track_sequence  # noqa: E402
from src.photometric import PhotometricRefineViewer  # noqa: E402

PIN_ALPHA = os.environ.get("PIN_ALPHA", "0") == "1"
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "0")) or None
OUT = "exp_traj_out"


def _metrics(results_dir, split, seq):
    npy = os.path.join(results_dir, "gt", split, f"{seq}.npy")
    est = np.load(npy)
    gt = load_gt_poses(os.path.join(DATASET_ROOT, split, seq))
    n = min(len(est), len(gt))
    pts = load_mesh_pts(DATASET_ROOT, split, seq)
    return eval_sequence(est[:n], gt[:n], pts)


def run_one(loftr, split, seq, refine):
    refl = split.split("_")[1]
    alpha = (1.0 - float(refl)) if PIN_ALPHA else None
    tag = "refined" if refine else "coarse"
    results_dir = os.path.join(OUT, tag, "gt", split)
    os.makedirs(results_dir, exist_ok=True)
    out_npy = os.path.join(results_dir, f"{seq}.npy")
    if os.path.exists(out_npy):
        os.remove(out_npy)
    rng = np.random.default_rng(seed=42)
    track_sequence(
        seq_path=f"{DATASET_ROOT}/{split}/{seq}",
        results_dir=results_dir,
        cache_dir=f"cache/diffuse/gt/{split}/{seq}",
        sequence_name=seq,
        alpha=alpha,
        depth_source="gt",
        loftr=loftr,
        rng=rng,
        separate=True,
        refine=refine,
        viewer=None,
        max_frames=MAX_FRAMES,
        refine_cfg=REFINE_CFG,
    )
    return _metrics(os.path.join(OUT, tag), split, seq)


def main():
    specs = []
    for arg in sys.argv[1:]:
        split, seqs = arg.split(":")
        for s in seqs.split(","):
            specs.append((split, s))

    loftr = LoftrRunner()
    rows = []
    for split, seq in specs:
        if not os.path.isdir(f"{DATASET_ROOT}/{split}/{seq}"):
            print(f"skip missing {split}/{seq}")
            continue
        c = run_one(loftr, split, seq, refine=False)
        r = run_one(loftr, split, seq, refine=True)
        rows.append((split, seq, c, r))
        print(f"\r{split}/{seq} done", flush=True)

    print(f"\n{'='*100}\nPIN_ALPHA={PIN_ALPHA}  (coarse = separation only;  refined = + 2-stage photometric/relight)")
    print(f"{'split/seq':40s} {'ADD-AUC c→r':>22s} {'Rot° c→r':>20s} {'ATE mm c→r':>20s}")
    for split, seq, c, r in rows:
        def arrow(cv, rv, better_up, scale=1.0, prec=2):
            cv, rv = cv*scale, rv*scale
            good = (rv > cv) if better_up else (rv < cv)
            mark = "✓" if good else ("·" if abs(rv-cv) < 1e-9 else "✗")
            return f"{cv:7.{prec}f}→{rv:7.{prec}f}{mark}"
        print(f"{split+'/'+seq:40s} "
              f"{arrow(c['add_auc'], r['add_auc'], True, 1, 4):>22s} "
              f"{arrow(c['mean_abs_rot_deg'], r['mean_abs_rot_deg'], False):>20s} "
              f"{arrow(c['ate_rmse'], r['ate_rmse'], False, 1000):>20s}")
    # aggregate
    if rows:
        ca = np.mean([c['add_auc'] for _,_,c,_ in rows]); ra = np.mean([r['add_auc'] for _,_,_,r in rows])
        cr = np.mean([c['mean_abs_rot_deg'] for _,_,c,_ in rows]); rr = np.mean([r['mean_abs_rot_deg'] for _,_,_,r in rows])
        print(f"{'AVG':40s} {ca:7.4f}→{ra:7.4f}  {cr:7.2f}→{rr:7.2f}")


if __name__ == "__main__":
    main()
