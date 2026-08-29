#!/usr/bin/env bash
# run_lift.sh — benchmark the pipeline on the real LiFT dataset in the lift6dof
# container.  All extra args pass through to main_lift.py / run_eval_lift.py.
#
# Usage:
#   ./run_lift.sh track  [args...]   # main_lift.py: track sequences → <exp>/<depth>/<seq>.npy
#   ./run_lift.sh depth  [args...]   # main_lift.py --write-lf-depth: write depth_lf/*.png
#   ./run_lift.sh eval   <exp> [...] # run_eval_lift.py: metrics + qualitative overlays
#
# Typical benchmark:
#   ./run_lift.sh track --gt-masks                    # LoFTR backbone, RealSense depth
#   ./run_lift.sh track --refine --gt-masks           # full method
#   ./run_lift.sh depth                               # pre-write LF plane-sweep depth
#   ./run_lift.sh track --refine --gt-masks --depth lf
#   ./run_lift.sh eval  lift_refine_est --gifs
#
# Tuning probe (one sequence, isolated output, fresh separation):
#   ./run_lift.sh track --refine --gt-masks --seqs jug_tilt --no-cache-separation \
#       --set refine.lambda_depth=1.0 --name sweeps/lift_ld1.0
#
# Env overrides:  GPU=0
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-0}"

CMD="${1:-}"; shift || true

case "$CMD" in
  track)
    docker exec -e CUDA_VISIBLE_DEVICES="$GPU" -w "$REPO" lift6dof \
      python -u main_lift.py "$@" ;;
  depth)
    docker exec -e CUDA_VISIBLE_DEVICES="$GPU" -w "$REPO" lift6dof \
      python -u main_lift.py --write-lf-depth "$@" ;;
  eval)
    docker exec -w "$REPO" lift6dof \
      python -u eval/run_eval_lift.py "$@" ;;
  *) sed -n '2,21p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
