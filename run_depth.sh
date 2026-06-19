#!/usr/bin/env bash
# run_depth.sh — DA3 depth estimation + viser GT-comparison for ReLiFT-6DoF.
#
# Thin wrapper around depth_estimator.py that runs inside the `lift6dof`
# container on GPU 1. All extra args are passed straight through to
# depth_estimator.py (see its --help / argparse for the full list).
#
# Usage:
#   ./run_depth.sh estimate [args...]   # run DA3 + GT-align, cache <out>/<split>_<refl>/<seq>.npz
#   ./run_depth.sh vis      [args...]   # launch viser comparing GT vs estimated depth
#   ./run_depth.sh both     [args...]   # estimate, then vis on the same args
#
# Quick look (one sequence, 5 frames) then inspect in viser:
#   ./run_depth.sh estimate --splits cube   --refls 0.0 --seqs bleach0 --limit 5
#   ./run_depth.sh vis                                   # opens http://localhost:8080
#
# Full sweep over ALL sequences (cube+objects × 0.0/0.5/0.7/1.0), resumable:
#   ./run_depth.sh estimate              # ~3 min/seq; skips already-cached .npz
#   ./run_depth.sh vis                   # browse every object once it's done
#
# Env overrides:  GPU=1  OUT=eval/depth_da3  PORT=8080
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-1}"
OUT="${OUT:-eval/depth_da3}"
PORT="${PORT:-8080}"

CMD="${1:-}"; shift || true

run_estimate() {
  echo ">> estimate (GPU $GPU) -> $OUT"
  docker exec -e CUDA_VISIBLE_DEVICES="$GPU" -w "$REPO" lift6dof \
    python depth_estimator.py --mode estimate --out "$OUT" "$@"
}

run_vis() {
  echo ">> viser at http://localhost:$PORT  (out=$OUT)"
  docker exec -w "$REPO" lift6dof \
    python depth_estimator.py --mode vis --out "$OUT" --port "$PORT" "$@"
}

case "$CMD" in
  estimate) run_estimate "$@" ;;
  vis)      run_vis "$@" ;;
  both)     run_estimate "$@"; run_vis ;;
  *) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
