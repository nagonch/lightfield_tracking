#!/usr/bin/env bash
# run_lf_depth.sh — light-field plane-sweep depth (lf_depth.py) in the lift6dof
# container.  A feed-forward, non-trainable depth estimator that replaces the noisy
# RealSense-style stereo depth (depth_synth) with dense, smooth, hole-free metric
# depth.  All extra args pass through to lf_depth.py (see --help).
#
# Usage:
#   ./run_lf_depth.sh estimate [args...]  # score vs GT + depth_synth -> <OUT>/<split>_<refl>/<seq>.npz
#   ./run_lf_depth.sh write    [args...]  # save depth_lf/*.png into each sequence (for main.py)
#   ./run_lf_depth.sh vis      [args...]  # viser: GT vs ours vs synth point clouds
#
# Quick look (diffuse + mirror cube, 5 frames) then inspect:
#   ./run_lf_depth.sh estimate --splits cube --refls 0.0,1.0 --seqs bleach0 --limit 5
#   ./run_lf_depth.sh vis
#
# Full sweep / write depth for the tracker:
#   ./run_lf_depth.sh estimate            # all cube+objects x 0.0/0.5/0.7/1.0
#   ./run_lf_depth.sh write               # then main.py can use depth_source="lf"
#
# Env overrides:  GPU=0  OUT=eval/depth_lf  PORT=8080
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-0}"
OUT="${OUT:-eval/depth_lf}"
PORT="${PORT:-8080}"

CMD="${1:-}"; shift || true

case "$CMD" in
  estimate)
    echo ">> estimate (GPU $GPU) -> $OUT"
    docker exec -e CUDA_VISIBLE_DEVICES="$GPU" -w "$REPO" lift6dof \
      python -u lf_depth.py --mode estimate --out "$OUT" "$@" ;;
  write)
    echo ">> write depth_lf/*.png into sequences (GPU $GPU)"
    docker exec -e CUDA_VISIBLE_DEVICES="$GPU" -w "$REPO" lift6dof \
      python -u lf_depth.py --mode write "$@" ;;
  vis)
    echo ">> viser at http://localhost:$PORT  (out=$OUT)"
    docker exec -w "$REPO" lift6dof \
      python lf_depth.py --mode vis --out "$OUT" --port "$PORT" "$@" ;;
  *) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
