#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  PRODUCTION — Experiment 3/4 : NO REFINEMENT + NO SEPARATION (LoFTR-only)
#
#  Both reflection separation AND photometric refinement are removed: LoFTR
#  tracks the RAW central view directly.  The plain-LoFTR baseline the pipeline
#  falls back to; isolates the contribution of reflection separation.
#  Runs on BOTH depths — GT (oracle) and ESTIMATED lf depth (computed live).
#  cube + objects, all reflectivities (from config.yaml).
#
#  Output:  ablation_no_separation/<gt|lf>/<split>_<refl>/<seq>.npy
#
#  Usage:   ./production/3_no_separation.sh <gpu:0|1> [extra main.py args...]
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${1:-}"
case "$GPU" in 0|1) ;; *) echo "usage: $0 <gpu:0|1> [extra main.py args...]"; exit 1 ;; esac
shift

echo ">> PRODUCTION 3/4: NO REFINE + NO SEPARATION (raw-view LoFTR, gt + live-lf depth) — GPU $GPU"
docker exec -e CUDA_VISIBLE_DEVICES="$GPU" -w "$REPO" lift6dof \
  python -u main.py --no-cache-separation --gt-masks --depth gt,lf --no-cache-depth "$@"
