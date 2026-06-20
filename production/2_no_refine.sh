#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  PRODUCTION — Experiment 2/4 : NO REFINEMENT (ablation)
#
#  Reflection separation stays ON, but the photometric refinement stage is
#  removed: LoFTR tracks the separated diffuse view and its coarse pose is the
#  final estimate.  Isolates the contribution of photometric refinement.
#  Runs on BOTH depths — GT (oracle) and ESTIMATED lf depth (computed live).
#  cube + objects, all reflectivities (from config.yaml).
#
#  Output:  ablation_loftr/<gt|lf>/<split>_<refl>/<seq>.npy
#
#  Usage:   ./production/2_no_refine.sh <gpu:0|1> [extra main.py args...]
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${1:-}"
case "$GPU" in 0|1) ;; *) echo "usage: $0 <gpu:0|1> [extra main.py args...]"; exit 1 ;; esac
shift

echo ">> PRODUCTION 2/4: NO REFINEMENT (separation only, gt + live-lf depth) — GPU $GPU"
docker exec -e CUDA_VISIBLE_DEVICES="$GPU" -w "$REPO" lift6dof \
  python -u main.py --name "ablation_no_refine_gt_mask" --depth gt,lf --gt-masks --no-cache-depth "$@"
