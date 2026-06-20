#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  PRODUCTION — Experiment 4/4 : NO REFINEMENT + NO SEPARATION on SYNTHETIC depth
#
#  Same LoFTR-only ablation as #3 (no separation, no refinement, raw central
#  view), but on the dataset's SYNTHETIC depth (depth_synth) instead of GT or
#  estimated depth — the lower-bound baseline with noisy off-the-shelf depth.
#  Synthetic depth is loaded from disk; nothing is estimated here.
#  cube + objects, all reflectivities (from config.yaml).
#
#  Output:  ablation_no_separation/synth/<split>_<refl>/<seq>.npy
#
#  Usage:   ./production/4_synth_depth.sh <gpu:0|1> [extra main.py args...]
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${1:-}"
case "$GPU" in 0|1) ;; *) echo "usage: $0 <gpu:0|1> [extra main.py args...]"; exit 1 ;; esac
shift

echo ">> PRODUCTION 4/4: NO REFINE + NO SEPARATION on SYNTHETIC depth — GPU $GPU"
docker exec -e CUDA_VISIBLE_DEVICES="$GPU" -w "$REPO" lift6dof \
  python -u main.py --name "ablation_synth_depth_gt_mask" --no-cache-separation --gt-masks --depth synth "$@"
