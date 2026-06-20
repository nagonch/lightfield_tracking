#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  PRODUCTION — Experiment 1/4 : FULL pipeline
#
#  The complete ReLiFT-6DoF system: reflection separation + LoFTR coarse pose +
#  photometric refinement.  Runs on BOTH depths — GT (oracle) and ESTIMATED
#  light-field plane-sweep depth (computed live, never read from disk).
#  cube + objects across all reflectivities (scope from config.yaml).
#
#  Output:  ablation_refine_est/<gt|lf>/<split>_<refl>/<seq>.npy
#
#  Usage:   ./production/1_full.sh <gpu:0|1> [run|fps] [extra main.py args...]
#    run   (default)  full sweep on gt + lf depth (reads separation cache).
#    fps              honest steady-state FPS of the realistic full system —
#                     estimated depth + separation recomputed live, segmentor
#                     live, frame 0 / model load excluded.  Reads each
#                     per-sequence "FPS …" line and a final "OVERALL FPS …".
#                     Bounded to MAX_FRAMES frames/sequence; Ctrl-C once you have
#                     enough samples.  Writes throwaway poses under _fps_full/.
#
#  Env overrides:  MAX_FRAMES=80 (fps per-sequence cap; 0 = unbounded)
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${1:-}"
case "$GPU" in 0|1) ;; *) echo "usage: $0 <gpu:0|1> [run|fps] [extra main.py args...]"; exit 1 ;; esac
shift
CMD="${1:-run}"; shift || true

case "$CMD" in
  run)
    echo ">> PRODUCTION 1/4: FULL pipeline (refine + separation, gt + live-lf depth) — GPU $GPU"
    docker exec -e CUDA_VISIBLE_DEVICES="$GPU" -w "$REPO" lift6dof \
      python -u main.py --refine --gt-masks --depth gt,lf --no-cache-depth "$@" ;;
  fps)
    echo ">> PRODUCTION 1/4: FPS of FULL pipeline (live lf depth + live separation) — GPU $GPU"
    fpscap="${MAX_FRAMES:-80}"            # per-sequence cap; set MAX_FRAMES=0 for unbounded
    extra=()
    [[ "$fpscap" != "0" ]] && extra+=(--max-frames "$fpscap")
    docker exec -e CUDA_VISIBLE_DEVICES="$GPU" -w "$REPO" lift6dof \
      python -u main.py --refine --depth lf --no-cache-depth --no-cache-separation \
        --fps --name _fps_full "${extra[@]}" "$@" ;;
  *) echo "usage: $0 <gpu:0|1> [run|fps] [extra main.py args...]"; exit 1 ;;
esac
