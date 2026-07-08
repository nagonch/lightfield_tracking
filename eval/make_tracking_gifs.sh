#!/usr/bin/env bash
# Build the lightweight per-method tracking GIFs (baseline + ablation sets).
#
# make_tracking_gifs.py is self-contained now — it renders the overlay frames and
# encodes each GIF with the static ffmpeg bundled in imageio-ffmpeg (two-pass
# palette, palette shrunk per sequence until the GIF is < 1 MB). This wrapper
# just runs it inside the lift6dof container as the *host* user so the output
# GIFs under eval/gifs/ are owned by you and not by root.
#
# Run on the host (needs docker):
#     bash eval/make_tracking_gifs.sh                       # both sets, all splits
#     bash eval/make_tracking_gifs.sh --set baseline
#     bash eval/make_tracking_gifs.sh --set ablation --split objects_1.0
#     bash eval/make_tracking_gifs.sh --split objects_1.0 --seq sugar_box1
set -euo pipefail

REPO="/home/ngoncharov/cvpr2026/ReLiFT-6DoF"
CONTAINER="lift6dof"

docker exec -u "$(id -u):$(id -g)" -w "$REPO" "$CONTAINER" \
    python eval/make_tracking_gifs.py "$@"
