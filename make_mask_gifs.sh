#!/usr/bin/env bash
# Build one animated GIF per sequence from the per-frame mask visualisations
# written by `python segmentor.py` (folders under masks_vis/<split>_<refl>__<seq>/
# containing 0000.png, 0001.png, ...).
#
# There is no ffmpeg on the training box — copy masks_vis/ somewhere with ffmpeg
# and run this there. It only reads/writes inside masks_vis/.
set -euo pipefail

VIS_ROOT="/home/ngoncharov/cvpr2026/ReLiFT-6DoF/masks_vis"
GIF_ROOT="$VIS_ROOT/gifs"
FPS=10

mkdir -p "$GIF_ROOT"

for seq_dir in "$VIS_ROOT"/*/; do
    seq_dir="${seq_dir%/}"
    name="$(basename "$seq_dir")"
    [ "$name" = "gifs" ] && continue

    shopt -s nullglob
    frames=("$seq_dir"/*.png)
    shopt -u nullglob
    if [ ${#frames[@]} -eq 0 ]; then
        echo "[$name] no frames, skipping"
        continue
    fi

    out="$GIF_ROOT/$name.gif"
    echo "[$name] ${#frames[@]} frames -> $out"

    # Two-pass palette → clean colours / small file. Frames are %04d from 0000.
    palette="$(mktemp --suffix=.png)"
    ffmpeg -y -framerate "$FPS" -i "$seq_dir/%04d.png" -vf "palettegen" "$palette"
    ffmpeg -y -framerate "$FPS" -i "$seq_dir/%04d.png" -i "$palette" \
        -lavfi "paletteuse" "$out"
    rm -f "$palette"
done

echo "done -> $GIF_ROOT"
