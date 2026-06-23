#!/usr/bin/env bash
# run_fps.sh — FPS + GPU benchmarks on a single object sequence at the highest
# reflectivity (objects_1.0). Run from the project root inside the container
# (see ../run_container.sh):
#
#   bash bench/run_fps.sh                 # bleach0, all three methods
#   SEQ=mustard0 bash bench/run_fps.sh    # different sequence
#
# Each method is a separate script; results print to stdout.
set -e
SEQ="${SEQ:-bleach0}"
REFL="${REFL:-1.0}"
SPLIT="${SPLIT:-objects}"
COMMON="--split $SPLIT --refl $REFL --seq $SEQ"

echo "==================== METHOD (ReLiFT-6DoF) ===================="
python bench/fps_method.py $COMMON

echo "==================== ICP baseline ===================="
python bench/fps_icp.py $COMMON

echo "==================== PnP baseline ===================="
python bench/fps_pnp.py $COMMON
