#!/usr/bin/env bash
# Run eval/run_eval_lift.py over every results folder under baselines_real/.
#
# Usage (inside the lift6dof container)
# -------------------------------------
#   ./run_eval_baselines_real.sh [extra run_eval_lift.py args...]
#
# Outputs go to eval/results/lift_<name>/ and eval/results_qual/lift_<name>/
# so they never collide with the synthetic baselines' eval outputs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$SCRIPT_DIR/baselines_real"

for dir in "$BASELINES_DIR"/*/; do
    name="$(basename "$dir")"
    if [[ ! -d "$dir/gt" && ! -d "$dir/lf" ]]; then
        continue  # not a results folder
    fi
    echo "=== $name ==="
    python "$SCRIPT_DIR/eval/run_eval_lift.py" "$dir" \
        --output-dir "$SCRIPT_DIR/eval/results/lift_${name}" \
        --qual-dir "$SCRIPT_DIR/eval/results_qual/lift_${name}" \
        "$@"
done
