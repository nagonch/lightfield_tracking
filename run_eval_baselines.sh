#!/usr/bin/env bash
# Run eval/run_eval.py over every results folder under baselines/.
#
# Usage
# -----
#   ./run_eval_baselines.sh [extra run_eval.py args...]
#
# Any extra arguments (e.g. --no-qual, --dataset-root PATH) are forwarded to
# every run_eval.py invocation. Each <baselines>/<name>/ directory is skipped
# unless it contains at least one of the expected {gt,synth,lf} pose subdirs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$SCRIPT_DIR/baselines"

for dir in "$BASELINES_DIR"/*/; do
    name="$(basename "$dir")"
    if [[ ! -d "$dir/gt" && ! -d "$dir/synth" && ! -d "$dir/lf" ]]; then
        continue  # not a results folder (e.g. baselines/__pycache__, scripts)
    fi
    echo "=== $name ==="
    python "$SCRIPT_DIR/eval/run_eval.py" "$dir" "$@"
done
