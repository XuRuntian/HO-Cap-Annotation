#!/usr/bin/env bash
# Legacy two-directory interface, backed by the provenance-checked batch pipeline.
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    printf '%s\n' \
      'Usage: JOBS=4 bash scripts/run_batch.sh DATA_PARENT HAND_SHAPE_DIRECTORY' \
      'Only direct child sequences containing rgbd/ are selected.' \
      'Environment: JOBS (default 2), MODE (run/precheck/report), BATCH_RESULTS,' \
      'MANO_PYTHON, MANO_ROOT, HAND_MODEL, DEVICE, CPU_THREADS, REPLACE (0/1).' \
      'Resume is always enabled; REPLACE=1 explicitly permits stale-stage recomputation.'
    exit 0
fi
if [[ $# -ne 2 || ! -d $1 || ! -d $2 ]]; then
    echo 'Usage: bash scripts/run_batch.sh DATA_PARENT HAND_SHAPE_DIRECTORY' >&2
    exit 2
fi
JOBS=${JOBS:-2}
if [[ ! $JOBS =~ ^[1-9][0-9]?$ ]] || (( JOBS > 32 )); then
    echo 'JOBS must be an integer between 1 and 32.' >&2
    exit 2
fi
REPO=${HOCAP_REPO:-"$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"}
ROOT=$(cd -- "$1" && pwd -P)
SHAPE=$(cd -- "$2" && pwd -P)
if [[ ! -f "$SHAPE/betas.json" || ! -f "$SHAPE/metadata.json" ]]; then
    echo 'A standard hand shape directory requires betas.json and metadata.json; convert legacy trial_shape first.' >&2
    exit 2
fi
if [[ -z ${MANO_PYTHON:-} ]]; then
    if [[ -x "$REPO/.venv/bin/python" ]]; then
        MANO_PYTHON="$REPO/.venv/bin/python"
    else
        MANO_PYTHON="$HOME/miniconda3/envs/hocap-annotation/bin/python"
    fi
fi
command -v "$MANO_PYTHON" >/dev/null || { echo 'Set MANO_PYTHON to the pipeline Python.' >&2; exit 2; }
MODE=${MODE:-run}
case "$MODE" in run|precheck|report) ;; *) echo 'Invalid MODE.' >&2; exit 2;; esac
REPLACE_ARGS=()
case "${REPLACE:-0}" in
    0) ;;
    1) REPLACE_ARGS=(--replace) ;;
    *) echo 'REPLACE must be 0 or 1.' >&2; exit 2;;
esac

shopt -s nullglob dotglob
SEQUENCES=()
for path in "$ROOT"/*; do
    [[ -d "$path/rgbd" && ! -L "$path" && ! -L "$path/rgbd" ]] || continue
    SEQUENCES+=("./${path##*/}")
done
if (( ${#SEQUENCES[@]} == 0 )); then
    echo 'No direct child sequences containing rgbd/.' >&2
    exit 2
fi
# Hash the canonical root to distinguish equal group names in different captures.
ROOT_ID=$("$MANO_PYTHON" -c 'import hashlib,sys; print(hashlib.sha256(sys.argv[1].encode()).hexdigest()[:12])' "$ROOT")
BATCH_RESULTS=${BATCH_RESULTS:-"${ROOT%/*}/batch_results/${ROOT##*/}-$ROOT_ID"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
unset LD_LIBRARY_PATH
printf 'Sequences: %s; workers: %s; mode: %s\nResults: %s\n' "${#SEQUENCES[@]}" "$JOBS" "$MODE" "$BATCH_RESULTS"
exec "$MANO_PYTHON" "$REPO/tools/batch_rgb_hand_pipeline.py" "$ROOT" \
    --batch-results "$BATCH_RESULTS" --hand-shape "$SHAPE" \
    --mano-root "${MANO_ROOT:-$HOME/models/mano}" \
    --hand-model "${HAND_MODEL:-$REPO/config/checkpoints/mediapipe/hand_landmarker.task}" \
    --device "${DEVICE:-cuda}" --cpu-threads "${CPU_THREADS:-2}" \
    --workers "$JOBS" --mode "$MODE" --resume "${REPLACE_ARGS[@]}" \
    --sequences "${SEQUENCES[@]}"
