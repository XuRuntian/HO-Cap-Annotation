#!/usr/bin/env bash
# Legacy two-directory interface, backed by the provenance-checked batch pipeline.
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    printf '%s\n' \
      'Usage: bash scripts/run_batch.sh DATA_PARENT HAND_SHAPE_DIRECTORY_OR_BETAS_JSON' \
      'Only direct child sequences containing rgbd/ are selected.' \
      'Default: FAH, one worker, results and predictions under DATA_PARENT/batch_results_fah.' \
      'DETECTOR=mediapipe restores the original detector and output locations.' \
      'Environment: DETECTOR, FAH_PYTHON, FAH_CONFIG, JOBS, MODE, BATCH_RESULTS,' \
      'MANO_PYTHON, MANO_ROOT, HAND_MODEL, DEVICE, CPU_THREADS, REPLACE (0/1).' \
      'Resume is always enabled; REPLACE=1 explicitly permits stale-stage recomputation.'
    exit 0
fi
if [[ $# -ne 2 || ! -d $1 ]]; then
    echo 'Usage: bash scripts/run_batch.sh DATA_PARENT HAND_SHAPE_DIRECTORY_OR_BETAS_JSON' >&2
    exit 2
fi
DETECTOR=${DETECTOR:-fah}
case "$DETECTOR" in
    fah) DEFAULT_JOBS=1; RESULTS_NAME=batch_results_fah ;;
    mediapipe) DEFAULT_JOBS=2; RESULTS_NAME=batch_results ;;
    *) echo 'DETECTOR must be fah or mediapipe.' >&2; exit 2 ;;
esac
JOBS=${JOBS:-$DEFAULT_JOBS}
if [[ ! $JOBS =~ ^[1-9][0-9]?$ ]] || (( JOBS > 32 )); then
    echo 'JOBS must be an integer between 1 and 32.' >&2
    exit 2
fi
REPO=${HOCAP_REPO:-"$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"}
ROOT=$(cd -- "$1" && pwd -P)
SHAPE=$2
if [[ -f "$SHAPE" && ${SHAPE##*/} == betas.json ]]; then
    SHAPE=$(dirname -- "$SHAPE")
fi
if [[ ! -f "$SHAPE/betas.json" || ! -f "$SHAPE/metadata.json" ]]; then
    echo 'A standard hand shape directory requires betas.json and metadata.json; convert legacy trial_shape first.' >&2
    exit 2
fi
SHAPE=$(cd -- "$SHAPE" && pwd -P)
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
if [[ $DETECTOR == fah && $MODE != report && $JOBS != 1 ]]; then
    echo 'Use JOBS=1 for FAH detection to avoid GPU memory contention.' >&2
    exit 2
fi
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
BATCH_RESULTS=${BATCH_RESULTS:-"$ROOT/$RESULTS_NAME"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
unset LD_LIBRARY_PATH
printf 'Sequences: %s; detector: %s; workers: %s; mode: %s\nResults: %s\n' \
    "${#SEQUENCES[@]}" "$DETECTOR" "$JOBS" "$MODE" "$BATCH_RESULTS"
exec "$MANO_PYTHON" "$REPO/tools/batch_rgb_hand_pipeline.py" "$ROOT" \
    --batch-results "$BATCH_RESULTS" --hand-shape "$SHAPE" \
    --mano-root "${MANO_ROOT:-$HOME/models/mano}" \
    --hand-model "${HAND_MODEL:-$REPO/config/checkpoints/mediapipe/hand_landmarker.task}" \
    --detector "$DETECTOR" --fah-python "${FAH_PYTHON:-$HOME/FAH/.venv/bin/python}" \
    --fah-config "${FAH_CONFIG:-$HOME/FAH/configs/local.json}" \
    --device "${DEVICE:-cuda}" --cpu-threads "${CPU_THREADS:-2}" \
    --workers "$JOBS" --mode "$MODE" --resume "${REPLACE_ARGS[@]}" \
    --sequences "${SEQUENCES[@]}"
