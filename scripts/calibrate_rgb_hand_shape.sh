#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo 'Usage: bash scripts/calibrate_rgb_hand_shape.sh CLIP_DIR NEW_SHAPE_DIR [--samples-per-camera N]'
    exit 0
fi
if [[ $# -lt 2 || ! -d $1 || -e $2 || -L $2 ]]; then
    echo 'Expected a clip directory and a new shape output directory.' >&2
    exit 2
fi
clip=$1
output=$2
shift 2
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
worker="$repo/tools/calibrate_rgb_hand_shape.py"
tmp=$(mktemp -d)
trap 'rm -f -- "$tmp/samples.json"; rmdir -- "$tmp"' EXIT
export TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-1}
unset LD_LIBRARY_PATH PYTHONPATH
"${FAH_PYTHON:-$HOME/FAH/.venv/bin/python}" "$worker" collect "$clip" \
    "${FAH_CONFIG:-$HOME/FAH/configs/local.json}" "$tmp/samples.json" "$@"
"${MANO_PYTHON:-$HOME/miniconda3/envs/hocap-annotation/bin/python}" \
    "$worker" publish "$tmp/samples.json" "$output" "${MANO_ROOT:-$HOME/models/mano}"
