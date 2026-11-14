#!/usr/bin/env bash
# Run the existing stages with separate detector/MANO Python environments.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -eq 0 || ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'HELP'
用法：bash scripts/run_rgb_hand_pipeline.sh 数据根目录 [选项]

自动执行：去畸变 → 2D检测 → 三角化 → MANO拟合 → 统一导出。
默认使用 CUDA、平均手型；MediaPipe 检测使用 CPU。
输出到 数据根目录/hand_annotations。

常用选项（直接传给已有 Python 入口）：
  --limit 3 --workdir /tmp/hand-test   先测试前三帧
  --device cpu                       改用 CPU 三角化和拟合
  --betas shape.json                  使用指定手型，替代平均手型
  --hand-shape /path/to/shape         EasyMocap 输出目录或标准手形目录（自动适配校验）
  --mano-root /path/to/models         指定 MANO 资产
  --hand-model /path/to/model.task    指定 MediaPipe 模型
  --exclude-cameras cam02            手动排除指定相机参与重建，保留重投影显示
  --no-camera-check                  关闭整段自动排除，仍逐关节筛选异常观测
  --camera-check-samples 48           整段均匀抽样数量
  --replace                          重算已有结果，不覆盖 final/

环境变量：MANO_PYTHON、DETECT_PYTHON、MANO_ROOT、HAND_MODEL。
可设置两个 Python 为同一个已安装全部依赖的环境。
中断后可使用 tools/rgbd_hand_pipeline.py --stage ... 单独继续。
HELP
    exit 0
fi
DATA_ROOT=$1
shift
SHAPE_ARGS=(--mean-hand)
for arg in "$@"; do
    case "$arg" in
        --stage|--stage=*) echo '一键脚本执行全部阶段；单阶段请使用 Python 入口。' >&2; exit 2 ;;
        --betas|--betas=*|--hand-shape|--hand-shape=*|--mean-hand) SHAPE_ARGS=() ;;
    esac
done
MANO_PYTHON=${MANO_PYTHON:-"$HOME/miniconda3/envs/hocap-annotation/bin/python"}
export MANO_PYTHON
DETECT_PYTHON=${DETECT_PYTHON:-"$MANO_PYTHON"}
MANO_ROOT=${MANO_ROOT:-"$HOME/models/mano"}
HAND_MODEL=${HAND_MODEL:-"$REPO_ROOT/config/checkpoints/mediapipe/hand_landmarker.task"}
for interpreter in "$MANO_PYTHON" "$DETECT_PYTHON"; do
    if ! command -v "$interpreter" >/dev/null 2>&1; then
        echo "找不到 Python：$interpreter；请设置 MANO_PYTHON / DETECT_PYTHON。" >&2
        exit 2
    fi
done
printf '数据：%s\n默认使用 CUDA 和平均手型；可通过参数覆盖。\n' "$DATA_ROOT"
for stage in prepare detect triangulate fit export; do
    interpreter=$MANO_PYTHON
    if [[ $stage == detect ]]; then interpreter=$DETECT_PYTHON; fi
    "$interpreter" "$REPO_ROOT/tools/rgbd_hand_pipeline.py" "$DATA_ROOT" \
        --stage "$stage" --device cuda --mano-root "$MANO_ROOT" \
        --hand-model "$HAND_MODEL" "${SHAPE_ARGS[@]}" "$@"
done
printf '流程完成。默认结果：%s/hand_annotations/predictions/mano.npz\n若指定 --workdir，结果位于该目录的 predictions/。\n' "$DATA_ROOT"
