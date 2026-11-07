# 六视角 RGB 手部估计链路

本分支给已同步的 `rgbd/` 数据增加独立入口。复用 MediaPipe 检测、HO-Cap 三角化及 MANO 损失，不运行原来的物体、SAM、HoloLens、深度拟合或离屏渲染链路。原始八阶段脚本仍保留。

输入是采集工具产生的 `rgbd/frames.jsonl`、`cam*/calibration.json`、`calibration/extrinsics.json` 和 RGB 图像。要求六台相机、米制外参，且外参明确声明 `p_master = T_master_from_camera @ p_camera (homogeneous column vectors)`。不猜测其他格式的外参方向。不读取深度，也不重新同步。

## 文件职责

默认工作目录是 `DATA_ROOT/hand_annotations`，也可用 `--workdir` 指定独立开发目录。

```text
DATA_ROOT/
├── rgbd/                           # 原始数据，只读
└── hand_annotations/
    ├── inputs/
    │   ├── manifest.json           # 帧号、相机、去畸变标定、时间戳及图像校验值
    │   └── rgb/camXX/*.jpg          # 单独生成的去畸变 RGB（JPEG quality 95）
    ├── predictions/
    │   ├── observations_2d.npz      # 可替换检测器的交接点
    │   ├── joints_3d.npz
    │   ├── fit_parameters.npz       # 内部 PCA 参数及输入来源校验值
    │   ├── fit_diagnostics.npz      # 误差、有效标志及损失曲线
    │   ├── run_config.json
    │   ├── mano.npz                # 标注工具的统一输入
    │   └── metadata.json
    └── final/                      # 仅由人工标注工具生成/维护
        ├── mano.npz
        └── metadata.json
```

`prepare → detect → triangulate → fit → export` 可逐阶段运行。替换检测器只需产生同样的 `observations_2d.npz`；标注工具仅依赖统一的 MANO 文件与图像清单，无需知道估计器名称。

- 中间 NPZ 都包含 `frame_ids`、`camera_ids`、`manifest_sha256`，不可改变帧或相机顺序。
- 2D：`keypoints [T,C,2,21,2]` 为去畸变图像像素，`valid [T,C,2,21]` 为 bool，关节顺序为 MediaPipe 21 点；手顺序固定右、左。`handedness_score` 是左右手分类置信度，不是关节置信度。重复同侧检测只保留分数最高者，不把另一检测伪装成另一只手。尚未实现跨相机纠正错误的左右手分类。
- 三角化：至少两视角、正深度、重投影内点检查；缺失点通过 `valid` 表达。
- 拟合：复用 2D/3D 关节、姿态先验及连续有效段平滑，不跨缺失段平滑。当前默认接受标准为内点重投影中位误差 ≤25 px，三维关节平均误差 ≤5 cm；这是初筛，不替代人工检查。
- 手型整段共享。必须明确选 `--mean-hand`（零 betas，平均手型，**不是个体手型估计**）或 `--betas shape.json`（右、左各10维）。本阶段不学习 betas。
- `mano.npz`：`frame_ids [T]`、`betas [2,10]`、`hand_pose [T,2,45]`、`global_orient [T,2,3]`、`transl [T,2,3]`、`valid [T,2]`。绝对轴角、弧度、MANO v1.2 原生左右手、平手均值约定、米制主相机坐标，translation 为原生 MANO 原点平移（不是直接把腕点当平移）。PCA+均值在导出时转换为45维绝对轴角。无效条目不应渲染成有效预测。
- 输入哈希阻止混用旧中间结果。重算用 `--replace`，下游需顺序重算；已有 `final/` 时禁止替换 `inputs/`，请另建工作目录。不会自动更新或覆盖人工结果。

## 本机已验证的环境与命令

检测与拟合可在不同环境/机器分阶段执行；移机复制完整工作目录，保持 manifest 和图像内容不变。本机现有 MANO 环境是 NumPy 1.23.5 / Torch 2.5.1 CPU / manopth；为避免 MediaPipe 新依赖升级其 NumPy，检测使用独立环境：

```bash
cd /home/xrt/HO-Cap-Annotation
uv venv --python /home/xrt/3DHandsForAll/.venv/bin/python .venv-detect
uv pip install --python .venv-detect/bin/python -r requirements-rgb-detect.txt
mkdir -p config/checkpoints/mediapipe
curl -L --fail -o config/checkpoints/mediapipe/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

模型来自 [MediaPipe 官方 Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)。MANO 资产需要自行取得，不随仓库提交。

完整数据处理（CPU 拟合可能较慢，先用 `--limit 3 --workdir /tmp/your-hand-test` 在 prepare 阶段做短测试）：

```bash
cd /home/xrt/HO-Cap-Annotation
DATA=/home/xrt/Downloads/3D_Trajectory_Continuity_01
PY=/home/xrt/3DHandsForAll/.venv/bin/python
$PY tools/rgbd_hand_pipeline.py "$DATA" --stage prepare
.venv-detect/bin/python tools/rgbd_hand_pipeline.py "$DATA" --stage detect \
  --hand-model config/checkpoints/mediapipe/hand_landmarker.task
$PY tools/rgbd_hand_pipeline.py "$DATA" --stage triangulate --device cpu
$PY tools/rgbd_hand_pipeline.py "$DATA" --stage fit --device cpu \
  --mano-root /home/xrt/models/mano_v1_2/models --mean-hand
$PY tools/rgbd_hand_pipeline.py "$DATA" --stage export
```

每一阶段都要传相同的自定义 `--workdir`（若使用）。独立拟合环境需要 torch、numpy、scipy、opencv、chumpy、manopth 及其 MANO 依赖；本次复用现有编辑器环境，尚未验证全新 GPU 环境安装。GPU 机器为 triangulate/fit 改成 `--device cuda`，检测默认仍是 CPU。如果一个环境已兼容全部依赖，可使用默认 `--stage all`，同时提供 `--hand-model --mano-root --mean-hand`。

打开人工标注：

```bash
cd /home/xrt/3DHandsForAll
.venv/bin/python pose_annotator.py \
  --project /home/xrt/Downloads/3D_Trajectory_Continuity_01/hand_annotations/inputs/manifest.json \
  --input /home/xrt/Downloads/3D_Trajectory_Continuity_01/hand_annotations/predictions/mano.npz \
  --mano-root /home/xrt/models/mano_v1_2/models --device cpu
```

## 验证

```bash
cd /home/xrt/HO-Cap-Annotation
/home/xrt/3DHandsForAll/.venv/bin/python -m unittest discover -s test -p 'test_rgbd_pipeline.py' -v
```

覆盖只读原始数据、覆盖保护、同侧重复检测、三角化离群视角、过期中间数据、短序列平滑、真实 MANO 的 CPU 拟合/统一导出，以及实际 3DHandsForAll 读取契约（本机有该仓库时）。真实数据另进行了前3帧六视角 CPU 流程测试，结果位于 `/tmp/hocap-rgb-smoke`，不在原始数据目录内。

未验证：完整259帧/GPU运行、GUI人工视觉验收、跨相机左右手误分类恢复、个体手型估计与深度约束。同步帧间平滑假设等时间间隔，未做不规则时间间隔加权。CPU测试成功也不代表每帧估计都可接受，应先检查 `fit_diagnostics.npz` 与六视角叠加。

本次真实3帧测试使用三角化5次迭代、拟合10,000次迭代：6个手部实例有5个通过检查，重投影中位误差5.4–8.1 px；第2帧右手误差35.5 px，被标为无效。左右手导出后以平手轴角重新构建 mesh，与内部 PCA mesh 的最大顶点差均为0 mm。实际编辑器加载器已成功读取导出；未进行GUI视觉验收。
