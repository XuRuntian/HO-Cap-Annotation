# 六视角 RGB 手部估计链路

本分支给已同步的 `rgbd/` 数据增加独立入口。支持 FAH 或 MediaPipe 作为 2D 检测后端，复用 HO-Cap 三角化及 MANO 损失，不运行原来的物体、SAM、HoloLens、深度拟合或离屏渲染链路。原始八阶段脚本仍保留。

输入是采集工具产生的 `rgbd/frames.jsonl`、`cam*/calibration.json`、`calibration/extrinsics.json` 和 RGB 图像。要求六台相机、米制外参，且外参明确声明 `p_master = T_master_from_camera @ p_camera (homogeneous column vectors)`。不猜测其他格式的外参方向。不读取深度，也不重新同步。

## 一键运行（本机）

```bash
cd ~/HO-Cap-Annotation
bash scripts/run_rgb_hand_pipeline.sh ~/ego_datas/3D_Trajectory_Continuity_01
```

一键入口默认 FAH、CUDA 和平均手型，输出到 `hand_annotations_fah/predictions/`。
FAH 检测使用 `$HOME/FAH/.venv/bin/python`，其余阶段默认使用
`$HOME/miniconda3/envs/hocap-annotation/bin/python`，无需手动激活环境。
要使用已标定的手型，显式传 `--hand-shape "$HOME/hand_shapes/02_standard"`。
MANO 默认位于 `$HOME/models/mano`，不会自动安装依赖或下载模型。

先做短测试：

```bash
bash scripts/run_rgb_hand_pipeline.sh ~/ego_datas/3D_Trajectory_Continuity_01 \
  --limit 3 --workdir /tmp/my-hand-test
```

参数沿用 Python 入口，如 `--device cpu`、`--betas shape.json`、`--replace`。
可通过 `FAH_PYTHON`、`FAH_CONFIG`、`MANO_PYTHON`、`MANO_ROOT` 指定路径；
`DETECT_PYTHON` 可单独覆盖检测解释器。失败立即停止。

本机单阶段续跑示例（不要重新执行已有的 prepare）：

```bash
~/miniconda3/envs/hocap-annotation/bin/python tools/rgbd_hand_pipeline.py \
  ~/ego_datas/3D_Trajectory_Continuity_01 --stage fit --device cuda \
  --detector fah --mano-root ~/models/mano --mean-hand
```

若之前指定了 `--workdir`，续跑时必须传入相同目录。

### 使用 FAH 做 2D 检测

一键入口已默认使用 FAH。要切回原流程，传 `--detector mediapipe`，
输出仍位于 `hand_annotations/predictions/`。Python 单阶段入口保留 MediaPipe 兼容默认，
使用 FAH 时指定 `--detector fah`。显式指定 FAH 环境和配置的示例：

```bash
DETECT_PYTHON="$HOME/FAH/.venv/bin/python" \
  bash scripts/run_rgb_hand_pipeline.sh "$DATA" \
  --detector fah --fah-config "$HOME/FAH/configs/local.json" \
  --limit 5 --workdir /tmp/hocap-fah-test
```

也可用 FAH 的 Python 单独运行 `tools/rgbd_hand_pipeline.py "$DATA" --stage detect`，
传入同样的 `--detector`、`--fah-config` 和已经准备好的 `--workdir`。
配置读取 `weights`、`sam3d`、`mano_dir`、`model` 和 `inference.device`；
相对资产路径以配置文件所在目录为基准。FAH 固定使用 PyTorch 单帧模式，
仅导出 2D 像素关节，不使用其 3D、视频跟踪或网格拟合。
观测中的 `detection_score` 是 FAH 手部检测分数，不是 MediaPipe 的
`handedness_score`。三角化、拟合、导出仍由 HO-Cap 环境执行。
批量入口也支持 FAH，见 [批量流程](BATCH_HAND_PIPELINE.md)。

当前机器验证（2026-11-07）：启动器3项测试通过；使用默认本机路径完成
前3帧的 prepare/detect/triangulate/fit/export，三角化与拟合使用 CUDA。
短测只执行5步三角化优化、10步拟合，结果在
`/tmp/hocap-local-launcher-20261107`；质量门限下有效手实例为0，
仅证明环境和启动链路可运行，不代表拟合收敛。完整序列尚未运行。

## 文件职责

默认工作目录为 FAH 的 `DATA_ROOT/hand_annotations_fah` 或 MediaPipe 的
`DATA_ROOT/hand_annotations`，也可用 `--workdir` 指定。

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

## 序列级相机检查与稳健三角化

`triangulate` 阶段默认先检查相机一致性，不增加独立阶段或重新执行检测。
每段独立判断，不会因为某段的 cam02 异常而在其他序列禁用它。

- 默认均匀抽取最多48帧。检查某台相机时，将它完全移出参考重建；
  参考点至少需要另外3个视角形成严格多数共识，并通过正深度、射线夹角检查。
- 每只手至少8个可靠关节才产生误差统计，再聚合成每帧一个样本。
  自动排除需要至少12个有效帧，并覆盖整段4个时间区间中的至少3个
  （每区间至少3帧）。总体和每个有充分样本的区间，异常比例均需达到80%。
  异常门限为 `max(30 px, 3 * --reprojection-px)`。
- 检测稀少、证据不足时只警告，不自动排除。自动排除后必须保留至少
  `max(3, --min-views)` 台相机，否则保留候选并报告歧义。
  检查发现的是几何不一致，不等同于证明外参错误，也不会自动修正标定或左右手分类。
- 三角化先按内点数量和误差选取两视角候选，再仅用其内点优化。
  保留不破坏原共识的最佳迭代，避免异常视角先把候选点拉偏。
  每个关节可使用不同相机组合；默认至少2个视角且射线夹角至少1度。
  平行或近反向射线、负深度及支持不足的关节保持无效，不强行补帧。
- DLT候选与预检查在CPU执行；`--device` 控制内点细化和MANO拟合。
  原始八阶段入口仍沿用原来的三角化函数，本次修改仅接入独立RGB流程。

参数可传给一键脚本或Python入口：

```bash
# 默认自动检查；必要时额外指定整段排除项
python tools/rgbd_hand_pipeline.py "$DATA" --workdir "$WORK" --stage triangulate \
  --exclude-cameras cam02

# 不自动排除整台相机，但仍执行逐帧逐关节内点筛选
python tools/rgbd_hand_pipeline.py "$DATA" --workdir "$WORK" --stage triangulate \
  --no-camera-check
```

`--camera-check-samples` 调整抽样数量，`--min-triangulation-angle` 调整最小夹角。
缩小到不足12帧将无法自动排除相机；这适合短测，不是绕过质量控制的手段。
以上默认值是保守起点，不保证适用于所有分辨率、机位或多人场景。

输出约定：

- `predictions/camera_consistency.json` 保存抽样帧、各相机误差、证据量、
  时间分布、判定原因、阈值及输入哈希。
- `predictions/metadata.json` 新增 `reconstruction`：`camera_selection`、
  `included_camera_ids`、`excluded_cameras`（含 `camera_id` 和 `reason`）、
  `diagnostics_file`、`diagnostics_sha256`、`triangulation_method`。
  included表示允许参与，不保证每帧每关节都实际使用。
- `joints_3d.npz/inliers` 保留全部相机维度，被排除相机全部为false。
  `reprojection_error_px` 仍记录全部有观测相机对有效3D点的残差，包括被排除相机，
  便于诊断；不能将这些数值直接混入内点验收指标。
- `fit_diagnostics.npz` 新增 `[T,C,2]` 的 `camera_observation_median_px`
  和 `camera_inlier_median_px`。前者包含被排除相机的观测残差，后者仅统计
  拟合内点；无对应样本处为NaN。原有整体拟合质量指标仍只使用内点。
- 图像清单、所有相机标定、统一世界坐标及 `mano.npz` 格式不变。
  3DHandsForAll可继续显示所有相机重投影，异常相机不参与拟合，但仍可能显示错位。
  标注软件需支持可选的 `reconstruction` 元数据字段；本机3DHandsForAll已同步
  增加接受并保留该字段的兼容修改，不改变显示或编辑逻辑。
  不应根据异常视角扭曲最终姿态。

报告与三角化通过哈希绑定，手动改动报告后必须重算三角化、拟合及导出。
已有导出的元数据不可原地改变，即便使用 `--replace`；更换相机选择方案或
报告内容时请使用新工作目录，避免旧姿态与新相机记录混配。
旧版三角化仍可拟合、导出，但不会凭空生成相机筛选记录。
采用新逻辑时推荐新建工作目录，复制原 `inputs/` 和
`predictions/observations_2d.npz`，依次运行 triangulate、fit、export，
fit使用与旧实验相同的手形；不要复制或覆盖人工 `final/`。

### 2026-11-14 本机对照验证

选取远端已有结果，复用其2D检测、固定手形及同一组MANO模型；本地重新
prepare后输入清单哈希一致。两段均执行100步内点细化、左右手各10000步CUDA拟合。
最终质量门限未放宽，未插值补帧，也没有覆盖旧结果或远端文件。

| 序列 | 自动排除 | 旧有效帧：右 / 左 | 新有效帧：右 / 左 |
| --- | --- | --- | --- |
| Basic_Grip_01（356帧） | cam02 | 192 / 255 | 356 / 356 |
| 3D_Trajectory_Continuity_01（259帧） | 无 | 259 / 259 | 259 / 259 |

抽查48帧的预检查耗时约0.8秒。两段完整三角化（含预检查）分别约13秒、10秒，
拟合分别约108秒、102秒。预检查把问题段cam02判为持续异常，但保留正常段cam02。

拟合重投影中位误差的跨帧中位数（右 / 左，px）：问题段由7.14 / 6.44变为
7.76 / 6.17，对照段由6.85 / 5.36变为7.04 / 5.52。误差并非全项下降，且
新旧有效帧和内点集合不完全相同；这些是内部残差，不是真值精度。
恢复有效标志不等于人工验收通过。已抽查首、中、末帧六视角网格叠加，
问题相机仍显示明显错位，其余视角仍有局部手形/关节残差，需继续人工检查。

本机输出分别位于：

```text
/home/user/文档/L1-L3-01/Basic_Grip_01/hand_annotations_consistency_20261114/
/home/user/文档/L1-L3-01/3D_Trajectory_Continuity_01/hand_annotations_consistency_20261114/
```

两目录均包含完整去畸变图像清单及预测，可供3DHandsForAll读取，六个相机均保留。
测试：HO-Cap共53项，52通过、1项因历史MANO路径跳过；标注软件无界面套件
48项，30通过、18项因显示/历史资产条件跳过。另用真实输出验证标注软件
加载及全部相机图像读取。GUI交互未在本次完整重测。

## 导入 EasyMocap 固定手形

### 当前三步流程（无需手工转换）

1. 在 EasyMocap 使用独立的 `scripts/calibrate_hand_shape.py` 生成个人手形。
2. 将其输出目录直接传给本仓库的 `--hand-shape`。
3. 使用 3DHandsForAll 打开动作数据进行人工修正。

```bash
cd ~/HO-Cap-Annotation
bash scripts/run_rgb_hand_pipeline.sh /path/to/new_dataset \
  --hand-shape ~/EasyMocap/output/person01_shape
```

新版 EasyMocap 标定入口直接输出 `betas.json` 和 `metadata.json`，
连同试验 NPZ、报告一起保留。加载器优先使用标准 JSON，绝不调用 EasyMocap。
v2 元数据记录模型文件和实际形状基底指纹，HO-Cap 在拟合前使用本机 manopth 校验，
无需生产端仓库、Python 环境或原始数据路径。复制这两个 JSON 即可跨机器使用。
原有 v1 标准目录（例如 `01_standard`、`02_standard`）继续兼容。

只有 `trial_shape.npz` 的历史目录需要在原标定机器使用下方手工工具
转换一次，或用新版入口重新生成。加载时会明确提示，不再偷偷启动转换进程。
这里简化的是手形输入，不改变动作数据的目录要求。

### 历史手工导入方式

EasyMocap 离线估计手形，本仓库只接收 betas，不导入它的姿态或平移。
目前适配的是 `scripts/test_mano_sequence.py` 使用的
`easymocap.smplmodel.load_model` 加载路径，不是任意 EasyMocap 模型配置。

先在 EasyMocap 仓库运行（输入需要已有本链路的检测、三角化和 predictions 元数据）：

```bash
cd ~/EasyMocap
env -u LD_LIBRARY_PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ~/miniconda3/envs/easymocap/bin/python scripts/test_mano_sequence.py \
  /home/user/test_ego_anno/3D_Trajectory_Continuity_01 \
  --out output/shape_rerun_20261112
```

转换时使用 hocap 环境，并校验两边实际加载的 MANO 模板和形状基底。
输出目录必须不存在；重复试验请换目录，不覆盖已经使用的手形：

```bash
cd ~/HO-Cap-Annotation
conda activate hocap-annotation
python tools/import_hand_shape.py \
  ~/EasyMocap/output/shape_rerun_20261112/trial_shape.npz \
  --output ~/hand_shapes/3D_Trajectory_Continuity_01_20261112 \
  --easymocap-root ~/EasyMocap \
  --easymocap-python ~/miniconda3/envs/easymocap/bin/python \
  --mano-root ~/models/mano
```

输入 NPZ 必须包含 `betas [2,10]` 和 `hand_order [2]`，左右手各一次。
工具规范化顺序为右、左，拒绝非有限数值、错误维度和不兼容基底。
输出 `betas.json` 是纯数值数组，可直接传给现有 `--betas`；
`metadata.json` 记录来源哈希、模型哈希、兼容性检查和试验状态。
请将这两个文件一起保留，之后同一个人的新数据不需要安装或运行 EasyMocap：

```bash
bash scripts/run_rgb_hand_pipeline.sh /path/to/new_dataset \
  --hand-shape ~/hand_shapes/3D_Trajectory_Continuity_01_20261112
```

推荐使用 `--hand-shape 目录`：自动读取 `betas.json` 和 `metadata.json`，
在执行阶段前检查格式、左右手顺序、betas 哈希，并在指定 MANO 目录时
校验模型哈希。一键脚本始终提供 MANO 目录，因此会执行模型校验。
原 `--betas 文件` 仍兼容，但不做配套元数据校验；这两个选项与
`--mean-hand` 互斥。Python 分阶段入口同样支持 `--hand-shape`。

已有数据更换手形时，只需重新 fit 和 export，但不能直接替换旧
`mano.npz` 中的 betas。推荐新建工作目录，复制原 `inputs/` 和
`predictions/{observations_2d.npz,joints_3d.npz,run_config.json}`，
然后运行（两条命令必须传同一个工作目录）：

```bash
DATA=/home/user/test_ego_anno/3D_Trajectory_Continuity_01
WORK="$DATA/hand_annotations_shape_20261112"
python tools/rgbd_hand_pipeline.py "$DATA" --workdir "$WORK" --stage fit \
  --mano-root ~/models/mano --device cuda --fit-steps 10000 \
  --betas ~/hand_shapes/3D_Trajectory_Continuity_01_20261112/betas.json
python tools/rgbd_hand_pipeline.py "$DATA" --workdir "$WORK" --stage export
```

不要复制或覆盖人工 `final/`。固定手形只对同一个人复用；模型资产或
加载方式改变后应重新验证兼容性。兼容性通过不代表手形估计准确，仍需
检查 EasyMocap 的保留集骨长误差、拟合诊断和多视角叠加效果。

### 2026-11-12 本机实测

已重新运行 EasyMocap，完成上述转换和全259帧 CUDA 拟合（左右手各10000步），
再导出到 `hand_annotations_shape_20261112/predictions/mano.npz`。
该工作目录的 `shape/` 同时保留本次 `betas.json` 和来源元数据。
原 `hand_annotations/` 和人工 `final/` 未覆盖。
两边 MANO 模板、形状基底的最大绝对差均为0，导出 betas 已与输入核对。

| 指标 | 平均手形：右 / 左 | EasyMocap 手形：右 / 左 |
| --- | --- | --- |
| 有效帧数 | 259 / 259 | 259 / 259 |
| 每帧重投影中位误差的跨帧中位数 (px) | 5.18 / 5.23 | 6.34 / 5.59 |
| 每帧3D关节平均误差的跨帧中位数 (mm) | 6.80 / 6.56 | 8.26 / 6.69 |

这些是相对原2D观测和三角化关节的残差，不是真值精度。
本次固定手形并未改善最终残差，故仅标记为试验手形，不升为默认配置；
尚未进行GUI多视角视觉验收。后续应检查大幅度 betas、骨长观测质量及
关节定义差异，或用专门的手形标定序列，而不是仅凭骨长拟合改善就验收。

测试：转换工具4项、启动脚本3项通过；RGB流水线4项通过、1项因旧机器
MANO路径跳过。该旧测试套件直接运行时存在 `patch.dict(sys.modules)`
触发Torch重复导入的问题，预先 `import torch` 后通过；未改动该无关测试。

## 历史 CPU 机器的环境与命令

本节保留原开发机 `/home/xrt` 的环境与验证记录，不是当前 `/home/user`
机器的默认配置。当前机器使用上方一键命令，无需创建 `.venv-detect`。

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

## 历史验证

```bash
cd /home/xrt/HO-Cap-Annotation
/home/xrt/3DHandsForAll/.venv/bin/python -m unittest discover -s test -p 'test_rgbd_pipeline.py' -v
```

覆盖只读原始数据、覆盖保护、同侧重复检测、三角化离群视角、过期中间数据、短序列平滑、真实 MANO 的 CPU 拟合/统一导出，以及实际 3DHandsForAll 读取契约（本机有该仓库时）。真实数据另进行了前3帧六视角 CPU 流程测试，结果位于 `/tmp/hocap-rgb-smoke`，不在原始数据目录内。

未验证：完整259帧/GPU运行、GUI人工视觉验收、跨相机左右手误分类恢复、个体手型估计与深度约束。同步帧间平滑假设等时间间隔，未做不规则时间间隔加权。CPU测试成功也不代表每帧估计都可接受，应先检查 `fit_diagnostics.npz` 与六视角叠加。

本次真实3帧测试使用三角化5次迭代、拟合10,000次迭代：6个手部实例有5个通过检查，重投影中位误差5.4–8.1 px；第2帧右手误差35.5 px，被标为无效。左右手导出后以平手轴角重新构建 mesh，与内部 PCA mesh 的最大顶点差均为0 mm。实际编辑器加载器已成功读取导出；未进行GUI视觉验收。
