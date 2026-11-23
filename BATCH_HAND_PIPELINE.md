# 批量检查、重建与分类

入口：`tools/batch_rgb_hand_pipeline.py`。每个序列独立判定，不修复外参，
不修改3DHandsForAll，不移动或删除原始数据，不触碰人工 `final/`。

## 模式

| 模式 | 行为 |
| --- | --- |
| `--mode report` | 只读检查原始输入、已有观测和阶段凭据；写报告及分类链接，不生成输入图像、不执行检测/三角化/拟合 |
| `--mode precheck` | 必要时prepare和检测，随后检查几何及覆盖；不拟合 |
| `--mode run` | 通过预检查后执行triangulate、fit、export，再按最终质量分类 |

预检查通过但尚未验证完整拟合/导出的序列仍归入 `03_needs_review`，
有效帧数留空，不会提前标成正常。程序返回0也不代表所有序列合格：
待检查属于正常处理结果；任一序列发生 `04_failed` 时批量入口最终返回1，
但仍继续处理后续序列。

## 命令

### RGB 手形标定与批量标注

每位受试者用一段多视角 RGB 采集标定一次。这里传入实际的 `clip_000`，
其中包含 `rgbd/camXX/color_*.jpg` 和彩色相机内参。抽帧不要求相机间同步，
不读取深度或相机外参。标定需要包含
[原生左右手 MANO 拟合支持](https://github.com/XuRuntian/FAH/commit/0fb2c540ca97d9844694ee391a3fe266c39f6548)
的 FAH 版本。标定环境、推理配置和 MANO 目录分别由 `FAH_PYTHON`、`FAH_CONFIG`、
`MANO_PYTHON`、`MANO_ROOT` 指定。

```bash
bash scripts/calibrate_rgb_hand_shape.sh "$HOME/文档/head_clip/01/clip_000" \
  "$HOME/hand_shapes/02_fah_v1_trial"
bash scripts/run_batch.sh "$HOME/文档/normal-samples-02/L1-L3-02" \
  "$HOME/hand_shapes/02_fah_v1_trial/betas.json"
```

标定输出目录必须是新的。双手各自按相机聚合，保存 `betas.json` 和同目录的
`metadata.json`（MANO 基底、文件校验值和标定来源）；新结果默认标记为待复核。
同目录自动生成两份空白中文表：`lengths.yaml`（MANO 骨段）和
`lengths_niosh.yaml`（真人外部尺寸），仅在需要人工修正时选填其中一份。
跑批可传上述文件或它所在的目录。批处理只读该手形，不更新 β；后续每段的
`mano.npz` 是包含逐帧姿态的标注结果，不是另一次标定。批量输入虽然仍位于
各序列的 `rgbd/`，但只需要 RGB、相机标定和同步帧清单，不需要深度图。

### 骨段长度修正（选做）

默认直接用 FAH 标定结果。只有拿到与 MANO 标志点定义一致的参考长度时，才运行
下面的命令。直接填写手形目录里的 `lengths.yaml`；
旧结果没有此文件时，可使用[带中文注释的 YAML 模板](config/hand_bone_lengths.template.yaml)。
也兼容 JSON。左右手可分别提供目标，保持 `null` 的字段跳过，不填的一侧原样保留。

```yaml
units: mm
left:
  thumb_ip_to_tip: 31.0  # 拇指末段
  index_total: 76.0      # 食指三段长度之和
```

```bash
"${MANO_PYTHON:-$HOME/miniconda3/envs/hocap-annotation/bin/python}" \
  tools/refine_hand_shape_lengths.py "$HOME/hand_shapes/02_fah_v1_trial/betas.json" \
  "$HOME/hand_shapes/02_fah_v1_trial/lengths.yaml" "$HOME/hand_shapes/02_fah_refined_v1"
```

示例数字不是 02 号被试的实测值。长度基于平手姿态下的 MANO 关节点和网格指尖，
并非从皮肤横纹量到指尖的外部长度，不能不加区分地填入：

- `thumb_cmc_to_mcp`、`thumb_mcp_to_ip`、`thumb_ip_to_tip`：拇指三段。
- `index`、`middle`、`ring`、`little` 可接 `_proximal`、`_middle`、
  `_distal` 分别指定三段；五指均可用 `_total` 指定三段长度之和。
- `palm_length` 是腕点到中指根，`palm_width` 是食指根到小指根。

修正限制 β 的变化，并保护未指定的长度，不强行追求目标零误差。
查看新版的 `refinement.json` 中的修正前后长度和剩余偏差，确认投影后再用于跑批。
输出到新目录，原手型不改；不在每帧改变 β，也不要求增加深度或真人测量流程。

### 真人外部尺寸修正（选做）

按 [NIOSH Figure 1](https://stacks.cdc.gov/view/cdc/36887/cdc_36887_DS1.pdf#page=16)
和[附录 A](https://stacks.cdc.gov/view/cdc/36887/cdc_36887_DS1.pdf#page=13)量手掌摊开时的
外部尺寸，填写手形目录内的 `lengths_niosh.yaml`。不需要把实测值换算成骨长。
早期手形目录没有测量表时，可使用
[NIOSH 中文模板](config/hand_niosh_external.template.yaml)。单位为毫米，可只填任意几项、
也可以只填一只手；全空时返回原手形文件，不生成新目录。

支持 Figure 1 的 14 项：整手长、手宽、掌长、掌宽，以及每根手指的长度和宽度。
其中手宽（2）是四指根部附近外缘之间的距离，掌宽（4）用掌横纹附近的外缘，
五指宽在各自从指根数的第一道关节处横向测量。工具比较平手姿态下 MANO 网格的
同类投影尺寸，用固定的网格表面标记近似腕横纹、指根和可见外缘。它们不是内部
关节中心，也不是对真人皮肤横纹或指宽的精确重建。整手长是掌长加中指长；
三项全填时，整手长仅写入报告（`used_in_fit: false`），不重复加权。
已有的 `lengths.yaml` 仍按 MANO 骨段解释；两套字段不能混填。

```bash
"${MANO_PYTHON:-$HOME/miniconda3/envs/hocap-annotation/bin/python}" \
  tools/refine_hand_shape_lengths.py "$HOME/hand_shapes/02_fah_v1_trial/betas.json" \
  "$HOME/hand_shapes/02_fah_v1_trial/lengths_niosh.yaml" "$HOME/hand_shapes/02_fah_niosh_v1"
```

输出仍是 `betas.json`、`metadata.json` 和 `refinement.json`。
修正保持接近原 β，保护没填写的尺寸；查看报告和独立片段上的双手投影后再使用。

### 兼容旧批量接口

新版 `scripts/run_batch.sh` 保留“数据组目录 + 手形路径”两个位置参数，
只选择该组直接子目录中的序列，不递归运行其它数据组，也不推断人员手形：

```bash
bash scripts/run_batch.sh "$DATA_PARENT" "$HOME/hand_shapes/02_standard"
```

默认用 FAH 做 2D 检测，三角化、MANO 拟合及导出保持不变。
检测使用 `$HOME/FAH/.venv/bin/python` 和 `$HOME/FAH/configs/local.json`，
可通过 `FAH_PYTHON`、`FAH_CONFIG` 覆盖，不必手动激活环境。
汇总在 `DATA_PARENT/batch_results_fah/summary.csv`，各序列输入与预测位于
`batch_results_fah/work/<序列ID>/`；具体工作目录可在该序列的 details 报告中查看。
分类链接仍指向原始序列，不替换原序列的 `hand_annotations/` 或人工 `final/`。
`MODE=precheck` 可先验证检测和重建条件，再用同一目录续跑完整拟合。

需要走原来的 MediaPipe 链路时：

```bash
DETECTOR=mediapipe bash scripts/run_batch.sh "$DATA_PARENT" "$HOME/hand_shapes/02_standard"
```

FAH 的检测和重建固定用一个 worker，避免多个序列同时加载模型；只读报告不受此限制。
MediaPipe 的 `JOBS` 默认2，范围1至32，表示同时处理的序列数，不是同时处理的数据组数。
多进程使用spawn，各序列独立工作；父进程统一维护一个summary.csv及四个分类目录。
CUDA设备由所有worker共享，4并行不保证显存足够，OOM时减少JOBS。
JOBS大于1时每个序列独立进程，普通异常或单个worker崩溃会记录失败，
其它序列继续处理。JOBS=1保留原串行实现，隔离Python异常，但原生崩溃会
终止该批次。机器级资源耗尽仍可能影响多个任务。不要同时启动两个任务写同一个结果目录。

MediaPipe 的默认结果位置为 `数据组目录/batch_results/`：全组一份summary.csv，每个序列一行。
四个分类目录内是指向原序列的软链接，不移动或复制原始数据。例如：

```text
data/L1-L3-01/
  Basic_Grip_01/
    rgbd/
    hand_annotations/
      inputs/
      predictions/
      final/        # 标注软件保存的人工结果，批量流程不覆盖
  3D_Trajectory_Continuity_01/
  batch_results/
    summary.csv
    01_normal/       # 原序列软链接，链接名包含唯一标识
    02_degraded/
    03_needs_review/
    04_failed/
    details/
```

在此 MediaPipe 布局中，`L1-L3-02`、`L4-L6-02` 各自在自己的目录下生成组级结果。
每个序列的准备图像和预测保存在自己的 `hand_annotations/`，与旧单序列流程一致。
分类链接指向原序列；打开后可找到该序列的hand_annotations。
标注软件仍可直接 `python pose_annotator.py /path/to/Basic_Grip_01`，
自动优先恢复final，否则读取predictions；本布局不要求改标注软件的路径逻辑。
可以通过 `BATCH_RESULTS=/absolute/path` 固定输出目录。两种后端应使用不同的输出目录。
默认总是启用resume，
不会像旧脚本那样看到hand_annotations存在就跳过；`REPLACE=1` 才允许重算已有
未验证阶段（仍受final及导出元数据保护约束）。不要指向正在运行的旧批次目录。

其它环境变量：`MODE=run|precheck|report`（默认run）、`DEVICE=cuda|cpu`、
`CPU_THREADS=2`（每个worker）、`MANO_ROOT`、`HAND_MODEL`、`MANO_PYTHON`。
优先使用仓库 `.venv/bin/python`，否则使用本机hocap-annotation Conda环境；
MediaPipe 使用该 Python；FAH 检测单独使用上文的 FAH 环境。
需要后台运行时可用 `nohup bash scripts/run_batch.sh ... > batch.log 2>&1 &`。

MediaPipe 的 Python 入口也可直接传 `--workers 4`（默认1）；改变并行数不改变阶段来源凭据。
此前版本把结果放在batch_results/work中，本接口不会自动移动旧结果。
迁移应在任务退出后，确认目标hand_annotations不存在，将完整工作目录迁回，
校验图像与预测哈希并更新详细报告中的workdir；不要覆盖已有目录或人工final。
保持文件内容和来源不变的完整迁移不会使阶段凭据失效，续跑仍需通过来源校验。

### 直接使用Python入口

Python 入口保留 MediaPipe 兼容默认。显式传 `--detector fah` 即可使用 FAH，
配置和解释器可用 `--fah-config`、`--fah-python` 指定。
MediaPipe 默认输出到各序列的 hand_annotations；需要隔离实验时，
显式增加 `--work-root /path/to/isolated_work`，不要直接覆盖生产结果：

```bash
PY="$HOME/miniconda3/envs/hocap-annotation/bin/python"
DATA=/path/to/data
BATCH=/path/to/batch_results/run_001
JOBS=/path/to/jobs.json
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# 仅报告；可在完整运行后反复核验，必须传相同的模型/形状/运行设置
$PY tools/batch_rgb_hand_pipeline.py "$DATA" \
  --batch-results "$BATCH" --jobs-json "$JOBS" --mode report --resume \
  --mano-root "$HOME/models/mano" \
  --hand-model config/checkpoints/mediapipe/hand_landmarker.task

# 预检查；没有可验证2D结果时执行检测，不运行MANO拟合
$PY tools/batch_rgb_hand_pipeline.py "$DATA" \
  --batch-results "$BATCH" --jobs-json "$JOBS" --mode precheck --resume

# 完整运行，按凭据复用未变化的已完成阶段
$PY tools/batch_rgb_hand_pipeline.py "$DATA" \
  --batch-results "$BATCH" --jobs-json "$JOBS" --mode run --resume \
  --mano-root "$HOME/models/mano" \
  --hand-model config/checkpoints/mediapipe/hand_landmarker.task
```

默认device为CUDA，检测为CPU/VIDEO；`--device cpu` 可在CPU运行。
检测和拟合使用启动入口的同一Python环境，不自动安装依赖或下载模型。
`--cpu-threads` 默认2。部分本机验证使用 `taskset -c 0-7` 限制CPU亲和性。

`jobs.json` 是明确的序列列表。sequence相对DATA，模型形状/复用目录路径相对
jobs.json所在目录，也可以使用绝对路径。不同人的数据应分别指定形状：

```json
[
  {
    "sequence": "L1-L3-01/Basic_Grip_01",
    "hand_shape": "/path/to/person01_shape",
    "reuse_workdir": "/path/to/old_hand_annotations"
  },
  {
    "sequence": "L1-L3-02/Basic_Joint_Accuracy_01",
    "betas": "/path/to/person02_betas.json",
    "exclude_cameras": ["cam02"]
  }
]
```

每个job可选 `hand_shape`、`betas` 或 `mean_hand: true`，三者互斥，覆盖全局
形状选项；没有job形状配置则使用全局 `--hand-shape` / `--betas` / `--mean-hand`。
不默认为平均手形。单个job的形状错误只让该序列失败，不中止整批。

不使用jobs.json时，可用 `--sequences L1-L3-01/Basic_Grip_01 ...`；
两个选项都不提供时递归查找DATA内的 `rgbd/` 目录，缺失必要文件仍会记录为失败。
建议先显式指定少数序列验证。

`--exclude-cameras cam02 cam05` 是额外的全局手动排除项，与job排除项合并。
`--no-camera-check` 仅关闭自动排除决策，不关闭几何预检查、覆盖门限或逐关节筛选。
合法排除造成视角不足时归入待检查，未知camera ID是配置错误。

## 输出与保护

```text
batch_results/
  summary.csv
  batch_index.json              # 本工具的链接所有权/更新事务索引，请勿删除
  01_normal/<唯一ID> -> 原序列
  02_degraded/<唯一ID> -> 原序列
  03_needs_review/<唯一ID> -> 原序列
  04_failed/<唯一ID> -> 原序列
  details/<唯一ID>/<运行ID>.json # 输入、几何、覆盖、质量、阈值、阶段复用记录
  details/<唯一ID>/<运行ID>.log  # 阶段输出和异常堆栈

<原序列>/hand_annotations/
  batch_state.json              # 阶段来源和输出哈希
  inputs/
  predictions/
  final/                        # 人工保存，由标注软件维护
```

分类目录内本工具只创建软链接，不复制或移动原序列。唯一ID包含规范化原序列
绝对路径的哈希，所以不同批次/根目录的同名序列不冲突。
同一序列只维护一个最终分类链接；旧链接只有在路径、inode及目标都匹配所有权
记录时才删除。用户文件、用户改指向的链接保留。若预定分类路径被用户占用，
记录失败，不覆盖它；连failed链接路径也被占用时仅在CSV记录失败。
链接更新有中断恢复记录。极端中断可能留下根目录下隐藏的暂存链接，不会将其
误当成用户文件删除或归入分类目录。

`summary.csv` 仅有7列：sequence、status、excluded_cameras、total_frames、
right_valid_frames、left_valid_frames、reason。有效帧数来自通过质量检查的拟合，
尚未验证拟合的字段为空；已验证拟合但该手确实无有效结果时才写0。
路径及工作目录详情可通过 `batch_index.json` 和 `details/` 对照。

MediaPipe 默认写原序列下的 `hand_annotations/`；FAH 默认写批量输出根下的 `work/`。
两者都不修改 rgbd 原始输入。
汇总输出根不能与任何原序列目录互相包含（可以放在数据组下，与各序列并列）。
显式指定的 `--work-root` 同样不能与原序列互相包含，工作目录为其下的唯一ID目录。
工作目录中存在 `final/` 时拒绝写入，即使指定replace也不例外；仅报告模式可只读检查。
已有输出目录中的软链接也不会被当作可写预测路径使用。

## 输入与复用

输入检查覆盖每一张RGB，不只检查抽样帧：必要文件、非空可解码图像、尺寸、
六台相机ID/serial、米制刚性标定、主相机单位变换、唯一帧及图像对应、
RGB时间戳有限且递增（含VIDEO毫秒精度）、跨相机时间跨度。
黑色全零RGB也视为无有效图像内容。深度和ego不参与此RGB估计器，不做深度质量评估。
输入异常记为 `failure_kind: input`，与几何不一致/证据不足分开。

原始RGB、标定及帧清单按内容哈希。复用的prepared图像必须与当前原始图像
去畸变后的清单和字节校验一致；输入改变、文件缺失或图像被替换均不能跳过。

默认寻找原序列中与后端对应的 `hand_annotations_fah/` 或 `hand_annotations/` 的 2D 观测，可用 `--reuse-workdir-name`
或job的 `reuse_workdir` 指定。只复用有匹配manifest、camera/frame顺序、有效数组、
检测模型哈希及完整检测配置的观测。默认直接验证本序列hand_annotations中的结果；
显式选择独立work-root时，合法的旧检测可以复制到独立工作目录，旧结果不动。
若缺少可验证的旧检测配置，会记录原因并重新检测；不会猜测其检测参数。
MediaPipe 需要本地 `--hand-model`；FAH 需要本地配置及其引用的资产，
会校验后端、姿态与检测权重指纹、配置和图像清单，不会复用旧的 MediaPipe 观测。

本批量工具完成的阶段在 `batch_state.json` 写入输入/配置/实现版本及输出内容哈希。
`--resume` 只跳过全部匹配的阶段。三角化绑定相机筛选配置和结果，拟合绑定
三角化、观测、betas、MANO模型和拟合参数，导出绑定拟合与三角化。
报告读取使用共享锁，执行使用独占锁，避免读取正在替换的阶段文件。

旧单序列结果没有批量阶段凭据时，只能导入经验证的inputs/2D，不能直接把旧
三角化/拟合视为当前批量配置下的已完成阶段。仅报告模式会明确标为待检查。
中断后留下的未验证/部分输出需要明确 `--replace` 才能重算；默认拒绝覆盖。
已有导出的reconstruction元数据仍遵循不可原地改变的约定，改变筛选策略时
请使用新的 `--work-root`（也建议新的batch_results），不要删除旧元数据绕过校验。

## 几何与最终质量

默认值集中在 `hocap_annotation/batch_quality.py::QualityPolicy`，可用
`--quality-config /path/to/quality.json` 覆盖部分字段；未知字段、无效数值会报错。
每份详细报告记录全部有效阈值。例：

```json
{
  "sample_frames": 48,
  "max_sample_frames": 192,
  "min_precheck_coverage": 0.8,
  "min_reference_coverage": 0.5,
  "min_final_coverage": 0.9
}
```

- 初始均匀抽48帧，无法确定可用性时递增至96、192（不超过序列长度）。
  达到上限仍无可靠参考或覆盖不足，标待检查，不拟合。
- 相机检查采用留一视角验证、正深度、射线夹角、重投影一致性和时间分布。
  持续异常默认需至少12个有效样本，覆盖4个时间区间中的至少3个，每区至少3帧，
  总体及每个充分采样区间异常比例至少80%。错误门限为max(30 px, 3倍三角化门限)。
- 自动判定为可信的参考相机至少3台，但**数量足够不是放行条件**：必须实际
  重建腕点及至少12个关节，射线夹角默认至少1度、重投影内点误差小于10 px。
  每只出现的手，抽样可见帧中至少50%具有三视角关节/腕点支持作为参考证据，
  至少80%能由不少于2个可信视角完成重建。三视角参考证据与实际重建覆盖分开统计。
- 手是否出现按左右手分别判断。完全无观测的手不视为拟合失败；有观测但缺少
  多视角支持是“不确定”或覆盖不足，不能当作没出现。已经确认出现的手，其
  单视角可见但重建不了的帧也计入覆盖率分母。
- 最终默认要求每只出现的手在至少90%的可见帧中有效，且每帧内点重投影中位
  误差不超过25 px、3D关节平均误差不超过5 cm。成功退出但质量不足仍归待检查。
- 这些是保守初筛，不是准确率保证。即使没有排除相机，也可能因弱几何或遮挡
  归待检查；排除相机较多也不直接等同于失败。多人、左右手错标或共同标定偏差
  仍可能需要人工检查，系统不会自动修正外参。

通过最终检查且未排除相机为normal；通过且排除部分相机为degraded；
几何不确定/覆盖不足/最终质量不足为needs_review；输入错误或运行异常为failed。

`predictions/metadata.json` 保留原字段，新增/保留reconstruction记录允许参与的
camera IDs、排除原因、诊断文件及哈希。实际逐帧逐关节内点仍在 `joints_3d.npz`。
全部相机图像/标定保留，被排除视角仍可显示重投影。本轮没有修改标注软件。

## 本地真实数据验证

2026-11-14 在独立目录 `/home/user/hocap_batch_validation_20261114/results`
验证，未远端批量重跑，未覆盖原序列结果。使用可验证的已有2D检测及各序列
固定betas，执行100步三角化与每手10000步MANO拟合：

| 序列 | 最终分类 | 排除相机 | 总帧 | 右手有效 | 左手有效 |
| --- | --- | --- | --- | --- | --- |
| Basic_Grip_01 | 02_degraded | cam02 | 356 | 356 | 356 |
| 3D_Trajectory_Continuity_01 | 01_normal | 无 | 259 | 259 | 259 |

额外对 `3D_Trajectory_Continuity_04` 检查几何，预检查通过且未排除相机，
未据此宣称最终拟合合格。默认覆盖门限结合上述异常及对照数据确定，并非
针对某个序列ID写死；对更多采集条件仍需扩大验证。

对已完成的两条序列分别重跑report及run --resume，所有predictions文件的
内容哈希和修改时间保持不变。质量通过仅表示满足已记录的数值门限，
不是人工标注精度的保证，也不能排除所有相机共同偏移等系统性误差。
