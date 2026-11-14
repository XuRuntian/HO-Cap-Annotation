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

### 兼容旧批量接口

新版 `scripts/run_batch.sh` 保留“数据组目录 + 手形目录”两个位置参数，
只选择该组直接子目录中的序列，不递归运行其它数据组，也不推断人员手形：

```bash
JOBS=4 bash scripts/run_batch.sh \
  /home/adminpc/hand-real-workspace/data/L1-L3-01 \
  /home/adminpc/hand-real-workspace/hand_shapes/01_standard
```

`JOBS` 默认2，范围1至32，表示同时处理的序列数，不是同时处理的数据组数。
多进程使用spawn，各序列独立工作；父进程统一维护一个summary.csv及四个分类目录。
CUDA设备由所有worker共享，4并行不保证显存足够，OOM时减少JOBS。
JOBS大于1时每个序列独立进程，普通异常或单个worker崩溃会记录失败，
其它序列继续处理。JOBS=1保留原串行实现，隔离Python异常，但原生崩溃会
终止该批次。机器级资源耗尽仍可能影响多个任务。不要同时启动两个任务写同一个结果目录。

默认结果位置为 `数据组的父目录/batch_results/组名-路径哈希/`，不写入原序列。
可以通过 `BATCH_RESULTS=/absolute/path` 固定输出目录。默认总是启用resume，
不会像旧脚本那样看到hand_annotations存在就跳过；`REPLACE=1` 才允许重算已有
未验证阶段（仍受final及导出元数据保护约束）。不要指向正在运行的旧批次目录。

其它环境变量：`MODE=run|precheck|report`（默认run）、`DEVICE=cuda|cpu`、
`CPU_THREADS=2`（每个worker）、`MANO_ROOT`、`HAND_MODEL`、`MANO_PYTHON`。
优先使用仓库 `.venv/bin/python`，否则使用本机hocap-annotation Conda环境；
检测和拟合使用同一Python，不再使用旧的DETECT_PYTHON分环境约定。
需要后台运行时可用 `nohup env JOBS=4 bash scripts/run_batch.sh ... > batch.log 2>&1 &`。

Python入口也可直接传 `--workers 4`（默认1）；改变并行数不改变阶段来源凭据。
此前临时四并行命令的独立结果布局不同，本接口不会自动导入它的三角化/拟合凭据。

### 直接使用Python入口

先使用小规模、独立目录，不直接在生产批次重跑：

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
  work/<唯一ID>/                # 默认独立工作目录，可用 --work-root 更换
    batch_state.json
    inputs/
    predictions/
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

工作目录默认在batch_results下，不写回原序列。输出根不能与任何原序列目录
互相包含；工作目录中存在 `final/` 时拒绝写入。原序列已有的 `final/` 完全不动。
已有输出目录中的软链接也不会被当作可写预测路径使用。

## 输入与复用

输入检查覆盖每一张RGB，不只检查抽样帧：必要文件、非空可解码图像、尺寸、
六台相机ID/serial、米制刚性标定、主相机单位变换、唯一帧及图像对应、
RGB时间戳有限且递增（含VIDEO毫秒精度）、跨相机时间跨度。
黑色全零RGB也视为无有效图像内容。深度和ego不参与此RGB估计器，不做深度质量评估。
输入异常记为 `failure_kind: input`，与几何不一致/证据不足分开。

原始RGB、标定及帧清单按内容哈希。复用的prepared图像必须与当前原始图像
去畸变后的清单和字节校验一致；输入改变、文件缺失或图像被替换均不能跳过。

默认寻找原序列 `hand_annotations/` 的2D观测，可用 `--reuse-workdir-name`
或job的 `reuse_workdir` 指定。只复用有匹配manifest、camera/frame顺序、有效数组、
检测模型哈希及完整检测配置的观测。合法复用会写到独立工作目录，旧结果不动。
若缺少可验证的旧检测配置，会记录原因并重新检测；不会猜测其检测参数。
缺少本地检测模型时不能验证模型身份，应提供 `--hand-model`。

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
