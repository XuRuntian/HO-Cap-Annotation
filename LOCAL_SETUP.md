# Local Six-Camera Setup

## Environment

Use `conda activate hocap-annotation`, then work from this repository root.
Acquisition and multical environments are not modified.

Validated local stack:

- Python 3.10.22, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121.
- NumPy 1.26.4, SciPy 1.14.1, JAX/JAXLIB 0.4.35.
- OpenCV/contrib 4.10.0.84, MediaPipe 0.10.14.
- chumpy 0.70 with the NumPy scalar-import compatibility patch; reapply after
  reinstalling with `python scripts/patch_chumpy_numpy.py`.
- manopth from `gobanana520/manopth`, commit
  `f66a5020d40afbf0b3b5b7ebedd3154aa859cf30` (the repository's recommended fork).
- `meshsdf_loss_cuda` built for RTX 4070 (SM 8.9) with CUDA 12.2 and GCC/G++ 11.

The repository's recommended chumpy fork was unavailable; PyPI chumpy 0.70 plus
the small explicit patch above is used instead. Full installed package inventory
is recorded locally in `logs/environment.freeze.txt`.

PyTorch cu121 wheels are listed in the [official previous-version instructions](https://pytorch.org/get-started/previous-versions/).
This host has CUDA 12.2 and a driver supporting CUDA 12.4. The cu121 build avoids
the CUDA-11-vs-12 compiler mismatch of the original README's cu118 command.
For uv use `--default-index https://download.pytorch.org/whl/cu121` explicitly:
this machine's `UV_DEFAULT_INDEX` takes precedence over the deprecated
`--index-url` option.

Rebuild the depth extension without modifying the system compiler:

```bash
CUDA_HOME=/usr/local/cuda-12.2 CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 \
MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 \
python -m pip install --no-build-isolation --no-deps ./hocap_annotation/loss/meshsdf_loss
```

## Configuration

`config/config.yaml` points MANO to `/home/user/models/mano` and optional SAM2 to
`/home/user/models/sam2.1/sam2.1_hiera_large.pt`. MediaPipe's official version-1
task model is downloaded under `config/checkpoints/mediapipe/hand_landmarker.task`.
MediaPipe uses its CPU delegate, while PyTorch reconstruction/fitting uses CUDA.

Six device serials and `valid_rs_serials` match the prepared dataset. Master is
`338522300613`. `no_hololens` is a loader-only placeholder; no HoloLens estimation
should be run. `max_workers: 4` limits image loading and detection workers.

`pointcloud.crop_limits: null` disables the original table-aligned spatial crop.
Invalid/nonpositive depth is still rejected. The prepared sequence's world frame
is the master RGB camera frame. To restrict the workspace later, supply
`[xmin, xmax, ymin, ymax, zmin, zmax]` in world-frame meters; bounds must be finite
and ordered. This is not a measured table frame or an automatic hand mask.

## Running

```bash
conda activate hocap-annotation
cd /home/user/HO-Cap-Annotation
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SEQ=datasets/realsense_hand_test/sequences/hand_test

taskset -c 0-7 python tools/02_mp_hand_detection.py --sequence_folder "$SEQ" --skip-vis
taskset -c 0-7 python tools/03_mp_3d_joints_generation.py --sequence_folder "$SEQ"
taskset -c 0-7 python tools/05_mano_pose_solver.py --sequence_folder "$SEQ"
```

The original tool 02 had its detection call commented out. It now detects before
rendering, honors the MediaPipe-specific device setting, and uses spawned workers
for detection. `--skip-vis` skips image/video rendering, not detection.

Validation performed: GPU tensor operations; dataset/MANO loading; MANO forward
pass; CUDA mesh-distance forward/backward; 2D detection over all 449 frames and
six cameras; sampled real-frame 3D reconstruction. This is an environment smoke
test, not a claim that the full sequence's 3D/MANO results or ground truth have
been validated. Review cross-view overlays, visibility and fused geometry before
using results as labels. Full MANO fitting is not run as part of installation.

## Missing-Observation Fix

The previous MANO path used missing `[-1, -1, -1]` joints as both translations
and 3D loss targets. The fix keeps original observations and initialization
seeds separate. It also removes the unconditional overwrite of frame 1 with
frame 0. No MINT/EgoHandKit code or dependencies are imported: their explicit
validity and missing-hand boundary rules are used as design references.

- `max_init_gap_frames: 5` controls initialization only; set 0 to disable filling.
  A short run requires hand detections in at least two views on every gap frame.
  Interior gaps use linear wrist interpolation; short supported edge gaps use
  the nearest observed wrist. This does not create 3D loss observations.
- Long runs, absent hands, and hands with no valid wrist anchor remain invalid.
  These are deliberately not globally filled from the cubic-interpolated file.
- Missing/NaN/Inf 3D targets are masked before residual computation. Invalid
  frames do not enter 2D fitting, depth mesh association, or temporal smoothing.
- Temporal penalties cannot span an invalid frame, including with wider windows.
  Translation penalties now correctly use the last three pose coordinates, not
  all coordinates after global rotation. No new post-fit smoother is added.
- Unsupported output poses stay at -1 and are omitted from rendering.
  `poses_m_valid.npy` is a canonical right/left eligibility mask, not a quality
  guarantee. `fit_observations.npz` records selected hand order, original joint
  validity, fit eligibility, and initialization source (0 invalid, 1 observed,
  2 linear interior fill, 3 edge hold). Filled frames remain estimated results.

Keep the old results for comparison by using a new output folder:

```bash
taskset -c 0-7 python tools/05_mano_pose_solver.py \
  --sequence_folder "$SEQ" \
  --output_folder "$SEQ/processed/hand_pose_solver_missing_fixed"
```

An explicitly supplied output folder must not exist. The default output path
retains its historical overwrite behavior. Existing 2D/3D input files are not
modified and do not need to be recomputed for this fix.

Regression tests are in `test/test_mano_missing_observations.py`, including the
actual initialization/loss/export/renderer call sites. At the user's request,
these new tests and a full refit have **not been run by the agent**:

```bash
PYOPENGL_PLATFORM=egl taskset -c 0-7 python -m unittest discover -s test -v
```
