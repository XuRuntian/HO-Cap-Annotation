"""Failure-isolated batch orchestration around the existing RGB estimator."""

import contextlib
import importlib.metadata
import multiprocessing
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from multiprocessing.connection import wait
from pathlib import Path

import cv2
import numpy as np

from . import rgb_hand_pipeline as pipeline
from .batch_cache import StageCache, StaleStage
from .batch_inputs import InputError, inspect_inputs, verify_prepared
from .batch_quality import QualityPolicy, final_quality, precheck
from .batch_results import BatchResults, sequence_id
from .hand_shape import load_hand_shape
from .rgbd_data import (
    PreparedSequence,
    atomic_json,
    atomic_npz,
    digest,
    prepare,
    read_json,
)

OUTPUTS = {
    "prepare": ["inputs/manifest.json"],
    "detect": ["predictions/observations_2d.npz"],
    "triangulate": ["predictions/joints_3d.npz", "predictions/camera_consistency.json"],
    "fit": ["predictions/fit_parameters.npz", "predictions/fit_diagnostics.npz"],
    "export": ["predictions/mano.npz", "predictions/metadata.json"],
}


@dataclass
class BatchOptions:
    mode: str = "report"
    resume: bool = False
    replace: bool = False
    work_root: Path | None = None
    reuse_workdir_name: str = "hand_annotations"
    hand_model: Path | None = None
    mano_root: Path | None = None
    detector_mode: str = "video"
    detector_device: str = "cpu"
    device: str = "cuda"
    triangulation_steps: int = 100
    fit_steps: int = 10000
    lr: float = 0.001
    mean_hand: bool = False
    betas: Path | None = None
    hand_shape: Path | None = None
    exclude_cameras: tuple = ()
    automatic: bool = True
    policy: QualityPolicy = field(default_factory=QualityPolicy)

    def __post_init__(self):
        if self.mode not in ("report", "precheck", "run"):
            raise ValueError("Invalid batch mode")
        if (
            self.triangulation_steps < 0
            or self.fit_steps <= 0
            or not np.isfinite(self.lr)
            or self.lr <= 0
        ):
            raise ValueError("Invalid optimization settings")
        if self.device not in ("cpu", "cuda") or self.detector_mode not in (
            "image",
            "video",
        ):
            raise ValueError("Invalid device/detector mode")
        if self.detector_device not in ("cpu", "gpu"):
            raise ValueError("Invalid detector device")
        if Path(self.reuse_workdir_name).name != self.reuse_workdir_name:
            raise ValueError("reuse-workdir-name must be a single directory name")


def implementation(stage):
    root = Path(__file__).parent
    names = ["rgbd_data.py"]
    if stage != "prepare":
        names += ["rgb_hand_pipeline.py"]
    if stage == "detect":
        names += ["wrappers/mediapipe.py"]
    if stage == "triangulate":
        names += ["triangulation.py", "camera_consistency.py", "batch_quality.py"]
    if stage == "fit":
        names += [
            str(p.relative_to(root)) for p in sorted((root / "loss").glob("*.py"))
        ]
    versions = dict(numpy=np.__version__, opencv=cv2.__version__)
    for name in ("torch", "manopth", "mediapipe"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return dict(files={name: digest(root / name) for name in names}, versions=versions)


def detection_signature(seq, options):
    if options.hand_model is None or not Path(options.hand_model).is_file():
        raise ValueError(
            "A local --hand-model is required to validate/recompute detections"
        )
    config = pipeline.detection_config(
        Path(options.hand_model).resolve(),
        options.detector_mode,
        options.detector_device,
    )
    config["model_asset_path"] = Path(options.hand_model).name
    config["model_sha256"] = digest(options.hand_model)
    return dict(
        manifest_sha256=seq.fingerprint,
        config=config,
        implementation=implementation("detect"),
    )


def verified_detection(seq, signature):
    obs = seq.load("observations_2d.npz")
    pipeline.check_observations(seq, obs)
    if str(obs.get("model_sha256", "")) != signature["config"]["model_sha256"]:
        raise StaleStage("Detection model fingerprint mismatch")
    config = read_json(seq.workdir / "predictions/run_config.json").get("detect")
    if config != signature["config"]:
        raise StaleStage(
            "Missing or changed detection configuration; cannot adopt old observations"
        )
    return obs


def reusable_detection(workdir, source, inspection, options):
    with StageCache(workdir, source, readonly=True) as original_cache:
        original = PreparedSequence(workdir)
        verify_prepared(original, inspection)
        key = detection_signature(original, options)
        if "detect" in original_cache.state["stages"] and not original_cache.valid(
            "detect", key, OUTPUTS["detect"]
        ):
            raise StaleStage("Reusable detection receipt no longer matches")
        return verified_detection(original, key), original.fingerprint


def shape_parameters(job, options):
    override = any(k in job for k in ("hand_shape", "betas", "mean_hand"))
    shape = job.get("hand_shape") if override else options.hand_shape
    betas = job.get("betas") if override else options.betas
    mean = job.get("mean_hand", False) if override else options.mean_hand
    if sum(bool(v) for v in (shape, betas, mean)) != 1:
        raise ValueError(
            "Choose exactly one hand_shape, betas or mean_hand for this sequence"
        )
    if options.mano_root is None:
        raise ValueError("--mano-root is required to validate/fit MANO")
    models = {
        side: digest(Path(options.mano_root) / f"MANO_{side.upper()}.pkl")
        for side in ("right", "left")
    }
    if shape:
        beta = load_hand_shape(shape, options.mano_root)
    elif betas:
        beta = read_json(betas)
    else:
        beta = np.zeros((2, 10)).tolist()
    beta = np.asarray(beta, dtype=np.float32)
    if beta.shape != (2, 10) or not np.isfinite(beta).all():
        raise ValueError("Invalid fixed hand shape")
    return beta, models


def bind_report(report, seq):
    report = dict(report)
    report.update(
        manifest_sha256=seq.fingerprint,
        observations_sha256=digest(seq.workdir / "predictions/observations_2d.npz"),
        sampled_frame_ids=[seq.frame_ids[i] for i in report["sampled_frame_indices"]],
    )
    return report


def _process(source, job, work, options, detail):
    inspection = inspect_inputs(source, options.policy)
    detail["input"] = {k: v for k, v in inspection.items() if k != "expected_manifest"}
    detail["total_frames"] = inspection["total_frames"]
    readonly = options.mode == "report"
    reuse = Path(
        job.get("reuse_workdir", source / options.reuse_workdir_name)
    ).resolve()
    if readonly and (
        not (work / "inputs/manifest.json").is_file()
        or not (work / "predictions/observations_2d.npz").is_file()
    ):
        work = reuse
    detail["workdir"] = str(work)
    if readonly and not (work / "inputs/manifest.json").is_file():
        return "03_needs_review", "no verified prepared inputs; run precheck mode"
    with StageCache(work, source, readonly=readonly) as cache:
        prepare_key = dict(
            source_sha256=inspection["source_sha256"],
            implementation=implementation("prepare"),
        )
        if (work / "inputs/manifest.json").exists():
            try:
                seq = PreparedSequence(work)
                verify_prepared(seq, inspection)
            except (OSError, ValueError, KeyError):
                if readonly or not options.replace:
                    raise
                cache.run(
                    "prepare",
                    prepare_key,
                    OUTPUTS["prepare"],
                    lambda: prepare(source, work, replace=True),
                    replace=True,
                )
                seq = PreparedSequence(work)
                verify_prepared(seq, inspection)
            if not readonly:
                cache.record("prepare", prepare_key, OUTPUTS["prepare"])
        else:
            cache.run(
                "prepare",
                prepare_key,
                OUTPUTS["prepare"],
                lambda: prepare(source, work),
                resume=options.resume,
                replace=options.replace,
            )
            seq = PreparedSequence(work)
            verify_prepared(seq, inspection)
        detect_key = detection_signature(seq, options)
        detection_path = work / "predictions/observations_2d.npz"
        detected = False
        if cache.valid("detect", detect_key, OUTPUTS["detect"]):
            obs = seq.load("observations_2d.npz")
            pipeline.check_observations(seq, obs)
            detail["stages"]["detect"] = "reused_verified_receipt"
            detected = True
        elif detection_path.exists():
            # Legacy adoption requires its own detector config/model and current raw manifest.
            try:
                obs = verified_detection(seq, detect_key)
                if "detect" in cache.state["stages"]:
                    raise StaleStage("Existing detection receipt no longer matches")
                if not readonly:
                    cache.record("detect", detect_key, OUTPUTS["detect"])
                detail["stages"]["detect"] = "adopted_verified_observations"
                detected = True
            except (OSError, ValueError, KeyError):
                if readonly or not options.replace:
                    raise StaleStage(
                        "Unverified existing 2D detections; use new work root or --replace"
                    )
        elif reuse != work and (reuse / "predictions/observations_2d.npz").is_file():
            try:
                obs, original_fingerprint = reusable_detection(
                    reuse, source, inspection, options
                )
                if original_fingerprint != seq.fingerprint:
                    raise StaleStage("Reusable detection manifest differs")
                if not readonly:
                    (work / "predictions").mkdir(exist_ok=True)
                    atomic_npz(detection_path, **obs)
                    pipeline.check_observations(seq, seq.load("observations_2d.npz"))
                    pipeline.log_config(seq, "detect", detect_key["config"])
                    cache.record("detect", detect_key, OUTPUTS["detect"])
                detail["stages"]["detect"] = "copied_verified_observations"
                detected = True
            except (OSError, ValueError, KeyError) as exc:
                detail["reuse_warning"] = str(exc)
        if not detected:
            if readonly:
                return (
                    "03_needs_review",
                    "no verified 2D observations; run precheck mode",
                )
            detail["stages"]["detect"] = cache.run(
                "detect",
                detect_key,
                OUTPUTS["detect"],
                lambda: pipeline.detect(
                    seq,
                    options.hand_model,
                    options.detector_mode,
                    options.detector_device,
                    replace=options.replace,
                ),
                resume=options.resume,
                replace=options.replace,
            )
            obs = seq.load("observations_2d.npz")
            pipeline.check_observations(seq, obs)
        excluded = sorted(
            set(options.exclude_cameras) | set(job.get("exclude_cameras", []))
        )
        report = precheck(
            obs["keypoints"],
            obs["valid"],
            seq.matrices,
            seq.camera_ids,
            options.policy,
            automatic=options.automatic,
            excluded=excluded,
        )
        report = bind_report(report, seq)
        detail["geometry"] = report
        detail["excluded_cameras"] = [
            c["camera_id"] for c in report["excluded_cameras"]
        ]
        if not report["usable"]:
            return "03_needs_review", "; ".join(report["reasons"])
        if options.mode == "precheck":
            return "03_needs_review", "precheck passed; final fit not verified"
        p = options.policy
        tri_key = dict(
            source_sha256=inspection["source_sha256"],
            selection=report,
            steps=options.triangulation_steps,
            device=options.device,
            min_views=p.min_views,
            implementation=implementation("triangulate"),
        )
        if readonly:
            if not cache.valid("triangulate", tri_key, OUTPUTS["triangulate"]):
                return (
                    "03_needs_review",
                    "precheck passed; triangulation provenance not verified",
                )
        else:
            previous = work / "predictions/metadata.json"
            if previous.exists():
                old_report = work / "predictions/camera_consistency.json"
                if not old_report.is_file() or read_json(old_report) != report:
                    raise StaleStage(
                        "Exported selection metadata is immutable; choose a new work root"
                    )
            detail["stages"]["triangulate"] = cache.run(
                "triangulate",
                tri_key,
                OUTPUTS["triangulate"],
                lambda: pipeline.triangulate(
                    seq,
                    device=options.device,
                    steps=options.triangulation_steps,
                    threshold=p.reprojection_px,
                    min_views=p.min_views,
                    min_angle=p.min_angle_deg,
                    selection_report=report,
                    replace=options.replace,
                ),
                resume=options.resume,
                replace=options.replace,
            )
        tri = seq.load("joints_3d.npz")
        pipeline.reconstruction_metadata(seq, tri)
        if str(tri["observations_sha256"]) != digest(detection_path):
            raise StaleStage("Triangulation observations changed")
        beta, models = shape_parameters(job, options)
        fit_key = dict(
            triangulation_sha256=digest(work / "predictions/joints_3d.npz"),
            observations_sha256=digest(detection_path),
            betas=beta.tolist(),
            models=models,
            steps=options.fit_steps,
            lr=options.lr,
            device=options.device,
            min_joints=p.min_joints,
            max_error_px=p.fit_reprojection_px,
            max_joint_error_m=p.fit_joint_error_m,
            implementation=implementation("fit"),
        )
        if readonly:
            if not cache.valid("fit", fit_key, OUTPUTS["fit"]):
                return "03_needs_review", "final fit provenance not verified"
        else:
            detail["stages"]["fit"] = cache.run(
                "fit",
                fit_key,
                OUTPUTS["fit"],
                lambda: pipeline.fit(
                    seq,
                    options.mano_root,
                    device=options.device,
                    steps=options.fit_steps,
                    lr=options.lr,
                    betas=beta,
                    min_joints=p.min_joints,
                    max_error_px=p.fit_reprojection_px,
                    max_joint_error_m=p.fit_joint_error_m,
                    replace=options.replace,
                ),
                resume=options.resume,
                replace=options.replace,
            )
        export_key = dict(
            fit_sha256=digest(work / "predictions/fit_parameters.npz"),
            triangulation_sha256=fit_key["triangulation_sha256"],
            implementation=implementation("export"),
        )
        if readonly:
            if not cache.valid("export", export_key, OUTPUTS["export"]):
                return "03_needs_review", "export provenance not verified"
        else:
            detail["stages"]["export"] = cache.run(
                "export",
                export_key,
                OUTPUTS["export"],
                lambda: pipeline.export(seq, replace=options.replace),
                resume=options.resume,
                replace=options.replace,
            )
        diagnostics = seq.load("fit_diagnostics.npz")
        result = final_quality(obs["valid"], diagnostics, p)
        detail["quality"] = result
        detail["valid_counts"] = [h["valid_frames"] for h in result["hands"]]
        if not result["passed"]:
            return "03_needs_review", "; ".join(result["reasons"])
        return (
            "02_degraded" if report["excluded_cameras"] else "01_normal"
        ), "quality passed"


def _process_job(source, job, workdir, options, report_path, detail):
    try:
        log = report_path.with_suffix(".log")
        detail["log_file"] = log.name
        with (
            log.open("x") as stream,
            contextlib.redirect_stdout(stream),
            contextlib.redirect_stderr(stream),
        ):
            try:
                status, reason = _process(source, job, workdir, options, detail)
            except Exception:
                traceback.print_exc()
                raise
    except Exception as exc:
        status, reason = "04_failed", f"{type(exc).__name__}: {exc}"
        detail["failure_kind"] = "input" if isinstance(exc, InputError) else "runtime"
        detail["exception"] = reason
    return source, report_path, detail, status, reason


def _worker_entry(sender, args, cpu_threads):
    import torch

    torch.set_num_threads(cpu_threads)
    try:
        sender.send(_process_job(*args))
    finally:
        sender.close()


def _job_results(results, resolved, work_root, options, workers, cpu_threads):
    # Only the parent owns classification/index writes. Spawn avoids inherited CUDA state.
    context = multiprocessing.get_context("spawn")
    pending = {}
    queue = iter(resolved)
    exhausted = False
    try:
        while pending or not exhausted:
            while len(pending) < workers and not exhausted:
                try:
                    source, job = next(queue)
                except StopIteration:
                    exhausted = True
                    break
                detail = dict(
                    sequence=str(source),
                    policy=asdict(options.policy),
                    mode=options.mode,
                    stages={},
                )
                try:
                    report_path = results.details(source) / f"{uuid.uuid4().hex}.json"
                    args = (
                        source,
                        job,
                        (work_root / sequence_id(source))
                        if work_root is not None
                        else source / "hand_annotations",
                        options,
                        report_path,
                        detail,
                    )
                    if workers == 1:
                        yield _process_job(*args)
                        continue
                    receiver, sender = context.Pipe(duplex=False)
                    process = context.Process(
                        target=_worker_entry, args=(sender, args, cpu_threads)
                    )
                    try:
                        process.start()
                    except BaseException:
                        receiver.close()
                        raise
                    finally:
                        sender.close()
                    pending[receiver] = (process, source, report_path, detail)
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    detail.update(failure_kind="runtime", exception=reason)
                    yield source, None, detail, "04_failed", reason
            if not pending:
                continue
            for receiver in wait(pending):
                process, source, report_path, detail = pending[receiver]
                try:
                    outcome = receiver.recv()
                except (EOFError, OSError):
                    process.join()
                    reason = f"Worker exited without a result (exit={process.exitcode})"
                    detail.update(failure_kind="runtime", exception=reason)
                    outcome = (source, report_path, detail, "04_failed", reason)
                process.join()
                receiver.close()
                del pending[receiver]
                yield outcome
    finally:
        # Each sequence has its own process so a native crash cannot break other workers.
        for receiver, (process, *_rest) in pending.items():
            if process.is_alive():
                process.terminate()
        for receiver, (process, *_rest) in pending.items():
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
            receiver.close()


def run_batch(data_root, batch_results, jobs, options, *, workers=1, cpu_threads=2):
    if not isinstance(workers, int) or not 1 <= workers <= 32 or cpu_threads < 1:
        raise ValueError("workers must be 1..32 and cpu_threads must be positive")
    data_root = Path(data_root).resolve()
    output = Path(batch_results).absolute()
    work_root = (
        Path(options.work_root).absolute() if options.work_root is not None else None
    )
    resolved = []
    for job in jobs:
        source = (data_root / job["sequence"]).resolve()
        if not source.is_relative_to(data_root):
            raise ValueError("Sequence outside data root")
        for dest in (output, *([work_root] if work_root is not None else [])):
            if dest.resolve().is_relative_to(source) or source.is_relative_to(
                dest.resolve()
            ):
                raise ValueError(
                    "Batch output/work root must not overlap a source sequence"
                )
        resolved.append((source, job))
    if len({s for s, _ in resolved}) != len(resolved):
        raise ValueError("Duplicate sequences in batch")
    rows = []
    with BatchResults(output) as results:
        for source, report_path, detail, status, reason in _job_results(
            results, resolved, work_root, options, workers, cpu_threads
        ):
            counts = detail.get("valid_counts", ["", ""])
            row = dict(
                sequence=source.relative_to(data_root).as_posix(),
                status=status,
                excluded_cameras=";".join(detail.get("excluded_cameras", [])),
                total_frames=detail.get("total_frames", ""),
                right_valid_frames=counts[0],
                left_valid_frames=counts[1],
                reason=reason[:200].replace("\n", " "),
            )
            detail["summary"] = row
            try:
                results.update(source, row)
            except (OSError, ValueError) as exc:
                row.update(
                    status="04_failed",
                    reason=f"classification link conflict: {exc}"[:200],
                )
                detail["classification_error"] = str(exc)
                try:
                    results.update(source, row)
                except (OSError, ValueError):
                    results.failed_without_link(source, row)
            if report_path is not None:
                try:
                    atomic_json(report_path, detail)
                except (OSError, ValueError) as exc:
                    row.update(
                        status="04_failed",
                        reason=f"detail report write failed: {exc}"[:200],
                    )
                    try:
                        results.update(source, row)
                    except (OSError, ValueError):
                        results.failed_without_link(source, row)
            rows.append(row)
            print(f"{row['sequence']}: {row['status']} - {row['reason']}", flush=True)
    return rows
