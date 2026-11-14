"""Inspect, classify and optionally reconstruct independent six-camera sequences."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hocap_annotation.batch_pipeline import BatchOptions, run_batch
from hocap_annotation.batch_quality import QualityPolicy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data_root", type=Path)
    p.add_argument("--batch-results", type=Path, required=True)
    p.add_argument("--mode", choices=["report", "precheck", "run"], default="report")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument(
        "--sequences",
        nargs="+",
        help="Sequence paths relative to data_root; default: discover rgbd directories",
    )
    selection.add_argument(
        "--jobs-json",
        type=Path,
        help="List of objects with sequence and optional per-sequence shape/exclusion/reuse settings",
    )
    p.add_argument(
        "--work-root",
        type=Path,
        help="Independent workdirs; default: batch_results/work/<unique-id>",
    )
    p.add_argument("--reuse-workdir-name", default="hand_annotations")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--replace", action="store_true")
    p.add_argument(
        "--quality-config",
        type=Path,
        help="JSON object overriding centralized quality thresholds",
    )
    p.add_argument(
        "--hand-model",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "config/checkpoints/mediapipe/hand_landmarker.task",
    )
    p.add_argument("--mano-root", type=Path, default=Path.home() / "models/mano")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--detector-device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--detector-mode", choices=["image", "video"], default="video")
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--triangulation-steps", type=int, default=100)
    p.add_argument("--fit-steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=0.001)
    shape = p.add_mutually_exclusive_group()
    shape.add_argument("--mean-hand", action="store_true")
    shape.add_argument("--betas", type=Path)
    shape.add_argument("--hand-shape", type=Path)
    p.add_argument("--exclude-cameras", nargs="+", default=[])
    p.add_argument(
        "--no-camera-check",
        action="store_true",
        help="Do not automatically exclude cameras; still check reference geometry and quality",
    )
    a = p.parse_args()
    try:
        if a.cpu_threads < 1:
            raise ValueError("cpu-threads must be positive")
        root = a.data_root.resolve()
        if a.jobs_json:
            jobs = json.loads(a.jobs_json.read_text())
            if not isinstance(jobs, list) or not all(
                isinstance(j, dict) and isinstance(j.get("sequence"), str) for j in jobs
            ):
                raise ValueError("jobs-json must contain a list of sequence objects")
            allowed = {
                "sequence",
                "betas",
                "hand_shape",
                "mean_hand",
                "exclude_cameras",
                "reuse_workdir",
            }
            for job in jobs:
                if set(job) - allowed:
                    raise ValueError(f"Unknown job fields: {set(job) - allowed}")
                for key in ("betas", "hand_shape", "reuse_workdir"):
                    if job.get(key):
                        job[key] = str(
                            (a.jobs_json.resolve().parent / job[key]).resolve()
                        )
        elif a.sequences:
            jobs = [dict(sequence=s) for s in a.sequences]
        else:
            jobs = [
                dict(sequence=str(path.parent.relative_to(root)))
                for path in sorted(root.rglob("rgbd"))
                if path.is_dir()
            ]
        if not jobs:
            raise ValueError("No sequences found")
        policy = QualityPolicy(
            **(json.loads(a.quality_config.read_text()) if a.quality_config else {})
        )
        options = BatchOptions(
            mode=a.mode,
            resume=a.resume,
            replace=a.replace,
            work_root=a.work_root,
            reuse_workdir_name=a.reuse_workdir_name,
            hand_model=a.hand_model,
            mano_root=a.mano_root,
            detector_mode=a.detector_mode,
            detector_device=a.detector_device,
            device=a.device,
            triangulation_steps=a.triangulation_steps,
            fit_steps=a.fit_steps,
            lr=a.lr,
            mean_hand=a.mean_hand,
            betas=a.betas,
            hand_shape=a.hand_shape,
            exclude_cameras=tuple(a.exclude_cameras),
            automatic=not a.no_camera_check,
            policy=policy,
        )
    except (ValueError, TypeError, OSError) as exc:
        p.error(str(exc))
    import torch

    torch.set_num_threads(a.cpu_threads)
    rows = run_batch(root, a.batch_results, jobs, options)
    print(f"Summary: {a.batch_results / 'summary.csv'}", flush=True)
    return 1 if any(r["status"] == "04_failed" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
