"""RGB-D dataset -> independent rectification -> RGB-only MANO predictions."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hocap_annotation.rgbd_data import prepare, PreparedSequence
from hocap_annotation.hand_shape import load_hand_shape


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "data_root", type=Path, help="Dataset containing rgbd/; ego/ is ignored"
    )
    p.add_argument(
        "--stage",
        choices=["all", "prepare", "detect", "triangulate", "fit", "export"],
        default="all",
    )
    p.add_argument(
        "--workdir",
        type=Path,
        help="Default: DATA_ROOT/hand_annotations; final/ is never written",
    )
    p.add_argument("--mano-root", type=Path)
    p.add_argument("--hand-model", type=Path, help="MediaPipe hand_landmarker.task")
    p.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Triangulation and fitting device",
    )
    p.add_argument("--detector-device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--detector-mode", choices=["image", "video"], default="video")
    p.add_argument(
        "--cpu-threads",
        type=int,
        default=2,
        help="Torch threads for CPU triangulation/fitting",
    )
    p.add_argument(
        "--limit",
        type=int,
        help="Limit prepare to first N synchronized frames (for testing)",
    )
    p.add_argument("--triangulation-steps", type=int, default=100)
    p.add_argument("--min-views", type=int, default=2)
    p.add_argument("--reprojection-px", type=float, default=10.0)
    p.add_argument("--fit-steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=0.001)
    shape = p.add_mutually_exclusive_group()
    shape.add_argument(
        "--mean-hand",
        action="store_true",
        help="Explicitly use zero betas; no individual shape estimation",
    )
    shape.add_argument(
        "--betas", type=Path, help="JSON [2,10] shared shape: right,left"
    )
    shape.add_argument(
        "--hand-shape", type=Path,
        help="EasyMocap output directory or fixed betas.json/metadata.json directory",
    )
    p.add_argument(
        "--replace",
        action="store_true",
        help="Explicitly recompute existing stage outputs; never final/",
    )
    a = p.parse_args()
    workdir = (a.workdir or a.data_root / "hand_annotations").resolve()
    if a.stage in ("all", "detect") and not a.hand_model:
        p.error("--hand-model is required for detection")
    if a.stage in ("all", "fit"):
        if not a.mano_root:
            p.error("--mano-root is required for fitting")
        if not a.mean_hand and not a.betas and not a.hand_shape:
            p.error("Choose --mean-hand, --betas or --hand-shape explicitly")
    betas = None
    if a.hand_shape:
        try:
            betas = load_hand_shape(a.hand_shape, a.mano_root)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            p.error(f"Invalid --hand-shape: {exc}")
    elif a.betas and a.stage in ("all", "fit"):
        betas = json.loads(a.betas.read_text())
    if a.cpu_threads < 1:
        p.error("--cpu-threads must be positive")
    stages = (
        ["prepare", "detect", "triangulate", "fit", "export"]
        if a.stage == "all"
        else [a.stage]
    )
    for stage in stages:
        print(f"=== {stage} ===", flush=True)
        if stage == "prepare":
            prepare(a.data_root, workdir, a.limit, a.replace)
            continue
        from hocap_annotation import rgb_hand_pipeline as pipeline

        seq = PreparedSequence(workdir)
        if stage in ("triangulate", "fit") and a.device == "cpu":
            import torch

            torch.set_num_threads(a.cpu_threads)
        if stage == "detect":
            pipeline.detect(
                seq, a.hand_model, a.detector_mode, a.detector_device, a.replace
            )
        elif stage == "triangulate":
            pipeline.triangulate(
                seq,
                a.device,
                a.triangulation_steps,
                a.reprojection_px,
                a.min_views,
                a.replace,
            )
        elif stage == "fit":
            pipeline.fit(
                seq,
                a.mano_root,
                a.device,
                a.fit_steps,
                a.lr,
                betas,
                replace=a.replace,
            )
        else:
            pipeline.export(seq, a.replace)


if __name__ == "__main__":
    main()
