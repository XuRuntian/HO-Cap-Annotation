"""Calibration worker: collect FAH samples, then write a reusable MANO bundle."""

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hocap_annotation.rgbd_data import atomic_json, atomic_write, digest, read_json, rectification_maps

SIDES = ("right", "left")
LENGTH_TEMPLATE = Path(__file__).resolve().parents[1] / "config/hand_bone_lengths.template.yaml"


def collect(clip, config, output, count):
    import cv2
    import fah
    from fah.api import method_args
    from fah.shape_calibration import NativeShapeFitter
    import pickle
    import torch

    clip = clip.resolve()
    cameras = sorted((clip / "rgbd").glob("cam*/calibration.json"))
    if len(cameras) != 6 or count < 5:
        raise ValueError("Expected six camera directories and at least five samples per camera")
    cfg = read_json(config)
    common = {k: cfg[k] for k in ("weights", "sam3d")}
    device = cfg.get("inference", {}).get("device", "auto")
    model = cfg["model"]
    torch.set_num_threads(2)
    infer = fah.load(model, mode="image", mesh="none", device=device,
                     mano_dir=cfg["mano_dir"], **common)
    options = method_args(model, "image", mesh="mano", device=device, **common)
    fitter = NativeShapeFitter(cfg["mano_dir"], options)
    bases = {}
    for side in SIDES:
        with (Path(cfg["mano_dir"]) / f"MANO_{side.upper()}.pkl").open("rb") as f:
            asset = pickle.load(f, encoding="latin1")
        bases[side] = {key: hashlib.sha256(np.asarray(asset[key], dtype="<f4").tobytes()).hexdigest()
                       for key in ("v_template", "shapedirs")}
    samples = []
    for cal in cameras:
        camera = cal.parent
        intr = read_json(cal)["original_calibration"]["intrinsics"]["color"]
        maps = rectification_maps(intr)
        width, height = intr["width"], intr["height"]
        k = np.float32([[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1]])
        files = sorted(camera.glob("color_*.jpg"))
        if len(files) < count:
            raise ValueError(f"{camera}: expected at least {count} RGB frames")
        counts = dict.fromkeys(SIDES, 0)
        for index in np.linspace(0, len(files) - 1, count, dtype=int):
            path = files[index]
            image = cv2.imread(str(path))
            if image is None or image.shape[:2] != (height, width):
                raise ValueError(f"Invalid RGB frame: {path}")
            hands = infer(cv2.remap(image, *maps, cv2.INTER_LINEAR), K=k)
            for side in SIDES:
                options = [h for h in hands if bool(h["is_right"]) == (side == "right")]
                if len(options) != 1:
                    continue
                hand = options[0]
                xmin, ymin, xmax, ymax = hand["bbox"]
                if (not np.isfinite(hand["score"]) or hand["score"] < 0.8
                        or min(xmin, ymin, width-xmax, height-ymax) < 3
                        or min(xmax-xmin, ymax-ymin) < 35):
                    continue
                vertices = hand.get("verts3d_dense")
                joints = hand["joints3d"]
                if vertices is None or vertices.shape != (778, 3) or not (
                    np.isfinite(vertices).all() and np.isfinite(joints).all()
                ):
                    continue
                beta, error = fitter.fit(vertices, joints, is_right=side == "right")
                if not np.isfinite(beta).all() or not np.isfinite(error) or error > 0.005:
                    continue
                counts[side] += 1
                samples.append(dict(camera=camera.name, side=side, image=path.name,
                                    betas=beta.tolist(), surface_error_mm=error*1000))
        print(f"{camera.name}: {counts}", flush=True)
    atomic_json(output, dict(clip=str(clip), rgb_only=True, mano_basis_sha256=bases,
                             fah_model=model, config_sha256=digest(config),
                             cameras=[p.parent.name for p in cameras], samples=samples))


def aggregate(samples, cameras, min_views=3, min_samples=5):
    if len(cameras) != 6 or len(set(cameras)) != 6:
        raise ValueError("Expected six distinct camera IDs")
    output, counts = [], {}
    for side in SIDES:
        groups = {c: [r["betas"] for r in samples if r["side"] == side and r["camera"] == c]
                  for c in cameras}
        counts[side] = {c: len(b) for c, b in groups.items()}
        valid = [np.asarray(b, dtype=np.float32) for b in groups.values() if len(b) >= min_samples]
        if len(valid) < min_views:
            raise ValueError(f"{side}: need at least {min_views} views with {min_samples} accepted samples")
        if any(b.ndim != 2 or b.shape[1] != 10 or not np.isfinite(b).all() for b in valid):
            raise ValueError(f"{side}: invalid native MANO samples")
        output.append(np.median([np.median(b, axis=0) for b in valid], axis=0))
    return np.asarray(output, dtype=np.float32), counts


def publish(source, output, mano):
    from hocap_annotation.hand_shape import model_basis_hashes

    candidate = read_json(source)
    basis = model_basis_hashes(mano)
    if basis != candidate["mano_basis_sha256"]:
        raise ValueError("FAH and HO-Cap MANO bases differ; cannot publish either side")
    betas, counts = aggregate(candidate["samples"], candidate["cameras"])
    models = {side: digest(mano / f"MANO_{side.upper()}.pkl") for side in SIDES}
    template = LENGTH_TEMPLATE.read_bytes()
    output.mkdir(parents=True, exist_ok=False)
    target = output / "betas.json"
    atomic_json(target, betas.tolist())
    atomic_json(output / "metadata.json", dict(
        schema="hocap.fixed_hand_shape.v2", hand_order=list(SIDES), rgb_only=True,
        source=candidate["clip"], source_samples_sha256=digest(source),
        fah_model=candidate["fah_model"], config_sha256=candidate["config_sha256"],
        quality_status="trial_not_manually_accepted", accepted_samples=counts,
        method="Per-camera median, then equal-camera median in native side MANO bases",
        basis_encoding="float32-le-c-order", basis_sha256=basis, model_sha256=models,
        betas_sha256=digest(target),
    ))
    atomic_json(output / "calibration_samples.json", candidate)
    atomic_write(output / "lengths.yaml", lambda f: f.write(template))
    print(target)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_subparsers(dest="mode", required=True)
    a = modes.add_parser("collect")
    for key in ("clip", "config", "samples"):
        a.add_argument(key, type=Path)
    a.add_argument("--samples-per-camera", type=int, default=24)
    b = modes.add_parser("publish")
    for key in ("samples", "output", "mano"):
        b.add_argument(key, type=Path)
    args = p.parse_args()
    if args.mode == "collect":
        collect(args.clip, args.config, args.samples, args.samples_per_camera)
    else:
        publish(args.samples, args.output, args.mano)
