"""Write HO-Cap 2D observations from the FAH environment."""

import json
from pathlib import Path

import cv2
import numpy as np

from ..rgb_hand_pipeline import log_config, outputs
from ..rgbd_data import atomic_npz, digest


def _infer(seq, model):
    t, c = len(seq.frame_ids), len(seq.camera_ids)
    xy = np.zeros((t, c, 2, 21, 2), np.float32)
    valid = np.zeros((t, c, 2, 21), bool)
    scores = np.zeros((t, c, 2), np.float32)
    duplicate = np.zeros((t, c, 2), bool)
    for ci, cid in enumerate(seq.camera_ids):
        intr = seq.manifest["cameras"][cid]["intrinsics"]
        k = np.array(
            [[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1]],
            dtype=np.float32,
        )
        for i in range(t):
            bgr = cv2.cvtColor(seq.image(i, cid), cv2.COLOR_RGB2BGR)
            for hand in model(bgr, K=k):
                hi = 0 if hand["is_right"] else 1
                points = np.asarray(hand["joints2d"], dtype=np.float32)
                score = float(hand["score"])
                if points.shape != (21, 2) or not np.isfinite(score):
                    continue
                finite = np.isfinite(points).all(axis=1)
                if not finite.any():
                    continue
                if valid[i, ci, hi].any():
                    duplicate[i, ci, hi] = True
                    if score <= scores[i, ci, hi]:
                        continue
                xy[i, ci, hi] = np.where(finite[:, None], points, 0)
                valid[i, ci, hi] = finite
                scores[i, ci, hi] = score
        print(f"Detected {cid}: {t} frames (FAH)", flush=True)
    return dict(
        keypoints=xy, valid=valid, detection_score=scores, duplicate_side=duplicate
    )


def detection_spec(config_path):
    """Resolve FAH assets and detection provenance without importing FAH."""
    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text())
    model_name = cfg.get("model", "fah")
    if model_name not in ("fah-s", "fah-b", "fah", "fah-h"):
        raise ValueError(f"Unknown FAH model: {model_name}")
    paths = {}
    for key in ("weights", "sam3d", "mano_dir"):
        if not cfg.get(key):
            raise ValueError(f"FAH config must specify {key}")
        paths[key] = str((config_path.parent / Path(cfg[key]).expanduser()).resolve())
    release_path = Path(paths["weights"]) / model_name / "config.json"
    release = json.loads(release_path.read_text())
    assets = {
        "model_config": release_path,
        "model": release_path.parent / release["model"],
        "detector": release_path.parent / release["detector"],
        "sam3d_config": Path(paths["sam3d"]) / "model_config.yaml",
        "mhr": Path(paths["sam3d"]) / "assets/mhr_model.pt",
    }
    for side in ("LEFT", "RIGHT"):
        assets[f"mano_{side.lower()}"] = Path(paths["mano_dir"]) / f"MANO_{side}.pkl"
    fingerprints = {key: digest(file) for key, file in assets.items()}
    device = cfg.get("inference", {}).get("device", "auto")
    return paths, dict(
        detector="fah",
        model=model_name,
        backend="torch",
        mode="image",
        mesh="none",
        device=device,
        model_sha256=fingerprints["model"],
        assets_sha256=fingerprints,
    )


def detect(seq, config_path, replace=False):
    (path,) = outputs(seq, ["observations_2d.npz"], replace)
    try:
        import fah
    except ModuleNotFoundError as exc:
        if exc.name != "fah":
            raise
        raise RuntimeError(
            "Run FAH detection using the Python environment with FAH installed"
        ) from exc

    paths, config = detection_spec(config_path)
    settings = {key: config[key] for key in ("mode", "mesh", "device")}
    observations = _infer(seq, fah.load(config["model"], **settings, **paths))
    atomic_npz(
        path,
        **seq.common(),
        **observations,
        detector=np.array("fah"),
        model_sha256=np.array(config["model_sha256"]),
    )
    log_config(seq, "detect", config)
    return path
