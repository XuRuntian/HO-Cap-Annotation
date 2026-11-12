"""Validate reusable fixed-hand-shape bundles before starting a pipeline."""

import json
from pathlib import Path

import numpy as np

from .rgbd_data import digest


def load_hand_shape(directory, mano_root=None):
    directory = Path(directory).expanduser()
    beta_path = directory / "betas.json"
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("schema") != "hocap.fixed_hand_shape.v1":
        raise ValueError("Unsupported hand shape metadata schema")
    if metadata.get("hand_order") != ["right", "left"]:
        raise ValueError("Hand shape order must be right,left")
    if metadata.get("betas_sha256") != digest(beta_path):
        raise ValueError("Hand shape betas hash mismatch")
    betas = np.asarray(json.loads(beta_path.read_text()), dtype=np.float32)
    if betas.shape != (2, 10) or not np.isfinite(betas).all():
        raise ValueError("betas must be finite [2,10], right then left")
    fingerprints = metadata.get("model_sha256")
    if not isinstance(fingerprints, dict) or any(
        not isinstance(fingerprints.get(side), str) for side in ("right", "left")
    ):
        raise ValueError("Hand shape metadata must identify both MANO models")
    if mano_root is not None:
        for side in ("right", "left"):
            asset = Path(mano_root).expanduser() / f"MANO_{side.upper()}.pkl"
            if fingerprints[side] != digest(asset):
                raise ValueError(f"Hand shape MANO model mismatch: {side}")
    return betas.tolist()
