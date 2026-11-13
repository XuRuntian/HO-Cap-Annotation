"""Validate reusable fixed-hand-shape bundles before starting a pipeline."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from .rgbd_data import digest


def load_hand_shape(directory, mano_root=None):
    directory = Path(directory).expanduser()
    if (directory / "trial_shape.npz").is_file():
        if (directory / "betas.json").exists():
            raise ValueError("Ambiguous hand shape directory: contains both trial_shape.npz and betas.json")
        return load_easymocap_shape(directory, mano_root)
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


def load_easymocap_shape(directory, mano_root):
    """Adapt an EasyMocap output without writing into the source directory."""
    if mano_root is None:
        mano_root = Path.home() / "models/mano"
    mano_root = Path(mano_root).expanduser().resolve()
    report = json.loads((directory / "report.json").read_text())
    if report.get("status") != "complete":
        raise ValueError("EasyMocap shape report must have status complete")
    for side in ("right", "left"):
        expected = report.get("model_sha256", {}).get(side)
        if expected != digest(mano_root / f"MANO_{side.upper()}.pkl"):
            raise ValueError(f"EasyMocap source MANO model mismatch: {side}")
    root = Path(os.environ.get("EASYMOCAP_ROOT", Path.home() / "EasyMocap")).expanduser()
    python = Path(os.environ.get("EASYMOCAP_PYTHON", Path.home() / "miniconda3/envs/easymocap/bin/python")).expanduser()
    # The launcher may use a lightweight detector environment for intermediate stages.
    importer_python = os.environ.get("MANO_PYTHON", sys.executable)
    tool = Path(__file__).resolve().parents[1] / "tools/import_hand_shape.py"
    with tempfile.TemporaryDirectory(prefix="hocap-hand-shape-") as tmp:
        output = Path(tmp) / "converted"
        result = subprocess.run(
            [importer_python, str(tool), str(directory / "trial_shape.npz"),
             "--output", str(output), "--easymocap-root", str(root),
             "--easymocap-python", str(python), "--mano-root", str(mano_root)],
            capture_output=True, text=True,
        )
        if result.returncode:
            raise ValueError("Automatic EasyMocap shape import failed:\n" + result.stderr[-4000:])
        return load_hand_shape(output, mano_root)
