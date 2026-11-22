"""Validate reusable fixed-hand-shape bundles before starting a pipeline."""

import json
import hashlib
from pathlib import Path

import numpy as np

from .rgbd_data import digest


def load_hand_shape(directory, mano_root=None, *, verify_basis=True):
    directory = Path(directory).expanduser()
    if directory.name == "betas.json" and not directory.is_dir():
        directory = directory.parent
    if (directory / "trial_shape.npz").is_file() and not (directory / "betas.json").is_file():
        raise ValueError("Legacy trial_shape.npz-only output: export betas.json and metadata.json "
                         "on the calibration machine first; EasyMocap is not invoked here")
    beta_path = directory / "betas.json"
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("schema") not in ("hocap.fixed_hand_shape.v1", "hocap.fixed_hand_shape.v2"):
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
    if metadata['schema'] == 'hocap.fixed_hand_shape.v2' and mano_root is None:
        mano_root = Path.home() / 'models/mano'
    if mano_root is not None:
        for side in ("right", "left"):
            asset = Path(mano_root).expanduser() / f"MANO_{side.upper()}.pkl"
            if fingerprints[side] != digest(asset):
                raise ValueError(f"Hand shape MANO model mismatch: {side}")
    if metadata['schema'] == 'hocap.fixed_hand_shape.v2':
        if metadata.get('basis_encoding') != 'float32-le-c-order':
            raise ValueError('Unsupported MANO basis encoding')
        expected = metadata.get('basis_sha256')
        if not isinstance(expected, dict) or set(expected) != {'right', 'left'}:
            raise ValueError('Missing MANO basis fingerprints')
        if verify_basis and expected != model_basis_hashes(mano_root):
            raise ValueError('Hand shape effective MANO basis mismatch')
    return betas.tolist()


def model_basis_hashes(mano_root):
    from manopth.manolayer import ManoLayer

    result = {}
    for side in ('right', 'left'):
        layer = ManoLayer(mano_root=str(mano_root), side=side,
                          use_pca=True, ncomps=45, flat_hand_mean=False)
        result[side] = {}
        for name in ('v_template', 'shapedirs'):
            array = getattr(layer, 'th_' + name).detach().cpu().numpy()
            result[side][name] = hashlib.sha256(
                np.asarray(array, dtype='<f4').tobytes(order='C')).hexdigest()
    return result
