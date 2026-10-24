"""Separate observed 3D joints from conservative MANO initialization seeds."""

import numpy as np


def prepare_hand_initialization(joints, view_counts, max_gap=5):
    """Return wrist seeds, fit validity, original observation mask, and seed source.

    Short missing-3D runs may be initialized only when every frame has hand
    detections in at least two views. Long gaps and absent hands remain invalid.
    Source codes: 0 invalid, 1 observed wrist, 2 interior interpolation, 3 edge hold.
    Inspired by MINT's validity-preserving processing and EgoHandKit's explicit
    missing-hand boundaries; no smoothing backend is imported from either repo.
    """
    joints = np.asarray(joints)
    if joints.ndim != 4 or joints.shape[2:] != (21, 3):
        raise ValueError('Expected joints[hand, frame, 21, 3]')
    views = np.asarray(view_counts)
    if views.shape != joints.shape[:2] or not np.isfinite(views).all() or np.any(views < 0):
        raise ValueError('Expected finite nonnegative view counts for each hand/frame')
    if type(max_gap) is not int or max_gap < 0:
        raise ValueError('max_gap must be a nonnegative integer')
    observed = np.isfinite(joints).all(axis=-1) & ~np.all(joints == -1, axis=-1)
    source = observed[..., 0].astype(np.uint8)
    seeds = np.zeros((*joints.shape[:2], 3), dtype=np.float32)
    seeds[source == 1] = joints[..., 0, :][source == 1]
    for hand in range(len(joints)):
        wrist_valid = observed[hand, :, 0]
        frame = 0
        while frame < len(wrist_valid):
            if wrist_valid[frame]:
                frame += 1
                continue
            start = frame
            while frame < len(wrist_valid) and not wrist_valid[frame]:
                frame += 1
            end = frame
            if end-start > max_gap or not np.all(views[hand, start:end] >= 2):
                continue
            left, right = start-1, end
            if left >= 0 and right < len(wrist_valid):
                weight = np.arange(1, end-start+1, dtype=float) / (end-start+1)
                seeds[hand, start:end] = (seeds[hand, left] * (1-weight[:, None])
                                          + seeds[hand, right] * weight[:, None])
                source[hand, start:end] = 2
            elif left >= 0 or right < len(wrist_valid):
                anchor = left if left >= 0 else right
                seeds[hand, start:end] = seeds[hand, anchor]
                source[hand, start:end] = 3
    return seeds, source > 0, observed, source
