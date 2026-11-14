"""Conservative, sequence-local camera selection from held-out geometry."""

import numpy as np

from .triangulation import robust_triangulate, projection_errors


def check_cameras(
    keypoints,
    valid,
    matrices,
    camera_ids,
    *,
    automatic=True,
    excluded=(),
    sample_frames=48,
    threshold=10.0,
    min_angle=1.0,
    min_views=2,
    bad_error_px=30.0,
    bad_fraction=0.8,
    min_frames=12,
    temporal_bins=4,
    min_covered_bins=3,
    min_frames_per_bin=3,
    min_reference_views=3,
    min_joints=8,
):
    """Require independent three-view consensus and temporally spread evidence.

    Each held-out camera gets one vote per sampled frame (not 42 correlated
    joint votes). A camera with too little evidence is retained, not condemned.
    """
    unknown = set(excluded) - set(camera_ids)
    if unknown:
        raise ValueError(f"Unknown excluded camera IDs: {sorted(unknown)}")
    if sample_frames < 1:
        raise ValueError("Camera check sample count must be positive")
    enabled = np.array([cid not in excluded for cid in camera_ids])
    if enabled.sum() < min_views:
        raise ValueError("Too few cameras after manual exclusions")
    total = len(keypoints)
    sampled = np.unique(np.linspace(0, total - 1, min(sample_frames, total), dtype=int))
    bad_threshold = max(bad_error_px, threshold * 3)
    config = dict(
        sample_frames=sample_frames,
        consensus_threshold_px=threshold,
        bad_error_px=bad_threshold,
        bad_fraction=bad_fraction,
        min_frames=min_frames,
        temporal_bins=temporal_bins,
        min_covered_bins=min_covered_bins,
        min_frames_per_bin=min_frames_per_bin,
        min_reference_views=min_reference_views,
        min_joints=min_joints,
        min_angle_deg=min_angle,
    )
    details = {}
    candidates = []
    for ci, cid in enumerate(camera_ids):
        records = []
        if automatic and enabled[ci]:
            for fi in sampled:
                xy = keypoints[fi].reshape(len(camera_ids), -1, 2)
                seen = valid[fi].reshape(len(camera_ids), -1) & enabled[:, None]
                held_out = seen[ci].copy()
                seen[ci] = False
                if not held_out.any():
                    continue
                points, inliers = robust_triangulate(
                    xy,
                    seen,
                    matrices,
                    threshold=threshold,
                    min_views=min_reference_views,
                    min_angle=min_angle,
                )
                # Reject ambiguous two-view matches and minority clusters.
                reliable = (
                    held_out
                    & (inliers.sum(0) >= min_reference_views)
                    & (inliers.sum(0) > seen.sum(0) / 2)
                )
                errors = projection_errors(points, xy, matrices)[ci]
                hand_errors = []
                for h in range(2):
                    select = reliable.reshape(2, 21)[h]
                    if select.sum() >= config["min_joints"]:
                        hand_errors.append(
                            float(np.median(errors.reshape(2, 21)[h, select]))
                        )
                if hand_errors:
                    records.append((int(fi), float(np.median(hand_errors))))
        errors = np.array([r[1] for r in records])
        bins = []
        for b in range(temporal_bins):
            values = [
                e
                for f, e in records
                if min(temporal_bins - 1, f * temporal_bins // total) == b
            ]
            bins.append(
                dict(
                    frames=len(values),
                    bad_fraction=(
                        float(np.mean(np.array(values) > bad_threshold))
                        if values
                        else None
                    ),
                )
            )
        covered = [b for b in bins if b["frames"] >= config["min_frames_per_bin"]]
        enough = len(records) >= min_frames and len(covered) >= min_covered_bins
        persistent = (
            enough
            and (errors > bad_threshold).mean() >= bad_fraction
            and all(b["bad_fraction"] >= bad_fraction for b in covered)
        )
        status = (
            "manually_excluded"
            if not enabled[ci]
            else "disabled"
            if not automatic
            else "insufficient_evidence"
            if not enough
            else "persistent_anomaly"
            if persistent
            else "retained"
        )
        if persistent:
            candidates.append(ci)
        median = float(np.median(errors)) if len(errors) else None
        details[cid] = dict(
            status=status,
            evaluated_frames=len(records),
            temporal_bins=bins,
            median_error_px=median
            if median is not None and np.isfinite(median)
            else None,
            bad_fraction=float((errors > bad_threshold).mean())
            if len(errors)
            else None,
            nonfinite_error_frames=int((~np.isfinite(errors)).sum()),
        )
    warnings = []
    # Do not choose arbitrary winners when the evidence would remove the core.
    if enabled.sum() - len(candidates) < max(min_reference_views, min_views):
        for ci in candidates:
            details[camera_ids[ci]]["status"] = "ambiguous_retained"
        if candidates:
            warnings.append(
                "Automatic exclusion would leave too few reference cameras; retained candidates"
            )
        candidates = []
    result_excluded = []
    for ci, cid in enumerate(camera_ids):
        if not enabled[ci] or ci in candidates:
            reason = (
                "manual_exclusion"
                if not enabled[ci]
                else "persistent_geometric_inconsistency"
            )
            result_excluded.append(dict(camera_id=cid, reason=reason))
            enabled[ci] = False
        elif details[cid]["status"] == "insufficient_evidence":
            warnings.append(
                f"{cid}: insufficient evidence; retained for per-joint screening"
            )
    return dict(
        schema="camera_consistency_v1",
        camera_selection="multiview_consistency" if automatic else "manual_only",
        included_camera_ids=[cid for ci, cid in enumerate(camera_ids) if enabled[ci]],
        excluded_cameras=result_excluded,
        cameras=details,
        config=config,
        sampled_frame_indices=sampled.tolist(),
        warnings=warnings,
    )
