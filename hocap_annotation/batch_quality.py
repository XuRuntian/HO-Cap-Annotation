"""Batch quality policy, independent of process success and filesystem layout."""

from dataclasses import asdict, dataclass

import numpy as np

from .camera_consistency import check_cameras
from .triangulation import robust_triangulate


@dataclass(frozen=True)
class QualityPolicy:
    sample_frames: int = 48
    max_sample_frames: int = 192
    reprojection_px: float = 10.0
    min_angle_deg: float = 1.0
    min_views: int = 2
    reference_views: int = 3
    min_joints: int = 12
    min_presence_frames: int = 3
    min_precheck_coverage: float = 0.8
    min_reference_coverage: float = 0.5
    min_final_coverage: float = 0.9
    fit_reprojection_px: float = 25.0
    fit_joint_error_m: float = 0.05
    max_timestamp_span_ms: float = 20.0
    camera_bad_px: float = 30.0
    camera_bad_fraction: float = 0.8
    camera_min_frames: int = 12
    camera_temporal_bins: int = 4
    camera_min_bins: int = 3
    camera_min_frames_per_bin: int = 3
    camera_min_joints: int = 8

    def __post_init__(self):
        integer_fields = (
            "sample_frames",
            "max_sample_frames",
            "min_views",
            "reference_views",
            "min_joints",
            "min_presence_frames",
            "camera_min_frames",
            "camera_temporal_bins",
            "camera_min_bins",
            "camera_min_frames_per_bin",
            "camera_min_joints",
        )
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid quality threshold: {key}")
            if key in integer_fields and type(value) is not int:
                raise ValueError(f"{key} must be an integer")
        if not (
            2 <= self.min_views <= self.reference_views <= 6
            and 3 <= self.reference_views
            and self.min_joints <= 21
            and self.camera_min_joints <= 21
            and self.min_angle_deg < 90
            and self.sample_frames <= self.max_sample_frames
            and self.camera_min_bins <= self.camera_temporal_bins
        ):
            raise ValueError("Inconsistent quality thresholds")
        for key in (
            "min_precheck_coverage",
            "min_reference_coverage",
            "min_final_coverage",
            "camera_bad_fraction",
        ):
            if getattr(self, key) > 1:
                raise ValueError(f"{key} must be in (0, 1]")

    def camera_options(self):
        return dict(
            bad_error_px=self.camera_bad_px,
            bad_fraction=self.camera_bad_fraction,
            min_frames=self.camera_min_frames,
            temporal_bins=self.camera_temporal_bins,
            min_covered_bins=self.camera_min_bins,
            min_frames_per_bin=self.camera_min_frames_per_bin,
            min_reference_views=self.reference_views,
            min_joints=self.camera_min_joints,
        )


def hand_presence(observed, policy):
    """Return evidence frames, without treating an unseen hand as a failed fit."""
    views = (observed.sum(-1) >= policy.camera_min_joints).sum(1)
    evidence = views > 0
    states = []
    for h in range(2):
        n = int((views[:, h] >= policy.min_views).sum())
        seen = int((views[:, h] > 0).sum())
        states.append(
            "present"
            if n >= policy.min_presence_frames
            else "unobserved"
            if seen == 0
            else "uncertain"
        )
    return evidence, states


def precheck(
    keypoints, observed, matrices, camera_ids, policy, *, automatic=True, excluded=()
):
    """Escalate samples, then require actual reference-group geometric coverage.

    Camera votes alone never authorize fitting. The reference cameras must
    jointly reconstruct wrists and enough joints with non-degenerate rays.
    """
    unknown = set(excluded) - set(camera_ids)
    if unknown:
        raise ValueError(f"Unknown excluded camera IDs: {sorted(unknown)}")
    if len(set(camera_ids) - set(excluded)) < policy.min_views:
        return dict(
            schema="camera_consistency_v1",
            camera_selection="multiview_consistency" if automatic else "manual_only",
            included_camera_ids=[c for c in camera_ids if c not in excluded],
            excluded_cameras=[
                dict(camera_id=c, reason="manual_exclusion")
                for c in camera_ids
                if c in excluded
            ],
            config=dict(
                consensus_threshold_px=policy.reprojection_px,
                min_angle_deg=policy.min_angle_deg,
                sample_frames=0,
            ),
            quality_policy=asdict(policy),
            cameras={},
            coverage=[],
            sampled_frame_indices=[],
            attempts=[],
            warnings=[],
            usable=False,
            reasons=["insufficient cameras after manual exclusions"],
        )
    attempts = []
    count = min(policy.sample_frames, len(keypoints))
    limit = min(policy.max_sample_frames, len(keypoints))
    while True:
        report = check_cameras(
            keypoints,
            observed,
            matrices,
            camera_ids,
            automatic=True,
            excluded=excluded,
            sample_frames=count,
            threshold=policy.reprojection_px,
            min_angle=policy.min_angle_deg,
            min_views=policy.min_views,
            **policy.camera_options(),
        )
        if not automatic:
            report["camera_selection"] = "manual_only"
            report["included_camera_ids"] = [c for c in camera_ids if c not in excluded]
            report["excluded_cameras"] = [
                dict(camera_id=c, reason="manual_exclusion")
                for c in camera_ids
                if c in excluded
            ]
        reference = [
            c
            for c in report["included_camera_ids"]
            if report["cameras"][c]["status"] == "retained"
        ]
        sample = report["sampled_frame_indices"]
        evidence, states = hand_presence(observed[sample], policy)
        supported = np.zeros((len(sample), 2), bool)
        reference_supported = np.zeros_like(supported)
        allowed = np.array([c in reference for c in camera_ids])
        if allowed.sum() >= policy.reference_views:
            for i, frame in enumerate(sample):
                points, inliers = robust_triangulate(
                    keypoints[frame].reshape(len(camera_ids), 42, 2),
                    (observed[frame] & allowed[:, None, None]).reshape(
                        len(camera_ids), 42
                    ),
                    matrices,
                    threshold=policy.reprojection_px,
                    min_views=policy.min_views,
                    min_angle=policy.min_angle_deg,
                )
                good = np.isfinite(points).all(-1).reshape(2, 21)
                supported[i] = good[:, 0] & (good.sum(-1) >= policy.min_joints)
                strong = good & (
                    inliers.sum(0).reshape(2, 21) >= policy.reference_views
                )
                reference_supported[i] = strong[:, 0] & (
                    strong.sum(-1) >= policy.min_joints
                )
        coverage = []
        reasons = []
        for h, side in enumerate(("right", "left")):
            n = int(evidence[:, h].sum())
            ratio = float((supported[:, h] & evidence[:, h]).sum() / n) if n else None
            reference_ratio = (
                float((reference_supported[:, h] & evidence[:, h]).sum() / n)
                if n
                else None
            )
            coverage.append(
                dict(
                    hand=side,
                    presence=states[h],
                    observed_frames=n,
                    supported_frames=int((supported[:, h] & evidence[:, h]).sum()),
                    coverage=ratio,
                    reference_coverage=reference_ratio,
                )
            )
            if states[h] == "uncertain":
                reasons.append(f"{side}: uncertain hand presence")
            if states[h] == "present" and ratio < policy.min_precheck_coverage:
                reasons.append(f"{side}: insufficient reconstruction coverage")
            if (
                states[h] == "present"
                and reference_ratio < policy.min_reference_coverage
            ):
                reasons.append(f"{side}: insufficient independent reference support")
        if "present" not in states:
            reasons.append("no sufficiently observed hand")
        if len(reference) < policy.reference_views:
            reasons.append("no reliable reference camera group")
        for ci, cid in enumerate(camera_ids):
            seen_frames = int(
                (observed[:, ci].sum(-1) >= policy.camera_min_joints).any(-1).sum()
            )
            if (
                cid in report["included_camera_ids"]
                and seen_frames >= policy.camera_min_frames
                and report["cameras"][cid]["status"] == "insufficient_evidence"
            ):
                reasons.append(f"{cid}: insufficient geometric evidence")
        if any(
            report["cameras"][c]["status"] == "ambiguous_retained" for c in camera_ids
        ):
            reasons.append("ambiguous camera selection")
        attempts.append(dict(samples=len(sample), reasons=reasons, coverage=coverage))
        if not reasons or count >= limit:
            break
        count = min(limit, max(count + 1, count * 2))
    report.update(
        quality_policy=asdict(policy),
        automatic_exclusion=automatic,
        reference_camera_ids=reference,
        coverage=coverage,
        attempts=attempts,
        usable=not reasons,
        reasons=reasons,
    )
    return report


def final_quality(observed, diagnostics, policy):
    evidence, states = hand_presence(observed, policy)
    valid = diagnostics["valid"]
    if valid.shape != evidence.shape or valid.dtype != np.bool_:
        raise ValueError("Invalid final hand validity")
    error = diagnostics["reprojection_median_px"]
    geometry = diagnostics["joint_error_m"]
    if error.shape != valid.shape or geometry.shape != valid.shape:
        raise ValueError("Invalid final diagnostics")
    quality_valid = (
        valid
        & np.isfinite(error)
        & np.isfinite(geometry)
        & (error <= policy.fit_reprojection_px)
        & (geometry <= policy.fit_joint_error_m)
    )
    reasons, hands = [], []
    for h, side in enumerate(("right", "left")):
        n = int(evidence[:, h].sum())
        coverage = (
            float((quality_valid[:, h] & evidence[:, h]).sum() / n) if n else None
        )
        hands.append(
            dict(
                hand=side,
                presence=states[h],
                observed_frames=n,
                valid_frames=int(quality_valid[:, h].sum()),
                coverage=coverage,
            )
        )
        if states[h] == "uncertain":
            reasons.append(f"{side}: uncertain hand presence")
        if states[h] == "present" and coverage < policy.min_final_coverage:
            reasons.append(f"{side}: final quality coverage below threshold")
    if "present" not in states:
        reasons.append("no sufficiently observed hand")
    return dict(passed=not reasons, hands=hands, reasons=reasons)
