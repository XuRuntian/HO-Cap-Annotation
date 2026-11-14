"""Small sequential RGB hand pipeline: MediaPipe -> HO-Cap triangulation -> MANO.

No HoloLens, object models, masks, depth fitting or offscreen-render imports.
The original full HO-Cap scripts remain available separately.
"""

import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .rgbd_data import PreparedSequence, atomic_json, atomic_npz, digest

SIDES = ("right", "left")


def outputs(seq, names, replace):
    directory = seq.workdir / "predictions"
    if directory.is_symlink():
        raise ValueError("predictions must be a real directory, not a symlink")
    paths = [directory / n for n in names]
    if not replace and any(p.exists() for p in paths):
        raise ValueError("Stage output exists; use --replace to recompute predictions")
    return paths


def log_config(seq, stage, config):
    path = seq.workdir / "predictions/run_config.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    previous[stage] = config
    atomic_json(path, previous)


def detection_config(model_path, mode="video", device="cpu"):
    return dict(
        model_asset_path=str(model_path),
        device=device,
        running_mode=mode,
        frame_rate=30,
        max_num_hands=2,
        min_hand_detection_confidence=0.1,
        min_tracking_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )


def detect(seq, model_path, mode="video", device="cpu", replace=False):
    (path,) = outputs(seq, ["observations_2d.npz"], replace)
    model_path = Path(model_path).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"MediaPipe model missing: {model_path}")
    from .wrappers.mediapipe import MPHandDetector

    config = detection_config(model_path, mode, device)
    t, c = len(seq.frame_ids), len(seq.camera_ids)
    xy = np.zeros((t, c, 2, 21, 2), np.float32)
    valid = np.zeros((t, c, 2, 21), bool)
    scores = np.zeros((t, c, 2), np.float32)
    duplicate = np.zeros((t, c, 2), bool)
    for ci, cid in enumerate(seq.camera_ids):
        detector = MPHandDetector(SimpleNamespace(**config))
        first = None
        previous = -1
        try:
            for i in range(t):
                if mode == "video":
                    timestamp = seq.manifest["frames"][i]["timestamps_ms"][cid]
                    if timestamp is None or not np.isfinite(timestamp):
                        raise ValueError(
                            "Video detection requires real RGB timestamps; use --detector-mode image otherwise"
                        )
                    if first is None:
                        first = timestamp
                    elapsed = round(timestamp - first)
                    if elapsed <= previous:
                        raise ValueError(
                            "RGB timestamps must increase for MediaPipe VIDEO mode"
                        )
                    detector._timestamp_ms = elapsed
                    previous = elapsed
                marks, sides, confidence = detector.detect_one(seq.image(i, cid))
                for hand_side in SIDES:
                    hi = SIDES.index(hand_side)
                    indices = [j for j, s in enumerate(sides) if s == hand_side]
                    if not indices:
                        continue
                    # Retain the strongest same-side detection, never fabricate the other side.
                    j = max(indices, key=lambda j: confidence[j])
                    duplicate[i, ci, hi] = len(indices) > 1
                    points = np.asarray(marks[j], dtype=np.float32)
                    if points.shape != (21, 2) or not np.isfinite(points).all():
                        continue
                    xy[i, ci, hi] = points
                    valid[i, ci, hi] = True
                    scores[i, ci, hi] = confidence[j]
        finally:
            detector.close()
        print(f"Detected {cid}: {t} frames", flush=True)
    atomic_npz(
        path,
        **seq.common(),
        keypoints=xy,
        valid=valid,
        handedness_score=scores,
        duplicate_side=duplicate,
        model_sha256=np.array(digest(model_path)),
    )
    log_config(
        seq,
        "detect",
        dict(config, model_asset_path=model_path.name, model_sha256=digest(model_path)),
    )
    return path


def check_observations(seq, obs):
    t, c = len(seq.frame_ids), len(seq.camera_ids)
    if (
        obs["camera_ids"].tolist() != seq.camera_ids
        or obs["keypoints"].shape != (t, c, 2, 21, 2)
        or obs["valid"].shape != (t, c, 2, 21)
        or obs["valid"].dtype != np.bool_
        or not np.isfinite(obs["keypoints"]).all()
    ):
        raise ValueError("Invalid 2D observation arrays")


def project(points, matrices):
    h = np.concatenate((points, np.ones((*points.shape[:-1], 1))), axis=-1)
    q = np.einsum("cij,...kj->...cki", matrices, h)
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = q[..., :2] / q[..., 2:3]
    return pixels, q[..., 2]


def triangulate(
    seq,
    device="cpu",
    steps=100,
    threshold=10.0,
    min_views=2,
    replace=False,
    *,
    camera_check=True,
    exclude_cameras=(),
    camera_check_samples=48,
    min_angle=1.0,
    selection_report=None,
):
    from .triangulation import robust_triangulate
    from .camera_consistency import check_cameras

    path, report_path = outputs(
        seq, ["joints_3d.npz", "camera_consistency.json"], replace
    )
    obs = seq.load("observations_2d.npz")
    check_observations(seq, obs)
    t, c = len(seq.frame_ids), len(seq.camera_ids)
    if (
        steps < 0
        or not np.isfinite(threshold)
        or threshold <= 0
        or not 2 <= min_views <= c
        or not 0 < min_angle < 90
    ):
        raise ValueError("Invalid triangulation options")
    report = (
        check_cameras(
            obs["keypoints"],
            obs["valid"],
            seq.matrices,
            seq.camera_ids,
            automatic=camera_check,
            excluded=exclude_cameras,
            sample_frames=camera_check_samples,
            threshold=threshold,
            min_angle=min_angle,
            min_views=min_views,
        )
        if selection_report is None
        else dict(selection_report)
    )
    if selection_report is not None:
        if (
            report.get("manifest_sha256") != seq.fingerprint
            or report.get("observations_sha256")
            != digest(seq.workdir / "predictions/observations_2d.npz")
            or report["config"]["consensus_threshold_px"] != threshold
            or report["config"]["min_angle_deg"] != min_angle
            or set(report["included_camera_ids"]) - set(seq.camera_ids)
        ):
            raise ValueError("Camera selection report is stale or incompatible")
        included = report["included_camera_ids"]
        omitted = [item["camera_id"] for item in report["excluded_cameras"]]
        if (
            len(set(included)) != len(included)
            or len(set(omitted)) != len(omitted)
            or set(included) & set(omitted)
            or set(included) | set(omitted) != set(seq.camera_ids)
            or len(included) < min_views
            or report.get("usable") is False
            or report.get("quality_policy", {}).get("min_views", min_views) != min_views
        ):
            raise ValueError("Invalid or unusable camera selection report")
        camera_check = report.get(
            "automatic_exclusion", report["camera_selection"] != "manual_only"
        )
        camera_check_samples = report["config"]["sample_frames"]
        exclude_cameras = [
            item["camera_id"]
            for item in report["excluded_cameras"]
            if item["reason"] == "manual_exclusion"
        ]
    report.update(
        manifest_sha256=seq.fingerprint,
        observations_sha256=digest(seq.workdir / "predictions/observations_2d.npz"),
        sampled_frame_ids=[seq.frame_ids[i] for i in report["sampled_frame_indices"]],
    )
    enabled = np.array([cid in report["included_camera_ids"] for cid in seq.camera_ids])
    print(
        f"Reconstruction cameras: {report['included_camera_ids']}; excluded: {report['excluded_cameras']}",
        flush=True,
    )
    for warning in report["warnings"]:
        print(f"Camera check warning: {warning}", flush=True)
    xyz = np.zeros((t, 2, 21, 3), np.float32)
    valid = np.zeros((t, 2, 21), bool)
    inliers = np.zeros((t, c, 2, 21), bool)
    error = np.full((t, c, 2, 21), np.nan, np.float32)
    for i in range(t):
        result, selected = robust_triangulate(
            obs["keypoints"][i].reshape(c, 42, 2),
            (obs["valid"][i] & enabled[:, None, None]).reshape(c, 42),
            seq.matrices,
            threshold=threshold,
            steps=steps,
            device=device,
            min_views=min_views,
            min_angle=min_angle,
        )
        result = result.reshape(2, 21, 3)
        selected = selected.reshape(c, 2, 21)
        usable = np.isfinite(result).all(-1)
        for h in range(2):
            uv, z = project(result[h], seq.matrices)
            e = np.linalg.norm(uv - obs["keypoints"][i, :, h], axis=-1)
            good = selected[:, h]
            accepted = usable[h] & (good.sum(axis=0) >= min_views)
            valid[i, h] = accepted
            xyz[i, h, accepted] = result[h, accepted]
            inliers[i, :, h] = good & accepted
            error[i, :, h] = np.where(obs["valid"][i, :, h] & accepted, e, np.nan)
        if i % 25 == 0:
            print(f"Triangulated {i + 1}/{t}", flush=True)
    atomic_json(report_path, report)
    reconstruction = {
        k: report[k]
        for k in ("camera_selection", "included_camera_ids", "excluded_cameras")
    }
    reconstruction.update(
        diagnostics_file=report_path.name,
        diagnostics_sha256=digest(report_path),
        triangulation_method="pairwise_consensus_inlier_refinement_v1",
    )
    atomic_npz(
        path,
        **seq.common(),
        joints=xyz,
        valid=valid,
        inliers=inliers,
        reprojection_error_px=error,
        reconstruction_json=np.array(json.dumps(reconstruction, sort_keys=True)),
        observations_sha256=np.array(
            digest(seq.workdir / "predictions/observations_2d.npz")
        ),
    )
    log_config(
        seq,
        "triangulate",
        dict(
            device=device,
            steps=steps,
            threshold_px=threshold,
            min_views=min_views,
            min_angle_deg=min_angle,
            camera_check=camera_check,
            camera_check_samples=camera_check_samples,
            exclude_cameras=list(exclude_cameras),
        ),
    )
    return path


def reconstruction_metadata(seq, tri):
    """Legacy triangulations have no camera-selection record; never invent one."""
    if "reconstruction_json" not in tri:
        return None
    metadata = json.loads(str(tri["reconstruction_json"]))
    report = seq.workdir / "predictions/camera_consistency.json"
    if not report.is_file() or digest(report) != metadata["diagnostics_sha256"]:
        raise ValueError("Camera consistency report is stale; rerun triangulate")
    return metadata


def fit(
    seq,
    mano_root,
    device="cuda",
    steps=1000,
    lr=0.001,
    betas=None,
    min_joints=12,
    max_error_px=25.0,
    replace=False,
    max_joint_error_m=0.05,
):
    import torch
    from manopth.manolayer import ManoLayer
    from .loss import Keypoint2DLoss, Keypoint3DLoss, MANORegLoss, PoseSmoothnessLoss

    path, diag = outputs(seq, ["fit_parameters.npz", "fit_diagnostics.npz"], replace)
    if (
        steps <= 0
        or lr <= 0
        or not 1 <= min_joints <= 21
        or max_error_px <= 0
        or max_joint_error_m <= 0
    ):
        raise ValueError("Invalid fit options")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(
            "CUDA unavailable; use --device cpu for a small validation run"
        )
    obs = seq.load("observations_2d.npz")
    check_observations(seq, obs)
    tri = seq.load("joints_3d.npz")
    if str(tri["observations_sha256"]) != digest(
        seq.workdir / "predictions/observations_2d.npz"
    ):
        raise ValueError(
            "Triangulation is stale; rerun triangulate after changing observations"
        )
    reconstruction_metadata(seq, tri)
    beta = (
        np.zeros((2, 10), np.float32)
        if betas is None
        else np.asarray(betas, dtype=np.float32)
    )
    if beta.shape != (2, 10) or not np.isfinite(beta).all():
        raise ValueError("betas must be finite [2,10], right then left")
    t = len(seq.frame_ids)
    poses = np.zeros((t, 2, 51), np.float32)
    valid = np.zeros((t, 2), bool)
    losses = np.full((2, steps), np.nan, np.float32)
    errors = np.full((t, 2), np.nan, np.float32)
    geometry_error = np.full((t, 2), np.nan, np.float32)
    camera_errors = np.full((t, len(seq.camera_ids), 2), np.nan, np.float32)
    camera_inlier_errors = np.full_like(camera_errors, np.nan)
    local = np.zeros((t, 2, 45), np.float32)
    fingerprints = []
    if (
        tri["joints"].shape != (t, 2, 21, 3)
        or tri["valid"].shape != (t, 2, 21)
        or tri["inliers"].shape != obs["valid"].shape
    ):
        raise ValueError("Invalid triangulation dimensions")
    M = torch.tensor(seq.matrices, dtype=torch.float32, device=device)
    sizes = torch.tensor(seq.sizes - 1, dtype=torch.float32, device=device)[
        None, :, None, :
    ]
    loss2 = Keypoint2DLoss("l2_norm")
    loss3 = Keypoint3DLoss("l2_norm")
    reg = MANORegLoss()
    smooth = PoseSmoothnessLoss(
        win_size=1, w_vel_r=0, w_vel_t=0, w_acc_r=1, w_acc_t=1
    ).to(device)
    for h, side in enumerate(SIDES):
        asset = Path(mano_root) / f"MANO_{side.upper()}.pkl"
        if not asset.is_file():
            raise FileNotFoundError(f"Missing licensed MANO asset: {asset}")
        fingerprints.append(digest(asset))
        indices = np.flatnonzero(
            tri["valid"][:, h, 0] & (tri["valid"][:, h].sum(-1) >= min_joints)
        )
        if not len(indices):
            continue
        layer = ManoLayer(
            mano_root=str(mano_root),
            side=side,
            use_pca=True,
            ncomps=45,
            flat_hand_mean=False,
            center_idx=None,
        ).to(device)
        b = torch.tensor(beta[h], device=device).expand(len(indices), -1)
        target = torch.tensor(tri["joints"][indices, h], device=device)
        mask3 = torch.tensor(tri["valid"][indices, h], device=device)
        targets2 = torch.tensor(obs["keypoints"][indices, :, h], device=device) / sizes
        mask2 = torch.tensor(tri["inliers"][indices, :, h], device=device)
        init = torch.zeros((len(indices), 51), device=device)
        with torch.no_grad():
            _, neutral = layer(init[:, :48], b)
            init[:, 48:] = target[:, 0] - neutral[:, 0] / 1000.0
        p = torch.nn.Parameter(init)
        optimizer = torch.optim.Adam([p], lr=lr)
        # Smooth only adjacent valid portions, never bridge missing hand instances.
        groups = np.split(
            np.arange(len(indices)), np.flatnonzero(np.diff(indices) != 1) + 1
        )
        groups = [g for g in groups if len(g) >= 3]
        for step in range(steps):
            optimizer.zero_grad()
            _, joints = layer(p[:, :48], b, p[:, 48:])
            joints = joints / 1000.0
            homogeneous = torch.cat((joints, torch.ones_like(joints[..., :1])), dim=-1)
            q = torch.einsum("cij,fkj->fcki", M, homogeneous)
            pixels = q[..., :2] / q[..., 2:3].clamp_min(1e-5) / sizes
            l3 = loss3(joints[mask3].reshape(1, -1, 3), target[mask3].reshape(1, -1, 3))
            l2 = loss2(pixels, targets2, mask2 & (q[..., 2] > 1e-5))
            ls = sum((smooth([p[g]]) * len(g) for g in groups), p.sum() * 0) / max(
                1, len(indices)
            )
            loss = l3 + l2 + 0.001 * reg([p]) + 0.1 * ls
            if not torch.isfinite(loss):
                raise ValueError("Non-finite fit loss; refusing export")
            loss.backward()
            optimizer.step()
            losses[h, step] = loss.item()
            if step == 0 or (step + 1) % 100 == 0:
                print(f"Fit {side} {step + 1}/{steps}: {loss.item():.6f}", flush=True)
        with torch.no_grad():
            _, j = layer(p[:, :48], b, p[:, 48:])
            j = j.cpu().numpy() / 1000.0
            full = p[:, 3:48] @ layer.th_selected_comps + layer.th_hands_mean
            local[indices, h] = full.cpu().numpy()
            poses[indices, h] = p.cpu().numpy()
        uv, z = project(j, seq.matrices)
        e = np.linalg.norm(uv - obs["keypoints"][indices, :, h], axis=-1)
        for n, i in enumerate(indices):
            selected = tri["inliers"][i, :, h]
            for ci in range(len(seq.camera_ids)):
                seen = obs["valid"][i, ci, h]
                if seen.any():
                    camera_errors[i, ci, h] = np.median(e[n, ci, seen])
                if selected[ci].any():
                    camera_inlier_errors[i, ci, h] = np.median(e[n, ci, selected[ci]])
            errors[i, h] = np.median(e[n][selected]) if selected.any() else np.inf
            geometry_error[i, h] = np.linalg.norm(j[n] - tri["joints"][i, h], axis=-1)[
                tri["valid"][i, h]
            ].mean()
            valid[i, h] = (
                np.isfinite(poses[i, h]).all()
                and np.isfinite(e[n][selected]).all()
                and np.all(z[n][selected] > 1e-5)
                and errors[i, h] <= max_error_px
                and geometry_error[i, h] <= max_joint_error_m
            )
    common = dict(
        **seq.common(),
        observations_sha256=np.array(
            digest(seq.workdir / "predictions/observations_2d.npz")
        ),
        triangulation_sha256=np.array(
            digest(seq.workdir / "predictions/joints_3d.npz")
        ),
    )
    atomic_npz(
        path,
        **common,
        poses=poses,
        hand_pose=local,
        betas=beta,
        valid=valid,
        model_sha256=np.array(fingerprints),
    )
    atomic_npz(
        diag,
        **common,
        loss=losses,
        reprojection_median_px=errors,
        joint_error_m=geometry_error,
        camera_observation_median_px=camera_errors,
        camera_inlier_median_px=camera_inlier_errors,
        valid=valid,
    )
    log_config(
        seq,
        "fit",
        dict(
            device=device,
            steps=steps,
            lr=lr,
            min_joints=min_joints,
            max_error_px=max_error_px,
            max_joint_error_m=max_joint_error_m,
            shape="mean" if betas is None else "provided",
            depth_enabled=False,
            model_sha256=fingerprints,
        ),
    )
    return path


def export(seq, replace=False):
    path, meta_path = outputs(seq, ["mano.npz", "metadata.json"], replace)
    fitted = seq.load("fit_parameters.npz")
    for field, name in [
        ("observations_sha256", "observations_2d.npz"),
        ("triangulation_sha256", "joints_3d.npz"),
    ]:
        if str(fitted[field]) != digest(seq.workdir / "predictions" / name):
            raise ValueError("Fit is stale; rerun fit")
    n = len(seq.frame_ids)
    arrays = dict(
        frame_ids=np.array(seq.frame_ids),
        betas=fitted["betas"],
        hand_pose=fitted["hand_pose"],
        global_orient=fitted["poses"][..., :3],
        transl=fitted["poses"][..., 48:],
        valid=fitted["valid"],
    )
    for key, shape in [
        ("betas", (2, 10)),
        ("hand_pose", (n, 2, 45)),
        ("global_orient", (n, 2, 3)),
        ("transl", (n, 2, 3)),
    ]:
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError("Invalid fitted MANO arrays")
    if arrays["valid"].shape != (n, 2) or arrays["valid"].dtype != np.bool_:
        raise ValueError("Invalid hand validity")
    meta = dict(
        schema="mano_sequence_v1",
        stage="multiview_fitted",
        model="MANO_v1.2",
        hand_order=list(SIDES),
        rotation="axis_angle_radians",
        hand_pose_convention="absolute_flat_hand_mean",
        translation_convention="native_mano_origin",
        units="meter",
        axes="x_right_y_down_z_forward",
        coordinate_system="master_camera:" + seq.manifest["master_serial"],
        image_manifest_sha256=seq.fingerprint,
    )
    reconstruction = reconstruction_metadata(seq, seq.load("joints_3d.npz"))
    if reconstruction is not None:
        meta["reconstruction"] = reconstruction
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise ValueError("Export metadata changed; use a new workdir")
    atomic_json(meta_path, meta)
    atomic_npz(path, **arrays)
    print(
        f"Exported {n} frames, {int(arrays['valid'].sum())} valid hand instances -> {path}",
        flush=True,
    )
    return path
