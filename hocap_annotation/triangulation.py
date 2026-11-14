"""HO-Cap pairwise DLT + reprojection refinement, without capture/render imports.

Extracted from tools/03_mp_3d_joints_generation.py; device and minimum views
are configurable. Invalid points retain the legacy -1 sentinel internally.
"""

import itertools
import numpy as np
import torch


def projection_errors(points, xy, matrices):
    q = np.einsum("cij,nj->cni", matrices, np.c_[points, np.ones(len(points))])
    with np.errstate(divide="ignore", invalid="ignore"):
        errors = np.linalg.norm(q[..., :2] / q[..., 2:] - xy, axis=-1)
    return np.where((q[..., 2] > 1e-5) & np.isfinite(errors), errors, np.inf)


def _wide_baseline(points, centers, a, b, min_angle):
    u, v = points - centers[a], points - centers[b]
    denominator = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = np.sum(u * v, axis=-1) / np.maximum(denominator, 1e-15)
    # Both nearly parallel and nearly antiparallel rays are ill-conditioned.
    return (denominator > 1e-15) & (np.abs(cosine) <= np.cos(np.deg2rad(min_angle)))


def robust_triangulate(
    xy,
    observed,
    matrices,
    threshold=10.0,
    min_views=2,
    min_angle=1.0,
    steps=0,
    device="cpu",
    lr=1e-3,
):
    """Pairwise consensus first, then refine only its inliers.

    Inputs are [camera, point, 2] and [camera, point]. Invalid output points
    are NaN; inliers keep the full camera axis, including excluded views.
    """
    xy = np.asarray(xy, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool) & np.isfinite(xy).all(-1)
    matrices = np.asarray(matrices, dtype=np.float64)
    c, n, _ = xy.shape
    if (
        observed.shape != (c, n)
        or matrices.shape != (c, 3, 4)
        or not np.isfinite(matrices).all()
        or not np.isfinite(threshold)
        or threshold <= 0
        or not 2 <= min_views <= c
        or not 0 < min_angle < 90
        or steps < 0
        or not np.isfinite(lr)
        or lr <= 0
    ):
        raise ValueError("Invalid robust triangulation inputs")
    xy = np.where(observed[..., None], xy, 0)
    centers = -np.linalg.solve(matrices[:, :, :3], matrices[:, :, 3])
    best = np.full((n, 3), np.nan)
    best_count = np.zeros(n, int)
    best_error = np.full(n, np.inf)
    for a, b in itertools.combinations(range(c), 2):
        A = np.stack(
            [
                xy[a, :, 0, None] * matrices[a, 2] - matrices[a, 0],
                xy[a, :, 1, None] * matrices[a, 2] - matrices[a, 1],
                xy[b, :, 0, None] * matrices[b, 2] - matrices[b, 0],
                xy[b, :, 1, None] * matrices[b, 2] - matrices[b, 1],
            ],
            axis=1,
        )
        _, _, vh = np.linalg.svd(A)
        with np.errstate(divide="ignore", invalid="ignore"):
            candidate = vh[:, -1, :3] / vh[:, -1, 3:]
        errors = projection_errors(candidate, xy, matrices)
        inliers = observed & (errors < threshold)
        supported = (
            inliers[a]
            & inliers[b]
            & _wide_baseline(candidate, centers, a, b, min_angle)
        )
        count = inliers.sum(0)
        cost = np.where(inliers, errors, 0).sum(0) / np.maximum(count, 1)
        better = (
            supported
            & (count >= min_views)
            & ((count > best_count) | ((count == best_count) & (cost < best_error)))
        )
        best[better], best_count[better], best_error[better] = (
            candidate[better],
            count[better],
            cost[better],
        )
    mask = observed & (projection_errors(best, xy, matrices) < threshold)
    usable = mask.sum(0) >= min_views
    if steps and usable.any():
        # Keep the best iterate per point; refinement must not destroy consensus.
        initial = best[usable].copy()
        p = torch.tensor(
            initial, dtype=torch.float64, device=device, requires_grad=True
        )
        target = torch.tensor(xy[:, usable], dtype=p.dtype, device=device)
        selected = torch.tensor(mask[:, usable], device=device)
        M = torch.tensor(matrices, dtype=p.dtype, device=device)
        optimizer = torch.optim.Adam([p], lr=lr)
        kept = p.detach().clone()
        kept_cost = torch.tensor(best_error[usable], dtype=p.dtype, device=device)
        for step in range(steps + 1):
            optimizer.zero_grad()
            q = torch.einsum(
                "cij,nj->cni", M, torch.cat([p, torch.ones_like(p[:, :1])], -1)
            )
            pixels = q[..., :2] / q[..., 2:].clamp_min(1e-5)
            errors = torch.linalg.vector_norm(pixels - target, dim=-1)
            cost = torch.where(selected, errors, 0).sum(0) / selected.sum(0)
            with torch.no_grad():
                supported = (
                    (~selected) | ((errors < threshold) & (q[..., 2] > 1e-5))
                ).all(0)
                better = supported & torch.isfinite(cost) & (cost <= kept_cost)
                kept[better], kept_cost[better] = p[better], cost[better]
            if step == steps:
                break
            cost.mean().backward()
            optimizer.step()
        best[usable] = kept.cpu().numpy()
    mask = observed & (projection_errors(best, xy, matrices) < threshold)
    wide = np.zeros(n, bool)
    for a, b in itertools.combinations(range(c), 2):
        wide |= mask[a] & mask[b] & _wide_baseline(best, centers, a, b, min_angle)
    usable = (mask.sum(0) >= min_views) & wide
    best[~usable] = np.nan
    mask[:, ~usable] = False
    return best.astype(np.float32), mask


def runner_hand_3d_joints_ransac_torch(
    handmarks, Ms, threshold=10.0, iterations=100, lr=1e-3, device="cuda", min_views=2
):
    def triangulate_handmarks_batch(p1, p2, M1, M2):
        """
        Triangulate handmarks from two camera views using PyTorch for batch processing.

        Parameters:
        - p1: Batch of points from the first camera, torch tensor of shape (B, 21, 2).
        - p2: Batch of points from the second camera, torch tensor of shape (B, 21, 2).
        - M1, M2: Corresponding camera projection matrices, torch tensors of shape (B, 3, 4).

        Returns:
        - A torch tensor of shape (B, 21, 3) containing the triangulated 3D handmarks.
        """
        B, N, _ = p1.shape  # B: Batch size, N: Number of handmarks (21)
        X = torch.zeros((B, N, 3), dtype=torch.float32, device=p1.device)

        for i in range(N):  # Process each handmark
            A = torch.zeros((B, 4, 4), dtype=torch.float32, device=p1.device)
            A[:, 0, :] = p1[:, i, 0].unsqueeze(1) * M1[:, 2, :] - M1[:, 0, :]
            A[:, 1, :] = p1[:, i, 1].unsqueeze(1) * M1[:, 2, :] - M1[:, 1, :]
            A[:, 2, :] = p2[:, i, 0].unsqueeze(1) * M2[:, 2, :] - M2[:, 0, :]
            A[:, 3, :] = p2[:, i, 1].unsqueeze(1) * M2[:, 2, :] - M2[:, 1, :]

            # Perform SVD on A
            U, S, Vh = torch.linalg.svd(A, full_matrices=False)
            # Extract the solution from Vh, normalize by the last element
            X[:, i, :] = Vh[:, -1, :3] / Vh[:, -1, 3].unsqueeze(1)

        return X

    def project_3d_to_2d_parallel(p_3d, Ms):
        """
        Project a batch of 3D points to 2D across multiple camera views in parallel.

        Parameters:
        - p_3d: A torch tensor of 3D points of shape (N, 3).
        - Ms: Camera projection matrices, torch tensor of shape (C, 3, 4).

        Returns:
        - projected_2d: Projected 2D points, torch tensor of shape (C, N, 2).
        """
        C = Ms.shape[0]
        N = p_3d.shape[0]
        ones = torch.ones((N, 1), device=p_3d.device, dtype=p_3d.dtype)
        p_3d_hom = (
            torch.cat((p_3d, ones), dim=1).unsqueeze(0).repeat(C, 1, 1)
        )  # Shape: (C, N, 4)
        p_2d_hom = torch.einsum("cij,ckj->cki", Ms, p_3d_hom)  # Shape: (C, N, 3)
        p_2d = p_2d_hom[:, :, :2] / p_2d_hom[:, :, [2]]  # Normalize
        return p_2d

    def parallel_reprojection_loss(p_3d, uv_coords, Ms):
        """
        Compute the reprojection loss for all 21 joints across all camera views in parallel.

        Parameters:
        - p_3d: A torch tensor of 3D points of shape (N, 3).
        - uv_coords: Observed 2D points for all cameras, torch tensor of shape (C, N, 2).
        - Ms: Camera projection matrices, torch tensor of shape (C, 3, 4).

        Returns:
        - Total reprojection loss, a single scalar value.
        """
        projected_2d = project_3d_to_2d_parallel(p_3d, Ms)
        loss = torch.norm(
            projected_2d - uv_coords, dim=-1
        )  # Compute L2 loss for each camera

        loss = loss.sum()
        loss /= uv_coords.size(0) * p_3d.size(0)  # Normalize by number of cameras
        return loss

    def optimize_all_joints(pts_3d, uv_coords, Ms, lr=1e-3, steps=100):
        """
        Optimize the 3D coordinates of all 21 MANO joints in parallel.

        Parameters:
        - pts_3d: Batch of initial 3D coordinates for all joints, torch tensor of shape (B, N, 3).
        - uv_coords: Observed 2D points for all cameras, torch tensor of shape (C, N, 2).
        - Ms: Camera projection matrices, torch tensor of shape (C, 3, 4).
        - lr: Learning rate for the optimizer.
        - steps: Number of optimization steps.

        Returns:
        - Optimized 3D coordinates for all 21 joints, torch tensor of shape (N, 3).
        """
        B, N, _ = pts_3d.shape
        X = torch.zeros(
            (B, 3),
            device=pts_3d.device,
            dtype=torch.float32,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([X], lr=lr)
        opt_pts = []

        for i in range(N):
            X.data = pts_3d[:, i].clone()
            optimizer = torch.optim.Adam([X], lr=lr)
            _uv_coords = uv_coords[:, i].unsqueeze(1).repeat(1, B, 1)
            for _ in range(steps):
                optimizer.zero_grad()
                loss = parallel_reprojection_loss(X, _uv_coords, Ms)
                loss.backward()
                optimizer.step()
            opt_pts.append(X.detach().clone())

        opt_pts = torch.stack(opt_pts)
        return opt_pts

    Ms_ts = torch.from_numpy(Ms).to(dtype=torch.float32, device=device)
    handmarks = torch.from_numpy(handmarks).to(dtype=torch.float32, device=device)
    hands, num_cams, joints, _ = handmarks.shape

    hand_joints_3d = torch.full((hands, joints, 3), -1, dtype=torch.float32)
    for hand in range(hands):
        marks = handmarks[hand]  # shape (num_cameras, num_joints, 2)
        if torch.all(marks == -1):  # no hand detected
            continue
        valid_views = torch.where(torch.all(marks[..., 0] != -1, dim=1))
        num_valid_views = len(valid_views[0])
        if num_valid_views < min_views:
            continue

        # create candidate 3D points by triangulating 2D points from valid views
        combinations = torch.tensor(
            list(itertools.combinations(valid_views[0].cpu().numpy(), 2)),
            device=marks.device,
        )
        b_pts_3d = triangulate_handmarks_batch(
            marks[combinations[:, 0]],
            marks[combinations[:, 1]],
            Ms_ts[combinations[:, 0]],
            Ms_ts[combinations[:, 1]],
        )

        # optimize each 3D point to minimize reprojection error
        pts_3d_optim = optimize_all_joints(
            b_pts_3d,
            marks[valid_views],
            Ms_ts[valid_views],
            lr=lr,
            steps=iterations,
        )

        best_points = torch.full((joints, 3), -1, dtype=torch.float32)
        for joint in range(joints):
            best_loss = 0.0
            best_point = None
            best_inliers = 0
            for i in range(len(pts_3d_optim[joint])):
                projected_2d = project_3d_to_2d_parallel(
                    pts_3d_optim[joint, i].unsqueeze(0),
                    Ms_ts[valid_views],
                )
                loss = torch.norm(
                    projected_2d - marks[valid_views][:, joint].unsqueeze(1), dim=-1
                )
                point_h = torch.cat(
                    (pts_3d_optim[joint, i], torch.ones(1, device=marks.device))
                )
                positive = (Ms_ts[valid_views] @ point_h)[:, 2] > 1e-5
                valid_mask = torch.where(
                    (loss[:, 0] < threshold) & positive & torch.isfinite(loss[:, 0])
                )[0]
                inliers = len(valid_mask)
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_point = pts_3d_optim[joint, i]
                    best_loss = loss[valid_mask].mean().item()
                if (
                    best_inliers > 0
                    and inliers == best_inliers
                    and loss[valid_mask].mean().item() < best_loss
                ):
                    best_inliers = inliers
                    best_point = pts_3d_optim[joint, i]
                    best_loss = loss[valid_mask].mean().item()
            if best_inliers >= min_views:
                best_points[joint] = best_point.cpu()
        hand_joints_3d[hand] = best_points

    hand_joints_3d = hand_joints_3d.numpy().astype(np.float32)

    return hand_joints_3d
