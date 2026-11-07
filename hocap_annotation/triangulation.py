"""HO-Cap pairwise DLT + reprojection refinement, without capture/render imports.

Extracted from tools/03_mp_3d_joints_generation.py; device and minimum views
are configurable. Invalid points retain the legacy -1 sentinel internally.
"""

import itertools
import numpy as np
import torch


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
