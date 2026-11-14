"""Sequence selection and per-joint robustness at the geometry/pipeline boundaries."""

import unittest
import json
import tempfile
import numpy as np

from hocap_annotation.triangulation import robust_triangulate
from hocap_annotation.camera_consistency import check_cameras
from hocap_annotation.rgb_hand_pipeline import triangulate, export
from hocap_annotation.rgbd_data import atomic_npz, digest
from test_rgbd_pipeline import capture, observations


def geometry():
    k = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    centers = np.array(
        [
            [-0.3, 0, 0],
            [0.3, 0, 0],
            [0, -0.3, 0],
            [0, 0.3, 0],
            [-0.2, -0.2, 0],
            [0.2, 0.2, 0],
        ]
    )
    matrices = np.array([k @ np.c_[np.eye(3), -c] for c in centers])
    points = np.stack(
        [np.linspace(-0.08, 0.08, 21), np.linspace(-0.05, 0.05, 21), np.ones(21)],
        axis=-1,
    )
    q = np.einsum("cij,kj->cki", matrices, np.c_[points, np.ones(21)])
    return matrices, points, q[..., :2] / q[..., 2:]


class RobustGeometryTests(unittest.TestCase):
    def test_refinement_does_not_pull_consensus_toward_outlier(self):
        matrices, points, xy = geometry()
        xy[2] += [45, -130]
        valid = np.ones(xy.shape[:-1], bool)
        valid[1, 0] = False
        xyz, inliers = robust_triangulate(xy, valid, matrices, threshold=10, steps=100)
        np.testing.assert_allclose(xyz, points, atol=1e-4)
        self.assertFalse(inliers[2].any())
        self.assertFalse(inliers[1, 0])
        self.assertTrue(inliers[0].all())

    def test_parallel_rays_and_missing_views_are_invalid(self):
        matrices, _, xy = geometry()
        matrices[:] = matrices[0]
        xy[:] = xy[0]
        xyz, mask = robust_triangulate(xy, np.ones(xy.shape[:-1], bool), matrices)
        self.assertTrue(np.isnan(xyz).all())
        self.assertFalse(mask.any())

    def test_minimum_support_applies_per_joint(self):
        matrices, points, xy = geometry()
        seen = np.zeros(xy.shape[:-1], bool)
        seen[:2, 0] = True
        seen[0, 1] = True
        xyz, mask = robust_triangulate(xy, seen, matrices)
        np.testing.assert_allclose(xyz[0], points[0], atol=1e-5)
        self.assertTrue(np.isnan(xyz[1:]).all())
        self.assertFalse(mask[:, 1:].any())

    def test_one_refinement_step_evaluates_updated_parameters(self):
        matrices, _, xy = geometry()
        xy += np.random.default_rng(5).normal(0, 0.5, xy.shape)
        seen = np.ones(xy.shape[:-1], bool)
        initial, _ = robust_triangulate(xy, seen, matrices, steps=0)
        refined, _ = robust_triangulate(xy, seen, matrices, steps=1, lr=1e-5)
        self.assertGreater(np.max(np.abs(initial - refined)), 1e-7)

    def test_cuda_refinement_agrees_with_cpu_when_available(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        matrices, _, xy = geometry()
        xy += np.random.default_rng(5).normal(0, 0.5, xy.shape)
        xy[2] += [45, -130]
        seen = np.ones(xy.shape[:-1], bool)
        cpu, cpu_mask = robust_triangulate(xy, seen, matrices, steps=10, device="cpu")
        gpu, gpu_mask = robust_triangulate(xy, seen, matrices, steps=10, device="cuda")
        np.testing.assert_allclose(cpu, gpu, atol=1e-5)
        np.testing.assert_array_equal(cpu_mask, gpu_mask)


class CameraSelectionTests(unittest.TestCase):
    def test_persistent_camera_excluded_but_occasional_error_kept(self):
        matrices, _, xy = geometry()
        xy = np.tile(xy[None, :, None], (48, 1, 2, 1, 1))
        valid = np.ones(xy.shape[:-1], bool)
        xy[:, 2] += [45, -130]
        xy[10:13, 4] += [60, 60]
        ids = [f"cam{i:02d}" for i in range(6)]
        report = check_cameras(xy, valid, matrices, ids)
        self.assertEqual(
            report["excluded_cameras"],
            [{"camera_id": "cam02", "reason": "persistent_geometric_inconsistency"}],
        )
        self.assertIn("cam04", report["included_camera_ids"])
        self.assertEqual(report["cameras"]["cam02"]["evaluated_frames"], 48)
        # A short capture or missing observations is not evidence of bad calibration.
        short = check_cameras(xy[:3], valid[:3], matrices, ids)
        self.assertEqual(short["excluded_cameras"], [])
        self.assertEqual(short["cameras"]["cam02"]["status"], "insufficient_evidence")

    def test_missing_camera_is_not_excluded_and_no_check_keeps_outlier(self):
        matrices, _, xy = geometry()
        xy = np.tile(xy[None, :, None], (48, 1, 2, 1, 1))
        valid = np.ones(xy.shape[:-1], bool)
        valid[:, 2] = False
        ids = [f"cam{i:02d}" for i in range(6)]
        report = check_cameras(xy, valid, matrices, ids)
        self.assertEqual(report["excluded_cameras"], [])
        self.assertEqual(report["cameras"]["cam02"]["status"], "insufficient_evidence")
        xy[:, 2] += 100
        valid[:, 2] = True
        report = check_cameras(xy, valid, matrices, ids, automatic=False)
        self.assertEqual(report["included_camera_ids"], ids)

    def test_concentrated_evidence_and_required_camera_count_prevent_exclusion(self):
        matrices, _, xy = geometry()
        xy = np.tile(xy[None, :, None], (48, 1, 2, 1, 1))
        xy[:, 2] += [45, -130]
        valid = np.ones(xy.shape[:-1], bool)
        valid[16:, 2] = False
        ids = [f"cam{i:02d}" for i in range(6)]
        report = check_cameras(xy, valid, matrices, ids)
        self.assertEqual(report["excluded_cameras"], [])
        self.assertEqual(report["cameras"]["cam02"]["status"], "insufficient_evidence")
        valid[:, 2] = True
        report = check_cameras(xy, valid, matrices, ids, min_views=6)
        self.assertEqual(report["excluded_cameras"], [])
        self.assertEqual(report["cameras"]["cam02"]["status"], "ambiguous_retained")


class SelectionPipelineTests(unittest.TestCase):
    def test_automatic_exclusion_uses_full_sequence_and_preserves_final(self):
        with tempfile.TemporaryDirectory() as root:
            _, seq = capture(root, 48)
            _, points, _ = geometry()
            path = observations(seq, points)
            obs = seq.load("observations_2d.npz")
            obs["keypoints"][:, 2, 0] += [45, -130]
            atomic_npz(path, **obs)
            final = seq.workdir / "final"
            final.mkdir()
            (final / "untouched").write_text("human result")
            triangulate(seq, steps=0, threshold=1)
            tri = seq.load("joints_3d.npz")
            self.assertFalse(tri["inliers"][:, 2].any())
            self.assertTrue(tri["valid"][:, 0].all())
            report = json.loads(
                (seq.workdir / "predictions/camera_consistency.json").read_text()
            )
            self.assertEqual(report["excluded_cameras"][0]["camera_id"], "cam02")
            self.assertEqual((final / "untouched").read_text(), "human result")

    def test_exclusion_reaches_inliers_and_export_without_removing_camera(self):
        with tempfile.TemporaryDirectory() as root:
            _, seq = capture(root, 1)
            _, points, _ = geometry()
            observations(seq, points)
            triangulate(seq, steps=2, exclude_cameras=["cam02"], camera_check=False)
            tri = seq.load("joints_3d.npz")
            self.assertFalse(tri["inliers"][:, 2].any())
            self.assertTrue(tri["valid"][:, 0].all())
            self.assertEqual(tri["inliers"].shape, (1, 6, 2, 21))
            directory = seq.workdir / "predictions"
            # An exported fit fixture exercises the actual provenance checks.
            atomic_npz(
                directory / "fit_parameters.npz",
                **seq.common(),
                observations_sha256=np.array(digest(directory / "observations_2d.npz")),
                triangulation_sha256=np.array(digest(directory / "joints_3d.npz")),
                poses=np.zeros((1, 2, 51)),
                hand_pose=np.zeros((1, 2, 45)),
                betas=np.zeros((2, 10)),
                valid=np.ones((1, 2), bool),
            )
            export(seq)
            meta = json.loads((directory / "metadata.json").read_text())
            self.assertNotIn("cam02", meta["reconstruction"]["included_camera_ids"])
            self.assertEqual(
                meta["reconstruction"]["excluded_cameras"],
                [{"camera_id": "cam02", "reason": "manual_exclusion"}],
            )
            self.assertEqual(len(seq.manifest["cameras"]), 6)
            old_meta = (directory / "metadata.json").read_bytes()
            old_mano = (directory / "mano.npz").read_bytes()
            triangulate(
                seq,
                steps=0,
                exclude_cameras=["cam03"],
                camera_check=False,
                replace=True,
            )
            fitted = seq.load("fit_parameters.npz")
            fitted["triangulation_sha256"] = np.array(
                digest(directory / "joints_3d.npz")
            )
            atomic_npz(directory / "fit_parameters.npz", **fitted)
            with self.assertRaisesRegex(ValueError, "metadata changed"):
                export(seq, replace=True)
            self.assertEqual((directory / "metadata.json").read_bytes(), old_meta)
            self.assertEqual((directory / "mano.npz").read_bytes(), old_mano)
            report = directory / "camera_consistency.json"
            report.write_text(report.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "stale"):
                export(seq, replace=True)

    def test_unknown_or_excessive_manual_exclusions_fail(self):
        with tempfile.TemporaryDirectory() as root:
            _, seq = capture(root, 1)
            _, points, _ = geometry()
            observations(seq, points)
            for excluded in [["typo"], seq.camera_ids[:5]]:
                with self.assertRaises(ValueError):
                    triangulate(seq, steps=0, exclude_cameras=excluded)
            self.assertFalse((seq.workdir / "predictions/joints_3d.npz").exists())


if __name__ == "__main__":
    unittest.main()
