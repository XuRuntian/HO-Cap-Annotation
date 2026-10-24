import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from hocap_annotation.loss.keypoint_3d_loss import Keypoint3DLoss
from hocap_annotation.loss.pose_smoothness_loss import PoseSmoothnessLoss
from hocap_annotation.rendering.hocap_renderer import HOCapRenderer
from hocap_annotation.utils.hand_observations import prepare_hand_initialization


spec = importlib.util.spec_from_file_location(
    'mano_missing_solver', Path(__file__).resolve().parents[1] / 'tools/05_mano_pose_solver.py'
)
solver_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver_module)


def joints_fixture(frames=9):
    joints = np.zeros((2, frames, 21, 3), dtype=np.float32)
    joints[..., 2] = 1.
    joints[:, :, :, 0] = np.arange(frames)[None, :, None] * .01
    return joints


class ObservationTests(unittest.TestCase):
    def test_short_gap_seeds_do_not_become_observations(self):
        joints = joints_fixture()
        joints[0, 3:5] = -1
        original = joints.copy()
        seeds, valid, observed, source = prepare_hand_initialization(joints, np.full((2, 9), 3))
        np.testing.assert_allclose(seeds[0, 3:5], [[.03, 0, 1], [.04, 0, 1]])
        self.assertTrue(valid.all())
        self.assertFalse(observed[0, 3:5].any())
        np.testing.assert_array_equal(source[0, 3:5], [2, 2])
        np.testing.assert_array_equal(joints, original)

    def test_long_or_unsupported_gaps_are_not_filled(self):
        joints = joints_fixture()
        joints[0, 1:7] = -1
        joints[1, 3:5] = -1
        views = np.full((2, 9), 3)
        views[1, 4] = 0
        seeds, valid, _, source = prepare_hand_initialization(joints, views)
        self.assertFalse(valid[0, 1:7].any())
        self.assertFalse(valid[1, 3:5].any())
        self.assertFalse(source[0, 1:7].any())
        self.assertTrue(np.isfinite(seeds).all())

    def test_edge_hold_requires_support_and_an_anchor(self):
        joints = joints_fixture()
        joints[0, :2] = -1
        joints[1] = -1
        seeds, valid, observed, source = prepare_hand_initialization(joints, np.full((2, 9), 3))
        np.testing.assert_allclose(seeds[0, :2], [[.02, 0, 1]]*2)
        np.testing.assert_array_equal(source[0, :2], [3, 3])
        self.assertFalse(observed[0, :2].any())
        self.assertFalse(valid[1].any())

    def test_zero_gap_limit_disables_initialization_fill(self):
        joints = joints_fixture()
        joints[0, 4] = np.nan
        _, valid, observed, _ = prepare_hand_initialization(joints, np.full((2, 9), 3), max_gap=0)
        self.assertFalse(valid[0, 4])
        self.assertFalse(observed[0, 4].any())


class MaskedLossTests(unittest.TestCase):
    def test_missing_only_loss_and_gradient_are_zero(self):
        pred = torch.zeros((1, 21, 3), requires_grad=True)
        loss = Keypoint3DLoss()(pred, torch.full_like(pred, -1))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(torch.equal(pred.grad, torch.zeros_like(pred)))

    def test_nan_inf_and_missing_targets_do_not_dilute_valid_loss(self):
        pred = torch.zeros((1, 4, 3), requires_grad=True)
        target = torch.tensor([[[1., 0, 0], [-1, -1, -1], [float('nan'), 0, 0], [0, float('inf'), 0]]])
        loss = Keypoint3DLoss()(pred, target)
        self.assertAlmostEqual(loss.item(), 1.)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertEqual(torch.count_nonzero(pred.grad[:, 1:]).item(), 0)

    def test_explicit_mask_and_joint_selection_work_together(self):
        pred = torch.zeros((1, 3, 3), requires_grad=True)
        target = torch.ones_like(pred)
        mask = torch.tensor([[True, False, False]])
        loss = Keypoint3DLoss()(pred, target, [1, 2], valid_mask=mask)
        self.assertEqual(loss.item(), 0.)

    def test_fully_observed_loss_retains_original_scale(self):
        pred, target = torch.zeros((2, 3, 3)), torch.ones((2, 3, 3))
        self.assertAlmostEqual(Keypoint3DLoss()(pred, target).item(), 3**.5, places=6)


class TemporalBoundaryTests(unittest.TestCase):
    def test_smoothing_does_not_bridge_a_missing_hand(self):
        poses = torch.zeros((7, 51), requires_grad=True)
        with torch.no_grad():
            poses[4:] = 10
        valid = torch.tensor([[True, True, True, False, True, True, True]])
        loss = PoseSmoothnessLoss(win_size=2)([poses], valid_mask=valid)
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(poses.grad).all())
        self.assertEqual(torch.count_nonzero(poses.grad).item(), 0)

    def test_translation_penalty_uses_only_last_three_coordinates(self):
        poses = torch.zeros((3, 51))
        poses[1, 3] = 5.
        loss = PoseSmoothnessLoss(w_vel_r=0, w_vel_t=1, w_acc_r=0, w_acc_t=1)([poses])
        self.assertEqual(loss.item(), 0.)

    def test_single_frame_has_finite_zero_temporal_loss(self):
        loss = PoseSmoothnessLoss()([torch.zeros((1, 51))])
        self.assertEqual(loss.item(), 0.)


class SolverIntegrationTests(unittest.TestCase):
    def solver(self, frames=9):
        solver = solver_module.ManoPoseSolver.__new__(solver_module.ManoPoseSolver)
        solver._device = 'cpu'
        solver._mano_sides = ['right', 'left']
        solver._num_frames = frames
        solver._mano_group_layer = SimpleNamespace(num_obj=2)
        solver._w_kpt_2d = 1
        solver._max_init_gap = 5
        solver._valid_mask_m = torch.ones((frames, 3, 42), dtype=torch.bool)
        solver._logger = Mock()
        return solver

    def test_real_initializer_never_uses_negative_one_as_translation(self):
        solver = self.solver()
        joints = joints_fixture()
        joints[0, 3:5] = -1
        joints[1, 1:7] = -1
        poses = solver._initialize_pose_m_from_3d_joints(joints)
        np.testing.assert_allclose(poses[0].detach()[3:5, -3:], [[.03, 0, 1], [.04, 0, 1]])
        self.assertFalse(solver._observed_3d[0, 3:5].any())
        self.assertFalse(solver._fit_valid[1, 1:7].any())
        self.assertFalse(solver._valid_mask_m[1:7, :, 21:].any())
        self.assertTrue(all(torch.isfinite(p).all() for p in poses))
        faces = torch.tensor([[0, 1, 2], [778, 779, 780]])
        self.assertEqual(solver._faces_for_frame(faces, 3).tolist(), [[0, 1, 2]])

    def test_loading_does_not_overwrite_frame_one(self):
        solver = self.solver(2)
        joints = joints_fixture(2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'joints.npy'
            np.save(path, joints)
            loaded = solver._load_hand_3d_joints(path)
        np.testing.assert_array_equal(loaded, joints)

    def test_invalid_export_and_validity_sidecars(self):
        solver = self.solver()
        joints = joints_fixture()
        joints[1, 1:7] = -1
        solver._pose_m = solver._initialize_pose_m_from_3d_joints(joints)
        with tempfile.TemporaryDirectory() as tmp:
            solver._save_folder = Path(tmp)
            solver._save_optimized_poses_m()
            saved = np.load(Path(tmp)/'poses_m.npy')
            valid = np.load(Path(tmp)/'poses_m_valid.npy')
            self.assertTrue(np.all(saved[1, 1:7] == -1))
            self.assertFalse(valid[1, 1:7].any())
            metadata = np.load(Path(tmp)/'fit_observations.npz')
            self.assertFalse(metadata['observed_3d'][1, 1:7].any())

    def test_explicit_output_directory_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                solver_module.ManoPoseSolver('unused', output_folder=tmp)

    def test_renderer_omits_invalid_hand_and_remaps_faces(self):
        renderer = HOCapRenderer.__new__(HOCapRenderer)
        renderer.render_dict = dict(mano_verts=np.zeros((1, 1556, 3)),
            mano_faces=np.array([[0, 1, 2], [778, 779, 780]]),
            mano_colors=['right', 'left'], mano_validity=np.array([[False, True]]))
        vertices, faces, colors = renderer._get_mano_geometry(0)
        self.assertEqual(vertices.shape, (778, 3))
        self.assertEqual(faces.tolist(), [[0, 1, 2]])
        self.assertEqual(colors, ['left'])
        renderer.render_dict['mano_validity'][:] = False
        self.assertIsNone(renderer._get_mano_geometry(0)[0])


if __name__ == '__main__':
    unittest.main()
