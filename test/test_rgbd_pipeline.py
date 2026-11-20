"""RGB-only contracts and geometry tests; no capture/segmentation/GPU dependencies."""

import copy
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from hocap_annotation.rgbd_data import (
    prepare,
    PreparedSequence,
    CONVENTION,
    atomic_json,
    atomic_npz,
    digest,
    rectification_maps,
)
from hocap_annotation.rgb_hand_pipeline import detect, triangulate, fit, export, project


def capture(root, count=3):
    data = Path(root) / "capture"
    rgbd = data / "rgbd"
    rgbd.mkdir(parents=True)
    intr = dict(
        width=80,
        height=64,
        fx=60.0,
        fy=60.0,
        ppx=40.0,
        ppy=32.0,
        model="distortion.none",
        coeffs=[0.0] * 5,
    )
    transforms = {}
    rows = [dict(frame_id=str(i * 2), cameras={}) for i in range(count)]
    for ci in range(6):
        cid = f"cam{ci:02d}"
        serial = str(ci)
        cam = rgbd / cid
        cam.mkdir()
        transform = np.eye(4)
        transform[0, 3] = ci * 0.03
        transforms[serial] = transform.tolist()
        atomic_json(
            cam / "calibration.json",
            dict(
                serial=serial,
                id=cid,
                original_calibration={"intrinsics": {"color": intr}},
            ),
        )
        for i in range(count):
            filename = f"{cid}/color_{i:06d}.png"
            cv2.imwrite(str(rgbd / filename), np.zeros((64, 80, 3), np.uint8))
            rows[i]["cameras"][cid] = dict(
                serial=serial,
                color_path=filename,
                color_source={"timestamp_ms": 1000 + i * 33.3},
            )
    atomic_json(
        rgbd / "calibration/extrinsics.json",
        dict(
            master_serial="0",
            translation_unit="meter",
            convention=CONVENTION,
            T_master_from_camera=transforms,
        ),
    )
    (rgbd / "frames.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    work = Path(root) / "work"
    prepare(data, work)
    return data, PreparedSequence(work)


def observations(seq, points):
    uv, _ = project(points, seq.matrices)
    n = len(seq.frame_ids)
    xy = np.zeros((n, 6, 2, 21, 2), np.float32)
    valid = np.zeros((n, 6, 2, 21), bool)
    xy[:, :, 0] = uv
    valid[:, :, 0] = True
    path = seq.workdir / "predictions/observations_2d.npz"
    atomic_npz(path, **seq.common(), keypoints=xy, valid=valid)
    return path


class PipelineTests(unittest.TestCase):
    def test_fah_observation_contract(self):
        from hocap_annotation.wrappers.fah_detector import _infer

        cameras = {
            "a": {"intrinsics": dict(fx=70, fy=71, ppx=40, ppy=32)},
            "b": {"intrinsics": dict(fx=80, fy=81, ppx=41, ppy=33)},
        }
        rgb = np.tile(np.array([17, 29, 43], np.uint8), (64, 80, 1))
        seq = SimpleNamespace(
            frame_ids=["10", "20"],
            camera_ids=list(cameras),
            manifest={"cameras": cameras},
            image=lambda i, cid: rgb,
        )
        calls = []

        def model(bgr, *, K):
            np.testing.assert_array_equal(bgr[0, 0], [43, 29, 17])
            calls.append(K.copy())
            if len(calls) != 1:
                return []
            winner = np.full((21, 2), 20.0)
            winner[1, 0] = np.nan
            return [
                {"is_right": False, "score": 0.9, "joints2d": np.full((21, 2), 30)},
                {"is_right": True, "score": 0.1, "joints2d": np.full((21, 2), 10)},
                {"is_right": True, "score": 0.7, "joints2d": winner},
            ]

        obs = _infer(seq, model)
        self.assertEqual(obs["keypoints"].shape, (2, 2, 2, 21, 2))
        np.testing.assert_array_equal(calls[0], [[70, 0, 40], [0, 71, 32], [0, 0, 1]])
        np.testing.assert_array_equal(calls[2], [[80, 0, 41], [0, 81, 33], [0, 0, 1]])
        np.testing.assert_array_equal(obs["keypoints"][0, 0, :, 0], [[20, 20], [30, 30]])
        np.testing.assert_array_equal(obs["keypoints"][0, 0, 0, 1], [0, 0])
        self.assertFalse(obs["valid"][0, 0, 0, 1])
        self.assertFalse(obs["valid"][1].any())
        self.assertFalse(obs["valid"][:, 1].any())
        self.assertTrue(obs["duplicate_side"][0, 0, 0])
        self.assertAlmostEqual(float(obs["detection_score"][0, 0, 0]), 0.7)
        self.assertNotIn("handedness_score", obs)

    def test_prepare_rectified_contract_and_source_immutable(self):
        with tempfile.TemporaryDirectory() as root:
            data, seq = capture(root)
            self.assertEqual(seq.frame_ids, ["0", "2", "4"])
            self.assertEqual(seq.image(0, "cam00").shape, (64, 80, 3))
            self.assertEqual(len(seq.camera_ids), 6)
            before = {
                str(p): digest(p) for p in (data / "rgbd").rglob("*") if p.is_file()
            }
            with self.assertRaisesRegex(ValueError, "already exists"):
                prepare(data, seq.workdir)
            (seq.workdir / "final").mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                prepare(data, seq.workdir, replace=True)
            self.assertEqual(
                before,
                {str(p): digest(p) for p in (data / "rgbd").rglob("*") if p.is_file()},
            )
            with self.assertRaises(ValueError):
                prepare(data, data / "rgbd" / "derived")
            cam = seq.manifest["cameras"]["cam00"]["intrinsics"]
            mx, my = rectification_maps(cam)
            yy, xx = np.mgrid[:64, :80]
            np.testing.assert_allclose(mx, xx, atol=1e-6)
            np.testing.assert_allclose(my, yy, atol=1e-6)

    def test_detection_duplicate_side_not_relabelled(self):
        with tempfile.TemporaryDirectory() as root:
            _, seq = capture(root, 1)
            model = Path(root) / "fake.task"
            model.write_text("fixture")
            module = ModuleType("hocap_annotation.wrappers.mediapipe")

            class Detector:
                def __init__(self, c):
                    pass

                def detect_one(self, image):
                    return (
                        [np.ones((21, 2)) * 0.25, np.ones((21, 2)) * 0.75],
                        ["right", "right"],
                        [0.3, 0.9],
                    )

                def close(self):
                    pass

            module.MPHandDetector = Detector
            with patch.dict(
                sys.modules, {"hocap_annotation.wrappers.mediapipe": module}
            ):
                detect(seq, model)
            obs = seq.load("observations_2d.npz")
            self.assertTrue(obs["valid"][:, :, 0].all())
            self.assertFalse(obs["valid"][:, :, 1].any())
            self.assertTrue(obs["duplicate_side"][:, :, 0].all())
            np.testing.assert_allclose(obs["keypoints"][:, :, 0], 0.75)

    def test_cpu_triangulation_outlier_and_stale_inputs(self):
        with tempfile.TemporaryDirectory() as root:
            _, seq = capture(root, 1)
            points = np.stack(
                (np.linspace(0, 0.1, 21), np.linspace(-0.05, 0.05, 21), np.ones(21)),
                axis=-1,
            )
            path = observations(seq, points)
            obs = seq.load("observations_2d.npz")
            obs["keypoints"][:, 0, 0] += 30
            atomic_npz(path, **obs)
            with patch.dict(sys.modules, {"meshsdf_loss_cuda": None}):
                triangulate(seq, device="cpu", steps=0, threshold=1.0)
            data = seq.load("joints_3d.npz")
            self.assertTrue(data["valid"][:, 0].all())
            self.assertFalse(data["valid"][:, 1].any())
            np.testing.assert_allclose(data["joints"][0, 0], points, atol=2e-5)
            self.assertFalse(data["inliers"][:, 0, 0].any())
            with self.assertRaisesRegex(ValueError, "Stage output exists"):
                triangulate(seq, steps=0)
            obs["keypoints"][0, 0, 0, 0, 0] += 1
            atomic_npz(path, **obs)
            with self.assertRaisesRegex(ValueError, "stale"):
                fit(seq, "unused", device="cpu", steps=1)

    def test_dependency_isolation_and_short_smoothing(self):
        with patch.dict(
            sys.modules, {"meshsdf_loss_cuda": None, "pyrender": None, "open3d": None}
        ):
            import torch
            from hocap_annotation.loss import Keypoint2DLoss, PoseSmoothnessLoss

            loss = PoseSmoothnessLoss(w_vel_r=0, w_vel_t=1, w_acc_r=0, w_acc_t=0)
            p = torch.zeros((2, 51), requires_grad=True)
            self.assertTrue(torch.isfinite(loss([p])))
            p = torch.zeros((3, 51))
            p[:, 5] = torch.tensor([0.0, 1.0, 2.0])
            self.assertEqual(float(loss([p])), 0.0)
            p[:, 48] = torch.tensor([0.0, 1.0, 2.0])
            self.assertGreater(float(loss([p])), 0.0)

    @unittest.skipUnless(
        Path("/home/xrt/models/mano_v1_2/models/MANO_RIGHT.pkl").exists(),
        "Licensed MANO assets required",
    )
    def test_cpu_fit_export_and_consumer_contract(self):
        import torch
        from manopth.manolayer import ManoLayer

        models = Path("/home/xrt/models/mano_v1_2/models")
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as root:
            _, seq = capture(root, 3)
            layer = ManoLayer(
                mano_root=str(models),
                side="right",
                use_pca=True,
                ncomps=45,
                flat_hand_mean=False,
                center_idx=None,
            )
            with torch.no_grad():
                _, j = layer(
                    torch.zeros(1, 48),
                    torch.zeros(1, 10),
                    torch.tensor([[0.0, 0.0, 0.8]]),
                )
            points = j[0].numpy() / 1000.0
            observations(seq, points)
            triangulate(seq, device="cpu", steps=0, threshold=1.0)
            with patch.dict(sys.modules, {"meshsdf_loss_cuda": None}):
                fit(seq, models, device="cpu", steps=2)
            out = export(seq)
            with np.load(out) as result:
                self.assertEqual(
                    set(result.files),
                    {
                        "frame_ids",
                        "betas",
                        "hand_pose",
                        "global_orient",
                        "transl",
                        "valid",
                    },
                )
                self.assertEqual(result["hand_pose"].shape, (3, 2, 45))
                self.assertFalse(result["valid"][:, 1].any())
                self.assertTrue(result["valid"][:, 0].all())
            meta = json.loads(out.with_name("metadata.json").read_text())
            self.assertEqual(meta["image_manifest_sha256"], digest(seq.path))
            self.assertFalse((seq.workdir / "final").exists())
            # Optional integration with the actual independent annotation consumer.
            consumer = Path("/home/xrt/3DHandsForAll")
            if (consumer / "sequence_store.py").exists():
                sys.path.insert(0, str(consumer))
                try:
                    from sequence_store import ManoSequence
                    from image_manifest import ImageSequence

                    images = ImageSequence(seq.path)
                    ManoSequence.load(
                        out, images.coordinate_system, images.manifest_sha256
                    )
                finally:
                    sys.path.remove(str(consumer))
            p = seq.workdir / "predictions/joints_3d.npz"
            p.write_bytes(p.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "stale"):
                export(seq, replace=True)


if __name__ == "__main__":
    unittest.main()
