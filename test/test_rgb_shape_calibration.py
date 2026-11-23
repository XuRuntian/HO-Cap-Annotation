import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hocap_annotation.hand_shape import load_hand_shape
from hocap_annotation.rgbd_data import atomic_json, digest

script = Path(__file__).resolve().parents[1] / "tools/calibrate_rgb_hand_shape.py"
spec = importlib.util.spec_from_file_location("calibrate_rgb_hand_shape", script)
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)


class AggregateShapeTests(unittest.TestCase):
    cameras = [f"cam{i:02}" for i in range(6)]

    def samples(self):
        return [dict(camera=cam, side=side, betas=[offset + i] * 10)
                for side, offset in (("right", 0), ("left", 10))
                for i, cam in enumerate(self.cameras)
                for _ in range(30 if i == 0 else 5)]

    def test_camera_weights_are_equal_and_sides_remain_separate(self):
        betas, counts = calibration.aggregate(self.samples(), self.cameras)
        np.testing.assert_array_equal(betas, [[2.5] * 10, [12.5] * 10])
        self.assertEqual(counts["right"]["cam00"], 30)

    def test_missing_left_views_do_not_fall_back_to_right(self):
        rows = [r for r in self.samples() if r["side"] == "right" or r["camera"] == "cam00"]
        with self.assertRaisesRegex(ValueError, "left: need"):
            calibration.aggregate(rows, self.cameras)

    def test_publish_includes_editable_measurement_template(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "samples.json"
            out = root / "shape"
            for side in calibration.SIDES:
                (root / f"MANO_{side.upper()}.pkl").write_bytes(side.encode())
            bases = {s: {"v_template": s, "shapedirs": s} for s in calibration.SIDES}
            atomic_json(source, dict(
                clip="clip_000", cameras=self.cameras, samples=self.samples(),
                mano_basis_sha256=bases, fah_model="fah", config_sha256="example"
            ))
            with patch("hocap_annotation.hand_shape.model_basis_hashes", return_value=bases):
                calibration.publish(source, out, root)
                lengths = out / "lengths.yaml"
                self.assertEqual(lengths.read_bytes(), calibration.LENGTH_TEMPLATE.read_bytes())
                self.assertEqual((out / "lengths_niosh.yaml").read_bytes(),
                                 calibration.SURFACE_TEMPLATE.read_bytes())
                base_hash = digest(out / "betas.json")
                lengths.write_text("units: mm\nleft:\n  thumb_ip_to_tip: 31.5\n")
                self.assertEqual(load_hand_shape(out / "betas.json", root), [[2.5] * 10, [12.5] * 10])
                self.assertEqual(digest(out / "betas.json"), base_hash)
                with self.assertRaises(FileExistsError):
                    calibration.publish(source, out, root)
                self.assertIn("31.5", lengths.read_text())
