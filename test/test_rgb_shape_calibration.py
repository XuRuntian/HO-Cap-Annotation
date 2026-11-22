import importlib.util
from pathlib import Path
import unittest

import numpy as np

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
