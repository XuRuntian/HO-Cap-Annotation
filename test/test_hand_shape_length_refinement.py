import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hocap_annotation.hand_shape import load_hand_shape
from hocap_annotation.rgbd_data import digest


path = Path(__file__).resolve().parents[1] / "tools/refine_hand_shape_lengths.py"
spec = importlib.util.spec_from_file_location("refine_hand_shape_lengths", path)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def neutral(beta):
    joints = np.zeros((21, 3))
    for i in range(5):
        joints[4*i+1:4*i+5, 0] = 12*i
        joints[4*i+1:4*i+5, 1] = [35, 55, 75, 95]
    joints[4, 1] += 2 * beta[0]
    return joints


class HandShapeLengthRefinementTests(unittest.TestCase):
    def test_chinese_yaml_template_with_one_measurement(self):
        from ruamel.yaml import YAML

        template = path.parents[1] / "config/hand_bone_lengths.template.yaml"
        raw = YAML(typ="safe").load(template.read_text(encoding="utf-8"))
        for side in tool.SIDES:
            self.assertEqual(set(raw[side]), set(tool.SEGMENTS) | set(tool.PALM))
            self.assertTrue(all(v is None for v in raw[side].values()))
        with self.assertRaisesRegex(ValueError, "at least one"):
            tool.read_targets(template)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "lengths.yaml"
            raw["left"]["thumb_ip_to_tip"] = 31.5
            with target.open("w", encoding="utf-8") as stream:
                YAML(typ="safe").dump(raw, stream)
            self.assertEqual(tool.read_targets(target), {"left": {"thumb_ip_to_tip": 31.5}})

    def test_strict_targets(self):
        for payload in (
            {"units": "cm", "left": {"thumb_ip_to_tip": 3}},
            {"units": "mm", "left": {"unknown_bone": 3}},
            {"units": "mm", "left": {"thumb_ip_to_tip": True}},
            {"units": "mm", "left": {"thumb_ip_to_tip": float("nan")}},
            {"units": "mm"},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                tool.parse_targets(payload)

    def test_writes_version_without_mutating_input_or_other_hand(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "original"
            base.mkdir()
            assets = root / "assets"
            assets.mkdir()
            models = {}
            for side in tool.SIDES:
                model = assets / f"MANO_{side.upper()}.pkl"
                model.write_bytes(side.encode())
                models[side] = digest(model)
            source = base / "betas.json"
            original = [[0.25] * 10, [0] * 10]
            source.write_text(json.dumps(original))
            source_hash = digest(source)
            metadata = base / "metadata.json"
            metadata.write_text(json.dumps(dict(schema="hocap.fixed_hand_shape.v1",
                                                hand_order=list(tool.SIDES),
                                                model_sha256=models, betas_sha256=source_hash)))
            meta_hash = digest(metadata)
            target = root / "target.json"
            target.write_text(json.dumps(dict(units="mm", left={"thumb_ip_to_tip": 18})))
            out = root / "v2"
            with patch.object(tool, "neutral_geometry", return_value=neutral), patch.object(
                tool, "model_basis_hashes", return_value=models
            ):
                beta_path = tool.run(source, target, out, assets)
                with self.assertRaises(FileExistsError):
                    tool.run(source, target, out, assets)
            fixed = load_hand_shape(beta_path, assets, verify_basis=False)
            self.assertEqual(fixed[0], original[0])
            self.assertLess(abs(20 + 2*fixed[1][0] - 18), 0.3)
            self.assertEqual(fixed[1][1:], [0] * 9)
            self.assertEqual(digest(source), source_hash)
            self.assertEqual(digest(metadata), meta_hash)
            self.assertEqual(json.loads((out / "refinement.json").read_text())["sides"].keys(),
                             {"left"})

    def test_does_not_exceed_adjustment_budget(self):
        beta, summary = tool.refine(np.zeros(10), neutral, {"thumb_ip_to_tip": 10.0}, .5)
        self.assertLessEqual(abs(beta[0]), .5)
        self.assertGreater(summary["target_mm"]["thumb_ip_to_tip"]["error_after"], 8)
