import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from hocap_annotation.hand_shape import load_hand_shape
from hocap_annotation.rgbd_data import digest


class HandShapeBundleTests(unittest.TestCase):
    def test_automatic_easymocap_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'trial_shape.npz').write_bytes(b'converter handles NPZ validation')
            models = {}
            for side in ('right', 'left'):
                asset = root / f'MANO_{side.upper()}.pkl'
                asset.write_bytes(side.encode())
                models[side] = digest(asset)
            report = dict(status='complete', model_sha256=models)
            report_path = root / 'report.json'
            report_path.write_text(json.dumps(report))
            before = {p.name: p.read_bytes() for p in root.iterdir()}

            def convert(command, **kwargs):
                output = Path(command[command.index('--output') + 1])
                output.mkdir()
                beta = output / 'betas.json'
                beta.write_text(json.dumps([[1.] * 10] * 2))
                (output / 'metadata.json').write_text(json.dumps(dict(
                    schema='hocap.fixed_hand_shape.v1', hand_order=['right', 'left'],
                    betas_sha256=digest(beta), model_sha256=models)))
                return SimpleNamespace(returncode=0)

            with patch('hocap_annotation.hand_shape.subprocess.run', side_effect=convert) as run:
                self.assertEqual(load_hand_shape(root, root), [[1.] * 10] * 2)
                self.assertEqual(run.call_count, 1)
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})
            with patch('hocap_annotation.hand_shape.subprocess.run', return_value=SimpleNamespace(returncode=1, stderr='bad basis')):
                with self.assertRaisesRegex(ValueError, 'bad basis'):
                    load_hand_shape(root, root)
            for invalid in [{**report, 'status': 'failed'},
                            {**report, 'model_sha256': {}}]:
                report_path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    load_hand_shape(root, root)
            (root / 'betas.json').write_text('[]')
            with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                load_hand_shape(root, root)

    def test_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            beta = root / "betas.json"
            beta.write_text(json.dumps([[0.] * 10] * 2))
            models = {}
            for side in ("right", "left"):
                path = root / f"MANO_{side.upper()}.pkl"
                path.write_bytes(side.encode())
                models[side] = digest(path)
            metadata = dict(schema="hocap.fixed_hand_shape.v1",
                            hand_order=["right", "left"],
                            betas_sha256=digest(beta), model_sha256=models)
            meta = root / "metadata.json"
            meta.write_text(json.dumps(metadata))
            self.assertEqual(load_hand_shape(root, root), [[0.] * 10] * 2)
            for field, value in [("schema", "unknown"),
                                 ("hand_order", ["left", "right"]),
                                 ("betas_sha256", "bad"), ("model_sha256", {})]:
                with self.subTest(field=field):
                    meta.write_text(json.dumps({**metadata, field: value}))
                    with self.assertRaises(ValueError):
                        load_hand_shape(root, root)
            for value in [[0.] * 10, [[float('nan')] * 10] * 2]:
                beta.write_text(json.dumps(value))
                meta.write_text(json.dumps({**metadata, 'betas_sha256': digest(beta)}))
                with self.assertRaises(ValueError):
                    load_hand_shape(root, root)
            beta.write_text(json.dumps([[0.] * 10] * 2))
            meta.write_text(json.dumps(metadata))
            (root / "MANO_LEFT.pkl").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "model mismatch"):
                load_hand_shape(root, root)
            meta.unlink()
            with self.assertRaises(FileNotFoundError):
                load_hand_shape(root)

    def test_cli_rejects_before_prepare(self):
        cli = Path(__file__).resolve().parents[1] / "tools/rgbd_hand_pipeline.py"
        with tempfile.TemporaryDirectory() as tmp:
            for options in [["--hand-shape", tmp],
                            ["--hand-shape", tmp, "--mean-hand"],
                            ["--hand-shape", tmp, "--betas", "unused.json"]]:
                result = subprocess.run(
                    [sys.executable, str(cli), tmp, "--stage", "prepare", *options],
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertNotIn("=== prepare ===", result.stdout)
                self.assertFalse((Path(tmp) / "hand_annotations").exists())
