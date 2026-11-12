import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location(
    'import_hand_shape', Path(__file__).resolve().parents[1] / 'tools/import_hand_shape.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ShapeImportTests(unittest.TestCase):
    def read(self, betas, order):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'shape.npz'
            np.savez(path, betas=betas, hand_order=order)
            return module.read_shape(path)

    def test_reorders_sides(self):
        betas = np.arange(20).reshape(2, 10)
        np.testing.assert_array_equal(self.read(betas, ['left', 'right']), betas[::-1])

    def test_preserves_canonical_order(self):
        betas = np.arange(20).reshape(2, 10)
        np.testing.assert_array_equal(self.read(betas, ['right', 'left']), betas)

    def test_rejects_invalid(self):
        for betas, order in [(np.zeros((10,)), ['right', 'left']),
                             (np.full((2, 10), np.nan), ['right', 'left']),
                             (np.zeros((2, 10)), ['right', 'right'])]:
            with self.subTest(order=order, shape=betas.shape):
                with self.assertRaises(ValueError):
                    self.read(betas, order)

    def test_conversion_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'trial_shape.npz'
            np.savez(source, betas=np.zeros((2, 10)), hand_order=['right', 'left'])
            for side in ('RIGHT', 'LEFT'):
                (root / f'MANO_{side}.pkl').write_bytes(b'test fixture')
            output = root / 'converted'
            argv = ['import_hand_shape', str(source), '--output', str(output),
                    '--easymocap-root', str(root), '--easymocap-python', 'python',
                    '--mano-root', str(root)]
            with patch('sys.argv', argv), patch.object(module, 'verify_models', return_value={}):
                module.main()
                self.assertEqual(json.loads((output / 'betas.json').read_text()),
                                 np.zeros((2, 10)).tolist())
                meta = json.loads((output / 'metadata.json').read_text())
                self.assertEqual(meta['betas_sha256'], module.sha256(output / 'betas.json'))
                self.assertEqual(meta['source_sha256'], module.sha256(source))
                with self.assertRaises(SystemExit):
                    module.main()
