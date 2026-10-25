import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


spec = importlib.util.spec_from_file_location(
    'coordinate_export', Path(__file__).resolve().parents[1]/'tools/export_mano_coordinates.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CoordinateExportTests(unittest.TestCase):
    def test_transform_preserves_nan_and_metric_translation(self):
        transform = np.eye(4)
        transform[:3, 3] = [1, 2, 3]
        points = np.array([[0., 0, 0], [np.nan, np.nan, np.nan]])
        out = module.transform_points(points, transform)
        np.testing.assert_allclose(out[0], [1, 2, 3])
        self.assertTrue(np.isnan(out[1]).all())

    def test_axis_confirmation_is_required_before_model_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'confirm-table-axes'):
                module.export_coordinates('unused', 'unused', 'unused', Path(tmp)/'out.npz')

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'out.npz'
            path.write_bytes(b'original')
            with self.assertRaises(FileExistsError):
                module.export_coordinates('unused', 'unused', 'unused', path, confirm_axes=True)
            self.assertEqual(path.read_bytes(), b'original')

    def test_quality_gate_is_explicit_and_requires_a_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = Path(tmp)/'table.json'
            table.write_text(json.dumps(dict(translation_unit='meter', master_serial='111')))
            with self.assertRaisesRegex(ValueError, 'require-rig-pass'):
                module.export_coordinates('unused', 'unused', table, Path(tmp)/'out.npz',
                                          confirm_axes=True, require_rig_pass=True)

    def test_bad_independent_report_is_rejected_in_strict_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = Path(tmp)/'table.json'
            report = Path(tmp)/'rig.json'
            table.write_text(json.dumps(dict(translation_unit='meter', master_serial='111')))
            report.write_text(json.dumps(dict(schema='sync_lab.rig_validation.v1', master_serial='111', status='inconsistent')))
            with self.assertRaisesRegex(ValueError, 'require-rig-pass'):
                module.export_coordinates('unused', 'unused', table, Path(tmp)/'out.npz', confirm_axes=True,
                                          rig_report_file=report, require_rig_pass=True)

    def test_legacy_quality_warning_does_not_gate_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = Path(tmp)/'table.json'
            table.write_text(json.dumps(dict(translation_unit='meter', master_serial='111',
                T_table_from_master=np.eye(4).tolist(), T_master_from_table=np.eye(4).tolist(),
                rig_validation_warning=True)))
            with patch.object(module, 'SequenceLoader', side_effect=RuntimeError('reached loader')):
                with self.assertRaisesRegex(RuntimeError, 'reached loader'):
                    module.export_coordinates('unused', 'unused', table, Path(tmp)/'out.npz',
                                              confirm_axes=True)


if __name__ == '__main__':
    unittest.main()
