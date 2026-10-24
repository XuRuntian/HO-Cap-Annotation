import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock


spec = importlib.util.spec_from_file_location(
    'hand_detection_entry', Path(__file__).resolve().parents[1] / 'tools/02_mp_hand_detection.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DetectionEntryTests(unittest.TestCase):
    def test_default_run_detects_before_rendering(self):
        detector = module.HandJointsDetector.__new__(module.HandJointsDetector)
        detector._logger = Mock()
        detector._data_folder = Path('test_sequence')
        detector.detect_mp_handmarks = Mock()
        detector.render_mp_handmarks = Mock()
        calls = Mock()
        calls.attach_mock(detector.detect_mp_handmarks, 'detect')
        calls.attach_mock(detector.render_mp_handmarks, 'render')
        detector.run()
        self.assertEqual([c[0] for c in calls.mock_calls], ['detect', 'render'])

    def test_skip_vis_still_runs_detection(self):
        detector = module.HandJointsDetector.__new__(module.HandJointsDetector)
        detector._logger = Mock()
        detector._data_folder = Path('test_sequence')
        detector.detect_mp_handmarks = Mock()
        detector.render_mp_handmarks = Mock()
        detector.run(skip_vis=True)
        detector.detect_mp_handmarks.assert_called_once()
        detector.render_mp_handmarks.assert_not_called()


if __name__ == '__main__':
    unittest.main()
