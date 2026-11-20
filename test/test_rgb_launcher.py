"""Launcher defaults and overrides without models, CUDA or input images."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_rgb_hand_pipeline.sh"


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.home = Path(self.temporary.name)
        self.log = self.home / "calls.jsonl"
        self.env = {k: v for k, v in os.environ.items()
                    if k not in ("MANO_PYTHON", "DETECT_PYTHON", "MANO_ROOT", "HAND_MODEL",
                                 "DETECTOR", "FAH_PYTHON", "FAH_CONFIG")}
        self.env.update(HOME=str(self.home), LAUNCH_LOG=str(self.log))
        self.python = self.make_interpreter("miniconda3/envs/hocap-annotation/bin/python")
        self.fah_python = self.make_interpreter("FAH/.venv/bin/python")

    def make_interpreter(self, relative):
        path = self.home / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "with open(os.environ['LAUNCH_LOG'], 'a') as f:\n"
            "    f.write(json.dumps(sys.argv) + '\\n')\n"
            "if os.environ.get('FAIL_STAGE') in sys.argv[1:]:\n"
            "    sys.exit(7)\n")
        path.chmod(0o755)
        return path

    def launch(self, *args):
        result = subprocess.run(["bash", str(SCRIPT), str(self.home / "data with spaces"), *args],
                                env=self.env, capture_output=True, text=True)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []
        return result, calls

    def test_local_defaults_and_five_stages(self):
        result, calls = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([c[c.index("--stage") + 1] for c in calls],
                         ["prepare", "detect", "triangulate", "fit", "export"])
        for call in calls:
            stage = call[call.index("--stage") + 1]
            self.assertEqual(call[0], str(self.fah_python if stage == "detect" else self.python))
            self.assertEqual(call[call.index("--detector") + 1], "fah")
            self.assertEqual(call[call.index("--fah-config") + 1],
                             str(self.home / "FAH/configs/local.json"))
            self.assertEqual(call[call.index("--device") + 1], "cuda")
            self.assertEqual(call[call.index("--mano-root") + 1], str(self.home / "models/mano"))
            self.assertIn("--mean-hand", call)
            self.assertIn(str(self.home / "data with spaces"), call)

    def test_environment_and_cli_overrides(self):
        custom = self.make_interpreter("custom/python")
        self.env.update(MANO_PYTHON=str(custom), MANO_ROOT="/custom/mano", HAND_MODEL="/custom/hand.task")
        result, calls = self.launch("--detector", "mediapipe", "--device", "cpu",
                                    "--betas", "shape.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        for call in calls:
            self.assertEqual(call[0], str(custom))
            self.assertIn("/custom/mano", call)
            self.assertIn("/custom/hand.task", call)
            self.assertEqual(call[call.index("--detector") + 1], "mediapipe")
            self.assertNotIn("--mean-hand", call)
            self.assertEqual(call[-4:], ["--device", "cpu", "--betas", "shape.json"])

    def test_failure_stops_following_stages(self):
        self.env["FAIL_STAGE"] = "detect"
        result, calls = self.launch()
        self.assertEqual(result.returncode, 7)
        self.assertEqual(len(calls), 2)

    def test_hand_shape_directory(self):
        for args in [("--hand-shape", "/shape with spaces"),
                     ("--hand-shape=/shape with spaces",)]:
            with self.subTest(args=args):
                result, calls = self.launch(*args)
                self.assertEqual(result.returncode, 0, result.stderr)
                for call in calls[-5:]:
                    self.assertNotIn("--mean-hand", call)
                    self.assertEqual(call[-len(args):], list(args))


if __name__ == "__main__":
    unittest.main()
