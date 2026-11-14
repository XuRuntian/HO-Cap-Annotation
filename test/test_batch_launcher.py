"""Legacy batch interface and spawn-worker isolation without real model fitting."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from hocap_annotation.batch_pipeline import BatchOptions, run_batch
from hocap_annotation.batch_pipeline import _worker_entry as real_worker_entry
from test_batch_pipeline import nonempty_capture


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_batch.sh"


def crash_one_worker(sender, args, cpu_threads):
    if args[0].name == "crash":
        os._exit(17)
    real_worker_entry(sender, args, cpu_threads)


class BatchLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "group with spaces"
        self.shape = self.root / "shape with spaces"
        self.shape.mkdir()
        for name in ("betas.json", "metadata.json"):
            (self.shape / name).write_text("{}")
        for name in ("one", "two with spaces"):
            (self.data / name / "rgbd").mkdir(parents=True)
        (self.data / "nested" / "ignored" / "rgbd").mkdir(parents=True)
        (self.data / "linked").symlink_to(self.data / "one", target_is_directory=True)
        self.repo = self.root / "repo"
        (self.repo / "tools").mkdir(parents=True)
        (self.repo / "tools/batch_rgb_hand_pipeline.py").write_text(
            "import json, os, sys\n"
            "print(json.dumps(sys.argv[1:]))\n"
            'sys.exit(int(os.environ.get("FAKE_EXIT", "0")))\n'
        )
        self.env = dict(os.environ)
        for key in (
            "JOBS",
            "MODE",
            "REPLACE",
            "BATCH_RESULTS",
            "MANO_ROOT",
            "HAND_MODEL",
        ):
            self.env.pop(key, None)
        self.env.update(HOCAP_REPO=str(self.repo), MANO_PYTHON=sys.executable)

    def launch(self, **env):
        return subprocess.run(
            ["bash", str(SCRIPT), str(self.data), str(self.shape)],
            env=self.env | env,
            capture_output=True,
            text=True,
        )

    def test_old_interface_selects_only_direct_children_and_new_pipeline(self):
        result = self.launch(JOBS="4")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(result.stdout.splitlines()[-1])
        self.assertEqual(args[0], str(self.data))
        self.assertEqual(args[args.index("--workers") + 1], "4")
        self.assertEqual(args[args.index("--hand-shape") + 1], str(self.shape))
        self.assertEqual(
            args[args.index("--sequences") + 1 :], ["./one", "./two with spaces"]
        )
        self.assertIn("--resume", args)
        self.assertEqual(
            args[args.index("--batch-results") + 1], str(self.data / "batch_results")
        )
        self.assertNotIn("--replace", args)
        self.assertFalse((self.data / "one" / "hand_annotations").exists())

    def test_modes_output_override_replace_and_exit_status(self):
        result = self.launch(
            MODE="precheck",
            REPLACE="1",
            BATCH_RESULTS="/tmp/custom batch",
            FAKE_EXIT="7",
        )
        self.assertEqual(result.returncode, 7)
        args = json.loads(result.stdout.splitlines()[-1])
        self.assertIn("--replace", args)
        self.assertEqual(args[args.index("--mode") + 1], "precheck")
        self.assertEqual(args[args.index("--batch-results") + 1], "/tmp/custom batch")

    def test_existing_group_results_are_not_selected_as_sequences(self):
        output = self.data / "batch_results"
        category = output / "01_normal"
        category.mkdir(parents=True)
        (category / "one-id").symlink_to(self.data / "one", target_is_directory=True)
        (output / "work" / "one-id" / "inputs").mkdir(parents=True)
        marker = output / "user-note.txt"
        marker.write_text("keep")
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(result.stdout.splitlines()[-1])
        self.assertEqual(
            args[args.index("--sequences") + 1 :], ["./one", "./two with spaces"]
        )
        self.assertEqual(marker.read_text(), "keep")
        self.assertTrue((category / "one-id").is_symlink())

    def test_invalid_settings_and_empty_group_fail_before_execution(self):
        for env in (
            {"JOBS": "0"},
            {"JOBS": "33"},
            {"JOBS": "oops"},
            {"MODE": "invalid"},
            {"REPLACE": "yes"},
        ):
            self.assertEqual(self.launch(**env).returncode, 2)
        self.data = self.root / "empty"
        self.data.mkdir()
        self.assertEqual(self.launch().returncode, 2)

    def test_legacy_shape_is_rejected_before_detection(self):
        (self.shape / "betas.json").unlink()
        (self.shape / "trial_shape.npz").touch()
        (self.shape / "report.json").write_text("{}")
        self.assertEqual(self.launch().returncode, 2)

    def test_worker_hard_exit_does_not_stop_remaining_sequences(self):
        for name in ("crash", "later1", "later2", "later3"):
            (self.root / name).mkdir()
        with patch("hocap_annotation.batch_pipeline._worker_entry", crash_one_worker):
            rows = run_batch(
                self.root,
                self.root / "crash-results",
                [dict(sequence=n) for n in ("crash", "later1", "later2", "later3")],
                BatchOptions(),
                workers=2,
                cpu_threads=1,
            )
        self.assertEqual(len(rows), 4)
        reasons = {r["sequence"]: r["reason"] for r in rows}
        self.assertIn("exit=17", reasons["crash"])
        for name in ("later1", "later2", "later3"):
            self.assertIn("InputError", reasons[name])

    def test_real_spawn_workers_publish_one_summary_and_isolate_failure(self):
        first, _ = nonempty_capture(self.root / "first")
        second, _ = nonempty_capture(self.root / "second")
        data_root = self.root
        missing = self.root / "missing"
        missing.mkdir()
        jobs = [
            dict(sequence=str(p.relative_to(data_root)))
            for p in (first, missing, second)
        ]
        output = data_root / "batch_results"
        rows = run_batch(
            data_root,
            output,
            jobs,
            BatchOptions(mode="report"),
            workers=2,
            cpu_threads=1,
        )
        self.assertEqual(len(rows), 3)
        statuses = {r["sequence"]: r["status"] for r in rows}
        self.assertEqual(statuses["missing"], "04_failed")
        for seq in (first, second):
            self.assertEqual(
                statuses[str(seq.relative_to(data_root))], "03_needs_review"
            )
        with (output / "summary.csv").open() as f:
            self.assertEqual(len(list(csv.DictReader(f))), 3)
        self.assertEqual(len(list(output.glob("0*/*"))), 3)
        self.assertFalse((output / "work").exists())
        links = {str(p): p.lstat().st_ino for p in output.glob("0*/*")}
        again = run_batch(
            data_root,
            output,
            jobs,
            BatchOptions(mode="report", resume=True),
            workers=2,
            cpu_threads=1,
        )
        self.assertEqual({r["sequence"]: r["status"] for r in again}, statuses)
        self.assertEqual({str(p): p.lstat().st_ino for p in output.glob("0*/*")}, links)


if __name__ == "__main__":
    unittest.main()
