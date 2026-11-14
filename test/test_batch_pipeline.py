"""Batch policy and data protection at public processing/storage boundaries."""

import csv
import io
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from hocap_annotation.batch_cache import StageCache, StaleStage
from hocap_annotation.batch_inputs import InputError, inspect_inputs, verify_prepared
from hocap_annotation.batch_quality import QualityPolicy, final_quality, precheck
from hocap_annotation.batch_pipeline import BatchOptions, run_batch
from hocap_annotation import rgb_hand_pipeline as pipeline
from hocap_annotation.batch_results import BatchResults, sequence_id
from hocap_annotation.rgbd_data import (
    PreparedSequence,
    atomic_json,
    atomic_npz,
    digest,
    prepare,
)
from test_camera_consistency import geometry
from test_rgbd_pipeline import capture, observations


def nonempty_capture(root, count=3):
    data, seq = capture(root, count)
    for path in (data / "rgbd").glob("cam*/color_*.png"):
        cv2.imwrite(str(path), np.full((64, 80, 3), 80, np.uint8))
    prepare(data, seq.workdir, replace=True)
    return data, PreparedSequence(seq.workdir)


class BatchPolicyTests(unittest.TestCase):
    def test_geometry_not_bad_camera_count_controls_usability(self):
        matrices, _, xy = geometry()
        xy = np.tile(xy[None, :, None], (48, 1, 2, 1, 1))
        seen = np.ones(xy.shape[:-1], bool)
        ids = [f"cam{i:02d}" for i in range(6)]
        policy = QualityPolicy()
        report = precheck(xy, seen, matrices, ids, policy)
        self.assertTrue(report["usable"])
        xy[:, 2] += [45, -130]
        report = precheck(xy, seen, matrices, ids, policy)
        self.assertTrue(report["usable"])
        self.assertEqual(report["excluded_cameras"][0]["camera_id"], "cam02")
        # Same number of cameras, but degenerate reference rays: do not fit.
        matrices[:] = matrices[0]
        report = precheck(xy, seen, matrices, ids, policy)
        self.assertFalse(report["usable"])

    def test_missing_hand_not_failure_but_final_low_coverage_is(self):
        seen = np.ones((20, 6, 2, 21), bool)
        seen[:, :, 1] = False
        valid = np.zeros((20, 2), bool)
        valid[:, 0] = True
        diag = dict(
            valid=valid,
            reprojection_median_px=np.ones((20, 2)),
            joint_error_m=np.ones((20, 2)) * 0.01,
        )
        self.assertTrue(final_quality(seen, diag, QualityPolicy())["passed"])
        diag["reprojection_median_px"][:5, 0] = 100
        result = final_quality(seen, diag, QualityPolicy())
        self.assertFalse(result["passed"])
        self.assertEqual(result["hands"][1]["presence"], "unobserved")
        self.assertEqual(result["hands"][0]["valid_frames"], 15)

    def test_single_view_visible_frames_are_not_treated_as_absence(self):
        seen = np.zeros((100, 6, 2, 21), bool)
        seen[:, 0, 0] = True
        seen[:10, 1, 0] = True
        valid = np.zeros((100, 2), bool)
        valid[:10, 0] = True
        result = final_quality(
            seen,
            dict(
                valid=valid,
                reprojection_median_px=np.ones((100, 2)),
                joint_error_m=np.ones((100, 2)) * 0.01,
            ),
            QualityPolicy(),
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["hands"][0]["observed_frames"], 100)
        self.assertEqual(result["hands"][0]["coverage"], 0.1)

    def test_short_evidence_remains_review_and_policy_validates(self):
        m, _, xy = geometry()
        xy = np.tile(xy[None, :, None], (6, 1, 2, 1, 1))
        report = precheck(
            xy, np.ones(xy.shape[:-1], bool), m, list("abcdef"), QualityPolicy()
        )
        self.assertFalse(report["usable"])
        with self.assertRaises(ValueError):
            replace(QualityPolicy(), min_final_coverage=1.2)
        report = precheck(
            xy,
            np.ones(xy.shape[:-1], bool),
            m,
            list("abcdef"),
            QualityPolicy(),
            excluded=list("abcde"),
        )
        self.assertFalse(report["usable"])
        self.assertIn("insufficient cameras", report["reasons"][0])


class BatchStorageTests(unittest.TestCase):
    def test_owned_links_update_without_touching_user_files(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            a, b = root / "a/same", root / "b/same"
            a.mkdir(parents=True)
            b.mkdir(parents=True)
            self.assertNotEqual(sequence_id(a), sequence_id(b))
            with BatchResults(root / "results") as results:
                user_file = results.root / "01_normal/user.txt"
                user_file.write_text("keep")
                row = dict(
                    sequence="a/same", status="03_needs_review", reason="not fitted"
                )
                results.update(a, row)
                row.update(status="02_degraded", right_valid_frames=10)
                results.update(a, row)
                results.update(
                    b, dict(sequence="b/same", status="04_failed", reason="input")
                )
                self.assertFalse(
                    (results.root / "03_needs_review" / sequence_id(a)).exists()
                )
                self.assertEqual(
                    (results.root / "02_degraded" / sequence_id(a)).resolve(), a
                )
                self.assertEqual(user_file.read_text(), "keep")
                rows = list(
                    csv.DictReader(
                        io.StringIO((results.root / "summary.csv").read_text())
                    )
                )
                self.assertEqual(len(rows), 2)
                self.assertEqual(
                    next(r for r in rows if r["sequence"] == "b/same")[
                        "right_valid_frames"
                    ],
                    "",
                )
                collision = results.root / "01_normal" / sequence_id(a)
                collision.write_text("user")
                with self.assertRaises(ValueError):
                    results.update(a, dict(row, status="01_normal"))
                self.assertEqual(collision.read_text(), "user")

    def test_resume_validates_configuration_and_output_content(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            with StageCache(root / "work", root / "data") as cache:
                output = cache.workdir / "result.txt"
                action = lambda: output.write_text("prediction")
                self.assertEqual(
                    cache.run("fit", {"beta": 1}, ["result.txt"], action), "ran"
                )
                self.assertEqual(
                    cache.run("fit", {"beta": 1}, ["result.txt"], action, resume=True),
                    "reused",
                )
                with self.assertRaises(StaleStage):
                    cache.run("fit", {"beta": 2}, ["result.txt"], action, resume=True)
                output.write_text("tampered")
                self.assertFalse(cache.valid("fit", {"beta": 1}, ["result.txt"]))
            (root / "work/final").mkdir()
            with self.assertRaises(ValueError):
                StageCache(root / "work", root / "data")

    def test_work_artifact_symlink_cannot_overwrite_external_data(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "work").mkdir()
            (root / "raw").mkdir()
            (root / "raw/observations_2d.npz").write_text("original")
            (root / "work/predictions").symlink_to(
                root / "raw", target_is_directory=True
            )
            with self.assertRaises(ValueError):
                StageCache(root / "work", root / "data")
            self.assertEqual((root / "raw/observations_2d.npz").read_text(), "original")

    def test_report_takes_shared_lock_and_link_publication_recovers(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "sequence"
            source.mkdir()
            with StageCache(root / "work", source):
                with self.assertRaises(BlockingIOError):
                    StageCache(root / "work", source, readonly=True)
            with BatchResults(root / "results") as results:
                results.update(
                    source, dict(sequence="sequence", status="03_needs_review")
                )
                original = os.link

                def interrupted(*args, **kwargs):
                    original(*args, **kwargs)
                    raise SystemExit("interrupted after publication")

                with patch(
                    "hocap_annotation.batch_results.os.link", side_effect=interrupted
                ):
                    with self.assertRaises(SystemExit):
                        results.update(
                            source, dict(sequence="sequence", status="02_degraded")
                        )
            with BatchResults(root / "results") as results:
                links = [p for p in results.root.glob("0*/*") if p.is_symlink()]
                self.assertEqual(len(links), 1)
                self.assertEqual(links[0].parent.name, "02_degraded")
                self.assertEqual(links[0].resolve(), source)

    def test_wrong_json_object_shape_is_an_input_error(self):
        with tempfile.TemporaryDirectory() as root:
            data, _ = nonempty_capture(root)
            (data / "rgbd/calibration/extrinsics.json").write_text("[]")
            with self.assertRaises(InputError):
                inspect_inputs(data, QualityPolicy())

    def test_input_validation_and_prepared_source_binding(self):
        with tempfile.TemporaryDirectory() as root:
            data, seq = nonempty_capture(root)
            check = inspect_inputs(data, QualityPolicy())
            verify_prepared(seq, check)
            raw = data / "rgbd/cam00/color_000000.png"
            cv2.imwrite(str(raw), np.full((64, 80, 3), 90, np.uint8))
            changed = inspect_inputs(data, QualityPolicy())
            self.assertNotEqual(check["source_sha256"], changed["source_sha256"])
            with self.assertRaises(InputError):
                verify_prepared(seq, changed)
            raw.write_bytes(b"")
            with self.assertRaises(InputError):
                inspect_inputs(data, QualityPolicy())


class BatchRunTests(unittest.TestCase):
    def setup_batch(self, root):
        root = Path(root)
        data, seq = nonempty_capture(root, 48)
        _, points, _ = geometry()
        path = observations(seq, points)
        model = root / "hand.task"
        model.write_bytes(b"detector fixture")
        obs = seq.load("observations_2d.npz")
        obs["model_sha256"] = np.array(digest(model))
        atomic_npz(path, **obs)
        config = pipeline.detection_config(model)
        config.update(model_asset_path=model.name, model_sha256=digest(model))
        atomic_json(seq.workdir / "predictions/run_config.json", {"detect": config})
        models = root / "models"
        models.mkdir()
        for side in ("RIGHT", "LEFT"):
            (models / f"MANO_{side}.pkl").write_bytes(
                b"licensed model boundary fixture"
            )
        options = BatchOptions(
            mode="run",
            hand_model=model,
            mano_root=models,
            device="cpu",
            mean_hand=True,
            triangulation_steps=0,
            fit_steps=1,
        )
        jobs = [dict(sequence="capture", reuse_workdir=str(seq.workdir))]
        return options, jobs

    @staticmethod
    def fitted_fixture(seq, *args, **kwargs):
        n = len(seq.frame_ids)
        common = dict(
            **seq.common(),
            observations_sha256=np.array(
                digest(seq.workdir / "predictions/observations_2d.npz")
            ),
            triangulation_sha256=np.array(
                digest(seq.workdir / "predictions/joints_3d.npz")
            ),
        )
        valid = np.zeros((n, 2), bool)
        valid[:, 0] = True
        atomic_npz(
            seq.workdir / "predictions/fit_parameters.npz",
            **common,
            poses=np.zeros((n, 2, 51)),
            hand_pose=np.zeros((n, 2, 45)),
            betas=np.zeros((2, 10)),
            valid=valid,
        )
        errors = np.ones((n, 2))
        if kwargs["steps"] == 2:
            errors[:20, 0] = 100
        atomic_npz(
            seq.workdir / "predictions/fit_diagnostics.npz",
            **common,
            valid=valid,
            reprojection_median_px=errors,
            joint_error_m=np.ones((n, 2)) * 0.01,
        )

    def test_full_resume_report_final_downgrade_and_failure_isolation(self):
        with (
            tempfile.TemporaryDirectory() as root,
            patch.object(pipeline, "fit", self.fitted_fixture),
        ):
            root = Path(root)
            options, jobs = self.setup_batch(root)
            jobs = [dict(sequence="missing"), *jobs]
            rows = run_batch(root, root / "results", jobs, options)
            self.assertEqual([r["status"] for r in rows], ["04_failed", "01_normal"])
            self.assertEqual(rows[0]["right_valid_frames"], "")
            self.assertEqual(rows[1]["right_valid_frames"], 48)
            with patch.object(
                pipeline, "fit", side_effect=AssertionError("must reuse")
            ):
                rows = run_batch(
                    root, root / "results", jobs[1:], replace(options, resume=True)
                )
                self.assertEqual(rows[0]["status"], "01_normal")
                rows = run_batch(
                    root, root / "results", jobs[1:], replace(options, mode="report")
                )
                self.assertEqual(rows[0]["status"], "01_normal")
            rows = run_batch(
                root,
                root / "results",
                jobs[1:],
                replace(options, fit_steps=2, resume=True, replace=True),
            )
            self.assertEqual(rows[0]["status"], "03_needs_review")
            self.assertEqual(rows[0]["right_valid_frames"], 28)
            rows = run_batch(
                root,
                root / "results",
                jobs[1:],
                replace(options, mode="report", fit_steps=3),
            )
            self.assertEqual(rows[0]["status"], "03_needs_review")
            self.assertEqual(rows[0]["right_valid_frames"], "")

    def test_report_only_never_creates_prepared_data_or_runs_detector(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            data, _ = nonempty_capture(root, 3)
            before = {p: digest(p) for p in data.rglob("*") if p.is_file()}
            with patch.object(
                pipeline, "detect", side_effect=AssertionError("must not detect")
            ):
                rows = run_batch(
                    root, root / "results", [dict(sequence="capture")], BatchOptions()
                )
            self.assertEqual(rows[0]["status"], "03_needs_review")
            self.assertFalse((root / "results/work").exists())
            self.assertEqual(
                before, {p: digest(p) for p in data.rglob("*") if p.is_file()}
            )

    def test_missing_detection_runs_once_and_precheck_does_not_fit(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            options, jobs = self.setup_batch(root)
            jobs[0]['reuse_workdir'] = str(root/'no_reuse')
            _, points, _ = geometry()
            def detector(seq, *args, **kwargs):
                path = observations(seq, points)
                obs = seq.load('observations_2d.npz')
                obs['model_sha256'] = np.array(digest(options.hand_model))
                atomic_npz(path, **obs)
            with patch.object(pipeline, 'detect', side_effect=detector) as detect_mock, \
                    patch.object(pipeline, 'fit', side_effect=AssertionError('no fit in precheck')):
                rows = run_batch(root, root/'results', jobs, replace(options, mode='precheck'))
                self.assertEqual(rows[0]['status'], '03_needs_review')
                self.assertEqual(rows[0]['right_valid_frames'], '')
                self.assertIn('precheck passed', rows[0]['reason'])
                run_batch(root, root/'results', jobs, replace(options, mode='precheck', resume=True))
                self.assertEqual(detect_mock.call_count, 1)

    def test_manual_exclusion_degrades_and_insufficient_views_skip_fit(self):
        with tempfile.TemporaryDirectory() as root, patch.object(pipeline, 'fit', self.fitted_fixture):
            root = Path(root)
            options, jobs = self.setup_batch(root)
            rows = run_batch(root, root/'results', jobs, replace(options, exclude_cameras=('cam02',)))
            self.assertEqual(rows[0]['status'], '02_degraded')
            self.assertEqual(rows[0]['excluded_cameras'], 'cam02')
            with patch.object(pipeline, 'fit', side_effect=AssertionError('insufficient geometry')):
                rows = run_batch(root, root/'results', jobs, replace(options, resume=True,
                    exclude_cameras=('cam00','cam01','cam02','cam03','cam04')))
            self.assertEqual(rows[0]['status'], '03_needs_review')
            self.assertEqual(rows[0]['right_valid_frames'], '')

    def test_user_replaced_category_link_is_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root/'source'; source.mkdir()
            elsewhere = root/'user_source'; elsewhere.mkdir()
            with BatchResults(root/'results') as results:
                results.update(source, dict(sequence='source', status='03_needs_review'))
                old = results.root/'03_needs_review'/sequence_id(source)
                old.unlink(); old.symlink_to(elsewhere)
                results.update(source, dict(sequence='source', status='01_normal'))
                self.assertEqual(old.resolve(), elsewhere)

    def test_unowned_detail_directory_does_not_abort_following_sequences(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            with BatchResults(root / "results"):
                collision = root / "results/details" / sequence_id(root / "first")
                collision.mkdir()
                (collision / "user.txt").write_text("keep")
            rows = run_batch(
                root,
                root / "results",
                [dict(sequence="first"), dict(sequence="second")],
                BatchOptions(),
            )
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r["status"] == "04_failed" for r in rows))
            self.assertEqual((collision / "user.txt").read_text(), "keep")

    def test_timestamps_and_frame_correspondence_are_input_errors(self):
        with tempfile.TemporaryDirectory() as root:
            data, _ = nonempty_capture(root)
            path = data / "rgbd/frames.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[1]["cameras"]["cam00"]["color_source"]["timestamp_ms"] = 1000
            path.write_text("\n".join(json.dumps(row) for row in rows))
            with self.assertRaisesRegex(InputError, "timestamp"):
                inspect_inputs(data, QualityPolicy())


if __name__ == "__main__":
    unittest.main()
