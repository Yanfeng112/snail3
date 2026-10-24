"""Regression checks for validation contracts and execution safeguards."""

import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import pandas as pd
from python.common import idw, classify_density
from python.contracts import (
    validate_observations,
    validate_site_split,
    require_identifiers,
)
from python.execution import (
    atomic_json,
    fingerprint,
    load_checkpoint,
    sha256_file,
    stage_status,
)
from python.spatial_validation import split_source_windows

cnn = importlib.import_module("python.02_downscaling")
correction = importlib.import_module("python.03_correction")
preflight = importlib.import_module("python.09_preflight")
prediction_audit = importlib.import_module("python.10_audit_predictions")


class ValidationContractTests(unittest.TestCase):
    def test_unknown_correction_is_not_sokol(self):
        with self.assertRaisesRegex(ValueError, "Unknown correction"):
            correction.apply_correction(np.array([1]), np.array([1]), "TYPO")

    def test_invalid_idw_parameters_fail_before_query(self):
        for parameters in [{"k": 0}, {"power": 0}, {"chunk": 0}]:
            with self.assertRaises(ValueError):
                idw([[0, 0]], [1], [[0, 0]], **parameters)

    def test_negative_density_is_not_hidden_by_rounding(self):
        with self.assertRaises(ValueError):
            classify_density([-0.00001])

    def test_missing_ids_do_not_disappear_in_groupby(self):
        data = pd.DataFrame(
            {
                "site_id": ["a", None],
                "period": ["2023", "2023"],
                "x": [0, 1],
                "y": [0, 1],
                "observed": [1, 2],
            }
        )
        with self.assertRaisesRegex(ValueError, "missing identifiers"):
            validate_observations(data)
        with self.assertRaisesRegex(ValueError, "empty input"):
            require_identifiers(pd.DataFrame({"id": []}), ["id"], "empty")

    def test_single_development_fold_rejected(self):
        observations = pd.DataFrame({"site_id": ["a", "b", "c"]})
        split = pd.DataFrame(
            {
                "site_id": ["a", "b", "c"],
                "set": ["development", "development", "test"],
                "fold": [1, 1, -1],
            }
        )
        with self.assertRaisesRegex(ValueError, "two development folds"):
            validate_site_split(split, observations)

    def test_buffered_cnn_windows_have_no_shared_pixels(self):
        rows, columns = np.indices((60, 60))
        # Duplicate coordinates represent the same cells observed in a second year.
        rows = np.tile(rows.ravel(), 2)
        columns = np.tile(columns.ravel(), 2)
        train, validation, audit = split_source_windows(
            rows,
            columns,
            mode="buffered_blocks",
            block_cells=15,
            buffer_cells=4,
            test_fraction=0.2,
        )
        self.assertEqual(audit["training_windows_sharing_validation_pixels"], 0)
        self.assertGreater(audit["minimum_chebyshev_distance_cells"], 4)
        a = set(zip(rows[train], columns[train]))
        b = set(zip(rows[validation], columns[validation]))
        self.assertTrue(a.isdisjoint(b))
        self.assertEqual(len(train), 2 * len(a))

    def test_too_small_buffer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "four-cell"):
            split_source_windows(
                [0, 10],
                [0, 10],
                mode="buffered_blocks",
                block_cells=10,
                buffer_cells=2,
                test_fraction=0.5,
            )

    def test_legacy_split_reports_window_overlap(self):
        rows, columns = np.indices((20, 20))
        _, _, audit = split_source_windows(
            rows.ravel(), columns.ravel(), mode="legacy_cell", test_fraction=0.2
        )
        self.assertGreater(audit["training_windows_sharing_validation_pixels"], 0)

    def test_batched_prediction_without_tensorflow(self):
        class MeanModel:
            def predict(self, patches, **kwargs):
                return patches.mean(axis=(1, 2, 3))[:, None]

        cube = np.ones((9, 10, 2))
        result = cnn.predict_map(
            cube, MeanModel(), np.zeros(2), np.ones(2), 3, 2, batch=7
        )
        np.testing.assert_allclose(result[2:-2, 2:-2], 5)
        self.assertTrue(np.isnan(result[:2]).all())

    def test_checkpoint_is_bound_to_input_bytes(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "input.txt"
            path.write_text("first")
            first = fingerprint({"alpha": 2}, [path])
            path.write_text("second")
            self.assertNotEqual(first, fingerprint({"alpha": 2}, [path]))

    def test_completed_checkpoint_detects_modified_outputs(self):
        with TemporaryDirectory() as folder:
            output = Path(folder) / "result.txt"
            output.write_text("valid")
            status = Path(folder) / "checkpoint.json"
            atomic_json(
                status,
                {
                    "fingerprint": "run1",
                    "status": "complete",
                    "outputs": [{"path": str(output), "sha256": sha256_file(output)}],
                },
            )
            self.assertIsNotNone(load_checkpoint(status, "run1"))
            output.write_text("tampered")
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                load_checkpoint(status, "run1")

    def test_failed_stage_is_recorded(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "status.json"
            with self.assertRaisesRegex(RuntimeError, "deliberate failure"):
                with stage_status(path, "run1", "test"):
                    raise RuntimeError("deliberate failure")
            result = json.loads(path.read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error_type"], "RuntimeError")

    def test_preflight_reports_multiple_blockers(self):
        report = preflight.inspect_configuration({}, "environment")
        self.assertFalse(report["ready"])
        self.assertGreater(report["errors"], 5)

    def test_paired_prediction_checker_rejects_unmatched_model_rows(self):
        records = []
        for model in ["OK", "RF"]:
            for repeat in [1, 2]:
                for fold in [1, 2]:
                    for dataset in ["training", "validation"]:
                        records.append(
                            dict(
                                region="toy",
                                repeat_id=repeat,
                                fold=fold,
                                survey_id=str(
                                    fold if dataset == "validation" else 3 - fold
                                ),
                                set=dataset,
                                model=model,
                                observed=0.1,
                                predicted=0.2,
                            )
                        )
        data = pd.DataFrame(records)
        summary = prediction_audit.audit_predictions(
            data, models=["OK", "RF"], expected_repeats=2, expected_folds=2
        )
        self.assertEqual(len(summary), 16)
        with self.assertRaisesRegex(ValueError, "differ"):
            prediction_audit.audit_predictions(
                data.iloc[:-1],
                models=["OK", "RF"],
                expected_repeats=2,
                expected_folds=2,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
