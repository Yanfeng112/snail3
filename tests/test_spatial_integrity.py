"""Regression checks for spatial integrity, numerical precision and calendar inputs."""

import importlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from rasterio.transform import from_origin
from python.common import idw
from python.environment_safeguards import filtered_idw, constrain_environment
from python.spatial_integrity import (
    pixel_areas_km2,
    cell_centres,
    validate_grid_predictions,
)
from python.raster_io import write_raster, read_raster

raw = importlib.import_module("python.01_raw_environment")
correction = importlib.import_module("python.03_correction")
audit = importlib.import_module("python.10_audit_predictions")


def profile():
    return dict(
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float64",
        crs="EPSG:32650",
        transform=from_origin(400000, 3300000, 100, 100),
    )


class SpatialIntegrityTests(unittest.TestCase):
    def test_back_transform_overflow_is_not_a_density_prediction(self):
        from python.common import back_transform

        for value in [1000, np.inf, np.nan]:
            with self.assertRaises(ValueError):
                back_transform([value], "retain")

    def test_projected_area_does_not_cancel_at_large_origins(self):
        p = profile()
        p["transform"] = from_origin(1e12, 1e12, 100, 100)
        np.testing.assert_allclose(
            pixel_areas_km2(p, [0, 1, 2, 3], "projected_cell"), 0.01, rtol=0, atol=0
        )

    def test_equal_area_feet_are_not_treated_as_square_metres(self):
        feet = "+proj=laea +lat_0=30 +lon_0=115 +datum=WGS84 +units=us-ft +type=crs"
        with self.assertRaisesRegex(ValueError, "metres"):
            pixel_areas_km2(profile(), [0], "equal_area", feet)
        metres = "+proj=laea +lat_0=30 +lon_0=115 +datum=WGS84 +units=m +type=crs"
        self.assertTrue(
            np.isfinite(pixel_areas_km2(profile(), [0], "equal_area", metres)).all()
        )

    def valid_grid(self):
        xy = cell_centres(profile(), [0, 1])
        return pd.DataFrame(
            dict(
                cell_id=[0, 1],
                region="dongting",
                x=xy[:, 0],
                y=xy[:, 1],
                predicted=[0.1, 0.2],
                sd_boot=[0.01, 0.02],
            )
        )

    def test_grid_id_is_insufficient_without_matching_coordinates(self):
        d = self.valid_grid()
        d.loc[0, "x"] += 100
        with self.assertRaisesRegex(ValueError, "coordinates"):
            validate_grid_predictions(d, [0, 1], profile(), "dongting")

    def test_negative_sd_and_wrong_region_are_rejected(self):
        d = self.valid_grid()
        d.loc[0, "sd_boot"] = -1
        with self.assertRaisesRegex(ValueError, "bootstrap SD"):
            validate_grid_predictions(d, [0, 1], profile(), "dongting")
        d = self.valid_grid()
        d.loc[0, "region"] = "anhui"
        with self.assertRaisesRegex(ValueError, "region"):
            validate_grid_predictions(d, [0, 1], profile(), "dongting")

    def test_fractional_pixel_ids_are_not_truncated(self):
        d = self.valid_grid()
        d["cell_id"] = [0, 0.5]
        with self.assertRaisesRegex(ValueError, "integers"):
            validate_grid_predictions(d, [0, 1], profile(), "dongting")

    def test_write_roundtrip_preserves_threshold_precision_and_minus9999(self):
        with TemporaryDirectory() as folder:
            a = np.array([[0.300500000001, -9999], [np.nan, 0.000500000001]])
            path = Path(folder) / "precise.tif"
            write_raster(path, a, profile())
            b, p = read_raster(path)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(p["dtype"], "float64")

    def test_float32_export_must_not_change_density_class(self):
        from python.common import classify_density

        value = float(np.nextafter(0.3005, 0))
        with TemporaryDirectory() as folder:
            path = Path(folder) / "boundary.tif"
            write_raster(path, np.full((2, 2), value), profile())
            restored, _ = read_raster(path)
            _, classes = classify_density(restored)
            np.testing.assert_array_equal(classes, np.full((2, 2), 3))

    def test_infinite_raster_does_not_silently_become_nodata(self):
        with TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "Infinite"):
                write_raster(
                    Path(folder) / "bad.tif", np.array([[np.inf, 0], [0, 0]]), profile()
                )

    def test_missing_cnn_support_is_not_extrapolated(self):
        a = np.arange(9, dtype=float).reshape(3, 3)
        a[1, 1] = np.nan
        rows, cols = np.indices(a.shape)
        points = np.column_stack([cols.ravel(), rows.ravel()])
        domain = np.ones(a.shape, bool)
        domain[0, 0] = False
        output, report = filtered_idw(
            a, points, domain, filter_outliers=False, k=12, power=2
        )
        self.assertTrue(np.isnan(output[1, 1]))
        self.assertTrue(np.isnan(output[0, 0]))
        self.assertEqual(report["missing_cells_extrapolated"], 0)
        self.assertEqual(report["unsupported_domain_cells"], 1)

    def test_outlier_replaced_without_changing_retained_values(self):
        a = np.zeros((5, 5))
        a[2, 2] = 100
        rows, cols = np.indices(a.shape)
        points = np.column_stack([cols.ravel(), rows.ravel()])
        output, report = filtered_idw(
            a, points, np.ones(a.shape, bool), filter_outliers=True, k=12, power=2
        )
        np.testing.assert_array_equal(output, np.zeros_like(a))
        self.assertEqual(report["outliers_replaced"], 1)

    def test_physical_bounds_distinguish_leap_years_and_count_changes(self):
        bounds = {"lower": 0, "annual_days": True}
        result, report = constrain_environment(
            [-1, 366, 367, np.nan], "2024", bounds, "clip"
        )
        np.testing.assert_array_equal(result, [0, 366, 366, np.nan])
        self.assertEqual(report["changed_values"], 2)
        with self.assertRaisesRegex(ValueError, "Nonphysical"):
            constrain_environment([366], "2023", bounds, "error")

    def test_invalid_bound_does_not_turn_output_into_nan(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            constrain_environment([1], "2023", {"lower": np.nan}, "clip")
        with self.assertRaisesRegex(ValueError, "Infinite"):
            constrain_environment([np.inf], "2023", {"lower": 0}, "clip")

    def test_idw_weights_remain_finite_for_tiny_distances(self):
        result = idw([[0, 0], [1e-10, 0]], [2, 4], [[1e-30, 0]], power=100)
        np.testing.assert_allclose(result, [2])
        with self.assertRaises(ValueError):
            idw([[0, 0]], [2], [[1, 0]], power=np.nan)

    def test_video_date_aliases_cannot_fake_complete_year(self):
        d = pd.DataFrame(
            [
                dict(site_id="v", date=str(date.date()), hour=hour, inundated=1)
                for date in pd.date_range("2023-01-01", "2023-12-31")
                for hour in [8, 12, 16]
            ]
        )
        d.loc[d.date == "2023-12-31", "date"] = "2023-1-1"
        with self.assertRaisesRegex(ValueError, "normalisation"):
            raw.video_inundation(d)

    def test_water_level_date_aliases_cannot_fake_complete_year(self):
        cells = pd.DataFrame(dict(cell_id=[0], x=[0], y=[0], elevation_1985_m=[5]))
        water = pd.DataFrame(
            dict(
                station_id="s",
                x=0,
                y=0,
                date=pd.date_range("2023-01-01", "2023-12-31").strftime("%Y-%m-%d"),
                level_1985_m=6,
            )
        )
        water.loc[364, "date"] = "2023-1-1"
        with self.assertRaisesRegex(ValueError, "normalisation"):
            raw.water_level_inundation(cells, water, True)

    def test_nat_and_non_hourly_temperature_are_rejected_before_grouping(self):
        for timestamp in ["NaT", "2023-01-01T00:30:00Z"]:
            d = pd.DataFrame(dict(site_id=["a"], time_utc=[timestamp], temp_c=[10]))
            with self.assertRaisesRegex(ValueError, "exact hourly"):
                raw.temperature_indicators(d)

    def test_correction_candidates_must_share_identical_ground_truth(self):
        rows = [
            dict(
                site_id=f"s{i}",
                period="2023",
                x=i * 100,
                y=0,
                observed=i + 2,
                downscaled=i + 1,
            )
            for i in range(30)
        ]
        d = pd.DataFrame(rows)
        split = correction.split_sites(d)
        different = d.copy()
        different.loc[0, "observed"] += 1
        with self.assertRaisesRegex(ValueError, "same observations"):
            correction.select_correction({"one": d, "two": different}, split, "period")

    def test_cv_self_prediction_is_not_an_out_of_sample_record(self):
        rows = []
        for model in ["OK", "RF"]:
            for repeat in [1, 2]:
                for fold in [1, 2]:
                    for dataset in ["training", "validation"]:
                        rows.append(
                            dict(
                                model=model,
                                repeat_id=repeat,
                                fold=fold,
                                region="toy",
                                survey_id="same_id",
                                set=dataset,
                                observed=0.1,
                                predicted=0.2,
                            )
                        )
        with self.assertRaisesRegex(ValueError, "exactly once"):
            audit.audit_predictions(
                pd.DataFrame(rows),
                models=["OK", "RF"],
                expected_repeats=2,
                expected_folds=2,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
