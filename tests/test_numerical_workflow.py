"""Small synthetic checks; these do not reproduce any published study result."""

import importlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from python.common import metrics, idw, classify_density, back_transform

area = importlib.import_module("python.06_density_area")
cor = importlib.import_module("python.03_correction")
raw = importlib.import_module("python.01_raw_environment")


class NumericalWorkflowTests(unittest.TestCase):
    def test_half_up_and_thresholds(self):
        rounded, classes = classify_density(
            np.array(
                [
                    0,
                    0.00049,
                    0.0005,
                    0.02049,
                    0.0205,
                    0.1005,
                    0.30049,
                    0.3005,
                    0.50049,
                    0.5005,
                ]
            )
        )
        np.testing.assert_array_equal(classes, [0, 0, 1, 1, 2, 3, 3, 4, 4, 5])
        np.testing.assert_allclose(
            rounded, [0, 0, 0.001, 0.020, 0.021, 0.101, 0.300, 0.301, 0.500, 0.501]
        )

    def test_area_includes_zero_domain(self):
        _, _, table = area.area_summary([0, 0.00049, 0.0005, 0.3005, 0.5005], 0.01)
        d = table.set_index("density_class")
        self.assertAlmostEqual(d.loc["positive_habitat_total", "area_km2"], 0.03)
        self.assertAlmostEqual(
            d.loc["prediction_domain_total_including_zero", "area_km2"], 0.05
        )
        self.assertAlmostEqual(
            d.loc["potential_high_density_gt_0.300", "area_km2"], 0.02
        )

    def test_metrics_and_inequalities(self):
        m = metrics([0, 1, 2], [0.2, 1.1, 2.3])
        self.assertGreaterEqual(m["rmse"], m["mae"])
        self.assertGreaterEqual(m["mae"] + 1e-12, abs(m["bias"]))
        self.assertEqual(m["mre_n"], 2)
        self.assertAlmostEqual(m["relative_bias"], 0.2)
        with self.assertRaises(ValueError):
            metrics([0, np.nan], [0, 1])

    def test_inverse_transform(self):
        np.testing.assert_allclose(
            back_transform(np.log1p([0, 0.2, 7]), "retain"), [0, 0.2, 7]
        )
        self.assertLess(back_transform([-1], "retain")[0], 0)
        self.assertEqual(back_transform([-1], "clip_zero")[0], 0)

    def test_idw_exact_and_duplicate_coordinates(self):
        y = idw([[0, 0], [0, 0], [10, 0]], [1, 3, 6], [[0, 0], [5, 0]])
        np.testing.assert_allclose(y, [2, 4])

    def test_correction_formulas_and_heldout_labels(self):
        train = pd.DataFrame(
            dict(
                site_id=["a", "b"],
                period=["2023", "2023"],
                x=[0, 10],
                y=[0, 0],
                observed=[12, 22],
                downscaled=[10, 20],
            )
        )
        query = pd.DataFrame(
            dict(
                site_id=["c"],
                period=["2023"],
                x=[5],
                y=[0],
                observed=[999],
                downscaled=[15],
            )
        )
        self.assertAlmostEqual(
            cor.predict_correction(train, query, "GDA", "period")[0], 17
        )
        # Changing held-out ground truth must not change its corrected prediction.
        q2 = query.assign(observed=-999)
        np.testing.assert_allclose(
            cor.predict_correction(train, query, "Sokol", "period"),
            cor.predict_correction(train, q2, "Sokol", "period"),
        )
        coeff = cor.correction_coefficients(train, "Sokol").coefficient.to_numpy()
        np.testing.assert_allclose(coeff, [22 / 20, 32 / 30])

    def test_station_years_remain_together(self):
        d = pd.DataFrame({"site_id": np.repeat([f"s{i}" for i in range(30)], 3)})
        s = cor.split_sites(d)
        self.assertEqual(s.site_id.nunique(), 30)
        self.assertEqual((s["set"] == "test").sum(), 9)
        self.assertFalse(
            set(s.loc[s["set"] == "test", "site_id"])
            & set(s.loc[s["set"] == "development", "site_id"])
        )

    def test_external_site_cv_selection(self):
        rows = []
        for i in range(30):
            for year in ["2023", "2024"]:
                rows.append(
                    dict(
                        site_id=f"s{i}",
                        period=year,
                        x=i * 100,
                        y=0,
                        observed=i + 2,
                        downscaled=i + 1,
                    )
                )
        d = pd.DataFrame(rows)
        split = cor.split_sites(d)
        winner, scores, pairs, train, test = cor.select_correction(
            {"candidate": d}, split, "pooled_site"
        )
        self.assertEqual(winner, ("candidate", "GDA"))
        self.assertAlmostEqual(metrics(test.observed, test.predicted)["rmse"], 0)
        self.assertTrue(set(train.site_id).isdisjoint(test.site_id))

    def test_leap_year_video_and_missingness(self):
        dates = pd.date_range("2024-01-01", "2024-12-31")
        d = pd.DataFrame(
            [
                dict(
                    site_id="v",
                    date=str(date.date()),
                    hour=h,
                    inundated=int(h in [8, 12]),
                )
                for date in dates
                for h in [8, 12, 16]
            ]
        )
        self.assertEqual(raw.video_inundation(d).observed.iloc[0], 366)
        with self.assertRaises(ValueError):
            raw.video_inundation(d.iloc[:-1])

    def test_water_level_datum_and_calendar(self):
        cells = pd.DataFrame(
            dict(cell_id=[0, 1], x=[0, 10], y=[0, 0], elevation_1985_m=[4, 6])
        )
        water = pd.DataFrame(
            dict(
                station_id="s",
                x=0,
                y=0,
                date=pd.date_range("2023-01-01", "2023-12-31").strftime("%Y-%m-%d"),
                level_1985_m=5,
            )
        )
        np.testing.assert_array_equal(
            raw.water_level_inundation(cells, water, True).inundation_days, [365, 0]
        )
        with self.assertRaises(ValueError):
            raw.water_level_inundation(cells, water, False)

    def test_raster_nodata_alignment_and_roundtrip(self):
        import rasterio
        from rasterio.transform import from_origin
        from python.raster_io import write_raster, read_raster, warp_to, sample_array

        with TemporaryDirectory() as td:
            p = dict(
                driver="GTiff",
                height=6,
                width=6,
                count=1,
                dtype="float32",
                crs="EPSG:32650",
                transform=from_origin(400000, 3300000, 100, 100),
            )
            a = np.ones((6, 6))
            a[0, 0] = np.nan
            file = Path(td) / "sample.tif"
            write_raster(file, a, p)
            b, profile = read_raster(file)
            self.assertTrue(np.isnan(b[0, 0]))
            np.testing.assert_allclose(warp_to(file, profile), b, equal_nan=True)
            self.assertEqual(
                sample_array(b, profile, [400150], [3299850], profile["crs"])[0], 1
            )

    def test_cnn_shape_if_tensorflow_available(self):
        try:
            import tensorflow
        except ImportError:
            self.skipTest(
                "TensorFlow 2.15 unavailable; CNN execution was not validated in this environment."
            )
        cnn = importlib.import_module("python.02_downscaling")
        model = cnn.build_cnn(4)
        self.assertEqual(model.output_shape, (None, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
