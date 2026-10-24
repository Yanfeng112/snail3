"""Synthetic integration check: shared extraction -> ensemble-format predictions -> area output."""

import importlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from rasterio.transform import from_origin
from python.raster_io import write_raster, read_raster

extract = importlib.import_module("python.05_extract_covariates")
area = importlib.import_module("python.06_density_area")


class FilePipelineTests(unittest.TestCase):
    def test_shared_grid_extraction_and_area(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            p = dict(
                driver="GTiff",
                height=8,
                width=8,
                count=1,
                dtype="float32",
                crs="EPSG:32650",
                transform=from_origin(400000, 3300000, 100, 100),
            )
            masks = {}
            for i, region in enumerate(["dongting", "poyang", "anhui"]):
                m = np.zeros((8, 8))
                m[1 + i * 2 : 3 + i * 2, 1:4] = 1
                path = root / f"{region}.tif"
                write_raster(path, m, p)
                masks[region] = str(path)
            continuous = root / "covariate.tif"
            write_raster(continuous, np.ones((8, 8)) * 3, p)
            survey = []
            external = []
            for i, region in enumerate(masks):
                yy = 3300000 - (1 + i * 2 + 0.5) * 100
                survey.append(
                    dict(
                        survey_id=f"000{i}", region=region, x=400150, y=yy, density=0.1
                    )
                )
                for marsh in range(3):
                    for q in range(100):
                        external.append(
                            dict(
                                quadrat_id=f"q_{i}_{marsh}_{q}",
                                marshland_id=f"m_{i}_{marsh}",
                                region=region,
                                x=400110 + (q % 10) * 20,
                                y=3300000 - (1 + i * 2) * 100 - 10 - (q // 10) * 20,
                                live_snails=int(q == 0),
                            )
                        )
            sf = root / "survey.csv"
            ef = root / "external.csv"
            pd.DataFrame(survey).to_csv(sf, index=False)
            pd.DataFrame(external).to_csv(ef, index=False)
            cfg = dict(
                survey=str(sf),
                external=str(ef),
                xy_crs="EPSG:32650",
                output=str(root / "tables"),
                region_masks=masks,
                covariates={
                    "tmean": dict(
                        corrected=str(continuous), uncorrected=str(continuous)
                    )
                },
            )
            file = root / "extract.json"
            file.write_text(json.dumps(cfg))
            extract.main(file)
            for region in masks:
                c = pd.read_csv(root / "tables" / f"{region}_grid_corrected.csv")
                u = pd.read_csv(root / "tables" / f"{region}_grid_uncorrected.csv")
                pd.testing.assert_frame_equal(c, u)
                self.assertEqual(len(c), 6)
                c["predicted"] = [0.00049, 0.0005, 0.3005] * 2
                c["sd_boot"] = 0
                dest = root / "outputs" / "bootstrap" / region
                dest.mkdir(parents=True)
                c.to_csv(dest / "grid_predictions.csv", index=False)
            cfg = dict(
                output=str(root / "outputs"),
                region_masks=masks,
                area_method="projected_cell",
                area_equal_area_crs=None,
            )
            file = root / "area.json"
            file.write_text(json.dumps(cfg))
            area.main(file)
            d = pd.read_csv(root / "outputs" / "area" / "density_class_areas.csv")
            self.assertEqual(len(d), 27)
            positive = d.loc[d.density_class == "positive_habitat_total", "area_km2"]
            np.testing.assert_allclose(positive, 0.04)
            for region in masks:
                a, _ = read_raster(
                    root / "outputs" / "area" / f"{region}_density_rounded.tif"
                )
                np.testing.assert_allclose(
                    a[np.isfinite(a)], [0, 0.001, 0.301] * 2, rtol=1e-6
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
