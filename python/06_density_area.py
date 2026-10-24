"""Round before classifying; include zero cells in domain but exclude them from habitat area."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from python.common import classify_density, read_json
from python.raster_io import (
    read_raster,
    projected_metres,
    write_raster,
    check_same_grid,
)
from python.spatial_integrity import validate_grid_predictions, pixel_areas_km2


def area_summary(predicted, cell_area_km2):
    rounded, classes = classify_density(predicted)
    if not np.isfinite(rounded).all():
        raise ValueError("Missing predictions within the stated prediction domain.")
    areas = np.broadcast_to(np.asarray(cell_area_km2, float), rounded.shape)
    if not np.isfinite(areas).all() or np.any(areas <= 0):
        raise ValueError("Invalid cell area.")
    labels = ["0", "0.001-0.020", "0.021-0.100", "0.101-0.300", "0.301-0.500", ">0.500"]
    rows = [
        dict(
            density_class=label,
            cells=int((classes == i).sum()),
            area_km2=float(areas[classes == i].sum()),
        )
        for i, label in enumerate(labels)
    ]
    habitat = float(areas[rounded > 0].sum())
    for row in rows:
        row["percent_of_positive_habitat"] = (
            100 * row["area_km2"] / habitat
            if habitat > 0 and row["density_class"] != "0"
            else None
        )
    rows.extend(
        [
            dict(
                density_class="positive_habitat_total",
                cells=int((rounded > 0).sum()),
                area_km2=habitat,
            ),
            dict(
                density_class="potential_high_density_gt_0.300",
                cells=int((rounded > 0.300).sum()),
                area_km2=float(areas[rounded > 0.300].sum()),
            ),
            dict(
                density_class="prediction_domain_total_including_zero",
                cells=int(rounded.size),
                area_km2=float(areas.sum()),
            ),
        ]
    )
    return rounded, classes, pd.DataFrame(rows)


def main(config):
    cfg = read_json(config)
    root = Path(cfg["output"]) / "area"
    root.mkdir(parents=True, exist_ok=True)
    results = []
    base_profile = None
    occupied = None
    for region, path in cfg["region_masks"].items():
        mask, p = read_raster(path)
        projected_metres(p)
        if p["transform"].b != 0 or p["transform"].d != 0:
            raise ValueError("Use an unrotated historical analysis grid.")
        if not np.allclose(
            [abs(p["transform"].a), abs(p["transform"].e)],
            [100, 100],
            rtol=0,
            atol=1e-8,
        ):
            raise ValueError("Area grid must use the declared 100 m resolution.")
        inside = np.isfinite(mask) & (mask > 0)
        if base_profile is None:
            base_profile = p
            occupied = np.zeros(mask.shape, bool)
        check_same_grid([base_profile, p])
        if np.any(inside & occupied):
            raise ValueError("Regional area masks overlap.")
        occupied |= inside
        cells = np.flatnonzero((np.isfinite(mask) & (mask > 0)).ravel())
        d = pd.read_csv(
            Path(cfg["output"]) / "bootstrap" / region / "grid_predictions.csv",
            float_precision="round_trip",
        )
        d = validate_grid_predictions(d, cells, p, region)
        area = pixel_areas_km2(
            p, cells, cfg.get("area_method"), cfg.get("area_equal_area_crs")
        )
        rounded, classes, summary = area_summary(d.predicted.to_numpy(), area)
        density_map = np.full(mask.shape, np.nan)
        class_map = np.full(mask.shape, np.nan)
        sd_map = np.full(mask.shape, np.nan)
        density_map.ravel()[cells] = rounded
        class_map.ravel()[cells] = classes
        sd_map.ravel()[cells] = d.sd_boot
        write_raster(root / f"{region}_density_rounded.tif", density_map, p)
        write_raster(root / f"{region}_density_class.tif", class_map, p)
        write_raster(root / f"{region}_sd_boot.tif", sd_map, p)
        results.append(summary.assign(region=region))
    pd.concat(results, ignore_index=True).to_csv(
        root / "density_class_areas.csv", index=False, float_format="%.10f"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)
