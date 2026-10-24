"""Build identically indexed corrected/uncorrected survey, grid and external tables."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from python.contracts import validate_density_inputs, require_within_mask
from python.common import read_json, write_json
from python.raster_io import (
    read_raster,
    check_same_grid,
    projected_metres,
    xy_centres,
    sample_array,
)


def main(config):
    cfg = read_json(config)
    projected_metres({"crs": cfg["xy_crs"]})
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    survey = pd.read_csv(cfg["survey"], dtype={"survey_id": str})
    if (
        survey.survey_id.duplicated().any()
        or not np.isfinite(survey.density).all()
        or (survey.density < 0).any()
    ):
        raise ValueError(
            "Invalid survey IDs or densities; retain valid zero-density habitats."
        )
    external = pd.read_csv(
        cfg["external"], dtype={"quadrat_id": str, "marshland_id": str}
    )
    if (
        external.quadrat_id.duplicated().any()
        or (external.live_snails < 0).any()
        or not np.isfinite(external.live_snails).all()
    ):
        raise ValueError("Invalid external quadrats.")
    if not external.groupby("marshland_id").size().eq(100).all():
        raise ValueError(
            "Each external marshland must contain all 100 quadrats, including zeros."
        )
    validate_density_inputs(survey, external, cfg["region_masks"])
    overlap = set(survey.survey_id) & set(external.quadrat_id)
    if overlap:
        raise ValueError(
            "Survey and external IDs overlap; verify independence before running."
        )
    base_profile = None
    used = None
    for region, mask_file in cfg["region_masks"].items():
        mask, profile = read_raster(mask_file)
        projected_metres(profile)
        if not np.allclose(
            [abs(profile["transform"].a), abs(profile["transform"].e)], [100, 100]
        ):
            raise ValueError("Prediction grid must use 100 m cells.")
        if base_profile is None:
            base_profile = profile
            used = np.zeros(mask.shape, bool)
        check_same_grid([base_profile, profile])
        inside = np.isfinite(mask) & (mask > 0)
        if np.any(inside & used):
            raise ValueError("Regional prediction masks overlap.")
        used |= inside
        cells = np.flatnonzero(inside.ravel())
        xy = xy_centres(profile)[cells]
        grid = pd.DataFrame(dict(cell_id=cells, region=region, x=xy[:, 0], y=xy[:, 1]))
        for kind in ["corrected", "uncorrected"]:
            records = {
                "survey": survey.loc[survey.region == region].copy(),
                "external": external.loc[external.region == region].copy(),
                "grid": grid.copy(),
            }
            for table in ["survey", "external"]:
                require_within_mask(
                    records[table], mask, profile, cfg["xy_crs"], f"{region}/{table}"
                )
            for name, item in cfg["covariates"].items():
                a, p = read_raster(item[kind])
                check_same_grid([profile, p])
                for table, d in records.items():
                    d[name] = (
                        a.ravel()[cells]
                        if table == "grid"
                        else sample_array(a, p, d.x, d.y, cfg["xy_crs"])
                    )
                    if not np.isfinite(d[name]).all():
                        raise ValueError(
                            f"{region}/{kind}/{table}/{name}: missing covariate values; use an explicit common domain."
                        )
                    if item.get("levels"):
                        d[name] = d[name].map(
                            lambda v: str(int(v)) if float(v).is_integer() else str(v)
                        )
                        if not d[name].isin(item["levels"]).all():
                            raise ValueError(f"{name}: unknown categorical code.")
            for table, d in records.items():
                d.to_csv(out / f"{region}_{table}_{kind}.csv", index=False)
    write_json(
        out / "extraction_metadata.json",
        dict(
            xy_crs=cfg["xy_crs"],
            density_unit="individuals/0.11 m2",
            covariates=cfg["covariates"],
            region_masks=cfg["region_masks"],
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)
