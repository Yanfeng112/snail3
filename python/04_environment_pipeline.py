"""Run all candidate scales, auxiliary combinations and correction methods for one indicator."""

import argparse
import importlib
from pathlib import Path
import numpy as np
import pandas as pd
from python.contracts import validate_observations, validate_site_split
from python.execution import fingerprint, sha256_file, load_checkpoint, stage_status
from python.common import read_json, write_json, metrics, idw
from python.environment_safeguards import constrain_environment
from python.raster_io import (
    read_raster,
    sample_array,
    projected_metres,
    xy_centres,
    write_raster,
)

cnn = importlib.import_module("python.02_downscaling")
cor = importlib.import_module("python.03_correction")


def main(config):
    spec = read_json(config)
    for key in [
        "xy_crs",
        "landcover_levels",
        "correction_time_mode",
        "final_idw_k",
        "final_idw_power",
        "cnn_split_mode",
        "processing_mask",
        "physical_bound_policy",
    ]:
        if spec.get(key) is None:
            raise ValueError(
                f"Set the required analysis parameter before running: {key}"
            )
    projected_metres({"crs": spec["xy_crs"]})
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=True)
    obs = pd.read_csv(spec["observations"], dtype={"site_id": str, "period": str})
    validate_observations(obs)
    if obs.duplicated(["site_id", "period"]).any():
        raise ValueError(
            "Aggregate observations to one matched indicator per site and period."
        )
    if obs.groupby("site_id")[["x", "y"]].nunique().max().max() > 1:
        raise ValueError("Site coordinates vary.")
    split_path = Path(spec["site_split_file"])
    if split_path.exists():
        split = pd.read_csv(split_path, dtype={"site_id": str})
    else:
        split = cor.split_sites(obs, folds=spec["correction_folds"])
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split.to_csv(split_path, index=False)
    if (
        split.site_id.duplicated().any()
        or set(split.site_id) != set(obs.site_id)
        or not split["set"].isin(["development", "test"]).all()
    ):
        raise ValueError("Saved site split does not match the observation network.")
    if not (split.loc[split["set"] == "development", "fold"] > 0).all():
        raise ValueError("Missing development folds.")
    validate_site_split(split, obs)
    candidates, rastersets = {}, {}
    expected = None
    for degrees in spec["candidate_degrees"]:
        for combination, names in spec["auxiliary_combinations"].items():
            name = f"{degrees}_{combination}"
            candidate_root = root / name
            source_files = [spec["template_100m"], spec["processing_mask"]]
            for period in spec["periods"]:
                source_files.append(period["source"])
                source_files.extend(period["aux"][v] for v in names)
            source_files.extend(str(p) for p in Path("python").glob("*.py"))
            signature = fingerprint(
                dict(specification=spec, names=names, degrees=degrees), source_files
            )
            status_path = candidate_root / "stage_status.json"
            checkpoint = load_checkpoint(status_path, signature)
            if checkpoint is not None:
                rasters = checkpoint["rasters"]
            else:
                with stage_status(status_path, signature, "cnn_downscaling") as status:
                    rasters = cnn.fit_candidate(spec, names, degrees, candidate_root)
                    status["rasters"] = rasters
                    status["outputs"] = [
                        dict(path=str(p), sha256=sha256_file(p))
                        for p in candidate_root.iterdir()
                        if p.is_file() and p != status_path
                    ]
            paired = []
            for period, file in rasters.items():
                a, profile = read_raster(file)
                projected_metres(profile)
                d = obs.loc[obs.period == str(period)].copy()
                if d.empty:
                    continue
                d["downscaled"] = sample_array(a, profile, d.x, d.y, spec["xy_crs"])
                paired.append(d)
            d = (
                pd.concat(paired, ignore_index=True)
                .sort_values(["site_id", "period"])
                .reset_index(drop=True)
            )
            if len(d) != len(obs) or not np.isfinite(d.downscaled).all():
                raise ValueError(
                    "Candidate lacks station predictions; do not change validation rows between candidates."
                )
            keys = list(zip(d.site_id, d.period))
            if expected is not None and keys != expected:
                raise ValueError("Candidate station pairs differ.")
            expected = keys
            candidates[name], rastersets[name] = d, rasters
    winner, scores, cvpairs, train, test = cor.select_correction(
        candidates,
        split,
        spec["correction_time_mode"],
        physical_bounds=spec.get("physical_bounds"),
        physical_policy=spec["physical_bound_policy"],
    )
    scores.to_csv(root / "development_fold_metrics.csv", index=False)
    cvpairs.to_csv(root / "development_predictions.csv", index=False)
    test.to_csv(root / "final_test_predictions.csv", index=False)
    write_json(
        root / "selected_scheme.json",
        dict(
            candidate=winner[0],
            method=winner[1],
            test_metrics=metrics(test.observed, test.predicted),
            calibration_periods=sorted(train.period.astype(str).unique().tolist()),
            prediction_periods=sorted(map(str, rastersets[winner[0]])),
            temporal_scope="Retrospective environmental reconstruction; no prospective temporal validation",
            source_validation_scope="CNN early-stopping validation is distinct from ground-station testing",
        ),
    )
    # Final products retain development-only calibration to preserve the stated final-test independence.
    # Site x/y coordinates and raster query points must use exactly the same projected CRS.
    for period, file in rastersets[winner[0]].items():
        a, p = read_raster(file)
        if str(p["crs"]) != spec["xy_crs"]:
            from pyproj import CRS

            if CRS(p["crs"]) != CRS(spec["xy_crs"]):
                raise ValueError("Observation x/y CRS differs from fine raster CRS.")
        coeff = cor.correction_basis(
            train, winner[1], period, spec["correction_time_mode"]
        )
        grid = xy_centres(p)
        surface = idw(
            coeff[["x", "y"]], coeff.coefficient, grid, k=12, power=2
        ).reshape(a.shape)
        corrected = cor.apply_correction(a, surface, winner[1])
        corrected, physical_audit = constrain_environment(
            corrected,
            period,
            spec.get("physical_bounds"),
            spec["physical_bound_policy"],
        )
        write_json(root / f"corrected_physical_bounds_{period}.json", physical_audit)
        write_raster(root / f"correction_surface_{period}.tif", surface, p)
        write_raster(root / f"corrected_{period}.tif", corrected, p)
    print("Environmental scheme and test predictions saved:", root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)
