"""Report all missing settings and inputs before starting CNN or INLA runs."""

from __future__ import annotations
import argparse
from pathlib import Path
from python.common import read_json, write_json


def inspect_configuration(configuration, kind):
    """Return every actionable issue; do not stop at the first null parameter."""
    issues = []

    def issue(path, message):
        issues.append({"severity": "error", "path": path, "message": message})

    def required(key):
        if configuration.get(key) is None:
            issue(key, "Set this parameter to the value used for the analysis.")

    def file(path, label):
        if not path or not Path(path).is_file():
            issue(label, f"Missing input file: {path}")

    if kind == "study":
        for key in [
            "xy_crs",
            "spde_coordinate_divisor",
            "alpha",
            "block_origin_m",
            "fixed_effect_mean",
            "fixed_effect_precision",
            "intercept_mean",
            "intercept_precision",
            "gaussian_precision_hyper",
            "negative_prediction_policy",
            "area_method",
            "bootstrap_variable_selection",
            "baseline_feature_policy",
            "map_prediction_source",
            "unseen_predictor_policy",
            "design_rank_policy",
        ]:
            required(key)
        for name, spec in configuration.get("covariates", {}).items():
            if spec.get("levels") is None and spec.get("transform") not in [
                "identity",
                "log1p",
            ]:
                issue(
                    f"covariates.{name}",
                    "Specify continuous transform or categorical levels.",
                )
        for region in configuration.get("regions", []):
            mesh = configuration.get("meshes", {}).get(region, {})
            if mesh.get("rds"):
                file(mesh["rds"], f"meshes.{region}.rds")
            else:
                file(mesh.get("boundary_file"), f"meshes.{region}.boundary_file")
                for key in ["max_edge_m", "cutoff_m", "offset_m"]:
                    if mesh.get(key) is None:
                        issue(
                            f"meshes.{region}.{key}",
                            "Missing historical mesh parameter.",
                        )
            for table in ["survey", "grid", "external"]:
                for version in ["corrected", "uncorrected"]:
                    path = (
                        Path(configuration.get("tables_directory", "data/processed"))
                        / f"{region}_{table}_{version}.csv"
                    )
                    file(path, f"tables.{region}.{table}.{version}")
    elif kind == "environment":
        for key in [
            "xy_crs",
            "landcover_levels",
            "correction_time_mode",
            "filter_outliers",
            "final_idw_k",
            "final_idw_power",
            "cnn_split_mode",
            "processing_mask",
            "physical_bound_policy",
        ]:
            required(key)
        if configuration.get("cnn_split_mode") == "buffered_blocks":
            required("cnn_block_cells")
        file(configuration.get("template_100m"), "template_100m")
        file(configuration.get("processing_mask"), "processing_mask")
        file(configuration.get("observations"), "observations")
        keys = [str(p.get("key")) for p in configuration.get("periods", [])]
        if not keys or len(keys) != len(set(keys)):
            issue("periods", "Periods must be nonempty and have unique keys.")
        for index, period in enumerate(configuration.get("periods", [])):
            file(period.get("source"), f"periods.{index}.source")
            needed = set(
                v
                for names in configuration.get("auxiliary_combinations", {}).values()
                for v in names
            )
            for name in sorted(needed):
                file(period.get("aux", {}).get(name), f"periods.{index}.aux.{name}")
    elif kind == "extract":
        required("xy_crs")
        for key in ["survey", "external"]:
            file(configuration.get(key), key)
        for name, item in configuration.get("covariates", {}).items():
            for version in ["corrected", "uncorrected"]:
                file(item.get(version), f"covariates.{name}.{version}")
    else:
        raise ValueError("Unknown configuration kind.")
    for name, path in configuration.get("region_masks", {}).items():
        file(path, f"region_masks.{name}")
    if configuration.get("area_method") == "equal_area":
        required("area_equal_area_crs")
    return {"kind": kind, "ready": not issues, "errors": len(issues), "issues": issues}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--kind", required=True, choices=["study", "environment", "extract"]
    )
    parser.add_argument("--output", default="validation/preflight.json")
    args = parser.parse_args()
    report = inspect_configuration(read_json(args.config), args.kind)
    write_json(args.output, report)
    print(f"Preflight: {report['errors']} issue(s); report: {args.output}")
    raise SystemExit(0 if report["ready"] else 2)


if __name__ == "__main__":
    main()
