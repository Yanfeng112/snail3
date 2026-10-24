"""Validate scientific input contracts before any expensive model fitting."""

from __future__ import annotations
from collections.abc import Iterable
import numpy as np
import pandas as pd


def require_columns(data: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    """Reject missing or ambiguous fields rather than silently changing the sample."""
    missing = set(columns) - set(data.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")
    if data.columns.duplicated().any():
        raise ValueError(f"{label}: duplicate column names")


def require_identifiers(data: pd.DataFrame, columns: list[str], label: str) -> None:
    require_columns(data, columns, label)
    if data.empty:
        raise ValueError(f"{label}: empty input table")
    if data[columns].isna().any().any():
        raise ValueError(f"{label}: missing identifiers")
    for column in columns:
        if data[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{label}: blank {column}")
    if data.duplicated(columns).any():
        raise ValueError(f"{label}: duplicate key {columns}")


def require_numeric(data: pd.DataFrame, columns: list[str], label: str) -> None:
    require_columns(data, columns, label)
    for column in columns:
        values = pd.to_numeric(data[column], errors="raise").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{label}: nonfinite {column}")


def validate_observations(data: pd.DataFrame) -> None:
    require_identifiers(data, ["site_id", "period"], "environment observations")
    require_numeric(data, ["x", "y", "observed"], "environment observations")
    if data.groupby("site_id")[["x", "y"]].nunique().gt(1).any().any():
        raise ValueError(
            "A site changes coordinates between periods; assign a new site ID."
        )


def validate_site_split(split: pd.DataFrame, observations: pd.DataFrame) -> None:
    require_identifiers(split, ["site_id"], "site split")
    require_columns(split, ["set", "fold"], "site split")
    if set(split.site_id.astype(str)) != set(observations.site_id.astype(str)):
        raise ValueError("Site split and observations do not contain the same network.")
    if not split["set"].isin(["development", "test"]).all():
        raise ValueError("Unknown site partition.")
    development = split.loc[split["set"] == "development"]
    test = split.loc[split["set"] == "test"]
    if development.empty or test.empty:
        raise ValueError(
            "Both development sites and independent test sites are required."
        )
    folds = pd.to_numeric(development.fold, errors="raise").to_numpy(float)
    if (
        not np.isfinite(folds).all()
        or np.any(folds < 1)
        or np.any(folds != np.floor(folds))
    ):
        raise ValueError("Development fold numbers must be positive integers.")
    if len(np.unique(folds)) < 2:
        raise ValueError("At least two development folds are required.")


def validate_density_inputs(survey, external, regions) -> None:
    """Check identity, count scale and region membership before raster extraction."""
    require_identifiers(survey, ["survey_id"], "survey")
    require_identifiers(external, ["quadrat_id"], "external quadrats")
    require_columns(survey, ["region"], "survey")
    require_columns(external, ["region", "marshland_id"], "external quadrats")
    require_numeric(survey, ["x", "y", "density"], "survey")
    require_numeric(external, ["x", "y", "live_snails"], "external quadrats")
    allowed = set(regions)
    for label, data in [("survey", survey), ("external", external)]:
        if data.region.isna().any() or not data.region.isin(allowed).all():
            raise ValueError(
                f"{label}: missing/unknown region; rows must not be discarded."
            )
        if set(data.region) != allowed:
            raise ValueError(f"{label}: one or more study regions have no records.")
    if (survey.density < 0).any():
        raise ValueError("Negative observed density.")
    counts = external.live_snails.to_numpy(float)
    if np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError("live_snails must contain nonnegative integer quadrat counts.")
    if external.marshland_id.isna().any():
        raise ValueError("Missing external marshland ID.")
    if external.groupby("marshland_id").region.nunique().gt(1).any():
        raise ValueError("A marshland ID is shared by different regions.")
    for marshland, frame in external.groupby("marshland_id"):
        if frame.duplicated(["x", "y"]).any():
            raise ValueError(
                f"{marshland}: repeated quadrat coordinates; inspect survey-grid layout."
            )
    sizes = external.groupby(["region", "marshland_id"]).size()
    if not sizes.eq(100).all():
        raise ValueError(
            "Each external marshland requires all 100 quadrats, including zeros."
        )
    marshlands = external.groupby("region").marshland_id.nunique()
    if not marshlands.eq(3).all():
        raise ValueError("Each region requires three independent external marshlands.")
    if set(survey.survey_id) & set(external.quadrat_id):
        raise ValueError(
            "Survey and external quadrat IDs overlap; inspect source independence."
        )


def require_within_mask(data, mask, profile, crs, label) -> None:
    from python.raster_io import sample_array

    values = sample_array(mask, profile, data.x, data.y, crs)
    if np.any(~np.isfinite(values) | (values <= 0)):
        raise ValueError(
            f"{label}: observations lie outside their declared prediction mask."
        )
