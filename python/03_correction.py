"""Site-grouped correction selection without using held-out station observations."""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from python.common import idw, metrics
from python.contracts import validate_observations, validate_site_split, require_numeric
from python.environment_safeguards import constrain_environment


def split_sites(observations, seed=42, folds=5):
    ids = np.sort(observations.site_id.unique())
    dev, test = train_test_split(ids, test_size=0.30, random_state=seed)
    if len(dev) < folds:
        raise ValueError("Too few development sites for correction folds.")
    out = pd.DataFrame(
        {
            "site_id": ids,
            "set": np.where(np.isin(ids, dev), "development", "test"),
            "fold": -1,
        }
    )
    for fold, (_, val) in enumerate(
        KFold(folds, shuffle=True, random_state=seed).split(dev), 1
    ):
        out.loc[out.site_id.isin(dev[val]), "fold"] = fold
    return out


def correction_coefficients(stations, method):
    o, p = stations.observed.to_numpy(float), stations.downscaled.to_numpy(float)
    if not np.isfinite(o).all() or not np.isfinite(p).all():
        raise ValueError(
            "Missing station/raster pairs; use a predeclared common valid-site mask."
        )
    if method == "GDA":
        coeff = o - p
    elif method == "GRA":
        if np.any(abs(p) < 1e-12):
            raise ValueError(
                "Zero denominator in GRA; historical handling rule is required."
            )
        coeff = o / p
    elif method == "Sokol":
        if np.any(abs(p + 10) < 1e-12):
            raise ValueError("Zero denominator in shifted Sokol ratio.")
        coeff = (o + 10) / (p + 10)
    else:
        raise ValueError("Unknown correction method.")
    return stations.assign(coefficient=coeff)


def correction_basis(stations, method, period, time_mode):
    if time_mode == "period":
        use = stations.loc[stations.period.astype(str) == str(period)]
        if use.empty:
            raise ValueError(
                "No calibration stations for this period; select and document temporal transfer explicitly."
            )
        coeff = correction_coefficients(use, method)
        return coeff.groupby(["site_id", "x", "y"], as_index=False).coefficient.mean()
    if time_mode == "pooled_site":
        coeff = correction_coefficients(stations, method)
        return coeff.groupby(["site_id", "x", "y"], as_index=False).coefficient.mean()
    raise ValueError(
        "Choose period or pooled_site; this temporal-transfer detail is not fully reported."
    )


def apply_correction(raw, coeff, method):
    if method == "GDA":
        return raw + coeff
    if method == "GRA":
        return raw * coeff
    if method == "Sokol":
        return np.maximum(coeff * (raw + 10) - 10, 0)
    raise ValueError(f"Unknown correction method: {method}")


def predict_correction(
    train, queries, method, time_mode, *, physical_bounds=None, physical_policy="error"
):
    out = np.empty(len(queries))
    for period in queries.period.astype(str).unique():
        ix = np.flatnonzero(queries.period.astype(str).to_numpy() == period)
        c = correction_basis(train, method, period, time_mode)
        surface = idw(
            c[["x", "y"]], c.coefficient, queries.iloc[ix][["x", "y"]], k=12, power=2
        )
        corrected = apply_correction(
            queries.iloc[ix].downscaled.to_numpy(), surface, method
        )
        out[ix], _ = constrain_environment(
            corrected, period, physical_bounds, physical_policy
        )
    return out


def select_correction(
    candidates, split, time_mode, *, physical_bounds=None, physical_policy="error"
):
    if not candidates:
        raise ValueError("No correction candidates supplied.")
    reference = None
    for name, observations in candidates.items():
        validate_observations(observations)
        validate_site_split(split, observations)
        require_numeric(observations, ["downscaled"], f"candidate {name}")
        if {"fold", "set"} & set(observations.columns):
            raise ValueError(
                "Candidate observations must not override site-split metadata."
            )
        identity = (
            observations[["site_id", "period", "x", "y", "observed"]]
            .sort_values(["site_id", "period"])
            .reset_index(drop=True)
        )
        if reference is None:
            reference = identity
        else:
            try:
                pd.testing.assert_frame_equal(reference, identity, check_dtype=False)
            except AssertionError as error:
                raise ValueError(
                    "Candidates do not share the same observations and coordinates."
                ) from error
    scores = []
    paired = []
    for name, observations in candidates.items():
        d = observations.merge(split, on="site_id", validate="many_to_one")
        if len(d) != len(observations):
            raise ValueError("Unassigned station records.")
        for method in ["GDA", "GRA", "Sokol"]:
            for fold in sorted(d.loc[d["set"] == "development", "fold"].unique()):
                tr = d.loc[(d["set"] == "development") & (d.fold != fold)]
                va = d.loc[(d["set"] == "development") & (d.fold == fold)].copy()
                pred = predict_correction(
                    tr,
                    va,
                    method,
                    time_mode,
                    physical_bounds=physical_bounds,
                    physical_policy=physical_policy,
                )
                scores.append(
                    dict(
                        candidate=name,
                        method=method,
                        fold=int(fold),
                        **metrics(va.observed, pred),
                    )
                )
                paired.append(
                    va[["site_id", "period", "observed"]].assign(
                        candidate=name, method=method, fold=fold, predicted=pred
                    )
                )
    scores = pd.DataFrame(scores)
    winner = (
        scores.groupby(["candidate", "method"], sort=True)
        .rmse.mean()
        .sort_values(kind="stable")
        .index[0]
    )
    d = candidates[winner[0]].merge(split, on="site_id", validate="many_to_one")
    tr = d.loc[d["set"] == "development"].copy()
    test = d.loc[d["set"] == "test"].copy()
    test["predicted"] = predict_correction(
        tr,
        test,
        winner[1],
        time_mode,
        physical_bounds=physical_bounds,
        physical_policy=physical_policy,
    )
    return winner, scores, pd.concat(paired, ignore_index=True), tr, test
