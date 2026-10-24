"""Audit paired five-model predictions and optionally plot descriptive diagnostics."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from python.common import metrics, write_json
from python.contracts import require_columns, require_identifiers, require_numeric

MODELS = ("OK", "RF", "RF_OK", "SPDE_uncorrected", "SPDE_corrected")
KEYS = ["region", "repeat_id", "fold", "survey_id", "set"]


def audit_predictions(data, *, models=MODELS, expected_repeats=10, expected_folds=10):
    """Refuse model comparisons when validation keys or outcomes are not identical."""
    require_columns(data, KEYS + ["model", "observed", "predicted"], "CV predictions")
    require_identifiers(data, KEYS + ["model"], "CV predictions")
    require_numeric(data, ["observed", "predicted"], "CV predictions")
    if not data["set"].isin(["training", "validation"]).all():
        raise ValueError("Unknown prediction set.")
    if set(data.model) != set(models):
        raise ValueError("Missing or unexpected model in paired comparison.")
    reference = None
    for model in models:
        current = (
            data.loc[data.model == model, KEYS + ["observed"]]
            .sort_values(KEYS)
            .reset_index(drop=True)
        )
        if reference is None:
            reference = current
        elif not current.equals(reference):
            raise ValueError(
                f"Validation records or observed values differ for {model}."
            )
    actual = set(zip(data.repeat_id, data.fold))
    expected = {
        (r, f)
        for r in range(1, expected_repeats + 1)
        for f in range(1, expected_folds + 1)
    }
    if actual != expected:
        raise ValueError("Missing/unexpected repeated CV evaluations.")
    if (data.observed < 0).any():
        raise ValueError("Negative observed density in CV output.")
    for region, part in data.groupby("region"):
        baseline = part.loc[part.model == models[0]]
        expected_ids = set(baseline.survey_id)
        # Internal consistency cannot prove that a row was omitted from every fold;
        # R's primary audit additionally checks the original survey table.
        truth_counts = baseline.groupby("survey_id")["observed"].nunique()
        if truth_counts.gt(1).any():
            raise ValueError("A survey has inconsistent observed density across folds.")
        for repeat_id in range(1, expected_repeats + 1):
            repeat_data = baseline.loc[baseline.repeat_id == repeat_id]
            validation = repeat_data.loc[repeat_data["set"] == "validation"]
            if (
                validation.survey_id.duplicated().any()
                or set(validation.survey_id) != expected_ids
            ):
                raise ValueError(
                    "Each survey must be validated exactly once per repeat."
                )
            for fold in range(1, expected_folds + 1):
                evaluation = repeat_data.loc[repeat_data.fold == fold]
                training_ids = set(
                    evaluation.loc[evaluation["set"] == "training", "survey_id"]
                )
                validation_ids = set(
                    evaluation.loc[evaluation["set"] == "validation", "survey_id"]
                )
                if (
                    not training_ids
                    or not validation_ids
                    or training_ids & validation_ids
                    or training_ids | validation_ids != expected_ids
                ):
                    raise ValueError(
                        "A CV evaluation has overlapping/incomplete train-validation sets."
                    )
    records = []
    for keys, frame in data.groupby(
        ["region", "repeat_id", "fold", "model", "set"], sort=True
    ):
        records.append(
            dict(
                zip(["region", "repeat_id", "fold", "model", "set"], keys),
                **metrics(frame.observed, frame.predicted),
            )
        )
    summary = pd.DataFrame(records)
    counts = summary.groupby(["region", "model", "set"]).size()
    if not counts.eq(expected_repeats * expected_folds).all():
        raise ValueError("A region/model/set lacks one or more CV evaluations.")
    return summary


def plot_diagnostics(data, summary, destination):
    """Use descriptive plots; repeated folds are not independent replicates for a t-test."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(destination)
    validation = data.loc[data["set"] == "validation"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    proposed = validation.loc[validation.model == "SPDE_corrected"]
    axes[0].scatter(
        proposed.observed, proposed.predicted, s=9, alpha=0.35, color="#39818A"
    )
    low = min(proposed.observed.min(), proposed.predicted.min())
    high = max(proposed.observed.max(), proposed.predicted.max())
    axes[0].plot([low, high], [low, high], color="#8A8A8A", linewidth=1)
    axes[0].set(
        xlabel="Observed density",
        ylabel="Predicted density",
        title="Held-out predictions",
    )
    groups = [
        summary.loc[
            (summary.model == m) & (summary["set"] == "validation"), "rmse"
        ].to_numpy()
        for m in MODELS
    ]
    axes[1].boxplot(
        groups,
        labels=["OK", "RF", "RF+OK", "SPDE raw", "SPDE corrected"],
        showfliers=False,
    )
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set(ylabel="RMSE (individuals / 0.11 m²)", title="Descriptive fold errors")
    fig.savefig(destination / "cv_diagnostics.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", default="outputs/cv/all_paired_predictions.csv"
    )
    parser.add_argument("--output", default="outputs/prediction_audit")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    data = pd.read_csv(args.predictions, dtype={"survey_id": str})
    summary = audit_predictions(data)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    summary.to_csv(destination / "checked_fold_metrics.csv", index=False)
    write_json(
        destination / "audit.json",
        {
            "passed": True,
            "rows": len(data),
            "evaluations": len(summary),
            "inference": "Descriptive errors; no independence of repeated folds is assumed.",
        },
    )
    if args.plots:
        plot_diagnostics(data, summary, destination)


if __name__ == "__main__":
    main()
