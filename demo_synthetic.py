"""Run a small RF example and threshold audit with synthetic data only.

This demonstrates executable data flow. It neither simulates INLA/CNN output nor
reproduces the manuscript. Toy blocks are 1 km, explicitly unlike the study's
20 km blocks, to keep the example small and quick.
"""

from __future__ import annotations
import importlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from python.common import metrics, write_json
from python.spatial_validation import split_source_windows


def main():
    destination = Path("demo_output")
    destination.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    n = 240
    x, y = rng.uniform(0, 6000, size=(2, n))
    ndvi = rng.uniform(0.1, 0.9, n)
    elevation = rng.normal(25, 5, n)
    latent = -0.25 + 0.7 * ndvi - 0.015 * elevation + 0.08 * np.sin(x / 800)
    observed = np.maximum(np.expm1(latent + rng.normal(0, 0.025, n)), 0)
    features = pd.DataFrame({"ndvi": ndvi, "elevation": elevation})
    groups = np.array([f"{int(a // 1000)}:{int(b // 1000)}" for a, b in zip(x, y)])
    predictions = []
    for fold, (train, validation) in enumerate(
        GroupKFold(5).split(features, groups=groups), 1
    ):
        model = RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=5,
            max_features=1,
            random_state=42,
            n_jobs=1,
        )
        model.fit(features.iloc[train], np.log1p(observed[train]))
        predicted = np.expm1(model.predict(features.iloc[validation]))
        predictions.append(
            pd.DataFrame(
                {
                    "synthetic_id": validation,
                    "fold": fold,
                    "observed": observed[validation],
                    "predicted": predicted,
                }
            )
        )
    paired = pd.concat(predictions, ignore_index=True)
    paired.to_csv(destination / "synthetic_rf_predictions.csv", index=False)
    rows, columns = np.indices((60, 60))
    _, _, split_audit = split_source_windows(
        rows.ravel(),
        columns.ravel(),
        mode="buffered_blocks",
        block_cells=15,
        buffer_cells=4,
        test_fraction=0.2,
    )
    area = importlib.import_module("python.06_density_area")
    _, _, areas = area.area_summary([0, 0.00049, 0.0005, 0.0205, 0.3005, 0.5005], 0.01)
    areas.to_csv(destination / "synthetic_threshold_areas.csv", index=False)
    write_json(
        destination / "demo_summary.json",
        {
            "synthetic_only": True,
            "model": "Random forest only; no CNN or INLA results are generated.",
            "toy_block_size_m": 1000,
            "metrics": metrics(paired.observed, paired.predicted),
            "cnn_split_geometry_audit": split_audit,
        },
    )
    print("Synthetic example completed:", destination)


if __name__ == "__main__":
    main()
