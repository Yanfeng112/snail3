"""Source-window partitions with explicit protection against overlapping CNN windows."""

from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupShuffleSplit


def split_source_windows(
    rows, columns, *, mode, test_fraction, block_cells=None, buffer_cells=4, seed=42
):
    """Split window centres and keep all repeated years at a centre together.

    A 5x5 window extends two cells from its centre. Two windows can share a
    source pixel when their Chebyshev centre distance is at most four cells.
    The buffered mode excludes such training centres around validation centres.
    Legacy mode is available only for explicitly reproducing the old design.
    """
    xy = np.column_stack([rows, columns]).astype(float)
    if not len(xy) or not np.isfinite(xy).all() or not 0 < test_fraction < 1:
        raise ValueError("Invalid source-window coordinates or validation fraction.")
    if mode == "legacy_cell":
        groups = np.array([f"{r}:{c}" for r, c in xy])
    elif mode == "buffered_blocks":
        if block_cells is None or int(block_cells) != block_cells or block_cells < 5:
            raise ValueError("Specify a CNN block size of at least five source cells.")
        if buffer_cells < 4:
            raise ValueError(
                "A 5x5 CNN needs at least a four-cell centre exclusion buffer."
            )
        groups = np.array(
            [f"{int(r // block_cells)}:{int(c // block_cells)}" for r, c in xy]
        )
    else:
        raise ValueError("Select cnn_split_mode: legacy_cell or buffered_blocks.")
    if len(np.unique(groups)) < 2:
        raise ValueError("Too few occupied source groups for internal validation.")
    train, validation = next(
        GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed).split(
            xy, groups=groups
        )
    )
    original_train_size = len(train)
    distance, _ = cKDTree(np.unique(xy[validation], axis=0)).query(xy[train], p=np.inf)
    if mode == "buffered_blocks":
        train = train[distance > buffer_cells]
        if len(train) < 2:
            raise ValueError(
                "CNN buffer removes all training centres; enlarge development domain."
            )
    distance, _ = cKDTree(np.unique(xy[validation], axis=0)).query(xy[train], p=np.inf)
    audit = {
        "mode": mode,
        "n_train": len(train),
        "n_validation": len(validation),
        "n_excluded_by_buffer": original_train_size - len(train),
        "minimum_chebyshev_distance_cells": float(distance.min()),
        "training_windows_sharing_validation_pixels": int(np.sum(distance <= 4)),
        "same_centres_across_years_kept_together": True,
        "scope": "Internal source-raster early-stopping split, not ground-station test validation.",
    }
    return train, validation, audit
