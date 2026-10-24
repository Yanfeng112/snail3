"""CNN coarse-to-fine reconstruction following Appendix 2, section 2.5."""

import os

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
import numpy as np
from pathlib import Path
from python.spatial_validation import split_source_windows
from scipy.ndimage import map_coordinates
from pyproj import Transformer
from python.common import metrics, write_json
from python.environment_safeguards import filtered_idw, constrain_environment
from python.raster_io import (
    read_raster,
    warp_to,
    geographic_profile,
    xy_centres,
    projected_metres,
    write_raster,
    check_same_grid,
)


def covariate_cube(aux, profile, names, landcover_levels):
    channels = []
    continuous = []
    for name in names:
        values = warp_to(
            aux[name], profile, "categorical" if name == "landcover" else "continuous"
        )
        if name == "landcover":
            if np.any(np.isfinite(values) & ~np.isin(values, landcover_levels)):
                raise ValueError("Land-cover codes absent from declared codebook.")
            channels.extend(
                [
                    np.where(
                        np.isfinite(values), (values == level).astype(float), np.nan
                    )
                    for level in landcover_levels
                ]
            )
            continuous.extend([False] * len(landcover_levels))
        else:
            channels.append(values)
            continuous.append(True)
    return np.stack(channels, axis=-1), np.asarray(continuous)


def windows(cube):
    if min(cube.shape[:2]) < 5:
        raise ValueError("Candidate raster must contain a 5x5 window.")
    return np.moveaxis(
        np.lib.stride_tricks.sliding_window_view(cube, (5, 5), axis=(0, 1)), 2, -1
    )


def build_cnn(channels):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    reg = tf.keras.regularizers.l2(0.0001)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((5, 5, channels)),
            tf.keras.layers.Conv2D(
                32,
                3,
                strides=1,
                padding="valid",
                activation="relu",
                kernel_regularizer=reg,
            ),
            tf.keras.layers.Conv2D(
                64,
                3,
                strides=1,
                padding="valid",
                activation="relu",
                kernel_regularizer=reg,
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=reg),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1, kernel_regularizer=reg),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse")
    return model


def predict_map(cube, model, xmean, xsd, ymean, ysd, batch=4096):
    w = windows(cube)
    out = np.full(cube.shape[:2], np.nan)
    # Edges remain missing. Input auxiliaries must extend two fine cells beyond the mask.
    width = w.shape[1]
    total = w.shape[0] * width
    for start in range(0, total, batch):
        indices = np.arange(start, min(start + batch, total))
        rows, columns = np.divmod(indices, width)
        patches = w[rows, columns]
        valid = np.isfinite(patches).all(axis=(1, 2, 3))
        if valid.any():
            scaled = (patches[valid] - xmean) / xsd
            pred = (
                model.predict(
                    scaled.astype("float32"), verbose=0, batch_size=32
                ).ravel()
                * ysd
                + ymean
            )
            out[rows[valid] + 2, columns[valid] + 2] = pred
    return out


def interpolate_residual(residual, coarse_profile, fine_profile):
    # Bilinear interpolation of SOURCE-RASTER residuals, distinct from station bias correction.
    query = xy_centres(fine_profile)
    x, y = Transformer.from_crs(
        fine_profile["crs"], coarse_profile["crs"], always_xy=True
    ).transform(query[:, 0], query[:, 1])
    inv = ~coarse_profile["transform"]
    col = inv.a * np.asarray(x) + inv.b * np.asarray(y) + inv.c
    row = inv.d * np.asarray(x) + inv.e * np.asarray(y) + inv.f
    rr, cc = np.asarray(row) - 0.5, np.asarray(col) - 0.5
    good = np.isfinite(residual)
    numerator = map_coordinates(
        np.where(good, residual, 0), [rr, cc], order=1, mode="constant", cval=0
    )
    denominator = map_coordinates(
        good.astype(float), [rr, cc], order=1, mode="constant", cval=0
    )
    z = np.divide(
        numerator, denominator, out=np.full(len(query), np.nan), where=denominator > 0
    )
    return z.reshape(fine_profile["height"], fine_profile["width"])


def fit_candidate(spec, names, degrees, output):
    import tensorflow as tf

    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _, fine_profile = read_raster(spec["template_100m"])
    projected_metres(fine_profile)
    if not spec.get("processing_mask"):
        raise ValueError(
            "An explicit processing_mask is required; a grid template is not a mask."
        )
    mask, mask_profile = read_raster(spec["processing_mask"])
    check_same_grid([fine_profile, mask_profile])
    domain = np.isfinite(mask) & (mask > 0)
    if not domain.any():
        raise ValueError("Empty environmental processing mask.")
    if not np.allclose(
        [abs(fine_profile["transform"].a), abs(fine_profile["transform"].e)], [100, 100]
    ):
        raise ValueError("Output template is not 100 metres.")
    coarse = geographic_profile(fine_profile, degrees)
    patches, targets, groups = [], [], []
    source_rows, source_columns = [], []
    inputs = []
    for period in spec["periods"]:
        cube, continuous = covariate_cube(
            period["aux"], coarse, names, spec["landcover_levels"]
        )
        response = warp_to(period["source"], coarse)
        w = windows(cube)
        yy = response[2:-2, 2:-2]
        valid = np.isfinite(w).all(axis=(2, 3, 4)) & np.isfinite(yy)
        rr, cc = np.where(valid)
        patches.append(w[rr, cc])
        targets.append(yy[rr, cc])
        # Same source-grid cell has the same internal split across all years.
        groups.extend([f"{r+2}:{c+2}" for r, c in zip(rr, cc)])
        source_rows.extend(rr + 2)
        source_columns.extend(cc + 2)
        inputs.append((period, cube, response))
    x, y = np.concatenate(patches), np.concatenate(targets)
    ix, jx, split_audit = split_source_windows(
        source_rows,
        source_columns,
        mode=spec.get("cnn_split_mode"),
        test_fraction=spec["internal_validation_fraction"],
        block_cells=spec.get("cnn_block_cells"),
        buffer_cells=spec.get("cnn_buffer_cells", 4),
    )
    write_json(root / "internal_split_audit.json", split_audit)
    if np.std(y[ix]) == 0:
        raise ValueError("Constant source-raster response.")
    mean = x[ix].mean(axis=(0, 1, 2))
    sd = x[ix].std(axis=(0, 1, 2), ddof=1)
    mean[~continuous] = 0
    sd[~continuous] = 1
    sd[sd == 0] = 1
    ym, ys = y[ix].mean(), y[ix].std(ddof=1)
    model = build_cnn(x.shape[-1])
    stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0.0001, patience=20, restore_best_weights=True
    )
    history = model.fit(
        ((x[ix] - mean) / sd).astype("float32"),
        ((y[ix] - ym) / ys).astype("float32"),
        validation_data=(
            ((x[jx] - mean) / sd).astype("float32"),
            ((y[jx] - ym) / ys).astype("float32"),
        ),
        batch_size=32,
        epochs=200,
        callbacks=[stop],
        verbose=2,
    )
    model.save(root / "cnn.keras")
    np.savez(
        root / "preprocessing.npz",
        xmean=mean,
        xsd=sd,
        ymean=ym,
        ysd=ys,
        train_index=ix,
        validation_index=jx,
        groups=np.asarray(groups),
    )
    write_json(
        root / "source_validation.json",
        metrics(
            y[jx],
            model.predict(((x[jx] - mean) / sd).astype("float32"), verbose=0).ravel()
            * ys
            + ym,
        ),
    )
    write_json(root / "training_history.json", history.history)
    results = {}
    for period, cube, response in inputs:
        fitted = predict_map(cube, model, mean, sd, ym, ys)
        residual = response - fitted
        write_raster(root / f"source_residual_{period['key']}.tif", residual, coarse)
        fine_cube, _ = covariate_cube(
            period["aux"], fine_profile, names, spec["landcover_levels"]
        )
        preliminary = predict_map(
            fine_cube, model, mean, sd, ym, ys
        ) + interpolate_residual(residual, coarse, fine_profile)
        final, outlier_audit = filtered_idw(
            preliminary,
            xy_centres(fine_profile),
            domain,
            filter_outliers=spec["filter_outliers"],
            k=spec["final_idw_k"],
            power=spec["final_idw_power"],
        )
        final, physical_audit = constrain_environment(
            final,
            period["key"],
            spec.get("physical_bounds"),
            spec["physical_bound_policy"],
        )
        target = root / f"downscaled_{period['key']}.tif"
        write_raster(target, final, fine_profile)
        write_json(root / f"outlier_audit_{period['key']}.json", outlier_audit)
        write_json(root / f"physical_bounds_{period['key']}.json", physical_audit)
        results[period["key"]] = str(target)
    return results
