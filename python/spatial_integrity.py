"""Spatial contracts used by area exports and raster postprocessing."""

from __future__ import annotations
import numpy as np
from pyproj import CRS, Transformer
from python.raster_io import projected_metres


def cell_centres(profile, cells):
    """Map explicit row-major pixel IDs to centres without materialising a full mesh."""
    indices = np.asarray(cells)
    if (
        indices.ndim != 1
        or not np.isfinite(indices).all()
        or np.any(indices != np.floor(indices))
    ):
        raise ValueError("Pixel IDs must be a finite one-dimensional integer vector.")
    indices = indices.astype(np.int64)
    if np.any(indices < 0) or np.any(indices >= profile["height"] * profile["width"]):
        raise ValueError("Pixel IDs fall outside the declared raster grid.")
    rows, columns = np.divmod(indices, profile["width"])
    transform = profile["transform"]
    x = transform.a * (columns + 0.5) + transform.b * (rows + 0.5) + transform.c
    y = transform.d * (columns + 0.5) + transform.e * (rows + 0.5) + transform.f
    return np.column_stack([x, y])


def validate_grid_predictions(data, cells, profile, region):
    required = {"cell_id", "region", "x", "y", "predicted", "sd_boot"}
    if not required.issubset(data.columns):
        raise ValueError(
            f"Missing grid columns: {sorted(required - set(data.columns))}"
        )
    ids = data.cell_id.to_numpy(float)
    if (
        not np.isfinite(ids).all()
        or np.any(ids != np.floor(ids))
        or data.cell_id.duplicated().any()
    ):
        raise ValueError("Grid IDs must be finite, unique integers.")
    if len(cells) == 0 or set(ids) != set(cells):
        raise ValueError("Prediction rows differ from the nonempty prediction mask.")
    if data.region.isna().any() or not data.region.eq(region).all():
        raise ValueError("Grid prediction region differs from the target mask.")
    ordered = data.set_index("cell_id").loc[cells]
    coordinates = ordered[["x", "y"]].to_numpy(float)
    if not np.isfinite(coordinates).all() or not np.allclose(
        coordinates, cell_centres(profile, cells), rtol=0, atol=1e-6
    ):
        raise ValueError("Grid coordinates do not match their pixel IDs/CRS.")
    values = ordered[["predicted", "sd_boot"]].to_numpy(float)
    if not np.isfinite(values).all() or np.any(values[:, 1] < 0):
        raise ValueError("Predictions and nonnegative bootstrap SD must be finite.")
    return ordered


def pixel_areas_km2(profile, cells, method, equal_area_crs=None):
    """Return stable areas, enforcing metre units on every area calculation CRS."""
    projected_metres(profile)
    transform = profile["transform"]
    if method == "projected_cell":
        # The affine determinant avoids subtracting products of large map coordinates.
        determinant = abs(transform.a * transform.e - transform.b * transform.d)
        if not np.isfinite(determinant) or determinant <= 0:
            raise ValueError("Invalid projected pixel area.")
        return np.full(len(cells), determinant / 1e6)
    if method != "equal_area" or equal_area_crs is None:
        raise ValueError("Specify projected_cell, or equal_area with an explicit CRS.")
    crs = CRS(equal_area_crs)
    projected_metres({"crs": crs})
    operation = crs.coordinate_operation
    name = "" if operation is None else operation.method_name.lower()
    if not any(term in name for term in ["equal area", "equal-area", "equal earth"]):
        raise ValueError("The requested area CRS is not an equal-area projection.")
    convert = Transformer.from_crs(profile["crs"], crs, always_xy=True)
    rows, columns = np.divmod(np.asarray(cells, dtype=np.int64), profile["width"])
    vertices = []
    for dc, dr in [(0, 0), (1, 0), (1, 1), (0, 1)]:
        x = transform.a * (columns + dc) + transform.b * (rows + dr) + transform.c
        y = transform.d * (columns + dc) + transform.e * (rows + dr) + transform.f
        vertices.append(np.column_stack(convert.transform(x, y)))
    vertices = np.stack(vertices, axis=1)
    if not np.isfinite(vertices).all():
        raise ValueError("Nonfinite transformed polygon coordinates.")
    # Translate each polygon to its own origin to avoid cancellation in shoelace sums.
    vertices -= vertices[:, :1, :].copy()
    following = np.roll(vertices, -1, axis=1)
    cross = vertices[..., 0] * following[..., 1] - following[..., 0] * vertices[..., 1]
    area = np.abs(cross.sum(axis=1)) * 0.5 / 1e6
    if np.any(area <= 0) or not np.isfinite(area).all():
        raise ValueError("Invalid equal-area pixel polygon.")
    return area
