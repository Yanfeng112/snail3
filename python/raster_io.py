"""Raster alignment and extraction. All distances for IDW use projected metres."""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import (
    reproject,
    Resampling,
    calculate_default_transform,
    transform_bounds,
)
from rasterio.transform import from_origin
from pyproj import CRS, Transformer


def read_raster(path):
    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError(
                "Select one indicator/time band explicitly before reading a raster."
            )
        return src.read(1, masked=True).astype(float).filled(np.nan), src.profile.copy()


def write_raster(path, a, profile):
    a = np.asarray(a, float)
    if a.shape != (profile["height"], profile["width"]):
        raise ValueError("Raster array shape differs from its georeferencing profile.")
    if np.isinf(a).any():
        raise ValueError("Infinite raster values must not silently become NoData.")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    p = profile.copy()
    p.update(
        driver="GTiff", count=1, dtype="float64", nodata=np.nan, compress="deflate"
    )
    with rasterio.open(path, "w", **p) as dst:
        dst.write(a.astype("float64"), 1)


def projected_metres(profile):
    crs = CRS(profile["crs"])
    if not crs.is_projected or any(
        abs(ax.unit_conversion_factor - 1) > 1e-10 for ax in crs.axis_info[:2]
    ):
        raise ValueError(
            "A projected CRS in metres is required; recover the historical analysis CRS."
        )


def xy_centres(profile):
    r, c = np.indices((profile["height"], profile["width"]))
    x, y = rasterio.transform.xy(profile["transform"], r, c, offset="center")
    return np.column_stack([np.asarray(x).ravel(), np.asarray(y).ravel()])


def warp_to(source, target_profile, kind="continuous", method=None):
    if kind not in ["continuous", "categorical"]:
        raise ValueError("Unknown raster variable kind.")
    with rasterio.open(source) as src:
        if src.count != 1:
            raise ValueError("Source raster must have one explicitly selected band.")
        if method is None:
            approx, _, _ = calculate_default_transform(
                src.crs, target_profile["crs"], src.width, src.height, *src.bounds
            )
            coarser = abs(
                target_profile["transform"].a * target_profile["transform"].e
            ) > abs(approx.a * approx.e) * (1 + 1e-6)
            method = (
                ("mode" if coarser else "nearest")
                if kind == "categorical"
                else ("average" if coarser else "bilinear")
            )
        out = np.full(
            (target_profile["height"], target_profile["width"]), np.nan, dtype="float64"
        )
        reproject(
            source=src.read(1, masked=True).astype(float).filled(np.nan),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=target_profile["transform"],
            dst_crs=target_profile["crs"],
            dst_nodata=np.nan,
            resampling=getattr(Resampling, method),
        )
    return out


def geographic_profile(fine_profile, degrees):
    if not np.isfinite(degrees) or not 0 < degrees <= 180:
        raise ValueError("Invalid candidate degree resolution.")
    # Global origin alignment avoids candidate-grid changes when clipping boundaries change.
    b = rasterio.transform.array_bounds(
        fine_profile["height"], fine_profile["width"], fine_profile["transform"]
    )
    west, south, east, north = transform_bounds(
        fine_profile["crs"], "EPSG:4326", *b, densify_pts=21
    )
    west = np.floor(west / degrees) * degrees
    north = np.ceil(north / degrees) * degrees
    p = fine_profile.copy()
    p.update(
        crs=rasterio.crs.CRS.from_epsg(4326),
        transform=from_origin(west, north, degrees, degrees),
        width=int(np.ceil((east - west) / degrees)),
        height=int(np.ceil((north - south) / degrees)),
    )
    return p


def sample_array(a, profile, x, y, xy_crs):
    xx, yy = Transformer.from_crs(xy_crs, profile["crs"], always_xy=True).transform(
        np.asarray(x), np.asarray(y)
    )
    r, c = rasterio.transform.rowcol(profile["transform"], xx, yy)
    r, c = np.asarray(r), np.asarray(c)
    if np.any((r < 0) | (r >= a.shape[0]) | (c < 0) | (c >= a.shape[1])):
        raise ValueError("Sampling location lies outside raster extent.")
    return a[r, c]


def check_same_grid(profiles):
    first = profiles[0]
    for p in profiles[1:]:
        if (
            p["crs"] != first["crs"]
            or p["transform"] != first["transform"]
            or (p["height"], p["width"]) != (first["height"], first["width"])
        ):
            raise ValueError("Rasters do not use the same grid.")
