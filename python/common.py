"""Shared numerical operations for prediction, interpolation and evaluation."""

from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def metrics(observed, predicted):
    o, p = np.asarray(observed, float), np.asarray(predicted, float)
    if (
        o.shape != p.shape
        or o.size == 0
        or not np.isfinite(o).all()
        or not np.isfinite(p).all()
    ):
        raise ValueError(
            "Expected the same nonempty finite paired observations; handle missingness explicitly."
        )
    e = p - o
    sst = np.sum((o - o.mean()) ** 2)
    nz = o != 0
    return dict(
        n=int(o.size),
        bias=float(e.mean()),
        relative_bias=float(p.sum() / o.sum() - 1) if o.sum() != 0 else None,
        rmse=float(np.sqrt(np.mean(e**2))),
        mae=float(np.mean(abs(e))),
        mre_percent=float(np.mean(abs(e[nz] / o[nz])) * 100) if nz.any() else None,
        mre_n=int(nz.sum()),
        r2=float(1 - np.sum(e**2) / sst) if sst > 0 else None,
        sse=float(np.sum(e**2)),
    )


def idw(xy, values, query, k=12, power=2, chunk=100000):
    if (
        not isinstance(k, (int, np.integer))
        or k < 1
        or not np.isfinite(power)
        or power <= 0
        or not isinstance(chunk, (int, np.integer))
        or chunk < 1
    ):
        raise ValueError("IDW requires positive k, power and chunk size.")
    xy, values, query = (
        np.asarray(xy, float),
        np.asarray(values, float),
        np.asarray(query, float),
    )
    if xy.ndim != 2 or xy.shape[1] != 2 or query.ndim != 2 or query.shape[1] != 2:
        raise ValueError(
            "IDW requires two-dimensional projected coordinates in metres."
        )
    if (
        len(values) != len(xy)
        or not len(values)
        or not np.isfinite(xy).all()
        or not np.isfinite(values).all()
        or not np.isfinite(query).all()
    ):
        raise ValueError("IDW received missing or incompatible data.")
    unique, inv = np.unique(xy, axis=0, return_inverse=True)
    values = np.bincount(inv, weights=values) / np.bincount(inv)
    tree = cKDTree(unique)
    out = np.empty(len(query))
    for first in range(0, len(query), chunk):
        d, ix = tree.query(query[first : first + chunk], k=min(k, len(unique)))
        if d.ndim == 1:
            d, ix = d[:, None], ix[:, None]
        if not np.isfinite(d).all():
            raise ValueError("Nonfinite IDW distances; check coordinate scale.")
        exact = d[:, 0] == 0
        y = np.empty(len(d))
        y[exact] = values[ix[exact, 0]]
        other = ~exact
        # Relative weights are bounded by one, avoiding overflow at small distances.
        weights = (d[other, :1] / d[other]) ** power
        y[other] = (weights * values[ix[other]]).sum(axis=1) / weights.sum(axis=1)
        out[first : first + len(y)] = y
    return out


def round_half_up(values, places=3):
    a = np.asarray(values, float)
    q = Decimal(1).scaleb(-places)
    return np.array(
        [
            (
                float(Decimal(str(v)).quantize(q, rounding=ROUND_HALF_UP))
                if np.isfinite(v)
                else np.nan
            )
            for v in a.flat
        ]
    ).reshape(a.shape)


def classify_density(density):
    raw = np.asarray(density, float)
    if np.any(raw[np.isfinite(raw)] < 0):
        raise ValueError(
            "Negative unrounded predictions require an explicit handling rule."
        )
    a = round_half_up(raw)
    if np.any(a[np.isfinite(a)] < 0):
        raise ValueError(
            "Negative predictions: recover and explicitly configure the historical handling rule first."
        )
    out = np.full(a.shape, -1, dtype=np.int16)
    good = np.isfinite(a)
    out[good] = np.searchsorted([0, 0.020, 0.100, 0.300, 0.500], a[good], side="left")
    return a, out


def back_transform(z, negative_policy):
    z = np.asarray(z, float)
    if not np.isfinite(z).all():
        raise ValueError("Back-transformation requires finite linear predictors.")
    try:
        with np.errstate(over="raise", invalid="raise"):
            y = np.expm1(z)
    except FloatingPointError as error:
        raise ValueError(
            "Back-transformed density overflowed; inspect the fitted model."
        ) from error
    if negative_policy == "clip_zero":
        return np.maximum(y, 0)
    if negative_policy == "retain":
        return y
    raise ValueError(
        "Specify negative_policy as retain or clip_zero after checking historical analysis."
    )


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
