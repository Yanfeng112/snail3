"""Preserve prediction support and make physical-value restrictions explicit."""

from __future__ import annotations
import calendar
import numpy as np
from python.common import idw


def filtered_idw(preliminary, points, domain, *, filter_outliers, k, power):
    """Fill removed outliers only inside the original valid prediction support.

    A missing CNN/residual prediction or a cell outside the processing mask is
    not converted into a prediction by unrestricted nearest-neighbour extrapolation.
    """
    if not isinstance(filter_outliers, (bool, np.bool_)):
        raise ValueError("filter_outliers must be an explicit boolean.")
    if (
        not isinstance(k, (int, np.integer))
        or k < 1
        or not np.isfinite(power)
        or power <= 0
    ):
        raise ValueError("Invalid final IDW settings.")
    values = np.asarray(preliminary, float)
    domain = np.asarray(domain, bool)
    if values.shape != domain.shape or np.asarray(points).shape != (values.size, 2):
        raise ValueError("Postprocessing shapes differ.")
    valid = domain & np.isfinite(values)
    if not valid.any():
        raise ValueError("No supported predictions inside the processing mask.")
    q1, q3 = np.quantile(values[valid], [0.25, 0.75])
    keep = valid.copy()
    if filter_outliers:
        keep &= (values >= q1 - 1.5 * (q3 - q1)) & (values <= q3 + 1.5 * (q3 - q1))
    if not keep.any():
        raise ValueError("Outlier filtering removed all supported predictions.")
    output = np.full(values.shape, np.nan)
    # Original retained values need no interpolation; this also preserves exact values.
    output[keep] = values[keep]
    replace = valid & ~keep
    if replace.any():
        output[replace] = idw(
            np.asarray(points)[keep.ravel()],
            values[keep],
            np.asarray(points)[replace.ravel()],
            k=k,
            power=power,
        )
    audit = {
        "domain_cells": int(domain.sum()),
        "supported_cells": int(valid.sum()),
        "unsupported_domain_cells": int((domain & ~np.isfinite(values)).sum()),
        "outliers_replaced": int(replace.sum()),
        "retained_cells": int(keep.sum()),
        "q1": float(q1),
        "q3": float(q3),
        "missing_cells_extrapolated": 0,
    }
    return output, audit


def constrain_environment(values, period, specification, policy):
    """Apply one prespecified physical policy to CV, final test and raster products."""
    if policy not in ["error", "clip"]:
        raise ValueError("Select physical_bound_policy explicitly: error or clip.")
    a = np.asarray(values, float)
    if np.isinf(a).any():
        raise ValueError(
            "Infinite environmental estimates are invalid inputs, not physical outliers."
        )
    lower = specification.get("lower") if specification else None
    upper = specification.get("upper") if specification else None
    if specification and specification.get("annual_days"):
        if (
            len(str(period)) != 4
            or not str(period).isdigit()
            or not 1 <= int(period) <= 9999
        ):
            raise ValueError(
                "Annual inundation bounds require a four-digit year period."
            )
        upper = 365 + calendar.isleap(int(period))
    if any(bound is not None and not np.isfinite(bound) for bound in [lower, upper]):
        raise ValueError("Physical bounds must be finite or null.")
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("Physical lower bound exceeds upper bound.")
    invalid = np.zeros(a.shape, dtype=bool)
    if lower is not None:
        invalid |= np.isfinite(a) & (a < lower)
    if upper is not None:
        invalid |= np.isfinite(a) & (a > upper)
    if policy == "error" and invalid.any():
        raise ValueError(
            f"Nonphysical environmental estimates in {period}: {invalid.sum()} cells/records."
        )
    result = np.clip(
        a, -np.inf if lower is None else lower, np.inf if upper is None else upper
    )
    return result, {
        "period": str(period),
        "policy": policy,
        "lower": lower,
        "upper": upper,
        "changed_values": int(invalid.sum()) if policy == "clip" else 0,
    }
