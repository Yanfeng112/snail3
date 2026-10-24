"""Functions for temporally matched temperature, precipitation and inundation inputs."""

import calendar
import numpy as np
import pandas as pd
from python.common import idw
from python.contracts import require_identifiers, require_numeric


def calendar_dates(values, label):
    parsed = pd.to_datetime(values, errors="raise")
    if parsed.isna().any() or parsed.dt.tz is not None:
        raise ValueError(
            f"{label}: missing dates or timezone-bearing values; supply local calendar dates."
        )
    return parsed.dt.normalize()


def temperature_indicators(hourly, local_timezone="Asia/Shanghai"):
    # Columns: site_id,time_utc,temp_c. ERA5 Kelvin must be converted before calling.
    require_identifiers(hourly, ["site_id", "time_utc"], "hourly temperature")
    require_numeric(hourly, ["temp_c"], "hourly temperature")
    d = hourly.copy()
    d["time"] = pd.to_datetime(d.time_utc, utc=True).dt.tz_convert(local_timezone)
    if (
        d.time.isna().any()
        or (d.time.dt.minute != 0).any()
        or (d.time.dt.second != 0).any()
        or (d.time.dt.microsecond != 0).any()
    ):
        raise ValueError("Temperature records must use valid, exact hourly timestamps.")
    if d.duplicated(["site_id", "time"]).any():
        raise ValueError("Duplicate hourly temperature records.")
    d["date"] = d.time.dt.date
    groups = d.groupby(["site_id", "date"]).temp_c
    daily = groups.agg(["min", "max", "mean", "count"]).reset_index()
    if (daily["count"] != 24).any() or not np.isfinite(d.temp_c).all():
        raise ValueError(
            "Incomplete daily temperature: apply and document the historical missing-data rule."
        )
    daily["date"] = pd.to_datetime(daily.date)
    daily["year"], daily["month"] = daily.date.dt.year, daily.date.dt.month
    result = []
    for (site, year), x in daily.groupby(["site_id", "year"]):
        if len(x) != 365 + calendar.isleap(int(year)):
            raise ValueError(f"Incomplete year: {site}, {year}")
        result.append(
            dict(
                site_id=site,
                year=int(year),
                tmean=x["mean"].mean(),
                tmin_jan=x.loc[x.month == 1, "min"].mean(),
                tmax_aug=x.loc[x.month == 8, "max"].mean(),
            )
        )
    return pd.DataFrame(result)


def seasonal_precipitation(monthly, season_months, reduction):
    # Exact winter-year convention and sum-versus-mean are intentionally explicit.
    if reduction not in ["sum", "mean"] or sorted(
        m for s in season_months.values() for m in s
    ) != list(range(1, 13)):
        raise ValueError(
            "Provide all 12 months once and specify sum or mean for seasonal aggregation."
        )
    require_identifiers(monthly, ["site_id", "year", "month"], "monthly precipitation")
    require_numeric(monthly, ["precip_mm", "year", "month"], "monthly precipitation")
    if (monthly.precip_mm < 0).any():
        raise ValueError("Negative precipitation must be resolved before aggregation.")
    if (
        np.any(monthly.year != np.floor(monthly.year))
        or not monthly.year.between(1, 9999).all()
    ):
        raise ValueError("Precipitation years must be valid integer calendar years.")
    d = monthly.copy()
    if (
        d.duplicated(["site_id", "year", "month"]).any()
        or not np.isfinite(d.precip_mm).all()
    ):
        raise ValueError("Duplicate/missing monthly precipitation.")
    result = []
    for (site, year), x in d.groupby(["site_id", "year"]):
        if sorted(x.month.tolist()) != list(range(1, 13)):
            raise ValueError("Missing precipitation month.")
        row = dict(site_id=site, year=int(year))
        for name, months in season_months.items():
            z = x.loc[x.month.isin(months), "precip_mm"]
            row[name] = float(z.sum() if reduction == "sum" else z.mean())
        result.append(row)
    return pd.DataFrame(result)


def video_inundation(video):
    # site_id,date,hour,inundated; annotations already quality-controlled.
    require_identifiers(video, ["site_id", "date", "hour"], "video annotations")
    d = video.copy()
    d["date"] = calendar_dates(d.date, "video annotations")
    if d.duplicated(["site_id", "date", "hour"]).any():
        raise ValueError(
            "Duplicate video observations after calendar-date normalisation."
        )
    if (
        not d.inundated.isin([0, 1]).all()
        or not d.hour.isin([8, 12, 16]).all()
        or d.duplicated(["site_id", "date", "hour"]).any()
    ):
        raise ValueError(
            "Video needs exactly one binary annotation per scheduled observation."
        )
    daily = d.groupby(["site_id", "date"]).inundated.agg(["sum", "count"]).reset_index()
    if (daily["count"] != 3).any():
        raise ValueError(
            "Missing video observations: do not silently classify them as dry."
        )
    daily["year"] = pd.to_datetime(daily.date).dt.year
    daily["wet"] = (daily["sum"] >= 2).astype(int)
    result = []
    for (site, year), x in daily.groupby(["site_id", "year"]):
        if len(x) != 365 + calendar.isleap(int(year)):
            raise ValueError(f"Incomplete video year: {site}, {year}")
        result.append(dict(site_id=site, year=int(year), observed=int(x.wet.sum())))
    return pd.DataFrame(result)


def water_level_inundation(cells, water, datum_verified):
    # cells: cell_id,x,y,elevation_1985_m; water: station_id,x,y,date,level_1985_m.
    # Input water levels are the daily 08:00 records, after documented datum conversion.
    from scipy.spatial import cKDTree

    require_identifiers(cells, ["cell_id"], "inundation cells")
    require_identifiers(water, ["station_id", "date"], "daily water levels")
    require_numeric(cells, ["x", "y", "elevation_1985_m"], "inundation cells")
    require_numeric(water, ["x", "y", "level_1985_m"], "daily water levels")
    if not datum_verified:
        raise ValueError(
            "Convert both vertical datums to the 1985 datum before water-level comparison."
        )
    water = water.copy()
    water["date"] = calendar_dates(water.date, "water levels")
    if water.duplicated(["station_id", "date"]).any():
        raise ValueError("Duplicate water levels after calendar-date normalisation.")
    stations = water[["station_id", "x", "y"]].drop_duplicates()
    if (
        stations.station_id.duplicated().any()
        or water.duplicated(["station_id", "date"]).any()
    ):
        raise ValueError("Inconsistent stations or duplicate daily levels.")
    if (
        not np.isfinite(water.level_1985_m).all()
        or not np.isfinite(cells.elevation_1985_m).all()
    ):
        raise ValueError("Missing elevation or water level.")
    _, ix = cKDTree(stations[["x", "y"]].to_numpy()).query(cells[["x", "y"]].to_numpy())
    station_ids = stations.station_id.to_numpy()[ix]
    d = water.copy()
    d["year"] = pd.to_datetime(d.date).dt.year
    result = []
    for year, w in d.groupby("year"):
        for station in np.unique(station_ids):
            levels = w.loc[w.station_id == station, "level_1985_m"].to_numpy()
            if len(levels) != 365 + calendar.isleap(int(year)):
                raise ValueError(f"Incomplete daily water levels: {station}, {year}")
            rows = np.flatnonzero(station_ids == station)
            for start in range(0, len(rows), 10000):
                rr = rows[start : start + 10000]
                days = (
                    levels[:, None]
                    > cells.iloc[rr].elevation_1985_m.to_numpy()[None, :]
                ).sum(axis=0)
                for j, n in zip(rr, days):
                    result.append(
                        dict(
                            cell_id=cells.iloc[j].cell_id,
                            x=cells.iloc[j].x,
                            y=cells.iloc[j].y,
                            year=int(year),
                            inundation_days=int(n),
                        )
                    )
    return pd.DataFrame(result)


def interpolate_inundation(points, query_xy, k, power):
    # Supply interpolation neighbour count and power explicitly.
    return idw(points[["x", "y"]], points.inundation_days, query_xy, k=k, power=power)
