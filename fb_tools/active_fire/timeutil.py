"""Time helpers for detections: local time, day bins and overpass grouping. Times are stored in UTC."""
import numpy as np
import pandas as pd


def to_local(gdf, tz: str, *, col: str = "acq_datetime", out: str = "acq_local"):
    """Copy of ``gdf`` with ``out`` = ``col`` converted to time zone ``tz`` (e.g. ``"America/Denver"``)."""
    return gdf.assign(**{out: gdf[col].dt.tz_convert(tz)})


def day_of(times: pd.Series, tz=None) -> pd.Series:
    """Calendar day of each time as a naive midnight timestamp: UTC days, or local days when ``tz`` is given."""
    return times.dt.tz_convert(tz or "UTC").dt.tz_localize(None).dt.normalize()


def assign_overpass(gdf, *, gap_min: float = 30, by: str = "satellite"):
    """
    Group detections into overpasses.

    Within each ``by`` group (one satellite), detections are ordered by time and a new
    overpass starts when the gap to the previous detection exceeds ``gap_min``.
    FIRMS times are to the minute, so one pass can carry two timestamps; consecutive
    VIIRS orbits are ~100 minutes apart. Satellites are never merged.

    Returns
    -------
    GeoDataFrame
        Copy with ``overpass_id`` (int, unique across groups) and ``overpass_time``
        (earliest ``acq_datetime`` in the overpass). Row order and index are kept.
    """
    out = gdf.copy()
    o = out[[by, "acq_datetime"]].reset_index(drop=True).sort_values([by, "acq_datetime"], kind="stable")
    new = (o["acq_datetime"].diff() > pd.Timedelta(minutes=gap_min)) | (o[by] != o[by].shift())
    oid = new.cumsum() - 1
    out["overpass_id"] = oid.sort_index().to_numpy(np.int64)
    out["overpass_time"] = o["acq_datetime"].groupby(oid).transform("min").sort_index().to_numpy()
    out["overpass_time"] = pd.to_datetime(out["overpass_time"], utc=True)
    return out
