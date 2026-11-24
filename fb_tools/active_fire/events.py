"""Fire events (perimeters) and the association of detections with them."""
import geopandas as gpd
import pandas as pd
import requests

_WFIGS = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/{}/FeatureServer"
WFIGS_SERVICES = {
    "current": _WFIGS.format("WFIGS_Interagency_Perimeters_Current"),
    "year_to_date": _WFIGS.format("WFIGS_Interagency_Perimeters_YearToDate"),
    "history": _WFIGS.format("WFIGS_Interagency_Perimeters"),
}


def _feature_service_gdf(url, *, where="1=1", bbox=None, layer=0) -> gpd.GeoDataFrame:
    """
    All features of an ArcGIS Feature Service layer matching ``where`` / ``bbox``, EPSG:4326.

    Pages over object IDs by the layer's ``maxRecordCount``; falls back from pgeojson
    to geojson. Ported from AspenAcres ``get_feature_service_gdf``.
    """
    query = f"{url}/{layer}/query"
    try:
        page = int(requests.get(f"{url}/{layer}?f=pjson", timeout=60).json().get("maxRecordCount", 1000))
    except Exception:
        page = 1000
    prm = {"where": where, "returnIdsOnly": "true", "f": "pjson"}
    if bbox is not None:
        prm.update(geometry=",".join(map(str, bbox)), geometryType="esriGeometryEnvelope", inSR=4326)
    info = requests.get(query, prm, timeout=120).json()
    if "error" in info:
        raise RuntimeError(f"Feature service query failed for {url}: {info['error']}")
    oids, parts = info.get("objectIds") or [], []
    for i in range(0, len(oids), page):
        prm = {"where": f"{info['objectIdFieldName']} in ({','.join(map(str, oids[i:i + page]))})",
               "outFields": "*", "returnGeometry": "true", "outSR": 4326, "f": "pgeojson"}
        try:
            feats = requests.post(query, data=prm, timeout=300).json()["features"]
        except (ValueError, KeyError):
            resp = requests.post(query, data={**prm, "f": "geojson"}, timeout=300)
            try:
                feats = resp.json()["features"]
            except (ValueError, KeyError) as e:
                raise RuntimeError(f"Failed to read features from {query}: {resp.text[:300]}") from e
        if feats:
            parts.append(gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326").dropna(axis=1, how="all"))
    if not parts:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
    return pd.concat(parts, ignore_index=True)


def fetch_wfigs_perimeters(name=None, bbox=None, which: str = "current") -> gpd.GeoDataFrame:
    """
    NIFC WFIGS interagency fire perimeters.

    Parameters
    ----------
    name : str, optional
        Matched anywhere in ``poly_IncidentName`` or ``attr_IncidentName``
        (case as stored; WFIGS names are usually title case).
    bbox : tuple, optional
        Lon/lat ``(W, S, E, N)``.
    which : {"current", "year_to_date", "history"}
        The WFIGS service: active perimeters, this calendar year, or all years.

    Returns
    -------
    GeoDataFrame
        EPSG:4326. Dates are epoch milliseconds, e.g.
        ``pd.to_datetime(gdf["attr_FireDiscoveryDateTime"], unit="ms", utc=True)``.
    """
    if which not in WFIGS_SERVICES:
        raise ValueError(f"which must be one of {tuple(WFIGS_SERVICES)}; got {which!r}.")
    where = "1=1"
    if name:
        n = name.replace("'", "''")
        where = f"poly_IncidentName LIKE '%{n}%' OR attr_IncidentName LIKE '%{n}%'"
    gdf = _feature_service_gdf(WFIGS_SERVICES[which], where=where, bbox=bbox)
    print(f"WFIGS {which}: {len(gdf)} perimeter(s)")
    return gdf


def fired_events(fired_fp, *, ids=None, bbox=None, years=None, layer=None) -> gpd.GeoDataFrame:
    """
    Read FIRED event polygons with their ignition and last burn dates.

    Parameters
    ----------
    fired_fp : path-like
        FIRED *events* GeoPackage (``id``, ``ig_date``, ``last_date``, ``ig_year``).
    ids : sequence, optional
        Event ids to read.
    bbox : tuple, optional
        Lon/lat ``(W, S, E, N)``.
    years : int or (first, last), optional
        Ignition year(s), inclusive.
    layer : str, optional

    Returns
    -------
    GeoDataFrame
        ``id, ig_date, last_date, geometry`` in the file's CRS; dates as datetimes.
    """
    import pyogrio
    from shapely.geometry import box

    where = []
    if ids is not None:
        where.append("id IN ({})".format(",".join(repr(i) if isinstance(i, str) else str(int(i)) for i in ids)))
    if years is not None:
        y0, y1 = (years, years) if isinstance(years, int) else years
        where.append(f"ig_year >= {int(y0)} AND ig_year <= {int(y1)}")
    if bbox is not None:
        crs = pyogrio.read_info(fired_fp, layer=layer)["crs"]
        bbox = tuple(gpd.GeoSeries([box(*bbox)], crs="EPSG:4326").to_crs(crs).total_bounds)
    g = pyogrio.read_dataframe(fired_fp, layer=layer, where=" AND ".join(where) or None, bbox=bbox,
                               columns=["id", "ig_date", "last_date"])
    for c in ("ig_date", "last_date"):
        g[c] = pd.to_datetime(g[c])
    return g


def associate_detections(
    detections,
    events,
    *,
    id_col=None,
    buffer_m: float = 0,
    start_col=None,
    end_col=None,
    pad_before_days: float = 0,
    pad_after_days: float = 0,
    hull: bool = False,
):
    """
    Assign detections to events by location and, optionally, time.

    A detection belongs to an event when its centre falls inside the event geometry
    (buffered by ``buffer_m``; the convex hull when ``hull``) and its UTC time falls in
    ``[start − pad_before, end + 1 day + pad_after)``. Event dates are read as UTC
    days and ``end`` is inclusive; a missing ``end`` leaves the window open, for
    active fires. Without ``start_col`` / ``end_col`` the match is spatial only.

    Parameters
    ----------
    detections : GeoDataFrame
        Any ``event_id`` already present (the bbox tag from ``fetch_firms``) is
        replaced, and rows repeated across events are reduced to one per
        ``detection_id`` first.
    events : GeoDataFrame
    id_col : str, optional
        Event id column; the index when omitted.
    buffer_m : float
    start_col, end_col : str, optional
    pad_before_days, pad_after_days : float
    hull : bool

    Returns
    -------
    GeoDataFrame
        Matched detections with ``event_id`` and ``multi_event``. A detection matching
        more than one event appears once per event, flagged ``multi_event``.
        Unmatched detections are dropped (count printed).
    """
    geom = events.geometry.convex_hull if hull else events.geometry
    if buffer_m:
        geom = geom.to_crs(geom.estimate_utm_crs()).buffer(buffer_m)
    zones = gpd.GeoDataFrame(
        {"event_id": (events[id_col] if id_col else events.index).to_numpy()},
        geometry=geom.to_crs(detections.crs).to_numpy(), crs=detections.crs,
    )
    for name, col in (("_start", start_col), ("_end", end_col)):
        zones[name] = pd.to_datetime(events[col], utc=True).dt.normalize().to_numpy() if col else pd.NaT
        zones[name] = pd.to_datetime(zones[name], utc=True)

    det = detections.drop(columns=["event_id", "multi_event"], errors="ignore")
    if "detection_id" in det:
        det = det.drop_duplicates("detection_id")
    det = det.reset_index(drop=True)
    j = gpd.sjoin(det, zones, predicate="intersects", how="inner").drop(columns="index_right")

    t = j["acq_datetime"]
    start = j["_start"] - pd.Timedelta(days=pad_before_days)
    end = j["_end"] + pd.Timedelta(days=1 + pad_after_days)
    j = j[(start.isna() | (t >= start)) & (end.isna() | (t < end))].drop(columns=["_start", "_end"])
    j["multi_event"] = j.index.duplicated(keep=False)
    print(f"associate: {j.index.nunique()} of {len(det)} detections matched to "
          f"{j['event_id'].nunique()} event(s); {j['multi_event'].sum()} multi-event rows")
    return j.sort_values(["event_id", "acq_datetime"], kind="stable").reset_index(drop=True)
