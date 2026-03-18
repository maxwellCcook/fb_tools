"""
OSM road and trail network extraction for SDI inputs.

Downloads highway features from OpenStreetMap via osmnx for a given area
of interest, splitting results into a roads GeoDataFrame (used for
accessibility calculations) and a trails GeoDataFrame (used for trail
density calculations in the penetrability sub-index).
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.ops import unary_union


# ── Highway type sets ────────────────────────────────────────────────────────

# Road types used for accessibility (Euclidean distance to nearest road).
# Tracks and informal paths are excluded; only maintained road surfaces count.
_ROAD_HIGHWAY_TYPES = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "service", "track",
}

# Integer codes 1–9 assigned to road types (ascending = slower / less maintained).
_ROAD_HIGHWAY_CODES = {
    "motorway": 1,
    "trunk": 2,
    "primary": 3,
    "secondary": 4,
    "tertiary": 5,
    "unclassified": 6,
    "residential": 7,
    "service": 8,
    "track": 9,
}

# Trail types used for penetrability density calculation.
# Track is included here as well (it bridges the road/trail boundary).
_TRAIL_HIGHWAY_TYPES = {"path", "footway", "bridleway", "track"}


# ── Private helpers ──────────────────────────────────────────────────────────

def _flatten_highway_tag(val):
    """Return the first element if *val* is a list, otherwise *val* as-is.

    osmnx occasionally returns OSM highway tags as a Python list when a
    way carries multiple values (e.g. ``['residential', 'service']``).
    This helper normalises the field to a scalar string before comparison.

    Parameters
    ----------
    val : str or list

    Returns
    -------
    str
    """
    if isinstance(val, list):
        return val[0] if val else ""
    return val if isinstance(val, str) else ""


# ── Public API ───────────────────────────────────────────────────────────────

def fetch_osm_roads(aoi, out_crs=None, cache=True):
    """
    Download OSM road and trail features for an area of interest.

    Uses osmnx to fetch all highway features within the bounding polygon
    of *aoi*, then splits them into a roads GeoDataFrame (motorway through
    track) and a trails GeoDataFrame (path, footway, bridleway, track).
    Both outputs are reprojected to *out_crs*.

    Parameters
    ----------
    aoi : GeoDataFrame or shapely geometry
        Area of interest.  Any CRS is accepted; the query is issued in
        WGS-84 (EPSG:4326).  If a GeoDataFrame, the union of all
        geometries is used as the query polygon.
    out_crs : int, str, or CRS, optional
        Target CRS for output GeoDataFrames (EPSG code or pyproj string).
        Defaults to the CRS of *aoi* when *aoi* is a GeoDataFrame, or
        EPSG:4326 when *aoi* is a bare shapely geometry.
    cache : bool
        Enable osmnx HTTP caching (default ``True``).  Set ``False`` to
        force a fresh download.

    Returns
    -------
    roads_gdf : GeoDataFrame
        Line features with columns:

        - ``highway``      — OSM highway tag value (scalar string)
        - ``highway_code`` — integer 1–9 (1 = motorway, 9 = track)
        - ``name``         — road name if present, else NaN
        - ``surface``      — surface type if present, else NaN
        - ``geometry``     — LineString or MultiLineString

    trails_gdf : GeoDataFrame
        Line features with columns ``highway``, ``name``, ``surface``,
        ``geometry`` for path/footway/bridleway/track types.

    Raises
    ------
    ImportError
        If osmnx is not installed.
    ValueError
        If no OSM features are found within *aoi*.

    Notes
    -----
    osmnx must be installed (``conda install -c conda-forge osmnx``).
    The returned GeoDataFrames are ready to pass directly to
    :func:`~fb_tools.suppression.sdi.calculate_sdi`.

    Examples
    --------
    >>> roads, trails = fetch_osm_roads(my_aoi_gdf, out_crs=26913)
    >>> sdi = calculate_sdi(lcp, fl, hua, roads, trails, rtc_path)
    """
    try:
        import osmnx as ox
    except ImportError as exc:
        raise ImportError(
            "osmnx is required for fetch_osm_roads. "
            "Install with: conda install -c conda-forge osmnx"
        ) from exc

    # Resolve output CRS and query polygon
    if isinstance(aoi, gpd.GeoDataFrame):
        if out_crs is None:
            out_crs = aoi.crs
        # Union all geometries, reproject to WGS-84 for the OSM query
        aoi_wgs84 = aoi.to_crs(4326)
        query_polygon = unary_union(aoi_wgs84.geometry)
    else:
        if out_crs is None:
            out_crs = 4326
        # Assume bare shapely geometry is already in WGS-84
        query_polygon = aoi

    ox.settings.use_cache = cache
    ox.settings.requests_timeout = 300

    # All highway types we care about in a single query
    all_types = _ROAD_HIGHWAY_TYPES | _TRAIL_HIGHWAY_TYPES
    highway_regex = "|".join(sorted(all_types))
    custom_filter = f'["highway"~"^({highway_regex})$"]'

    print("Querying OSM for road and trail features...")
    raw = ox.features_from_polygon(
        query_polygon,
        tags={"highway": True},
    )

    if raw.empty:
        raise ValueError("No OSM features found within the provided area of interest.")

    # Keep only line geometries
    raw = raw[raw.geometry.geom_type.isin({"LineString", "MultiLineString"})].copy()

    # Flatten list-valued highway tags and apply type filter
    raw["highway"] = raw["highway"].apply(_flatten_highway_tag)
    known_types = _ROAD_HIGHWAY_TYPES | _TRAIL_HIGHWAY_TYPES
    raw = raw[raw["highway"].isin(known_types)].copy()

    if raw.empty:
        raise ValueError(
            "No recognised highway types found. "
            f"Expected one of: {sorted(known_types)}"
        )

    # Preserve optional attributes; reset index to drop OSM multi-index
    keep_cols = ["highway", "name", "surface", "geometry"]
    available = [c for c in keep_cols if c in raw.columns]
    raw = raw[available].reset_index(drop=True)

    # Reproject to output CRS
    raw = raw.to_crs(out_crs)

    # ── Split into roads and trails ──────────────────────────────────────────
    road_mask  = raw["highway"].isin(_ROAD_HIGHWAY_TYPES)
    trail_mask = raw["highway"].isin(_TRAIL_HIGHWAY_TYPES)

    roads_gdf  = raw[road_mask].copy().reset_index(drop=True)
    trails_gdf = raw[trail_mask].copy().reset_index(drop=True)

    roads_gdf["highway_code"] = roads_gdf["highway"].map(_ROAD_HIGHWAY_CODES)

    print(f"OSM: {len(roads_gdf)} road segments, {len(trails_gdf)} trail segments")

    return roads_gdf, trails_gdf
