"""
OSM road and trail network extraction for SDI inputs.

Downloads highway features from OpenStreetMap via osmnx for a given area
of interest, splitting results into a roads GeoDataFrame (used for
accessibility calculations) and a trails GeoDataFrame (used for trail
density calculations in the penetrability sub-index).

For large landscapes that exceed the Overpass API area limit, use the
``chunk_by`` parameter to split the download into smaller sub-regions:

    # Chunk by US counties (recommended for multi-county study areas)
    counties = fetch_counties(aoi, state_fips='35')
    roads, trails = fetch_osm_roads(aoi, out_crs=26913, chunk_by=counties)

    # Chunk by individual treatment polygon
    roads, trails = fetch_osm_roads(aoi, out_crs=26913, chunk_by='polygon')
"""

import io
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import box as shapely_box
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

# Census Bureau 20m cartographic county boundaries (all US, ~2.5 MB).
_CENSUS_COUNTIES_URL = (
    "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_20m.zip"
)


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


def _query_one(polygon, cache):
    """
    Query OSM for a single WGS-84 polygon and return raw line features.

    Parameters
    ----------
    polygon : shapely geometry
        Query polygon in WGS-84 (EPSG:4326).
    cache : bool
        Enable osmnx HTTP caching.

    Returns
    -------
    GeoDataFrame or None
        Raw features with osmnx MultiIndex (element_type, osmid) intact,
        or ``None`` if the query returns no results.
    """
    import osmnx as ox

    ox.settings.use_cache = cache
    ox.settings.requests_timeout = 300

    try:
        raw = ox.features_from_polygon(polygon, tags={"highway": True})
    except Exception as exc:
        # osmnx raises InsufficientResponseError when the area returns nothing
        if "InsufficientResponseError" in type(exc).__name__ or "Response Error" in str(exc):
            return None
        raise

    if raw is None or raw.empty:
        return None

    # Keep only line geometries
    raw = raw[raw.geometry.geom_type.isin({"LineString", "MultiLineString"})].copy()
    if raw.empty:
        return None

    return raw


# ── Public API ───────────────────────────────────────────────────────────────

def fetch_counties(aoi, state_fips=None):
    """
    Fetch US county boundaries overlapping *aoi* from the Census Bureau.

    Downloads the 20m cartographic county shapefile (~2.5 MB) from the
    Census FTP, extracts it in memory, and returns the counties that
    intersect the *aoi* bounding box.

    Parameters
    ----------
    aoi : GeoDataFrame or shapely geometry
        Area of interest (any CRS). The bounding box is used for the spatial
        query, so counties that overlap the bbox but not the aoi itself may
        be included. Pass the result to :func:`fetch_osm_roads` as
        ``chunk_by`` to use county-level chunked downloads.
    state_fips : str or list of str, optional
        One or more 2-digit state FIPS codes to restrict results (e.g.
        ``'35'`` for New Mexico, ``['35', '49']`` for NM + Utah). ``None``
        returns all counties intersecting the bounding box.

    Returns
    -------
    GeoDataFrame
        County boundaries in WGS-84 (EPSG:4326) with columns:
        ``GEOID``, ``NAME``, ``STATEFP``, ``geometry``.

    Raises
    ------
    RuntimeError
        If the download fails or no counties are found for the AOI.

    Examples
    --------
    >>> counties = fetch_counties(my_aoi, state_fips='35')
    >>> roads, trails = fetch_osm_roads(my_aoi, out_crs=26913, chunk_by=counties)
    """
    # Resolve AOI bounding box in WGS-84
    if isinstance(aoi, gpd.GeoDataFrame):
        bounds = aoi.to_crs(4326).total_bounds       # [minx, miny, maxx, maxy]
    else:
        bounds = tuple(aoi.bounds)                    # (minx, miny, maxx, maxy)
    bbox_geom = shapely_box(*bounds)

    # Download the Census 20m cartographic county shapefile for all US (~2.5 MB).
    # Using a direct shapefile download is more reliable than the TIGER REST API
    # which has scale-dependent layer visibility that can return empty results.
    print("Downloading Census county boundaries (~2.5 MB)...")
    resp = requests.get(_CENSUS_COUNTIES_URL, timeout=120)
    resp.raise_for_status()

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            zf.extractall(tmpdir)
        shp = next(Path(tmpdir).glob("*.shp"))
        counties_all = gpd.read_file(shp).to_crs(4326)

    # Clip to AOI bounding box
    counties = counties_all[counties_all.geometry.intersects(bbox_geom)].copy()

    # Optionally filter to specific state(s)
    if state_fips is not None:
        if isinstance(state_fips, str):
            state_fips = [state_fips]
        state_fips = [str(s).zfill(2) for s in state_fips]
        counties = counties[counties["STATEFP"].isin(state_fips)].copy()

    if counties.empty:
        raise RuntimeError(
            "No counties found for the provided AOI"
            + (f" and state_fips={state_fips}" if state_fips else "")
            + ". Check that the AOI is within the conterminous US."
        )

    keep = [c for c in ["GEOID", "NAME", "STATEFP", "geometry"] if c in counties.columns]
    counties = counties[keep].reset_index(drop=True)
    print(f"  Found {len(counties)} counties")
    return counties


def fetch_osm_roads(aoi, out_crs=None, cache=True, chunk_by=None):
    """
    Download OSM road and trail features for an area of interest.

    Uses osmnx to fetch all highway features within *aoi*, then splits
    them into a roads GeoDataFrame (motorway through track) and a trails
    GeoDataFrame (path, footway, bridleway, track).

    For large areas that exceed the Overpass API limit, use *chunk_by* to
    split the download into smaller sub-regions (counties or individual
    polygons). Results from all chunks are concatenated and deduplicated
    by OSM feature ID before returning.

    Parameters
    ----------
    aoi : GeoDataFrame or shapely geometry
        Area of interest.  Any CRS is accepted; the query is issued in
        WGS-84 (EPSG:4326).  If a GeoDataFrame, the union of all
        geometries is used as the query polygon (or individual polygons
        when ``chunk_by='polygon'``).
    out_crs : int, str, or CRS, optional
        Target CRS for output GeoDataFrames (EPSG code or pyproj string).
        Defaults to the CRS of *aoi* when *aoi* is a GeoDataFrame, or
        EPSG:4326 when *aoi* is a bare shapely geometry.
    cache : bool
        Enable osmnx HTTP caching (default ``True``).  Set ``False`` to
        force a fresh download.
    chunk_by : None, ``'polygon'``, or GeoDataFrame
        Sub-region strategy for large AOIs:

        ``None``
            Single query using the union of all *aoi* geometries. Use only
            for small areas that stay within the Overpass area limit.

        ``'polygon'``
            One query per row of the *aoi* GeoDataFrame. Useful when the
            AOI is already a set of non-overlapping treatment units.

        GeoDataFrame
            One query per row of the provided regions GeoDataFrame (e.g.,
            county boundaries from :func:`fetch_counties`). Only regions
            that intersect the *aoi* bounding box are queried.

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
    >>> # Small area — single query
    >>> roads, trails = fetch_osm_roads(my_aoi, out_crs=26913)

    >>> # Large landscape — chunk by county
    >>> counties = fetch_counties(my_aoi, state_fips='35')
    >>> roads, trails = fetch_osm_roads(my_aoi, out_crs=26913, chunk_by=counties)

    >>> # Chunk by individual polygon
    >>> roads, trails = fetch_osm_roads(my_aoi, out_crs=26913, chunk_by='polygon')
    """
    try:
        import osmnx as ox
    except ImportError as exc:
        raise ImportError(
            "osmnx is required for fetch_osm_roads. "
            "Install with: conda install -c conda-forge osmnx"
        ) from exc

    # Resolve output CRS and WGS-84 query polygon / GeoDataFrame
    if isinstance(aoi, gpd.GeoDataFrame):
        if out_crs is None:
            out_crs = aoi.crs
        aoi_wgs84    = aoi.to_crs(4326)
        query_polygon = unary_union(aoi_wgs84.geometry)
    else:
        if out_crs is None:
            out_crs = 4326
        aoi_wgs84    = None
        query_polygon = aoi   # assume already WGS-84

    # ── Build list of chunk polygons ─────────────────────────────────────────

    if chunk_by is None:
        chunks      = [query_polygon]
        chunk_names = [None]

    elif isinstance(chunk_by, gpd.GeoDataFrame):
        regions = chunk_by.to_crs(4326)
        # Keep only regions that intersect the aoi bounding polygon
        mask    = regions.geometry.intersects(query_polygon)
        regions = regions[mask].reset_index(drop=True)
        chunks  = list(regions.geometry)
        # Use NAME column if present, else index
        if "NAME" in regions.columns:
            chunk_names = list(regions["NAME"].astype(str))
        else:
            chunk_names = [str(i) for i in range(len(chunks))]

    elif isinstance(chunk_by, str) and chunk_by == "polygon":
        if aoi_wgs84 is None:
            raise ValueError(
                "chunk_by='polygon' requires aoi to be a GeoDataFrame, "
                "not a bare shapely geometry."
            )
        chunks      = list(aoi_wgs84.geometry)
        chunk_names = [str(i) for i in range(len(chunks))]

    else:
        raise ValueError(
            "chunk_by must be None, 'polygon', or a GeoDataFrame. "
            f"Got: {type(chunk_by)}"
        )

    # ── Query each chunk ─────────────────────────────────────────────────────

    n      = len(chunks)
    pieces = []

    for i, (polygon, name) in enumerate(zip(chunks, chunk_names)):
        label = f"({name})" if name else ""
        print(f"  Querying OSM chunk {i + 1}/{n} {label}...")

        chunk_raw = _query_one(polygon, cache)

        if chunk_raw is None or chunk_raw.empty:
            print(f"    No features found — skipping")
            continue

        print(f"    {len(chunk_raw)} raw features")
        pieces.append(chunk_raw)

    if not pieces:
        raise ValueError("No OSM features found within the provided area of interest.")

    # ── Concatenate and deduplicate by OSM feature ID ─────────────────────────

    raw = pd.concat(pieces)   # osmnx MultiIndex (element_type, osmid) preserved
    raw = raw[~raw.index.duplicated(keep="first")]

    # ── Filter and flatten ───────────────────────────────────────────────────

    raw["highway"] = raw["highway"].apply(_flatten_highway_tag)
    known_types    = _ROAD_HIGHWAY_TYPES | _TRAIL_HIGHWAY_TYPES
    raw            = raw[raw["highway"].isin(known_types)].copy()

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
