"""
Ignition generation for FSPro transmission experiments.

FSPro's ``IgnitionFile`` is a **starting fire perimeter**, not a sampling
domain: every simulated fire begins with that polygon already burned, so burn
probability is ≈ 1.0 inside it by construction.  The functions here build
ignitions that respect that semantics — small day-1 perimeters placed at
locations drawn from an FPA-FOD ignition-likelihood surface — and carry the
per-ignition weight *w_i* needed by the Ager et al. (2014) transmission
estimator

    Expected transmission to j = Σ_i w_i · TF_ij,
    TF_ij ≈ Σ_{p ∈ j} BP_p(ignition at i) × cell_area

Public API
----------
footprint_radius_m, build_ignition_footprints, write_ignition_shapefiles
    Ignition footprint sizing (P2.4) and one-shapefile-per-ignition output
    (P2.5).  FSPro treats a single ``IgnitionFile`` as *one* fire, so N
    ignitions in one shapefile is N simultaneous disjoint fires, not N runs.
ignition_density_surface, sample_density_at_points, write_density_raster
    Kernel-density ignition-likelihood surface from FPA-FOD, masked to
    burnable FBFM40 (P2.2).
check_ignition_clustering
    Monte-Carlo test of Ager's uniform-ignition assumption against a
    burnable-restricted CSR null (P2.2).
wind_cone_half_angle, downwind_cone
    Downwind spread cone replacing the zero-width ray test (P2.3).
select_design_ignitions
    Stratified, density-weighted design-fire selection (P2.3).

Notes
-----
All spatial work happens in the LCP CRS (projected metres).  Inputs may be in
any CRS; they are reprojected at the function boundary.
"""

from pathlib import Path

import numpy as np

# ── Module-level constants ─────────────────────────────────────────────────────

_ACRE_M2 = 4046.8564224

#: Default ignition footprint.  A small realistic day-1 perimeter rather than a
#: point: it matches the vendor sample's IR-perimeter framing and skips the
#: initial-growth phase FSPro is not well parameterized for, while pre-burning
#: little enough area that masking it out of Δ statistics costs almost nothing.
#: P0.3 (half-pixel vs 10 ac vs 100 ac) has not been run — revisit when it is.
DEFAULT_FOOTPRINT_ACRES = 10.0

#: Fraction of non-calm fire-hour wind frequency the default spread cone spans.
DEFAULT_CONE_COVERAGE = 0.5

#: Gaussian sigma, in working-grid cells, used when smoothing the ignition
#: density surface.  The working grid is coarsened to hit this, so the kernel
#: stays a few cells wide no matter how large the bandwidth is in metres.
_SMOOTH_CELLS = 4.0


# ── Ignition footprints (P2.4) ─────────────────────────────────────────────────

def footprint_radius_m(
    acres: "float | None" = DEFAULT_FOOTPRINT_ACRES,
    lcp_res_m: "float | None" = None,
) -> float:
    """
    Radius in metres of a circular ignition footprint of a given area.

    Parameters
    ----------
    acres : float or None
        Footprint area in acres.  ``None`` requests a half-pixel circle (a
        point ignition), which requires *lcp_res_m*.
    lcp_res_m : float, optional
        LCP cell size in metres.  Required when ``acres is None``.  When both
        are given, the returned radius is floored at half a pixel so the
        footprint always covers at least one cell.

    Returns
    -------
    float
        Circle radius in metres.

    Raises
    ------
    ValueError
        If *acres* is None and *lcp_res_m* is not supplied, or if *acres*
        is not positive.
    """
    if acres is None:
        if lcp_res_m is None:
            raise ValueError(
                "footprint_radius_m(acres=None) requests a half-pixel circle "
                "and therefore needs lcp_res_m."
            )
        return float(lcp_res_m) / 2.0

    acres = float(acres)
    if acres <= 0:
        raise ValueError(f"acres must be positive, got {acres}")

    radius = float(np.sqrt(acres * _ACRE_M2 / np.pi))
    if lcp_res_m is not None:
        radius = max(radius, float(lcp_res_m) / 2.0)
    return radius


def build_ignition_footprints(
    points_gdf,
    acres: "float | None" = DEFAULT_FOOTPRINT_ACRES,
    lcp_fp: "str | Path | None" = None,
    lcp_res_m: "float | None" = None,
):
    """
    Buffer ignition points into circular day-1 fire perimeters.

    Parameters
    ----------
    points_gdf : geopandas.GeoDataFrame
        Ignition point locations.  Must be in a projected CRS with metre
        units (normally the LCP CRS).  All non-geometry columns are carried
        through.
    acres : float or None
        Footprint area in acres.  Default 10.  ``None`` gives a half-pixel
        circle and requires *lcp_fp* or *lcp_res_m*.
    lcp_fp : str or Path, optional
        Landscape raster, read only for its cell size.
    lcp_res_m : float, optional
        Cell size in metres, if already known (skips reading *lcp_fp*).

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of *points_gdf* with circular polygon geometry and a
        ``footprint_ac`` column.

    Raises
    ------
    ValueError
        If *points_gdf* has a geographic CRS, where a metre buffer is
        meaningless.
    """
    if points_gdf.crs is not None and points_gdf.crs.is_geographic:
        raise ValueError(
            "build_ignition_footprints needs a projected CRS (metres); got "
            f"{points_gdf.crs.to_string()}. Reproject to the LCP CRS first."
        )

    if lcp_res_m is None and lcp_fp is not None:
        import rasterio
        with rasterio.open(lcp_fp) as src:
            lcp_res_m = min(src.res)

    radius = footprint_radius_m(acres, lcp_res_m)

    out = points_gdf.copy()
    # resolution=64 gives a 256-gon; the default 16 undershoots the true circle
    # area by ~0.08%, which would make footprint_ac a small lie.
    out["geometry"] = out.geometry.buffer(radius, resolution=64)
    out["footprint_ac"] = (out.geometry.area / _ACRE_M2).round(3)
    return out


def write_ignition_shapefiles(
    ign_gdf,
    out_dir: "str | Path",
    prefix: str = "ign",
    id_col: "str | None" = None,
):
    """
    Write one single-feature shapefile per ignition.

    FSPro treats one ``IgnitionFile`` as one fire's starting perimeter, so N
    features in a single shapefile become N simultaneous disjoint fires within
    every simulation rather than N independent design fires.  Each ignition
    therefore needs its own file and its own run.

    Parameters
    ----------
    ign_gdf : geopandas.GeoDataFrame
        One row per ignition, polygon geometry.
    out_dir : str or Path
        Destination directory, created if absent.
    prefix : str
        Filename stem prefix.  Files are ``{prefix}_000.shp``, ``_001`` …
    id_col : str, optional
        Column supplying the numeric suffix.  When omitted, the positional
        index is used.

    Returns
    -------
    list of Path
        Absolute paths, in row order.

    Notes
    -----
    Only the geometry and *id_col* are written.  FSPro reads no attributes
    from ``IgnitionFile``, and the dBase format silently truncates field names
    to 10 characters — so design metadata (``w_i``, ``dist_m``, stratum) stays
    on the returned GeoDataFrame and in the run manifest, where it keeps its
    full name.
    """
    import geopandas as gpd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keep = ["geometry"] + ([id_col] if id_col is not None else [])
    slim = ign_gdf[keep]

    paths = []
    for pos, (_, row) in enumerate(slim.iterrows()):
        ident = int(row[id_col]) if id_col is not None else pos
        shp_path = out_dir / f"{prefix}_{ident:03d}.shp"
        one = gpd.GeoDataFrame([row], crs=ign_gdf.crs)
        one.to_file(shp_path)
        paths.append(shp_path.resolve())

    print(f"  [write_ignition_shapefiles] {len(paths)} single-feature "
          f"shapefile(s) → {out_dir}")
    return paths


# ── Burnable-fuel raster helpers ───────────────────────────────────────────────

def _read_burnable_mask(lcp_fp):
    """
    Read the FBFM40 band and return a boolean burnable mask plus grid metadata.

    Returns
    -------
    dict
        ``mask`` (bool ndarray, True = burnable), ``transform``, ``crs``,
        ``shape``, ``res_m`` (min cell dimension), ``cell_area_m2``.
    """
    import rasterio
    from .lcp import _NB_CODES

    with rasterio.open(lcp_fp) as src:
        fbfm_band = 4
        for i, desc in enumerate(src.descriptions, start=1):
            if desc and "FBFM" in desc.upper():
                fbfm_band = i
                break
        data = src.read(fbfm_band)
        nodata = src.nodata
        meta = {
            "transform": src.transform,
            "crs": src.crs,
            "shape": data.shape,
            "res_m": float(min(src.res)),
            "cell_area_m2": float(abs(src.res[0] * src.res[1])),
        }

    mask = ~np.isin(data, list(_NB_CODES))
    if nodata is not None:
        mask &= data != nodata
    meta["mask"] = mask
    return meta


def _points_from_mask(mask, transform, crs, sub_mask=None):
    """Pixel-centre point GeoDataFrame for every True cell in *mask*."""
    import geopandas as gpd
    from rasterio.transform import xy as rio_xy

    sel = mask if sub_mask is None else (mask & sub_mask)
    rows, cols = np.where(sel)
    if len(rows) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=crs), rows, cols
    xs, ys = rio_xy(transform, rows, cols)
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys), crs=crs)
    return gdf, rows, cols


# ── Ignition density surface (P2.2) ────────────────────────────────────────────

def _scott_bandwidth_m(xy: np.ndarray) -> float:
    """
    Scott's-rule bandwidth for a 2-D point set, in the point CRS units.

    ``h = n**(-1/6) * σ``, with σ the mean per-axis standard deviation.  This
    is the isotropic simplification of ``scipy.stats.gaussian_kde``'s Scott
    factor and is only a starting point — for a sparse large-fire point set
    over a single landscape it collapses toward the domain scale, which is an
    honest statement that the data carry little local information but is
    rarely what you want.  Prefer a bandwidth estimated over the wider region.
    """
    n = len(xy)
    if n < 2:
        raise ValueError("Need at least 2 points to estimate a bandwidth.")
    sigma = float(np.mean(xy.std(axis=0, ddof=1)))
    return float(n ** (-1.0 / 6.0) * sigma)


def ignition_density_surface(
    fod_gdf,
    lcp_fp: "str | Path",
    bandwidth_m: "float | None" = None,
    mask_burnable: bool = True,
    clip_buffer_bandwidths: float = 3.0,
    normalize: bool = True,
    min_bandwidth_cells: float = 2.0,
) -> dict:
    """
    Kernel-density ignition-likelihood surface from historical fire points.

    Bins the points onto the LCP grid, smooths with an isotropic Gaussian of
    the requested bandwidth, masks to burnable FBFM40, and normalizes so the
    surface sums to 1 over the burnable domain.  The result serves both as the
    sampling frame for design fires (:func:`select_design_ignitions`) and as
    the weights *w_i* in ``Σ_i w_i · TF_ij``.

    Parameters
    ----------
    fod_gdf : geopandas.GeoDataFrame
        Historical ignition points — normally FPA-FOD, pre-filtered to
        large-fire-capable records (size class D–G) and to the years and
        causes of interest.  Any CRS.  Points outside the LCP are used for
        smoothing near the edge and then dropped, so pass a set that extends
        beyond the domain.
    lcp_fp : str or Path
        Landscape raster.  Defines the output grid, CRS, and burnable mask.
    bandwidth_m : float, optional
        Gaussian kernel standard deviation in metres.  When omitted, Scott's
        rule is applied to the supplied points and the value is printed —
        check it, because on a small domain it degenerates toward uniform.
    mask_burnable : bool
        Zero out non-burnable FBFM40 cells (default True).  Fires cannot start
        where there is no fuel, so this is also the correct null for
        :func:`check_ignition_clustering`.
    clip_buffer_bandwidths : float
        Points are retained out to this many bandwidths beyond the LCP extent
        so the surface is not biased low near the boundary.  Default 3.
    normalize : bool
        Scale the surface to sum to 1 over unmasked cells (default True).
        With ``False`` the units are points per cell.
    min_bandwidth_cells : float
        Floor on the bandwidth expressed in cells.  A bandwidth below ~1 cell
        degenerates to the raw point pattern.  Default 2.

    Returns
    -------
    dict
        ``"density"``      : 2-D float array on the LCP grid (0 where masked)
        ``"transform"``    : affine transform
        ``"crs"``          : rasterio CRS
        ``"shape"``        : (rows, cols)
        ``"bandwidth_m"``  : float, the bandwidth actually used
        ``"n_points"``     : int, points contributing to the surface
        ``"n_in_domain"``  : int, points falling inside the LCP extent
        ``"cell_area_m2"`` : float
        ``"normalized"``   : bool

    Raises
    ------
    ValueError
        If no points fall within the buffered LCP extent.
    """
    from scipy.ndimage import gaussian_filter, map_coordinates
    from shapely.geometry import box

    grid = _read_burnable_mask(lcp_fp)
    transform, crs, shape = grid["transform"], grid["crs"], grid["shape"]
    res_m = grid["res_m"]
    nrows, ncols = shape

    pts = fod_gdf.to_crs(crs)
    pts = pts[pts.geometry.notna() & ~pts.geometry.is_empty]

    # ── Bandwidth ─────────────────────────────────────────────────────────────
    if bandwidth_m is None:
        xy_all = np.column_stack([pts.geometry.x.values, pts.geometry.y.values])
        bandwidth_m = _scott_bandwidth_m(xy_all)
        print(f"  [ignition_density_surface] Scott's-rule bandwidth: "
              f"{bandwidth_m:,.0f} m over {len(xy_all)} points — verify this is "
              f"a sensible spatial scale for your domain.")
    bandwidth_m = max(float(bandwidth_m), min_bandwidth_cells * res_m)

    # ── Retain points out to N bandwidths beyond the grid ─────────────────────
    left = transform.c
    top = transform.f
    right = left + ncols * transform.a
    bottom = top + nrows * transform.e  # transform.e is negative
    pad = clip_buffer_bandwidths * bandwidth_m
    padded = box(min(left, right) - pad, min(top, bottom) - pad,
                 max(left, right) + pad, max(top, bottom) + pad)
    pts = pts[pts.within(padded)]
    if len(pts) == 0:
        raise ValueError(
            "No ignition points fall within the LCP extent buffered by "
            f"{pad:,.0f} m. Check that fod_gdf covers the study area and is "
            "not over-filtered."
        )

    # ── Bin, smooth, and resample back onto the LCP grid ─────────────────────
    # The kernel is smooth at scale `bandwidth_m`, so binning and smoothing on
    # a working grid of ~bandwidth/4 costs nothing in accuracy and keeps the
    # separable Gaussian at a few cells wide.  Doing it at LCP resolution would
    # mean a sigma of hundreds of cells over a heavily padded array.
    work_res = max(res_m, bandwidth_m / _SMOOTH_CELLS)
    scale = work_res / res_m                       # LCP cells per working cell
    sigma_cells = bandwidth_m / work_res

    pad_cells = int(np.ceil(pad / work_res))
    work_rows = int(np.ceil(nrows / scale)) + 2 * pad_cells
    work_cols = int(np.ceil(ncols / scale)) + 2 * pad_cells

    inv = ~transform
    cols_f, rows_f = inv * (pts.geometry.x.values, pts.geometry.y.values)
    n_in_domain = int(
        ((rows_f >= 0) & (rows_f < nrows) & (cols_f >= 0) & (cols_f < ncols)).sum()
    )

    rows_w = np.floor(rows_f / scale).astype(int) + pad_cells
    cols_w = np.floor(cols_f / scale).astype(int) + pad_cells

    counts = np.zeros((work_rows, work_cols), dtype=float)
    ok = (
        (rows_w >= 0) & (rows_w < work_rows)
        & (cols_w >= 0) & (cols_w < work_cols)
    )
    np.add.at(counts, (rows_w[ok], cols_w[ok]), 1.0)

    smoothed = gaussian_filter(counts, sigma=sigma_cells, mode="constant", cval=0.0)

    if scale == 1.0:
        density = smoothed[pad_cells:pad_cells + nrows,
                           pad_cells:pad_cells + ncols]
    else:
        # Bilinear sample of the working grid at every LCP cell centre
        rr = np.arange(nrows) / scale + pad_cells
        cc = np.arange(ncols) / scale + pad_cells
        coords = np.array(np.meshgrid(rr, cc, indexing="ij"))
        density = map_coordinates(smoothed, coords, order=1, mode="nearest")
        # Working-cell mass → LCP-cell mass, so `normalize=False` stays in
        # points per LCP cell.
        density = density / (scale ** 2)

    if mask_burnable:
        density = np.where(grid["mask"], density, 0.0)

    if normalize:
        total = density.sum()
        if total <= 0:
            raise ValueError(
                "Ignition density surface is empty after masking — no burnable "
                "cell received any kernel mass. Check the bandwidth and the "
                "overlap between fod_gdf and the LCP."
            )
        density = density / total

    print(f"  [ignition_density_surface] bandwidth={bandwidth_m:,.0f} m "
          f"({sigma_cells:.1f} cells), {len(pts)} points "
          f"({n_in_domain} inside the LCP), "
          f"{'burnable-masked, ' if mask_burnable else ''}"
          f"{'normalized' if normalize else 'counts per cell'}")

    return {
        "density":      density,
        "transform":    transform,
        "crs":          crs,
        "shape":        shape,
        "bandwidth_m":  float(bandwidth_m),
        "n_points":     int(len(pts)),
        "n_in_domain":  n_in_domain,
        "cell_area_m2": grid["cell_area_m2"],
        "normalized":   bool(normalize),
    }


def sample_density_at_points(density: dict, points_gdf) -> np.ndarray:
    """
    Look up the density surface value at each point (nearest cell).

    Parameters
    ----------
    density : dict
        Output of :func:`ignition_density_surface`.
    points_gdf : geopandas.GeoDataFrame
        Point geometries, any CRS.

    Returns
    -------
    numpy.ndarray
        Density value per point; ``0.0`` for points outside the grid.
    """
    pts = points_gdf.to_crs(density["crs"])
    inv = ~density["transform"]
    cols_f, rows_f = inv * (pts.geometry.x.values, pts.geometry.y.values)
    rows = np.floor(rows_f).astype(int)
    cols = np.floor(cols_f).astype(int)

    nrows, ncols = density["shape"]
    out = np.zeros(len(pts), dtype=float)
    ok = (rows >= 0) & (rows < nrows) & (cols >= 0) & (cols < ncols)
    out[ok] = density["density"][rows[ok], cols[ok]]
    return out


def write_density_raster(density: dict, out_path: "str | Path") -> Path:
    """
    Write an ignition density surface to a float32 GeoTIFF.

    Parameters
    ----------
    density : dict
        Output of :func:`ignition_density_surface`.
    out_path : str or Path
        Destination ``.tif``.  Parent directory is created if needed.

    Returns
    -------
    Path
        Absolute path to the written raster.
    """
    import rasterio

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nrows, ncols = density["shape"]
    profile = {
        "driver":    "GTiff",
        "dtype":     "float32",
        "count":     1,
        "height":    nrows,
        "width":     ncols,
        "transform": density["transform"],
        "crs":       density["crs"],
        "nodata":    None,
        "compress":  "deflate",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(density["density"].astype("float32"), 1)
        dst.update_tags(
            bandwidth_m=density["bandwidth_m"],
            n_points=density["n_points"],
            normalized=str(density["normalized"]),
        )
    print(f"  [write_density_raster] → {out_path.name}")
    return out_path.resolve()


# ── Ager's uniform-ignition assumption (P2.2) ──────────────────────────────────

def check_ignition_clustering(
    fod_gdf,
    lcp_fp: "str | Path | None" = None,
    domain_gdf=None,
    n_sim: int = 199,
    radii_m: "list[float] | None" = None,
    seed: "int | None" = None,
    mask_burnable: bool = True,
) -> dict:
    """
    Test historical ignitions against a burnable-restricted CSR null.

    Ager et al. (2014) found *"no evidence of spatial correlation in the
    ignition locations of large fires"* and consequently used uniform random
    ignitions.  This function checks whether that holds for the study area.
    If it does, uniform sampling is defensible; if it does not, the
    density-weighted design of :func:`select_design_ignitions` is a genuine
    methodological improvement worth reporting.

    The null is complete spatial randomness **restricted to burnable fuel**,
    not to the bounding box.  Testing against an unrestricted null would
    report clustering merely because rock, water, and agriculture exist.

    Two statistics are computed:

    - mean nearest-neighbour distance (a Clark–Evans-style summary; smaller
      than the null envelope ⇒ clustered, larger ⇒ regular);
    - Ripley's *L(r) − r* at each radius in *radii_m*, which localizes the
      scale at which any departure occurs.

    Parameters
    ----------
    fod_gdf : geopandas.GeoDataFrame
        Historical ignition points, pre-filtered.  Any CRS.
    lcp_fp : str or Path, optional
        Landscape raster supplying the burnable study region and CRS.
        Required unless *domain_gdf* is given.
    domain_gdf : geopandas.GeoDataFrame, optional
        Study-region polygon.  Used together with *lcp_fp* to restrict the
        region, or alone (with ``mask_burnable=False``) when no LCP exists.
    n_sim : int
        Monte-Carlo realizations of the null.  Default 199, giving a minimum
        two-sided p-value of 0.01.
    radii_m : list of float, optional
        Radii for Ripley's L.  Default ``[1000, 2500, 5000, 10000, 20000]``.
    seed : int, optional
        Random seed.
    mask_burnable : bool
        Restrict the null to burnable FBFM40 (default True).  Requires
        *lcp_fp*.

    Returns
    -------
    dict
        ``"n_points"``, ``"mean_nn_m"``, ``"null_mean_nn_m"``,
        ``"null_nn_lo"`` / ``"null_nn_hi"`` (2.5/97.5 percentiles),
        ``"nn_p_value"`` (two-sided), ``"verdict"`` — one of
        ``"clustered"`` / ``"regular"`` / ``"consistent with CSR"`` —
        ``"radii_m"``, ``"L_minus_r"``, ``"L_null_lo"``, ``"L_null_hi"``,
        ``"n_sim"``, ``"region_area_m2"``.

    Raises
    ------
    ValueError
        If fewer than 3 points fall inside the region, or if neither
        *lcp_fp* nor *domain_gdf* is supplied.
    """
    from scipy.spatial import cKDTree

    if lcp_fp is None and domain_gdf is None:
        raise ValueError("check_ignition_clustering needs lcp_fp or domain_gdf.")
    if mask_burnable and lcp_fp is None:
        raise ValueError("mask_burnable=True requires lcp_fp.")

    if radii_m is None:
        radii_m = [1000.0, 2500.0, 5000.0, 10000.0, 20000.0]
    radii = np.asarray(radii_m, dtype=float)
    rng = np.random.default_rng(seed)

    # ── Candidate region as a set of admissible cell centres ─────────────────
    if lcp_fp is not None:
        grid = _read_burnable_mask(lcp_fp)
        crs = grid["crs"]
        region_mask = grid["mask"] if mask_burnable else np.ones(grid["shape"], bool)
        cell_area = grid["cell_area_m2"]

        if domain_gdf is not None:
            from rasterio.features import geometry_mask
            dom = domain_gdf.to_crs(crs)
            inside = ~geometry_mask(
                [g for g in dom.geometry],
                out_shape=grid["shape"],
                transform=grid["transform"],
                invert=False,
            )
            region_mask = region_mask & inside

        region_pts, _, _ = _points_from_mask(region_mask, grid["transform"], crs)
        region_xy = np.column_stack(
            [region_pts.geometry.x.values, region_pts.geometry.y.values]
        )
        region_area = float(region_mask.sum() * cell_area)
        from shapely.geometry import box as _box
        bounds = region_pts.total_bounds
        region_geom = _box(*bounds)
    else:
        dom = domain_gdf.to_crs(domain_gdf.estimate_utm_crs())
        crs = dom.crs
        region_geom = dom.geometry.union_all()
        region_area = float(region_geom.area)
        region_xy = None

    pts = fod_gdf.to_crs(crs)
    pts = pts[pts.geometry.notna() & ~pts.geometry.is_empty]
    if region_xy is None:
        pts = pts[pts.within(region_geom)]
    else:
        pts = pts[pts.within(region_geom)]
    n = len(pts)
    if n < 3:
        raise ValueError(
            f"Only {n} ignition point(s) inside the region — need at least 3."
        )
    obs_xy = np.column_stack([pts.geometry.x.values, pts.geometry.y.values])

    def _mean_nn(xy):
        tree = cKDTree(xy)
        d, _ = tree.query(xy, k=2)
        return float(d[:, 1].mean())

    def _ripley_l(xy, area):
        tree = cKDTree(xy)
        m = len(xy)
        lam = m / area
        out = np.empty(len(radii))
        for i, r in enumerate(radii):
            # counts within r, excluding self
            k = np.array(tree.query_ball_point(xy, r, return_length=True)) - 1
            k_hat = k.mean() / lam
            out[i] = np.sqrt(k_hat / np.pi) - r
        return out

    obs_nn = _mean_nn(obs_xy)
    obs_l = _ripley_l(obs_xy, region_area)

    def _draw_null():
        if region_xy is not None:
            idx = rng.choice(len(region_xy), size=n, replace=False)
            return region_xy[idx]
        minx, miny, maxx, maxy = region_geom.bounds
        from shapely.geometry import Point
        got = []
        while len(got) < n:
            xs = rng.uniform(minx, maxx, n * 3)
            ys = rng.uniform(miny, maxy, n * 3)
            for x, y in zip(xs, ys):
                if region_geom.contains(Point(x, y)):
                    got.append((x, y))
                    if len(got) == n:
                        break
        return np.asarray(got)

    null_nn = np.empty(n_sim)
    null_l = np.empty((n_sim, len(radii)))
    for s in range(n_sim):
        sim_xy = _draw_null()
        null_nn[s] = _mean_nn(sim_xy)
        null_l[s] = _ripley_l(sim_xy, region_area)

    # Two-sided Monte-Carlo p-value (Besag–Diggle rank form)
    rank_lo = (null_nn <= obs_nn).sum()
    rank_hi = (null_nn >= obs_nn).sum()
    p_value = 2.0 * (min(rank_lo, rank_hi) + 1) / (n_sim + 1)
    p_value = float(min(p_value, 1.0))

    nn_lo, nn_hi = np.percentile(null_nn, [2.5, 97.5])
    if obs_nn < nn_lo:
        verdict = "clustered"
    elif obs_nn > nn_hi:
        verdict = "regular"
    else:
        verdict = "consistent with CSR"

    print(f"  [check_ignition_clustering] n={n}, mean NN {obs_nn:,.0f} m vs null "
          f"{null_nn.mean():,.0f} m [{nn_lo:,.0f}, {nn_hi:,.0f}] "
          f"→ {verdict} (p={p_value:.3f}, {n_sim} sims)")

    return {
        "n_points":       n,
        "mean_nn_m":      obs_nn,
        "null_mean_nn_m": float(null_nn.mean()),
        "null_nn_lo":     float(nn_lo),
        "null_nn_hi":     float(nn_hi),
        "nn_p_value":     p_value,
        "verdict":        verdict,
        "radii_m":        radii.tolist(),
        "L_minus_r":      obs_l.tolist(),
        "L_null_lo":      np.percentile(null_l, 2.5, axis=0).tolist(),
        "L_null_hi":      np.percentile(null_l, 97.5, axis=0).tolist(),
        "n_sim":          int(n_sim),
        "region_area_m2": region_area,
    }


# ── Downwind spread cone (P2.3) ────────────────────────────────────────────────

def wind_cone_half_angle(
    pyrome_id: "str | int",
    cache_dir: "str | Path",
    coverage: float = DEFAULT_CONE_COVERAGE,
    prefix: str = "pyrome",
) -> dict:
    """
    Half-angle of the downwind spread cone, from the pyrome wind climatology.

    Finds the narrowest contiguous arc of the cached ``WindCellValues``
    directional marginal that contains *coverage* of the non-calm fire-hour
    frequency, interpolating within the boundary bin so the result is
    continuous rather than a multiple of the bin width.  This ties the cone to
    the site's own wind variability instead of an arbitrary angle.

    Parameters
    ----------
    pyrome_id : str or int
        Group identifier matching the cached JSON filename.
    cache_dir : str or Path
        Directory of ``{prefix}_{id}_wind.json`` files.
    coverage : float
        Fraction of non-calm directional frequency the arc must span, in
        (0, 1).  Default 0.5.
    prefix : str
        Cache filename prefix.  Default ``"pyrome"``.

    Returns
    -------
    dict
        ``"half_angle_deg"`` : float — half the arc width
        ``"arc_deg"``        : float — full arc width
        ``"center_az"``      : float — frequency-weighted arc centre, degrees FROM
        ``"coverage"``       : float — the coverage actually achieved
        ``"dominant_az"``    : float — centre of the single modal bin

    Raises
    ------
    ValueError
        If *coverage* is not in (0, 1).
    """
    from ..weather.hrrr import load_pyrome_wind_cells

    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage must be in (0, 1), got {coverage}")

    meta = load_pyrome_wind_cells(pyrome_id, cache_dir, return_meta=True,
                                  prefix=prefix)
    cells = np.asarray(meta["WindCellValues"], dtype=float)
    breaks = np.asarray(meta["WindDirBreaks_deg"], dtype=float)

    freq = cells.sum(axis=0)
    total = freq.sum()
    if total <= 0:
        raise ValueError(f"Wind rose for '{pyrome_id}' has zero total frequency.")
    freq = freq / total

    nbins = len(freq)
    lowers = np.concatenate([[0.0], breaks[:-1]])
    widths = breaks - lowers
    centers = lowers + widths / 2.0

    target = coverage
    best = None  # (arc_deg, start_bin, n_bins, partial_fraction)
    for start in range(nbins):
        acc = 0.0
        arc = 0.0
        for k in range(nbins):
            b = (start + k) % nbins
            if acc + freq[b] >= target:
                # Take only the fraction of this bin needed to reach coverage
                need = (target - acc) / freq[b] if freq[b] > 0 else 1.0
                arc += widths[b] * need
                acc = target
                cand = (arc, start, k + 1, need)
                if best is None or cand[0] < best[0]:
                    best = cand
                break
            acc += freq[b]
            arc += widths[b]
        else:
            continue

    if best is None:  # pragma: no cover — coverage < 1 always terminates
        raise ValueError(
            f"Could not span coverage={coverage} in the wind rose for '{pyrome_id}'."
        )

    arc_deg, start, n_used, _ = best

    # Frequency-weighted circular mean over the bins the arc spans
    used = [(start + k) % nbins for k in range(n_used)]
    w = freq[used]
    ang = np.radians(centers[used])
    center_az = float((np.degrees(np.arctan2((w * np.sin(ang)).sum(),
                                             (w * np.cos(ang)).sum())) + 360.0) % 360.0)

    dom_idx = int(np.argmax(freq))
    return {
        "half_angle_deg": float(arc_deg / 2.0),
        "arc_deg":        float(arc_deg),
        "center_az":      round(center_az, 1),
        "coverage":       float(coverage),
        "dominant_az":    float(centers[dom_idx]),
    }


def downwind_cone(origin, downwind_az: float, half_angle_deg: float,
                  length_m: float, n_arc: int = 48):
    """
    Triangular/sector spread cone from an ignition point.

    Replaces the zero-width ``LineString`` ray test, which asked only whether
    one exact bearing clipped the treatment polygon and ignored fire width
    entirely.  A real fire spreads angularly with distance, so a candidate
    ignition whose axis misses the treatment by 200 m still burns through it.

    Parameters
    ----------
    origin : shapely.geometry.Point
        Cone apex — the ignition location, in projected metres.
    downwind_az : float
        Direction fire travels, degrees from north (``wind_from + 180``).
    half_angle_deg : float
        Half the cone's angular width.
    length_m : float
        Cone length.
    n_arc : int
        Points sampled along the far arc.  Default 48.

    Returns
    -------
    shapely.geometry.Polygon

    Raises
    ------
    ValueError
        If *half_angle_deg* is not in (0, 180) or *length_m* is not positive.
    """
    from shapely.geometry import Polygon

    if not 0.0 < half_angle_deg < 180.0:
        raise ValueError(
            f"half_angle_deg must be in (0, 180), got {half_angle_deg}"
        )
    if length_m <= 0:
        raise ValueError(f"length_m must be positive, got {length_m}")

    azimuths = np.linspace(downwind_az - half_angle_deg,
                           downwind_az + half_angle_deg, n_arc)
    math_ang = np.radians(90.0 - azimuths)
    arc = [(origin.x + length_m * np.cos(a), origin.y + length_m * np.sin(a))
           for a in math_ang]
    return Polygon([(origin.x, origin.y)] + arc)


# ── Stratified design-fire selection (P2.3) ────────────────────────────────────

def _allocate_largest_remainder(mass: np.ndarray, total: int) -> np.ndarray:
    """
    Allocate *total* draws across strata proportional to *mass*.

    Every stratum carrying mass receives at least one draw so the design spans
    the transmission geometry rather than piling into the densest stratum.
    """
    nonzero = mass > 0
    k = int(nonzero.sum())
    if k == 0:
        raise ValueError("No stratum carries any ignition-density mass.")
    if total < k:
        # Not enough ignitions for one per stratum — keep the heaviest strata.
        order = np.argsort(-mass)
        out = np.zeros(len(mass), dtype=int)
        out[order[:total]] = 1
        return out

    share = mass / mass.sum() * total
    out = np.where(nonzero, np.maximum(np.floor(share), 1.0), 0.0).astype(int)
    # Largest-remainder pass to hit the exact total
    while out.sum() > total:
        cand = np.where(out > 1, out - share, -np.inf)
        out[int(np.argmax(cand))] -= 1
    while out.sum() < total:
        cand = np.where(nonzero, share - out, -np.inf)
        out[int(np.argmax(cand))] += 1
    return out


def select_design_ignitions(
    treatments_gdf,
    values_gdf,
    lcp_fp: "str | Path",
    out_dir: "str | Path",
    wind_from_deg: float,
    density: "dict | None" = None,
    n_ignitions: int = 10,
    dist_band_km: tuple = (5.0, 25.0),
    cone_half_angle_deg: float = 30.0,
    upwind_sector_deg: "float | None" = None,
    n_bearing_strata: int = 3,
    n_distance_strata: int = 2,
    footprint_acres: "float | None" = DEFAULT_FOOTPRINT_ACRES,
    require_ordering: bool = True,
    cone_length_m: "float | None" = None,
    max_candidates: int = 20000,
    seed: "int | None" = None,
    prefix: str = "ign",
) -> dict:
    """
    Select stratified, density-weighted design fires upwind of a treatment.

    Implements the Part-A sampling design: rather than a dense ensemble, take
    8–12 source locations that **cover** the transmission geometry and carry
    ignition-likelihood weights, so that

        Expected transmission to j = Σ_i w_i · TF_ij.

    Candidates are burnable pixel centres inside an upwind annular sector
    around the treatment centroid.  They are filtered on transmission geometry
    — the candidate's downwind spread **cone** must intersect the treatment,
    and (optionally) the treatment must lie nearer than the values — then
    stratified by approach bearing and distance, with draws allocated
    proportional to each stratum's ignition-density mass and taken within a
    stratum proportional to the per-cell density.

    The resulting weights are ``w_i = (stratum density mass) / (draws in that
    stratum)``, normalized to sum to 1 — the Horvitz–Thompson form, so
    ``Σ_i w_i · TF_ij`` is unbiased for the density-weighted expectation.

    Parameters
    ----------
    treatments_gdf : geopandas.GeoDataFrame
        Treatment polygons under test.  Any CRS.
    values_gdf : geopandas.GeoDataFrame or None
        Values to protect (WUI, structures, POD).  Used to orient the
        ignition → treatment → values geometry and for *require_ordering*.
        Pass ``None`` to skip both checks.
    lcp_fp : str or Path
        Landscape raster — CRS, cell size, and burnable fuels.
    out_dir : str or Path
        Directory for the per-ignition shapefiles.
    wind_from_deg : float
        Dominant wind direction, degrees FROM (meteorological).  See
        :func:`~fb_tools.weather.hrrr.dominant_wind_direction`.
    density : dict, optional
        Output of :func:`ignition_density_surface`.  When omitted, every
        candidate is weighted equally — Ager's uniform-ignition assumption,
        which :func:`check_ignition_clustering` can check for the study area.
    n_ignitions : int
        Number of design fires.  Default 10.
    dist_band_km : tuple of float
        ``(min, max)`` upwind distance from the treatment centroid, km.
        Default ``(5, 25)`` — wide enough that fire reaches *and passes* the
        treatment within a 21-day ``Duration``.  Calibrate against P0.2.
    cone_half_angle_deg : float
        Half-width of the downwind spread cone.  Default 30.  Derive it from
        the site's wind climatology with :func:`wind_cone_half_angle`.
    upwind_sector_deg : float, optional
        Angular width of the candidate placement wedge.  Defaults to
        ``2 * cone_half_angle_deg + 30`` so candidates exist slightly outside
        the strict cone geometry and the cone filter does real work.
    n_bearing_strata, n_distance_strata : int
        Stratification grid over approach bearing and distance.  Defaults
        3 × 2 = 6 strata.
    footprint_acres : float or None
        Ignition footprint area.  Default 10 ac; ``None`` gives a half-pixel
        circle.  See :func:`footprint_radius_m`.
    require_ordering : bool
        Require the treatment to be nearer the ignition than the values are,
        so a transmission pathway through the treatment actually exists.
        Ignored when *values_gdf* is None.
    cone_length_m : float, optional
        Cone length.  Defaults to the outer distance band plus the diagonal
        of the treatments-plus-values envelope.
    max_candidates : int
        Cap on candidate pixels carried into the per-candidate cone test,
        randomly subsampled above this.  Default 20,000.
    seed : int, optional
        Random seed for subsampling and within-stratum draws.
    prefix : str
        Shapefile stem prefix.  Default ``"ign"``.

    Returns
    -------
    dict
        ``"ignition_shapefiles"``  : list of Path, one single-feature file each
        ``"ignitions_gdf"``        : GeoDataFrame — footprint polygons with
        ``ign_id``, ``w_i``, ``density``, ``dist_m``, ``az_from_trt``,
        ``bear_stratum``, ``dist_stratum``, ``stratum``, ``footprint_ac``,
        ``shp_path``
        ``"ignition_points_gdf"``  : GeoDataFrame of the point locations
        ``"wind_from_deg"``, ``"downwind_az"``, ``"cone_half_angle_deg"``
        ``"n_candidates"``, ``"n_after_cone"``, ``"n_after_ordering"``
        ``"uniform_weights"``      : bool — True when *density* was omitted

    Raises
    ------
    ValueError
        If no burnable candidate survives the placement wedge, the cone
        filter, or the ordering filter.
    """
    import geopandas as gpd
    import rasterio
    from rasterio.features import geometry_mask
    from shapely.geometry import Point

    from .lcp import _annular_sector, _bearing_deg

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    grid = _read_burnable_mask(lcp_fp)
    crs, transform = grid["crs"], grid["transform"]

    trt = treatments_gdf.to_crs(crs)
    trt_union = trt.geometry.union_all()
    trt_centroid = trt_union.centroid

    val_union = None
    if values_gdf is not None:
        val = values_gdf.to_crs(crs)
        val_union = val.geometry.union_all()

    downwind_az = (float(wind_from_deg) + 180.0) % 360.0

    if val_union is not None:
        trt_to_val_az = _bearing_deg(trt_centroid, val_union.centroid)
        off = abs((trt_to_val_az - downwind_az + 180.0) % 360.0 - 180.0)
        if off > 90.0:
            print(f"  [select_design_ignitions] Warning: values are not downwind "
                  f"of treatments (treatment→values {trt_to_val_az:.0f}°, downwind "
                  f"{downwind_az:.0f}°, off by {off:.0f}°). The "
                  f"ignition→treatment→values geometry is inconsistent with the "
                  f"prevailing wind.")

    # ── Candidate pixels inside the upwind placement wedge ───────────────────
    if upwind_sector_deg is None:
        upwind_sector_deg = 2.0 * cone_half_angle_deg + 30.0
    r_in, r_out = dist_band_km[0] * 1000.0, dist_band_km[1] * 1000.0
    wedge = _annular_sector(
        trt_centroid.x, trt_centroid.y,
        center_az=float(wind_from_deg),
        sector_deg=float(upwind_sector_deg),
        r_in=r_in, r_out=r_out,
    )
    in_wedge = ~geometry_mask([wedge], out_shape=grid["shape"],
                              transform=transform, invert=False)
    cand, rows, cols = _points_from_mask(grid["mask"], transform, crs,
                                         sub_mask=in_wedge)
    n_candidates = len(cand)
    if n_candidates == 0:
        raise ValueError(
            "No burnable pixels in the upwind placement wedge. Check "
            "dist_band_km, upwind_sector_deg, and that the LCP covers the zone."
        )

    # Per-candidate ignition-likelihood weight
    if density is None:
        w = np.ones(n_candidates, dtype=float)
        uniform_weights = True
    else:
        w = density["density"][rows, cols].astype(float)
        uniform_weights = False

    # ── Subsample before the per-candidate cone test ─────────────────────────
    if n_candidates > max_candidates:
        keep = rng.choice(n_candidates, size=max_candidates, replace=False)
        keep.sort()
        cand = cand.iloc[keep].reset_index(drop=True)
        w = w[keep]
        print(f"  [select_design_ignitions] {n_candidates:,} candidates "
              f"subsampled to {max_candidates:,} for the cone test.")

    # ── Downwind cone filter (replaces the zero-width ray test) ──────────────
    if cone_length_m is None:
        env = trt_union if val_union is None else trt_union.union(val_union)
        minx, miny, maxx, maxy = env.bounds
        diag = float(np.hypot(maxx - minx, maxy - miny))
        cone_length_m = r_out + diag

    cones = gpd.GeoSeries(
        [downwind_cone(pt, downwind_az, cone_half_angle_deg, cone_length_m,
                       n_arc=24) for pt in cand.geometry],
        crs=crs,
    )
    hits = cones.intersects(trt_union).values
    cand = cand[hits].reset_index(drop=True)
    w = w[hits]
    n_after_cone = len(cand)
    if n_after_cone == 0:
        raise ValueError(
            "No candidate's downwind cone reaches the treatment. Widen "
            "cone_half_angle_deg or upwind_sector_deg, or shorten dist_band_km."
        )

    # ── Ordering: treatment must lie between ignition and values ─────────────
    n_after_ordering = n_after_cone
    if require_ordering and val_union is not None:
        d_trt = cand.geometry.distance(trt_union).values
        d_val = cand.geometry.distance(val_union).values
        keep = d_trt < d_val
        cand = cand[keep].reset_index(drop=True)
        w = w[keep]
        n_after_ordering = len(cand)
        if n_after_ordering == 0:
            raise ValueError(
                "No candidate has the treatment nearer than the values — there "
                "is no transmission pathway through the treatment to measure. "
                "Set require_ordering=False or revisit the design geometry."
            )

    # ── Stratify by approach bearing × distance ──────────────────────────────
    az = np.array([_bearing_deg(trt_centroid, pt) for pt in cand.geometry])
    dist = cand.geometry.distance(trt_centroid).values

    # Bearing relative to the wedge centre, unwrapped to [-180, 180]
    rel_az = (az - float(wind_from_deg) + 180.0) % 360.0 - 180.0
    half = upwind_sector_deg / 2.0
    bear_edges = np.linspace(-half, half, n_bearing_strata + 1)
    bear_idx = np.clip(np.digitize(rel_az, bear_edges[1:-1]), 0,
                       n_bearing_strata - 1)

    dist_edges = np.linspace(r_in, r_out, n_distance_strata + 1)
    dist_idx = np.clip(np.digitize(dist, dist_edges[1:-1]), 0,
                       n_distance_strata - 1)

    stratum = bear_idx * n_distance_strata + dist_idx
    n_strata = n_bearing_strata * n_distance_strata
    mass = np.array([w[stratum == s].sum() for s in range(n_strata)])
    if mass.sum() <= 0:
        # Density is zero everywhere admissible — fall back to uniform so the
        # geometry design still yields a set, and say so.
        print("  [select_design_ignitions] Warning: the density surface is zero "
              "at every admissible candidate; falling back to uniform weights.")
        w = np.ones_like(w)
        mass = np.array([w[stratum == s].sum() for s in range(n_strata)])
        uniform_weights = True

    alloc = _allocate_largest_remainder(mass, int(n_ignitions))

    # ── Draw within strata, proportional to density ──────────────────────────
    picks = []
    for s in range(n_strata):
        k = int(alloc[s])
        if k == 0:
            continue
        idx = np.flatnonzero(stratum == s)
        p = w[idx]
        p = p / p.sum() if p.sum() > 0 else None
        k = min(k, len(idx))
        chosen = rng.choice(idx, size=k, replace=False, p=p)
        stratum_w = mass[s] / k  # Horvitz–Thompson weight for this stratum
        for c in chosen:
            picks.append((int(c), s, float(stratum_w)))

    picks.sort(key=lambda t: (t[1], t[0]))
    sel_idx = np.array([p[0] for p in picks])
    sel_stratum = np.array([p[1] for p in picks])
    sel_w = np.array([p[2] for p in picks], dtype=float)
    sel_w = sel_w / sel_w.sum()

    points = cand.iloc[sel_idx].reset_index(drop=True).copy()
    points["ign_id"]       = np.arange(len(points))
    points["w_i"]          = np.round(sel_w, 6)
    points["density"]      = w[sel_idx]
    points["dist_m"]       = np.round(dist[sel_idx], 1)
    points["az_from_trt"]  = np.round(az[sel_idx], 1)
    points["bear_stratum"] = bear_idx[sel_idx]
    points["dist_stratum"] = dist_idx[sel_idx]
    points["stratum"]      = sel_stratum

    # ── Footprints and one shapefile per ignition ────────────────────────────
    foot = build_ignition_footprints(points, acres=footprint_acres,
                                     lcp_res_m=grid["res_m"])
    shp_paths = write_ignition_shapefiles(foot, out_dir, prefix=prefix,
                                          id_col="ign_id")
    foot["shp_path"] = [str(p) for p in shp_paths]

    print(f"  [select_design_ignitions] {len(foot)} design fire(s) from "
          f"{n_candidates:,} candidates → {n_after_cone:,} in cone → "
          f"{n_after_ordering:,} ordered; wind FROM {wind_from_deg:.0f}°, "
          f"cone ±{cone_half_angle_deg:.0f}°, band "
          f"{dist_band_km[0]}–{dist_band_km[1]} km, "
          f"{'uniform' if uniform_weights else 'density'} weights")

    return {
        "ignition_shapefiles":  shp_paths,
        "ignitions_gdf":        foot,
        "ignition_points_gdf":  points,
        "wind_from_deg":        float(wind_from_deg),
        "downwind_az":          downwind_az,
        "cone_half_angle_deg":  float(cone_half_angle_deg),
        "n_candidates":         int(n_candidates),
        "n_after_cone":         int(n_after_cone),
        "n_after_ordering":     int(n_after_ordering),
        "uniform_weights":      bool(uniform_weights),
    }
