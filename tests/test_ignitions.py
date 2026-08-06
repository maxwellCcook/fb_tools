"""
Phase 2 — ignition footprints, density surfaces, and design-fire selection.

Covers `fb_tools.fuelscape.ignitions` plus the P2.5 fix to the two ignition
writers in `lcp.py`.  Everything here runs against the synthetic landscape
fixture, so the suite needs nothing from `data/`.
"""

import math

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, box

from fb_tools.fuelscape.ignitions import (
    DEFAULT_FOOTPRINT_ACRES,
    _ACRE_M2,
    _read_burnable_mask,
    build_ignition_footprints,
    downwind_cone,
    footprint_radius_m,
    ignition_density_surface,
    sample_density_at_points,
    select_design_ignitions,
    check_ignition_clustering,
    wind_cone_half_angle,
    write_density_raster,
    write_ignition_shapefiles,
)


# ── P2.4 — ignition footprint sizing ───────────────────────────────────────────

@pytest.mark.parametrize("acres", [1.0, 10.0, 100.0, 630.0])
def test_footprint_radius_round_trips_to_requested_area(acres):
    """A circle of the returned radius has exactly the requested area."""
    r = footprint_radius_m(acres)
    assert math.isclose(math.pi * r * r / _ACRE_M2, acres, rel_tol=1e-9)


def test_footprint_default_is_ten_acres():
    """The shipped default is the 10-acre day-1 perimeter (P2.4)."""
    assert DEFAULT_FOOTPRINT_ACRES == 10.0
    assert math.isclose(footprint_radius_m(), footprint_radius_m(10.0))


def test_footprint_none_gives_half_pixel():
    assert footprint_radius_m(None, lcp_res_m=90.0) == 45.0


def test_footprint_none_without_resolution_raises():
    with pytest.raises(ValueError, match="lcp_res_m"):
        footprint_radius_m(None)


def test_footprint_floored_at_half_pixel():
    """A footprint smaller than one cell is widened, never silently lost."""
    r = footprint_radius_m(0.001, lcp_res_m=90.0)
    assert r == 45.0


def test_footprint_rejects_nonpositive():
    with pytest.raises(ValueError, match="positive"):
        footprint_radius_m(0.0)
    with pytest.raises(ValueError, match="positive"):
        footprint_radius_m(-5.0)


def test_build_footprints_areas_and_columns(synth_lcp):
    pts = gpd.GeoDataFrame(
        {"ign_id": [0, 1]},
        geometry=[Point(-799_000, 1_955_000), Point(-798_000, 1_954_000)],
        crs="EPSG:5070",
    )
    foot = build_ignition_footprints(pts, acres=10.0, lcp_fp=synth_lcp)
    assert list(foot.ign_id) == [0, 1]
    assert np.allclose(foot.geometry.area / _ACRE_M2, 10.0, rtol=1e-3)
    # footprint_ac is measured off the written polygon, not the nominal circle,
    # so it is honest about the polygonal approximation (~0.01% short).
    assert np.allclose(foot.footprint_ac, foot.geometry.area / _ACRE_M2,
                       atol=5e-4)
    # Input is not mutated (copy-not-mutate)
    assert pts.geometry.iloc[0].geom_type == "Point"


def test_build_footprints_rejects_geographic_crs():
    pts = gpd.GeoDataFrame(geometry=[Point(-105.0, 39.0)], crs="EPSG:4326")
    with pytest.raises(ValueError, match="projected CRS"):
        build_ignition_footprints(pts, acres=10.0, lcp_res_m=30.0)


# ── P2.5 — one shapefile per ignition ──────────────────────────────────────────

def test_write_ignition_shapefiles_one_feature_each(tmp_path):
    pts = gpd.GeoDataFrame(
        {"ign_id": [0, 1, 2], "w_i": [0.5, 0.3, 0.2]},
        geometry=[Point(0, 0), Point(1000, 0), Point(2000, 0)],
        crs="EPSG:5070",
    )
    foot = build_ignition_footprints(pts, acres=10.0)
    paths = write_ignition_shapefiles(foot, tmp_path, prefix="ign",
                                      id_col="ign_id")

    assert len(paths) == 3
    assert [p.name for p in paths] == ["ign_000.shp", "ign_001.shp", "ign_002.shp"]
    for p in paths:
        # FSPro treats one IgnitionFile as ONE fire's starting perimeter, so
        # each file must hold exactly one feature.
        assert len(gpd.read_file(p)) == 1


def test_write_ignition_shapefiles_drops_long_field_names(tmp_path):
    """Design metadata stays off the DBF, which truncates names to 10 chars."""
    pts = gpd.GeoDataFrame(
        {"ign_id": [0], "bear_stratum": [1], "footprint_ac": [10.0]},
        geometry=[Point(0, 0)], crs="EPSG:5070",
    )
    paths = write_ignition_shapefiles(pts, tmp_path, id_col="ign_id")
    cols = set(gpd.read_file(paths[0]).columns)
    assert cols == {"ign_id", "geometry"}


def test_create_random_ignitions_returns_one_file_per_ignition(synth_lcp, tmp_path):
    """P2.5: this used to write all N circles into a single shapefile."""
    from fb_tools.fuelscape.lcp import create_random_ignitions
    import rasterio

    with rasterio.open(synth_lcp) as src:
        container = gpd.GeoDataFrame(geometry=[box(*src.bounds)], crs=src.crs)

    paths = create_random_ignitions(container, 5, synth_lcp, tmp_path,
                                    seed=3, footprint_acres=10.0)
    assert isinstance(paths, list) and len(paths) == 5
    assert len({p.name for p in paths}) == 5
    for p in paths:
        assert len(gpd.read_file(p)) == 1


def test_create_fod_ignitions_returns_one_file_per_ignition(
    synth_lcp, synth_fod, tmp_path
):
    from fb_tools.fuelscape.lcp import create_fod_ignitions
    import rasterio

    with rasterio.open(synth_lcp) as src:
        container = gpd.GeoDataFrame(geometry=[box(*src.bounds)], crs=src.crs)

    paths = create_fod_ignitions(container, synth_fod, synth_lcp, tmp_path,
                                 footprint_acres=10.0)
    assert isinstance(paths, list) and len(paths) > 1
    for p in paths:
        assert len(gpd.read_file(p)) == 1


# ── P2.2 — ignition density surface ────────────────────────────────────────────

def test_density_surface_normalizes_to_one(synth_lcp, synth_fod):
    d = ignition_density_surface(synth_fod, synth_lcp, bandwidth_m=600.0)
    assert math.isclose(d["density"].sum(), 1.0, rel_tol=1e-6)
    assert d["shape"] == (200, 200)
    assert d["bandwidth_m"] == 600.0


def test_density_surface_masks_nonburnable(synth_lcp, synth_fod):
    """Fires cannot start where there is no fuel."""
    d = ignition_density_surface(synth_fod, synth_lcp, bandwidth_m=600.0)
    grid = _read_burnable_mask(synth_lcp)
    assert (d["density"][~grid["mask"]] == 0).all()
    assert d["density"][grid["mask"]].sum() > 0


def test_density_surface_unmasked_covers_nonburnable(synth_lcp, synth_fod):
    d = ignition_density_surface(synth_fod, synth_lcp, bandwidth_m=600.0,
                                 mask_burnable=False)
    grid = _read_burnable_mask(synth_lcp)
    assert d["density"][~grid["mask"]].sum() > 0


def test_density_surface_bandwidth_floored_to_cells(synth_lcp, synth_fod):
    """A sub-cell bandwidth would degenerate to the raw point pattern."""
    d = ignition_density_surface(synth_fod, synth_lcp, bandwidth_m=1.0)
    assert d["bandwidth_m"] >= 2 * 30.0


def test_density_surface_tracks_the_clusters(synth_lcp, synth_fod):
    """Density is higher near the seeded clusters than far from them."""
    d = ignition_density_surface(synth_fod, synth_lcp, bandwidth_m=600.0)
    near = sample_density_at_points(d, synth_fod.head(20))
    far = gpd.GeoDataFrame(
        geometry=[Point(-795_500, 1_954_500)], crs=synth_fod.crs
    )
    assert near.mean() > sample_density_at_points(d, far).max()


def test_density_surface_large_bandwidth_is_fast_and_finite(synth_lcp, synth_fod):
    """A bandwidth of many hundreds of cells must not blow up the smoother."""
    d = ignition_density_surface(synth_fod, synth_lcp, bandwidth_m=50_000.0)
    assert np.isfinite(d["density"]).all()
    assert math.isclose(d["density"].sum(), 1.0, rel_tol=1e-6)


def test_density_surface_raises_when_points_are_elsewhere(synth_lcp):
    far = gpd.GeoDataFrame(
        geometry=[Point(0, 0), Point(1000, 1000)], crs="EPSG:5070"
    )
    with pytest.raises(ValueError, match="No ignition points"):
        ignition_density_surface(far, synth_lcp, bandwidth_m=600.0)


def test_sample_density_off_grid_is_zero(synth_lcp, synth_fod):
    d = ignition_density_surface(synth_fod, synth_lcp, bandwidth_m=600.0)
    off = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:5070")
    assert sample_density_at_points(d, off)[0] == 0.0


def test_write_density_raster_round_trips(synth_lcp, synth_fod, tmp_path):
    import rasterio

    d = ignition_density_surface(synth_fod, synth_lcp, bandwidth_m=600.0)
    p = write_density_raster(d, tmp_path / "density.tif")
    with rasterio.open(p) as src:
        assert src.crs == d["crs"]
        assert (src.height, src.width) == d["shape"]
        assert np.allclose(src.read(1), d["density"].astype("float32"))


# ── P2.2 — Ager's uniform-ignition assumption ──────────────────────────────────

def test_clustering_detects_clustered_points(synth_lcp, synth_fod):
    """The seeded fixture is three tight clusters — the test must say so."""
    r = check_ignition_clustering(synth_fod, lcp_fp=synth_lcp, n_sim=99, seed=1)
    assert r["verdict"] == "clustered"
    assert r["mean_nn_m"] < r["null_nn_lo"]
    assert r["nn_p_value"] <= 0.05


def test_clustering_accepts_random_points(synth_lcp):
    """Points drawn from the null itself must not be called clustered."""
    import rasterio

    grid = _read_burnable_mask(synth_lcp)
    with rasterio.open(synth_lcp) as src:
        crs = src.crs
    from fb_tools.fuelscape.ignitions import _points_from_mask

    cells, _, _ = _points_from_mask(grid["mask"], grid["transform"], crs)
    rng = np.random.default_rng(5)
    pts = cells.iloc[rng.choice(len(cells), 120, replace=False)].reset_index(drop=True)

    r = check_ignition_clustering(pts, lcp_fp=synth_lcp, n_sim=99, seed=2)
    assert r["verdict"] == "consistent with CSR"


def test_clustering_null_excludes_nonburnable(synth_lcp, synth_fod):
    """With mask_burnable the admissible region is smaller than the grid."""
    grid = _read_burnable_mask(synth_lcp)
    masked = check_ignition_clustering(synth_fod, lcp_fp=synth_lcp, n_sim=19,
                                      seed=3, mask_burnable=True)
    unmasked = check_ignition_clustering(synth_fod, lcp_fp=synth_lcp, n_sim=19,
                                        seed=3, mask_burnable=False)
    assert masked["region_area_m2"] < unmasked["region_area_m2"]
    assert math.isclose(
        masked["region_area_m2"], grid["mask"].sum() * grid["cell_area_m2"]
    )


def test_clustering_needs_a_region():
    pts = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:5070")
    with pytest.raises(ValueError, match="lcp_fp or domain_gdf"):
        check_ignition_clustering(pts)


def test_clustering_needs_enough_points(synth_lcp):
    pts = gpd.GeoDataFrame(
        geometry=[Point(-799_000, 1_955_000)], crs="EPSG:5070"
    )
    with pytest.raises(ValueError, match="at least 3"):
        check_ignition_clustering(pts, lcp_fp=synth_lcp, n_sim=9)


# ── P2.3 — spread cone geometry ────────────────────────────────────────────────

def test_cone_area_matches_the_sector_formula():
    half, length = 30.0, 10_000.0
    c = downwind_cone(Point(0, 0), 90.0, half, length, n_arc=256)
    expected = math.pi * length ** 2 * (2 * half / 360.0)
    assert math.isclose(c.area, expected, rel_tol=1e-3)


def test_cone_points_downwind():
    """A due-east cone reaches east and not west."""
    c = downwind_cone(Point(0, 0), 90.0, 30.0, 10_000.0)
    assert c.contains(Point(5000, 0))
    assert not c.contains(Point(-5000, 0))


def test_cone_catches_what_a_zero_width_ray_misses():
    """
    The defect P2.3 fixes: a zero-width ray tests one exact bearing and
    ignores fire width, so it rejects sources that burn straight through.
    """
    from shapely.geometry import LineString

    origin = Point(0, 0)
    # Treatment offset 1.5 km off the due-east axis, 5 km out
    trt = box(4_000, 1_000, 6_000, 2_000)

    ray = LineString([(0, 0), (50_000, 0)])
    cone = downwind_cone(origin, 90.0, 30.0, 20_000.0)

    assert not ray.intersects(trt)
    assert cone.intersects(trt)


@pytest.mark.parametrize("half", [0.0, 180.0, -10.0, 200.0])
def test_cone_rejects_invalid_half_angle(half):
    with pytest.raises(ValueError, match="half_angle_deg"):
        downwind_cone(Point(0, 0), 90.0, half, 1000.0)


def test_cone_rejects_nonpositive_length():
    with pytest.raises(ValueError, match="length_m"):
        downwind_cone(Point(0, 0), 90.0, 30.0, 0.0)


# ── P2.3 — wind-derived cone half-angle ────────────────────────────────────────

def _write_wind_cache(tmp_path, cells, dir_breaks, pyrome_id="test"):
    import json

    path = tmp_path / f"pyrome_{pyrome_id}_wind.json"
    path.write_text(json.dumps({
        "pyrome_id": pyrome_id,
        "NumWindSpeeds": len(cells),
        "NumWindDirs": len(dir_breaks),
        "WindSpeedBreaks_mph": [5, 10, 15, 20, 25, 30][:len(cells)],
        "WindDirBreaks_deg": dir_breaks,
        "WindCellValues": cells,
        "CalmValue": 1.0,
        "n_observations": 1000,
        "years_covered": [2016, 2024],
    }))
    return path


def test_cone_half_angle_on_a_single_direction(tmp_path):
    """All frequency in one 45 deg bin: half the coverage needs half the bin."""
    cells = [[0.0] * 8]
    cells[0][2] = 100.0            # bin 2 spans (90, 135]
    _write_wind_cache(tmp_path, cells, [45, 90, 135, 180, 225, 270, 315, 360])

    r = wind_cone_half_angle("test", tmp_path, coverage=0.5)
    assert math.isclose(r["arc_deg"], 22.5, rel_tol=1e-6)
    assert math.isclose(r["half_angle_deg"], 11.25, rel_tol=1e-6)
    assert math.isclose(r["dominant_az"], 112.5, rel_tol=1e-6)


def test_cone_half_angle_widens_with_coverage(tmp_path):
    cells = [[5.0, 10.0, 40.0, 25.0, 10.0, 5.0, 3.0, 2.0]]
    _write_wind_cache(tmp_path, cells, [45, 90, 135, 180, 225, 270, 315, 360])

    angles = [wind_cone_half_angle("test", tmp_path, coverage=c)["half_angle_deg"]
              for c in (0.25, 0.5, 0.75, 0.9)]
    assert angles == sorted(angles)
    assert all(a > 0 for a in angles)


def test_cone_half_angle_wraps_across_north(tmp_path):
    """Mass split across the 315-360 / 0-45 boundary must give a narrow arc."""
    cells = [[50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0]]
    _write_wind_cache(tmp_path, cells, [45, 90, 135, 180, 225, 270, 315, 360])

    r = wind_cone_half_angle("test", tmp_path, coverage=0.5)
    # Bin 7 (315-360] plus part of bin 0 — not a 315 deg trip the long way.
    assert r["arc_deg"] <= 90.0


@pytest.mark.parametrize("coverage", [0.0, 1.0, -0.5, 1.5])
def test_cone_half_angle_rejects_invalid_coverage(tmp_path, coverage):
    cells = [[12.5] * 8]
    _write_wind_cache(tmp_path, cells, [45, 90, 135, 180, 225, 270, 315, 360])
    with pytest.raises(ValueError, match="coverage"):
        wind_cone_half_angle("test", tmp_path, coverage=coverage)


# ── P2.3 — stratified design-fire selection ────────────────────────────────────

@pytest.fixture
def design_geometry(synth_lcp):
    """Treatment east of centre, values further east — a west wind carries."""
    import rasterio

    with rasterio.open(synth_lcp) as src:
        crs, b = src.crs, src.bounds
    cx = (b.left + b.right) / 2
    cy = (b.bottom + b.top) / 2
    trt = gpd.GeoDataFrame(geometry=[box(cx + 400, cy - 500, cx + 1200, cy + 500)],
                           crs=crs)
    val = gpd.GeoDataFrame(geometry=[Point(cx + 2400, cy).buffer(300)], crs=crs)
    return trt, val


def test_design_ignitions_weights_sum_to_one(synth_lcp, synth_fod, design_geometry,
                                             tmp_path):
    trt, val = design_geometry
    d = ignition_density_surface(synth_fod, synth_lcp, bandwidth_m=1000.0)
    res = select_design_ignitions(
        trt, val, synth_lcp, tmp_path, wind_from_deg=270.0, density=d,
        n_ignitions=6, dist_band_km=(0.5, 2.0), cone_half_angle_deg=30.0, seed=1,
    )
    g = res["ignitions_gdf"]
    assert len(g) == 6
    assert math.isclose(g.w_i.sum(), 1.0, abs_tol=1e-5)
    assert (g.w_i > 0).all()
    assert not res["uniform_weights"]


def test_design_ignitions_one_file_per_fire(synth_lcp, design_geometry, tmp_path):
    trt, val = design_geometry
    res = select_design_ignitions(
        trt, val, synth_lcp, tmp_path, wind_from_deg=270.0,
        n_ignitions=6, dist_band_km=(0.5, 2.0), cone_half_angle_deg=30.0, seed=1,
    )
    paths = res["ignition_shapefiles"]
    assert len(paths) == 6
    for p in paths:
        assert len(gpd.read_file(p)) == 1


def test_design_ignitions_span_the_strata(synth_lcp, design_geometry, tmp_path):
    """Stratification exists so the design covers the transmission geometry."""
    trt, val = design_geometry
    res = select_design_ignitions(
        trt, val, synth_lcp, tmp_path, wind_from_deg=270.0,
        n_ignitions=6, dist_band_km=(0.5, 2.0), cone_half_angle_deg=30.0,
        n_bearing_strata=3, n_distance_strata=2, seed=1,
    )
    g = res["ignitions_gdf"]
    assert g.bear_stratum.nunique() == 3
    assert g.dist_stratum.nunique() == 2
    assert g.stratum.nunique() == 6


def test_design_ignitions_are_upwind_and_ordered(synth_lcp, design_geometry,
                                                 tmp_path):
    """Every ignition sits west of the treatment and nearer it than the values."""
    trt, val = design_geometry
    res = select_design_ignitions(
        trt, val, synth_lcp, tmp_path, wind_from_deg=270.0,
        n_ignitions=6, dist_band_km=(0.5, 2.0), cone_half_angle_deg=30.0,
        require_ordering=True, seed=1,
    )
    pts = res["ignition_points_gdf"]
    trt_u = trt.geometry.union_all()
    val_u = val.geometry.union_all()
    assert (pts.geometry.x < trt_u.centroid.x).all()
    assert (pts.geometry.distance(trt_u) < pts.geometry.distance(val_u)).all()


def test_design_ignitions_respect_the_distance_band(synth_lcp, design_geometry,
                                                    tmp_path):
    trt, val = design_geometry
    res = select_design_ignitions(
        trt, val, synth_lcp, tmp_path, wind_from_deg=270.0,
        n_ignitions=6, dist_band_km=(0.5, 2.0), cone_half_angle_deg=30.0, seed=1,
    )
    g = res["ignitions_gdf"]
    assert (g.dist_m >= 500).all() and (g.dist_m <= 2000).all()


def test_design_ignitions_land_on_burnable_fuel(synth_lcp, design_geometry,
                                                tmp_path):
    trt, val = design_geometry
    res = select_design_ignitions(
        trt, val, synth_lcp, tmp_path, wind_from_deg=270.0,
        n_ignitions=6, dist_band_km=(0.5, 2.0), cone_half_angle_deg=30.0, seed=1,
    )
    grid = _read_burnable_mask(synth_lcp)
    inv = ~grid["transform"]
    for pt in res["ignition_points_gdf"].geometry:
        col, row = inv * (pt.x, pt.y)
        assert grid["mask"][int(row), int(col)]


def test_design_ignitions_default_footprint_is_ten_acres(synth_lcp,
                                                         design_geometry,
                                                         tmp_path):
    trt, val = design_geometry
    res = select_design_ignitions(
        trt, val, synth_lcp, tmp_path, wind_from_deg=270.0,
        n_ignitions=4, dist_band_km=(0.5, 2.0), cone_half_angle_deg=30.0, seed=1,
    )
    assert np.allclose(res["ignitions_gdf"].geometry.area / _ACRE_M2, 10.0,
                       rtol=1e-3)


def test_design_ignitions_uniform_without_density(synth_lcp, design_geometry,
                                                  tmp_path):
    """No density surface reproduces Ager's uniform-ignition assumption."""
    trt, val = design_geometry
    res = select_design_ignitions(
        trt, val, synth_lcp, tmp_path, wind_from_deg=270.0, density=None,
        n_ignitions=6, dist_band_km=(0.5, 2.0), cone_half_angle_deg=30.0, seed=1,
    )
    assert res["uniform_weights"]
    assert math.isclose(res["ignitions_gdf"].w_i.sum(), 1.0, abs_tol=1e-5)


def test_design_ignitions_reproducible(synth_lcp, design_geometry, tmp_path):
    trt, val = design_geometry
    kwargs = dict(wind_from_deg=270.0, n_ignitions=6, dist_band_km=(0.5, 2.0),
                  cone_half_angle_deg=30.0, seed=99)
    a = select_design_ignitions(trt, val, synth_lcp, tmp_path / "a", **kwargs)
    b = select_design_ignitions(trt, val, synth_lcp, tmp_path / "b", **kwargs)
    assert list(a["ignition_points_gdf"].geometry.x) == \
           list(b["ignition_points_gdf"].geometry.x)


def test_design_ignitions_raise_when_band_leaves_the_domain(synth_lcp,
                                                            design_geometry,
                                                            tmp_path):
    """A distance band larger than the landscape has no candidates at all."""
    trt, val = design_geometry
    with pytest.raises(ValueError, match="placement wedge"):
        select_design_ignitions(
            trt, val, synth_lcp, tmp_path, wind_from_deg=270.0,
            n_ignitions=6, dist_band_km=(20.0, 30.0),
            cone_half_angle_deg=30.0, seed=1,
        )


def test_design_ignitions_raise_when_values_come_first(synth_lcp, tmp_path):
    """
    With the values upwind of the treatment there is no pathway *through* the
    treatment to measure, so require_ordering must reject every candidate.
    """
    import rasterio

    with rasterio.open(synth_lcp) as src:
        crs, b = src.crs, src.bounds
    cx, cy = (b.left + b.right) / 2, (b.bottom + b.top) / 2

    trt = gpd.GeoDataFrame(geometry=[box(cx + 400, cy - 500, cx + 1200, cy + 500)],
                           crs=crs)
    # Values blanket the whole upwind candidate zone, so fire reaches them
    # before it ever reaches the treatment — no pathway through it to measure.
    val = gpd.GeoDataFrame(geometry=[box(cx - 2500, cy - 2500, cx + 600, cy + 2500)],
                           crs=crs)

    with pytest.raises(ValueError, match="no transmission pathway"):
        select_design_ignitions(
            trt, val, synth_lcp, tmp_path, wind_from_deg=270.0,
            n_ignitions=6, dist_band_km=(0.5, 2.0), cone_half_angle_deg=30.0,
            require_ordering=True, seed=1,
        )


def test_design_ignitions_without_values(synth_lcp, design_geometry, tmp_path):
    """values_gdf=None skips the ordering and downwind checks."""
    trt, _ = design_geometry
    res = select_design_ignitions(
        trt, None, synth_lcp, tmp_path, wind_from_deg=270.0,
        n_ignitions=4, dist_band_km=(0.5, 2.0), cone_half_angle_deg=30.0, seed=1,
    )
    assert len(res["ignitions_gdf"]) == 4
    assert res["n_after_ordering"] == res["n_after_cone"]
