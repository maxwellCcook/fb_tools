"""
Ignition-density grids by cause (`ignition_density_grid`, `load_fod`).

Engine-free and data-free: points, AOIs and a tiny FOD GeoPackage are built
in-process.
"""

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
from shapely.geometry import box

from fb_tools.fuelscape.ignitions import (
    LANDFIRE_GRID_ORIGIN,
    _smooth_points_onto_grid,
    convert_density_units,
    ignition_density_grid,
    load_fod,
    write_density_grids,
)

CRS = "EPSG:5070"
X0, Y0 = -800_015.0, 1_900_015.0          # on LANDFIRE pixel edges (15 mod 30)


def _aoi(w=60_000.0, h=40_000.0):
    return gpd.GeoDataFrame(geometry=[box(X0, Y0, X0 + w, Y0 + h)], crs=CRS)


def _points(xy, cause, year=2000):
    xy = np.atleast_2d(xy)
    return gpd.GeoDataFrame(
        {"cause": [cause] * len(xy), "FIRE_YEAR": [year] * len(xy)},
        geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=CRS,
    )


def _cluster(rng, cx, cy, n, sd, cause):
    xy = np.column_stack([rng.normal(cx, sd, n), rng.normal(cy, sd, n)])
    return _points(xy, cause)


@pytest.fixture
def two_cause_fod():
    rng = np.random.default_rng(3)
    hum = _cluster(rng, X0 + 15_000, Y0 + 20_000, 400, 3_000, "human")
    nat = _cluster(rng, X0 + 45_000, Y0 + 20_000, 100, 3_000, "natural")
    unk = _cluster(rng, X0 + 30_000, Y0 + 20_000, 50, 3_000, "unknown")
    return gpd.pd.concat([hum, nat, unk], ignore_index=True)


# ── the shared smoother ───────────────────────────────────────────────────────

@pytest.mark.parametrize("res,bw", [(1000.0, 3000.0), (90.0, 5000.0), (30.0, 600.0)])
def test_smoother_conserves_mass(res, bw):
    from rasterio.transform import from_origin
    rng = np.random.default_rng(0)
    x = rng.normal(50_000, 2_000, 300)
    y = rng.normal(-50_000, 2_000, 300)
    n = int(100_000 / res)
    d, n_used, n_in, _ = _smooth_points_onto_grid(
        x, y, from_origin(0, 0, res, res), (n, n), bw, 3 * bw)
    assert n_used == n_in == 300
    assert d.sum() == pytest.approx(300, rel=0.01)


def test_smoother_peak_sits_on_the_point():
    """Regression: resampling used to sample working cells half a cell off."""
    from rasterio.transform import from_origin
    res, bw = 90.0, 5000.0
    n = 400
    px, py = 18_045.0, -18_045.0          # centre of cell (200, 200)
    d, *_ = _smooth_points_onto_grid([px], [py], from_origin(0, 0, res, res),
                                     (n, n), bw, 3 * bw)
    r, c = np.unravel_index(np.argmax(d), d.shape)
    assert abs(r - 200) <= 1 and abs(c - 200) <= 1


# ── ignition_density_grid ────────────────────────────────────────────────────

def test_grid_edges_snap_to_origin():
    aoi = gpd.GeoDataFrame(geometry=[box(X0 + 7, Y0 + 11, X0 + 9_100, Y0 + 5_050)], crs=CRS)
    pts = _points([[X0 + 4_000, Y0 + 2_000]], "human")
    ds = ignition_density_grid(pts, aoi, res_m=90.0, bandwidth_m=1000.0, years=(2000, 2000))
    t = ds.rio.transform()
    ox, oy = LANDFIRE_GRID_ORIGIN
    assert (t.c - ox) % 90 == pytest.approx(0) and (t.f - oy) % 90 == pytest.approx(0)
    left, bottom, right, top = aoi.total_bounds
    assert t.c <= left and t.f >= top
    assert t.c + ds.sizes["x"] * 90 >= right and t.f - ds.sizes["y"] * 90 <= bottom


def test_non_albers_grid_snaps_to_zero_origin():
    aoi = _aoi().to_crs("EPSG:26913")
    pts = _points([[X0 + 30_000, Y0 + 20_000]], "human")
    ds = ignition_density_grid(pts, aoi, crs="EPSG:26913", res_m=90.0,
                               bandwidth_m=2000.0, years=(2000, 2000))
    t = ds.rio.transform()
    assert ds.rio.crs.to_epsg() == 26913
    assert t.c % 90 == pytest.approx(0) and t.f % 90 == pytest.approx(0)
    assert float(ds["human"].sum()) * 90.0 ** 2 / 1e8 == pytest.approx(1.0, rel=0.02)


def test_rate_units_integrate_to_fires_per_year(two_cause_fod):
    ds = ignition_density_grid(two_cause_fod, _aoi(), res_m=500.0, bandwidth_m=1500.0,
                               years=(2000, 2004), mask_to_aoi=False)
    cell_100km2 = 500.0 ** 2 / 1e8
    # 400 human fires over 5 years, well inside the AOI → 80 fires/yr
    assert float(ds["human"].sum()) * cell_100km2 == pytest.approx(80.0, rel=0.02)
    assert float(ds["natural"].sum()) * cell_100km2 == pytest.approx(20.0, rel=0.02)
    assert float(ds["all"].sum()) * cell_100km2 == pytest.approx(110.0, rel=0.02)
    assert ds.attrs["n_years"] == 5


def test_all_includes_unknown_and_difference_is_h_minus_n(two_cause_fod):
    ds = ignition_density_grid(two_cause_fod, _aoi(), res_m=1000.0, bandwidth_m=2000.0,
                               years=(2000, 2000))
    np.testing.assert_allclose(
        ds["all"], ds["human"] + ds["natural"] + ds["unknown"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(ds["difference"], ds["human"] - ds["natural"],
                               rtol=1e-5, atol=1e-6)


def test_human_share_bounded_and_follows_the_clusters(two_cause_fod):
    ds = ignition_density_grid(two_cause_fod, _aoi(), res_m=1000.0, bandwidth_m=2000.0,
                               years=(2000, 2000), share_min_fires=1.0)
    s = ds["human_share"].values
    ok = np.isfinite(s)
    assert ok.any() and (s[ok] >= 0).all() and (s[ok] <= 1).all()
    row = ds.sizes["y"] // 2
    assert s[row, 15] > 0.95          # human cluster
    assert s[row, 45] < 0.05          # natural cluster


def test_human_share_masked_where_sparse(two_cause_fod):
    ds = ignition_density_grid(two_cause_fod, _aoi(), res_m=1000.0, bandwidth_m=2000.0,
                               years=(2000, 2000), share_min_fires=10.0)
    s = ds["human_share"].values
    assert np.isnan(s[0, 0])          # far corner — no fires under the kernel
    assert np.isfinite(s[ds.sizes["y"] // 2, 15])


def test_mask_to_aoi_sets_outside_to_nan(two_cause_fod):
    aoi = gpd.GeoDataFrame(
        geometry=[box(X0, Y0, X0 + 60_000, Y0 + 40_000).difference(
            box(X0, Y0, X0 + 20_000, Y0 + 20_000))], crs=CRS)
    ds = ignition_density_grid(two_cause_fod, aoi, res_m=1000.0, bandwidth_m=2000.0,
                               years=(2000, 2000))
    h = ds["human"].values
    assert np.isnan(h[-5, 5])                       # in the notch (bottom-left)
    assert np.isfinite(h[5, 5])


def test_template_defines_the_grid(synth_lcp, two_cause_fod):
    import rasterio
    with rasterio.open(synth_lcp) as src:
        b, t, shape = src.bounds, src.transform, src.shape
    rng = np.random.default_rng(1)
    pts = _cluster(rng, (b.left + b.right) / 2, (b.top + b.bottom) / 2, 50, 500, "human")
    ds = ignition_density_grid(pts, template=synth_lcp, bandwidth_m=600.0,
                               years=(2000, 2000))
    assert ds["human"].shape == shape
    assert ds.rio.transform() == t


def test_needs_an_extent(two_cause_fod):
    with pytest.raises(ValueError, match="aoi"):
        ignition_density_grid(two_cause_fod)


def test_rejects_geographic_crs(two_cause_fod):
    with pytest.raises(ValueError, match="projected"):
        ignition_density_grid(two_cause_fod, _aoi(), crs="EPSG:4326")


def test_write_density_grids_round_trips(two_cause_fod, tmp_path):
    import rasterio
    ds = ignition_density_grid(two_cause_fod, _aoi(), res_m=1000.0, bandwidth_m=2000.0,
                               years=(2000, 2000))
    paths = write_density_grids(ds, tmp_path, prefix="t")
    assert {p.name for p in paths} == {f"t_{k}.tif" for k in ds.data_vars}
    with rasterio.open(tmp_path / "t_human.tif") as src:
        np.testing.assert_allclose(src.read(1), ds["human"].values, equal_nan=True)
        assert src.tags()["bandwidth_m"] == "2000.0"
        assert src.crs.to_epsg() == 5070


# ── load_fod ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_fod_gpkg(tmp_path):
    rows = [
        # lon, lat, year, size class, cause
        (-105.5, 40.5, 1995, "A", "Human"),
        (-105.4, 40.6, 2010, "D", "Natural"),
        (-105.3, 40.4, 2020, "B", "Missing data/not specified/undetermined"),
        (-110.0, 45.0, 2010, "D", "Human"),          # far outside the AOI
    ]
    gdf = gpd.GeoDataFrame(
        {
            "FOD_ID": range(len(rows)),
            "FIRE_YEAR": [r[2] for r in rows],
            "DISCOVERY_DATE": "2000-01-01",
            "FIRE_SIZE": 1.0,
            "FIRE_NAME": "x",
            "FIRE_SIZE_CLASS": [r[3] for r in rows],
            "NWCG_CAUSE_CLASSIFICATION": [r[4] for r in rows],
            "NWCG_GENERAL_CAUSE": "x",
            "STATE": "CO",
        },
        geometry=gpd.points_from_xy([r[0] for r in rows], [r[1] for r in rows]),
        crs="EPSG:4269",
    )
    fp = tmp_path / "fod.gpkg"
    gdf.to_file(fp, layer="Fires", driver="GPKG")
    return fp


def test_load_fod_maps_causes_and_clips_to_aoi(tiny_fod_gpkg):
    aoi = gpd.GeoDataFrame(geometry=[box(-105.6, 40.3, -105.2, 40.7)], crs="EPSG:4326")
    gdf = load_fod(tiny_fod_gpkg, aoi=aoi, buffer_m=1000)
    assert len(gdf) == 3
    assert sorted(gdf["cause"]) == ["human", "natural", "unknown"]


def test_load_fod_filters_years_and_sizes(tiny_fod_gpkg):
    gdf = load_fod(tiny_fod_gpkg, years=(2000, 2024), size_classes="CDEFG")
    assert sorted(gdf["FIRE_YEAR"]) == [2010, 2010]
    assert set(gdf["FIRE_SIZE_CLASS"]) == {"D"}


# ── adaptive kernel ───────────────────────────────────────────────────────────

def test_adaptive_bandwidths_shrink_where_dense():
    from fb_tools.fuelscape.ignitions import _adaptive_bandwidths
    rng = np.random.default_rng(5)
    dense = np.column_stack([rng.normal(0, 1_000, 500), rng.normal(0, 1_000, 500)])
    sparse = rng.uniform(-40_000, 40_000, (60, 2))
    xy = np.vstack([dense, sparse])
    h = _adaptive_bandwidths(xy[:, 0], xy[:, 1], h0=2_000.0, trim=5.0)
    assert np.median(h[:500]) < 2_000 < np.median(h[500:])
    assert h.min() >= 2_000 / 5 - 1e-6 and h.max() <= 2_000 * 5 + 1e-6


@pytest.fixture
def mixed_density_fod():
    rng = np.random.default_rng(8)
    town = _cluster(rng, X0 + 15_000, Y0 + 20_000, 600, 1_500, "human")
    xy = np.column_stack([rng.uniform(X0 + 5_000, X0 + 55_000, 120),
                          rng.uniform(Y0 + 5_000, Y0 + 35_000, 120)])
    lightning = _points(xy, "natural")
    unk = _cluster(rng, X0 + 40_000, Y0 + 20_000, 30, 3_000, "unknown")
    return gpd.pd.concat([town, lightning, unk], ignore_index=True)


def test_adaptive_conserves_mass_and_sums(mixed_density_fod):
    ds = ignition_density_grid(mixed_density_fod, _aoi(), res_m=250.0, bandwidth_m=2_000.0,
                               years=(2000, 2009), adaptive=True, mask_to_aoi=False)
    cell_100km2 = 250.0 ** 2 / 1e8
    # every point sits ≥ 5 km inside the grid; the widest kernel leaks a little past the edge
    assert float(ds["human"].sum()) * cell_100km2 == pytest.approx(60.0, rel=0.03)
    np.testing.assert_allclose(ds["all"], ds["human"] + ds["natural"] + ds["unknown"],
                               rtol=1e-4, atol=1e-6)
    assert ds.attrs["adaptive"] == "True"


def test_adaptive_local_bandwidth_is_narrow_in_town(mixed_density_fod):
    ds = ignition_density_grid(mixed_density_fod, _aoi(), res_m=250.0, bandwidth_m=2_000.0,
                               years=(2000, 2009), adaptive=True)
    bw = ds["bandwidth_local"]
    town = float(bw.sel(x=X0 + 15_000, y=Y0 + 20_000, method="nearest"))
    rural = float(bw.sel(x=X0 + 50_000, y=Y0 + 8_000, method="nearest"))
    assert town < 2_000 < rural
    assert ds["bandwidth_local"].attrs["units"] == "m"


def test_fixed_kernel_has_no_local_bandwidth_layer(two_cause_fod):
    ds = ignition_density_grid(two_cause_fod, _aoi(), res_m=1000.0, bandwidth_m=2000.0,
                               years=(2000, 2000))
    assert "bandwidth_local" not in ds


def test_convert_density_units_record_total_per_sq_mi(two_cause_fod):
    ds = ignition_density_grid(two_cause_fod, _aoi(), res_m=500.0, bandwidth_m=1500.0,
                               years=(2000, 2004), mask_to_aoi=False)
    tot = convert_density_units(ds, per_year=False, area="mi2")
    cell_mi2 = 500.0 ** 2 / 1e6 / 2.589988110336
    # 400 human fires over the record, well inside the AOI
    assert float(tot["human"].sum()) * cell_mi2 == pytest.approx(400.0, rel=0.02)
    assert tot["human"].attrs["units"] == "fires / sq mi (2000-2004)"
    # share untouched, input not mutated, round trip exact
    np.testing.assert_array_equal(tot["human_share"].values, ds["human_share"].values)
    assert ds["human"].attrs["units"] == "fires / yr / 100 km2"
    back = convert_density_units(tot, per_year=True, area="100km2")
    np.testing.assert_allclose(back["all"].values, ds["all"].values, rtol=1e-5)


def test_convert_density_units_rejects_unknown_area(two_cause_fod):
    ds = ignition_density_grid(two_cause_fod, _aoi(), res_m=500.0, bandwidth_m=1500.0, years=(2000, 2004))
    with pytest.raises(ValueError, match="area"):
        convert_density_units(ds, area="acre")
