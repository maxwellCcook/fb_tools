"""
Phase 3a regression tests for :mod:`fb_tools.spread.bp`.

Every test here pins a defect that was live on ``main`` before Phase 3a:

- the int16 delta write had no nodata, so masked pixels descaled to −327.68;
- ``summarize_bp_treatments`` called ``geom_to_raster_crs`` with three
  arguments against a two-argument signature (``TypeError`` on the first
  polygon) and differenced two independently clipped, independently
  NaN-filtered flat arrays positionally;
- ``downwind_treatment_effect`` set ``src_crs = None`` on both branches, so its
  reprojection was unreachable and a WGS84 polygon returned all-NaN;
- ``aggregate_ignition_bp`` had no design weights and reported an
  ``n_ignitions`` count that cannot detect co-burn for burn probability;
- nothing masked the ignition footprint, inside which BP ≈ 1 in every arm.

All fixtures are built in-process — nothing here touches ``data/``.
"""

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
rxr = pytest.importorskip("rioxarray")
gpd = pytest.importorskip("geopandas")

from rasterio.transform import from_origin
from shapely.geometry import box

from fb_tools.spread.bp import (
    DELTA_NODATA,
    delta_burn_probability,
    aggregate_ignition_bp,
    summarize_bp_treatments,
    downwind_treatment_effect,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

N = 100
RES = 30.0
LEFT, TOP = -800_000.0, 1_960_000.0
CRS = "EPSG:5070"


def _write_bp(path, arr, res=RES, left=LEFT, top=TOP, crs=CRS, nodata=np.nan):
    """Write a float32 single-band raster and return its path."""
    arr = np.asarray(arr, dtype="float32")
    profile = dict(
        driver="GTiff", dtype="float32", count=1,
        height=arr.shape[0], width=arr.shape[1],
        crs=crs, transform=from_origin(left, top, res, res),
        nodata=nodata,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
    return path


def _xy(row, col):
    """Centre coordinate of (row, col) on the test grid."""
    return LEFT + (col + 0.5) * RES, TOP - (row + 0.5) * RES


@pytest.fixture
def bp_pair(tmp_path):
    """
    Baseline and treated BP rasters with a known, exactly recoverable delta.

    Baseline is 0.8 over the left half and 0.4 over the right half.  Treated
    subtracts 0.2 from a 20x20 block, so the true delta is 0.2 there and 0.0
    everywhere else.
    """
    bl = np.where(np.arange(N)[None, :] < N // 2, 0.8, 0.4).repeat(N, axis=0)
    bl = np.broadcast_to(bl, (N, N)).astype("float32").copy()
    tr = bl.copy()
    tr[40:60, 20:40] -= 0.2

    return (
        _write_bp(tmp_path / "baseline.tif", bl),
        _write_bp(tmp_path / "treated.tif", tr),
    )


@pytest.fixture
def ignition_gdf():
    """A 10x10-cell ignition footprint at rows/cols 45–55."""
    x0, y0 = _xy(55, 45)
    x1, y1 = _xy(45, 55)
    return gpd.GeoDataFrame(
        geometry=[box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))],
        crs=CRS,
    )


# ── delta_burn_probability ───────────────────────────────────────────────────

def test_delta_is_baseline_minus_treated(bp_pair):
    bl, tr = bp_pair
    d = delta_burn_probability(bl, tr, verbose=False)

    assert d.shape == (N, N)
    # Treated block was lowered by 0.2, so the delta is +0.2 there.
    assert np.allclose(d.values[40:60, 20:40], 0.2, atol=1e-6)
    # Everywhere else the arms are identical.
    untouched = d.values.copy()
    untouched[40:60, 20:40] = 0.0
    assert np.allclose(untouched, 0.0, atol=1e-6)


def test_int16_write_roundtrips_nodata_not_minus_327(bp_pair, tmp_path, ignition_gdf):
    """
    The headline regression: masked pixels must read back as NaN.

    Before Phase 3a the int16 cast had no nodata set, so NaN became −32768 and
    descaling by ``scale`` produced −327.68 — a plausible-looking, wildly wrong
    burn-probability delta that no downstream check would flag.
    """
    bl, tr = bp_pair
    out = tmp_path / "delta.tif"

    delta_burn_probability(
        bl, tr, out_path=out, scale=100, ignition=ignition_gdf, verbose=False
    )

    with rasterio.open(out) as src:
        assert src.nodata == DELTA_NODATA, "nodata tag missing from the delta raster"
        band = src.read(1, masked=True)

    # The ignition footprint must come back masked, never as a real value.
    assert band.mask.any(), "no pixels were masked in the written delta"
    assert band.mask[50, 50], "ignition footprint pixel is not masked"

    descaled = band.astype("float32") / 100.0
    assert not np.any(np.isclose(descaled.compressed(), -327.68, atol=0.01)), \
        "the -327.68 descaling artefact is back"

    # Unmasked values still round-trip at 1/scale precision.
    assert np.isclose(descaled[45, 25], 0.2, atol=0.01)
    assert np.isclose(descaled[5, 5], 0.0, atol=0.01)


def test_write_rejects_int16_overflow(bp_pair, tmp_path):
    bl, tr = bp_pair
    with pytest.raises(ValueError, match="does not fit in int16"):
        delta_burn_probability(
            bl, tr, out_path=tmp_path / "d.tif", scale=1_000_000, verbose=False
        )


def test_strict_grid_rejects_shape_mismatch(bp_pair, tmp_path):
    bl, _ = bp_pair
    small = _write_bp(tmp_path / "small.tif", np.zeros((N // 2, N // 2)))
    with pytest.raises(ValueError, match="shape"):
        delta_burn_probability(bl, small, verbose=False)


def test_strict_grid_rejects_shifted_transform(bp_pair, tmp_path):
    """A half-pixel origin shift mispairs every cell but keeps the same shape."""
    bl, _ = bp_pair
    shifted = _write_bp(
        tmp_path / "shifted.tif", np.zeros((N, N)), left=LEFT + RES / 2
    )
    with pytest.raises(ValueError, match="transform"):
        delta_burn_probability(bl, shifted, verbose=False)


def test_strict_grid_can_be_disabled(bp_pair, tmp_path):
    bl, _ = bp_pair
    shifted = _write_bp(
        tmp_path / "shifted.tif", np.zeros((N, N)), left=LEFT + RES / 2
    )
    d = delta_burn_probability(bl, shifted, strict_grid=False, verbose=False)
    assert d.shape == (N, N)


def test_ignition_footprint_is_masked(bp_pair, ignition_gdf):
    bl, tr = bp_pair
    d = delta_burn_probability(bl, tr, ignition=ignition_gdf, verbose=False)

    assert np.isnan(d.values[50, 50]), "footprint interior should be NaN"
    assert not np.isnan(d.values[5, 5]), "pixels outside the footprint survive"
    # Roughly 10x10 cells masked (all_touched can add a boundary ring).
    assert 100 <= int(np.isnan(d.values).sum()) <= 144


# ── aggregate_ignition_bp ────────────────────────────────────────────────────

@pytest.fixture
def three_ignitions(tmp_path):
    """Three baseline/treated pairs with per-ignition deltas of 0.1/0.2/0.3."""
    pairs = []
    for i, step in enumerate((0.1, 0.2, 0.3)):
        bl = np.full((N, N), 0.6, dtype="float32")
        tr = bl - step
        pairs.append((
            _write_bp(tmp_path / f"bl_{i}.tif", bl),
            _write_bp(tmp_path / f"tr_{i}.tif", tr),
        ))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def test_equal_weights_give_the_plain_mean(three_ignitions):
    bls, trs = three_ignitions
    res = aggregate_ignition_bp(bls, trs, verbose=False)
    # (0.1 + 0.2 + 0.3) / 3
    assert np.isclose(float(res["delta_mean"].values[10, 10]), 0.2, atol=1e-5)
    assert np.allclose(res["weights"], 1 / 3)


def test_weights_shift_the_ensemble_mean(three_ignitions):
    """Design weights must actually reach the estimator."""
    bls, trs = three_ignitions
    res = aggregate_ignition_bp(bls, trs, weights=[0.7, 0.2, 0.1], verbose=False)
    expected = 0.7 * 0.1 + 0.2 * 0.2 + 0.1 * 0.3
    assert np.isclose(float(res["delta_mean"].values[10, 10]), expected, atol=1e-5)


def test_weights_are_normalized(three_ignitions):
    bls, trs = three_ignitions
    a = aggregate_ignition_bp(bls, trs, weights=[2, 2, 2], verbose=False)
    b = aggregate_ignition_bp(bls, trs, verbose=False)
    assert np.isclose(
        float(a["delta_mean"].values[10, 10]),
        float(b["delta_mean"].values[10, 10]),
        atol=1e-6,
    )


def test_weights_keyed_by_ignition_id(three_ignitions):
    bls, trs = three_ignitions
    res = aggregate_ignition_bp(
        bls, trs,
        weights={"a": 0.7, "b": 0.2, "c": 0.1},
        ignition_ids=["a", "b", "c"],
        verbose=False,
    )
    expected = 0.7 * 0.1 + 0.2 * 0.2 + 0.1 * 0.3
    assert np.isclose(float(res["delta_mean"].values[10, 10]), expected, atol=1e-5)


def test_weights_stay_paired_when_an_ignition_is_dropped(three_ignitions):
    """
    A missing output must not shift the weight-to-ignition correspondence.

    Weights are resolved against the full input list *before* filtering, so
    dropping the middle ignition keeps w=[0.7, 0.1] (renormalized), not the
    first two weights [0.7, 0.2].
    """
    bls, trs = three_ignitions
    bls = [bls[0], None, bls[2]]
    trs = [trs[0], None, trs[2]]

    res = aggregate_ignition_bp(bls, trs, weights=[0.7, 0.2, 0.1], verbose=False)

    assert np.allclose(res["weights"], [0.875, 0.125], atol=1e-6)
    expected = 0.875 * 0.1 + 0.125 * 0.3
    assert np.isclose(float(res["delta_mean"].values[10, 10]), expected, atol=1e-5)


@pytest.mark.parametrize("bad,err", [
    ([0.5, 0.5], "length"),
    ([1.0, -1.0, 1.0], "negative"),
    ([0.0, 0.0, 0.0], "sum to zero"),
    ([np.nan, 0.5, 0.5], "NaN"),
])
def test_invalid_weights_raise(three_ignitions, bad, err):
    bls, trs = three_ignitions
    with pytest.raises(ValueError, match=err):
        aggregate_ignition_bp(bls, trs, weights=bad, verbose=False)


def test_missing_weight_key_raises(three_ignitions):
    bls, trs = three_ignitions
    with pytest.raises(ValueError, match="missing entries"):
        aggregate_ignition_bp(
            bls, trs, weights={"a": 1.0}, ignition_ids=["a", "b", "c"], verbose=False
        )


def test_n_burned_distinguishes_unburned_bp_from_n_ignitions(tmp_path):
    """
    ``n_ignitions`` cannot detect co-burn for burn probability.

    Unburned interior pixels are ``0.0``, not nodata, so ``notnull()`` counts
    every ignition everywhere.  ``n_burned`` is the count that means something.
    """
    bls, trs = [], []
    for i in range(3):
        bl = np.zeros((N, N), dtype="float32")
        bl[:, : (i + 1) * 20] = 0.5      # each ignition burns a wider strip
        trs.append(_write_bp(tmp_path / f"t{i}.tif", bl * 0.9))
        bls.append(_write_bp(tmp_path / f"b{i}.tif", bl))

    res = aggregate_ignition_bp(bls, trs, verbose=False)

    # Valid-data count is the full ignition count everywhere — the old metric.
    assert int(res["n_ignitions"].values[10, 90]) == 3
    # Burned-support count reflects which ignitions actually reached the pixel.
    assert int(res["n_burned"].values[10, 10]) == 3   # burned by all three
    assert int(res["n_burned"].values[10, 30]) == 2   # by ignitions 1 and 2
    assert int(res["n_burned"].values[10, 90]) == 0   # burned by none


def test_delta_std_is_written_to_disk(three_ignitions, tmp_path):
    bls, trs = three_ignitions
    out = tmp_path / "ens"
    res = aggregate_ignition_bp(bls, trs, out_dir=out, verbose=False)

    assert (out / "ensemble_delta_std.tif").exists(), \
        "delta_std was computed but never written"
    assert (out / "ensemble_n_burned.tif").exists()
    # Population sd of (0.1, 0.2, 0.3) about their mean.
    assert np.isclose(float(res["delta_std"].values[10, 10]),
                      np.std([0.1, 0.2, 0.3]), atol=1e-5)


def test_per_ignition_footprints_are_masked(three_ignitions, ignition_gdf):
    bls, trs = three_ignitions
    res = aggregate_ignition_bp(
        bls, trs, ignitions=[ignition_gdf] * 3, verbose=False
    )
    assert np.isnan(float(res["delta_mean"].values[50, 50]))
    assert not np.isnan(float(res["delta_mean"].values[5, 5]))


def test_ignitions_length_mismatch_raises(three_ignitions, ignition_gdf):
    bls, trs = three_ignitions
    with pytest.raises(ValueError, match="one footprint per design fire"):
        aggregate_ignition_bp(bls, trs, ignitions=[ignition_gdf], verbose=False)


# ── summarize_bp_treatments ──────────────────────────────────────────────────

@pytest.fixture
def zones():
    """Two zones: one over the treated block, one well away from it."""
    x0, y0 = _xy(59, 20)
    x1, y1 = _xy(40, 39)
    inside = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    x0, y0 = _xy(15, 70)
    x1, y1 = _xy(5, 80)
    outside = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    return gpd.GeoDataFrame(
        {"TRT_ID": [1, 2], "TRT_TYPE": ["thin", "control"]},
        geometry=[inside, outside],
        crs=CRS,
    )


def test_summarize_runs_and_matches_hand_computation(bp_pair, zones):
    """Regression: this function used to raise TypeError on the first polygon."""
    bl, tr = bp_pair
    df = summarize_bp_treatments(
        zones, baseline_bp=bl, treatment_bp=tr,
        id_col="TRT_ID", type_col="TRT_TYPE", verbose=False,
    )

    assert list(df["TRT_ID"]) == [1, 2]
    treated = df.set_index("TRT_ID").loc[1]
    control = df.set_index("TRT_ID").loc[2]

    assert np.isclose(treated["dBP_mean"], 0.2, atol=0.02)
    assert np.isclose(control["dBP_mean"], 0.0, atol=1e-6)
    assert treated["pct_improved"] > 90.0
    assert control["pct_improved"] == 0.0


def test_summarize_accepts_a_wgs84_zones_frame(bp_pair, zones):
    """Zones are reprojected to the raster CRS, not assumed to match it."""
    bl, tr = bp_pair
    df = summarize_bp_treatments(
        zones.to_crs("EPSG:4326"), baseline_bp=bl, treatment_bp=tr,
        id_col="TRT_ID", verbose=False,
    )
    assert np.isclose(df.set_index("TRT_ID").loc[1, "dBP_mean"], 0.2, atol=0.02)


def test_summarize_pairs_cellwise_across_arm_nodata(tmp_path, zones):
    """
    The pairing regression.

    Treated carries nodata where baseline does not.  The old implementation
    dropped NaNs from each side *independently* and then subtracted the
    flattened arrays positionally, so every value after the first hole was
    paired with the wrong cell.  Differencing before filtering is the fix:
    a cell missing in either arm simply drops out.
    """
    bl = np.full((N, N), 0.6, dtype="float32")
    tr = np.full((N, N), 0.4, dtype="float32")
    tr[45:50, 25:30] = np.nan          # hole in the treated arm only

    bl_p = _write_bp(tmp_path / "b.tif", bl)
    tr_p = _write_bp(tmp_path / "t.tif", tr)

    df = summarize_bp_treatments(
        zones, baseline_bp=bl_p, treatment_bp=tr_p,
        id_col="TRT_ID", verbose=False,
    )
    row = df.set_index("TRT_ID").loc[1]

    # Every surviving cell pairs 0.6 against 0.4 — the delta is exactly 0.2.
    assert np.isclose(row["dBP_mean"], 0.2, atol=1e-6)
    assert np.isclose(row["dBP_min"], 0.2, atol=1e-6)
    assert np.isclose(row["dBP_max"], 0.2, atol=1e-6)
    # The 5x5 hole is excluded from the 20x20 zone.
    assert row["n_pixels"] == pytest.approx(400 - 25, abs=45)


def test_sum_ha_is_the_area_integral(bp_pair, zones):
    bl, tr = bp_pair
    df = summarize_bp_treatments(
        zones, baseline_bp=bl, treatment_bp=tr, id_col="TRT_ID", verbose=False
    )
    row = df.set_index("TRT_ID").loc[1]
    cell_ha = RES ** 2 / 10_000.0
    assert np.isclose(row["dBP_sum_ha"], row["dBP_mean"] * row["n_pixels"] * cell_ha,
                      rtol=1e-5)


def test_min_pixels_flags_small_zones(bp_pair, zones):
    bl, tr = bp_pair
    df = summarize_bp_treatments(
        zones, baseline_bp=bl, treatment_bp=tr, id_col="TRT_ID",
        min_pixels=100_000, verbose=False,
    )
    assert not df["reliable"].any()


def test_summarize_requires_a_delta_source(zones):
    with pytest.raises(ValueError, match="Provide either delta_bp"):
        summarize_bp_treatments(zones, id_col="TRT_ID", verbose=False)


def test_summarize_rejects_a_missing_id_col(bp_pair, zones):
    bl, tr = bp_pair
    with pytest.raises(ValueError, match="id_col"):
        summarize_bp_treatments(
            zones, baseline_bp=bl, treatment_bp=tr, id_col="NOPE", verbose=False
        )


# ── downwind_treatment_effect ────────────────────────────────────────────────

@pytest.fixture
def east_signal_delta(tmp_path):
    """A delta raster that is 0.5 on the eastern half and 0.0 on the western."""
    arr = np.zeros((N, N), dtype="float32")
    arr[:, N // 2:] = 0.5
    return _write_bp(tmp_path / "delta.tif", arr)


@pytest.fixture
def centre_polygon():
    x0, y0 = _xy(55, 45)
    x1, y1 = _xy(45, 55)
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def test_downwind_sector_follows_the_wind(east_signal_delta, centre_polygon):
    """Wind FROM the west blows east, so the eastern signal must be found."""
    east = downwind_treatment_effect(
        centre_polygon, east_signal_delta, wind_direction=270,
        buffer_km=2.0, src_crs=CRS, verbose=False,
    )
    west = downwind_treatment_effect(
        centre_polygon, east_signal_delta, wind_direction=90,
        buffer_km=2.0, src_crs=CRS, verbose=False,
    )

    assert east["downwind_azimuth"] == 90.0
    assert east["mean_delta_bp"] > 0.4

    # The sector apex sits on the signal boundary, so `all_touched=True` grabs a
    # few eastern pixels at the cone tip even when pointing west.  That is
    # correct geometry, not leakage — assert the separation, not an exact zero.
    assert west["mean_delta_bp"] < 0.005
    assert east["mean_delta_bp"] > 100 * west["mean_delta_bp"]


def test_wgs84_polygon_is_reprojected(east_signal_delta, centre_polygon):
    """
    The CRS regression.

    ``src_crs`` was previously hard-set to ``None`` on both branches, making the
    reprojection unreachable.  A WGS84 polygon then landed nowhere near the
    projected raster and every statistic came back NaN.
    """
    wgs = gpd.GeoSeries([centre_polygon], crs=CRS).to_crs("EPSG:4326")

    result = downwind_treatment_effect(
        wgs.iloc[0], east_signal_delta, wind_direction=270,
        buffer_km=2.0, src_crs="EPSG:4326", verbose=False,
    )

    assert result["n_pixels"] > 0, "WGS84 polygon still yields an empty sector"
    assert result["mean_delta_bp"] > 0.4


def test_geodataframe_carries_its_own_crs(east_signal_delta, centre_polygon):
    gdf = gpd.GeoDataFrame(geometry=[centre_polygon], crs=CRS).to_crs("EPSG:4326")
    result = downwind_treatment_effect(
        gdf, east_signal_delta, wind_direction=270, buffer_km=2.0, verbose=False
    )
    assert result["n_pixels"] > 0
    assert result["mean_delta_bp"] > 0.4


@pytest.mark.parametrize("sentinel,kind", [(-1, "uphill"), (-2, "downhill")])
def test_flammap_wind_sentinels_are_rejected(
    east_signal_delta, centre_polygon, sentinel, kind
):
    """``-1``/``-2`` are slope flags, not azimuths — a sector is undefined."""
    with pytest.raises(ValueError, match=kind):
        downwind_treatment_effect(
            centre_polygon, east_signal_delta, wind_direction=sentinel,
            src_crs=CRS, verbose=False,
        )


def test_out_of_range_wind_direction_is_rejected(east_signal_delta, centre_polygon):
    with pytest.raises(ValueError, match="outside 0"):
        downwind_treatment_effect(
            centre_polygon, east_signal_delta, wind_direction=451,
            src_crs=CRS, verbose=False,
        )


def test_missing_wind_direction_is_rejected(east_signal_delta, centre_polygon):
    with pytest.raises(ValueError, match="wind_direction"):
        downwind_treatment_effect(
            centre_polygon, east_signal_delta, src_crs=CRS, verbose=False
        )
