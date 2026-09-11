"""
Stage 1 — the LCP de-stacker's conversion table, cell by cell.

No pyretechnics import anywhere in this file: the de-stacker is pure
rasterio/numpy and must stay testable in the ``fb_tools`` env, where the engine
is deliberately absent.

Every assertion here guards a conversion that fails *silently* — each wrong
answer is physically plausible, lands in the right ballpark, and would survive
a look at the map. The slope test is the one that matters most: LANDFIRE SLPD
is degrees, ``pyretechnics.load_landfire`` expects SLPP (percent), and reading
one as the other turns a 20 deg slope into 11.3 deg with no error.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from fb_tools.spread.convert import (
    PYRE_LAYERS,
    _canonical_band_name,
    _resolve_slope_units,
    destack_lcp,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read(fp):
    with rasterio.open(fp) as src:
        return src.read(1), src


def _write_lcp(path, bands, names, *, nodata=-9999, dtype="int16", n=20, res=30.0):
    """Minimal LCP writer for the negative cases the session fixture cannot cover."""
    profile = dict(
        driver="GTiff", dtype=dtype, count=len(bands), height=n, width=n,
        crs="EPSG:5070", transform=from_origin(-800_000.0, 1_960_000.0, res, res),
        nodata=nodata,
    )
    with rasterio.open(path, "w", **profile) as dst:
        for i, arr in enumerate(bands, start=1):
            dst.write(arr.astype(dtype), i)
            dst.set_band_description(i, names[i - 1])
    return path


@pytest.fixture(scope="module")
def destacked(synth_lcp, tmp_path_factory):
    """De-stack the session's synthetic LCP once and reuse across tests."""
    out = tmp_path_factory.mktemp("destack")
    return destack_lcp(synth_lcp, out, verbose=False), out


# ---------------------------------------------------------------------------
# the conversion table
# ---------------------------------------------------------------------------

def test_slope_degrees_become_tangent_not_percent(destacked):
    """SLPD 20 deg -> tan(20 deg) = 0.3640, NOT 0.20.

    This is the regression test for the silent-underslope bug the de-stacker
    exists to prevent. 0.20 rise/run is 11.3 deg: a 43% under-slope on exactly
    the terrain that drives spread, with a perfectly plausible-looking output.
    """
    layers, _ = destacked
    arr, _ = _read(layers["slope"])
    assert np.allclose(arr, np.tan(np.radians(20.0)), atol=1e-6)
    assert not np.allclose(arr, 0.20, atol=1e-3), "slope was scaled as percent"


def test_canopy_cover_percent_to_fraction(destacked):
    layers, _ = destacked
    arr, _ = _read(layers["canopy_cover"])
    assert np.allclose(arr, 0.45, atol=1e-6)


def test_canopy_height_decimetres_to_metres(destacked):
    layers, _ = destacked
    arr, _ = _read(layers["canopy_height"])
    assert np.allclose(arr, 15.0, atol=1e-5)


def test_canopy_base_height_decimetres_to_metres(destacked):
    layers, _ = destacked
    arr, _ = _read(layers["canopy_base_height"])
    assert np.allclose(arr, 1.5, atol=1e-6)


def test_canopy_bulk_density_scaled_by_100(destacked):
    layers, _ = destacked
    arr, _ = _read(layers["canopy_bulk_density"])
    assert np.allclose(arr, 0.10, atol=1e-6)


def test_elevation_and_aspect_pass_through(destacked):
    layers, _ = destacked
    elev, _ = _read(layers["elevation"])
    asp, _ = _read(layers["aspect"])
    assert np.allclose(elev, 2500.0)
    assert np.allclose(asp, 180.0)


def test_fuel_model_codes_survive_including_non_burnables(destacked):
    """FBFM40 is an index, not a measurement: it must arrive unscaled."""
    layers, _ = destacked
    arr, src = _read(layers["fuel_model"])
    assert set(np.unique(arr).tolist()) == {93, 98, 165}
    assert src.dtypes[0] == "int16"
    assert not np.isnan(arr.astype("float64")).any()


def test_all_eight_layers_written_and_grid_congruent(destacked, synth_lcp):
    layers, _ = destacked
    assert set(layers) == set(PYRE_LAYERS)
    with rasterio.open(synth_lcp) as ref:
        for fp in layers.values():
            with rasterio.open(fp) as dst:
                assert dst.transform == ref.transform
                assert (dst.width, dst.height) == (ref.width, ref.height)
                assert dst.crs == ref.crs


def test_continuous_layers_are_float32(destacked):
    """pyretechnics copies any non-float32 array into its SpaceTimeCube."""
    layers, _ = destacked
    for name, fp in layers.items():
        with rasterio.open(fp) as src:
            expected = "int16" if name == "fuel_model" else "float32"
            assert src.dtypes[0] == expected, f"{name} is {src.dtypes[0]}"


def test_units_recorded_in_band_tags(destacked):
    layers, _ = destacked
    with rasterio.open(layers["slope"]) as src:
        tags = src.tags()
    assert tags["units"] == "rise/run"
    assert tags["slope_units"] == "degrees"
    assert tags["source_band"] == "LF2020_SlpD_CONUS"


# ---------------------------------------------------------------------------
# nodata
# ---------------------------------------------------------------------------

def test_nodata_becomes_nb_fill_in_fuel_model(tmp_path):
    """pyretechnics has no nodata concept — an unfilled cell is undefined."""
    n = 20
    fbfm = np.full((n, n), 165, dtype="int16")
    fbfm[:5, :] = -9999
    bands = [np.full((n, n), 2500, dtype="int16"),
             np.full((n, n), 20, dtype="int16"),
             np.full((n, n), 180, dtype="int16"),
             fbfm,
             np.full((n, n), 45, dtype="int16"),
             np.full((n, n), 150, dtype="int16"),
             np.full((n, n), 15, dtype="int16"),
             np.full((n, n), 10, dtype="int16")]
    names = ["LF2020_Elev_CONUS", "LF2020_SlpD_CONUS", "LF2020_Asp_CONUS",
             "LF2016_FBFM40_CONUS", "LF2016_CC_CONUS", "LF2016_CH_CONUS",
             "LF2016_CBH_CONUS", "LF2016_CBD_CONUS"]
    lcp = _write_lcp(tmp_path / "holes.tif", bands, names, n=n)

    layers = destack_lcp(lcp, tmp_path / "out", verbose=False)
    fuel, _ = _read(layers["fuel_model"])
    assert (fuel[:5, :] == 91).all(), "nodata did not become NB1"
    assert (fuel[5:, :] == 165).all()

    # And the nodata region must not leak a -9999 into a scaled layer.
    cc, _ = _read(layers["canopy_cover"])
    assert cc.min() >= 0.0 and cc.max() <= 1.0


def test_flat_aspect_sentinel_is_resolved(tmp_path):
    """LANDFIRE writes -1 where aspect is undefined; -1 is not an azimuth."""
    n = 20
    asp = np.full((n, n), 180, dtype="int16")
    asp[:4, :] = -1
    bands = [np.full((n, n), 2500, dtype="int16"),
             np.zeros((n, n), dtype="int16"),
             asp,
             np.full((n, n), 165, dtype="int16"),
             np.full((n, n), 45, dtype="int16"),
             np.full((n, n), 150, dtype="int16"),
             np.full((n, n), 15, dtype="int16"),
             np.full((n, n), 10, dtype="int16")]
    names = ["LF2020_Elev_CONUS", "LF2020_SlpD_CONUS", "LF2020_Asp_CONUS",
             "LF2016_FBFM40_CONUS", "LF2016_CC_CONUS", "LF2016_CH_CONUS",
             "LF2016_CBH_CONUS", "LF2016_CBD_CONUS"]
    lcp = _write_lcp(tmp_path / "flat.tif", bands, names, n=n)

    layers = destack_lcp(lcp, tmp_path / "out", flat_aspect=0.0, verbose=False)
    arr, _ = _read(layers["aspect"])
    assert (arr[:4, :] == 0.0).all()
    assert (arr[4:, :] == 180.0).all()
    assert arr.min() >= 0.0 and arr.max() < 360.0


# ---------------------------------------------------------------------------
# resampling
# ---------------------------------------------------------------------------

def test_resampling_invents_no_fuel_codes(synth_lcp, tmp_path):
    """Bilinear on an FBFM40 band would average 98 and 165 into ~131 (SH-block).

    The de-stacker forces nearest/mode for categorical layers regardless of the
    `resample` argument, so the 90 m grid may only contain codes present at 30 m.
    """
    layers = destack_lcp(synth_lcp, tmp_path / "r90", resolution=90.0,
                         resample="bilinear", verbose=False)
    coarse, src = _read(layers["fuel_model"])
    assert set(np.unique(coarse).tolist()) <= {93, 98, 165}
    assert src.res == (90.0, 90.0)


def test_resampled_grid_is_congruent_across_arms(synth_lcp, tmp_path):
    """Two arms de-stacked separately at the same resolution must align exactly.

    A half-pixel offset between arms makes the paired difference meaningless
    while still producing a plausible surface, which is why Stage 3 asserts grid
    congruence — but the cheapest place to get it right is here.
    """
    a = destack_lcp(synth_lcp, tmp_path / "arm_a", resolution=90.0, verbose=False)
    b = destack_lcp(synth_lcp, tmp_path / "arm_b", resolution=90.0, verbose=False)
    with rasterio.open(a["slope"]) as sa, rasterio.open(b["slope"]) as sb:
        assert sa.transform == sb.transform
        assert (sa.width, sa.height) == (sb.width, sb.height)
        assert np.array_equal(sa.read(1), sb.read(1))


# ---------------------------------------------------------------------------
# the things that must raise
# ---------------------------------------------------------------------------

def test_missing_band_raises_and_names_what_was_found(tmp_path):
    n = 20
    bands = [np.full((n, n), 2500, dtype="int16"),
             np.full((n, n), 20, dtype="int16")]
    names = ["LF2020_Elev_CONUS", "LF2020_SlpD_CONUS"]
    lcp = _write_lcp(tmp_path / "short.tif", bands, names, n=n)
    with pytest.raises(ValueError, match="not found"):
        destack_lcp(lcp, tmp_path / "out", verbose=False)


def test_unresolvable_slope_units_raise_rather_than_guess(tmp_path):
    """An LCP whose slope band does not say SlpD or SlpP must not be guessed at."""
    n = 20
    bands = [np.full((n, n), 2500, dtype="int16"),
             np.full((n, n), 20, dtype="int16"),
             np.full((n, n), 180, dtype="int16"),
             np.full((n, n), 165, dtype="int16"),
             np.full((n, n), 45, dtype="int16"),
             np.full((n, n), 150, dtype="int16"),
             np.full((n, n), 15, dtype="int16"),
             np.full((n, n), 10, dtype="int16")]
    names = ["LF2020_Elev_CONUS", "LF2020_Slope_CONUS", "LF2020_Asp_CONUS",
             "LF2016_FBFM40_CONUS", "LF2016_CC_CONUS", "LF2016_CH_CONUS",
             "LF2016_CBH_CONUS", "LF2016_CBD_CONUS"]
    lcp = _write_lcp(tmp_path / "ambig.tif", bands, names, n=n)

    with pytest.raises(ValueError, match="Cannot resolve slope units"):
        destack_lcp(lcp, tmp_path / "out", verbose=False)

    # ... but an explicit declaration is honoured.
    layers = destack_lcp(lcp, tmp_path / "out2", slope_units="degrees", verbose=False)
    arr, _ = _read(layers["slope"])
    assert np.allclose(arr, np.tan(np.radians(20.0)), atol=1e-6)


def test_slpp_band_is_scaled_as_percent(tmp_path):
    """The other side of the same coin: SLPP 20 means 0.20 rise/run, not tan(20)."""
    n = 20
    bands = [np.full((n, n), 2500, dtype="int16"),
             np.full((n, n), 20, dtype="int16"),
             np.full((n, n), 180, dtype="int16"),
             np.full((n, n), 165, dtype="int16"),
             np.full((n, n), 45, dtype="int16"),
             np.full((n, n), 150, dtype="int16"),
             np.full((n, n), 15, dtype="int16"),
             np.full((n, n), 10, dtype="int16")]
    names = ["LF2020_Elev_CONUS", "LF2020_SlpP_CONUS", "LF2020_Asp_CONUS",
             "LF2016_FBFM40_CONUS", "LF2016_CC_CONUS", "LF2016_CH_CONUS",
             "LF2016_CBH_CONUS", "LF2016_CBD_CONUS"]
    lcp = _write_lcp(tmp_path / "slpp.tif", bands, names, n=n)
    layers = destack_lcp(lcp, tmp_path / "out", verbose=False)
    arr, _ = _read(layers["slope"])
    assert np.allclose(arr, 0.20, atol=1e-6)


def test_fabricated_fbfm_code_raises(tmp_path):
    """94-97 are unassigned in Scott & Burgan 40 and must not pass."""
    n = 20
    fbfm = np.full((n, n), 165, dtype="int16")
    fbfm[:3, :] = 95
    bands = [np.full((n, n), 2500, dtype="int16"),
             np.full((n, n), 20, dtype="int16"),
             np.full((n, n), 180, dtype="int16"),
             fbfm,
             np.full((n, n), 45, dtype="int16"),
             np.full((n, n), 150, dtype="int16"),
             np.full((n, n), 15, dtype="int16"),
             np.full((n, n), 10, dtype="int16")]
    names = ["LF2020_Elev_CONUS", "LF2020_SlpD_CONUS", "LF2020_Asp_CONUS",
             "LF2016_FBFM40_CONUS", "LF2016_CC_CONUS", "LF2016_CH_CONUS",
             "LF2016_CBH_CONUS", "LF2016_CBD_CONUS"]
    lcp = _write_lcp(tmp_path / "badfuel.tif", bands, names, n=n)
    with pytest.raises(ValueError, match="Scott & Burgan"):
        destack_lcp(lcp, tmp_path / "out", verbose=False)


def test_out_of_range_conversion_raises(tmp_path):
    """A CC band already stored as a fraction would convert to 0.0045 — but a
    CC band stored as per-mille converts to 8.5, far outside [0, 1]."""
    n = 20
    bands = [np.full((n, n), 2500, dtype="int16"),
             np.full((n, n), 20, dtype="int16"),
             np.full((n, n), 180, dtype="int16"),
             np.full((n, n), 165, dtype="int16"),
             np.full((n, n), 850, dtype="int16"),     # per-mille, not percent
             np.full((n, n), 150, dtype="int16"),
             np.full((n, n), 15, dtype="int16"),
             np.full((n, n), 10, dtype="int16")]
    names = ["LF2020_Elev_CONUS", "LF2020_SlpD_CONUS", "LF2020_Asp_CONUS",
             "LF2016_FBFM40_CONUS", "LF2016_CC_CONUS", "LF2016_CH_CONUS",
             "LF2016_CBH_CONUS", "LF2016_CBD_CONUS"]
    lcp = _write_lcp(tmp_path / "permille.tif", bands, names, n=n)
    with pytest.raises(ValueError, match="outside the\n?\\s*physical range|outside the physical range"):
        destack_lcp(lcp, tmp_path / "out", verbose=False)


def test_unknown_layer_name_raises(synth_lcp, tmp_path):
    with pytest.raises(ValueError, match="Unknown pyretechnics layer"):
        destack_lcp(synth_lcp, tmp_path / "out", layers=["slope", "wind"], verbose=False)


def test_missing_lcp_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        destack_lcp(tmp_path / "nope.tif", tmp_path / "out", verbose=False)


# ---------------------------------------------------------------------------
# name resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("desc,expected", [
    ("LF2020_Elev_CONUS", "ELEV"),
    ("LF2020_SlpD_CONUS", "SLP"),
    ("LF2020_SlpP_CONUS", "SLP"),
    ("LF2020_Asp_CONUS", "ASP"),
    ("LF2016_FBFM40_CONUS", "FBFM40"),
    ("LF2016_CC_CONUS", "CC"),
    ("LF2016_CBD_CONUS", "CBD"),
    ("LF2016_EVT_CONUS", "EVT"),
    ("CC", "CC"),
    ("", None),
    (None, None),
    ("LF2016_Nonsense_CONUS", None),
])
def test_band_name_normalization(desc, expected):
    assert _canonical_band_name(desc) == expected


@pytest.mark.parametrize("desc,expected", [
    ("LF2020_SlpD_CONUS", "degrees"),
    ("LF2020_SlpP_CONUS", "percent"),
])
def test_slope_unit_resolution(desc, expected):
    assert _resolve_slope_units(desc, "auto") == expected


def test_bad_slope_units_argument_raises():
    with pytest.raises(ValueError, match="must be 'auto'"):
        _resolve_slope_units("LF2020_SlpD_CONUS", "radians")


# ---------------------------------------------------------------------------
# reuse / overwrite
# ---------------------------------------------------------------------------

def test_existing_layers_are_reused_unless_overwrite(synth_lcp, tmp_path):
    out = tmp_path / "reuse"
    first = destack_lcp(synth_lcp, out, layers=["slope"], verbose=False)
    stamp = first["slope"].stat().st_mtime_ns

    again = destack_lcp(synth_lcp, out, layers=["slope"], verbose=False)
    assert again["slope"].stat().st_mtime_ns == stamp

    forced = destack_lcp(synth_lcp, out, layers=["slope"], overwrite=True, verbose=False)
    arr, _ = _read(forced["slope"])
    assert np.allclose(arr, np.tan(np.radians(20.0)), atol=1e-6)


def test_cell2fire_stubs_are_kept():
    """FlamMap/FSim/MTT/Cell2Fire paths survive the port — nothing is deleted."""
    from fb_tools.spread.convert import build_cell2fire_weather, lcp_to_cell2fire

    with pytest.raises(NotImplementedError):
        lcp_to_cell2fire()
    with pytest.raises(NotImplementedError):
        build_cell2fire_weather()
