"""
Tests for mask_lcp — restrict an LCP to an analysis area.

Every band is masked outside the analysis area: FlamMap is pixel-based, so it
returns NoData for masked cells and does not need continuous terrain.  The
contract worth pinning is that the fill is explicit and the dtype only changes
when the caller asks for NaN.  Uses the synthetic LCP from conftest, so nothing
here touches ``data/``.
"""

import numpy as np
import pytest
import rasterio
import rioxarray as rxr
import xarray as xr

from fb_tools import mask_lcp


BAND = {"ELEV": 1, "SLP": 2, "ASP": 3, "FBFM40": 4,
        "CC": 5, "CH": 6, "CBH": 7, "CBD": 8, "EVT": 9}


@pytest.fixture
def lcp(synth_lcp):
    return rxr.open_rasterio(synth_lcp, masked=False)


@pytest.fixture
def keep(lcp):
    """Keep the western half of the landscape."""
    m = xr.zeros_like(lcp.isel(band=0), dtype=bool)
    m[:, :100] = True
    return m


@pytest.mark.parametrize("band", sorted(BAND))
def test_every_band_is_masked_outside(lcp, keep, band):
    out = mask_lcp(lcp, keep)
    assert (out.sel(band=BAND[band]).values[:, 100:] == -9999).all()


@pytest.mark.parametrize("band", sorted(BAND))
def test_every_band_is_untouched_inside(lcp, keep, band):
    out = mask_lcp(lcp, keep)
    idx = BAND[band]
    assert (out.sel(band=idx).values[:, :100] == lcp.sel(band=idx).values[:, :100]).all()


def test_integer_dtype_is_preserved(lcp, keep):
    out = mask_lcp(lcp, keep)
    assert out.dtype == lcp.dtype == np.int16


def test_default_fill_comes_from_the_raster_nodata(lcp, keep):
    assert lcp.rio.nodata == -9999
    out = mask_lcp(lcp, keep)
    assert out.rio.nodata == -9999
    assert (out.values[:, :, 100:] == -9999).all()


def test_explicit_integer_nodata_is_used(lcp, keep):
    out = mask_lcp(lcp, keep, nodata=-32768)
    assert out.dtype == np.int16
    assert (out.values[:, :, 100:] == -32768).all()
    assert out.rio.nodata == -32768


def test_nan_nodata_upcasts_to_float(lcp, keep):
    out = mask_lcp(lcp, keep, nodata=np.nan)
    assert out.dtype == np.float32
    assert np.isnan(out.values[:, :, 100:]).all()
    # kept half still carries the original values
    assert (out.values[:, :, :100] == lcp.values[:, :, :100]).all()


def test_grid_shape_and_georeferencing_are_unchanged(lcp, keep):
    out = mask_lcp(lcp, keep)
    assert out.shape == lcp.shape
    assert out.dims == lcp.dims
    assert out.rio.transform() == lcp.rio.transform()
    assert out.rio.crs == lcp.rio.crs


def test_band_names_survive(lcp, keep):
    out = mask_lcp(lcp, keep)
    assert out.attrs["long_name"] == tuple(lcp.attrs["long_name"])


def test_accepts_an_integer_mask(lcp, keep):
    a = mask_lcp(lcp, keep)
    b = mask_lcp(lcp, keep.astype("int16"))
    assert (a.values == b.values).all()


def test_accepts_a_mask_carrying_a_band_dim(lcp, keep):
    out = mask_lcp(lcp, keep.expand_dims("band"))
    assert out.shape == lcp.shape
    assert (out.values[:, :, 100:] == -9999).all()


def test_accepts_a_path(synth_lcp, keep):
    out = mask_lcp(synth_lcp, keep)
    assert out.dtype == np.int16
    assert (out.values[:, :, 100:] == -9999).all()


def test_all_true_mask_is_a_no_op(lcp):
    everything = xr.ones_like(lcp.isel(band=0), dtype=bool)
    out = mask_lcp(lcp, everything)
    assert (out.values == lcp.values).all()


def test_all_false_mask_blanks_the_grid(lcp):
    nothing = xr.zeros_like(lcp.isel(band=0), dtype=bool)
    out = mask_lcp(lcp, nothing)
    assert (out.values == -9999).all()


def test_missing_band_metadata_raises(lcp, keep):
    bare = lcp.copy()
    bare.attrs.pop("long_name", None)
    with pytest.raises(ValueError, match="long_name"):
        mask_lcp(bare, keep)


def test_written_raster_round_trips(lcp, keep, tmp_path):
    out_fp = tmp_path / "masked.tif"
    mask_lcp(lcp, keep, out_path=out_fp)

    with rasterio.open(out_fp) as src:
        assert src.dtypes[0] == "int16"
        assert src.count == lcp.sizes["band"]
        assert src.nodata == -9999
        # band names survive, so get_band_by_longname still works downstream
        assert src.descriptions == tuple(lcp.attrs["long_name"])
        for name, idx in BAND.items():
            band = src.read(idx)
            assert (band[:, 100:] == -9999).all(), f"{name} not masked on disk"
            assert (band[:, :100] == lcp.sel(band=idx).values[:, :100]).all()


def test_written_raster_reopens_as_nan_when_masked(lcp, keep, tmp_path):
    """NoData is declared on the file, so masked=True gives NaN downstream."""
    out_fp = tmp_path / "masked.tif"
    mask_lcp(lcp, keep, out_path=out_fp)
    reopened = rxr.open_rasterio(out_fp, masked=True)
    assert np.isnan(reopened.values[:, :, 100:]).all()
    assert not np.isnan(reopened.values[:, :, :100]).any()


def test_out_path_creates_parent_directory(lcp, keep, tmp_path):
    out_fp = tmp_path / "nested" / "deeper" / "masked.tif"
    mask_lcp(lcp, keep, out_path=out_fp)
    assert out_fp.exists()


def test_input_is_not_mutated(lcp, keep):
    before = lcp.values.copy()
    mask_lcp(lcp, keep)
    assert (lcp.values == before).all()


def test_output_is_lzw_compressed_and_tiled_by_default(lcp, keep, tmp_path):
    out_fp = tmp_path / "masked.tif"
    mask_lcp(lcp, keep, out_path=out_fp)
    with rasterio.open(out_fp) as src:
        assert src.profile["compress"] == "lzw"
        assert src.profile["tiled"] is True


def test_compression_can_be_changed(lcp, keep, tmp_path):
    out_fp = tmp_path / "masked.tif"
    mask_lcp(lcp, keep, out_path=out_fp, compress="deflate")
    with rasterio.open(out_fp) as src:
        assert src.profile["compress"] == "deflate"


def test_compression_can_be_disabled(lcp, keep, tmp_path):
    out_fp = tmp_path / "masked.tif"
    mask_lcp(lcp, keep, out_path=out_fp, compress=None)
    with rasterio.open(out_fp) as src:
        assert src.profile.get("compress") is None


def test_compression_shrinks_a_mostly_masked_grid(lcp, keep, tmp_path):
    """The point of the default: masked cells are one constant, so LZW wins."""
    raw = tmp_path / "raw.tif"
    lzw = tmp_path / "lzw.tif"
    mask_lcp(lcp, keep, out_path=raw, compress=None)
    mask_lcp(lcp, keep, out_path=lzw)
    assert lzw.stat().st_size < raw.stat().st_size


def test_compression_does_not_alter_pixel_values(lcp, keep, tmp_path):
    raw = tmp_path / "raw.tif"
    lzw = tmp_path / "lzw.tif"
    mask_lcp(lcp, keep, out_path=raw, compress=None)
    mask_lcp(lcp, keep, out_path=lzw)
    with rasterio.open(raw) as a, rasterio.open(lzw) as b:
        assert (a.read() == b.read()).all()
        assert a.descriptions == b.descriptions
        assert a.nodata == b.nodata
