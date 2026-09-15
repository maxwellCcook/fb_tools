"""raster_to_layers: a statewide raster lands on the run grid cell for cell, nodata outside it. Engine-free."""
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from fb_tools.spread import raster_to_layers

RES = 30.0


def _write(fp, arr, x0, y0, nodata=None, dtype="float32"):
    with rasterio.open(fp, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=dtype,
                       crs="EPSG:5070", transform=from_origin(x0, y0, RES, RES), nodata=nodata) as dst:
        dst.write(arr.astype(dtype), 1)


@pytest.fixture
def layers(tmp_path):
    d = tmp_path / "arm"; d.mkdir()
    _write(d / "slope.tif", np.zeros((10, 8)), x0=1000.0, y0=5000.0)
    return d


def test_window_is_cell_aligned_and_nodata_outside(layers, tmp_path):
    big = np.arange(20 * 20, dtype="float32").reshape(20, 20)
    _write(tmp_path / "state.tif", big, x0=1000.0 - 5 * RES, y0=5000.0 + 3 * RES, nodata=-9999)
    out = raster_to_layers(tmp_path / "state.tif", layers, "sdi")
    with rasterio.open(out) as src:
        a = src.read(1); assert src.nodata == -9999 and a.dtype == np.int16 and a.shape == (10, 8)
    assert np.array_equal(a, big[3:13, 5:13].astype("int16"))
    small = np.full((4, 4), 7.0, dtype="float32")
    _write(tmp_path / "small.tif", small, x0=1000.0 + 2 * RES, y0=5000.0 - 2 * RES)
    out = raster_to_layers(tmp_path / "small.tif", layers, "sdi2")
    with rasterio.open(out) as src:
        a = src.read(1)
    assert (a[2:6, 2:6] == 7).all() and (a == -9999).sum() == 80 - 16


def test_off_lattice_or_wrong_resolution_is_rejected(layers, tmp_path):
    _write(tmp_path / "shift.tif", np.zeros((20, 20)), x0=1000.0 + 7.0, y0=5000.0)
    with pytest.raises(ValueError, match="lattice"):
        raster_to_layers(tmp_path / "shift.tif", layers, "x")
    with rasterio.open(tmp_path / "coarse.tif", "w", driver="GTiff", height=5, width=5, count=1, dtype="float32",
                       crs="EPSG:5070", transform=from_origin(1000.0, 5000.0, 60.0, 60.0)) as dst:
        dst.write(np.zeros((1, 5, 5), "float32"))
    with pytest.raises(ValueError, match=" m"):
        raster_to_layers(tmp_path / "coarse.tif", layers, "y")
