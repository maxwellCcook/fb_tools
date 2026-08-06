"""Shared fixtures for the fb_tools test suite.

Tests that need real model output or cached weather are skipped rather than
failed when the data is absent, so the suite stays runnable on a clean clone.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Vendor sample distributed with TestFSPro — the golden reference for the
# FSPro input format.
VENDOR_DIR = REPO_ROOT / "code" / "FB" / "TestFSPro" / "SampleData"
VENDOR_INPUT = VENDOR_DIR / "416inputsfile.input"

# A real FSPro run produced by this package (pyrome 47, 100 fires, 7 days).
FSPRO_RUN_DIR = REPO_ROOT / "data" / "fspro_test" / "build_test"
FSPRO_RUN_BASE = "fspro_p47"

# Cached GridMET ERC climatology, one JSON per pyrome.
PYROME_ERC_DIR = REPO_ROOT / "data" / "weather" / "pyrome_erc"

# Weather cache root and the GEE-exported GridMET fire-season climatology.
WEATHER_DIR = REPO_ROOT / "data" / "weather"
GRIDMET_CSV = (
    REPO_ROOT / "data" / "tabular" / "raw" / "weather" / "gridmet_clim_CO_pyromes.csv"
)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def vendor_input() -> Path:
    """Path to the vendor's ``416inputsfile.input`` golden reference."""
    if not VENDOR_INPUT.exists():
        pytest.skip(f"vendor sample not found: {VENDOR_INPUT}")
    return VENDOR_INPUT


@pytest.fixture(scope="session")
def fspro_run_dir() -> Path:
    """Directory holding the on-disk pyrome 47 FSPro run outputs."""
    if not FSPRO_RUN_DIR.is_dir():
        pytest.skip(f"FSPro run outputs not found: {FSPRO_RUN_DIR}")
    return FSPRO_RUN_DIR


@pytest.fixture(scope="session")
def weather_dir() -> Path:
    """Root of the cached weather data (pyrome_erc/, flammap_rtma/, ...)."""
    if not WEATHER_DIR.is_dir():
        pytest.skip(f"weather cache not found: {WEATHER_DIR}")
    return WEATHER_DIR


@pytest.fixture(scope="session")
def synth_lcp(tmp_path_factory) -> Path:
    """
    A small heterogeneous landscape raster, built in-process.

    200 x 200 cells at 30 m (6 x 6 km), EPSG:5070, nine bands in LANDFIRE
    order so band 4 is FBFM40.  Fuels are burnable (TU5 = 165) except a
    non-burnable water stripe (NB8 = 98) and an agriculture block (NB3 = 93),
    which gives the burnable-mask code something real to exclude.  No
    dependency on `data/`, so ignition tests run on a clean clone.
    """
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    path = tmp_path_factory.mktemp("synth") / "synth_lcp.tif"
    n, res = 200, 30.0
    left, top = -800_000.0, 1_960_000.0

    fbfm = np.full((n, n), 165, dtype="int16")
    fbfm[90:100, :] = 98        # water stripe across the middle
    fbfm[:, 160:180] = 93       # agriculture block on the east side

    bands = {
        1: np.full((n, n), 2500, dtype="int16"),   # ELEV
        2: np.full((n, n), 20, dtype="int16"),     # SLPD
        3: np.full((n, n), 180, dtype="int16"),    # ASP
        4: fbfm,                                   # FBFM40
        5: np.full((n, n), 45, dtype="int16"),     # CC
        6: np.full((n, n), 150, dtype="int16"),    # CH
        7: np.full((n, n), 15, dtype="int16"),     # CBH
        8: np.full((n, n), 10, dtype="int16"),     # CBD
        9: np.full((n, n), 9999, dtype="int16"),   # EVT
    }
    names = ["LF2020_Elev_CONUS", "LF2020_SlpD_CONUS", "LF2020_Asp_CONUS",
             "LF2022_FBFM40_CONUS", "LF2022_CC_CONUS", "LF2022_CH_CONUS",
             "LF2022_CBH_CONUS", "LF2022_CBD_CONUS", "LF2022_EVT_CONUS"]

    profile = dict(
        driver="GTiff", dtype="int16", count=9, height=n, width=n,
        crs="EPSG:5070", transform=from_origin(left, top, res, res),
        nodata=-9999,
    )
    with rasterio.open(path, "w", **profile) as dst:
        for i, arr in bands.items():
            dst.write(arr, i)
            dst.set_band_description(i, names[i - 1])
    return path


@pytest.fixture(scope="session")
def synth_fod(synth_lcp):
    """Clustered synthetic ignition points over the west half of `synth_lcp`."""
    import numpy as np
    import geopandas as gpd
    import rasterio

    with rasterio.open(synth_lcp) as src:
        crs, b = src.crs, src.bounds

    rng = np.random.default_rng(11)
    # Three tight clusters, all in the western half
    centres = [(b.left + 1200, b.bottom + 1200),
               (b.left + 1500, b.top - 1500),
               (b.left + 2500, (b.top + b.bottom) / 2)]
    xs, ys = [], []
    for cx, cy in centres:
        xs.extend(rng.normal(cx, 300, 40))
        ys.extend(rng.normal(cy, 300, 40))
    return gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys), crs=crs)


@pytest.fixture(scope="session")
def gridmet_df():
    """
    The GridMET fire-season climatology, loaded once per session.

    Session-scoped because the CSV is ~30k rows and several tests rebuild ERC
    class tables from it.  Consumers must not mutate the frame in place.
    """
    if not GRIDMET_CSV.exists():
        pytest.skip(f"GridMET climatology not found: {GRIDMET_CSV}")
    from fb_tools.weather.gridmet import load_gridmet_csv

    return load_gridmet_csv(GRIDMET_CSV)
