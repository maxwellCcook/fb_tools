"""
Tests for the FSPro wind-rose plumbing (P1.5).

Three corrections:

- ``build_wind_cells`` counted NaN winds as calm, because ``ws >= threshold``
  is False for NaN. That reported HRRR archive gaps as meteorological calms and
  inflated ``CalmValue``.
- The wind cache records the bin edges its matrix was built on, but
  ``_load_weather_for_pyrome`` dropped them, so ``build_fspro_inputs`` fell back
  to its defaults — a cache built with custom breaks silently contradicted its
  own frequency table.
- ``WindCellValues`` normalizes to ~100 on its own and ``CalmValue`` is a
  separate field. Several docstrings claimed the table summed to 100 only after
  subtracting the calm share.
"""

import json

import numpy as np
import pytest

from fb_tools.weather.hrrr import build_wind_cells

SPEED_BREAKS = [5, 10, 15, 20, 25, 30]
DIR_BREAKS = [45, 90, 135, 180, 225, 270, 315, 360]


# ── Missing vs calm ───────────────────────────────────────────────────────────

def test_nan_speeds_are_missing_not_calm():
    """
    Ten valid observations, all windy, plus ten NaN speeds. Calm must be 0% —
    the pre-P1.5 code reported 50%.
    """
    ws = np.array([12.0] * 10 + [np.nan] * 10)
    wd = np.array([100.0] * 20)
    cells, calm = build_wind_cells(ws, wd)
    assert calm == pytest.approx(0.0)
    assert cells.sum() == pytest.approx(100.0)


def test_nan_directions_are_missing_not_calm():
    ws = np.array([12.0] * 20)
    wd = np.array([100.0] * 10 + [np.nan] * 10)
    cells, calm = build_wind_cells(ws, wd)
    assert calm == pytest.approx(0.0)
    assert cells.sum() == pytest.approx(100.0)


def test_calm_fraction_excludes_missing_from_the_denominator():
    """
    5 calm, 5 windy, 10 missing → calm is 5/10 of *valid* observations, not
    5/20 of everything.
    """
    ws = np.array([0.5] * 5 + [12.0] * 5 + [np.nan] * 10)
    wd = np.array([100.0] * 20)
    _, calm = build_wind_cells(ws, wd)
    assert calm == pytest.approx(50.0)


def test_genuine_calms_still_counted():
    ws = np.array([0.5] * 4 + [12.0] * 6)
    wd = np.array([100.0] * 10)
    _, calm = build_wind_cells(ws, wd)
    assert calm == pytest.approx(40.0)


def test_all_missing_raises():
    ws = np.full(10, np.nan)
    wd = np.full(10, np.nan)
    with pytest.raises(ValueError, match="No usable observations"):
        build_wind_cells(ws, wd)


def test_all_calm_raises():
    ws = np.full(10, 0.1)
    wd = np.full(10, 100.0)
    with pytest.raises(ValueError, match="No usable observations"):
        build_wind_cells(ws, wd)


# ── Normalization: matrix and CalmValue are independent ───────────────────────

def test_matrix_sums_to_100_independently_of_calm():
    """
    The table normalizes over non-calm observations, so its sum does not move
    when the calm share changes. Vendor 416: matrix 99.74, CalmValue 10.25.
    """
    wd = np.array([100.0] * 20)
    no_calm, calm_a = build_wind_cells(np.full(20, 12.0), wd)
    with_calm, calm_b = build_wind_cells(
        np.array([0.5] * 10 + [12.0] * 10), wd
    )
    assert no_calm.sum() == pytest.approx(100.0)
    assert with_calm.sum() == pytest.approx(100.0)
    assert calm_a == pytest.approx(0.0)
    assert calm_b == pytest.approx(50.0)


def test_vendor_sample_matches_the_separate_calm_convention(vendor_input):
    """The golden file is the authority on this: 99.74 + 10.25 > 100."""
    from fb_tools.models.fspro_validate import parse_fspro_input

    p = parse_fspro_input(vendor_input)
    assert p["wind_cells"].sum() == pytest.approx(100.0, abs=1.0)
    assert p["CalmValue"] > 0
    assert p["wind_cells"].sum() + p["CalmValue"] > 101.0


# ── Binning ───────────────────────────────────────────────────────────────────

def test_table_shape_follows_the_breaks():
    ws = np.full(50, 12.0)
    wd = np.linspace(1, 359, 50)
    cells, _ = build_wind_cells(
        ws, wd, speed_breaks=SPEED_BREAKS, dir_breaks=DIR_BREAKS
    )
    assert cells.shape == (len(SPEED_BREAKS), len(DIR_BREAKS))


def test_custom_breaks_change_the_shape():
    ws = np.full(50, 12.0)
    wd = np.linspace(1, 359, 50)
    cells, _ = build_wind_cells(ws, wd, speed_breaks=[10, 20], dir_breaks=[180, 360])
    assert cells.shape == (2, 2)


def test_north_wind_lands_in_the_last_direction_bin():
    """0 degrees is north and maps to the 360 bin, not the 45 bin."""
    cells, _ = build_wind_cells(
        np.full(10, 12.0), np.zeros(10),
        speed_breaks=SPEED_BREAKS, dir_breaks=DIR_BREAKS,
    )
    assert cells[:, -1].sum() == pytest.approx(100.0)


def test_speeds_above_the_last_break_land_in_the_last_row():
    cells, _ = build_wind_cells(
        np.full(10, 99.0), np.full(10, 100.0),
        speed_breaks=SPEED_BREAKS, dir_breaks=DIR_BREAKS,
    )
    assert cells[-1, :].sum() == pytest.approx(100.0)


# ── Cache plumbing: breaks reach the input file ───────────────────────────────

def test_cached_wind_matrix_matches_its_declared_breaks(weather_dir):
    """
    A cache whose matrix shape disagrees with its own bin edges would produce
    an input file FSPro reads wrongly, whichever side wins.
    """
    caches = sorted((weather_dir / "pyrome_wind").glob("*_wind.json"))
    if not caches:
        pytest.skip("no cached wind climatology")
    for path in caches:
        data = json.loads(path.read_text())
        cells = np.asarray(data["WindCellValues"], dtype=float)
        assert cells.shape == (
            len(data["WindSpeedBreaks_mph"]),
            len(data["WindDirBreaks_deg"]),
        ), f"{path.name}: matrix shape disagrees with its bin edges"
        assert cells.shape == (data["NumWindSpeeds"], data["NumWindDirs"])


def test_load_weather_forwards_the_cached_breaks(weather_dir):
    """
    The regression P1.5 fixes: the breaks were read from the cache and then
    dropped, so ``build_fspro_inputs`` silently used its defaults.
    """
    from fb_tools.models.container import _load_weather_for_pyrome

    caches = sorted((weather_dir / "pyrome_wind").glob("*_wind.json"))
    if not caches:
        pytest.skip("no cached wind climatology")
    pyrome_id = json.loads(caches[0].read_text())["pyrome_id"]
    if not (weather_dir / "pyrome_erc" / f"pyrome_{pyrome_id}_gridmet.json").exists():
        pytest.skip(f"no ERC cache for pyrome {pyrome_id}")

    wx = _load_weather_for_pyrome(
        pyrome_id, weather_dir, ignition_season_day=80, duration=7, max_lag=30
    )
    assert wx["speed_breaks"] is not None
    assert wx["dir_breaks"] is not None
    assert len(wx["speed_breaks"]) == wx["wind_cells"].shape[0]
    assert len(wx["dir_breaks"]) == wx["wind_cells"].shape[1]


def test_written_input_declares_the_cached_breaks(weather_dir, tmp_path):
    """
    End-to-end: non-default breaks must survive into the written file, with
    ``NumWindSpeeds``/``NumWindDirs`` agreeing with the matrix.
    """
    from fb_tools.models.fspro import build_fspro_inputs
    from fb_tools.models.fspro_validate import parse_fspro_input

    speed_breaks = [8, 16, 24, 32]
    dir_breaks = [90, 180, 270, 360]
    wind_cells = np.full((4, 4), 100.0 / 16.0)

    out = build_fspro_inputs(
        tmp_path / "custom.input",
        wind_cells=wind_cells,
        calm_value=7.5,
        erc_historic=np.full((15, 214), 50.0),
        erc_avg=np.full(214, 50.0),
        erc_std=np.full(214, 5.0),
        erc_classes=np.array([
            [80, 100, 3.0, 5.0, 7.0, 40.0, 60.0, 360, 0.15, 0],
            [60, 80, 4.0, 6.0, 9.0, 50.0, 70.0, 300, 0.10, 0],
            [40, 60, 5.0, 7.0, 11.0, 60.0, 80.0, 240, 0.05, 0],
            [20, 40, 6.0, 9.0, 14.0, 70.0, 95.0, 180, 0.01, 0],
            [0, 20, 8.0, 12.0, 18.0, 90.0, 120.0, 120, 0.00, 0],
        ], dtype=float),
        current_erc=np.full(79, 50),
        speed_breaks=speed_breaks,
        dir_breaks=dir_breaks,
        ignition_file="ign.shp",
    )

    p = parse_fspro_input(out)
    assert p["NumWindSpeeds"] == 4
    assert p["NumWindDirs"] == 4
    assert p["wind_cells"].shape == (4, 4)
    text = out.read_text()
    assert " ".join(str(v) for v in speed_breaks) in text
    assert " ".join(str(v) for v in dir_breaks) in text
