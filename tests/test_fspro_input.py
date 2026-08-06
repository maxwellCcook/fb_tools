"""
P0.4 — FSPro input file round-trip and specification conformance.

Golden reference is the vendor's ``416inputsfile.input``.  Byte equality with
it is deliberately *not* the invariant: the vendor writes minimal float repr
(``1.8``, ``0.1``) where :func:`~fb_tools.models.fspro.build_fspro_inputs`
writes fixed decimals (``1.80``, ``0.10``).  Both parse identically, so the
tests assert

* **semantic round-trip** — parse vendor, rebuild through the writer, parse
  again, and require every scalar and array to match numerically;
* **writer idempotence** — build, parse, build again, and require the two
  *written* files to be byte-identical.

The remaining tests pin each spec constraint by mutating the golden file in a
single targeted way and asserting the validator reports it.
"""

import json

import numpy as np
import pytest

from fb_tools.models.fspro import build_fspro_inputs
from fb_tools.models.fspro_validate import (
    assert_valid_fspro_input,
    parse_fspro_input,
    validate_fspro_input,
)

from conftest import PYROME_ERC_DIR

# Array keys every complete input file must round-trip.
ARRAY_KEYS = (
    "dir_breaks", "speed_breaks", "wind_cells", "erc_classes",
    "historic_erc", "avg_erc", "std_erc", "current_erc", "forecast",
)

# Scalar switches carried through the writer.
SCALAR_KEYS = (
    "Dimension", "Resolution", "Duration", "NumFires", "MaxLag", "PolyDegree",
    "ThreadsPerFire", "UseCustomFuels", "SPOTTING_SEED", "CROWN_FIRE_METHOD",
    "CalmValue", "NumWindDirs", "NumWindSpeeds", "NumERCClasses", "NumERCYears",
    "NumWxPerYear", "NumWxCurrYear", "NumForecast", "BarrierFill",
    "SavePerimeters", "IgnitionFile",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def errors(findings: list[str]) -> list[str]:
    """Keep only ``ERROR:`` findings."""
    return [f for f in findings if f.startswith("ERROR")]


def rebuild(parsed: dict, out_path):
    """Write *parsed* back out through :func:`build_fspro_inputs`."""
    forecast = parsed.get("forecast")
    forecast_rows = (
        [tuple(row) for row in forecast]
        if forecast is not None and forecast.size
        else None
    )
    return build_fspro_inputs(
        output_path=out_path,
        wind_cells=parsed["wind_cells"],
        calm_value=parsed["CalmValue"],
        erc_historic=parsed["historic_erc"],
        erc_avg=parsed["avg_erc"],
        erc_std=parsed["std_erc"],
        erc_classes=parsed["erc_classes"],
        current_erc=parsed["current_erc"],
        ignition_file=parsed["IgnitionFile"],
        speed_breaks=list(parsed["speed_breaks"]),
        dir_breaks=list(parsed["dir_breaks"]),
        forecast=forecast_rows,
        **{k: parsed[k] for k in (
            "Dimension", "Resolution", "Duration", "NumFires", "MaxLag",
            "PolyDegree", "ThreadsPerFire", "UseCustomFuels", "SPOTTING_SEED",
            "CROWN_FIRE_METHOD", "BarrierFill", "SavePerimeters",
        ) if k in parsed},
    )


def mutate(text: str, old: str, new: str) -> str:
    """Replace *old* with *new*, asserting the edit actually landed."""
    assert old in text, f"fixture text no longer contains {old!r}"
    return text.replace(old, new, 1)


def write_variant(tmp_path, vendor_input, old: str, new: str):
    """Write a copy of the golden file with a single targeted mutation."""
    text = mutate(vendor_input.read_text(), old, new)
    path = tmp_path / "variant.input"
    path.write_text(text)
    return path


# ── Parsing the golden file ───────────────────────────────────────────────────

def test_vendor_parses_without_errors(vendor_input):
    parsed = parse_fspro_input(vendor_input)
    assert parsed["_parse_errors"] == []
    assert parsed["header"] == "FSPRO-Inputs-File-Version-4"


def test_vendor_block_shapes(vendor_input):
    p = parse_fspro_input(vendor_input)
    assert p["dir_breaks"].shape == (p["NumWindDirs"],)
    assert p["speed_breaks"].shape == (p["NumWindSpeeds"],)
    assert p["wind_cells"].shape == (p["NumWindSpeeds"], p["NumWindDirs"])
    assert p["erc_classes"].shape == (p["NumERCClasses"], 10)
    assert p["historic_erc"].shape == (p["NumERCYears"], p["NumWxPerYear"])
    assert p["avg_erc"].shape == (p["NumWxPerYear"],)
    assert p["std_erc"].shape == (p["NumWxPerYear"],)
    assert p["current_erc"].shape == (p["NumWxCurrYear"],)
    assert p["forecast"].shape == (p["NumForecast"], 3)


def test_vendor_scalars(vendor_input):
    p = parse_fspro_input(vendor_input)
    assert p["Duration"] == 7
    assert p["NumFires"] == 100
    assert p["MaxLag"] == 30
    assert p["CROWN_FIRE_METHOD"] == "Finney"
    assert p["Resolution"] == pytest.approx(90.0)
    # A Windows drive-letter colon must not be mistaken for the switch separator.
    assert p["IgnitionFile"] == r".\416ign.shp"


def test_vendor_has_no_validation_errors(vendor_input):
    """The vendor file is legal FSPro; only WARN-level findings are acceptable."""
    assert errors(validate_fspro_input(vendor_input)) == []


def test_burn_period_is_column_eight(vendor_input):
    """
    Column 8 is the daily burn period in minutes, not spot distance.

    Spec p.4 names the field ``Duration``; the ``_DayTypes.txt`` output echoes
    the same values under ``burnPeriod``.  The 360/300/240/180/120 ladder is
    6 h down to 2 h, shortening as ERC falls.
    """
    p = parse_fspro_input(vendor_input)
    burn_period = p["erc_classes"][:, 7]
    np.testing.assert_array_equal(burn_period, [360, 300, 240, 180, 120])
    assert np.all(np.diff(burn_period) < 0)


# ── Round-trip ────────────────────────────────────────────────────────────────

def test_semantic_round_trip(vendor_input, tmp_path):
    """Parse -> build -> parse must preserve every scalar and array."""
    original = parse_fspro_input(vendor_input)
    rebuilt_path = rebuild(original, tmp_path / "roundtrip.input")
    rebuilt = parse_fspro_input(rebuilt_path)

    assert rebuilt["_parse_errors"] == []

    for key in SCALAR_KEYS:
        assert key in rebuilt, f"{key} lost in round-trip"
        if isinstance(original[key], float):
            assert rebuilt[key] == pytest.approx(original[key]), key
        else:
            assert rebuilt[key] == original[key], key

    for key in ARRAY_KEYS:
        np.testing.assert_allclose(
            rebuilt[key], original[key], rtol=0, atol=1e-6,
            err_msg=f"{key} changed across the round-trip",
        )


def test_writer_is_idempotent(vendor_input, tmp_path):
    """build -> parse -> build must produce byte-identical files."""
    first = rebuild(parse_fspro_input(vendor_input), tmp_path / "first.input")
    second = rebuild(parse_fspro_input(first), tmp_path / "second.input")
    assert first.read_bytes() == second.read_bytes()


def test_rebuilt_file_still_validates(vendor_input, tmp_path):
    rebuilt = rebuild(parse_fspro_input(vendor_input), tmp_path / "rebuilt.input")
    assert errors(validate_fspro_input(rebuilt)) == []


# ── Spec constraints ──────────────────────────────────────────────────────────

def test_num_forecast_without_rows_is_an_error(vendor_input, tmp_path):
    """
    Defect #8 — ``NumForecast`` set but no rows written.

    FSPro then reads ``BarrierFill`` / ``SavePerimeters`` / ``IgnitionFile``
    as forecast records.
    """
    text = vendor_input.read_text()
    text = text.replace("71 10 250\n76 14 270\n78 9 270\n", "")
    path = tmp_path / "no_forecast_rows.input"
    path.write_text(text)

    found = errors(validate_fspro_input(path))
    assert any("NumForecast" in f for f in found), found


def test_num_forecast_above_duration_is_an_error(vendor_input, tmp_path):
    """Spec p.5: NumForecast must be in [0, Duration - 1]."""
    path = write_variant(vendor_input=vendor_input, tmp_path=tmp_path,
                         old="Duration: 7", new="Duration: 2")
    found = errors(validate_fspro_input(path))
    assert any("Duration-1" in f for f in found), found


def test_num_wx_curr_year_below_max_lag_is_an_error(vendor_input, tmp_path):
    """Spec p.5: NumWxCurrYear >= MaxLag."""
    path = write_variant(vendor_input=vendor_input, tmp_path=tmp_path,
                         old="MaxLag: 30", new="MaxLag: 120")
    found = errors(validate_fspro_input(path))
    assert any("MaxLag" in f for f in found), found


def test_num_wx_curr_year_above_window_is_an_error(vendor_input, tmp_path):
    """Spec p.5: NumWxCurrYear < NumWxPerYear - Duration."""
    path = write_variant(vendor_input=vendor_input, tmp_path=tmp_path,
                         old="NumWxCurrYear: 79", new="NumWxCurrYear: 210")
    found = errors(validate_fspro_input(path))
    assert any("NumWxPerYear - Duration" in f for f in found), found


@pytest.mark.parametrize("method", ["Scott/Reinhardt", "scottrheinhardt", "Rothermel"])
def test_invalid_crown_fire_method(vendor_input, tmp_path, method):
    """
    Spec p.2 admits only ``Finney`` and ``ScottRheinhardt``.

    ``Scott/Reinhardt`` is the spelling several fb_tools docstrings recommend
    and FSPro does not accept it.
    """
    path = write_variant(vendor_input=vendor_input, tmp_path=tmp_path,
                         old="CROWN_FIRE_METHOD: Finney",
                         new=f"CROWN_FIRE_METHOD: {method}")
    found = errors(validate_fspro_input(path))
    assert any("CROWN_FIRE_METHOD" in f for f in found), found


@pytest.mark.parametrize("degree", [3, 16])
def test_poly_degree_out_of_range(vendor_input, tmp_path, degree):
    """Spec p.2: PolyDegree in 4-15."""
    path = write_variant(vendor_input=vendor_input, tmp_path=tmp_path,
                         old="PolyDegree: 9", new=f"PolyDegree: {degree}")
    found = errors(validate_fspro_input(path))
    assert any("PolyDegree" in f for f in found), found


def test_wind_matrix_shape_mismatch(vendor_input, tmp_path):
    """
    Declared bin counts must match the matrix.

    This is what happens when custom breaks are chosen but not forwarded to
    the writer: the header says one thing and the table another.
    """
    path = write_variant(vendor_input=vendor_input, tmp_path=tmp_path,
                         old="NumWindDirs: 8\n45 90 135 180 225 270 315 360",
                         new="NumWindDirs: 4\n90 180 270 360")
    found = errors(validate_fspro_input(path))
    assert any("WindCellValues shape" in f for f in found), found


def test_wind_speed_breaks_must_ascend(vendor_input, tmp_path):
    path = write_variant(vendor_input=vendor_input, tmp_path=tmp_path,
                         old="5 10 15 20 25 30", new="5 10 15 20 30 25")
    found = errors(validate_fspro_input(path))
    assert any("ascending" in f for f in found), found


def test_erc_classes_must_descend(vendor_input, tmp_path):
    """Spec p.4: class rows run in descending order by ERC."""
    p = parse_fspro_input(vendor_input)
    flipped = p["erc_classes"][::-1].copy()
    text = vendor_input.read_text()
    original_block = "\n".join(
        line for line in text.splitlines()
        if line.startswith(("81 91", "70 80", "66 71", "60 65", "55 59"))
    )
    flipped_block = "\n".join(
        " ".join(f"{v:g}" for v in row) for row in flipped
    )
    path = tmp_path / "ascending_classes.input"
    path.write_text(mutate(text, original_block, flipped_block))

    found = errors(validate_fspro_input(path))
    assert any("descending" in f for f in found), found


def test_live_fm_inversion_is_an_error(vendor_input, tmp_path):
    """
    Defect #2 — live fuel moisture must fall as ERC rises.

    Deriving live FM from the bin-median day-of-year gives the extreme-ERC
    class the greenest fuels, because high-ERC days cluster mid-summer while
    low-ERC days pool spring green-up with cured autumn.  Here the top class
    is given herb/woody values from the mildest class.
    """
    path = write_variant(
        vendor_input=vendor_input, tmp_path=tmp_path,
        old="81 91 2.9 3.2 4.3 36.3 60.0 360 0.15 0",
        new="81 91 2.9 3.2 4.3 142.1 193.4 360 0.15 0",
    )
    found = errors(validate_fspro_input(path))
    assert any("fm_herb is not monotonic" in f for f in found), found
    assert any("fm_woody is not monotonic" in f for f in found), found


def test_erc_class_gap_is_an_error(vendor_input, tmp_path):
    """A gap between class bounds leaves ERC values with no matching class."""
    path = write_variant(vendor_input=vendor_input, tmp_path=tmp_path,
                         old="60 65 5.1 5.6 7.5 40.5 76.0 180 0.01 0",
                         new="60 62 5.1 5.6 7.5 40.5 76.0 180 0.01 0")
    found = errors(validate_fspro_input(path))
    assert any("ERC gap" in f for f in found), found


def test_shared_class_edges_are_only_a_warning(vendor_input):
    """
    Quantile-derived bounds touch, and the vendor's own table overlaps by 2.

    Overlap is therefore a WARN — FSPro resolves to the first matching class —
    while a gap is an ERROR.
    """
    findings = validate_fspro_input(vendor_input)
    assert any(f.startswith("WARN") and "overlap" in f for f in findings), findings


def test_burn_period_out_of_range_is_an_error(vendor_input, tmp_path):
    path = write_variant(vendor_input=vendor_input, tmp_path=tmp_path,
                         old="81 91 2.9 3.2 4.3 36.3 60.0 360 0.15 0",
                         new="81 91 2.9 3.2 4.3 36.3 60.0 2000 0.15 0")
    found = errors(validate_fspro_input(path))
    assert any("burn period" in f for f in found), found


def test_forecast_column_order_is_erc_speed_direction(vendor_input, tmp_path):
    """
    Spec p.5: ``ERC WindSpeed WindDirection``.

    Several fb_tools docstrings give the reversed order.  A row written that
    way puts a direction in the speed column and a speed in the direction
    column; the latter is caught by the 0-360 bound only when the swap is
    large, so this test pins the specific failure.
    """
    path = write_variant(vendor_input=vendor_input, tmp_path=tmp_path,
                         old="71 10 250", new="71 250 400")
    found = errors(validate_fspro_input(path))
    assert any("wind direction" in f for f in found), found


def test_empty_ignition_file_is_an_error(vendor_input, tmp_path):
    path = write_variant(vendor_input=vendor_input, tmp_path=tmp_path,
                         old=r"IgnitionFile: .\416ign.shp", new="IgnitionFile: ")
    found = errors(validate_fspro_input(path))
    assert any("IgnitionFile" in f for f in found), found


def test_wrong_header_is_an_error(vendor_input, tmp_path):
    path = write_variant(vendor_input=vendor_input, tmp_path=tmp_path,
                         old="FSPRO-Inputs-File-Version-4",
                         new="FSPRO-Inputs-File-Version-3")
    found = errors(validate_fspro_input(path))
    assert any("header" in f for f in found), found


def test_comment_lines_are_ignored(vendor_input, tmp_path):
    """Spec p.1: '#' in the first column marks a comment."""
    text = vendor_input.read_text()
    path = tmp_path / "commented.input"
    path.write_text(text.replace("Duration: 7", "#Duration: 99\nDuration: 7", 1))
    p = parse_fspro_input(path)
    assert p["Duration"] == 7


def test_assert_raises_on_error(vendor_input, tmp_path):
    path = write_variant(vendor_input=vendor_input, tmp_path=tmp_path,
                         old="PolyDegree: 9", new="PolyDegree: 99")
    with pytest.raises(ValueError, match="PolyDegree"):
        assert_valid_fspro_input(path)


def test_assert_passes_on_vendor_file(vendor_input):
    warnings = assert_valid_fspro_input(vendor_input, warn=False)
    assert all(w.startswith("WARN") for w in warnings)


# ── Cached ERC climatology (defect #2 in production data) ─────────────────────

PYROME_IDS = ["42", "43", "45", "46", "47", "52", "53", "56", "128"]


@pytest.mark.xfail(
    reason="P1.1 — build_erc_classes derives live FM from the bin-median DOY, "
           "so the extreme-ERC class gets the greenest fuels. Expected to pass "
           "once live FM comes from RTMA or calc_live_fm_from_dead.",
    strict=False,
)
@pytest.mark.parametrize("pyrome_id", PYROME_IDS)
def test_cached_erc_classes_have_monotonic_live_fm(pyrome_id):
    """Live herb/woody FM in the cached class tables must fall as ERC rises."""
    cache = PYROME_ERC_DIR / f"pyrome_{pyrome_id}_gridmet.json"
    if not cache.exists():
        pytest.skip(f"no cached ERC climatology for pyrome {pyrome_id}")

    classes = np.asarray(json.loads(cache.read_text())["ERCClasses"], dtype=float)
    for name, col in (("fm_herb", 5), ("fm_woody", 6)):
        diffs = np.diff(classes[:, col])
        assert np.all(diffs >= -1e-9), (
            f"pyrome {pyrome_id}: {name} rises with ERC — "
            f"top class (ERC {classes[0, 0]:.0f}-{classes[0, 1]:.0f}) has "
            f"{classes[0, col]:.1f}, bottom class has {classes[-1, col]:.1f}"
        )
