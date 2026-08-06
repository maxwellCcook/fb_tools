"""
Tests for write-time validation in ``build_fspro_inputs`` (P1.6, defects #7/#8).

FSPro does not validate its input file — it reads whatever is there and
misinterprets anything malformed, usually without complaint. The most damaging
case is ``NumForecast`` set without matching rows: FSPro then consumes
``BarrierFill``, ``SavePerimeters``, and ``IgnitionFile`` as forecast records,
and the run proceeds with no ignition file at all.

So the writer validates what it wrote, and quarantines the file if it fails.
"""

import numpy as np
import pytest

from fb_tools.models.fspro import build_fspro_inputs
from fb_tools.models.fspro_validate import parse_fspro_input

# A minimal, spec-valid parameter set. Fuel moisture rises as ERC falls, class
# bounds are contiguous, and the wind matrix sums to 100.
VALID_ERC_CLASSES = np.array([
    [80, 100, 3.0, 5.0, 7.0, 40.0, 60.0, 360, 0.15, 0],
    [60, 80, 4.0, 6.0, 9.0, 50.0, 70.0, 300, 0.10, 0],
    [40, 60, 5.0, 7.0, 11.0, 60.0, 80.0, 240, 0.05, 0],
    [20, 40, 6.0, 9.0, 14.0, 70.0, 95.0, 180, 0.01, 0],
    [0, 20, 8.0, 12.0, 18.0, 90.0, 120.0, 120, 0.00, 0],
], dtype=float)


@pytest.fixture
def valid_args():
    return dict(
        wind_cells=np.full((6, 8), 100.0 / 48.0),
        calm_value=10.0,
        erc_historic=np.full((15, 214), 50.0),
        erc_avg=np.full(214, 50.0),
        erc_std=np.full(214, 5.0),
        erc_classes=VALID_ERC_CLASSES.copy(),
        current_erc=np.full(79, 50),
        ignition_file="ign.shp",
    )


def test_valid_input_writes_and_validates(tmp_path, valid_args):
    out = build_fspro_inputs(tmp_path / "ok.input", **valid_args)
    assert out.exists()
    assert parse_fspro_input(out)["NumFires"] == 1000


# ── Defect #8: NumForecast desync ─────────────────────────────────────────────

def test_num_forecast_without_rows_is_reset_to_zero(tmp_path, valid_args):
    """
    The corruption case. ``NumForecast=3`` with no forecast rows made FSPro
    read the three trailing fields as forecast records.
    """
    out = build_fspro_inputs(tmp_path / "a.input", NumForecast=3, **valid_args)
    p = parse_fspro_input(out)
    assert p["NumForecast"] == 0
    # The three fields that used to be swallowed are still fields.
    text = out.read_text()
    assert "BarrierFill: 0" in text
    assert "SavePerimeters: 1" in text
    assert "IgnitionFile: ign.shp" in text


def test_forecast_rows_set_the_count(tmp_path, valid_args):
    out = build_fspro_inputs(
        tmp_path / "b.input",
        forecast=[(80, 12, 225), (85, 15, 240)],
        **valid_args,
    )
    assert parse_fspro_input(out)["NumForecast"] == 2


def test_forecast_count_overrides_a_conflicting_kwarg(tmp_path, valid_args):
    out = build_fspro_inputs(
        tmp_path / "c.input",
        forecast=[(80, 12, 225)],
        NumForecast=7,
        **valid_args,
    )
    assert parse_fspro_input(out)["NumForecast"] == 1


def test_empty_forecast_list_means_zero(tmp_path, valid_args):
    out = build_fspro_inputs(tmp_path / "d.input", forecast=[], NumForecast=4, **valid_args)
    assert parse_fspro_input(out)["NumForecast"] == 0


def test_forecast_rows_are_written_in_spec_order(tmp_path, valid_args):
    """Spec p.5: ``ERC WindSpeed WindDirection``."""
    out = build_fspro_inputs(
        tmp_path / "e.input", forecast=[(80, 12, 225)], **valid_args
    )
    lines = out.read_text().splitlines()
    row = lines[lines.index("NumForecast: 1") + 1]
    assert row.split() == ["80", "12", "225"]


# ── Defect #7: unvalidated spec constraints ───────────────────────────────────

def test_invalid_crown_fire_method_rejected(tmp_path, valid_args):
    """The docstrings used to recommend the invalid 'Scott/Reinhardt'."""
    with pytest.raises(ValueError, match="CROWN_FIRE_METHOD"):
        build_fspro_inputs(
            tmp_path / "f.input", CROWN_FIRE_METHOD="Scott/Reinhardt", **valid_args
        )


@pytest.mark.parametrize("method", ["Finney", "ScottRheinhardt"])
def test_valid_crown_fire_methods_accepted(tmp_path, valid_args, method):
    out = build_fspro_inputs(
        tmp_path / f"{method}.input", CROWN_FIRE_METHOD=method, **valid_args
    )
    assert parse_fspro_input(out)["CROWN_FIRE_METHOD"] == method


def test_max_lag_above_current_erc_length_rejected(tmp_path, valid_args):
    """Spec p.5: MaxLag <= NumWxCurrYear."""
    with pytest.raises(ValueError, match="NumWxCurrYear"):
        build_fspro_inputs(tmp_path / "g.input", MaxLag=200, **valid_args)


def test_poly_degree_out_of_range_rejected(tmp_path, valid_args):
    with pytest.raises(ValueError, match="PolyDegree"):
        build_fspro_inputs(tmp_path / "h.input", PolyDegree=99, **valid_args)


def test_non_monotonic_fuel_moisture_rejected(tmp_path, valid_args):
    """
    Defect #2's signature, now caught at write time rather than after a run.
    """
    classes = VALID_ERC_CLASSES.copy()
    classes[0, 5] = 200.0            # extreme class becomes the greenest
    valid_args["erc_classes"] = classes
    with pytest.raises(ValueError, match="fm_herb"):
        build_fspro_inputs(tmp_path / "i.input", **valid_args)


def test_erc_class_gap_rejected(tmp_path, valid_args):
    classes = VALID_ERC_CLASSES.copy()
    classes[1, 1] = 55.0             # leaves 55-60 in no class
    valid_args["erc_classes"] = classes
    with pytest.raises(ValueError, match="gap"):
        build_fspro_inputs(tmp_path / "j.input", **valid_args)


# ── NaN guards ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "field", ["erc_avg", "erc_std", "erc_historic", "current_erc", "erc_classes"]
)
def test_non_finite_arrays_rejected_by_name(tmp_path, valid_args, field):
    """
    ``int(round(nan))`` used to raise a bare ValueError from inside the write
    loop, naming nothing. An all-NaN day-of-season column is the realistic way
    to get here.
    """
    arr = np.asarray(valid_args[field], dtype=float).copy()
    arr.flat[0] = np.nan
    valid_args[field] = arr
    with pytest.raises(ValueError, match=field):
        build_fspro_inputs(tmp_path / "k.input", **valid_args)


def test_non_finite_calm_value_rejected(tmp_path, valid_args):
    valid_args["calm_value"] = float("nan")
    with pytest.raises(ValueError, match="calm_value"):
        build_fspro_inputs(tmp_path / "l.input", **valid_args)


# ── Failure handling ──────────────────────────────────────────────────────────

def test_rejected_file_is_quarantined(tmp_path, valid_args):
    """
    A spec-violating file must not be left where it could be run. FSPro does
    not validate, so an invalid .input left on disk is a loaded gun.
    """
    out = tmp_path / "bad.input"
    with pytest.raises(ValueError):
        build_fspro_inputs(out, MaxLag=200, **valid_args)
    assert not out.exists()
    assert (tmp_path / "bad.input.invalid").exists()


def test_validation_can_be_disabled(tmp_path, valid_args):
    """
    The escape hatch, for deliberately writing a file the validator rejects.
    Note the array/enum guards still apply — only the file-level check is off.
    """
    out = build_fspro_inputs(
        tmp_path / "m.input", MaxLag=200, validate=False, **valid_args
    )
    assert out.exists()
    assert parse_fspro_input(out)["MaxLag"] == 200


def test_treatment_pair_also_validates(tmp_path, valid_args):
    """``build_treatment_pair`` forwards to the same writer."""
    from fb_tools.models.fspro import build_treatment_pair

    args = dict(valid_args)
    ignition = args.pop("ignition_file")
    with pytest.raises(ValueError, match="NumWxCurrYear"):
        build_treatment_pair(
            tmp_path / "pair.input", ignition_file=ignition, MaxLag=200, **args
        )
