"""
Phase 2 — simulation domain, N-arm API, and the ignition_mode default.

Covers P2.0 (pre-defined domain, grid congruence, display-only clipping),
P2.0b (`lcps` / `contrasts` replacing the hard-wired two-arm pair), and
P2.1 (`ignition_mode` no longer defaults to the container-as-ignition trap).
"""

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from fb_tools.models.container import (
    _assert_grid_congruence,
    _check_domain,
    _resolve_arms,
    _resolve_single_ignition,
)


def _write_lcp(path, *, width=200, height=200, res=30.0,
               left=-800_000.0, top=1_960_000.0, crs="EPSG:5070"):
    """Minimal single-band raster standing in for a landscape."""
    profile = dict(
        driver="GTiff", dtype="int16", count=1, height=height, width=width,
        crs=crs, transform=from_origin(left, top, res, res), nodata=-9999,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.full((height, width), 165, dtype="int16"), 1)
    return path


# ── P2.0b — N-arm resolution ───────────────────────────────────────────────────

def test_arms_from_lcps_mapping(synth_lcp):
    arms, contrasts = _resolve_arms(
        {"untreated": synth_lcp, "background": synth_lcp, "coswap": synth_lcp},
        None, None, None,
    )
    assert list(arms) == ["untreated", "background", "coswap"]


def test_default_contrasts_are_every_ordered_pair(synth_lcp):
    _, contrasts = _resolve_arms(
        {"untreated": synth_lcp, "background": synth_lcp, "coswap": synth_lcp},
        None, None, None,
    )
    assert contrasts == [
        ("untreated", "background"),
        ("untreated", "coswap"),
        ("background", "coswap"),
    ]


def test_contrast_additivity_holds_for_the_defaults(synth_lcp):
    """
    untreated-coswap should decompose as
    (untreated-background) + (background-coswap) — the Phase 3 additivity
    sanity check only works if all three contrasts are actually declared.
    """
    _, contrasts = _resolve_arms(
        {"untreated": synth_lcp, "background": synth_lcp, "coswap": synth_lcp},
        None, None, None,
    )
    assert ("untreated", "background") in contrasts
    assert ("background", "coswap") in contrasts
    assert ("untreated", "coswap") in contrasts


def test_explicit_contrasts_are_kept(synth_lcp):
    _, contrasts = _resolve_arms(
        {"untreated": synth_lcp, "background": synth_lcp, "coswap": synth_lcp},
        None, None, [("background", "coswap")],
    )
    assert contrasts == [("background", "coswap")]


def test_two_arm_shim_still_works(synth_lcp):
    arms, contrasts = _resolve_arms(None, synth_lcp, synth_lcp, None)
    assert list(arms) == ["baseline", "treated"]
    assert contrasts == [("baseline", "treated")]


def test_mixing_both_forms_raises(synth_lcp):
    with pytest.raises(ValueError, match="not both"):
        _resolve_arms({"a": synth_lcp, "b": synth_lcp}, synth_lcp, None, None)


def test_no_landscapes_raises():
    with pytest.raises(ValueError, match="No landscapes supplied"):
        _resolve_arms(None, None, None, None)


def test_single_arm_raises(synth_lcp):
    with pytest.raises(ValueError, match="at least two arms"):
        _resolve_arms({"only": synth_lcp}, None, None, None)


def test_half_a_shim_raises(synth_lcp):
    with pytest.raises(ValueError, match="both baseline_lcp_path and"):
        _resolve_arms(None, synth_lcp, None, None)


def test_unknown_contrast_arm_raises(synth_lcp):
    with pytest.raises(ValueError, match="not in lcps"):
        _resolve_arms({"a": synth_lcp, "b": synth_lcp}, None, None,
                      [("a", "nope")])


def test_missing_lcp_raises(tmp_path, synth_lcp):
    with pytest.raises(FileNotFoundError, match="arm 'ghost'"):
        _resolve_arms({"real": synth_lcp, "ghost": tmp_path / "nope.tif"},
                      None, None, None)


# ── P2.0 — grid congruence across arms ─────────────────────────────────────────

def test_congruent_arms_pass(tmp_path):
    a = _write_lcp(tmp_path / "a.tif")
    b = _write_lcp(tmp_path / "b.tif")
    grid = _assert_grid_congruence({"untreated": a, "coswap": b})
    assert grid["width"] == 200 and grid["height"] == 200
    assert grid["crs"].to_epsg() == 5070


def test_shape_mismatch_is_rejected(tmp_path):
    a = _write_lcp(tmp_path / "a.tif", width=200)
    b = _write_lcp(tmp_path / "b.tif", width=180)
    with pytest.raises(ValueError, match="not on a congruent grid"):
        _assert_grid_congruence({"untreated": a, "coswap": b})


def test_origin_shift_is_rejected(tmp_path):
    """
    Same shape, same CRS, shifted origin: xr.align(join="left") would happily
    subtract these and return a plausible-looking, wrong Δ surface.
    """
    a = _write_lcp(tmp_path / "a.tif", left=-800_000.0)
    b = _write_lcp(tmp_path / "b.tif", left=-799_970.0)
    with pytest.raises(ValueError, match="transform"):
        _assert_grid_congruence({"untreated": a, "coswap": b})


def test_resolution_mismatch_is_rejected(tmp_path):
    a = _write_lcp(tmp_path / "a.tif", res=30.0)
    b = _write_lcp(tmp_path / "b.tif", res=90.0)
    with pytest.raises(ValueError, match="not on a congruent grid"):
        _assert_grid_congruence({"untreated": a, "coswap": b})


def test_crs_mismatch_is_rejected(tmp_path):
    a = _write_lcp(tmp_path / "a.tif", crs="EPSG:5070")
    b = _write_lcp(tmp_path / "b.tif", crs="EPSG:32613")
    with pytest.raises(ValueError, match="CRS"):
        _assert_grid_congruence({"untreated": a, "coswap": b})


def test_congruence_error_names_the_offending_arm(tmp_path):
    a = _write_lcp(tmp_path / "a.tif", width=200)
    b = _write_lcp(tmp_path / "b.tif", width=180)
    with pytest.raises(ValueError, match="coswap"):
        _assert_grid_congruence({"untreated": a, "coswap": b})


# ── P2.0 — domain is validated, never used to clip ─────────────────────────────

def test_domain_none_gives_empty_provenance(tmp_path):
    grid = _assert_grid_congruence({"a": _write_lcp(tmp_path / "a.tif")})
    assert _check_domain(None, grid) == {}


def test_domain_inside_the_landscape_is_covered(tmp_path):
    a = _write_lcp(tmp_path / "a.tif")
    grid = _assert_grid_congruence({"a": a})
    with rasterio.open(a) as src:
        b = src.bounds
        crs = src.crs
    dom = gpd.GeoDataFrame(
        geometry=[box(b.left + 500, b.bottom + 500, b.right - 500, b.top - 500)],
        crs=crs,
    )
    info = _check_domain(dom, grid)
    assert info["covered_by_lcp"] is True
    assert info["n_features"] == 1
    assert info["area_ha"] > 0


def test_domain_larger_than_the_landscape_is_flagged(tmp_path, capsys):
    a = _write_lcp(tmp_path / "a.tif")
    grid = _assert_grid_congruence({"a": a})
    with rasterio.open(a) as src:
        b = src.bounds
        crs = src.crs
    dom = gpd.GeoDataFrame(
        geometry=[box(b.left - 3000, b.bottom - 3000, b.right + 3000, b.top + 3000)],
        crs=crs,
    )
    info = _check_domain(dom, grid)
    assert info["covered_by_lcp"] is False
    assert "outside the landscape extent" in capsys.readouterr().out


def test_domain_is_reprojected_not_rejected(tmp_path):
    """A domain in a different CRS is handled, not refused."""
    a = _write_lcp(tmp_path / "a.tif")
    grid = _assert_grid_congruence({"a": a})
    with rasterio.open(a) as src:
        b = src.bounds
        crs = src.crs
    dom = gpd.GeoDataFrame(
        geometry=[box(b.left + 500, b.bottom + 500, b.right - 500, b.top - 500)],
        crs=crs,
    ).to_crs("EPSG:4326")
    info = _check_domain(dom, grid)
    assert info["covered_by_lcp"] is True


# ── P2.1 — ignition_mode has no default ────────────────────────────────────────

def _container(lcp):
    with rasterio.open(lcp) as src:
        return gpd.GeoDataFrame(geometry=[box(*src.bounds)], crs=src.crs)


def test_omitting_ignition_mode_raises(synth_lcp, tmp_path):
    """
    Defect #1: the old default dissolved the analysis unit and handed it to
    FSPro as a starting fire perimeter, burning the whole container at BP ~ 1.
    """
    with pytest.raises(ValueError, match="requires an explicit ignition_mode"):
        _resolve_single_ignition(
            None, _container(synth_lcp), synth_lcp, tmp_path,
            1, None, None, caller="prepare_container_fspro",
        )


def test_ignition_mode_error_explains_the_trap(synth_lcp, tmp_path):
    with pytest.raises(ValueError, match="starting fire perimeter"):
        _resolve_single_ignition(
            None, _container(synth_lcp), synth_lcp, tmp_path,
            1, None, None, caller="prepare_container_fspro",
        )


def test_unknown_ignition_mode_raises(synth_lcp, tmp_path):
    with pytest.raises(ValueError, match="Unknown ignition_mode"):
        _resolve_single_ignition(
            "wedge", _container(synth_lcp), synth_lcp, tmp_path,
            1, None, None, caller="prepare_container_fspro",
        )


def test_container_mode_still_available(synth_lcp, tmp_path):
    """Kept for reproducing an observed fire from its real perimeter."""
    p = _resolve_single_ignition(
        "container", _container(synth_lcp), synth_lcp, tmp_path,
        1, None, None, caller="prepare_container_fspro",
    )
    assert p.exists()
    assert len(gpd.read_file(p)) == 1


def test_fod_mode_without_points_raises(synth_lcp, tmp_path):
    with pytest.raises(ValueError, match="requires fod_gdf"):
        _resolve_single_ignition(
            "fod", _container(synth_lcp), synth_lcp, tmp_path,
            1, None, None, caller="prepare_container_fspro",
        )


def test_single_random_ignition_is_accepted(synth_lcp, tmp_path):
    p = _resolve_single_ignition(
        "random", _container(synth_lcp), synth_lcp, tmp_path,
        1, None, 5, caller="prepare_container_fspro",
    )
    assert len(gpd.read_file(p)) == 1


def test_multiple_ignitions_are_refused_by_the_single_run_path(synth_lcp,
                                                               tmp_path):
    """
    N ignitions in one IgnitionFile would be N simultaneous starts in every
    simulated fire, not N design fires — so the single-run entry points must
    refuse rather than quietly produce it.
    """
    with pytest.raises(ValueError, match="prepare_fspro_experiment"):
        _resolve_single_ignition(
            "random", _container(synth_lcp), synth_lcp, tmp_path,
            8, None, 5, caller="prepare_container_fspro",
        )


def test_multiple_fod_ignitions_are_refused(synth_lcp, synth_fod, tmp_path):
    with pytest.raises(ValueError, match="prepare_fspro_experiment"):
        _resolve_single_ignition(
            "fod", _container(synth_lcp), synth_lcp, tmp_path,
            1, synth_fod, None, caller="prepare_container_fspro",
        )


# ── P2.0b — the deprecated alias ───────────────────────────────────────────────

def test_deprecated_alias_rejects_the_removed_ray_test_kwargs():
    from fb_tools.models.container import prepare_counterfactual_ignition_set

    for gone in ("sector_deg", "require_treatment_intersect"):
        with pytest.raises(TypeError, match="spread cone"):
            prepare_counterfactual_ignition_set(**{gone: 45.0})
