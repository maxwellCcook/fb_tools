"""
Engine identity, version pinning, and the crown-gate sweep parameter.

Split into two halves:

* **No engine required** — version-string handling and the module's constants.
  These run in the ``fb_tools`` env, where pyretechnics is deliberately absent.
* **Engine required** (``pytest.importorskip``) — the live threshold accessors.
  These only run in ``fb_pyre``, and the sweep-parameter tests additionally skip
  on a stock build, since the constant is compiled in there.

The property under test throughout is that **a result can always be traced to the
engine that produced it**. The crown canopy-cover gate is a swept physics setting
worth 5.7x in fireline-intensity p95 on the real landscape, so a manifest that
does not record it cannot be interpreted, and a patched build that reports itself
as stock would be the worst available outcome.
"""

import pytest

from fb_tools.models import pyre


# ---------------------------------------------------------------------------
# no engine required
# ---------------------------------------------------------------------------

def test_pinned_version_is_explicit():
    """An unpinned campaign is not reproducible: 15 releases in 23 months, with
    the spotting API added and crown physics retuned along the way."""
    assert pyre.PYRETECHNICS_VERSION == "2026.8.10"
    assert pyre.PYRETECHNICS_PATCH_LABEL == "cfri1"
    assert pyre.STOCK_CROWN_COVER_THRESHOLD == 0.4


@pytest.mark.parametrize("version,base", [
    ("2026.8.10", "2026.8.10"),
    ("2026.8.10+cfri1", "2026.8.10"),
    ("2026.8.10+cfri2", "2026.8.10"),
    ("2025.6.13", "2025.6.13"),
])
def test_base_version_strips_local_label(version, base):
    assert pyre._base_version(version) == base


def test_require_pyretechnics_error_names_the_remedy():
    """The engine does not build in the fb_tools env (numpy 2.x / Cython 3.1+),
    so the ImportError has to say where it does build or the next person repeats
    the whole Stage 0 investigation."""
    pytest.importorskip  # noqa: B018 — documentational
    try:
        import pyretechnics  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="fb_pyre"):
            pyre.require_pyretechnics()
    else:
        pytest.skip("pyretechnics is installed; nothing to assert about its absence")


# ---------------------------------------------------------------------------
# engine required
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    return pytest.importorskip("pyretechnics", reason="run in the fb_pyre env")


@pytest.fixture
def patched(engine):
    if not pyre.is_patched_engine():
        pytest.skip(
            "stock pyretechnics hard-codes the crown gate; build the CFRI engine "
            "with code/dev/pyre/build_patched_engine.py"
        )
    original = pyre.crown_cover_threshold()
    yield
    pyre.set_crown_cover_threshold(original)


def test_installed_version_matches_the_pin(engine):
    """A local patch label is accepted; a different upstream release is not."""
    pyre.require_pyretechnics()
    assert pyre._base_version(pyre._installed_version()) == pyre.PYRETECHNICS_VERSION


def test_wrong_upstream_version_raises(engine):
    with pytest.raises(RuntimeError, match="pinned"):
        pyre.require_pyretechnics(version="2025.6.13")


def test_version_info_records_the_physics_setting(engine):
    info = pyre.pyre_version_info()
    for key in ("pyretechnics_version", "pyretechnics_pinned", "pyretechnics_patched",
                "crown_cover_threshold", "python_version", "numpy_version",
                "scipy_version", "platform", "conda_env"):
        assert key in info, key
    # numpy is recorded because the extension is compiled against the numpy C-API
    # present at build time: same pinned source, different env, different binary.
    assert info["numpy_version"]
    assert isinstance(info["pyretechnics_patched"], bool)
    assert 0.0 <= info["crown_cover_threshold"] <= 1.0


def test_threshold_readable_on_any_build(engine):
    """Stock reports the compiled-in 0.4; patched reports the live value."""
    assert pyre.crown_cover_threshold() == pytest.approx(
        pyre.STOCK_CROWN_COVER_THRESHOLD
        if not pyre.is_patched_engine()
        else pyre.crown_cover_threshold()
    )


def test_stock_build_refuses_to_pretend_the_gate_is_settable(engine):
    if pyre.is_patched_engine():
        pytest.skip("patched build; the setter is expected to work")
    with pytest.raises(RuntimeError, match="build_patched_engine"):
        pyre.set_crown_cover_threshold(0.0)


def test_threshold_defaults_to_stock_on_the_patched_build(patched):
    """The parameter must be inert at its default, or installing the patched
    engine over stock would silently change every existing result."""
    assert pyre.crown_cover_threshold() == pytest.approx(pyre.STOCK_CROWN_COVER_THRESHOLD)


def test_threshold_round_trips(patched):
    for value in (0.0, 0.2, 0.4, 1.0):
        pyre.set_crown_cover_threshold(value)
        assert pyre.crown_cover_threshold() == pytest.approx(value)


@pytest.mark.parametrize("bad", [-0.1, 1.1, 40.0])
def test_out_of_range_threshold_raises(patched, bad):
    """40.0 is the percent-vs-fraction slip. canopy_cover is a fraction — the
    engine's own load_landfire applies CC * 0.01 -> [0, 0.95]."""
    with pytest.raises(ValueError):
        pyre.set_crown_cover_threshold(bad)


def test_gate_actually_changes_crown_initiation(patched):
    """The compiled code must read the global, not inline the constant.

    CC 0.25 with ample fireline intensity: blocked at 0.4, allowed at 0.2. This is
    the whole landscape's situation — North Cottonwood is 71% below 0.4, and the
    intensity criterion is satisfied on 82% of the cells the gate blocks.
    """
    import pyretechnics.crown_fire as cf

    args = (5000.0, 0.25, 0.3, 1.0)   # intensity kW/m, CC fraction, CBH m, foliar
    pyre.set_crown_cover_threshold(0.4)
    assert cf.van_wagner_crown_fire_initiation(*args) is False
    pyre.set_crown_cover_threshold(0.2)
    assert cf.van_wagner_crown_fire_initiation(*args) is True


def test_gate_reaches_the_spread_model_not_just_burn_cells(patched):
    """The gate is hard-coded TWICE, on two different code paths.

    ``van_wagner_crown_fire_initiation`` serves ``burn_cells``;
    ``van_wagner_crowning_spread_rate`` serves ``spread_fire_with_phi_field``,
    which is what the campaign runs. Patching only the first — the one a static
    diagnostic surfaces — would leave the actual simulation unchanged.

    The second site is a ``cy.cfunc`` and so is unreachable from Python, which is
    exactly why this test drives a real spread run instead of calling it: an
    end-to-end assertion is the only one available, and it is the one that
    matters. On a uniform CC = 0.25 landscape the gate is worth roughly 30x the
    area burned, so the margin here is enormous and the test is not delicate.
    """
    import numpy as np
    import pyretechnics.eulerian_level_set as els
    from pyretechnics.space_time_cube import SpaceTimeCube

    n, res, bands = 60, 30.0, 6
    shape = (bands, n, n)
    f = np.float32
    full = lambda v: np.full((n, n), v, dtype="float32")  # noqa: E731

    # Uniformly canopied TU5 timber at CC 0.25 — below stock's 0.4 gate, above 0.
    cubes = {
        "slope":               SpaceTimeCube(shape, full(0.15)),
        "aspect":              SpaceTimeCube(shape, full(90.0)),
        "fuel_model":          SpaceTimeCube(shape, np.full((n, n), 165, dtype="int64")),
        "canopy_cover":        SpaceTimeCube(shape, full(0.25)),
        "canopy_height":       SpaceTimeCube(shape, full(18.0)),
        "canopy_base_height":  SpaceTimeCube(shape, full(1.0)),
        "canopy_bulk_density": SpaceTimeCube(shape, full(0.10)),
        "wind_speed_10m":      SpaceTimeCube(shape, f(42.8)),    # km/hr, NOT m/s
        "upwind_direction":    SpaceTimeCube(shape, f(270.0)),   # degrees FROM
        "fuel_moisture_dead_1hr":        SpaceTimeCube(shape, f(0.04)),
        "fuel_moisture_dead_10hr":       SpaceTimeCube(shape, f(0.055)),
        "fuel_moisture_dead_100hr":      SpaceTimeCube(shape, f(0.066)),
        "fuel_moisture_live_herbaceous": SpaceTimeCube(shape, f(0.593)),
        "fuel_moisture_live_woody":      SpaceTimeCube(shape, f(0.787)),
        "foliar_moisture":               SpaceTimeCube(shape, f(1.0)),
    }

    def run():
        state = (els.SpreadState(shape)
                 .set_resolution((60.0, res, res))
                 .set_start_time(0.0))
        state.ignite_cell((n // 2, n // 4))
        out = els.spread_fire_with_phi_field(cubes, state, max_duration=3 * 60.0)
        m = out["spread_state"].get_full_matrices()
        ft = np.asarray(m["fire_type"])
        burned = ft > 0
        return int(burned.sum()), int((ft >= 2).sum())

    pyre.set_crown_cover_threshold(0.4)
    n_burned_gated, n_crown_gated = run()

    pyre.set_crown_cover_threshold(0.2)
    n_burned_open, n_crown_open = run()

    assert n_burned_gated > 0, "the gated run did not spread at all; test is vacuous"
    assert n_crown_gated == 0, (
        f"CC 0.25 is below the 0.4 gate but the spread model produced "
        f"{n_crown_gated} crown cells"
    )
    assert n_crown_open > 0, (
        "lowering the threshold produced no crown fire in the spread model — "
        "van_wagner_crowning_spread_rate was NOT patched, so the campaign path "
        "still has the constant compiled in"
    )
    assert n_burned_open > n_burned_gated, (
        f"crown fire appeared ({n_crown_open} cells) but area did not grow "
        f"({n_burned_open} vs {n_burned_gated})"
    )


# ---------------------------------------------------------------------------
# engine required: coarse 3-D wind cubes
# ---------------------------------------------------------------------------

def test_build_cubes_accepts_coarse_3d_wind():
    """A (t, rows/ky, cols/kx) wind array must be repeated by the engine, not rejected or resampled."""
    pytest.importorskip("pyretechnics", reason="run in the fb_pyre env")
    import numpy as np
    rows, cols, n_t = 20, 30, 4
    static = {k: np.full((rows, cols), 0.1, "float32") for k in pyre.STATIC_LAYERS}
    static["fuel_model"] = np.full((rows, cols), 102, "int64")
    weather = {k: np.full(n_t, 0.05, "float32") for k in
               ("temperature", "fuel_moisture_dead_1hr", "fuel_moisture_dead_10hr", "fuel_moisture_dead_100hr",
                "fuel_moisture_live_herbaceous", "fuel_moisture_live_woody")}
    ws = np.arange(n_t * 2 * 3, dtype="float32").reshape(n_t, 2, 3)
    weather["wind_speed_10m"] = ws
    weather["upwind_direction"] = np.full((n_t, 2, 3), 270.0, "float32")
    cubes, shape = pyre.build_cubes(static, weather)
    assert shape == (n_t, rows, cols)
    c = cubes["wind_speed_10m"]
    assert c.get(0, 0, 0) == ws[0, 0, 0] and c.get(3, 19, 29) == ws[3, 1, 2] and c.get(2, 9, 10) == ws[2, 0, 1]
    with pytest.raises(ValueError):
        pyre.build_cubes(static, {**weather, "wind_speed_10m": np.zeros((n_t, 3, 3), "float32")})
