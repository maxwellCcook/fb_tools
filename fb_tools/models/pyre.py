"""
pyretechnics engine wrapper — the COSWAP transmission model.

Stage 0 scope: version pinning and provenance only. The realization runner
(``run_pyre_realization`` / ``run_pyre_batch``) lands in Stage 3, once the
de-stacker (Stage 1) and the benchmark (Stage 2) have settled the adapter and
the production resolution.

See ``docs/pyretechnics_transmission_plan.md``.

Why the version is pinned and stamped
-------------------------------------
Fifteen pyretechnics releases in 23 months, with API additions (``spot_ignitions``
and spot-from-spot in 2025.4.8) and retuned crown-fire physics in 2026.8.10. An
unpinned campaign is not reproducible, and a campaign whose outputs do not carry
the engine version cannot be audited after the fact. Every manifest written by
the transmission pipeline therefore carries ``pyre_version_info()``.

Environment
-----------
pyretechnics does **not** install into the ``fb_tools`` env — see
``environment-pyre.yml`` for the measured reason. Import it from the ``fb_pyre``
env. ``import pyretechnics`` is deferred to call time throughout ``fb_tools``,
per the existing ``osmnx``/``herbie`` convention, so ``import fb_tools`` still
succeeds in an env that has never seen the engine.
"""

import platform
import sys

#: The only pyretechnics release this analysis is validated against.
#: Changing it invalidates every campaign manifest stamped with the old value.
PYRETECHNICS_VERSION = "2026.8.10"

#: Local patch label (PEP 440) of the CFRI build, which exposes pyretechnics'
#: hard-coded crown canopy-cover gate as a runtime parameter. Stock pyretechnics
#: has no local label, and both are accepted — but the *full* version string is
#: what gets stamped into a manifest, so a patched result can never be mistaken
#: for a stock one. See code/dev/pyre/build_patched_engine.py.
PYRETECHNICS_PATCH_LABEL = "cfri1"

#: Stock pyretechnics' crown gate: crown fire cannot occur at or below this
#: canopy-cover fraction. Hard-coded in stock; settable in the CFRI build.
STOCK_CROWN_COVER_THRESHOLD = 0.4


def require_pyretechnics(version=PYRETECHNICS_VERSION):
    """
    Import pyretechnics, asserting the pinned version.

    Parameters
    ----------
    version : str or None
        Required version string. ``None`` skips the check (for exploratory work
        only — never for a production run).

    Returns
    -------
    module
        The imported ``pyretechnics`` top-level module.

    Raises
    ------
    ImportError
        If pyretechnics is not installed, with the ``fb_pyre`` remedy.
    RuntimeError
        If the installed version differs from *version*.
    """
    try:
        import pyretechnics  # noqa: F401
    except ImportError as exc:                                   # pragma: no cover
        raise ImportError(
            "pyretechnics is not installed in this environment. It does not build "
            "against the fb_tools env (numpy 2.x / Cython 3.1+); use the fb_pyre env:\n"
            "    conda env create -f environment-pyre.yml\n"
            "    conda activate fb_pyre\n"
            "    pip install -e . --no-deps\n"
            '    pip install "pyretechnics==%s" --no-deps --no-build-isolation'
            % PYRETECHNICS_VERSION
        ) from exc

    installed = _installed_version()
    # A local version label (2026.8.10+cfri1) is the CFRI patched build and is
    # accepted: it is the same upstream release with one constant exposed as a
    # parameter, verified inert at its default against the stock engine on the
    # real landscape. The base version must still match exactly.
    if version is not None and _base_version(installed) != _base_version(version):
        raise RuntimeError(
            f"pyretechnics {installed!r} is installed but this analysis is pinned to "
            f"{version!r}. Crown-fire physics and the spotting API have both changed "
            "across releases, so results are not comparable across versions. Either "
            "install the pinned version or pass version=None deliberately."
        )
    return pyretechnics


def _base_version(v):
    """Strip a PEP 440 local label: ``"2026.8.10+cfri1" -> "2026.8.10"``."""
    return str(v).split("+", 1)[0]


def is_patched_engine():
    """True when the installed pyretechnics is the CFRI patched build."""
    return _installed_version().endswith(f"+{PYRETECHNICS_PATCH_LABEL}")


def crown_cover_threshold():
    """
    The canopy-cover fraction above which the engine allows crown fire.

    Returns
    -------
    float
        The live value on the patched build; ``STOCK_CROWN_COVER_THRESHOLD``
        (0.4, hard-coded and unchangeable) on stock pyretechnics.
    """
    if not is_patched_engine():
        return STOCK_CROWN_COVER_THRESHOLD
    import pyretechnics.crown_fire as cf
    return float(cf.get_crown_cover_threshold())


def set_crown_cover_threshold(value):
    """
    Set the engine's crown canopy-cover gate — one level of the sweep axis.

    Parameters
    ----------
    value : float
        Canopy-cover fraction in [0, 1]. ``0.4`` is stock pyretechnics; ``0.0``
        disables the gate, reproducing Van Wagner (1977) as published.

    Raises
    ------
    RuntimeError
        On stock pyretechnics, where the constant is compiled in. Install the
        CFRI build (``code/dev/pyre/build_patched_engine.py``).

    Notes
    -----
    **Why this is a sweep axis and not a tuned constant.** Stock's 0.4 blocks
    crown fire on 71% of North Cottonwood, where the *intensity* criterion is
    satisfied on 82% of the blocked cells; setting it to 0 brings crown extent to
    within 0.5 pp of FlamMap at 99.5% pixel agreement. But pyretechnics has no
    passive-crown regime at all (``fire_type == 2`` is ~0 everywhere at p97), so
    with the gate off it classes torching-grade fire as full active crown fire and
    flame length overshoots as far as it undershoots at 0.4. Neither end is
    correct; both are reported.

    **Process-global.** Seeds and weather are per realization, this is not. Set it
    once per condition level *before* spawning a worker pool — a "spawn" worker
    re-imports the module and gets the default, so a value set after the pool
    exists will not reach the workers.
    """
    if not is_patched_engine():
        raise RuntimeError(
            f"pyretechnics {_installed_version()} hard-codes the crown canopy-cover "
            "gate at 0.4 in two places (van_wagner_crowning_spread_rate and "
            "van_wagner_crown_fire_initiation). Build the CFRI engine to make it "
            "settable:\n    python code/dev/pyre/build_patched_engine.py"
        )
    import pyretechnics.crown_fire as cf
    cf.set_crown_cover_threshold(float(value))


def _installed_version():
    """Installed pyretechnics version, or ``"unknown"``.

    The compiled package exposes no ``__version__`` attribute (it is a Cython
    extension module), so the distribution metadata is the only source.
    """
    from importlib.metadata import PackageNotFoundError, version as _dist_version

    try:
        return _dist_version("pyretechnics")
    except PackageNotFoundError:                                 # pragma: no cover
        return "unknown"


def pyre_version_info(strict=True):
    """
    Provenance stamp for a campaign manifest.

    Parameters
    ----------
    strict : bool
        Assert the pinned version (default True). Pass False to stamp an
        exploratory run without failing on a mismatch.

    Returns
    -------
    dict
        ``pyretechnics_version``, ``pyretechnics_pinned``, ``pyretechnics_patched``,
        ``crown_cover_threshold``, ``python_version``, ``numpy_version``,
        ``scipy_version``, ``platform``, ``conda_env``.

    Notes
    -----
    numpy and scipy are recorded because pyretechnics is compiled against the
    numpy C-API present at build time: the same source pinned to the same
    version can produce different binaries in different envs.

    ``crown_cover_threshold`` is recorded because it is a swept physics setting
    (see ``set_crown_cover_threshold``), and a results table that does not say
    which end of that sweep produced it cannot be interpreted.
    """
    if strict:
        require_pyretechnics()

    import numpy as np

    try:
        import scipy
        scipy_version = scipy.__version__
    except ImportError:                                          # pragma: no cover
        scipy_version = None

    return {
        "pyretechnics_version": _installed_version(),
        "pyretechnics_pinned": PYRETECHNICS_VERSION,
        "pyretechnics_patched": is_patched_engine(),
        # Recorded on every run, patched or not: it is a physics setting, and a
        # results table that does not say which end of the sweep it came from
        # cannot be interpreted.
        "crown_cover_threshold": crown_cover_threshold(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy_version,
        "platform": f"{platform.system()}-{platform.machine()}",
        "conda_env": sys.prefix.rsplit("/", 1)[-1],
    }
