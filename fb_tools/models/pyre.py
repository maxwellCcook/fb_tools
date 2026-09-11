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
    if version is not None and installed != version:
        raise RuntimeError(
            f"pyretechnics {installed!r} is installed but this analysis is pinned to "
            f"{version!r}. Crown-fire physics and the spotting API have both changed "
            "across releases, so results are not comparable across versions. Either "
            "install the pinned version or pass version=None deliberately."
        )
    return pyretechnics


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
        ``pyretechnics_version``, ``pyretechnics_pinned``, ``python_version``,
        ``numpy_version``, ``scipy_version``, ``platform``, ``conda_env``.

    Notes
    -----
    numpy and scipy are recorded because pyretechnics is compiled against the
    numpy C-API present at build time: the same source pinned to the same
    version can produce different binaries in different envs.
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
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy_version,
        "platform": f"{platform.system()}-{platform.machine()}",
        "conda_env": sys.prefix.rsplit("/", 1)[-1],
    }
