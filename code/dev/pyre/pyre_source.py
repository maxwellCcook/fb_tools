"""
Access to pyretechnics' **pure-Python-mode source**, beside the compiled engine.

pyretechnics is written in Cython's pure-Python mode: every ``src/pyretechnics/*.py``
file is valid Python that runs unmodified when it is imported instead of its
compiled ``.so``. The ``cython`` shadow module turns ``@cy.cfunc`` and friends into
no-op decorators, so the *same source* the wheel was built from executes as plain
Python, ~100x slower and fully introspectable.

That buys two things this analysis needs and the compiled build cannot give:

1. **``cy.cfunc`` functions become callable.** The compiled extension exports only
   ``cy.ccall`` names, so ``spot_fire.expected_firebrand_production`` — the
   function Stage 1b is built on — is unreachable from Python in the installed
   package. In pure-Python mode it is an ordinary function, which is what lets a
   vectorized reimplementation be checked against the reference rather than
   against a docstring.

2. **Constants can be patched for an experiment without shipping a fork.** The
   ``canopy_cover > 0.4`` crown gate is one line; swapping it in pure-Python mode
   measures its effect on the real landscape while leaving the installed engine
   untouched and the campaign reproducible.

**This is a diagnostic tool, not a production path.** Nothing under ``fb_tools/``
imports it. Production runs use the compiled, pinned, version-stamped engine; the
pure-Python tree exists so we can see inside it.
"""

import sys
import tarfile
import urllib.request
from pathlib import Path

from fb_tools.models.pyre import PYRETECHNICS_VERSION

#: Where the extracted sdist lives. Under the scratch root, not the repo: it is
#: ~1 MB of third-party EPL-2.0 source that is re-fetchable on demand.
from g1_flammap_diagnostic import SCRATCH

SRC_ROOT = SCRATCH / "pyretechnics-src"
PYPI_JSON = "https://pypi.org/pypi/pyretechnics/{version}/json"


def ensure_pyre_source(version=PYRETECHNICS_VERSION, verbose=True):
    """
    Download and extract the pinned pyretechnics sdist, returning its ``src`` dir.

    Parameters
    ----------
    version : str
        Must match the installed, pinned version — reading a different release's
        source than the one that produced the results is exactly the kind of
        quiet mismatch this whole port keeps tripping over.
    verbose : bool

    Returns
    -------
    Path
        The ``src/`` directory to prepend to ``sys.path``.
    """
    src = SRC_ROOT / f"pyretechnics-{version}" / "src"
    if (src / "pyretechnics" / "crown_fire.py").exists():
        return src

    import json

    SRC_ROOT.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(PYPI_JSON.format(version=version)) as fh:
        meta = json.load(fh)
    url = next(u["url"] for u in meta["urls"] if u["packagetype"] == "sdist")
    tgz = SRC_ROOT / f"pyretechnics-{version}.tar.gz"
    if verbose:
        print(f"fetching {url} -> {tgz}")
    urllib.request.urlretrieve(url, tgz)
    with tarfile.open(tgz) as tf:
        tf.extractall(SRC_ROOT, filter="data")

    if not (src / "pyretechnics" / "crown_fire.py").exists():
        raise RuntimeError(f"sdist extracted but {src} is not where expected")
    if verbose:
        print(f"pyretechnics {version} source at {src}")
    return src


def import_pure_python(version=PYRETECHNICS_VERSION, verbose=True):
    """
    Swap the process over to the pure-Python pyretechnics modules.

    Purges any already-imported compiled ``pyretechnics.*`` modules and puts the
    source tree first on ``sys.path``. **Irreversible within a process** — do the
    compiled-engine work first, then call this.

    Returns
    -------
    module
        The pure-Python ``pyretechnics`` package.

    Raises
    ------
    RuntimeError
        If the import still resolves to a compiled extension, which would make
        every patched-constant result silently meaningless.
    """
    src = ensure_pyre_source(version, verbose=verbose)
    for name in [k for k in sys.modules if k == "pyretechnics" or k.startswith("pyretechnics.")]:
        del sys.modules[name]
    sys.path.insert(0, str(src))

    import pyretechnics
    if not pyretechnics.__file__.endswith(".py"):
        raise RuntimeError(
            f"pyretechnics still resolved to {pyretechnics.__file__} — the compiled "
            "extension is shadowing the source tree, so nothing patched here would "
            "take effect."
        )
    if verbose:
        print(f"pure-Python pyretechnics: {pyretechnics.__file__}")
    return pyretechnics
