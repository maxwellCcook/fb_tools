"""
Build pyretechnics 2026.8.10 with the crown canopy-cover gate as a **runtime
parameter** instead of a hard-coded 0.4, and install it over the stock build.

Why
---
``g1b``/``g1c`` established that the entire 21%-vs-51% crown-fire disagreement
with FlamMap is one constant, that removing it aligns crown extent to 99.5%
pixel agreement, and that the constant is plausibly a crude stand-in for
pyretechnics' missing passive-crown regime — so neither setting is simply
correct. The decision (2026-09-10) is to **carry the gate as a sweep axis**
beside firebrand flux, reporting the treatment effect at both ends rather than
picking one. Running the low end in production needs a compiled build: the
pure-Python source is ~100x too slow for a campaign.

The gate occurs TWICE, in two functions on two different code paths, and they
are easy to confuse:

  crown_fire.py:83  ``canopy_cover > 0.4``           in van_wagner_crown_fire_initiation
                    -> used by burn_cells (what G1 measured)
  crown_fire.py:54  ``if canopy_cover <= 0.4: inf``  in van_wagner_crowning_spread_rate
                    -> used by spread_fire_with_phi_field, **the campaign path**

Patching only the first — the one the static diagnostic surfaces — would change
nothing about the actual spread simulation. Both are replaced here, and the patch
**asserts its target text** so a version bump fails loudly instead of silently
no-opping.

Safety properties this script enforces
--------------------------------------
1. **The default is byte-compatible with stock.** The threshold initializes to
   0.4, so an unmodified call sequence must reproduce the stock engine exactly.
   ``verify_default_matches_stock()`` is the gate on that, and it runs the real
   landscape, not a fixture.
2. **The build is identifiable.** The version becomes ``2026.8.10+cfri1`` (PEP 440
   local label), so ``pyre_version_info()`` stamps every campaign manifest with
   the fact that a patched engine produced it. A result from this build must never
   be reported as coming from stock pyretechnics.
3. **The threshold is stamped too.** Reading it back at run time is how a manifest
   records which end of the sweep a realization came from.

Run:
    conda activate fb_pyre
    python code/dev/pyre/build_patched_engine.py            # build + install
    python code/dev/pyre/build_patched_engine.py --verify   # verify only
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from fb_tools.models.pyre import PYRETECHNICS_VERSION
from pyre_source import SRC_ROOT, ensure_pyre_source

#: PEP 440 local version label. Bump when the patch itself changes.
PATCH_LABEL = "cfri1"
PATCHED_VERSION = f"{PYRETECHNICS_VERSION}+{PATCH_LABEL}"

BUILD_DIR = SRC_ROOT / f"patched-{PATCHED_VERSION}"


# ---------------------------------------------------------------------------
# the patch
# ---------------------------------------------------------------------------

#: Inserted after crown_fire.py's import block. A module-level typed global plus
#: two ccall accessors: typed so the per-cell read in the spread loop stays a C
#: double compare rather than a Python object compare.
_THRESHOLD_BLOCK = '''
# ---------------------------------------------------------------------------
# CFRI PATCH (%(label)s): crown canopy-cover gate as a runtime parameter
# ---------------------------------------------------------------------------
# Stock pyretechnics hard-codes 0.4 in two places (van_wagner_crowning_spread_rate
# and van_wagner_crown_fire_initiation). Van Wagner (1977) has no canopy-cover
# term, and on open Colorado montane forest the constant blocks crown fire on 71%%
# of the landscape -- where the gate allows crowning, pyretechnics and FlamMap
# agree to within 0.6 pp, so this one number is the whole difference. It is
# plausibly a stand-in for the missing passive-crown regime, so it is exposed as a
# parameter to be swept, NOT silently lowered.
#
# Default 0.4 == stock behaviour exactly.
_cfri_crown_cover_threshold = cy.declare(cy.double, 0.4)


@cy.ccall
def set_crown_cover_threshold(value: cy.double):
    """
    Set the canopy-cover fraction above which crown fire can occur.

    0.4 is stock pyretechnics. 0.0 disables the gate, which reproduces Van Wagner
    (1977) as published (intensity criterion only) and matches FlamMap's crown
    extent on North Cottonwood to 0.5 pp -- but see the module header: with no
    passive-crown regime, the engine then classes torching-grade fire as full
    active crown fire, so flame length overshoots as much as it undershoots at 0.4.

    Process-global and not thread-safe by design: it is set once per condition
    level, before a worker pool is spawned, and stamped into the manifest.
    """
    global _cfri_crown_cover_threshold
    if not (0.0 <= value <= 1.0):
        raise ValueError("crown cover threshold must be a fraction in [0, 1]")
    _cfri_crown_cover_threshold = value


@cy.ccall
def get_crown_cover_threshold() -> cy.double:
    """The canopy-cover fraction above which crown fire can occur."""
    return _cfri_crown_cover_threshold
# --------------------------- end CFRI PATCH --------------------------------
''' % {"label": PATCH_LABEL}

#: (description, exact text to find, replacement). Order matters only for clarity.
_PATCHES = [
    (
        "insert the threshold global + accessors after the import block",
        "# crown-fire-imports ends here\n",
        "# crown-fire-imports ends here\n" + _THRESHOLD_BLOCK,
    ),
    (
        "van_wagner_crowning_spread_rate -- THE CAMPAIGN PATH "
        "(spread_fire_with_phi_field)",
        "    if canopy_cover <= 0.4:\n        return inf\n",
        "    if canopy_cover <= _cfri_crown_cover_threshold:\n        return inf\n",
    ),
    (
        "van_wagner_crown_fire_initiation -- the burn_cells / static path",
        "        canopy_cover > 0.4\n",
        "        canopy_cover > _cfri_crown_cover_threshold\n",
    ),
]


def apply_patch(build_dir, verbose=True):
    """Copy the pinned sdist into *build_dir* and patch it, asserting every target."""
    src = ensure_pyre_source(verbose=verbose)
    sdist_root = src.parent

    if build_dir.exists():
        shutil.rmtree(build_dir)
    shutil.copytree(sdist_root, build_dir)

    # Stale generated C from a previous cythonize would be used in preference to
    # re-cythonizing the patched .py, silently producing an unpatched extension.
    removed = 0
    for c in build_dir.rglob("*.c"):
        c.unlink()
        removed += 1
    if verbose:
        print(f"removed {removed} pre-generated .c file(s) so the patch is cythonized")

    target = build_dir / "src" / "pyretechnics" / "crown_fire.py"
    text = target.read_text()
    for desc, find, repl in _PATCHES:
        n = text.count(find)
        if n != 1:
            raise RuntimeError(
                f"patch target not found exactly once ({n} matches): {desc}\n"
                f"  looking for: {find!r}\n"
                f"pyretechnics {PYRETECHNICS_VERSION} source has changed shape. Do NOT "
                "loosen this check — an unpatched build that reports itself as patched "
                "is the worst outcome available here."
            )
        text = text.replace(find, repl)
        if verbose:
            print(f"patched: {desc}")
    target.write_text(text)

    # Identify the build so pyre_version_info() stamps it into every manifest.
    pp = build_dir / "pyproject.toml"
    ppt = pp.read_text()
    stock = f'version = "{PYRETECHNICS_VERSION}"'
    if ppt.count(stock) != 1:
        raise RuntimeError(f"could not find {stock!r} exactly once in {pp}")
    pp.write_text(ppt.replace(stock, f'version = "{PATCHED_VERSION}"'))
    if verbose:
        print(f"version -> {PATCHED_VERSION}")

    # Leave the patched source readable beside the build for review.
    return build_dir


def build_and_install(build_dir, verbose=True):
    cmd = [sys.executable, "-m", "pip", "install", str(build_dir),
           "--no-deps", "--no-build-isolation", "--force-reinstall"]
    if verbose:
        print("\n" + " ".join(cmd))
    subprocess.run(cmd, check=True)


def verify_installed(verbose=True):
    """Confirm the installed engine is the patched one and the setter works."""
    from importlib.metadata import version as dist_version

    got = dist_version("pyretechnics")
    if got != PATCHED_VERSION:
        raise RuntimeError(
            f"installed pyretechnics is {got!r}, expected {PATCHED_VERSION!r}"
        )

    import pyretechnics.crown_fire as cf

    if not hasattr(cf, "set_crown_cover_threshold"):
        raise RuntimeError(
            "the installed extension has no set_crown_cover_threshold — the patch "
            "did not make it into the compiled module"
        )
    assert cf.get_crown_cover_threshold() == 0.4, "default must be stock 0.4"

    # The gate must actually move. A canopy cover of 0.25 with ample intensity is
    # blocked at 0.4 and allowed at 0.2.
    args = (5000.0, 0.25, 0.3, 1.0)   # intensity kW/m, CC, CBH m, foliar
    assert cf.van_wagner_crown_fire_initiation(*args) is False
    cf.set_crown_cover_threshold(0.2)
    assert cf.get_crown_cover_threshold() == 0.2
    assert cf.van_wagner_crown_fire_initiation(*args) is True, (
        "threshold changed but the gate did not — the compiled code inlined the "
        "constant instead of reading the global"
    )
    cf.set_crown_cover_threshold(0.4)
    assert cf.van_wagner_crown_fire_initiation(*args) is False

    try:
        cf.set_crown_cover_threshold(1.5)
    except ValueError:
        pass
    else:
        raise RuntimeError("out-of-range threshold was accepted")

    if verbose:
        print(f"\nverified: pyretechnics {got}, threshold settable, default 0.4, "
              "gate responds, range checked")
    return got


#: Measured on the STOCK build, all three arms, before patching
#: (code/dev/pyre/, 2026-09-10; raw values in
#: $FB_PYRE_SCRATCH/stock_engine_baseline.json). The patched build at threshold
#: 0.4 must reproduce these to the last digit.
STOCK_BASELINE = {
    "untreated":  {"crown_frac": 0.213076, "fl_median": 1.800107,
                   "fl_sum": 6483587.2285, "fi_sum": 36316809166.73, "n_active": 304487},
    "background": {"crown_frac": 0.210515, "fl_median": 1.792601,
                   "fl_sum": 6427336.8169, "fi_sum": 35968587044.33, "n_active": 300827},
    "coswap":     {"crown_frac": 0.210360, "fl_median": 1.792601,
                   "fl_sum": 6424923.1632, "fi_sum": 35948941779.97, "n_active": 300606},
}


def verify_default_matches_stock(verbose=True):
    """
    At the default threshold the patched build must reproduce STOCK exactly.

    This is the property that makes the patch safe to install over the stock
    engine: the parameter is genuinely inert at 0.4. Checked on the real
    landscape, all three arms, against numbers measured before patching -- not on
    a fixture, and not on a tolerance loose enough to hide a change.

    `n_active` is an exact integer count and `fi_sum` a sum over 3.0 M cells, so
    between them any per-cell difference anywhere on the landscape shows up.
    """
    import numpy as np
    import pyretechnics.burn_cells as bc
    import pyretechnics.crown_fire as cf

    from g1_flammap_diagnostic import LAYER_DIR, LCP_DIR, build_cubes
    from g1b_crown_divergence import NB
    from fb_tools.spread.convert import destack_lcp

    cf.set_crown_cover_threshold(0.4)
    if verbose:
        print("\nverifying the default threshold is inert (patched == stock):")

    failures = []
    for arm, expected in STOCK_BASELINE.items():
        layers = destack_lcp(LCP_DIR / f"{arm}.tif", LAYER_DIR / arm, verbose=False)
        cubes, arrays, _, _, _ = build_cubes(layers)
        r = bc.burn_all_cells_as_head_fire(cubes, t=0)

        ft = r["fire_type"].astype("int32")
        fl = r["flame_length"].astype("float64")
        fi = r["fireline_intensity"].astype("float64")
        burnable = ~np.isin(arrays["fuel_model"].astype("int32"), NB)

        got = {
            "crown_frac": float((ft[burnable] >= 2).mean()),
            "fl_median": float(np.median(fl[burnable & (fl > 0)])),
            "fl_sum": float(fl[burnable].sum()),
            "fi_sum": float(fi[burnable].sum()),
            "n_active": int((ft == 3).sum()),
        }
        bad = [k for k, v in expected.items()
               if (got[k] != v if k == "n_active"
                   else abs(got[k] - v) > 1e-6 * max(abs(v), 1.0))]
        status = "OK" if not bad else f"MISMATCH on {bad}"
        if verbose:
            print(f"  {arm:11s} crown {100*got['crown_frac']:7.4f}%  "
                  f"FL median {got['fl_median']:.6f}  I sum {got['fi_sum']:.2f}  "
                  f"active {got['n_active']:,}   {status}")
        if bad:
            failures.append((arm, bad, got, expected))

    if failures:
        raise RuntimeError(
            "the patched build does NOT reproduce stock at threshold 0.4:\n"
            + "\n".join(f"  {a}: {b}\n    got      {g}\n    expected {e}"
                         for a, b, g, e in failures)
            + "\nThe parameter is not inert at its default, so the patch changed "
              "physics it should not have. Do not use this build."
        )
    if verbose:
        print("  -> the threshold parameter is inert at 0.4; the patch is safe to install")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="verify the already-installed build, do not rebuild")
    args = ap.parse_args()

    if not args.verify:
        print(f"building pyretechnics {PATCHED_VERSION} in {BUILD_DIR}\n")
        apply_patch(BUILD_DIR)
        build_and_install(BUILD_DIR)

    verify_installed()
    verify_default_matches_stock()
    print(f"\nPatched source kept at {BUILD_DIR / 'src' / 'pyretechnics' / 'crown_fire.py'} "
          "for review.")
    print("Reinstall stock with:\n"
          f'  pip install "pyretechnics=={PYRETECHNICS_VERSION}" --no-deps '
          "--no-build-isolation --force-reinstall")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
