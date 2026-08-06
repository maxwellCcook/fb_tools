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
