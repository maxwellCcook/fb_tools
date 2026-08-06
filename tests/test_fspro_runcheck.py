"""
Tests for the FSPro run-success check.

``TestFSPro.exe`` is Windows-only, so :func:`run_fspro` itself cannot execute
here.  :func:`_assert_fspro_succeeded` carries all the logic and is
platform-independent, so it is tested directly.

The defect being pinned: ``subprocess.run`` was called without ``check=True``
and the return code was never inspected, while ``run_fspro_batch`` recorded
``status="success"`` for anything that did not raise.  A crashed run was
indistinguishable from a good one — across a ~5-day, 30-run campaign that means
silent gaps discovered only at analysis time.
"""

from types import SimpleNamespace

import pytest

from fb_tools.models.fspro import _assert_fspro_succeeded


def _base(tmp_path, name="fspro_out"):
    return tmp_path / name


def test_clean_exit_with_output_passes(tmp_path):
    base = _base(tmp_path)
    (tmp_path / "fspro_out_DailyAcres.txt").write_text("day,acres\n1,10\n")

    assert _assert_fspro_succeeded(
        SimpleNamespace(returncode=0), base, tmp_path / "run.log"
    )


@pytest.mark.parametrize("rc", [1, 2, -1, 255])
def test_non_zero_return_code_raises(tmp_path, rc):
    base = _base(tmp_path)
    (tmp_path / "fspro_out_DailyAcres.txt").write_text("day,acres\n1,10\n")

    with pytest.raises(RuntimeError, match=f"return code {rc}"):
        _assert_fspro_succeeded(
            SimpleNamespace(returncode=rc), base, tmp_path / "run.log"
        )


def test_clean_exit_without_output_raises(tmp_path):
    """
    A zero exit is not on its own proof of success.

    FSPro can terminate early and leave the output directory empty; without
    this check that run would be reported as a success and silently drop an
    arm from the contrast.
    """
    base = _base(tmp_path)

    with pytest.raises(RuntimeError, match="wrote no files"):
        _assert_fspro_succeeded(
            SimpleNamespace(returncode=0), base, tmp_path / "run.log"
        )


def test_output_from_a_different_basename_does_not_count(tmp_path):
    """Another arm's output in the same directory must not mask a failure."""
    base = _base(tmp_path, "coswap")
    (tmp_path / "background_DailyAcres.txt").write_text("day,acres\n1,10\n")

    with pytest.raises(RuntimeError, match="wrote no files"):
        _assert_fspro_succeeded(
            SimpleNamespace(returncode=0), base, tmp_path / "run.log"
        )


def test_error_message_points_at_the_log(tmp_path):
    base = _base(tmp_path)
    log = tmp_path / "TestFSPro_run.log"

    with pytest.raises(RuntimeError, match=str(log)):
        _assert_fspro_succeeded(SimpleNamespace(returncode=3), base, log)


def test_missing_returncode_attribute_is_tolerated(tmp_path):
    """Some callers pass a Popen that has not been polled; absence != failure."""
    base = _base(tmp_path)
    (tmp_path / "fspro_out_DailyAcres.txt").write_text("day,acres\n1,10\n")

    assert _assert_fspro_succeeded(
        SimpleNamespace(returncode=None), base, tmp_path / "run.log"
    )
