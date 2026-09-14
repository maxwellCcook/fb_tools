"""load_rtma_csv: header-only monthly exports must not poison the dtypes of the concatenated record."""
import warnings

import numpy as np
import pandas as pd
import pytest

from fb_tools.weather.rtma import load_rtma_csv, _REQUIRED_RTMA_COLS

_HEADER = ",".join(_REQUIRED_RTMA_COLS) + "\n"


def _rows(n, start="2013-08-01T00:00:00Z"):
    t = pd.date_range(start, periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"pyrome_id": 45, "datetime_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "tmp_f": np.linspace(60, 90, n),
                         "rh_pct": np.linspace(40, 10, n), "emc_pct": np.linspace(8, 3, n), "vpd_pa": np.linspace(500, 3000, n),
                         "pcp_mm_hr": [np.nan] * n})


def test_header_only_month_is_skipped_and_dtypes_stay_numeric(tmp_path):
    full, empty = tmp_path / "rtma_hourly_x_2013_08.csv", tmp_path / "rtma_hourly_x_2013_09.csv"
    _rows(6).to_csv(full, index=False)
    empty.write_text(_HEADER)                                   # the real 2013-09 export
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)           # pandas 2.2 warned here; pandas 3 silently made every column object
        out = load_rtma_csv([full, empty])
    assert len(out) == 6
    for c in ("tmp_f", "rh_pct", "emc_pct", "vpd_pa", "pcp_mm_hr"):
        assert pd.api.types.is_float_dtype(out[c]), f"{c} is {out[c].dtype}"
    assert pd.api.types.is_integer_dtype(out["pyrome_id"]) and pd.api.types.is_datetime64_any_dtype(out["datetime_utc"])
    assert (out["pcp_mm_hr"] == 0.0).all(), "masked precipitation is filled to zero"
    assert load_rtma_csv(tmp_path).equals(out), "a directory loads the same files"


def test_all_header_only_files_raise(tmp_path):
    (tmp_path / "rtma_hourly_x_2013_09.csv").write_text(_HEADER)
    with pytest.raises(ValueError, match="No RTMA rows"):
        load_rtma_csv(tmp_path)
