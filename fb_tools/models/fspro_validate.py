"""
FSPro input file parser and validator.

Round-trips ``FSPRO-Inputs-File-Version-4`` files and checks them against the
vendor specification (``code/FB/TestFSPro/FSProInputsFileDocumentation.pdf``).

Three entry points:

1. :func:`parse_fspro_input` — read a written ``.input`` file back into a dict
   of scalars and numpy arrays.
2. :func:`validate_fspro_input` — return a list of ``ERROR:`` / ``WARN:``
   findings.
3. :func:`assert_valid_fspro_input` — raise ``ValueError`` on any ``ERROR:``.

The validator exists because several defects can be written to disk silently
and only surface as implausible model output:

- **Live fuel moisture inverted with ERC.**  Deriving live FM from the
  bin-median day-of-year gives the extreme-ERC class the *greenest* fuels,
  because high-ERC days cluster mid-summer while low-ERC days pool spring
  green-up with cured autumn.  :func:`_check_erc_classes` makes that
  unwritable.
- **``NumForecast`` set without rows.**  FSPro then consumes ``BarrierFill`` /
  ``SavePerimeters`` / ``IgnitionFile`` as forecast records.
- **``NumWxCurrYear`` outside the ``MaxLag`` / ``Duration`` window** (spec p.5).
- **Wind matrix shape disagreeing with the declared bin edges**, which happens
  whenever custom breaks are used but not forwarded to the writer.

Column 8 of an ERC class row is the daily **burn period in minutes**.  The
spec calls this field ``Duration`` — distinct from the run-level ``Duration:``
switch — and it is echoed as ``burnPeriod`` in the ``_DayTypes.txt`` output.
It was historically mislabelled ``spot_dist`` in this package.

Notes
-----
Byte-for-byte equality with the vendor's ``416inputsfile.input`` is not
achievable and is not the invariant to test: the vendor file uses minimal
float repr (``1.8``, ``0.1``, ``0.0``) where
:func:`~fb_tools.models.fspro.build_fspro_inputs` writes fixed decimals
(``1.80``, ``0.10``, ``0.00``).  Both parse identically.  Test *semantic*
round-trip plus writer idempotence instead — see ``tests/test_fspro_input.py``.
"""

from pathlib import Path

import numpy as np


# ── Spec constants ────────────────────────────────────────────────────────────

_HEADER = "FSPRO-Inputs-File-Version-4"

# Spec p.2: "The only acceptable values are 'Finney' and 'ScottRheinhardt'".
_VALID_CROWN_FIRE_METHODS = frozenset({"Finney", "ScottRheinhardt"})

# Switch name -> python type for scalar values.
_SCALAR_TYPES: dict[str, type] = {
    "Dimension": int,
    "Resolution": float,
    "Duration": int,
    "NumFires": int,
    "MaxLag": int,
    "PolyDegree": int,
    "ThreadsPerFire": int,
    "UseCustomFuels": int,
    "SPOTTING_SEED": int,
    "CROWN_FIRE_METHOD": str,
    "CalmValue": float,
    "NumWindDirs": int,
    "NumWindSpeeds": int,
    "NumERCClasses": int,
    "NumERCYears": int,
    "NumWxPerYear": int,
    "NumWxCurrYear": int,
    "NumForecast": int,
    "BarrierFill": int,
    "SavePerimeters": int,
    "IgnitionFile": str,
    "BarriersFile": str,
}

# Block header switch -> (output key, row-count resolver, is_2d).
# An int resolver is a literal row count; a str resolver names the scalar
# switch that carries the count.
_BLOCKS: dict[str, tuple[str, "str | int", bool]] = {
    "NumWindDirs":       ("dir_breaks",   1,               False),
    "NumWindSpeeds":     ("speed_breaks", 1,               False),
    "WindCellValues":    ("wind_cells",   "NumWindSpeeds", True),
    "NumERCClasses":     ("erc_classes",  "NumERCClasses", True),
    "HistoricERCValues": ("historic_erc", "NumERCYears",   True),
    "AvgERCValues":      ("avg_erc",      1,               False),
    "StdDevERCValues":   ("std_erc",      1,               False),
    "CurrentERCValues":  ("current_erc",  1,               False),
    "NumForecast":       ("forecast",     "NumForecast",   True),
}

# ERC class row layout (spec p.4):
#   MinERC MaxERC FM1 FM10 FM100 FMHerb FMWoody Duration SpotProbability SpotDelay
_ERC_COLS = (
    "min_erc", "max_erc", "fm1", "fm10", "fm100", "fm_herb", "fm_woody",
    "burn_period_min", "spot_probability", "spot_delay",
)
_N_ERC_COLS = len(_ERC_COLS)

# Column index of each fuel moisture field in an ERC class row.
_FM_COLS = {"fm1": 2, "fm10": 3, "fm100": 4, "fm_herb": 5, "fm_woody": 6}

# Vendor README: "In the range 1000-3000 fires at a minimum."
_MIN_RECOMMENDED_FIRES = 1000


# ── Parsing ───────────────────────────────────────────────────────────────────

def _is_switch_line(line: str) -> bool:
    """True when *line* starts a switch in column 1 (spec p.1: no leading space)."""
    if not line or line[0] in " \t#":
        return False
    key, sep, _ = line.partition(":")
    if not sep or not key:
        return False
    return key.replace("_", "").isalnum() and not key[0].isdigit()


def _coerce(key: str, raw: str):
    """Cast a raw switch value to its spec type, returning the string on failure."""
    typ = _SCALAR_TYPES.get(key, str)
    text = raw.strip()
    if typ is str or not text:
        return text
    try:
        return typ(float(text)) if typ is int else typ(text)
    except ValueError:
        return text


def parse_fspro_input(path: "str | Path") -> dict:
    """
    Parse an FSPro input file into scalars and numpy arrays.

    Switches may appear in any order (spec p.1), so scalars are collected in a
    first pass and the data blocks whose row counts depend on them are read in
    a second.

    Parameters
    ----------
    path : str or Path
        Path to an ``FSPRO-Inputs-File-Version-4`` file.

    Returns
    -------
    dict
        Every scalar switch found, plus these array keys when present:
        ``dir_breaks``, ``speed_breaks``, ``wind_cells`` ``(n_speed, n_dir)``,
        ``erc_classes`` ``(n_classes, 10)``, ``historic_erc``
        ``(n_years, n_wx)``, ``avg_erc``, ``std_erc``, ``current_erc``,
        ``forecast`` ``(n_forecast, 3)``.  Also ``header`` (the version line)
        and ``_parse_errors`` (list of str, empty when the file parsed cleanly).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.

    Examples
    --------
    >>> p = parse_fspro_input("code/FB/TestFSPro/SampleData/416inputsfile.input")
    >>> p["NumFires"], p["erc_classes"].shape
    (100, (5, 10))
    """
    path = Path(path)
    raw_lines = path.read_text().splitlines()

    parsed: dict = {"header": None, "_parse_errors": []}
    errors: list[str] = parsed["_parse_errors"]

    # Blank out comments but keep index alignment for the second pass.
    lines = ["" if ln.startswith("#") else ln for ln in raw_lines]

    for ln in lines:
        if ln.strip():
            parsed["header"] = ln.strip()
            break
    if parsed["header"] != _HEADER:
        errors.append(f"header is {parsed['header']!r}, expected {_HEADER!r}")

    # ── Pass 1: scalars ───────────────────────────────────────────────────────
    for ln in lines:
        if not _is_switch_line(ln):
            continue
        key, _, raw = ln.partition(":")
        parsed[key] = _coerce(key, raw)

    # ── Pass 2: data blocks ───────────────────────────────────────────────────
    for i, ln in enumerate(lines):
        if not _is_switch_line(ln):
            continue
        key = ln.partition(":")[0]
        if key not in _BLOCKS:
            continue
        out_key, n_spec, is_2d = _BLOCKS[key]

        if isinstance(n_spec, int):
            n_rows = n_spec
        else:
            n_rows = parsed.get(n_spec)
            if not isinstance(n_rows, int):
                errors.append(
                    f"{key}: cannot read block — {n_spec} missing or non-integer"
                )
                continue
        if n_rows <= 0:
            parsed[out_key] = np.zeros((0, 0) if is_2d else 0, dtype=float)
            continue

        rows: list[list[float]] = []
        for raw_row in lines[i + 1 : i + 1 + n_rows]:
            if _is_switch_line(raw_row):
                break
            try:
                rows.append([float(tok) for tok in raw_row.split()])
            except ValueError:
                errors.append(f"{key}: non-numeric token in data row {len(rows) + 1}")
                rows = []
                break

        if not rows:
            errors.append(f"{key}: expected {n_rows} data row(s), found none")
            continue
        if len(rows) != n_rows:
            errors.append(f"{key}: expected {n_rows} data row(s), found {len(rows)}")

        widths = {len(r) for r in rows}
        if len(widths) > 1:
            errors.append(f"{key}: ragged block, row widths {sorted(widths)}")
            continue

        arr = np.asarray(rows, dtype=float)
        parsed[out_key] = arr if is_2d else arr.reshape(-1)

    return parsed


# ── Individual checks ─────────────────────────────────────────────────────────

def _check_required(p: dict, out: list[str]) -> None:
    """Presence and domain of the scalar switches."""
    required = (
        "Dimension", "Resolution", "Duration", "NumFires", "MaxLag", "PolyDegree",
        "CalmValue", "NumWindDirs", "NumWindSpeeds", "NumERCClasses",
        "NumERCYears", "NumWxPerYear", "NumWxCurrYear", "NumForecast",
        "SavePerimeters", "IgnitionFile",
    )
    for key in required:
        if key not in p:
            out.append(f"ERROR: required switch {key!r} is missing")

    if p.get("Dimension") not in (None, 2):
        out.append(f"ERROR: Dimension must be 2 (spec p.1), got {p['Dimension']}")

    res = p.get("Resolution")
    if isinstance(res, float) and res <= 0:
        out.append(f"ERROR: Resolution must be > 0, got {res}")

    for key in ("Duration", "NumFires", "MaxLag"):
        val = p.get(key)
        if isinstance(val, int) and val < 1:
            out.append(f"ERROR: {key} must be >= 1, got {val}")

    n_fires = p.get("NumFires")
    if isinstance(n_fires, int) and 1 <= n_fires < _MIN_RECOMMENDED_FIRES:
        out.append(
            f"WARN: NumFires={n_fires} is below the vendor minimum of "
            f"{_MIN_RECOMMENDED_FIRES}-3000; the BP surface will be too coarse "
            "to resolve a treatment signal"
        )

    poly = p.get("PolyDegree")
    if isinstance(poly, int) and not 4 <= poly <= 15:
        out.append(f"ERROR: PolyDegree must be in 4-15 (spec p.2), got {poly}")

    cfm = p.get("CROWN_FIRE_METHOD")
    if cfm is not None and cfm not in _VALID_CROWN_FIRE_METHODS:
        out.append(
            f"ERROR: CROWN_FIRE_METHOD must be one of "
            f"{sorted(_VALID_CROWN_FIRE_METHODS)} (spec p.2), got {cfm!r}"
        )

    for key in ("BarrierFill", "SavePerimeters", "UseCustomFuels"):
        if key in p and p[key] not in (0, 1):
            out.append(f"ERROR: {key} must be 0 or 1, got {p[key]}")

    if not str(p.get("IgnitionFile", "")).strip():
        out.append("ERROR: IgnitionFile is empty")


def _check_wind(p: dict, out: list[str]) -> None:
    """Wind rose block: declared counts, bin ordering, matrix shape and mass."""
    dirs = p.get("dir_breaks")
    speeds = p.get("speed_breaks")
    cells = p.get("wind_cells")
    n_dir, n_speed = p.get("NumWindDirs"), p.get("NumWindSpeeds")

    if dirs is not None and isinstance(n_dir, int) and len(dirs) != n_dir:
        out.append(f"ERROR: NumWindDirs={n_dir} but {len(dirs)} direction breaks")
    if speeds is not None and isinstance(n_speed, int) and len(speeds) != n_speed:
        out.append(f"ERROR: NumWindSpeeds={n_speed} but {len(speeds)} speed breaks")

    if cells is not None and isinstance(n_speed, int) and isinstance(n_dir, int):
        if cells.shape != (n_speed, n_dir):
            out.append(
                f"ERROR: WindCellValues shape {cells.shape} != "
                f"(NumWindSpeeds, NumWindDirs) = ({n_speed}, {n_dir})"
            )

    if dirs is not None and len(dirs):
        if np.any(np.diff(dirs) <= 0):
            out.append("ERROR: wind direction breaks must be strictly ascending")
        if dirs.min() < 0 or dirs.max() > 360:
            out.append(
                f"ERROR: wind direction breaks outside 0-360 (spec p.3) — "
                f"got {dirs.min():g} to {dirs.max():g}"
            )
    if speeds is not None and len(speeds):
        if np.any(np.diff(speeds) <= 0):
            out.append("ERROR: wind speed breaks must be strictly ascending (spec p.3)")
        if speeds.min() < 0:
            out.append("ERROR: negative wind speed break")

    if cells is not None and cells.size:
        if np.any(~np.isfinite(cells)):
            out.append("ERROR: WindCellValues contains NaN/inf")
        elif np.any(cells < 0):
            out.append("ERROR: WindCellValues contains negative frequencies")
        else:
            # Verified against 416inputsfile.input: the matrix sums to 99.74 on
            # its own and CalmValue (10.25) is stored separately, not subtracted.
            total = float(cells.sum())
            if not 95.0 <= total <= 105.0:
                out.append(
                    f"WARN: WindCellValues sums to {total:.2f}, expected ~100 "
                    "(CalmValue is stored separately, not subtracted)"
                )

    calm = p.get("CalmValue")
    if isinstance(calm, float) and not 0.0 <= calm <= 100.0:
        out.append(f"ERROR: CalmValue must be a percentage in 0-100, got {calm}")


def _check_erc_classes(p: dict, out: list[str]) -> None:
    """ERC class table: shape, ordering, coverage, and fuel-moisture physics."""
    cls = p.get("erc_classes")
    n_cls = p.get("NumERCClasses")
    if cls is None:
        return
    if isinstance(n_cls, int) and cls.shape[0] != n_cls:
        out.append(f"ERROR: NumERCClasses={n_cls} but {cls.shape[0]} class rows")
    if cls.ndim != 2 or cls.shape[1] != _N_ERC_COLS:
        out.append(
            f"ERROR: ERC class rows must have {_N_ERC_COLS} columns "
            f"({' '.join(_ERC_COLS)}), got shape {cls.shape}"
        )
        return
    if not cls.size:
        out.append("ERROR: ERC class table is empty")
        return
    if np.any(~np.isfinite(cls)):
        out.append("ERROR: ERC class table contains NaN/inf")
        return

    lo, hi = cls[:, 0], cls[:, 1]

    if np.any(hi < lo):
        out.append("ERROR: ERC class has max_erc < min_erc")
    # Spec p.4: "X lines of ERC class definitions, in descending order by ERC value"
    if len(lo) > 1 and np.any(np.diff(lo) >= 0):
        out.append(
            "ERROR: ERC classes must be in descending ERC order "
            "(highest first, spec p.4)"
        )

    # Coverage is checked on the values as WRITTEN — build_fspro_inputs uses
    # :.0f for the bounds, and that rounded text is what FSPro reads.
    lo_w, hi_w = np.rint(lo).astype(int), np.rint(hi).astype(int)
    for i in range(len(lo_w) - 1):
        gap = lo_w[i] - hi_w[i + 1]
        if gap > 1:
            out.append(
                f"ERROR: ERC gap between classes {i} and {i + 1} — values "
                f"{hi_w[i + 1] + 1}-{lo_w[i] - 1} fall in no class"
            )
        elif gap < 0:
            # Shared or overlapping edges are common: quantile-derived bounds
            # touch, and the vendor's own 416 table overlaps by 2 (70-80 / 66-71).
            out.append(
                f"WARN: ERC classes {i} and {i + 1} overlap "
                f"({lo_w[i]} <= {hi_w[i + 1]}); FSPro resolves to the first match"
            )

    # Fuels must dry as ERC rises.  Rows run highest ERC first, so every FM
    # column must be non-decreasing down the rows.  This is the rule that
    # permanently catches the DOY-median live-FM inversion.
    for name, col in _FM_COLS.items():
        d = np.diff(cls[:, col])
        if np.any(d < -1e-9):
            bad = int(np.argmin(d))
            out.append(
                f"ERROR: {name} is not monotonic with ERC — class {bad} "
                f"(ERC {lo[bad]:.0f}-{hi[bad]:.0f}) has {name}={cls[bad, col]:.1f} "
                f"but the moister class {bad + 1} "
                f"(ERC {lo[bad + 1]:.0f}-{hi[bad + 1]:.0f}) has "
                f"{name}={cls[bad + 1, col]:.1f}; fuel moisture must fall as ERC rises"
            )

    if np.any(cls[:, 2:5] <= 0) or np.any(cls[:, 2:5] > 60):
        out.append("WARN: dead fuel moisture outside a plausible 0-60% range")
    if np.any(cls[:, 5:7] < 30) or np.any(cls[:, 5:7] > 300):
        out.append("WARN: live fuel moisture outside a plausible 30-300% range")

    burn_period = cls[:, 7]
    if np.any(burn_period <= 0) or np.any(burn_period > 1440):
        out.append(
            "ERROR: burn period (column 8, spec field 'Duration') must be in "
            "0-1440 minutes"
        )
    if len(burn_period) > 1 and np.any(np.diff(burn_period) > 1e-9):
        out.append(
            "WARN: burn period lengthens as ERC falls; expected the reverse "
            "(longer burn periods on the most extreme days)"
        )

    spot_prob = cls[:, 8]
    if np.any(spot_prob < 0) or np.any(spot_prob > 1):
        out.append("ERROR: spot probability (column 9) must be in 0-1")
    if np.any(cls[:, 9] < 0):
        out.append("ERROR: spot delay (column 10) must be >= 0")


def _check_erc_streams(p: dict, out: list[str]) -> None:
    """Historic / average / current ERC blocks and the spec p.5 window."""
    n_years, n_wx = p.get("NumERCYears"), p.get("NumWxPerYear")
    hist = p.get("historic_erc")

    if hist is not None:
        if isinstance(n_years, int) and hist.shape[0] != n_years:
            out.append(
                f"ERROR: NumERCYears={n_years} but HistoricERCValues has "
                f"{hist.shape[0]} rows"
            )
        if isinstance(n_wx, int) and hist.ndim == 2 and hist.shape[1] != n_wx:
            out.append(
                f"ERROR: NumWxPerYear={n_wx} but HistoricERCValues rows have "
                f"{hist.shape[1]} values"
            )
        if np.any(~np.isfinite(hist)):
            out.append("ERROR: HistoricERCValues contains NaN/inf")

    for key, label in (("avg_erc", "AvgERCValues"), ("std_erc", "StdDevERCValues")):
        arr = p.get(key)
        if arr is None:
            continue
        if isinstance(n_wx, int) and len(arr) != n_wx:
            out.append(
                f"ERROR: {label} has {len(arr)} values, expected NumWxPerYear={n_wx}"
            )
        if np.any(~np.isfinite(arr)):
            out.append(f"ERROR: {label} contains NaN/inf")
    std = p.get("std_erc")
    if std is not None and np.any(std < 0):
        out.append("ERROR: StdDevERCValues contains negative values")

    cur = p.get("current_erc")
    n_cur, max_lag, dur = p.get("NumWxCurrYear"), p.get("MaxLag"), p.get("Duration")

    if cur is not None:
        if isinstance(n_cur, int) and len(cur) != n_cur:
            out.append(
                f"ERROR: NumWxCurrYear={n_cur} but CurrentERCValues has "
                f"{len(cur)} values"
            )
        if np.any(~np.isfinite(cur)):
            out.append("ERROR: CurrentERCValues contains NaN/inf")

    # Spec p.5: NumWxCurrYear >= MaxLag and < NumWxPerYear - Duration.
    if isinstance(n_cur, int):
        if isinstance(max_lag, int) and n_cur < max_lag:
            out.append(f"ERROR: NumWxCurrYear={n_cur} < MaxLag={max_lag} (spec p.5)")
        if isinstance(n_wx, int) and isinstance(dur, int) and n_cur >= n_wx - dur:
            out.append(
                f"ERROR: NumWxCurrYear={n_cur} must be < NumWxPerYear - Duration "
                f"= {n_wx - dur} (spec p.5)"
            )

    # ERC classes should span the streams FSPro will draw from; out-of-range
    # values fall back on whatever FSPro clamps to, which is undocumented.
    cls = p.get("erc_classes")
    if cls is not None and cls.ndim == 2 and cls.shape[0]:
        c_lo = float(np.rint(cls[:, 0].min()))
        c_hi = float(np.rint(cls[:, 1].max()))
        for arr, label in ((hist, "HistoricERCValues"), (cur, "CurrentERCValues")):
            if arr is None or not np.size(arr) or np.all(~np.isfinite(arr)):
                continue
            a_lo, a_hi = float(np.nanmin(arr)), float(np.nanmax(arr))
            if a_lo < c_lo:
                out.append(
                    f"WARN: {label} minimum {a_lo:.0f} is below the lowest ERC "
                    f"class bound {c_lo:.0f}"
                )
            if a_hi > c_hi:
                out.append(
                    f"WARN: {label} maximum {a_hi:.0f} exceeds the highest ERC "
                    f"class bound {c_hi:.0f}"
                )


def _check_forecast(p: dict, out: list[str]) -> None:
    """NumForecast range and the presence/shape of the forecast rows."""
    n_fc, dur = p.get("NumForecast"), p.get("Duration")
    fc = p.get("forecast")

    if not isinstance(n_fc, int):
        return
    if n_fc < 0:
        out.append(f"ERROR: NumForecast must be >= 0, got {n_fc}")
        return
    if isinstance(dur, int) and n_fc > dur - 1:
        out.append(
            f"ERROR: NumForecast={n_fc} must be <= Duration-1 = {dur - 1} (spec p.5)"
        )
    if n_fc == 0:
        return

    if fc is None or fc.size == 0:
        out.append(
            f"ERROR: NumForecast={n_fc} but no forecast rows follow — FSPro will "
            "read BarrierFill/SavePerimeters/IgnitionFile as forecast records"
        )
        return
    if fc.shape[0] != n_fc:
        out.append(f"ERROR: NumForecast={n_fc} but {fc.shape[0]} forecast rows")
    if fc.ndim != 2 or fc.shape[1] != 3:
        out.append(
            f"ERROR: forecast rows must be 'ERC WindSpeed WindDirection' "
            f"(3 values, spec p.5), got shape {fc.shape}"
        )
        return
    if np.any(fc != np.rint(fc)):
        out.append("WARN: forecast values should be integers (spec p.5)")
    if np.any(fc[:, 2] < 0) or np.any(fc[:, 2] > 360):
        out.append(
            "ERROR: forecast wind direction (column 3) outside 0-360 — "
            "column order is 'ERC WindSpeed WindDirection'"
        )


def _check_lcp(p: dict, lcp_path: "str | Path", out: list[str]) -> None:
    """Spec p.1: Resolution should be a multiple of the LCP cell size."""
    try:
        import rasterio
    except ImportError:
        out.append("WARN: rasterio unavailable, skipped LCP resolution check")
        return

    lcp_path = Path(lcp_path)
    if not lcp_path.exists():
        out.append(f"WARN: lcp_path not found, skipped resolution check: {lcp_path}")
        return

    res = p.get("Resolution")
    if not isinstance(res, float):
        return
    try:
        with rasterio.open(lcp_path) as src:
            cell = float(src.res[0])
    except Exception as exc:
        # Binary .lcp files are not GDAL-readable without the FARSITE driver.
        out.append(f"WARN: could not read cell size from {lcp_path.name}: {exc}")
        return
    if cell <= 0:
        return
    ratio = res / cell
    if abs(ratio - round(ratio)) > 1e-6:
        out.append(
            f"ERROR: Resolution {res:g} m is not a multiple of the LCP cell size "
            f"{cell:g} m (spec p.1)"
        )


# ── Public validation API ─────────────────────────────────────────────────────

def validate_fspro_input(
    path: "str | Path | dict",
    lcp_path: "str | Path | None" = None,
) -> list[str]:
    """
    Validate an FSPro input file against the vendor specification.

    Parameters
    ----------
    path : str or Path or dict
        Path to an ``.input`` file, or an already-parsed dict from
        :func:`parse_fspro_input`.
    lcp_path : str or Path, optional
        Landscape file.  When given, ``Resolution`` is checked for being a
        multiple of the LCP cell size (spec p.1).

    Returns
    -------
    list of str
        Findings, each prefixed ``ERROR:`` or ``WARN:``.  Empty when clean.
        ``ERROR`` means FSPro will misread the file or the physics is wrong;
        ``WARN`` means the value is legal but implausible.

    Examples
    --------
    >>> problems = validate_fspro_input("inputs/pyrome_47.input")
    >>> [x for x in problems if x.startswith("ERROR")]
    []
    """
    p = path if isinstance(path, dict) else parse_fspro_input(path)

    out: list[str] = [f"ERROR: parse — {e}" for e in p.get("_parse_errors", [])]

    _check_required(p, out)
    _check_wind(p, out)
    _check_erc_classes(p, out)
    _check_erc_streams(p, out)
    _check_forecast(p, out)
    if lcp_path is not None:
        _check_lcp(p, lcp_path, out)

    return out


def assert_valid_fspro_input(
    path: "str | Path | dict",
    lcp_path: "str | Path | None" = None,
    warn: bool = True,
) -> list[str]:
    """
    Validate and raise on any ``ERROR:`` finding.

    Parameters
    ----------
    path : str or Path or dict
        Input file or parsed dict.
    lcp_path : str or Path, optional
        Passed through to :func:`validate_fspro_input`.
    warn : bool
        When ``True`` (default), print ``WARN:`` findings instead of
        discarding them silently.

    Returns
    -------
    list of str
        The ``WARN:`` findings, so callers can record them in a manifest.

    Raises
    ------
    ValueError
        If any ``ERROR:`` finding is present.
    """
    findings = validate_fspro_input(path, lcp_path=lcp_path)
    errors = [f for f in findings if f.startswith("ERROR")]
    warnings = [f for f in findings if f.startswith("WARN")]

    name = path if isinstance(path, (str, Path)) else "<parsed input>"
    if errors:
        joined = "\n  ".join(errors)
        raise ValueError(
            f"Invalid FSPro input file ({name}) — {len(errors)} error(s):\n  {joined}"
        )
    if warn:
        for w in warnings:
            print(f"  [validate_fspro_input] {w}")
    return warnings
