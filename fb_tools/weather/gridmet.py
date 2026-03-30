"""
fb_tools/weather/gridmet.py
============================
Process GEE-exported GridMET CSV into per-pyrome FSPro weather arrays.

The expected input is a CSV exported from the ``GridMET_ERC_Climatology``
GEE notebook: one row per (pyrome, date) covering the full fire season
(April 1 – October 31, DOY 91–304, 214 days) for years 2008–2022.

Required CSV columns
--------------------
pyrome_id, date, year, doy, erc, fm100, fm1000, tmmx, rmin

- ``tmmx``: daily maximum temperature (°K) — converted to °F on load
- ``rmin``: daily minimum relative humidity (%) — used with tmmx for NFDRS FM

Outputs feed directly into ``build_fspro_inputs()`` in ``fb_tools.models.fspro``:

- ``build_historic_erc_arrays()``  → ``HistoricERCValues`` block
- ``build_erc_stats()``            → ``AvgERCValues`` / ``StdDevERCValues``
- ``build_erc_classes()``          → ``NumERCClasses`` 5-row table
- ``build_current_erc_values()``   → ``CurrentERCValues`` (scenario-based median)

JSON cache pattern
------------------
Each function that writes a cache saves ``pyrome_{id}_gridmet.json`` in
``out_dir`` / ``cache_dir``.  Matches the pattern used in ``hrrr.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fb_tools.weather.nfdrs import calc_1hr_fm, calc_10hr_fm, kelvin_to_fahrenheit

# FSPro fire-season constants (April 1 – October 31)
_N_SEASON_DAYS: int = 214
_SEASON_START_MONTH: int = 4   # April
_SEASON_END_MONTH: int = 10    # October

# Default spotting parameter rows per ERC class (high → low danger, 5 rows)
# Format: [spot_dist_ft, spot_prob, spot2]
# Matches the 416inputsfile.input example file ordering (highest ERC first).
_DEFAULT_SPOTTING: list[list[float]] = [
    [360, 0.15, 0],
    [300, 0.10, 0],
    [240, 0.05, 0],
    [180, 0.01, 0],
    [120, 0.00, 0],
]


# ── Public API ─────────────────────────────────────────────────────────────────

def load_gridmet_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Load and normalize the GEE-exported GridMET fire-season CSV.

    Parses the ``date`` column, adds ``year`` and ``doy`` integer columns if
    missing, and converts ``tmmx`` from Kelvin to Fahrenheit (stored as
    ``tmmx_f``).

    Parameters
    ----------
    csv_path : str or Path
        Path to the exported CSV file (``gridmet_erc_pyromes.csv``).

    Returns
    -------
    pd.DataFrame
        Columns: ``pyrome_id``, ``date``, ``year``, ``doy``, ``erc``,
        ``fm100``, ``fm1000``, ``tmmx_f``, ``rmin``.
        All numeric columns cast to float.

    Raises
    ------
    FileNotFoundError
        If ``csv_path`` does not exist.
    KeyError
        If any required columns are missing from the CSV.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"GridMET CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"pyrome", "date", "erc", "fm100", "fm1000", "tmmx", "rmin"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])

    if "year" not in df.columns:
        df["year"] = df["date"].dt.year
    if "doy" not in df.columns:
        df["doy"] = df["date"].dt.dayofyear

    df["year"] = df["year"].astype(int)
    df["doy"] = df["doy"].astype(int)

    # Convert tmmx from °K → °F
    df["tmmx_f"] = kelvin_to_fahrenheit(df["tmmx"].values)
    df = df.drop(columns=["tmmx"])

    for col in ["erc", "fm100", "fm1000", "tmmx_f", "rmin"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing key values
    n_before = len(df)
    df = df.dropna(subset=["erc", "pyrome"]).reset_index(drop=True)
    if len(df) < n_before:
        print(f"  [load_gridmet_csv] Dropped {n_before - len(df)} rows with missing erc/PYROME")

    print(f"  [load_gridmet_csv] {len(df):,} rows, {df['pyrome'].nunique()} pyromes, "
          f"years {df['year'].min()}–{df['year'].max()}")
    return df


def build_historic_erc_arrays(
    df: pd.DataFrame,
    pyrome_col: str = "pyrome",
    n_years: int = 15,
    out_dir: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """
    Build per-pyrome historic ERC arrays for FSPro ``HistoricERCValues``.

    Each array has shape ``(n_years, 214)`` — one row per year, 214 columns
    representing fire-season days 1–214 (day 1 = April 1, day 214 = Oct 31).
    Years are sorted chronologically.  Missing day/year combinations are
    filled with the column (day-of-season) median across available years.

    The pivot uses **day-of-season** (1–214 anchored to April 1 of each
    calendar year) rather than raw DOY.  This makes the result leap-year-safe:
    April 1 is always season day 1 regardless of whether the year is a leap
    year, so all years produce exactly 214 columns.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_gridmet_csv`.
    pyrome_col : str
        Column name identifying the spatial grouping unit (default
        ``"pyrome"``).
    n_years : int
        Expected number of years in the record (default 15 for 2008–2022).
        Used only for the warning message; does not truncate data.
    out_dir : str or Path, optional
        If provided, write ``pyrome_{id}_gridmet.json`` cache files here.

    Returns
    -------
    dict[str, np.ndarray]
        ``{pyrome_id: ndarray(n_years, 214)}`` with ERC values.
    """
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # Compute day-of-season (1 = April 1, 214 = Oct 31) anchored per year.
    # This is leap-year-safe: raw DOY shifts by 1 in leap years, but
    # (date - April_1_of_that_year).days is always consistent.
    df2 = df.copy()
    april_1 = pd.to_datetime(df2["year"].astype(str) + "-04-01")
    df2["_dos"] = (df2["date"] - april_1).dt.days + 1

    # Keep only the 214-day fire season (April 1 – Oct 31)
    df2 = df2[(df2["_dos"] >= 1) & (df2["_dos"] <= _N_SEASON_DAYS)]

    season_days = list(range(1, _N_SEASON_DAYS + 1))  # always [1 … 214]

    result: dict[str, np.ndarray] = {}

    for pyrome_id, group in df2.groupby(pyrome_col):
        # Pivot: rows = year, columns = day-of-season (1–214)
        pivot = (
            group.pivot_table(index="year", columns="_dos", values="erc", aggfunc="mean")
            .reindex(columns=season_days)
        )
        pivot = pivot.sort_index()

        # Fill missing day/year cells with column (day-of-season) median
        pivot = pivot.fillna(pivot.median(axis=0))

        arr = pivot.values.astype(float)

        if arr.shape[0] != n_years:
            print(f"  [build_historic_erc_arrays] Pyrome {pyrome_id}: "
                  f"{arr.shape[0]} years (expected {n_years})")

        result[str(pyrome_id)] = arr

        if out_dir is not None:
            cache = {
                "pyrome_id": str(pyrome_id),
                "years": list(map(int, pivot.index)),
                "season_days": season_days,   # 1-based, leap-year-safe
                "NumERCYears": int(arr.shape[0]),
                "NumWxPerYear": int(arr.shape[1]),
                "HistoricERCValues": arr.tolist(),
            }
            _write_json(cache, out_dir / f"pyrome_{pyrome_id}_gridmet.json")

    print(f"  [build_historic_erc_arrays] {len(result)} pyromes → "
          f"arrays shape ({n_years}, {_N_SEASON_DAYS})")
    return result


def build_erc_stats(
    historic_dict: dict[str, np.ndarray],
) -> dict[str, dict]:
    """
    Compute per-DOY ERC statistics for FSPro ``AvgERCValues`` / ``StdDevERCValues``.

    Parameters
    ----------
    historic_dict : dict
        Output of :func:`build_historic_erc_arrays`.

    Returns
    -------
    dict[str, dict]
        ``{pyrome_id: {"avg": ndarray(214), "std": ndarray(214)}}``.
        Both arrays are rounded to 2 decimal places to match FSPro format.
    """
    stats: dict[str, dict] = {}
    for pyrome_id, arr in historic_dict.items():
        stats[str(pyrome_id)] = {
            "avg": np.round(np.nanmean(arr, axis=0), 2),
            "std": np.round(np.nanstd(arr, axis=0, ddof=1), 2),
        }
    return stats


def build_erc_classes(
    df: pd.DataFrame,
    pyrome_col: str = "pyrome",
    n_classes: int = 5,
    erc_col: str = "erc",
    fm100_col: str = "fm100",
    tmmx_col: str = "tmmx_f",
    rmin_col: str = "rmin",
    spotting: list[list[float]] | None = None,
) -> dict[str, np.ndarray]:
    """
    Build the 5-row ERC class table for FSPro ``NumERCClasses`` block.

    Each row represents one ERC percentile class (highest ERC first) with
    associated fuel moisture values derived from GridMET data.

    Row format (10 columns):
    ``[lower, upper, fm1, fm10, fm100, fm_herb, fm_woody, spot_dist, spot_prob, spot2]``

    Fuel moisture derivation:
    - ``fm1``   : median NFDRS 1-hr FM (from tmmx_f + rmin via :func:`calc_1hr_fm`)
    - ``fm10``  : median NFDRS 10-hr FM (from tmmx_f + rmin via :func:`calc_10hr_fm`)
    - ``fm100`` : median GridMET fm100 (direct)
    - ``fm_herb``: fm100 + 5, clipped to [10, 100]  (NFDRS live herbaceous proxy)
    - ``fm_woody``: fm_herb + 10  (NFDRS live woody proxy)

    Quintile ERC bins are computed per pyrome from the full fire-season record.
    Classes are ordered highest-to-lowest ERC (class 1 = most extreme).

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_gridmet_csv`.
    pyrome_col : str
        Column identifying the spatial grouping unit.
    n_classes : int
        Number of ERC classes (default 5).
    erc_col, fm100_col, tmmx_col, rmin_col : str
        Column names in ``df``.
    spotting : list of list, optional
        Override default spotting parameters. Must be ``n_classes`` rows of
        ``[spot_dist_ft, spot_prob, spot2]``. Ordered highest-to-lowest ERC.
        Defaults to :data:`_DEFAULT_SPOTTING`.

    Returns
    -------
    dict[str, np.ndarray]
        ``{pyrome_id: ndarray(n_classes, 10)}``.
    """
    if spotting is None:
        spotting = _DEFAULT_SPOTTING
    if len(spotting) != n_classes:
        raise ValueError(f"len(spotting)={len(spotting)} must equal n_classes={n_classes}")

    result: dict[str, np.ndarray] = {}

    for pyrome_id, group in df.groupby(pyrome_col):
        erc = group[erc_col].dropna().values

        # Quintile bin edges (0%, 20%, 40%, 60%, 80%, 100%)
        quantiles = np.linspace(0, 100, n_classes + 1)
        edges = np.percentile(erc, quantiles)

        # Build one row per class (highest ERC class first)
        rows = []
        for i in range(n_classes - 1, -1, -1):
            lower = edges[i]
            upper = edges[i + 1]

            # Mask observations in this ERC bin
            mask = (group[erc_col] >= lower) & (group[erc_col] <= upper)
            sub = group[mask]

            if len(sub) == 0:
                # Fall back to all observations if bin is empty
                sub = group

            fm1 = float(np.nanmedian(
                calc_1hr_fm(sub[tmmx_col].values, sub[rmin_col].values)
            ))
            fm10 = float(np.nanmedian(
                calc_10hr_fm(sub[tmmx_col].values, sub[rmin_col].values)
            ))
            fm100 = float(np.nanmedian(sub[fm100_col].values))
            fm_herb = float(np.clip(fm100 + 5.0, 10.0, 100.0))
            fm_woody = float(np.clip(fm_herb + 10.0, 10.0, 200.0))

            spot = spotting[n_classes - 1 - i]  # highest ERC → spotting[0]
            rows.append([
                round(lower, 1), round(upper, 1),
                round(fm1, 1), round(fm10, 1), round(fm100, 1),
                round(fm_herb, 1), round(fm_woody, 1),
                float(spot[0]), float(spot[1]), float(spot[2]),
            ])

        result[str(pyrome_id)] = np.array(rows)

    print(f"  [build_erc_classes] {len(result)} pyromes × {n_classes} ERC classes")
    return result


def build_current_erc_values(
    historic_dict: dict[str, np.ndarray],
    start_doy: int,
    n_days: int = 79,
) -> dict[str, np.ndarray]:
    """
    Build ``CurrentERCValues`` from the historic-median approach.

    Computes the column-wise median across all years for a consecutive
    ``n_days`` window starting at ``start_doy`` in the fire season.  The
    DOY index is relative to the fire season (DOY 1 = April 1).

    For scenario-based (non-operational) FSPro runs this provides a
    climatologically representative current-season ERC sequence without
    requiring a real-time GEE fetch.

    Parameters
    ----------
    historic_dict : dict
        Output of :func:`build_historic_erc_arrays` —
        ``{pyrome_id: ndarray(n_years, 214)}``.
    start_doy : int
        1-based fire-season DOY (1 = April 1) at which the current-year
        window begins.
    n_days : int
        Length of the current-year ERC sequence (default 79, matching the
        416 example file).

    Returns
    -------
    dict[str, np.ndarray]
        ``{pyrome_id: ndarray(n_days)}`` of median ERC values.

    Raises
    ------
    ValueError
        If ``start_doy + n_days - 1`` exceeds the fire season length (214).
    """
    end_idx = start_doy - 1 + n_days
    if end_idx > _N_SEASON_DAYS:
        raise ValueError(
            f"start_doy={start_doy} + n_days={n_days} - 1 = {end_idx} exceeds "
            f"fire season length {_N_SEASON_DAYS}"
        )

    result: dict[str, np.ndarray] = {}
    for pyrome_id, arr in historic_dict.items():
        window = arr[:, start_doy - 1 : start_doy - 1 + n_days]
        result[str(pyrome_id)] = np.round(np.nanmedian(window, axis=0)).astype(int)
    return result


def load_gridmet_pyrome_cache(
    pyrome_id: str | int,
    cache_dir: str | Path,
    return_meta: bool = False,
) -> np.ndarray | dict:
    """
    Load cached GridMET historic ERC array from JSON.

    Parameters
    ----------
    pyrome_id : str or int
        Pyrome identifier matching the ``pyrome_{id}_gridmet.json`` filename.
    cache_dir : str or Path
        Directory containing the JSON cache files.
    return_meta : bool
        If False (default), return only the ``HistoricERCValues`` array.
        If True, return the full metadata dictionary.

    Returns
    -------
    np.ndarray or dict
        Shape ``(NumERCYears, 214)`` array, or full dict if ``return_meta=True``.

    Raises
    ------
    FileNotFoundError
        If the cache file does not exist.
    """
    cache_path = Path(cache_dir) / f"pyrome_{pyrome_id}_gridmet.json"
    if not cache_path.exists():
        raise FileNotFoundError(f"GridMET cache not found: {cache_path}")
    with open(cache_path) as f:
        meta = json.load(f)
    if return_meta:
        return meta
    return np.array(meta["HistoricERCValues"])


# ── Internal helpers ───────────────────────────────────────────────────────────

def _write_json(data: dict, path: Path) -> None:
    """Write ``data`` to ``path`` as pretty-printed JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [gridmet] Wrote {path.name}")
