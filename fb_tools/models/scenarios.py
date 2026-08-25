"""
FlamMap scenario management — build, load, and batch-run fire scenarios.

Three entry points
------------------
``load_scenarios``
    Read a scenario CSV (same format as ``fire_scenarios.csv``) into a
    DataFrame that ``run_batch`` can consume directly.

``build_scenarios``
    Construct the same DataFrame programmatically from a table of fire-weather
    conditions and a list of LCP files.  The cross-product of every condition ×
    every LCP is returned.

``run_batch``
    Iterate over every row of a scenarios DataFrame and execute FlamMap,
    organising outputs under ``output_root/{lcp_stem}/{scenario}/``.
    Returns a summary DataFrame with run status for each row.
"""

import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from .flammap import run_flammap_scenarios

# FlamMap writes "FlamMap 1: 47.300 complete" to stdout as it works, and
# run_cli redirects that into the run log.  Tailing the log is the only
# progress signal available without changing how the executable is invoked.
_PCT_RE = re.compile(r"(\d+\.\d+)\s+complete")


@contextmanager
def _progress(log_path, label, every=30):
    """Echo percent-complete from a model's run log while it executes.

    Prints are flushed because batch runs are usually driven from a notebook
    or a pipe, where Python block-buffers stdout and nothing would appear
    until the process exits.
    """
    stop = threading.Event()

    def tail():
        while not stop.wait(every):
            try:
                hits = _PCT_RE.findall(Path(log_path).read_text(errors="ignore"))
            except OSError:
                continue
            pct = f"{float(hits[-1]):.0f}%" if hits else "starting"
            print(f"    {label} {pct}", flush=True)

    threading.Thread(target=tail, daemon=True).start()
    try:
        yield
    finally:
        stop.set()


# Columns that must be present in a valid scenarios DataFrame.
_REQUIRED_COLS = {
    "Scenario",
    "LCP",
    "WIND_SPEED",
    "WIND_DIRECTION",
    "FM_1hr",
    "FM_10hr",
    "FM_100hr",
    "FM_herb",
    "FM_woody",
    "CROWN_FIRE_METHOD",
    "Outputs",
}

# Defaults applied by build_scenarios when not supplied by the caller.
_DEFAULTS = {
    "CROWN_FIRE_METHOD":        "ScottReinhardt",
    "GRIDDED_WINDS_GENERATE":   "No",
    "GRIDDED_WINDS_RESOLUTION": 30,
    "Outputs": "FLAMELENGTH, CROWNSTATE, SPREADRATE, MIDFLAME, HEATAREA",
    "FM_1000hr": None,
    "ERC":       None,
    "FM_NAME":   None,
}

# MTT-specific defaults applied by build_mtt_scenarios.
_MTT_DEFAULTS = {
    "MTT_RESOLUTION":           30,
    "MTT_SIM_TIME":             400,
    "MTT_TRAVEL_PATH_INTERVAL": 50,
    "MTT_SPOT_PROBABILITY":     0.0,
    "MTT_FILL_BARRIERS":        0,
}


def stacked_output_path(output_root, lcp_path, scenario):
    """
    Return the expected path to the stacked FlamMap output for one run.

    Mirrors the naming convention used by :func:`run_batch` and
    :func:`~fb_tools.fuelscape.lcp.stack_rasters`::

        output_root/<lcp_stem>/<scenario>/<scenario>_<LCP_STEM>.tif

    Parameters
    ----------
    output_root : str or Path
        Root directory passed to :func:`run_batch`.
    lcp_path : str or Path
        LCP file path (same value used in the scenarios DataFrame).
    scenario : str
        Scenario name (``Scenario`` column value, e.g. ``"Pct97"``).

    Returns
    -------
    Path
        Full path to the expected stacked GeoTIFF.  The file is not
        checked for existence; call ``.exists()`` if needed.
    """
    lcp_stem = Path(lcp_path).stem
    out_dir  = Path(output_root) / lcp_stem / scenario
    return out_dir / f"{scenario}_{lcp_stem.upper()}.tif"


def load_scenarios(csv_path, lcp_dir=None):
    """
    Load a FlamMap scenario CSV into a DataFrame.

    Parameters
    ----------
    csv_path : str or Path
        Path to the scenario CSV.  Expected columns match ``fire_scenarios.csv``:
        ``Scenario``, ``LCP``, ``WIND_SPEED``, ``WIND_DIRECTION``,
        ``FM_1hr`` … ``FM_woody``, ``CROWN_FIRE_METHOD``, ``Outputs``.
        Additional metadata columns (``FM_NAME``, ``ERC``) are kept as-is.
    lcp_dir : str or Path, optional
        Directory that contains the LCP files referenced in the ``LCP`` column.
        If provided, ``LCP`` values that are bare filenames are resolved to full
        paths.  Absolute paths in the CSV are left unchanged.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If required columns are missing from the CSV.
    """
    df = pd.read_csv(Path(csv_path))

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Scenario CSV missing required columns: {missing}")

    if lcp_dir is not None:
        lcp_dir = Path(lcp_dir)
        df["LCP"] = df["LCP"].apply(
            lambda p: str(lcp_dir / p) if not Path(p).is_absolute() else p
        )

    return df


def build_scenarios(conditions, lcps, outputs=None, **defaults):
    """
    Build a scenarios DataFrame from weather conditions and LCP files.

    Produces the cross-product of every row in *conditions* with every path
    in *lcps*, resulting in ``len(conditions) × len(lcps)`` rows.

    Parameters
    ----------
    conditions : pd.DataFrame
        One row per fire-weather / fuel-moisture condition.  Required columns:

        - ``Scenario`` — name for this condition (e.g. ``"Pct25"``)
        - ``WIND_SPEED`` — 20-ft wind speed (mph)
        - ``WIND_DIRECTION`` — wind azimuth; ``-1`` = uphill, ``-2`` = downhill
        - ``FM_1hr``, ``FM_10hr``, ``FM_100hr``, ``FM_herb``, ``FM_woody``

        Optional columns (filled from *defaults* if absent):
        ``CROWN_FIRE_METHOD``, ``GRIDDED_WINDS_GENERATE``,
        ``GRIDDED_WINDS_RESOLUTION``, ``Outputs``, ``FM_1000hr``,
        ``ERC``, ``FM_NAME``.
    lcps : list of str or Path
        LCP files (baseline, treated variants, etc.) to pair with every
        condition row.
    outputs : str, optional
        Comma-separated FlamMap output names to use for every scenario.
        Overrides the ``Outputs`` column in *conditions* if provided.
    **defaults
        Override any of ``_DEFAULTS`` (e.g.
        ``CROWN_FIRE_METHOD="Rothermel"``).

    Returns
    -------
    pd.DataFrame
        Same column layout as :func:`load_scenarios`.

    Examples
    --------
    Build a 6-condition × 3-LCP scenario table (18 runs):

    >>> conditions = pd.DataFrame({
    ...     "Scenario":      ["Pct25", "Pct50", "Pct75", "Pct90", "Pct97", "Pct100"],
    ...     "WIND_SPEED":    [8, 9, 10.5, 12.5, 17, 33],
    ...     "WIND_DIRECTION":[-1, -1, -1, -1, -1, -1],
    ...     "FM_1hr":        [21, 12, 9, 7.5, 5.8, 5.8],
    ...     "FM_10hr":       [17, 13, 11, 9.5, 7.5, 7.5],
    ...     "FM_100hr":      [15, 13, 12, 10.5, 8.5, 8.5],
    ...     "FM_herb":       [100, 80, 70, 50, 30, 30],
    ...     "FM_woody":      [130, 110, 90, 80, 60, 60],
    ... })
    >>> lcps = ["baseline.tif", "hand_thin.tif", "mech_thin.tif"]
    >>> df = build_scenarios(conditions, lcps)
    >>> len(df)
    18
    """
    resolved_defaults = {**_DEFAULTS, **defaults}
    if outputs is not None:
        resolved_defaults["Outputs"] = outputs

    # fill any missing optional columns with defaults
    cond = conditions.copy()
    for col, val in resolved_defaults.items():
        if col not in cond.columns:
            cond[col] = val

    rows = []
    for lcp in lcps:
        lcp = Path(lcp)
        block = cond.copy()
        block.insert(1, "LCP", str(lcp))
        block["FM_NAME"] = block["FM_NAME"].fillna(lcp.stem)
        rows.append(block)

    df = pd.concat(rows, ignore_index=True)
    return df


# Fuel-moisture / wind keys copied verbatim from a scenario cache entry into a
# conditions row.  Names match both the cache entry (gridmet.build_flammap_
# scenario_cache) and the conditions columns build_scenarios expects.
_CACHE_CONDITION_KEYS = (
    "WIND_SPEED",
    "WIND_DIRECTION",
    "FM_1hr",
    "FM_10hr",
    "FM_100hr",
    "FM_herb",
    "FM_woody",
)


def scenario_cache_to_conditions(cache, percentiles=None, scenario_prefix="Pct"):
    """
    Convert a loaded FlamMap scenario cache into a ``build_scenarios`` table.

    Closes the manual glue step between the weather layer and the run layer:
    :func:`~fb_tools.weather.gridmet.load_flammap_scenario_cache` returns a
    nested dict, and this turns its ``"scenarios"`` block into a flat
    ``conditions`` DataFrame ready to pass to :func:`build_scenarios`.

    Parameters
    ----------
    cache : dict
        A cache dict as returned by
        :func:`~fb_tools.weather.gridmet.load_flammap_scenario_cache` (must
        contain a ``"scenarios"`` mapping keyed by percentile, e.g. ``"p97"``).
    percentiles : iterable, optional
        Subset / ordering of percentile bands to emit.  Items may be ints
        (``97``) or cache keys (``"p97"``).  Defaults to every scenario in the
        cache, in its stored order.
    scenario_prefix : str
        Prefix for the generated ``Scenario`` name; ``"p97"`` → ``"Pct97"``
        (default prefix ``"Pct"``).

    Returns
    -------
    pd.DataFrame
        One row per percentile with columns ``Scenario`` + the fuel-moisture /
        wind fields consumed by :func:`build_scenarios`.

    Raises
    ------
    KeyError
        If *cache* lacks a ``"scenarios"`` block, a requested percentile is
        absent, or a scenario entry is missing a required field (e.g.
        ``WIND_SPEED`` was ``None`` because no wind source was supplied when
        the cache was built).

    Examples
    --------
    >>> from fb_tools import load_flammap_scenario_cache, scenario_cache_to_conditions
    >>> cache = load_flammap_scenario_cache(42, "data/weather/pyrome_flammap/")
    >>> conditions = scenario_cache_to_conditions(cache, percentiles=[50, 90, 97])
    >>> scenarios = build_scenarios(conditions, ["baseline.tif", "treated.tif"])
    """
    if "scenarios" not in cache:
        raise KeyError("cache has no 'scenarios' block — pass a FlamMap scenario cache dict")
    scenarios = cache["scenarios"]

    if percentiles is None:
        keys = list(scenarios.keys())
    else:
        keys = [f"p{p}" if not str(p).startswith("p") else str(p) for p in percentiles]

    rows = []
    for key in keys:
        if key not in scenarios:
            raise KeyError(f"percentile {key!r} not in cache scenarios ({list(scenarios)})")
        entry = scenarios[key]
        name = f"{scenario_prefix}{key[1:]}" if key.startswith("p") else key
        row = {"Scenario": name}
        for k in _CACHE_CONDITION_KEYS:
            if k not in entry or entry[k] is None:
                raise KeyError(
                    f"scenario {key!r} is missing {k!r} (value None?). "
                    "Rebuild the cache with a wind source / FM source that populates it."
                )
            row[k] = entry[k]
        rows.append(row)

    return pd.DataFrame(rows)


def build_wind_sweep_conditions(
    cache,
    wind_speeds,
    percentile=97,
    wind_direction=None,
    scenario_fmt="{pct}_W{ws:g}",
):
    """
    Hold fuel moisture at one percentile and sweep wind speed.

    Answers "under fixed hot/dry conditions, how does wind speed drive fire
    behaviour?".  Every returned row carries the *same* fuel moistures — the
    ones cached for *percentile* — and differs only in ``WIND_SPEED``, so any
    difference in the resulting FlamMap surfaces is attributable to wind alone.

    Parameters
    ----------
    cache : dict
        A FlamMap scenario cache from
        :func:`~fb_tools.weather.gridmet.load_flammap_scenario_cache`.
    wind_speeds : iterable of float
        20-ft wind speeds (mph) to run.  Duplicates are dropped and the values
        are sorted ascending.
    percentile : int or str
        Which cached percentile supplies the fuel moistures (default ``97``).
        Accepts ``97`` or ``"p97"``.
    wind_direction : int or float, optional
        Overrides the cached ``WIND_DIRECTION`` for every row.  FlamMap reads
        ``-1`` as upslope and ``-2`` as downslope; a value in ``[0, 360)`` is a
        fixed azimuth.  Defaults to whatever the cache stored.
    scenario_fmt : str
        Format string for the ``Scenario`` name, given ``pct`` (e.g. ``"Pct97"``)
        and ``ws`` (the wind speed).  The default yields ``"Pct97_W20"``.

    Returns
    -------
    pd.DataFrame
        One row per wind speed, with the columns
        :func:`build_scenarios` expects plus a ``WIND_SPEED_baseline`` column
        recording the cached wind speed the percentile was derived with.

    Raises
    ------
    ValueError
        If *wind_speeds* is empty or holds a negative value.
    KeyError
        If *percentile* is not in the cache (raised by
        :func:`scenario_cache_to_conditions`).

    Examples
    --------
    >>> cache = load_flammap_scenario_cache(46, cache_dir)
    >>> conditions = build_wind_sweep_conditions(cache, [5, 10, 15, 20, 25, 30, 35, 40])
    >>> scenarios = build_scenarios(conditions, [lcp])
    >>> len(scenarios)
    8
    """
    speeds = sorted({float(w) for w in wind_speeds})
    if not speeds:
        raise ValueError("wind_speeds is empty — pass at least one wind speed.")
    if any(w < 0 for w in speeds):
        raise ValueError(f"wind_speeds must be non-negative, got {speeds}.")

    # One clean row of cached fuel moistures for the requested percentile.
    base = scenario_cache_to_conditions(cache, percentiles=[percentile]).iloc[0]
    pct_name = base["Scenario"]

    rows = []
    for ws in speeds:
        row = base.to_dict()
        row["WIND_SPEED_baseline"] = base["WIND_SPEED"]
        row["WIND_SPEED"] = ws
        if wind_direction is not None:
            row["WIND_DIRECTION"] = wind_direction
        row["Scenario"] = scenario_fmt.format(pct=pct_name, ws=ws)
        rows.append(row)

    return pd.DataFrame(rows)


def run_batch(
    fm_exe,
    scenarios_df,
    output_root,
    lcp_dir=None,
    n_process=1,
    stack_out=False,
    cleanup=False,
    mask=None,
    skip_existing=False,
):
    """
    Run all scenarios in *scenarios_df* and return a status summary.

    Outputs are organised as::

        output_root/
          <lcp_stem>/
            <scenario>/
              FlamMap.input
              FMcommand.txt
              TestFlamMap_run.log
              <FlamMap output TIFFs>

    Parameters
    ----------
    fm_exe : str or Path
        Path to ``TestFlamMap.exe``.
    scenarios_df : pd.DataFrame
        Scenario table from :func:`load_scenarios` or :func:`build_scenarios`.
    output_root : str or Path
        Root directory for all run outputs.
    lcp_dir : str or Path, optional
        Prepended to relative ``LCP`` paths in the table.
    n_process : int
        Processor threads for each FlamMap run (default ``1``).
    stack_out : bool
        Stack per-output TIFFs into a multi-band file after each run.
    cleanup : bool
        Delete single-band TIFFs after stacking.
    mask : GeoDataFrame, optional
        Passed through to :func:`run_flammap_scenarios`.  If provided,
        output TIFFs are clipped to this geometry before stacking after
        each scenario completes.  Default ``None``.
    skip_existing : bool
        If ``True``, skip any scenario whose output directory already
        contains at least one ``.tif`` file.  Useful for resuming a
        partially-completed batch without re-running finished scenarios.
        Default ``False``.

    Returns
    -------
    pd.DataFrame
        One row per scenario with columns:
        ``Scenario``, ``LCP``, ``output_dir``, ``status``, ``log_path``.
    """
    output_root = Path(output_root)
    if lcp_dir is not None:
        lcp_dir = Path(lcp_dir)

    summary_rows = []
    n_runs = len(scenarios_df)

    for i, (_, row) in enumerate(scenarios_df.iterrows(), start=1):
        lcp_path = Path(row["LCP"])
        if lcp_dir and not lcp_path.is_absolute():
            lcp_path = lcp_dir / lcp_path

        scenario_name = str(row["Scenario"])
        lcp_stem = lcp_path.stem

        out_dir = output_root / lcp_stem / scenario_name
        out_dir.mkdir(parents=True, exist_ok=True)

        log_path = out_dir / "TestFlamMap_run.log"

        if skip_existing and any(out_dir.glob("*.tif")):
            print(f"[skip] {lcp_stem} / {scenario_name}", flush=True)
            summary_rows.append({
                "Scenario":   scenario_name,
                "LCP":        str(lcp_path),
                "output_dir": str(out_dir),
                "status":     "skipped",
                "log_path":   str(log_path),
            })
            continue

        status = "success"
        label = f"[{i}/{n_runs}] {lcp_stem} / {scenario_name}"
        print(f"{label} starting", flush=True)
        t0 = time.time()

        try:
            with _progress(log_path, label):
                run_flammap_scenarios(
                    fm_exe=fm_exe,
                    lcp_fp=lcp_path,
                    fm_params=row.to_dict(),
                    output_directory=out_dir,
                    n_process=n_process,
                    stack_out=stack_out,
                    cleanup=cleanup,
                    mask=mask,
                )
        except Exception as exc:
            status = f"error: {exc}"

        summary_rows.append({
            "Scenario":   scenario_name,
            "LCP":        str(lcp_path),
            "output_dir": str(out_dir),
            "status":     status,
            "log_path":   str(log_path),
        })

        print(f"{label} {status} in {time.time() - t0:.0f}s", flush=True)

    return pd.DataFrame(summary_rows)


def build_mtt_scenarios(conditions, lcps, outputs=None, **defaults):
    """
    Build an MTT scenarios DataFrame from weather conditions and LCP files.

    Identical to :func:`build_scenarios` but also fills ``_MTT_DEFAULTS``
    (``MTT_RESOLUTION``, ``MTT_SIM_TIME``, ``MTT_TRAVEL_PATH_INTERVAL``,
    ``MTT_SPOT_PROBABILITY``, ``MTT_FILL_BARRIERS``) so the resulting
    DataFrame is accepted by :func:`~fb_tools.models.mtt.run_mtt_batch`.

    Parameters
    ----------
    conditions : pd.DataFrame
        Same as :func:`build_scenarios`.
    lcps : list of str or Path
        LCP files to pair with every condition row.
    outputs : str, optional
        Comma-separated MTT output flag names (e.g. ``"FLAMELENGTH, CROWNSTATE"``).
        These are written as flag-style lines (``FLAMELENGTH:`` with no value)
        in the MTT input file.  Overrides the ``Outputs`` column in *conditions*.
    **defaults
        Override any ``_DEFAULTS`` or ``_MTT_DEFAULTS`` key.

    Returns
    -------
    pd.DataFrame
        Same column layout as :func:`build_scenarios`, plus the five MTT
        columns.
    """
    resolved_defaults = {**_DEFAULTS, **_MTT_DEFAULTS, **defaults}
    if outputs is not None:
        resolved_defaults["Outputs"] = outputs

    cond = conditions.copy()
    for col, val in resolved_defaults.items():
        if col not in cond.columns:
            cond[col] = val

    rows = []
    for lcp in lcps:
        lcp = Path(lcp)
        block = cond.copy()
        block.insert(1, "LCP", str(lcp))
        block["FM_NAME"] = block["FM_NAME"].fillna(lcp.stem)
        rows.append(block)

    return pd.concat(rows, ignore_index=True)
