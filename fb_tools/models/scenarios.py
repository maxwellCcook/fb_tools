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

from pathlib import Path

import pandas as pd

from .flammap import run_flammap_scenarios


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
        rows.append(block)

    df = pd.concat(rows, ignore_index=True)
    return df


def run_batch(
    fm_exe,
    scenarios_df,
    output_root,
    lcp_dir=None,
    n_process=1,
    stack_out=False,
    cleanup=False,
    mask=None,
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

    for _, row in scenarios_df.iterrows():
        lcp_path = Path(row["LCP"])
        if lcp_dir and not lcp_path.is_absolute():
            lcp_path = lcp_dir / lcp_path

        scenario_name = str(row["Scenario"])
        lcp_stem = lcp_path.stem

        out_dir = output_root / lcp_stem / scenario_name
        out_dir.mkdir(parents=True, exist_ok=True)

        log_path = out_dir / "TestFlamMap_run.log"
        status = "success"

        try:
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

        print(f"[{status}] {lcp_stem} / {scenario_name}")

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
        rows.append(block)

    return pd.concat(rows, ignore_index=True)
