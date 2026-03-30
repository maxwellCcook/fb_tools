"""
FSPro (Fire Spread Probability) CLI wrapper and input builder.

Provides two layers of functionality:

1. **Input file builder** — :func:`build_fspro_inputs` writes a complete
   ``FSPRO-Inputs-File-Version-4`` text file from pyrome-specific weather
   climatology arrays produced by :mod:`fb_tools.weather.gridmet` (ERC) and
   :mod:`fb_tools.weather.hrrr` (wind).

2. **CLI runner** — :func:`run_fspro` and :func:`run_fspro_batch` invoke
   ``TestFSPro.exe`` with a pre-written input file.

FSPro command-line invocation::

    TestFSPro {lcp_path} {input_file} {output_base}

All three arguments are required positional arguments; there is no separate
command file.

Platform note
-------------
TestFSPro.exe is a Windows-only executable.  Calling :func:`run_fspro` on
macOS or Linux raises ``RuntimeError``.  :func:`build_fspro_inputs` has no
platform restriction and can be used on any OS for input preparation.
"""

import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


# ── FSPro simulation defaults ─────────────────────────────────────────────────

_FSPRO_DEFAULTS: dict = {
    "Dimension": 2,
    "Resolution": 90.0,
    "Duration": 7,
    "NumFires": 1000,
    "MaxLag": 30,
    "PolyDegree": 9,
    "ThreadsPerFire": 1,
    "UseCustomFuels": 0,
    "SPOTTING_SEED": 617327,
    "CROWN_FIRE_METHOD": "Finney",
    "BarrierFill": 0,
    "SavePerimeters": 1,
    "NumForecast": 0,
}

# Default wind speed / direction bin edges (matches 416inputsfile.input)
_DEFAULT_SPEED_BREAKS_MPH: list[float] = [5, 10, 15, 20, 25, 30]
_DEFAULT_DIR_BREAKS_DEG: list[float] = [45, 90, 135, 180, 225, 270, 315, 360]


# ── Input file builder ────────────────────────────────────────────────────────

def build_fspro_inputs(
    output_path: str | Path,
    wind_cells: np.ndarray,
    calm_value: float,
    erc_historic: np.ndarray,
    erc_avg: np.ndarray,
    erc_std: np.ndarray,
    erc_classes: np.ndarray,
    current_erc: np.ndarray,
    ignition_file: str | Path,
    speed_breaks: list[float] | None = None,
    dir_breaks: list[float] | None = None,
    forecast: list[tuple] | None = None,
    **kwargs,
) -> Path:
    """
    Write a complete ``FSPRO-Inputs-File-Version-4`` input file.

    Assembles all required FSPro weather blocks from pyrome-specific arrays
    produced by :mod:`fb_tools.weather.gridmet` and :mod:`fb_tools.weather.hrrr`,
    then writes them in the exact section order expected by TestFSPro.exe.

    Parameters
    ----------
    output_path : str or Path
        Destination path for the written input file (e.g.
        ``"data/fspro_inputs/pyrome_42_baseline.input"``).
        Parent directory is created if it does not exist.
    wind_cells : np.ndarray
        Wind frequency table, shape ``(NumWindSpeeds, NumWindDirs)``.
        Produced by :func:`~fb_tools.weather.hrrr.build_wind_cells` or
        :func:`~fb_tools.weather.hrrr.load_pyrome_wind_cells`.
        Values are percentage frequencies; rows sum to ~100 after subtracting
        ``calm_value``.
    calm_value : float
        Percentage of calm observations (wind speed below threshold).
        Stored as ``CalmValue`` in the FSPro input.
    erc_historic : np.ndarray
        Historic daily ERC values, shape ``(NumERCYears, 214)``.
        Produced by :func:`~fb_tools.weather.gridmet.build_historic_erc_arrays`.
    erc_avg : np.ndarray
        Per-DOY average ERC, shape ``(214,)``.
        Produced by :func:`~fb_tools.weather.gridmet.build_erc_stats`.
    erc_std : np.ndarray
        Per-DOY standard deviation of ERC, shape ``(214,)``.
        Produced by :func:`~fb_tools.weather.gridmet.build_erc_stats`.
    erc_classes : np.ndarray
        ERC class table, shape ``(5, 10)`` — one row per class (highest ERC
        first).  Row format:
        ``[lower, upper, fm1, fm10, fm100, fm_herb, fm_woody, spot_dist, spot_prob, spot2]``
        Produced by :func:`~fb_tools.weather.gridmet.build_erc_classes`.
    current_erc : np.ndarray
        Current-season ERC sequence, shape ``(n_days,)``.
        Produced by :func:`~fb_tools.weather.gridmet.build_current_erc_values`.
    ignition_file : str or Path
        Path to the ignition shapefile (polygon or polyline).  Written
        as-is into the ``IgnitionFile`` field.
    speed_breaks : list of float, optional
        Upper bounds of wind speed bins in mph.
        Defaults to ``[5, 10, 15, 20, 25, 30]``.
    dir_breaks : list of float, optional
        Upper bounds of wind direction bins in degrees (meteorological FROM).
        Defaults to ``[45, 90, 135, 180, 225, 270, 315, 360]``.
    forecast : list of tuple, optional
        Forecast rows, each ``(erc, wind_dir_deg, wind_spd_mph)``.
        Overrides ``NumForecast`` in *kwargs* to ``len(forecast)``.
    **kwargs
        Override any key in :data:`_FSPRO_DEFAULTS`, e.g.
        ``NumFires=2000``, ``Duration=14``, ``SPOTTING_SEED=99999``.

    Returns
    -------
    Path
        Absolute path to the written file.

    Examples
    --------
    >>> from fb_tools.weather import load_pyrome_wind_cells, load_gridmet_pyrome_cache
    >>> from fb_tools.weather.gridmet import build_erc_stats, build_current_erc_values
    >>> wind = load_pyrome_wind_cells("42", "data/weather/pyrome_wind", return_meta=True)
    >>> erc_meta = load_gridmet_pyrome_cache("42", "data/weather/pyrome_erc", return_meta=True)
    >>> historic = np.array(erc_meta["HistoricERCValues"])
    >>> stats = build_erc_stats({"42": historic})["42"]
    >>> build_fspro_inputs(
    ...     "inputs/pyrome_42.input",
    ...     wind_cells=wind["wind_cells"],
    ...     calm_value=wind["CalmValue"],
    ...     erc_historic=historic,
    ...     erc_avg=stats["avg"],
    ...     erc_std=stats["std"],
    ...     erc_classes=erc_classes["42"],
    ...     current_erc=current_erc["42"],
    ...     ignition_file="data/ignitions/watershed_42_ign.shp",
    ...     NumFires=2000,
    ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if speed_breaks is None:
        speed_breaks = _DEFAULT_SPEED_BREAKS_MPH
    if dir_breaks is None:
        dir_breaks = _DEFAULT_DIR_BREAKS_DEG

    # Merge defaults with caller overrides
    params = dict(_FSPRO_DEFAULTS)
    params.update(kwargs)

    if forecast is not None:
        params["NumForecast"] = len(forecast)

    n_speed = wind_cells.shape[0]
    n_dir = wind_cells.shape[1]
    n_erc_years = erc_historic.shape[0]
    n_wx_per_year = erc_historic.shape[1]
    n_erc_classes = erc_classes.shape[0]
    n_current = len(current_erc)

    def _fmt_row(values, fmt="{:.2f}"):
        return " ".join(fmt.format(v) for v in values)

    with open(output_path, "w") as f:
        # ── Header block ──────────────────────────────────────────────────────
        f.write("FSPRO-Inputs-File-Version-4\n")
        f.write(f"Dimension: {params['Dimension']}\n")
        f.write(f"Resolution: {params['Resolution']}\n")
        f.write(f"Duration: {params['Duration']}\n")
        f.write(f"NumFires: {params['NumFires']}\n")
        f.write(f"MaxLag: {params['MaxLag']}\n")
        f.write(f"PolyDegree: {params['PolyDegree']}\n")
        f.write(f"ThreadsPerFire: {params['ThreadsPerFire']}\n")
        f.write(f"UseCustomFuels: {params['UseCustomFuels']}\n")
        f.write(f"SPOTTING_SEED: {params['SPOTTING_SEED']}\n")
        f.write(f"CROWN_FIRE_METHOD: {params['CROWN_FIRE_METHOD']}\n")

        # ── Wind rose ─────────────────────────────────────────────────────────
        f.write(f"CalmValue: {calm_value:.2f}\n")
        f.write(f"NumWindDirs: {n_dir}\n")
        f.write(" ".join(str(int(d)) for d in dir_breaks) + "\n")
        f.write(f"NumWindSpeeds: {n_speed}\n")
        f.write(" ".join(str(int(s)) for s in speed_breaks) + "\n")
        f.write("WindCellValues:\n")
        for row in wind_cells:
            f.write(" ".join(f"{v:.2f}" for v in row) + "\n")

        # ── ERC classes ───────────────────────────────────────────────────────
        f.write(f"NumERCClasses: {n_erc_classes}\n")
        for row in erc_classes:
            # lower upper fm1 fm10 fm100 fm_herb fm_woody spot_dist spot_prob spot2
            f.write(
                f"{row[0]:.0f} {row[1]:.0f} "
                f"{row[2]:.1f} {row[3]:.1f} {row[4]:.1f} "
                f"{row[5]:.1f} {row[6]:.1f} "
                f"{row[7]:.0f} {row[8]:.2f} {row[9]:.0f}\n"
            )

        # ── Historic ERC ─────────────────────────────────────────────────────
        f.write(f"NumERCYears: {n_erc_years}\n")
        f.write(f"NumWxPerYear: {n_wx_per_year}\n")
        f.write("HistoricERCValues: \n")
        for year_row in erc_historic:
            f.write(" ".join(str(int(round(v))) for v in year_row) + "\n")

        # ── ERC statistics ────────────────────────────────────────────────────
        f.write("AvgERCValues: \n")
        f.write(" ".join(f"{v:.1f}" for v in erc_avg) + "\n")
        f.write("StdDevERCValues:\n")
        f.write(" ".join(f"{v:.2f}" for v in erc_std) + "\n")

        # ── Current season ERC ────────────────────────────────────────────────
        f.write(f"NumWxCurrYear: {n_current}\n")
        f.write("CurrentERCValues: \n")
        f.write(" ".join(str(int(round(v))) for v in current_erc) + "\n")

        # ── Forecast ─────────────────────────────────────────────────────────
        f.write(f"NumForecast: {params['NumForecast']}\n")
        if forecast:
            for fc_row in forecast:
                f.write(" ".join(str(v) for v in fc_row) + "\n")

        # ── Barrier / output / ignition ───────────────────────────────────────
        f.write(f"BarrierFill: {params['BarrierFill']}\n")
        f.write(f"SavePerimeters: {params['SavePerimeters']}\n")
        f.write(f"IgnitionFile: {ignition_file}\n")

    print(f"  [build_fspro_inputs] Wrote {output_path.name} "
          f"(NumFires={params['NumFires']}, {n_erc_years}yr ERC, "
          f"wind {n_speed}×{n_dir})")
    return output_path.resolve()


def build_treatment_pair(
    out_path: str | Path,
    ignition_file: str | Path,
    wind_cells: np.ndarray,
    calm_value: float,
    erc_historic: np.ndarray,
    erc_avg: np.ndarray,
    erc_std: np.ndarray,
    erc_classes: np.ndarray,
    current_erc: np.ndarray,
    seed: int = 617327,
    **kwargs,
) -> Path:
    """
    Write a single FSPro input file suitable for paired baseline/treated runs.

    Because ``TestFSPro.exe`` takes the LCP as a run-time positional argument
    (not embedded in the input file), one input file can serve both the
    baseline and treated landscape scenarios.  Fixing ``SPOTTING_SEED``
    guarantees identical weather draws across both runs, giving a clean
    counterfactual comparison without external scenario management.

    This is a thin wrapper around :func:`build_fspro_inputs` that enforces
    the ``SPOTTING_SEED`` parameter and documents the pairing intent.

    Parameters
    ----------
    out_path : str or Path
        Destination for the written ``.input`` file.
    ignition_file : str or Path
        Ignition polygon/polyline shapefile (same for both runs).
    wind_cells : np.ndarray
        Wind frequency table ``(NumWindSpeeds, NumWindDirs)``.
    calm_value : float
        Calm percentage for FSPro ``CalmValue`` field.
    erc_historic : np.ndarray
        Historic ERC array ``(NumERCYears, 214)``.
    erc_avg, erc_std : np.ndarray
        Per-day-of-season ERC mean and std, shape ``(214,)``.
    erc_classes : np.ndarray
        ERC class table ``(5, 10)`` — highest ERC first.
    current_erc : np.ndarray
        Current-season ERC sequence ``(n_days,)``.
    seed : int
        ``SPOTTING_SEED`` shared by both runs (default 617327).
        **Do not vary between baseline and treated runs.**
    **kwargs
        Passed to :func:`build_fspro_inputs` (e.g. ``NumFires=2000``).
        ``SPOTTING_SEED`` in kwargs is overridden by *seed*.

    Returns
    -------
    Path
        Absolute path to the written input file.

    Examples
    --------
    >>> path = build_treatment_pair(
    ...     "inputs/watershed_42.input",
    ...     ignition_file="data/watershed_42_ign.shp",
    ...     wind_cells=wind_arr, calm_value=calm_pct,
    ...     erc_historic=hist, erc_avg=avg, erc_std=std,
    ...     erc_classes=classes, current_erc=curr,
    ...     NumFires=1000,
    ... )
    >>> # Run FSPro twice with the same input, different LCPs:
    >>> run_fspro(exe, lcp_baseline, path, out_dir / "baseline")
    >>> run_fspro(exe, lcp_treated,  path, out_dir / "treated")
    """
    # Enforce shared seed — override any caller-supplied value
    kwargs["SPOTTING_SEED"] = seed

    return build_fspro_inputs(
        output_path=out_path,
        wind_cells=wind_cells,
        calm_value=calm_value,
        erc_historic=erc_historic,
        erc_avg=erc_avg,
        erc_std=erc_std,
        erc_classes=erc_classes,
        current_erc=current_erc,
        ignition_file=ignition_file,
        **kwargs,
    )


def run_fspro(
    fspro_exe,
    lcp_fp,
    input_file,
    output_directory,
    output_basename=None,
    num_fires_warn=1000,
):
    """
    Run a single FSPro scenario.

    Parameters
    ----------
    fspro_exe : str or Path
        Path to ``TestFSPro.exe``.
    lcp_fp : str or Path
        Path to the landscape file (.lcp or .tif).  Must be in the same
        datum and projection as the ignition file referenced in *input_file*.
    input_file : str or Path
        Path to the FSPro inputs text file (``FSPRO-Inputs-File-Version-4``
        format).  This file contains weather climatology, NumFires, ERC
        distributions, and the ``IgnitionFile`` path.  It is typically
        generated by WFDSS or prepared manually.
    output_directory : str or Path
        Directory where FSPro outputs are written.  Created automatically
        including parents.
    output_basename : str, optional
        Base name for FSPro output files (no extension).  Defaults to
        ``"fspro_out"``.  Outputs are written to
        ``output_directory/{output_basename}*``.
    num_fires_warn : int
        If the ``NumFires`` value found in *input_file* is below this
        threshold, a warning is printed.  Default is ``1000``; production
        runs typically require 1000–3000+ fires.  Set to ``0`` to suppress.

    Returns
    -------
    subprocess.CompletedProcess

    Raises
    ------
    RuntimeError
        On non-Windows platforms (TestFSPro.exe is Windows-only).
    FileNotFoundError
        If *fspro_exe*, *lcp_fp*, or *input_file* does not exist.

    Notes
    -----
    FSPro does not use a separate command file — the executable takes three
    positional CLI arguments directly.  Unlike FlamMap and MTT, this function
    does not use :func:`~fb_tools.models.base.run_cli` because FSPro's
    argument order is ``[exe, lcp, input_file, output_base]``, which differs
    from the ``[exe, command_file]`` convention used by FlamMap.
    """
    if platform.system() != "Windows":
        raise RuntimeError(
            "TestFSPro.exe is a Windows-only executable. "
            "Run this function on Windows (e.g. in Parallels)."
        )

    fspro_exe  = Path(fspro_exe)
    lcp_fp     = Path(lcp_fp)
    input_file = Path(input_file)
    output_directory = Path(output_directory)

    if not fspro_exe.exists():
        raise FileNotFoundError(f"FSPro executable not found: {fspro_exe}")
    if not lcp_fp.exists():
        raise FileNotFoundError(f"LCP file not found: {lcp_fp}")
    if not input_file.exists():
        raise FileNotFoundError(f"FSPro input file not found: {input_file}")

    # Warn if NumFires is low
    if num_fires_warn > 0:
        try:
            text = input_file.read_text()
            for line in text.splitlines():
                if line.strip().lower().startswith("numfires"):
                    val = int(line.split(":")[-1].strip())
                    if val < num_fires_warn:
                        print(
                            f"Warning: NumFires={val} in {input_file.name}. "
                            f"Production runs typically need {num_fires_warn}+ fires."
                        )
                    break
        except Exception:
            pass  # non-critical; proceed regardless

    output_directory.mkdir(parents=True, exist_ok=True)

    if output_basename is None:
        output_basename = "fspro_out"

    output_base = output_directory / output_basename

    log_path = output_directory / f"{fspro_exe.stem}_run.log"
    cmd = [str(fspro_exe), str(lcp_fp), str(input_file), str(output_base)]

    with open(log_path, "w") as log:
        result = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(output_directory),
        )

    return result


def run_fspro_batch(
    fspro_exe,
    scenarios_df,
    output_root,
    lcp_dir=None,
    input_file_col="FSPro_input",
    output_basename_col="output_basename",
    num_fires_warn=1000,
):
    """
    Run all FSPro scenarios in *scenarios_df* and return a status summary.

    Outputs are organised as::

        output_root/
          <lcp_stem>/
            <scenario>/
              TestFSPro_run.log
              fspro_out* (FSPro output files)

    Parameters
    ----------
    fspro_exe : str or Path
        Path to ``TestFSPro.exe``.
    scenarios_df : pd.DataFrame
        Scenario table.  Required columns:

        - ``Scenario`` — scenario name
        - ``LCP`` — path to the landscape file
        - Column named by *input_file_col* — path to the FSPro inputs file

        Optional column named by *output_basename_col* — base name for
        outputs; defaults to ``"fspro_out"`` when absent.

    output_root : str or Path
        Root directory for all run outputs.
    lcp_dir : str or Path, optional
        Prepended to relative ``LCP`` paths in the table.
    input_file_col : str
        Name of the column containing FSPro input file paths
        (default ``"FSPro_input"``).
    output_basename_col : str
        Name of the column containing output base names
        (default ``"output_basename"``).
    num_fires_warn : int
        Passed through to :func:`run_fspro`.  Set to ``0`` to silence.

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
        log_path = out_dir / "TestFSPro_run.log"
        status = "success"

        inp = row.get(input_file_col)
        if inp is None:
            status = f"error: column '{input_file_col}' not found in scenarios_df row"
            summary_rows.append({
                "Scenario":   scenario_name,
                "LCP":        str(lcp_path),
                "output_dir": str(out_dir),
                "status":     status,
                "log_path":   str(log_path),
            })
            print(f"[{status}]")
            continue

        out_basename = row.get(output_basename_col, "fspro_out")

        try:
            run_fspro(
                fspro_exe=fspro_exe,
                lcp_fp=lcp_path,
                input_file=inp,
                output_directory=out_dir,
                output_basename=str(out_basename) if out_basename else None,
                num_fires_warn=num_fires_warn,
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
