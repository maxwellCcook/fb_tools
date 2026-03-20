"""
FlamMap MTT (Minimum Travel Time) CLI wrapper.

Writes the MTT short-term input file and command file, then invokes
TestMTT.exe via the shared CLI runner.

MTT command file format (one line per run, multiple runs supported)::

    {lcp_path} {input_file} {ignition_shp} {barrier_or_0} {output_base_path} {output_type}

Where *output_base_path* is ``{output_dir}/{basename}`` (no extension).
The directory component must pre-exist — TestMTT will not create directories.

MTT output flags in the input file use flag-only syntax (``FLAMELENGTH:`` with
no value), unlike FlamMap which uses ``FLAMELENGTH: 1``.

Platform note
-------------
TestMTT.exe is a Windows-only executable.  Calling :func:`run_mtt` on macOS
or Linux raises ``RuntimeError`` with a message pointing to ``run_cell2fire``
as a cross-platform alternative.
"""

import platform
from pathlib import Path

import pandas as pd

from .base import run_cli, _write_shortterm_inputs
from .scenarios import _MTT_DEFAULTS, _DEFAULTS


# Required MTT-specific keys in the params dict.
_MTT_REQUIRED = {
    "MTT_RESOLUTION",
    "MTT_SIM_TIME",
    "MTT_TRAVEL_PATH_INTERVAL",
    "MTT_SPOT_PROBABILITY",
    "MTT_FILL_BARRIERS",
}


def run_mtt(
    mtt_exe,
    lcp_fp,
    mtt_params,
    ignition_shp,
    output_directory,
    barrier_shp=None,
    n_process=1,
    output_type=2,
):
    """
    Run a single MTT (Minimum Travel Time) scenario.

    Writes ``MTT.input`` and ``MTTcommand.txt`` into *output_directory*,
    then calls :func:`~fb_tools.models.base.run_cli`.

    Parameters
    ----------
    mtt_exe : str or Path
        Path to ``TestMTT.exe``.
    lcp_fp : str or Path
        Path to the landscape file (.lcp or .tif).
    mtt_params : dict or pandas.Series
        Scenario parameters.  Standard moisture / wind keys (see
        :func:`~fb_tools.models.flammap.run_flammap_scenarios`) plus the
        five MTT-specific keys:

        - ``MTT_RESOLUTION`` — grid resolution (m), e.g. ``30``
        - ``MTT_SIM_TIME`` — simulation time in seconds, e.g. ``400``
        - ``MTT_TRAVEL_PATH_INTERVAL`` — travel path interval (m), e.g. ``50``
        - ``MTT_SPOT_PROBABILITY`` — spotting probability (0.0–1.0)
        - ``MTT_FILL_BARRIERS`` — fill barriers flag (``1`` or ``0``)

        Optional: ``Outputs`` — comma-separated FlamMap output names to
        request alongside MTT outputs (e.g. ``"FLAMELENGTH, CROWNSTATE"``).
        Written as flag-only lines (``FLAMELENGTH:`` with no value).

    ignition_shp : str or Path
        Path to the ignition shapefile (.shp).
    output_directory : str or Path
        Directory where input files and MTT outputs are written.
        Created automatically including parents.
    barrier_shp : str or Path, optional
        Path to the barrier shapefile.  Pass ``None`` (default) for no
        barriers; the command file will use ``0``.
    n_process : int
        Processor threads (default ``1``).
    output_type : int
        Output file type: ``0`` = ASCII + GeoTIFF, ``1`` = ASCII only,
        ``2`` = GeoTIFF only (default ``2``).

    Returns
    -------
    subprocess.CompletedProcess

    Raises
    ------
    RuntimeError
        On non-Windows platforms (TestMTT.exe is Windows-only).
    FileNotFoundError
        If *mtt_exe*, *lcp_fp*, or *ignition_shp* does not exist.
    ValueError
        If any required MTT key is missing from *mtt_params*.
    """
    if platform.system() != "Windows":
        raise RuntimeError(
            "TestMTT.exe is a Windows-only executable. "
            "Run this function on Windows (e.g. in Parallels). "
            "For cross-platform burn probability, use run_cell2fire() instead."
        )

    mtt_exe = Path(mtt_exe)
    lcp_fp  = Path(lcp_fp)
    ignition_shp = Path(ignition_shp)
    output_directory = Path(output_directory)

    if not mtt_exe.exists():
        raise FileNotFoundError(f"MTT executable not found: {mtt_exe}")
    if not lcp_fp.exists():
        raise FileNotFoundError(f"LCP file not found: {lcp_fp}")
    if not ignition_shp.exists():
        raise FileNotFoundError(f"Ignition shapefile not found: {ignition_shp}")

    missing = _MTT_REQUIRED - set(mtt_params.keys())
    if missing:
        raise ValueError(
            f"mtt_params is missing required MTT keys: {sorted(missing)}. "
            f"Use build_mtt_scenarios() to fill defaults automatically."
        )

    output_directory.mkdir(parents=True, exist_ok=True)

    input_file   = output_directory / "MTT.input"
    command_file = output_directory / "MTTcommand.txt"

    # MTT output base path: directory + LCP stem (no extension)
    output_base = output_directory / lcp_fp.stem

    # --- Write the MTT input file
    mtt_extra = {
        "MTT_RESOLUTION":           mtt_params["MTT_RESOLUTION"],
        "MTT_SIM_TIME":             mtt_params["MTT_SIM_TIME"],
        "MTT_TRAVEL_PATH_INTERVAL": mtt_params["MTT_TRAVEL_PATH_INTERVAL"],
        "MTT_SPOT_PROBABILITY":     mtt_params["MTT_SPOT_PROBABILITY"],
        "MTT_FILL_BARRIERS":        mtt_params["MTT_FILL_BARRIERS"],
    }

    with open(input_file, "w") as f:
        _write_shortterm_inputs(f, mtt_params, n_process=n_process, extra_keys=mtt_extra)

        # Output flags: MTT uses flag-only syntax (no value after colon)
        outputs_str = mtt_params.get("Outputs", "")
        if outputs_str:
            outputs = [o.strip() for o in str(outputs_str).split(",") if o.strip()]
            for out in outputs:
                f.write(f"{out}:\n")

    # --- Write the MTT command file (6-arg format)
    barrier_arg = str(barrier_shp) if barrier_shp is not None else "0"
    with open(command_file, "w") as f:
        f.write(
            f"{lcp_fp} {input_file.name} {ignition_shp} "
            f"{barrier_arg} {output_base} {output_type}\n"
        )

    # --- Run the model
    return run_cli(
        exe_path=mtt_exe,
        command_file=command_file,
        output_dir=output_directory,
    )


def run_mtt_batch(
    mtt_exe,
    scenarios_df,
    output_root,
    ignitions,
    lcp_dir=None,
    barrier_shp=None,
    n_process=1,
    output_type=2,
):
    """
    Run all MTT scenarios in *scenarios_df* and return a status summary.

    Outputs are organised as::

        output_root/
          <lcp_stem>/
            <scenario>/
              MTT.input
              MTTcommand.txt
              TestMTT_run.log
              <MTT output TIFFs>

    Parameters
    ----------
    mtt_exe : str or Path
        Path to ``TestMTT.exe``.
    scenarios_df : pd.DataFrame
        Scenario table from :func:`~fb_tools.models.scenarios.build_mtt_scenarios`
        or :func:`~fb_tools.models.scenarios.load_scenarios` (with MTT columns
        added).
    output_root : str or Path
        Root directory for all run outputs.
    ignitions : str, Path, or dict
        Ignition shapefile path used for all scenarios, OR a
        ``{scenario_name: shp_path}`` dict for per-scenario ignitions.
    lcp_dir : str or Path, optional
        Prepended to relative ``LCP`` paths in the table.
    barrier_shp : str, Path, or dict, optional
        Barrier shapefile(s).  Single path used for all scenarios, or a
        ``{scenario_name: shp_path}`` dict for per-scenario barriers.
        ``None`` means no barriers (default).
    n_process : int
        Processor threads for each MTT run (default ``1``).
    output_type : int
        Output file type passed to :func:`run_mtt` (default ``2`` = GeoTIFF).

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
        log_path = out_dir / "TestMTT_run.log"
        status = "success"

        # Resolve ignition
        if isinstance(ignitions, dict):
            ign = ignitions.get(scenario_name)
            if ign is None:
                status = f"error: no ignition shapefile for scenario '{scenario_name}'"
                summary_rows.append({
                    "Scenario":   scenario_name,
                    "LCP":        str(lcp_path),
                    "output_dir": str(out_dir),
                    "status":     status,
                    "log_path":   str(log_path),
                })
                print(f"[{status}]")
                continue
        else:
            ign = ignitions

        # Resolve barrier
        if isinstance(barrier_shp, dict):
            barr = barrier_shp.get(scenario_name)
        else:
            barr = barrier_shp

        try:
            run_mtt(
                mtt_exe=mtt_exe,
                lcp_fp=lcp_path,
                mtt_params=row.to_dict(),
                ignition_shp=ign,
                output_directory=out_dir,
                barrier_shp=barr,
                n_process=n_process,
                output_type=output_type,
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
