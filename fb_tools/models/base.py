"""
Shared CLI runner for fire behavior models.

All fire behavior models (FlamMap, FSPro, FSim, MTT, FARSITE) follow the
same pattern:
  1. Write a model-specific input file.
  2. Write a command file that points the executable at the input file.
  3. Invoke the executable as a subprocess.
  4. Capture stdout/stderr to a log file.

This module provides the shared subprocess helper used by every model
wrapper.  Model-specific logic lives in the individual model modules.
"""

import subprocess
from pathlib import Path


def run_cli(exe_path, command_file, output_dir, extra_args=None):
    """
    Run a fire behavior model executable and capture output to a log file.

    The process is launched with *output_dir* as its working directory, which
    is required by FlamMap and similar tools that resolve relative paths in the
    command file against CWD.

    Parameters
    ----------
    exe_path : str or Path
        Absolute path to the model executable (e.g. ``TestFlamMap.exe``).
    command_file : str or Path
        Path to the command file (may be relative to *output_dir*).
    output_dir : str or Path
        Working directory for the subprocess; log file is written here.
    extra_args : list of str, optional
        Additional positional arguments appended after the command file name.
        Used by FSPro, which takes ``lcp_path`` and ``output_basename`` as
        extra CLI arguments.  FlamMap callers leave this as ``None``.

    Returns
    -------
    subprocess.CompletedProcess

    Raises
    ------
    FileNotFoundError
        If *exe_path* does not exist.
    """
    exe_path = Path(exe_path)
    output_dir = Path(output_dir)
    command_file = Path(command_file)

    if not exe_path.exists():
        raise FileNotFoundError(f"Model executable not found: {exe_path}")

    cmd = [str(exe_path), str(command_file.name)]
    if extra_args:
        cmd.extend([str(a) for a in extra_args])

    log_path = output_dir / f"{exe_path.stem}_run.log"
    with open(log_path, "w") as log:
        result = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(output_dir),
        )

    return result


def _write_shortterm_inputs(f, params, n_process=1, extra_keys=None):
    """
    Write the shared ``ShortTerm-Inputs-File-Version-1`` header block.

    Used by MTT and FSPro wrappers to avoid duplicating the fuel-moisture /
    wind / crown-fire block that is already written inline in
    :func:`~fb_tools.models.flammap.run_flammap_scenarios`.

    Parameters
    ----------
    f : file object
        Opened for writing (text mode).
    params : dict or pandas.Series
        Scenario parameters.  Required keys: ``FM_1hr``, ``FM_10hr``,
        ``FM_100hr``, ``FM_herb``, ``FM_woody``, ``CROWN_FIRE_METHOD``,
        ``WIND_SPEED``, ``WIND_DIRECTION``.
        Optional: ``GRIDDED_WINDS_GENERATE``, ``GRIDDED_WINDS_RESOLUTION``.
    n_process : int
        Number of processor threads (default ``1``).
    extra_keys : dict, optional
        Additional ``KEY: value`` lines written after the shared block, in
        insertion order.  Pass ``None`` for no additional keys.
    """
    f.write("ShortTerm-Inputs-File-Version-1\n\n")
    f.write("FUEL_MOISTURES_DATA: 1\n")
    f.write(
        f"0 {params['FM_1hr']} {params['FM_10hr']} "
        f"{params['FM_100hr']} {params['FM_herb']} {params['FM_woody']}\n"
    )
    f.write("FOLIAR_MOISTURE_CONTENT: 100\n")
    f.write(f"CROWN_FIRE_METHOD: {params['CROWN_FIRE_METHOD']}\n")
    f.write(f"NUMBER_PROCESSORS: {n_process}\n")
    f.write(f"WIND_SPEED: {params['WIND_SPEED']}\n")
    f.write(f"WIND_DIRECTION: {params['WIND_DIRECTION']}\n")

    if params.get("GRIDDED_WINDS_GENERATE") == "Yes":
        f.write("GRIDDED_WINDS_GENERATE: Yes\n")
        f.write(f"GRIDDED_WINDS_RESOLUTION: {params['GRIDDED_WINDS_RESOLUTION']}\n")

    if extra_keys:
        for key, value in extra_keys.items():
            f.write(f"{key}: {value}\n")
