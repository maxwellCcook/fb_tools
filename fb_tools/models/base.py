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


def run_cli(exe_path, command_file, output_dir):
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

    log_path = output_dir / f"{exe_path.stem}_run.log"
    with open(log_path, "w") as log:
        result = subprocess.run(
            [str(exe_path), str(command_file.name)],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(output_dir),
        )

    return result
