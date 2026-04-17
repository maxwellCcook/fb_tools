"""
FlamMap CLI wrapper.

Writes the FlamMap short-term input file and command file, then invokes
TestFlamMap.exe via the shared CLI runner.

Notes on cross-platform paths
------------------------------
All paths are handled through ``pathlib.Path``.  The executable itself must
be a Windows binary and is therefore typically run inside Parallels or a
Windows VM — pass the *Windows* path to *fm_exe* when running on Windows,
or a Parallels-accessible UNC / mapped path when calling from macOS.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .base import run_cli
from ..fuelscape.lcp import stack_rasters


def run_flammap_scenarios(
    fm_exe,
    lcp_fp,
    fm_params,
    output_directory,
    n_process=1,
    tag=None,
    stack_out=False,
    cleanup=False,
    mask=None,
):
    """
    Run a single FlamMap scenario for a given landscape and weather condition.

    Writes ``FlamMap.input`` and ``FMcommand.txt`` into *output_directory*,
    then executes ``TestFlamMap.exe``.

    Parameters
    ----------
    fm_exe : str or Path
        Path to ``TestFlamMap.exe``.
    lcp_fp : str or Path
        Path to the landscape file (.lcp or .tif).
    fm_params : dict or pandas.Series
        Scenario parameters.  Expected keys:

        - ``FM_1hr``, ``FM_10hr``, ``FM_100hr``, ``FM_herb``, ``FM_woody``
          — dead/live fuel moistures (%)
        - ``CROWN_FIRE_METHOD`` — ``1`` (Rothermel) or ``2`` (Scott & Reinhardt)
        - ``WIND_SPEED`` — 20-ft wind speed (mph)
        - ``WIND_DIRECTION`` — wind azimuth (degrees); ``-1`` = uphill, ``-2`` = downhill
        - ``GRIDDED_WINDS_GENERATE`` — ``"Yes"`` to enable WindNinja
        - ``GRIDDED_WINDS_RESOLUTION`` — WindNinja resolution (m), required when above is ``"Yes"``
        - ``Outputs`` — comma-separated output names, e.g.
          ``"FLAMELENGTH, CROWNSTATE, SPREADRATE"``
        - ``LCPType`` — label for this fuelscape (used as *tag* if not provided)

    output_directory : str or Path
        Directory where input files and outputs are written.
    n_process : int
        Number of processor threads passed to FlamMap (default ``1``).
    tag : str, optional
        Override the run label (defaults to ``fm_params["LCPType"]``).
    stack_out : bool
        Stack individual output TIFFs into a single multi-band file
        (default ``False``).
    cleanup : bool
        Delete single-band TIFFs after stacking; only relevant when
        *stack_out* is ``True`` (default ``False``).
    mask : GeoDataFrame, optional
        If provided, every output GeoTIFF in *output_directory* is clipped
        to the union of these geometries before stacking.  Reduces file sizes
        when the LCP covers a larger extent than the region of interest.
        Default ``None``.

    Returns
    -------
    subprocess.CompletedProcess
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    if tag is None:
        tag = fm_params.get("LCPType", Path(lcp_fp).stem)

    input_file = output_directory / "FlamMap.input"
    command_file = output_directory / "FMcommand.txt"

    # --- Write the FlamMap short-term input file
    with open(input_file, "w") as f:
        f.write("ShortTerm-Inputs-File-Version-1\n\n")

        f.write("FUEL_MOISTURES_DATA: 1\n")
        f.write(
            f"0 {fm_params['FM_1hr']} {fm_params['FM_10hr']} "
            f"{fm_params['FM_100hr']} {fm_params['FM_herb']} {fm_params['FM_woody']}\n"
        )
        f.write("FOLIAR_MOISTURE_CONTENT: 100\n")
        f.write(f"CROWN_FIRE_METHOD: {fm_params['CROWN_FIRE_METHOD']}\n")
        f.write(f"NUMBER_PROCESSORS: {n_process}\n")
        f.write(f"WIND_SPEED: {fm_params['WIND_SPEED']}\n")
        f.write(f"WIND_DIRECTION: {fm_params['WIND_DIRECTION']}\n")

        if fm_params.get("GRIDDED_WINDS_GENERATE") == "Yes":
            f.write("GRIDDED_WINDS_GENERATE: Yes\n")
            f.write(f"GRIDDED_WINDS_RESOLUTION: {fm_params['GRIDDED_WINDS_RESOLUTION']}\n")

        outputs = [o.strip() for o in str(fm_params["Outputs"]).split(",") if o.strip()]
        for output_type in outputs:
            f.write(f"{output_type}: 1\n")

    # --- Write the FlamMap command file
    # Format: <lcp_path> <input_filename> <output_dir> <mode>
    with open(command_file, "w") as f:
        f.write(f"{Path(lcp_fp)} {input_file.name} . 2\n")

    # --- Run the model
    result = run_cli(
        exe_path=fm_exe,
        command_file=command_file,
        output_dir=output_directory,
    )

    # --- Optionally clip outputs before stacking (smaller files → faster stack)
    if mask is not None:
        from ..utils.geo import clip_raster_inplace
        tifs = list(output_directory.glob("*.tif"))
        print(f"Clipping {len(tifs)} output TIFF(s) to mask ...")
        for tif in tifs:
            clip_raster_inplace(tif, mask)

    # --- Optionally stack outputs
    if stack_out:
        stack_rasters(in_dir=output_directory, tag=tag, cleanup=cleanup)

    return result


def run_flammap_conditioning(
    fm_exe,
    lcp_fp,
    fm_params,
    weather_data: list[tuple],
    wind_data: list[tuple],
    conditioning_end: datetime | str,
    output_directory,
    elevation_ft: int,
    n_process: int = 1,
    tag: str | None = None,
    stack_out: bool = False,
    cleanup: bool = False,
    mask=None,
):
    """
    Run FlamMap with a multi-day conditioning period using GridMET weather data.

    Writes a ``FlamMap-Inputs-File-Version-1`` input file with a
    ``WEATHER_DATA`` block (daily GridMET) and a ``WIND_DATA`` block (HRRR
    fire-hour winds).  The ``FUEL_MOISTURES_DATA`` row provides **initial live
    FM** (fm_herb, fm_woody), which remains constant throughout conditioning
    while FlamMap dynamically updates dead FM from the weather stream.

    Use :func:`~fb_tools.weather.gridmet.build_flammap_weather_data` to
    produce ``weather_data`` and
    :func:`~fb_tools.weather.hrrr.build_flammap_wind_data` to produce
    ``wind_data``.  Use
    :func:`~fb_tools.weather.gridmet.build_flammap_fuel_moistures` to derive
    ``fm_params`` from GridMET climatology.

    Parameters
    ----------
    fm_exe : str or Path
        Path to ``TestFlamMap.exe``.
    lcp_fp : str or Path
        Path to the landscape file (.lcp).
    fm_params : dict
        Initial fuel moisture values (Model 0).  Required keys:
        ``FM_1hr``, ``FM_10hr``, ``FM_100hr``, ``FM_herb``, ``FM_woody``,
        ``CROWN_FIRE_METHOD``, ``WIND_SPEED``, ``WIND_DIRECTION``, ``Outputs``.
        Optional: ``GRIDDED_WINDS_GENERATE``, ``GRIDDED_WINDS_RESOLUTION``.
    weather_data : list of tuple
        Rows from :func:`~fb_tools.weather.gridmet.build_flammap_weather_data`.
        Each tuple: ``(Mth, Day, Pcp, mTH, xTH, mT, xT, mH, xH, Elv, PST, PET)``.
    wind_data : list of tuple
        Rows from :func:`~fb_tools.weather.hrrr.build_flammap_wind_data`.
        Each tuple: ``(Mth, Day, HHMM, ws_mph, wd_deg, cc)``.
    conditioning_end : datetime or str
        End of the conditioning period. If str, parsed as ``"YYYY-MM-DD HH:MM"``.
        FlamMap format: ``MM DD HHMM``.
    output_directory : str or Path
        Directory where input files and outputs are written.
    elevation_ft : int
        Elevation of the representative weather location in feet (written to
        ``RAWS_ELEVATION`` for reference; not used by ``WEATHER_DATA`` mode).
    n_process : int
        Processor threads (default 1).
    tag : str, optional
        Override the run label.
    stack_out : bool
        Stack output TIFFs into multi-band file (default False).
    cleanup : bool
        Delete single-band TIFFs after stacking (default False).
    mask : GeoDataFrame, optional
        Clip outputs to this geometry before stacking.

    Returns
    -------
    subprocess.CompletedProcess
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    if tag is None:
        tag = fm_params.get("LCPType", Path(lcp_fp).stem)

    if isinstance(conditioning_end, str):
        conditioning_end = datetime.strptime(conditioning_end, "%Y-%m-%d %H:%M")

    cond_end_str = conditioning_end.strftime("%m %d %H%M")

    input_file = output_directory / "FlamMap.input"
    command_file = output_directory / "FMcommand.txt"

    with open(input_file, "w") as f:
        f.write("FlamMap-Inputs-File-Version-1\n")
        f.write(f"CONDITIONING_PERIOD_END: {cond_end_str}\n\n")

        # Initial fuel moistures (live FM stays constant through conditioning)
        f.write("FUEL_MOISTURES_DATA: 1\n")
        f.write(
            f"0 {fm_params['FM_1hr']} {fm_params['FM_10hr']} "
            f"{fm_params['FM_100hr']} {fm_params['FM_herb']} {fm_params['FM_woody']}\n"
        )

        # Daily weather stream (GridMET)
        f.write(f"\nWEATHER_DATA: {len(weather_data)}\n")
        f.write("# Mth Day Pcp mTH xTH mT xT mH xH Elv PST PET\n")
        for row in weather_data:
            f.write(" ".join(str(v) for v in row) + "\n")

        # Hourly wind data (HRRR fire hours)
        f.write(f"\nWIND_DATA: {len(wind_data)}\n")
        f.write("# Mth Day Hour Speed Direction CloudCover\n")
        for row in wind_data:
            f.write(" ".join(str(v) for v in row) + "\n")

        f.write(f"\nFOLIAR_MOISTURE_CONTENT: 100\n")
        f.write(f"CROWN_FIRE_METHOD: {fm_params['CROWN_FIRE_METHOD']}\n")
        f.write(f"NUMBER_PROCESSORS: {n_process}\n")
        f.write(f"WIND_SPEED: {fm_params['WIND_SPEED']}\n")
        f.write(f"WIND_DIRECTION: {fm_params['WIND_DIRECTION']}\n")

        if fm_params.get("GRIDDED_WINDS_GENERATE") == "Yes":
            f.write("GRIDDED_WINDS_GENERATE: Yes\n")
            f.write(f"GRIDDED_WINDS_RESOLUTION: {fm_params['GRIDDED_WINDS_RESOLUTION']}\n")

        outputs = [o.strip() for o in str(fm_params["Outputs"]).split(",") if o.strip()]
        for output_type in outputs:
            f.write(f"{output_type}:\n")

    with open(command_file, "w") as f:
        f.write(f"{Path(lcp_fp)} {input_file.name} . 2\n")

    result = run_cli(
        exe_path=fm_exe,
        command_file=command_file,
        output_dir=output_directory,
    )

    if mask is not None:
        from ..utils.geo import clip_raster_inplace
        tifs = list(output_directory.glob("*.tif"))
        print(f"Clipping {len(tifs)} output TIFF(s) to mask ...")
        for tif in tifs:
            clip_raster_inplace(tif, mask)

    if stack_out:
        stack_rasters(in_dir=output_directory, tag=tag, cleanup=cleanup)

    return result
