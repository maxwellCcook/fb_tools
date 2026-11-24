# Plan: `fb_tools.active_fire` — access, process and plot active-fire detections

> **Status: build steps 1–2 and treatments done (2026-11-14)** — `firms.py`, `qa.py`, `timeutil.py`, `events.py`, `footprints.py`, `grids.py`, `tests/test_active_fire.py` (30 tests). The FIRMS fetch has run live; steps 2's functions have only run on synthetic data. Working notebook: `code/notebooks/active_fire/01_FIRMS.ipynb`. Next is step 3 (`swath.py`). Settle the remaining "Open at build time" items with Max as you reach them.
>
> **Swath module deferred (Max, 2026-11-14).** FIRMS now serves version 2 detections (the Aspen Acres cache: S-NPP / NOAA-20 SP `version = 2`, NOAA-21 `2.0NRT`), so earthaccess no longer has a version advantage. `swath.py` moves behind treatments, point-pattern and plots; revisit it for the cloud / not-observed mask, exact sample/line geometry, sub-minute times, or science-quality NOAA-21. Build order is now: treatments → pointpattern → plots/animate → showcase notebook → swath.
>
> **Treatments built (2026-11-14)** — `treatments.py::treatment_detections`, `plot.py::plot_grid` / `color_norm` (`scale=` linear / log / sqrt; later plots take the same argument). 33 tests in `tests/test_active_fire.py`. Decisions (Max):
> - Attribution `method="footprint"` is the default (a detection counts when its scan × track box overlaps the polygon; `cover_fraction` = share of the polygon under the boxes); `"centre"` is the alternative.
> - The showcase summarises the **atomic interactions layer** (CFT + TWIG for the fire AOI through `treatment_interactions`), `id_col="ATOM_ID"`.
> - FRP statistics are over the attributed detections' whole-pixel FRP, **not overlap-weighted** — built that way, not yet confirmed by Max.
>
> **Notebooks (Max, 2026-11-14):** `code/notebooks/active_fire/01_FIRMS.ipynb` stays the short module demo. The showcase lives in the case study: `projects/case_studies/AspenAcres/code/Python/ActiveFire_Showcase.ipynb` (detections → grid products → Overture structures from `colorado-structure-graph/data/raw/co_buildings.parquet` → treatment atoms → LANDFIRE download). Structures: each building takes the cell holding its centroid, each cell a count. LANDFIRE is download-only for now — how it is analysed is deliberately undecided. BAER severity landed 2026-11-14 (`AspenAcres/data/spatial/raw/BAER/baer_sbs.tif` classes 1–4, `baer_rdnbr.tif`; 30 m, EPSG:32613, Landsat 8 pre 2026-06-18 / post 2026-07-12) and is loaded and mapped in section 7; its join to the detections is undecided. Perimeter-to-grid steps were checked on the cached fetch (first-detection cells 408 km² vs a 413 km² perimeter); the structure, treatment and LANDFIRE cells are unrun.
>
> **Grid size (discussed 2026-11-14):** 480 m stays the analysis unit; 30 m layers are summarised up to it. `frp_fr` values depend on cell size (pixel density × share of the pixel in the cell), so finer grids are not comparable until note A is evaluated. A 240/360/480/960 m sensitivity on Aspen Acres was offered, not run.
>
> **Step 2 decisions (Max, 2026-11-14):**
> - An overpass is per satellite, split where the time gap exceeds `gap_min` (default 30 min); satellites are never merged (`timeutil.assign_overpass`). This is the unit for `first_detection_grid` and, later, `pointpattern`.
> - `gridded_frp` drops FRP ≤ 0 or missing by default (`drop_zero_frp`), as the potr5 notebook did before gridding. Footprints and first detection keep every detection.
>
> **Step 2 as built, where it goes beyond the plan text:**
> - `timeutil.py` (not in the plan layout) holds `to_local`, `day_of` and `assign_overpass`.
> - `utils/geo.py` has `snapped_grid` (moved from `ignitions.py`, unchanged) and `snap_origin`, which resolves None / `"landfire"` / raster. A snap raster sets the grid CRS; `res_m` must be a whole multiple of its pixel.
> - `gridded_frp` is vectorized (spatial-index pairs + `shapely.intersection`, no `gpd.overlay`); day/night stats are joined on `grid_id`. That equals potr5's positional assignment, because both tables there are sorted by the same cell set. Day/night columns use the same percentiles as the overall ones (potr5 had no p95 day/night). `daynight` is read by first letter, so FIRMS `D`/`N` and potr5 `Day`/`Night` both work.
> - `first_detection_grid` `"fraction"` unions an overpass's pieces only where their summed area could reach the threshold; single pieces use their own area.
> - `associate_detections` replaces the bbox `event_id` from `fetch_firms` and de-duplicates on `detection_id` first. A missing event end date leaves the window open. Event dates are read as UTC days.
> - `fetch_wfigs_perimeters(which=)` also takes `"year_to_date"`; all three service URLs were checked live. `fired_events` returns the file's CRS.
> - `to_raster` writes a datetime column as float days since `t0` (tag `T0`).
>
> **Step 1 decisions (Max, 2026-11-14):**
> - MODIS interscan: not screened for now; only VIIRS (187.5 m / 7 s). MODIS rows pass through with a printed count.
> - FIRMS interscan uses `cKDTree.query_pairs`, pairs resolved in time order (not the potr5 row loop). Chains of three or more can resolve differently from potr5.
> - `map_events` has the `n_workers` process pool.
> - `request_pause` stays 1.0 s (limit ~5,000 transactions / 10 min).
>
> **Step 1 as built, where it goes beyond the plan text:**
> - A cached day is final once its file was written ≥ `refetch_recent_days` after the day began (file mtime). Today's partial day is therefore always re-pulled, and so is a past day that was cached while still in progress.
> - Days outside a source's live `data_availability` range are skipped and not cached, so an SP day requested before publication is not frozen as empty. Costs one availability request per fetch that needs any data.
> - `apply_qa` order: low confidence → interscan → static (the plan's listed order). Order matters when a low-confidence detection is the higher-FRP member of a duplicate pair.
> - The MAP_KEY is redacted from every error message.
> - `firms_sources()` reads availability only when `map_key` is passed (not from `$FIRMS_MAP_KEY`), so it is offline by default.
> - `satellite` is the AspenAcres label (`VIIRS S-NPP`, …, `MODIS`); Terra and Aqua are pooled as in AspenAcres. Split them before any MODIS interscan work.

**Porting rule (applies throughout):** code from potr5-fb and AspenAcres is ported as-is in behaviour. Efficiency or method improvements are listed under "Improvement options — consult first" and are not applied without Max's OK. The only exception Max has already approved is the first-detection default (note B).

## Context
Active-fire detections (AFD) are handled ad hoc in two projects:

- **AspenAcres** — `projects/case_studies/AspenAcres/code/Python/`:
  - `firms_nrt.py`: FIRMS Area API, all sensors, unified schema.
  - `plotting.py`: Plotly and matplotlib progression animations.
  - No QA, footprints or derived metrics.
- **potr5-fb** — `projects/potr5-fb/code/Python/`:
  - `01b_VIIRS_earthaccess.ipynb`: earthaccess VNP14IMG/VJ114IMG swaths.
  - `01c_VIIRS-Grid.ipynb` and `__functions.py`: interscan duplicates, scan×track boxes, overlap-weighted FRP on a 480 m grid snapped to TreeMap.

fb_tools has no AFD code yet. The goal is a general **toolkit** — not a standardized report — for:
- incident and post-fire work
- near-real-time (NRT) assessment
- prescribed-fire and treatment monitoring

Plots are minimal and manuscript-style, and get tuned as we iterate.

## Decisions (with Max, 2026-11-13/14)

| Topic | Decision |
|---|---|
| Scope | Functions for access, QA, geometry, gridding, association, treatment summaries, and simple plots. No "report" object. |
| Access | **FIRMS API is primary** (NRT + SP). **earthaccess is minimal now**: search and read VNP14IMG / VJ114IMG / VJ214IMG fire pixels into the shared schema. Footprints come from **potr5's sample lookup table, ported now**, so swath detections work in every downstream tool. |
| Many perimeters | Every fetch accepts a set of events (geometry + per-event dates). Fetching **loops over fires one at a time** (no parallel API requests), with a pause between requests and a cache, to keep FIRMS requests manageable. Parallelism, if any, goes to **processing** fires once the data are gathered. |
| Footprint | potr5 method: axis-aligned scan×track box (FIRMS `scan`/`track`, km). |
| FRP per cell | **Keep the potr5 method until it is evaluated**: `frp_wkm2 = frp / pix_area`, `fraction = overlap / pix_area`, `frp_fr = frp_wkm2 × fraction`, summarised per cell. Alternatives are listed under "Improvement options" (note A). |
| QA | Drop by default: low confidence, interscan duplicates, non-vegetation `type`. Each can be switched off and reports what it dropped. |
| Interscan | VIIRS at 187.5 m / 7 s, keep max FRP. MODIS through the same function with its own radius if it stays simple; otherwise later. |
| Grid | Grid builder with `res_m` (default 480), `crs` (default EPSG:5070), and an **optional `snap`**: None, `"landfire"`, or any raster path (e.g. TreeMap). |
| First-detection grid | `rule` option; **default overlap fraction ≥ 0.5**. Also `"any"` and `"centre"`. See note B. |
| Sensors | VIIRS (SNPP / NOAA-20 / NOAA-21) by default for spatial products; others opt-in. A printable catalog shows what is available. FRP is never pooled across instruments. |
| Time | Stored in UTC. `to_local(gdf, tz)` converts. Day-binning functions take an optional `tz` (UTC days if omitted). |
| Cache | Caller passes `cache_dir`; otherwise `$FB_FIRMS_CACHE`; otherwise `./firms_cache` next to the running script/notebook. NRT and SP are separate sources. The last **1** UTC day of NRT is always re-pulled. |
| Growth metrics | **No footprint-union "growth".** Use point-pattern analysis of detection centres instead (`pointpattern.py`). It is an optional, separate toolset with its own calls, distinct from the gridded/footprint path. |
| Module names | Subpackage `active_fire`; perimeter/event code in `events.py`, not a second `perimeters.py`. Reuse fb_tools helpers wherever they exist. |
| Treatments | Per-polygon AFD summary for a TWIG/CFT treatment table, with FRP statistics (default mean). Treatments are assumed to predate the fire, so the window comes from the fire or detection set, not the treatment date. Treatment attributes (e.g. `YEAR_COMP`) are carried through. |
| Plotting | Simple static plots, plus a ported and streamlined version of the AspenAcres animations. |
| Demo | A minimalist notebook that exercises every public function, as for other modules. |

**Note A (2026-11-14).** The potr5 allocation is kept as-is for now. Two alternatives were discussed and are deferred until an evaluation compares them:
- `frp × overlap / cell_area`;
- area-proportional `frp × overlap / pix_area` (which conserves each detection's FRP across cells), then ÷ cell area for density.

**Note B.** potr5 did **not** use a 50% rule. `01c` set `first_obs_date` as the minimum date over every intersecting footprint piece (any overlap), and `overlap` (the sum of fractions) was carried only as a model covariate in `04a`. Also, a single 375 m box covers at most ~61% of a 480 m cell. The proposed default is therefore the **earliest overpass whose unioned boxes cover ≥ 50% of the cell**.

## Subpackage layout — `fb_tools/active_fire/`

### `firms.py` — FIRMS client (ported from AspenAcres `FIRMSAccess`, as functions)

**`firms_sources(map_key=None, *, print_table=True)`**
- Prints and returns the source catalog: name, platform, instrument, nominal pixel, NRT/SP.
- With a key, it adds live availability dates from `/data_availability`.

**`fetch_firms(aoi_or_events, start=None, end=None, *, sources=VIIRS_NRT, ...)`**
- Other arguments: `start_col=None, end_col=None, id_col=None, buffer_m=0, map_key=None, cache_dir=None, request_pause=1.0, refetch_recent_days=1`.
- Input is a bbox, one geometry, or an events GeoDataFrame with per-row dates.
- **Execution — a sequential loop, no parallel requests:**
  - Loop over fires one at a time. Each fire's buffered bbox × date window is split into ≤5-day windows (2 for GOES), per AspenAcres `DEFAULT_MAX_DAYS` / `SOURCE_MAX_DAYS`.
  - Sleep `request_pause` between requests (AspenAcres behaviour).
  - Skip windows already cached.
  - Print progress per fire (fire k of N, requests made vs served from cache).
  - Keep `_read_firms_csv`'s check for plain-text errors. A failed window is reported and skipped, as in AspenAcres; a simple retry is an improvement option.
- **Cache:**
  - One Parquet file per fire/bbox, source and UTC day, so file names encode their inputs.
  - Atomic `.part` → `replace` writes, as in `hrrr_uv10`.
  - An empty day writes an empty marker so it is not re-requested.
  - A re-run therefore only requests new or recent NRT days.
- **Return:** one GeoDataFrame, tagged with an `event_id` when events are given.

**`map_events(func, detections, events, *, id_col, n_workers=1, **kw)`** — processing, not fetching:
- Applies a per-fire function (e.g. footprints → `gridded_frp`) to each fire's detections and concatenates the results.
- `n_workers > 1` runs fires in parallel processes on data already gathered. Proposed; confirm at build time.

**`standardize_firms(raw)`**
- Builds the AspenAcres `STD_COLS` schema, and **adds** `scan`, `track`, `type`, `version`, `bright_ti4/ti5`, a `detection_id` and a `product` field.
- Normalizes confidence with `_norm_confidence`.
- Sets GOES −999 FRP to NaN.
- `acq_datetime` is tz-aware UTC.

MAP_KEY comes from the argument, then `$FIRMS_MAP_KEY`; never from a file.

### `swath.py` — minimal earthaccess path (port of potr5 `Access_VIIRS_AFD.create_fire_gdf`)

**`swath_products()`** prints the supported short names.

**`search_swaths(aoi_or_events, start=None, end=None, *, short_names=("VNP14IMG", "VJ114IMG", "VJ214IMG"), ...)`**
- Also takes `start_col` / `end_col`.
- Calls `earthaccess.login(persist=True)` once, then searches by polygon and window per event, deduplicating granules across events.

**`read_fire_pixels(granules, *, stream=True, cache_dir=None)`** — sequential over granules, as in potr5.
- Opens each granule with `xr.open_dataset(..., phony_dims="access")`.
- Reads `FP_longitude/latitude/power/T4/T5/sample/line` and the fire mask class (7/8/9 → l/n/h), plus `DayNightFlag` and platform.
- Takes time from the granule ID.
- Joins **`FP_sample` → the sample lookup table**, as potr5 does: `along_scan` and `along_track` (km), `scan_angle`, `pix_area`.
  - `scan`/`track` are filled from `along_scan`/`along_track`, so swath and FIRMS rows share **one schema** and feed the same footprint, grid and QA tools.
- **Lookup table ships as package data**: `fb_tools/data/viirs_pix_size_lut.csv`, copied from `potr5-fb/data/tabular/raw/pix_size_lut.csv` (6,400 rows: `sample, along_scan, along_track, scan_angle, pix_area`). `pyproject` already ships `data/*.csv`. Record its provenance (source or generating code) in the docstring; ask Max if it is unknown.
- **Swath interscan duplicates**: port potr5 `01c` `correct_interscan_duplicates` (Li et al. 2018) as-is:
  - group by `(satellite, granule_id)`
  - radius `along_track × 1000` m
  - exact `scan_angle` match
  - ≤ 8 s
  - keep max FRP
- Granule caching: port potr5's processed-granule log and per-granule outputs. Fix only its plain bugs (the `dataraw` global; the missing `continue` in the -geo variant, if ported).

Dependencies `earthaccess` and `h5netcdf` are imported inside functions and added to `environment.yml`. `import fb_tools` never needs them.

### `qa.py`
Every step returns `(kept, dropped)` and prints counts.

- **`apply_qa(gdf, low_conf=True, interscan=True, static=True)`** chains the three steps below.
- **`drop_low_confidence`**: drops VIIRS `l` and MODIS < 30.
- **`remove_interscan_duplicates(..., method="firms"|"swath", radius_m=..., time_tol_s=...)`**
  - `"firms"` is potr5 `correct_interscan_duplicates_firms` ported as-is: same scan/track, ≤ 7 s, radius (VIIRS 187.5 m; MODIS radius to confirm), keep max FRP.
    - `group_cols` stays an argument. potr5 grouped by `(satellite, fireid)`; before association the default is `(satellite, event_id)` when tagged, else `(satellite,)`.
  - `"swath"` is the `01c` version above.
  - The `method` default is chosen by `product`.
- **`drop_static_sources`**: keeps `type == 0`. NRT has no `type` column, so it prints that static screening was not possible.

### `footprints.py`
**`footprint_boxes(gdf, crs="EPSG:5070", instruments=("VIIRS",))`**
- Builds scan×1000 × track×1000 m boxes, ported from potr5 `pixel_area()`: axis-aligned, no rotation.
- Adds `pix_area_km2` and `frp_wkm2`.
- Works on FIRMS and swath rows alike.

### `grids.py`
- **`afd_grid(aoi, res_m=480, crs="EPSG:5070", snap=None)`**
  - `snap` is None (multiples of `res_m`), `"landfire"` (15 m mod 30), or a raster path (its pixel edges).
  - Promote `fuelscape/ignitions.py::_snapped_grid` to `utils/geo.py::snapped_grid` (behaviour unchanged, re-imported in `ignitions.py`) and extend it to take a snap raster.
- **`gridded_frp(boxes, grid, stats=(...))`**
  - Ports potr5 `aggregate_frp` as-is, including its `frp_fr` allocation (note A).
  - Per cell: count, unique days, overlap, `frp_fr` sum/max/min/mean/p90/p95/p97/p99/first, first/last obs, and a day/night split.
  - The positional `unstack` issue is listed under "Improvement options".
  - Returns a GeoDataFrame; `to_raster(...)` writes a GeoTIFF via `utils/io.py::raster_write_kwargs`.
- **`first_detection_grid(boxes, grid, rule="fraction", min_fraction=0.5)`** gives the earliest overpass time per cell. It is not interpolated.

### `pointpattern.py` — optional point-pattern toolset on detection centres

A separate set of small calls that treat detections as points (not footprints, not grids). Nothing else in the module depends on it.

Every call takes `step="overpass"|"day"` and an optional `tz`. Pooling across VIIRS satellites is a `by=` argument, not fixed.

**Candidate calls — pick and confirm at build time:**
- **Per step:**
  - `step_summary`: count and the centre of detections.
  - `hull_metrics`: convex-hull area of the centres.
  - `principal_axis`: PCA length and orientation.
  - `max_extent`: maximum pairwise distance.
- **Between consecutive steps:**
  - `centroid_track`: displacement distance and bearing.
  - `advance_distance`: for each new detection, the distance to the nearest earlier detection, summarised as median / p90 / max.
  - `reach`: the maximum distance from the first-detection location.

### `events.py` — perimeters and association (reusing fb_tools pieces)

- **`fetch_wfigs_perimeters(name=None, bbox=None, which="current"|"history")`**
  - Ports AspenAcres `get_feature_service_gdf`: paging by object ID, pgeojson → geojson fallback.
- **`fired_events(fired_fp, *, ids=None, bbox=None, years=None)`**
  - Uses a pyogrio `where=` read, following `spread/episodes.py::build_fast_fires_table`.
  - Returns geometry, `ig_date` and `last_date`.
- **`associate_detections(detections, events, *, buffer_m, start_col, end_col, ...)`**
  - Other arguments: `pad_before_days=0, pad_after_days=0, hull=False`.
  - A vectorized `sjoin` plus a per-event time filter.
  - A detection matching more than one event is kept in each and flagged `multi_event`.

### `treatments.py` — TWIG/CFT summaries

**`treatment_detections(treatments, detections, ...)`**
- Arguments:
  - `id_col="SOURCE_UID"`
  - `keep_cols=("ACTIVITY", "YEAR_COMP")`, carried through unchanged
  - `buffer_m=0`, `start=None`, `end=None`
  - `frp_stats=("mean",)`
- Column defaults match the harmonized output of `treatment_interactions` (CFT v2 + TWIG, EPSG:26913). Any table works through the column arguments.
- Treatments are assumed to predate the fire. The detections passed in, optionally limited by `start`/`end` (e.g. the fire window), define the period. Treatment dates do not set any window.
- **Returns, per polygon:** detected, n, first/last time, satellites, and FRP statistics per instrument.
- **Caveats to document (not to fix):**
  - Oversized CFT planning boundaries (`ac_flag`).
  - Double counting of CFT+TWIG overlaps when both raw sources are passed.
  - Pile vs broadcast uncertainty in TWIG's fallback fields.
- Inside-vs-outside comparisons are deferred.

### `plot.py` — minimal, manuscript-style
Few labels, no titles by default, matplotlib, returns `(fig, ax)`:
- `plot_detections(gdf, color="time"|"frp", events=None, ax=None)`
- `plot_overpass_series(gdf, value="count"|"frp_sum", by="satellite")`
- `plot_grid(grid_gdf, column)`
- `plot_geometry_metrics(df)`

### `animate.py` — ported AspenAcres animations, streamlined
- One shared frame builder, used by `animate_html` (Plotly MapLibre) and `animate_static` (matplotlib GIF/MP4).
- Removes the hard-coded "Aspen Acres Fire" title and "MT" labels; `tz` comes from the argument.
- Keeps the shadow/trail modes and the token-free Esri basemaps.
- Drops `frp_by_satellite` and `normalize_frp`.
- `plotly` and `imageio-ffmpeg` are imported inside functions and added to `environment.yml`.

### Exports
A subpackage `__init__` with `__all__`, plus a block in `fb_tools/__init__.py`, in the style of `weather/`.

Conventions: NumPy docstrings, `pathlib`, copy-not-mutate, `print` progress, deferred heavy imports.

## Suggested build order (next session)
1. Schema + `firms.py` (catalog, per-fire fetch loop, cache, `map_events`) + `qa.py`.
2. `events.py` association + `footprints.py` + `grids.py` (with the `snapped_grid` promotion).
3. `swath.py` (earthaccess + lookup table + swath interscan).
4. `treatments.py` + `pointpattern.py`.
5. `plot.py` + `animate.py`.
6. Demo notebook + `CLAUDE.md` section + memory update.

## Tests — `tests/test_active_fire.py` (offline; no network marker)

**FIRMS parsing and fetch**
- Synthetic FIRMS CSV for VIIRS, MODIS and GOES → schema, UTC times, confidence classes, −999 → NaN.
- `fetch_firms` with `requests.get` monkeypatched:
  - per-fire loop and 5-day / 2-day chunking
  - cached windows are not re-requested
  - NRT re-pull covers the last day only
  - atomic cache write and empty-day marker
  - MAP_KEY error
- `map_events` gives the same result at `n_workers=1` and `2`.

**QA**
- Interscan pairs inside/outside the radius and time tolerance; static `type` filter; NRT without `type` prints and keeps everything.

**Footprints and grids**
- Box area = scan × track.
- `gridded_frp`: reproduces the potr5 `frp_fr` values on a hand-built case (a box split across two cells).
- `first_detection_grid` under each rule on hand-built boxes.
- `afd_grid` snapping for None / `"landfire"` / raster; `test_ignition_density.py` still passes after the move.

**Point pattern**
- Each call on a constructed moving point cloud with known displacement, advance and reach.

**Treatments and events**
- `treatment_detections` on a burned and an unburned polygon; `keep_cols` carried through; FRP statistics.
- `associate_detections` multi-event flag.

**Swath reader**
- `read_fire_pixels` on a tiny synthetic netCDF built in `tmp_path` with the VNP14IMG variable names (`pytest.importorskip("h5netcdf")`).
- Lookup-table join reproduces `pix_area` for known samples.
- Swath interscan duplicates on constructed pairs.

## Verification
1. `pytest tests/test_active_fire.py tests/test_ignition_density.py`, then the full suite, in env `fb_tools`.
2. Demo notebook `code/notebooks/01g_ActiveFire.ipynb`, minimal, one cell group per function, on Aspen Acres:
   - WFIGS perimeter → `firms_sources` → `fetch_firms` → `apply_qa` → footprints → `gridded_frp` / `first_detection_grid` → geometry metrics → plots → `animate_static`.
   - A short multi-event fetch over two to three FIRED fires (sequential loop; a second run is served from the cache), then `map_events` over them.
   - `treatment_detections` on a handful of CFT/TWIG burn polygons.
   - `search_swaths` / `read_fire_pixels` for one day.
   - Max runs it with his own `$FIRMS_MAP_KEY` and Earthdata login.
   - Check counts against the AspenAcres snapshots.

## Open at build time
- MODIS interscan radius.
- Which point-pattern calls to keep.
- Lookup-table provenance.
- `request_pause` default and FIRMS rate limits: check the API docs.

## Improvement options — consult first (not applied in the port)
- **FRP allocation (note A):** evaluate the potr5 `frp_fr` against `frp × overlap / cell_area` and area-proportional allocation before switching.
- **Shared requests across fires:** fixed lon/lat tiles × UTC days would let overlapping fires reuse cached data. Consider it only if per-fire looping proves wasteful.
- **FIRMS interscan:** replace the `iterrows` + per-row KD query loop with `cKDTree.query_pairs` per group (same result, much faster on large sets).
- **Swath interscan:**
  - Grouping by `granule_id` misses duplicates across granule edges.
  - The docstring mentions `angle_tol` and "average FRP" but the code uses an exact angle match and max FRP. Reconcile the two.
- **Footprints:** boxes are axis-aligned in EPSG:5070, not rotated to the scan direction.
- **`gridded_frp`:** the potr5 day/night stats are assigned by position after `unstack`. Merging on keys is a bug fix, flagged here because it changes output if the order ever differed.
- **potr5 date windows:** `01c` effectively applied ±28 d (a ±14 d buffer twice); `01b` set `last_date` = `ig_date`. Note these as bugs; the port takes windows as explicit arguments.
- **Swath reading:** parallel granule reads, and streaming vs download. Keep the potr5 behaviour (sequential, streamed) as the default.
- **FIRMS fetch:** retry with backoff on a failed window (AspenAcres only logs and skips).
- **Lookup table:** could be derived analytically from the VIIRS aggregation-zone geometry instead of shipped as a CSV.

## Next round (logged)
- Cross-sensor FRP harmonization.
- Inside-vs-outside treatment comparisons.
- Fuel joins for FRP modelling.
- Space-time clustering of detections into events.
- Parks 2014-style interpolated progression.
- FRE.

## Candidate fire pixels (deferred, scoped 2026-11-15)
Source: Coffield et al. (2026), *Remote Sensing of Environment* 334, 115156, doi:10.1016/j.rse.2025.115156. Scoped for the potr5-fb revision and set aside so the revision could go out; kept here because it depends on the deferred `swath.py`.

- **What they are:** pixels the VIIRS 375 m algorithm flags as "potential background fire" or "candidate fire" (L2 `algorithm QA` flags 8 and 10) but does not classify as low/nominal/high. Confirm the exact bit test against the paper's code before use.
- **Reported effect (95 western US fires > 25 km², 2020):** +45% daytime and +12% nighttime detections, +20% persistence, +6.5% FRP (Cameron Peak 9.6%, East Troublesome 11.4%, Pine Gulch 10.6%, Mullen 12.2%). Gains are largest under smoke and cloud and at moderate-to-high tree cover.
- **Validity limit:** only supported inside known large-fire perimeters (FEDS perimeter + 2 km). Outside that, expect false positives from hot land surfaces. Do not apply to small fires without new validation.
- **Data needs beyond FIRMS (which does not carry them):**
  - Full 2-D `algorithm QA` from VNP14IMG / VJ114IMG, not just the sparse `FP_*` list `swath.py` is specified to read.
  - Geolocation from VNP03IMG / VJ103IMG for the flagged non-fire pixels.
  - For FRP: 750 m M13 radiance (VNP02MOD / VJ102MOD), background radiance borrowed from the nearest known fire pixel in the same swath; negative values set to zero.
- **Released code and data:** `github.com/scoffiel/candidate_fires`; doi:10.13016/evnn-j0zz (2020 western US).
  - `joined_dets_2020.parquet` (42.5 MB, CC0): known + candidate detections with lon/lat, UTC time, `fire_mask`, `confidence` (`x` = candidate), sample index `j`, `frp` (their estimate, candidates included), `frp_old`, FEDS `fireid`, MTBS `name` for the 20 largest fires (includes Cameron Peak and East Troublesome). A pilot on those fires needs no new extraction.
  - Their extraction keeps pixels with `algorithm QA` bit 8 or 10 set (or fire mask > 6), reads lat/lon from 03IMG `geolocation_data`, and skips granules with no known detections.
- **Faster swath design than potr5 `01b_VIIRS_earthaccess-geo.ipynb`** (which reads full 03IMG arrays per granule and matches FRP with a per-row KD-tree loop): read L2 first, take flagged line/sample indices, read only that row window from 03IMG, join FRP on `FP_line`/`FP_sample`, run granules in parallel. Not yet timed.
- **Pilot design (potr5-fb, not run):**
  - Fires: 2020 Southern Rockies fires > 25 km² in the model sample (Cameron Peak, East Troublesome, Pine Gulch, Mullen, others to confirm).
  - Grid candidates on the existing 480 m grid with the same footprint logic, tagged known vs candidate.
  - Report per cell: change in overlap (exposure), unique days, day proportion, and cells gained that had no detection, each by forest type, aspen share and canopy cover. The question is whether the gain differs by forest type.
  - Screen: share of candidates corroborated by a known or candidate pixel in the adjacent overpass (500 m, 60 min).
- **Possible API:** `read_fire_pixels(..., include_candidates=False)` returning a `detection_class` column (`known` / `candidate`), with `footprint_boxes` and `gridded_frp` unchanged.

## Aside
`AspenAcres/code/Python/FIRMS-NRT_animation.ipynb` cell 1 holds a plain-text FIRMS MAP_KEY. Remove it and rotate the key.
