"""
TEALOM Python Class
"""

import os, sys

import pandas as pd
import xarray as xr
import numpy as np

from pathlib import Path
from os.path import join as pjoin
from os.path import basename

# import the __functions.py (helper functions)
sys.path.append(os.getcwd()) # add code folder to system path
from __functions import *  # imports all helper functions


class TEALOM:
    """
    IN DEVELOPMENT

    Treatment Evaluation and Landscape-scale Outcomes Monitoring (TEALOM)
    ~ Colorado Forest Restoration Institute (CFRI)
    ~ Dept. of Forest and Rangeland Stewardship, Colorado State University
    ~ Author: Maxwell.Cook@colostate.edu

    TEALOM extends a classic RADS/QWRA (GTR-315) by assessing the potential outcomes \
    of planned or completed treatments across a landscape specific to Highly Valued \
    Resources and Assets (HVRAs) and assigned hazard/risk to those assets. \
    Outputs include Delta NVC (expected or conditional), fire behavior change, and SDI
    """
    DEFAULT_METRICS = ['eNVC', 'Flamelength', 'Feas', 'Cost', 'SDI']
    PATTERNS = { # file naming patterns
        "eNVC": re.compile(r"envc", re.I),
        "cNVC": re.compile(r"cnvc", re.I),
        "Feas": re.compile(r"feas", re.I),
        "Cost": re.compile(r"cost", re.I),
        "SDI": re.compile(r"sdi90", re.I),
        "Flamelength": re.compile(r"flamelength", re.I),
        "Crownstate": re.compile(r"crownstate", re.I),
        "Spreadrate": re.compile(r"spreadrate", re.I),
    } # can add to these based on standard QWRA outputs/file formats

    def __init__(self, treatments: gpd.GeoDataFrame, type_col: str, id_col: str,
                 qwra_directory: str, results_directory: str,
                 hvra_categories: list, hvra_lut=None, metrics=None,
                 landscape_units: dict = None, nvc_type: str = 'eNVC',
                 feasibility: str = None,
                 hdist: str = None, nb: str = None, bps: str = None,
                 proj_crs='EPSG:5070', save_outputs=False):

        self.trts = treatments # input completed/planned/hypothetical fuels treatments
        self.id_col = id_col # the unique treatment identifier
        self.type_col = type_col  # treatment type/scenario column
        self.qwradir = Path(qwra_directory) # main directory with input rasters from the QWRA (e.g., 'Geodatabase')
        self.hvras = hvra_categories # what HVRA categories are being considered?
        self.hvra_lut = hvra_lut  # HVRA lookup table (for different file naming conventions)
        self.metrics = metrics or TEALOM.DEFAULT_METRICS  # input metrics to consider
        self.landscape_units = landscape_units # a dictionary with format {name, geodataframe}
        self.region_cols = self.landscape_units.keys() # dictionary keys should be the unit name
        self.nvc_type = nvc_type # optional specify either eNVC or cNVC
        self.feasibility = feasibility # use-defined composite feasibility
        self.hdist = hdist  # use-defined historic disturbance raster
        self.nb = nb # non-burnable mask
        self.bps = bps # constrained to BPS, use-defined
        self.crs = proj_crs # projection
        self.results_dir = results_directory
        os.makedirs(self.results_dir, exist_ok=True)
        self.save_outputs = save_outputs

        # standardized treatment scenarios
        self.scenarios = self.trts[self.type_col].unique().tolist()
        self.scenarios = [(k, str(k).lower()) for k in sorted(self.scenarios, key=len, reverse=True)]

        # search the QWRA directory and build dictionary
        self.dd = self._parse_inputs()

    @staticmethod
    def _geom_to_raster_crs(zones_gdf, raster_fp):
        """
        Reprojects a geometry to a raster CRS given a file path

        :param zones_gdf:
        :param raster_fp:
        :return:
        """
        with rxr.open_rasterio(raster_fp, masked=True) as da:
            r_crs = da.rio.crs
        if zones_gdf.crs != r_crs:
            return zones_gdf.to_crs(r_crs)
        return zones_gdf

    @staticmethod
    def _bin_raster(arr: str, bins: list):
        """

        :param arr:
        :param bins:
        :return:
        """
        with rxr.open_rasterio(arr, masked=True) as da:
            tr_arr = da.squeeze().values.astype("float32")
            tr_nd = da.rio.nodata
            tr_trf = da.rio.transform()
        if tr_nd is not None:
            tr_arr = np.where(tr_arr == tr_nd, np.nan, tr_arr)
        tr_binned = np.where(
            np.isnan(tr_arr), -9999,
            np.digitize(tr_arr, bins=bins)
        ).astype("int16")
        del tr_arr

        return tr_binned, tr_trf, tr_nd

    @staticmethod
    def _normalize(s: str) -> str:
        """Normalize strings for fuzzy filename/scenario matching."""
        return re.sub(r"[-_]", "", s).lower()

    @staticmethod
    def _mask_raster(raster_path, geom, nodata_val=None):
        with rio.open(raster_path) as src:
            out_image, _ = rio_mask(src, [geom], crop=True)
            arr = out_image[0]
            if nodata_val is not None:
                arr = np.ma.masked_equal(arr, nodata_val)
                arr = arr.filled(np.nan)
            return arr

    @staticmethod
    def _parse_fb_pct(fps: list, pcts: list):
        # locate one baseline raster per percentile
        bl_files = {
            pct: next(
                (f for f in fps
                 if "baseline" in f.lower() and pct.lower() in Path(f).stem.lower()),
                None
            )
            for pct in pcts
        }
        tr_files = [f for f in fps if "treated" in f.lower()]

        return bl_files, tr_files

    def _parse_inputs(self):
        """
        Create an analysis dictionary for pre- and post-treatment layers
        :return: Dictionary with keys=outcomes, values=raster list
        """
        inputs = {k: [] for k in TEALOM.PATTERNS.keys()}  # the data inputs / analysis dictionary
        for p in self.qwradir.rglob("*.tif"):
            name = p.name.lower() # file basename
            full = p.as_posix().lower() # full path
            if "maps" in full:
                continue # skips non-spatial tifs, need to improve this
            for metric, rgx in TEALOM.PATTERNS.items():
                if self.metrics is not None and metric not in self.metrics:
                    continue # skip metrics if needed (user input or default)
                if rgx.search(name):
                    inputs[metric].append(str(p)) # append the tif file
                    break  # keeps the first match
        # drop empty keys
        inputs = {k: v for k, v in inputs.items() if v}

        return inputs

    def _parse_hvra(self, hvra: str):
        """
        Parse an optional HVRA lookup table
        :return:
        """
        if self.hvra_lut is not None:
            return self.hvra_lut.loc[self.hvra_lut['Category'] == hvra, 'Category_num'].values[0]
        else:
            return hvra  # need a better way to handle this

    def parse_feasibility(self, fps: list):
        """
        Return a matched feasibility scenarios dictionary
        :param fps: file paths
        :return:
        """

        def _match_scenario(fpath: str):
            stem = Path(fpath).stem.lower()
            # Sort longest first so mechRxFire is checked before mech
            for orig, low in sorted(self.scenarios, key=lambda x: len(x[1]), reverse=True):
                # Require delimiter after prefix to avoid mech matching mechRxFire
                if stem.startswith(low + "_") or stem == low:
                    return orig
            return None

        feas_files = fps  # gather the feasibility rasters
        scenario_feas = {}
        for fp in feas_files:
            sc = _match_scenario(fp)
            if sc is None:
                continue
            scenario_feas.setdefault(sc, fp)
        if not scenario_feas:
            raise ValueError("! Could not match any feasibility rasters to treatment scenario keys.")

        return scenario_feas

    def project_nvc(self):
        """
        Calculate the pre/post NVC at the project (treatment) scale
        """
        # Find the correct NVC files (geotiffs) from the data dictionary
        envc_files = self.dd[self.nvc_type]
        print(f"\n{self.nvc_type} files: {len(envc_files)}\n")

        # Reproject the treatment data once
        trts_base = TEALOM._geom_to_raster_crs(self.trts, envc_files[0])
        trts_base['ACRES_GIS'] = trts_base.geometry.area * 0.000247105

        hvra_nvcs = [] # stores the project-scale NVC metrics
        for hvra in self.hvras:
            hvra_cat = self._parse_hvra(hvra) # uses a lookup table if needed
            print(f"\tCalculating project-scale delta eNVC for HVRA:{hvra}")

            # --- Find the baseline NVC file
            cat_bl = "Composite" if hvra in ("Total", "Composite") else hvra # for composite/total
            bl_file = [f for f in envc_files if 'baseline' in f.lower()
                       and cat_bl.lower() in basename(f).lower()][0] # the baseline raster file path (e.g., Water_eNVC)
            print(f"\t\tBaseline file: {bl_file}")

            # --- Run the baseline zonal statistics for this HVRA
            bl_zs = compute_band_stats(
                trts_base, bl_file,
                id_col=self.id_col,
                attr=f"{self.nvc_type}_bl", # column naming convention
                stats=['sum','mean','count'], # sum and mean NVC also count
                ztype="continuous" # continuous or categorical
            )
            bl_zs['HVRA'] = hvra # assign the HVRA category
            bl_zs.columns = [c.replace("_count", "_npix") for c in bl_zs.columns]

            # --- Find the treated files paths
            treated_files = [
                f for f in envc_files if 'treated' in f.lower()
                and hvra_cat.lower() in pjoin(f).lower()
            ] # returns treated file paths (need to work on this)
            print(f"\tTreated files: {treated_files}")

            # --- Run the treated zonal statistics
            treated_envc = [] # to store results by treatment scenario
            for treated_fp in treated_files:
                stem = Path(treated_fp).stem.lower()
                scenario = next((orig for orig, low in self.scenarios if stem.startswith(low + "_")), None)
                if scenario is None:
                    continue
                print(f"\t\tTreated file: {basename(treated_fp)}")

                # filter treatments to the current type
                trts = trts_base[trts_base[self.type_col] == scenario]
                trts = trts[[self.id_col, self.type_col, 'ACRES_GIS', 'geometry']]
                # run the zonal stats
                trt_zs = compute_band_stats(
                    trts, treated_fp,
                    id_col=self.id_col,
                    attr=f'{self.nvc_type}_tr',
                    stats=['sum','mean'],
                    ztype="continuous"
                )

                # force numeric dtype for stats (keeps NaN if missing)
                for c in [f"{self.nvc_type}_tr_sum", f"{self.nvc_type}_tr_mean"]:
                    if c in trt_zs.columns:
                        trt_zs[c] = pd.to_numeric(trt_zs[c], errors="coerce")
                    else:
                        trt_zs[c] = np.nan

                trt_zs['HVRA'] = hvra
                trt_zs[self.type_col] = scenario
                treated_envc.append(trt_zs) # append to output list

            # --- Safely concatenate treatment scenarios for this HVRA
            treated_envc = pd.concat(treated_envc, ignore_index=True)

            # --- Merge baseline + treated stats WORK HERE
            envc_stats = pd.merge(
                bl_zs, treated_envc,
                on=[self.id_col, 'HVRA'],  # include scenario
                how="left", validate='one_to_many'
            )

            # --- compute deltas (sum & mean)
            envc_stats[f'{self.nvc_type}_dlt_sum'] = (envc_stats[f"{self.nvc_type}_tr_sum"] -
                                                      envc_stats[f"{self.nvc_type}_bl_sum"])
            envc_stats[f'{self.nvc_type}_dlt_mean'] = (envc_stats[f"{self.nvc_type}_tr_mean"] -
                                                       envc_stats[f"{self.nvc_type}_bl_mean"])
            envc_stats[f'{self.nvc_type}_pct_chg'] = (
                (envc_stats[f'{self.nvc_type}_dlt_sum'] / envc_stats[f'{self.nvc_type}_bl_sum'].abs()) * 100
            )
            hvra_nvcs.append(envc_stats) # append the HVRA results out

        # --- Concatenate the HVRA-scenario-specific outputs
        nvc_project_df = pd.concat(hvra_nvcs, ignore_index=True)

        # --- Merge back to the treatment data
        nvc_project_gdf = pd.merge(self.trts, nvc_project_df, on=[self.id_col, self.type_col], how="left")

        if self.save_outputs is True:
            out_fp = pjoin(self.results_dir, f'{self.nvc_type}_project.gpkg')
            nvc_project_gdf.to_file(out_fp)  # spatial data
            # tabular data
            out_fp = pjoin(self.results_dir, f'{self.nvc_type}_project.csv')
            nvc_project_df = nvc_project_df.merge(self.trts[[self.id_col,'ACRES_GIS']])
            nvc_project_df.to_csv(out_fp)


        return nvc_project_df, nvc_project_gdf


    def landscape_nvc(self, region_col: str,
                      units: list or dict,
                      majority: bool):
        """
        Calculate the landscape NVC metrics
        :param units:
        :param region_col:
        :param majority:
        :return:
        """

        nvc_files = self.dd[f'{self.nvc_type}']
        feas_by_scenario = self.parse_feasibility(self.dd['Feas']) # match feasibility to scenarios

        # Grab a reference raster to match
        ref_da = rxr.open_rasterio(nvc_files[0]).squeeze()
        r_crs = ref_da.rio.crs

        # --- Load and align hdist mask once, if provided
        if self.hdist is not None:
            print("\nMasking out previously disturbed lands")
            hdist_da = rxr.open_rasterio(self.hdist).squeeze().rio.reproject_match(ref_da)
            # Guard against nodata bleed-through
            nd = hdist_da.rio.nodata
            if nd is not None:
                hdist_da = hdist_da.where(hdist_da != nd)
            undisturbed = (hdist_da == 1) # True where NOT disturbed -> keep
        else:
            undisturbed = None

        # --- Load and align non-burnable mask once, if provided
        if self.nb is not None:
            print("Masking out non-burnable fuel types")
            nb_da = rxr.open_rasterio(self.nb).squeeze().rio.reproject_match(ref_da)
            nd = nb_da.rio.nodata
            if nd is not None:
                nb_da = nb_da.where(nb_da != nd)
            burnable = (nb_da == 1)  # True where burnable -> keep
        else:
            burnable = None

        # --- Load and align non-burnable mask once, if provided
        if self.bps is not None:
            print("Masking out BPS codes")
            bps_da = rxr.open_rasterio(self.bps).squeeze().rio.reproject_match(ref_da)
            nd = bps_da.rio.nodata
            if nd is not None:
                bps_da = bps_da.where(bps_da != nd)
            bps = (bps_da == 1)  # True where burnable -> keep
        else:
            bps = None

        # --- Combine masks into a single treatment mask
        masks = [m for m in [undisturbed, burnable, bps] if m is not None]
        if masks:
            treatment_mask = masks[0]
            for m in masks[1:]:
                treatment_mask = treatment_mask & m
        else:
            treatment_mask = None

        # --- Pre-load the feasibility rasters by scenario
        print(f"\nAligning feasibility rasters to NVC")
        feas_aligned = {}
        for scenario, feas_fp in feas_by_scenario.items():
            with rxr.open_rasterio(feas_fp).squeeze() as da:
                aligned = da.rio.reproject_match(ref_da)
                if treatment_mask is not None:
                    aligned = aligned.where(treatment_mask==1)  # zero out disturbed pixels
                feas_aligned[scenario] = aligned

        # --- Create/load the composite feasibility
        if self.feasibility is not None:
            print(f"\nLoading composite feasibility")
            feas_composite = rxr.open_rasterio(self.feasibility).squeeze()
            feas_composite = feas_composite.rio.reproject_match(ref_da)
            if treatment_mask is not None:
                feas_composite = feas_composite.where(treatment_mask==1)
        else:
            count = None
            for da in feas_aligned.values():  # already masked above
                mi = (da >= 1).astype("int16")
                count = mi if count is None else (count + mi)
            feas_composite = (count >= 3).astype("int16")
            if treatment_mask is not None:
                feas_composite = feas_composite.where(treatment_mask==1)
            feas_composite.rio.to_raster('feas_composite.tif')
        del ref_da

        # ---- Reproject units once to match raster CRS
        print("Aligning landscape units CRS")
        units = {k: v.to_crs(r_crs) for k, v in units.items()}

        # --- Map over landscape units, build the dataframe
        landscape_nvc = []
        for hvra in self.hvras:
            cat_bl = "Composite" if hvra in ("Total", "Composite") else hvra
            try:
                bl_file = [f for f in nvc_files
                           if "baseline" in f.lower() and cat_bl.lower() in basename(f).lower()][0]
            except Exception as e:
                print(e)
                continue

            # --- Open the baseline NVC raster for this HVRA
            print(f"\tCalculating landscape {self.nvc_type} for HVRA:{hvra}\n\t{bl_file}")
            bl_da = rxr.open_rasterio(bl_file).squeeze()
            r_crs, nd, trf = bl_da.rio.crs, bl_da.rio.nodata, bl_da.rio.transform()
            if nd is not None and np.isnan(nd):
                bl_da = bl_da.astype("float32")

            # --- Apply feasibility mask(s) to the baseline NVC
            bl_feas = bl_da.where(feas_composite == 1) # composite feasibility

            for unit_type, gdf in units.items():
                # print(f"\t{unit_type}: {len(gdf)} units")
                if region_col not in gdf.columns:
                    raise KeyError(f"{unit_type} units missing required column: {region_col}")

                # --- Calculate the total baseline NVC, regardless of scenario
                total_nvc = compute_band_stats(
                    gdf, bl_file,
                    id_col=region_col,
                    attr=f'{self.nvc_type}_total',
                    stats=["sum","count"],
                    ztype="continuous"
                )
                total_nvc.columns = [c.replace("_sum", "") for c in total_nvc.columns]
                total_nvc.columns = [c.replace("_count", "_npix") for c in total_nvc.columns]

                # --- Baseline feasibility
                feas_nvc = compute_band_stats(
                    gdf, bl_feas.data,
                    id_col=region_col,
                    attr=f"{self.nvc_type}_feas",
                    stats=['sum', 'count'],
                    ztype="continuous",
                    nodata=nd, transform=trf
                )
                feas_nvc.columns = [c.replace("_sum", "") for c in feas_nvc.columns]
                feas_nvc.columns = [c.replace("_count", "_npix") for c in feas_nvc.columns]

                # --- Scenario-specific feasible NVC
                for scenario, feas_da in feas_aligned.items():
                    # Mask the scenario feasibility
                    bl_feas_scenario = bl_da.where(feas_da == 1)
                    # print(f"\t\tScenario:{scenario}")
                    # run zonal stats
                    scenario_nvc = compute_band_stats(
                        gdf, bl_feas_scenario.data,
                        id_col=region_col,
                        attr=f"{self.nvc_type}_feas_scen",
                        stats=["sum", "count"],
                        ztype="continuous",
                        nodata=nd, transform=trf
                    )
                    scenario_nvc.columns = [c.replace("_sum", "") for c in scenario_nvc.columns]
                    scenario_nvc.columns = [c.replace("_count", "_npix") for c in scenario_nvc.columns]

                    nvc_sums = (total_nvc
                          .merge(scenario_nvc, on=region_col, how="left")
                          .merge(feas_nvc, on=region_col, how="left"))

                    nvc_sums['UNIT_NAME'] = unit_type
                    nvc_sums['UNIT_ID'] = nvc_sums[region_col]
                    nvc_sums['HVRA'] = hvra
                    nvc_sums['scenario'] = scenario

                    landscape_nvc.append(nvc_sums)

        landscape_nvc = pd.concat(landscape_nvc, ignore_index=True)

        # # convert number of pixels to acres
        landscape_nvc['ACRES_total'] = landscape_nvc['eNVC_total_npix'] * (30 * 30) / 4046.86
        landscape_nvc['ACRES_feas'] = landscape_nvc['eNVC_feas_npix'] * (30 * 30) / 4046.86
        landscape_nvc['ACRES_feas_scenario'] = landscape_nvc['eNVC_feas_scen_npix'] * (30 * 30) / 4046.86
        landscape_nvc.drop(columns=['eNVC_total_npix','eNVC_feas_npix','eNVC_feas_scen_npix'], inplace=True)

        if self.save_outputs is True:
            out_fp = pjoin(self.results_dir, f'{self.nvc_type}_landscape.csv')
            landscape_nvc.to_csv(out_fp)

        return landscape_nvc


    def cost_benefit(self):
        """
        Calculated the expected cost of treatment
        :return:
        """
        cost_files = self.dd.get('Cost', [])
        if not cost_files: raise ValueError("No Cost rasters found in the data dictionary.")


    def fl_change(self,
                    percentiles=("Pct25", "Pct50", "Pct90", "Pct97"),
                    fl_bins=None,
                    fl_labels=None):
        """
        Calculate the pre-post flame length (binned)

        :param percentiles:
        :param fl_bins:
        :param fl_labels:
        :return:
        """

        # --- Define the flame length bins
        fl_bins = fl_bins or [0, 2, 4, 8, 12, np.inf]
        fl_labels = fl_labels or ["0-2ft", "2-4ft", "4-8ft", "8-12ft", ">12ft"]
        bin_edges = fl_bins[1:-1]  # interior edges for np.digitize -> [2, 4, 8, 12]
        bin_map = dict(enumerate(fl_labels))  # {0:"0-2ft", 1:"2-4ft", ...}

        # --- Get the flame length raster files
        fl_files = self.dd.get("Flamelength", [])
        if not fl_files: raise ValueError("No Flamelength rasters found in the data dictionary.")

        # --- Locate baseline and treated rasters
        bl_files, tr_files = TEALOM._parse_fb_pct(fl_files, percentiles)

        # --- Reproject treatments to raster CRS once
        trts_bl = TEALOM._geom_to_raster_crs(self.trts, fl_files[0])

        # --- Run the zonal statistics
        fl_results = []
        for pct in percentiles:
            base_fp = bl_files.get(pct) # get base file for this percentile
            if base_fp is None:
                print(f"\t! No baseline FL raster found for {pct}, skipping")
                continue
            print(f"\nFlame length change [{pct}]")

            # --- Baseline binned array -> categorical zonal stats
            bl_binned, bl_trf, bl_nd = TEALOM._bin_raster(base_fp, bin_edges)
            # --- Compute band stats (categorical, percentages)
            bl_cat = compute_band_stats(
                trts_bl, bl_binned, id_col=self.id_col,
                attr="FL_bl", ztype="categorical",
                transform=bl_trf, nodata=-9999
            )
            bl_cat["FL_class"] = bl_cat["FL_bl"].map(bin_map)
            bl_cat.rename(columns={'FL_bl_pct_cover': 'FL_bl_pct'}, inplace=True)
            bl_cat["percentile"] = pct

            # --- scenario-specific treated rasters
            fl_scens = []
            for fp in tr_files:
                stem = Path(fp).stem.lower()
                if pct.lower() not in stem:
                    continue
                scenario = next(
                    (orig for orig, low in sorted(self.scenarios, key=lambda x: len(x[1]), reverse=True)
                     if TEALOM._normalize(stem).startswith(TEALOM._normalize(low))), None
                )
                if scenario is None:
                    continue
                print(f"\t  Scenario: {scenario}  |  {Path(fp).name}")

                # --- Filter treatments to the current scenario
                trts_sc = trts_bl[trts_bl[self.type_col] == scenario].copy()
                if trts_sc.empty:
                    continue

                # treated binned array -> categorical stats
                tr_binned, tr_trf, tr_nd = TEALOM._bin_raster(fp, bin_edges)
                tr_cat = compute_band_stats(
                    trts_sc, tr_binned, id_col=self.id_col,
                    attr="FL_tr", ztype="categorical",
                    transform=tr_trf, nodata=-9999
                )
                tr_cat["FL_class"] = tr_cat["FL_tr"].map(bin_map)
                tr_cat.rename(columns={'FL_tr_pct_cover': 'FL_tr_pct'}, inplace=True)
                tr_cat["percentile"] = pct
                tr_cat[self.type_col] = scenario
                fl_scens.append(tr_cat)

            # concatenate the scenarios
            fl_scens = pd.concat(fl_scens, ignore_index=True)
            # merge with baseline stats
            fl_cat = pd.merge(
                bl_cat, fl_scens,
                on=[self.id_col, "FL_class", "percentile"],
                how="outer")
            fl_cat['delta_pct'] = fl_cat["FL_tr_pct"] - fl_cat["FL_bl_pct"]
            fl_results.append(fl_cat)

        fl_cat = pd.concat(fl_results, ignore_index=True) if fl_results else pd.DataFrame()
        fl_cat["metric"] = "Flamelength"

        if self.save_outputs:
            fl_cat.to_csv(pjoin(self.results_dir, "fl_change_categorical.csv"), index=False)
        del bl_binned, bl_cat # clean up

        return fl_cat


    def crownstate_change(self,
                          percentiles=("Pct25", "Pct50", "Pct90", "Pct97"),
                          cs_labels: list = None):
        """

        :param percentiles:
        :param cs_labels:
        :return:
        """

        cs_labels = cs_labels or ['NB','Surface','Passive','Active']
        bin_map = dict(enumerate(cs_labels))

        cs_files = self.dd.get('Crownstate', [])
        if not cs_files: raise ValueError("No Crownstate rasters found in the data dictionary.")

        # --- Parse baseline and treated
        bl_files, tr_files = TEALOM._parse_fb_pct(cs_files, percentiles)

        # --- Reproject treatments to raster CRS once
        trts_bl = TEALOM._geom_to_raster_crs(self.trts, cs_files[0])

        cs_results = []
        for pct in percentiles:
            base_fp = bl_files.get(pct)
            if base_fp is None:
                print(f"  ! No baseline FL raster found for {pct}, skipping")
                continue
            print(f"\n  Crown state change [{pct}]")

            # --- Calculate the baseline crown state percentages
            bl_cat = compute_band_stats(
                trts_bl, base_fp, id_col=self.id_col,
                attr="CS_bl", ztype="categorical"
            )
            bl_cat["CS_class"] = bl_cat["CS_bl"].map(bin_map)
            bl_cat.rename(columns={'CS_bl_pct_cover': 'CS_bl_pct'}, inplace=True)
            bl_cat["percentile"] = pct

            cs_scens = []
            for fp in tr_files:
                stem = Path(fp).stem.lower()
                if pct.lower() not in stem:
                    continue
                scenario = next(
                    (orig for orig, low in sorted(self.scenarios, key=lambda x: len(x[1]), reverse=True)
                     if TEALOM._normalize(stem).startswith(TEALOM._normalize(low))),
                    None
                )
                if scenario is None:
                    continue

                subset = self.trts[self.trts[self.type_col] == scenario].copy()
                if subset.empty:
                    continue
                subset = TEALOM._geom_to_raster_crs(subset, base_fp)
                print(f"\t  Scenario: {scenario}  |  {Path(fp).name}")

                # treated binned array -> categorical stats
                tr_cat = compute_band_stats(
                    subset, fp, id_col=self.id_col,
                    attr="CS_tr", ztype="categorical",
                )
                tr_cat["CS_class"] = tr_cat["CS_tr"].map(bin_map)
                tr_cat.rename(columns={'CS_tr_pct_cover': 'CS_tr_pct'}, inplace=True)
                tr_cat["percentile"] = pct
                tr_cat[self.type_col] = scenario

                # --- Filter baseline to this scenario's IDs at merge time
                scenario_ids = subset[self.id_col]
                bl_sc = bl_cat[bl_cat[self.id_col].isin(scenario_ids)]

                cs_merge = pd.merge(
                    bl_sc, tr_cat,
                    on=[self.id_col, "CS_class", "percentile"],
                    how="outer"
                )
                cs_merge[self.type_col] = cs_merge[self.type_col].fillna(scenario)
                cs_merge["CS_bl_pct"] = cs_merge["CS_bl_pct"].fillna(0)
                cs_merge["CS_tr_pct"] = cs_merge["CS_tr_pct"].fillna(0)
                cs_merge["delta_pct"] = cs_merge["CS_tr_pct"] - cs_merge["CS_bl_pct"]
                cs_scens.append(cs_merge)
                del cs_merge

            if cs_scens:
                cs_results.append(pd.concat(cs_scens, ignore_index=True))

        cs = pd.concat(cs_results, ignore_index=True) if cs_results else pd.DataFrame()
        cs["metric"] = "Crownstate"

        if self.save_outputs:
            cs.to_csv(pjoin(self.results_dir, "cs_change_categorical.csv"), index=False)

        return cs


    def sdi_pod_summary(self, pods: gpd.GeoDataFrame, buffer_dist: int = 300) -> pd.DataFrame:
        """
        Calculate pre/post SDI statistics for treatment polygons, flagged by
        proximity to POD boundaries (Line vs. Interior).

        :param pods:         GeoDataFrame of POD polygons
        :param buffer_dist:  buffer distance (meters) around POD boundaries for Line flag
        :return:             DataFrame with TRT_ID, scenario, POD_Flag, SDI_bl, SDI_tr, delta_SDI
        """
        sdi_files = self.dd.get('SDI', [])
        if not sdi_files:
            raise ValueError("No SDI rasters found in the data dictionary.")

        # --- Baseline file
        bl_files = [f for f in sdi_files if 'baseline' in f.lower()]
        if not bl_files:
            raise ValueError("No baseline SDI raster found.")
        bl_fp = bl_files[0]
        print(f"\t  SDI raster: {os.path.basename(bl_fp)}")

        # --- Prep treatment data
        trts_bl = TEALOM._geom_to_raster_crs(self.trts, bl_fp)

        # --- POD boundary buffer -> Line vs. Interior flag
        pod_buffer = gpd.GeoDataFrame(
            geometry=pods.boundary.buffer(buffer_dist), crs=pods.crs
        ).to_crs(trts_bl.crs)
        # --- Add a POD Line flag
        trts_bl['PODLine'] = np.where(
            trts_bl.intersects(pod_buffer.union_all()), 'Line', 'Interior'
        ).copy()

        # --- Baseline zonal stats over ALL treatment polygons (computed once)
        bl_stats = compute_band_stats(
            trts_bl, bl_fp,
            id_col=self.id_col,
            attr='SDI_bl',
            stats=['mean'],
            ztype='continuous'
        )
        bl_stats.rename(columns={'SDI_bl_mean': 'SDI_bl'}, inplace=True)

        # --- Treated files
        tr_files = [f for f in sdi_files if 'treated' in f.lower()]
        if not tr_files:
            raise ValueError("No treated SDI rasters found.")

        results = []
        for fp in tr_files:
            stem = Path(fp).stem
            scenario = next(
                (orig for orig, low in sorted(self.scenarios, key=lambda x: len(x[1]), reverse=True)
                 if TEALOM._normalize(stem).startswith(TEALOM._normalize(low))),
                None
            )
            if scenario is None:
                print(f"  ! Could not match scenario for: {stem}, skipping")
                continue
            print(f"\tScenario: {scenario}  |  {Path(fp).name}")

            subset = trts_bl[trts_bl[self.type_col] == scenario].copy()
            if subset.empty:
                print(f"  ! No treatment polygons for scenario: {scenario}, skipping")
                continue
            subset = TEALOM._geom_to_raster_crs(subset, fp)

            tr_stats = compute_band_stats(
                subset, fp,
                id_col=self.id_col,
                attr='SDI_tr',
                stats=['mean'],
                ztype='continuous'
            )
            tr_stats.rename(columns={'SDI_tr_mean': 'SDI_tr'}, inplace=True)

            # filter baseline to this scenario's polygons and merge
            bl_sc = bl_stats[bl_stats[self.id_col].isin(subset[self.id_col])]
            df = bl_sc.merge(tr_stats, on=self.id_col, how='inner')

            # carry PODLine via map — no merge fan-out risk
            df['PODLine'] = df[self.id_col].map(subset.set_index(self.id_col)['PODLine'])
            df['scenario'] = scenario
            df['delta_SDI'] = df['SDI_tr'] - df['SDI_bl']
            df["delta_SDI_pct"] = np.where(
                df["SDI_bl"] >= 1.0,
                (df["delta_SDI"] / df["SDI_bl"]) * 100,
                np.nan
            )

            results.append(df[[self.id_col, 'scenario', 'PODLine',
                               'SDI_bl', 'SDI_tr', 'delta_SDI', 'delta_SDI_pct']])

        sdi = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

        if self.save_outputs:
            sdi.to_csv(pjoin(self.results_dir, "sdi_summary.csv"), index=False)

        return sdi