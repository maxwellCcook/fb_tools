"""Episodes, weather units, seeds, CRN contract, and outputs — no engine required."""

import json

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from fb_tools.models import pyre
from fb_tools.spread import episodes as ep
from fb_tools.spread import pyre_outputs as po


# ── select_episodes ──────────────────────────────────────────────────────────

@pytest.fixture
def erc_cache(tmp_path):
    rng = np.random.default_rng(1)
    years = list(range(2011, 2021))
    erc = rng.uniform(20, 60, size=(len(years), 214))
    erc[:, 100:140] += 40                     # a hot mid-season block every year
    (tmp_path / "pyrome_9_gridmet.json").write_text(json.dumps(
        {"pyrome_id": "9", "years": years, "HistoricERCValues": erc.tolist()}))
    return tmp_path, erc


def test_episodes_meet_threshold_and_do_not_overlap(erc_cache):
    cache, erc = erc_cache
    df = ep.select_episodes(9, cache, severity="p90", n=5, duration_days=21, seed=3)
    assert len(df) == 5
    assert (df.erc_day1 >= np.percentile(erc, 90)).all()
    assert (df.end_date - df.start_date).dt.days.eq(20).all()
    assert (df.season_day + 20 <= 214).all()
    for y, sub in df.groupby("year"):
        d = np.sort(sub.season_day.values)
        assert (np.diff(d) >= 21).all()


def test_year_filter_and_too_few_raise(erc_cache):
    cache, _ = erc_cache
    df = ep.select_episodes(9, cache, severity="p90", n=2, years=[2015, 2016], seed=0)
    assert set(df.year) <= {2015, 2016}
    with pytest.raises(ValueError, match="non-overlapping"):
        ep.select_episodes(9, cache, severity="p90", n=50, seed=0)


def test_rtma_coverage_filter_skips_gappy_windows(erc_cache):
    cache, _ = erc_cache
    yrs = [2014, 2015]
    full = pd.DataFrame({"datetime_utc": pd.concat(
        [pd.Series(pd.date_range(f"{y}-04-01", f"{y}-10-31 23:00", freq="h")) for y in yrs])})
    df = ep.select_episodes(9, cache, severity="p90", n=2, years=yrs, rtma_df=full, seed=0)
    # punch a 2-day hole into each chosen window; neither may be chosen again
    holes = pd.concat([pd.Series(pd.date_range(r.start_date + pd.Timedelta(days=5), periods=48, freq="h"))
                       for r in df.itertuples()])
    gappy = full[~full.datetime_utc.isin(holes)]
    df2 = ep.select_episodes(9, cache, severity="p90", n=2, years=yrs, rtma_df=gappy, seed=0)
    assert all(ep._longest_gap_h(gappy, r.start_date, 22) <= 3 for r in df2.itertuples())
    assert not set(df2.start_date) & set(df.start_date)


# ── episode_hourly_frame ─────────────────────────────────────────────────────

@pytest.fixture
def rtma_df():
    idx = pd.date_range("2018-07-01", periods=10 * 24, freq="h")
    return pd.DataFrame({
        "datetime_utc": idx, "tmp_f": 86.0, "FM1": 4.0, "FM10": 5.5, "FM100": 8.0,
        "u_ms": -5.0, "v_ms": 0.0, "ws_ms": 5.0,
    })


def test_frame_units_match_engine(rtma_df):
    row = {"episode_id": "ep_000", "start_date": "2018-07-02", "end_date": "2018-07-04"}
    f = ep.episode_hourly_frame(row, rtma_df, ignition_hour_local=12, utc_offset_h=-6)
    assert len(f) == 72
    assert f.datetime_utc.iloc[0] == pd.Timestamp("2018-07-02 18:00")   # noon MDT
    assert f.tmp_c.iloc[0] == pytest.approx(30.0)                        # °F -> °C
    assert f.ws_kmh.iloc[0] == pytest.approx(18.0)                       # m/s -> km/h
    assert f.wd_from_deg.iloc[0] == pytest.approx(90.0)                  # u<0 blows west: FROM east
    assert f.fm1.iloc[0] == pytest.approx(0.04)                          # % -> fraction
    assert f.fm_herb.iloc[0] == pytest.approx(0.52) and f.fm_woody.iloc[0] == pytest.approx(0.72)
    assert f.attrs["wind_source"] == "rtma"
    arrs = ep.weather_arrays(f)
    assert set(arrs) == set(ep.WEATHER_CUBES) and all(a.dtype == np.float32 for a in arrs.values())


def test_missing_wind_and_gap_raise(rtma_df):
    row = {"start_date": "2018-07-02", "end_date": "2018-07-03"}
    with pytest.raises(ValueError, match="lacks"):
        ep.episode_hourly_frame(row, rtma_df.drop(columns=["u_ms"]))
    gappy = rtma_df.drop(rtma_df.index[50:60])      # inside the window
    with pytest.raises(ValueError, match="gap"):
        ep.episode_hourly_frame(row, gappy)


def test_weather_from_scenario_units():
    sc = {"WIND_SPEED": 23.1, "FM_1hr": 4.0, "FM_10hr": 5.5, "FM_100hr": 6.6, "FM_herb": 59.3, "FM_woody": 78.7}
    w = ep.weather_from_scenario(sc, wind_from_deg=292.5, tmp_c=30.0, n_hours=3)
    assert set(w) == set(ep.WEATHER_CUBES) and all(v.shape == (3,) and v.dtype == np.float32 for v in w.values())
    assert w["wind_speed_10m"][0] == pytest.approx(23.1 * 1.609344, rel=1e-5)
    assert w["fuel_moisture_dead_1hr"][0] == pytest.approx(0.04) and w["upwind_direction"][0] == 292.5


# ── seeds and the CRN contract ───────────────────────────────────────────────

def test_pad_bands_gives_one_spare_hour():
    """The engine reads band ``stop_time // 60`` on its last step: that must be a real band."""
    w = {"temperature": np.arange(10, dtype="float32"), "wind_speed_10m": np.ones((10, 2, 3), "float32")}
    out = pyre.pad_bands(w, duration_h=8)
    assert out["temperature"].shape == (9,) and out["wind_speed_10m"].shape == (9, 2, 3)   # ceil(8) + 1
    assert out["temperature"].tolist() == list(range(9))                                    # trimmed, not repeated
    short = pyre.pad_bands({"temperature": np.arange(3, dtype="float32")}, duration_h=4.5)
    assert short["temperature"].tolist() == [0, 1, 2, 2, 2, 2]                              # ceil(4.5) + 1, last repeated
    assert all(a.dtype == np.float32 for a in out.values())


def test_seed_is_arm_independent_and_key_sensitive():
    s = pyre.seed_for(617327, "ign_000", "ep_000", "central")
    assert s == pyre.seed_for(617327, "ign_000", "ep_000", "central")
    assert s != pyre.seed_for(617327, "ign_000", "ep_001", "central")
    assert s != pyre.seed_for(617328, "ign_000", "ep_000", "central")
    assert 0 <= s < 2 ** 31


def test_build_runs_pairs_arms_and_contract_catches_a_broken_seed(tmp_path):
    runs = pyre.build_runs({"a": "la", "b": "lb"}, {"i0": "i0.shp", "i1": "i1.shp"},
                           {"e0": "e0.csv"}, {"c": {"spot_config": None, "crown_gate": 0.0}},
                           base_seed=1, out_root=tmp_path)
    assert len(runs) == 4
    assert runs.groupby(["ignition_id", "episode_id"]).seed.nunique().eq(1).all()
    runs.loc[0, "seed"] += 1
    with pytest.raises(AssertionError, match="CRN"):
        pyre.assert_crn_contract(runs, n_arms=2)


# ── outputs ──────────────────────────────────────────────────────────────────

def test_firebrand_production_matches_engine_formula():
    fb = po.firebrand_production([1000.0, 500.0, 0.0], [10.0, 0.0, 5.0], 900.0,
                                 firebrands_per_unit_heat=1e-6)
    assert fb[0] == pytest.approx(1e-6 * 60 * 1000 / 10 * 900)
    assert fb[1] == 0.0 and fb[2] == 0.0
    sloped = po.firebrand_production([1000.0], [10.0], 900.0, slope=[1.0], firebrands_per_unit_heat=1e-6)
    assert sloped[0] == pytest.approx(fb[0] * np.sqrt(2))


def _write_run(run_dir, toa, meta):
    run_dir.mkdir(parents=True)
    prof = dict(driver="GTiff", height=toa.shape[0], width=toa.shape[1], count=1, dtype="float32",
                transform=from_origin(0, 100, 10, 10), crs="EPSG:32613", nodata=np.nan)
    layers = {"time_of_arrival": toa, "fire_type": np.where(np.isfinite(toa), 1.0, np.nan),
              "fireline_intensity": np.where(np.isfinite(toa), 100.0, np.nan),
              "flame_length": np.where(np.isfinite(toa), 2.0, np.nan),
              "spread_rate": np.where(np.isfinite(toa), 5.0, np.nan),
              "spread_direction": np.where(np.isfinite(toa), 90.0, np.nan)}
    for k, a in layers.items():
        with rasterio.open(run_dir / f"{k}.tif", "w", **prof) as dst:
            dst.write(a.astype("float32"), 1)
    (run_dir / "manifest.json").write_text(json.dumps({**meta, "seed": 7, "spot_config": None, "duration_h": 48.0}))


def test_aggregate_and_paired_deltas(tmp_path):
    toa_a = np.full((4, 4), np.nan, dtype="float32")
    toa_a[:2, :] = 1440.0                     # day 1
    toa_b = toa_a.copy()
    toa_b[:1, :] = 2880.0                     # treated: half the burn arrives day 2
    key = {"condition_id": "c", "ignition_id": "i", "episode_id": "e"}
    _write_run(tmp_path / "a", toa_a, {**key, "arm": "a"})
    _write_run(tmp_path / "b", toa_b, {**key, "arm": "b"})

    paths = po.aggregate_realizations([tmp_path / "a", tmp_path / "b"], tmp_path / "agg",
                                      day_marks=(1, 2, 21), verbose=False)
    rd = lambda k: rasterio.open(paths[k]).read(1)                  # noqa: E731
    bp, at = rd("BurnProb"), rd("AvgTime")
    assert bp[0, 0] == 1.0 and bp[3, 0] == 0.0
    assert at[0, 0] == pytest.approx(1.5) and np.isnan(at[3, 0])   # days, NaN outside
    # arrival probability: row 0 reached on day 1 in a, day 2 in b; row 1 day 1 in both
    assert rd("ArrivalProb_d1")[0, 0] == 0.5 and rd("ArrivalProb_d1")[1, 0] == 1.0
    assert rd("ArrivalProb_d2")[0, 0] == 1.0 and rd("ArrivalProb_d2")[3, 0] == 0.0
    assert "ArrivalProb_d21" not in paths                           # past the 2-day runs
    # expected firebrands: 5e-11 * 60 * I / R * cell area, averaged over realizations
    fb = rd("Firebrands")
    assert fb[0, 0] == pytest.approx(5e-11 * 60 * 100.0 / 5.0 * 100.0, rel=1e-5) and fb[3, 0] == 0.0
    assert set(paths) == {"BurnProb", "AvgFlameLength", "AvgTime", "n_burned", "ArrivalProb_d1",
                          "ArrivalProb_d2", "Firebrands"}

    res = pd.DataFrame({"out_dir": [tmp_path / "a", tmp_path / "b"], "status": ["ok", "ok"]})
    summ = po.summarize_runs(res, day_marks=(1, 2))
    assert summ.area_d1_ha.tolist() == pytest.approx([0.08, 0.04])
    d = po.paired_deltas(summ, baseline="a", treated="b")
    assert len(d) == 1 and d.d_area_d1_ha.iloc[0] == pytest.approx(-0.04)
    assert d.d_max_toa_h.iloc[0] == pytest.approx(24.0)


# ── load_rtma_hourly: yearly and per-month export spellings ─────────────────

def test_load_rtma_hourly_accepts_yearly_and_monthly_files(tmp_path):
    def rows(start, n, pid=9):
        idx = pd.date_range(start, periods=n, freq="h")
        return pd.DataFrame({"pyrome_id": pid, "datetime_utc": idx.strftime("%Y-%m-%d %H:%M:%S"),
                             "tmp_f": 80.0, "rh_pct": 20.0, "emc_pct": 5.0, "vpd_pa": 2000.0,
                             "pcp_mm_hr": 0.0})
    emc, wind = tmp_path / "emc", tmp_path / "wind"
    emc.mkdir(); wind.mkdir()
    rows("2018-07-01", 48).to_csv(emc / "rtma_hourly_X_pyromes_2018_07.csv", index=False)
    rows("2018-08-01", 48).to_csv(emc / "rtma_hourly_X_pyromes_2018_08.csv", index=False)
    rows("2019-07-01", 48).to_csv(emc / "rtma_hourly_X_pyromes_2019.csv", index=False)
    rows("2018-07-01", 48, pid=8).to_csv(emc / "rtma_hourly_X_pyromes_2018_09.csv", index=False)
    for name, start in [("rtma_wind_X_pyromes_2018_07.csv", "2018-07-01"),
                        ("rtma_wind_X_pyromes_2018_08.csv", "2018-08-01"),
                        ("rtma_wind_X_pyromes_2019.csv", "2019-07-01")]:
        r = rows(start, 48)[["pyrome_id", "datetime_utc"]].assign(u_ms=-3.0, v_ms=0.0, ws_ms=4.0)
        r.to_csv(wind / name, index=False)
    # a header-only month (a failed GEE task) must not poison the dtypes of the concat
    (wind / "rtma_wind_X_pyromes_2018_09.csv").write_text("pyrome_id,datetime_utc,u_ms,v_ms,ws_ms\n")

    df = ep.load_rtma_hourly(emc, 9, [2018, 2019], wind_dir=wind)
    assert all(df[c].dtype.kind == "f" for c in ("u_ms", "v_ms", "ws_ms", "FM1"))
    assert len(df) == 144 and df["datetime_utc"].is_monotonic_increasing
    assert {"FM1", "FM10", "FM100", "u_ms", "v_ms", "ws_ms"} <= set(df.columns)
    assert df["ws_ms"].notna().all() and (df["pyrome_id"].astype(int) == 9).all()

    with pytest.raises(FileNotFoundError):
        ep.load_rtma_hourly(emc, 9, [2017])
    with pytest.raises(FileNotFoundError):
        ep.load_rtma_hourly(emc, 9, [2018], wind_dir=tmp_path)


def test_rank_top_by_erc_and_explicit_dates(erc_cache):
    cache, erc = erc_cache
    top = ep.select_episodes(9, cache, severity="p90", n=3, duration_days=7, rank="erc_day1", seed=0)
    rnd = ep.select_episodes(9, cache, severity="p90", n=3, duration_days=7, rank="random", seed=0)
    assert top.erc_day1.min() >= rnd.erc_day1.max() - 1e-9 or top.erc_day1.sum() >= rnd.erc_day1.sum()
    assert top.erc_day1.iloc[0] <= np.nanmax(erc) and (top.erc_pct >= 90).all()
    # explicit dates: a named event is kept without a severity filter, and its ERC percentile is reported
    named = ep.select_episodes(9, cache, severity=None, n=None, duration_days=7, dates=["2015-07-20", "2015-07-22", "2030-01-01"])
    assert len(named) == 1 and named.source.iloc[0] == "dates" and 0 <= named.erc_pct.iloc[0] <= 100   # overlap + out-of-range dropped
    assert named.start_date.iloc[0] == pd.Timestamp("2015-07-20")
    with pytest.raises(ValueError, match="rank must be"):
        ep.select_episodes(9, cache, rank="hottest")


def test_dates_from_fod_groups_by_day():
    import geopandas as gpd
    f = gpd.GeoDataFrame({"DISCOVERY_DATE": ["07/06/2005", "07/06/2005", "06/22/2005", "05/01/2006"],
                          "FIRE_SIZE_CLASS": ["G", "D", "E", "C"], "FIRE_SIZE": [11357.0, 150.0, 1182.0, 20.0],
                          "FIRE_NAME": ["MASON", "SMALL", "RIO", "TINY"]}, geometry=gpd.points_from_xy([0, 0, 0, 0], [0, 0, 0, 0]))
    pool = ep.dates_from_fod(f, min_class="D")
    assert len(pool) == 2 and pool.fire_name.tolist() == ["RIO", "MASON"] and pool.n_fires.tolist() == [1, 2]


def test_dates_from_fast_fires_windows_and_area():
    tab = pd.DataFrame({"fired_id": [1, 2, 3], "mx_grw_dte": ["2020-09-08", "2018-06-28", "2016-07-10"],
                        "mx_grw_ha": [15283.7, 13265.9, 1000.0], "tot_ar_ha": [48040.6, 41622.3, 6783.2],
                        "lon": [-105.69, -105.15, -106.1], "lat": [40.65, 37.55, 38.6]})
    pool = ep.dates_from_fast_fires(tab, bbox=(-106.0, 40.0, -105.0, 41.0), days_before=3)
    assert len(pool) == 1 and pool.date.iloc[0] == pd.Timestamp("2020-09-05")     # 3 d before the biggest day
    assert len(ep.dates_from_fast_fires(tab, days_before=0)) == 2                    # the 1,000 ha/day event is below the threshold
    assert len(ep.dates_from_fast_fires(tab, min_growth_ha=500, days_before=0)) == 3


def test_build_fast_fires_table_from_fired(tmp_path):
    import pytest as _pt
    fp = ep.Path("/Users/mcc/Library/CloudStorage/Box-Box/MCC/data/wildfire/FIRED/raw/fired_v2025var_usa_lower48_2000-2026_s05_t11_events.gpkg")
    if not fp.exists():
        _pt.skip("FIRED events file not on this machine")
    t = ep.build_fast_fires_table(fp, tmp_path / "fast.csv")
    assert (t.mx_grw_ha >= ep.FAST_FIRE_GROWTH_HA).all() and {"fired_id", "mx_grw_dte", "lon", "lat"} <= set(t.columns)
    assert (tmp_path / "fast.csv").exists() and len(t) > 1000
