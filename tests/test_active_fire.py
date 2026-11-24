"""Offline tests for fb_tools.active_fire (FIRMS client + QA). No network: requests.get is faked."""
import io
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from fb_tools.active_fire import events, firms, footprints, grids, plot, qa, timeutil, treatments

KEY = "testkey123"
BBOX = (-105.25, 37.85, -104.70, 38.30)
TODAY = pd.Timestamp.now("UTC").date()

VIIRS_CSV = """latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight
38.1,-105.0,330.1,0.39,0.36,2025-07-01,0912,N,VIIRS,n,2.0NRT,290.2,5.5,N
38.2,-105.1,340.0,0.40,0.37,2025-07-01,2030,N,VIIRS,l,2.0NRT,295.0,1.2,D
38.3,-105.2,350.0,0.41,0.38,2025-07-02,45,N,VIIRS,h,2.0NRT,300.0,60.0,N
"""
MODIS_CSV = """latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight,type
38.1,-105.0,320.0,1.0,1.0,2025-07-01,1830,Terra,MODIS,29,6.1,290.0,12.0,D,0
38.1,-105.1,320.0,1.0,1.0,2025-07-01,1830,Terra,MODIS,30,6.1,290.0,12.0,D,2
38.1,-105.2,320.0,1.0,1.0,2025-07-01,1830,Aqua,MODIS,81,6.1,290.0,12.0,D,0
"""
GOES_CSV = """latitude,longitude,acq_date,acq_time,satellite,confidence,frp,daynight
38.1,-105.0,2025-07-01,1800,G18,10,-999.0,D
38.1,-105.0,2025-07-01,1805,G18,10,150.0,D
"""


def _raw(csv, source):
    return pd.read_csv(io.StringIO(csv)).assign(source=source)


# ── Schema ─────────────────────────────────────────────────────────────────────

def test_standardize_schema_times_confidence():
    raw = pd.concat([_raw(VIIRS_CSV, "VIIRS_SNPP_NRT"), _raw(MODIS_CSV, "MODIS_SP"),
                     _raw(GOES_CSV, "GOES_NRT")], ignore_index=True)
    g = firms.standardize_firms(raw)

    assert list(g.columns) == firms.STD_COLS and g.crs.to_epsg() == 4326
    assert str(g["acq_datetime"].dt.tz) == "UTC"
    assert g["acq_datetime"].iloc[2] == pd.Timestamp("2025-07-02 00:45", tz="UTC")   # 45 → 0045
    assert g["detection_id"].is_unique and (g["product"] == "FIRMS").all()

    v, m, go = (g[g["instrument"] == i] for i in ("VIIRS", "MODIS", "ABI"))
    assert list(v["confidence_norm"]) == ["nominal", "low", "high"]
    assert list(m["confidence_norm"]) == ["low", "nominal", "high"]                  # 29 / 30 / 81
    assert set(v["satellite"]) == {"VIIRS S-NPP"} and v["scan"].iloc[0] == 0.39
    assert np.isnan(go["frp"].iloc[0]) and go["frp"].iloc[1] == 150.0                # −999 → NaN
    assert v["type"].isna().all() and list(m["type"]) == [0, 2, 0]


def test_standardize_empty_and_exact_duplicates():
    assert list(firms.standardize_firms(None).columns) == firms.STD_COLS
    raw = _raw(VIIRS_CSV, "VIIRS_SNPP_NRT")
    assert len(firms.standardize_firms(pd.concat([raw, raw]))) == 3


def test_sources_catalog_offline(capsys):
    cat = firms.firms_sources(print_table=False)
    assert set(firms.VIIRS_NRT) | set(firms.VIIRS_SP) <= set(cat["source"])
    assert capsys.readouterr().out == ""


# ── Fetch ──────────────────────────────────────────────────────────────────────

class FakeFIRMS:
    """Stands in for requests.get: one detection per requested day at the bbox centre."""

    def __init__(self, empty_days=(), fail_first=None, max_date=TODAY):
        self.urls, self.empty, self.fail_first, self.max_date = [], set(empty_days), fail_first, max_date

    def area_calls(self):
        return [u.split("/area/csv/")[1].split("/")[1:] for u in self.urls if "/area/" in u]

    def __call__(self, url, timeout=None):
        self.urls.append(url)
        if "data_availability" in url:
            rows = [f"{s},2012-01-20,{self.max_date}" for s in firms.SOURCE_META]
            return _Resp("data_id,min_date,max_date\n" + "\n".join(rows))
        _, src, area, n, first = url.split("/area/csv/")[1].split("/")
        if KEY not in url:
            return _Resp("Invalid MAP_KEY.")
        if first == self.fail_first:
            return _Resp("Error: transaction limit exceeded")
        w, s, e, n_ = map(float, area.split(","))
        days = [str(d.date()) for d in pd.date_range(first, periods=int(n))]
        rows = [f"{(s + n_) / 2},{(w + e) / 2},0.39,0.36,{d},1900,n,2.0NRT,5.0,D"
                for d in days if d not in self.empty]
        return _Resp("latitude,longitude,scan,track,acq_date,acq_time,confidence,version,frp,daynight\n"
                     + "\n".join(rows))


class _Resp:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


@pytest.fixture
def fake(monkeypatch):
    f = FakeFIRMS()
    monkeypatch.setattr(firms.requests, "get", f)
    monkeypatch.setattr(firms.time, "sleep", lambda s: None)
    return f


def _fetch(tmp_path, *a, **kw):
    kw.setdefault("sources", ["VIIRS_SNPP_SP"])
    return firms.fetch_firms(*a, map_key=KEY, cache_dir=tmp_path, **kw)


def test_fetch_chunks_caches_and_marks_empty_days(fake, tmp_path):
    fake.empty = {"2025-07-03"}
    g = _fetch(tmp_path, BBOX, "2025-07-01", "2025-07-12")
    assert len(g) == 11 and "event_id" not in g
    assert [(c[2], c[3]) for c in fake.area_calls()] == [
        ("5", "2025-07-01"), ("5", "2025-07-06"), ("2", "2025-07-11")]
    assert fake.area_calls()[0][1] == "-105.2500,37.8500,-104.7000,38.3000"

    files = sorted(tmp_path.glob("*.parquet"))
    assert len(files) == 12 and not list(tmp_path.glob("*.part"))                 # one per day
    assert len(pd.read_parquet(next(tmp_path.glob("*20250703.parquet")))) == 0    # empty marker

    n = len(fake.urls)
    g2 = _fetch(tmp_path, BBOX, "2025-07-01", "2025-07-12")
    assert len(fake.urls) == n                                                    # all from cache
    pd.testing.assert_frame_equal(pd.DataFrame(g2), pd.DataFrame(g))

    _fetch(tmp_path, BBOX, "2025-07-10", "2025-07-14")                            # only new days
    assert [(c[2], c[3]) for c in fake.area_calls()[3:]] == [("2", "2025-07-13")]


def test_fetch_goes_two_day_windows(fake, tmp_path):
    _fetch(tmp_path, BBOX, "2025-07-01", "2025-07-05", sources=["GOES_NRT"])
    assert [c[2] for c in fake.area_calls()] == ["2", "2", "1"]


def test_fetch_repulls_only_the_last_nrt_day(fake, tmp_path):
    start = TODAY - pd.Timedelta(days=3)
    _fetch(tmp_path, BBOX, start, TODAY, sources=["VIIRS_SNPP_NRT"])
    n = len(fake.area_calls())
    _fetch(tmp_path, BBOX, start, TODAY, sources=["VIIRS_SNPP_NRT"])
    assert [(c[2], c[3]) for c in fake.area_calls()[n:]] == [("1", str(TODAY))]

    # a past day that was cached while still in progress (written at its own noon) is not final
    stale = TODAY - pd.Timedelta(days=2)
    fp = next(tmp_path.glob(f"*{stale:%Y%m%d}.parquet"))
    os.utime(fp, (pd.Timestamp(stale, tz="UTC").timestamp() + 43200,) * 2)
    n = len(fake.area_calls())
    _fetch(tmp_path, BBOX, start, TODAY, sources=["VIIRS_SNPP_NRT"])
    assert [(c[2], c[3]) for c in fake.area_calls()[n:]] == [("1", str(stale)), ("1", str(TODAY))]


def test_fetch_skips_days_outside_availability(fake, tmp_path, capsys):
    fake.max_date = "2025-07-02"
    g = _fetch(tmp_path, BBOX, "2025-07-01", "2025-07-04")
    assert len(g) == 2 and len(list(tmp_path.glob("*.parquet"))) == 2   # unpublished days not cached
    assert "outside availability" in capsys.readouterr().out


def test_fetch_events_loop_and_failed_window(fake, tmp_path, capsys):
    events = gpd.GeoDataFrame(
        {"fid": ["A 1", "B"], "ig": ["2025-07-01", "2025-08-01"], "last": ["2025-07-02", "2025-08-07"]},
        geometry=[box(-105.2, 38.0, -105.0, 38.2), box(-107.2, 39.0, -107.0, 39.2)], crs="EPSG:4326",
    )
    fake.fail_first = "2025-08-06"
    g = _fetch(tmp_path, events, id_col="fid", start_col="ig", end_col="last", buffer_m=1000)
    out = capsys.readouterr().out
    assert g.groupby("event_id").size().to_dict() == {"A 1": 2, "B": 5}
    assert "[2/2] B: 5 detections (2 requests, 1 failed" in out and KEY not in out
    assert not list(tmp_path.glob("*2025080[67].parquet"))               # failed window not cached
    w = float(fake.area_calls()[0][1].split(",")[0])
    assert -105.22 < w < -105.21                                         # 1 km buffer, ~0.0115°
    assert len(list(tmp_path.glob("A-1_*"))) == 2                        # id is in the file name


def test_fetch_key_errors(fake, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MAP_KEY"):
        firms.fetch_firms(BBOX, "2025-07-01", "2025-07-01", cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="<MAP_KEY>"):                 # key redacted in the message
        firms._read_firms_csv(f"{firms.FIRMS_BASE}/area/csv/badkey/VIIRS_SNPP_SP/0,0,1,1/1/2025-07-01",
                              "badkey")
    with pytest.raises(ValueError, match="before start"):
        _fetch(tmp_path, BBOX, "2025-07-02", "2025-07-01")


# ── map_events ─────────────────────────────────────────────────────────────────

def _count(dets, event=None, scale=1):
    out = {"n": [len(dets) * scale]}
    if event is not None:
        out["name"] = [event["name"].iloc[0]]
    return pd.DataFrame(out)


def _detections(n=6):
    raw = pd.DataFrame({
        "latitude": 38.0, "longitude": -105.0 + 0.01 * np.arange(n), "scan": 0.39, "track": 0.36,
        "acq_date": "2025-07-01", "acq_time": 1900, "confidence": "n", "frp": 1.0 + np.arange(n),
        "daynight": "D", "source": "VIIRS_SNPP_SP", "type": 0,
    })
    return firms.standardize_firms(raw)


def test_map_events_serial_matches_parallel():
    d = _detections().assign(event_id=["a", "a", "a", "b", "b", "c"])
    one = firms.map_events(_count, d, scale=2)
    assert one.to_dict("list") == {"n": [6, 4, 2], "event_id": ["a", "b", "c"]}
    pd.testing.assert_frame_equal(one, firms.map_events(_count, d, n_workers=2, scale=2))

    ev = pd.DataFrame({"fid": ["c", "a", "b"], "name": ["C", "A", "B"]})
    assert list(firms.map_events(_count, d, ev, id_col="fid")["name"]) == ["A", "B", "C"]
    with pytest.raises(ValueError, match="event_id"):
        firms.map_events(_count, _detections())


# ── QA ─────────────────────────────────────────────────────────────────────────

def _pair(dx_m=100.0, dt_s=0, frp=(5.0, 9.0), **over):
    """Two VIIRS detections dx_m apart (east-west, at 38°N) and dt_s apart."""
    g = _detections(2)
    g["longitude"] = [-105.0, -105.0 + dx_m / (111_320 * np.cos(np.radians(38)))]
    g = g.set_geometry(gpd.points_from_xy(g["longitude"], g["latitude"]), crs="EPSG:4326")
    g["acq_datetime"] = g["acq_datetime"].iloc[0] + pd.to_timedelta([0, dt_s], unit="s")
    g["frp"] = list(frp)
    for k, v in over.items():
        g[k] = v
    return g


def test_interscan_keeps_max_frp():
    kept, dropped = qa.remove_interscan_duplicates(_pair())
    assert list(kept["frp"]) == [9.0] and list(dropped["frp"]) == [5.0]
    assert kept.index.tolist() == [1]                                    # index preserved


@pytest.mark.parametrize("kw", [
    dict(dx_m=250.0),                      # outside 187.5 m
    dict(dt_s=8),                          # outside 7 s
    dict(scan=[0.39, 0.40]),               # different scan
    dict(track=[0.36, 0.37]),              # different track
    dict(satellite=["VIIRS S-NPP", "VIIRS NOAA-20"]),
    dict(event_id=["a", "b"]),
    dict(instrument="MODIS"),              # only VIIRS is screened
])
def test_interscan_non_duplicates_are_kept(kw):
    kept, dropped = qa.remove_interscan_duplicates(_pair(**kw))
    assert len(kept) == 2 and len(dropped) == 0


def test_interscan_inside_tolerances_and_missing_frp():
    assert len(qa.remove_interscan_duplicates(_pair(dx_m=180.0, dt_s=7))[0]) == 1
    kept, _ = qa.remove_interscan_duplicates(_pair(frp=(np.nan, 2.0)))
    assert list(kept["frp"]) == [2.0]
    with pytest.raises(NotImplementedError):
        qa.remove_interscan_duplicates(_pair(), method="swath")


def test_low_confidence_and_static(capsys):
    g = _detections(4)
    g["confidence_norm"] = ["low", "nominal", "high", None]
    kept, dropped = qa.drop_low_confidence(g)
    assert len(kept) == 3 and list(dropped.index) == [0]

    g["type"] = [0, 2, 3, 0]
    kept, dropped = qa.drop_static_sources(g)
    assert list(kept.index) == [0, 3] and len(dropped) == 2

    capsys.readouterr()
    kept, dropped = qa.drop_static_sources(g.assign(type=pd.NA))         # NRT: nothing to screen
    assert len(kept) == 4 and "not screened" in capsys.readouterr().out


def test_apply_qa_chain_and_switches():
    g = pd.concat([_pair(), _pair(dx_m=5000.0)], ignore_index=True)      # rows 0/1 duplicates
    g["confidence_norm"] = ["nominal", "nominal", "low", "nominal"]
    g["type"] = [0, 0, 0, 2]
    kept, dropped = qa.apply_qa(g)
    assert list(kept.index) == [1]
    assert dropped["qa_reason"].to_dict() == {2: "low_confidence", 0: "interscan", 3: "static_source"}
    kept, dropped = qa.apply_qa(g, low_conf=False, interscan=False, static=False)
    assert len(kept) == 4 and len(dropped) == 0 and "qa_reason" in dropped


# ── Overpasses, footprints, grids ──────────────────────────────────────────────

T0 = pd.Timestamp("2025-07-01 20:00", tz="UTC")       # 14:00 MDT


def _det5070(xy, minutes, frp=10.0, scan=0.4, track=0.4, sat="VIIRS S-NPP", dn="D"):
    """Detections placed directly in EPSG:5070 metres, so box/cell overlaps are exact."""
    n = len(xy)
    return gpd.GeoDataFrame({
        "detection_id": [f"d{i}" for i in range(n)], "frp": frp, "satellite": sat, "instrument": "VIIRS",
        "acq_datetime": [T0 + pd.Timedelta(minutes=m) for m in minutes], "daynight": dn,
        "scan": scan, "track": track, "bright_ti4": 330.0, "bright_ti5": 290.0,
    }, geometry=gpd.points_from_xy(*zip(*xy)), crs="EPSG:5070")


def _two_cells():
    return grids.afd_grid(gpd.GeoSeries([box(1, 1, 959, 479)], crs="EPSG:5070"), res_m=480)


def test_assign_overpass_and_local_time():
    d = _det5070([(0, 0)] * 4, [0, 1, 100, 2], sat=["A", "A", "A", "B"])
    o = timeutil.assign_overpass(d)
    assert list(o["overpass_id"]) == [0, 0, 1, 2] and o.index.equals(d.index)
    assert list(o["overpass_time"]) == [T0, T0, T0 + pd.Timedelta(minutes=100), T0 + pd.Timedelta(minutes=2)]
    assert timeutil.assign_overpass(d, gap_min=200)["overpass_id"].nunique() == 2

    late = _det5070([(0, 0)], [9 * 60])                # 05:00 UTC next day = 23:00 MDT same day
    assert timeutil.day_of(late["acq_datetime"]).iloc[0] == pd.Timestamp("2025-07-02")
    assert timeutil.day_of(late["acq_datetime"], "America/Denver").iloc[0] == pd.Timestamp("2025-07-01")
    assert timeutil.to_local(late, "America/Denver")["acq_local"].iloc[0].hour == 23


def test_footprint_box_is_scan_by_track(capsys):
    d = _det5070([(1000, 2000), (0, 0)], [0, 0], frp=[8.0, 8.0], scan=[0.5, 0.4], track=0.4)
    d["instrument"] = ["VIIRS", "MODIS"]
    b = footprints.footprint_boxes(d)
    assert len(b) == 1 and "dropped 1" in capsys.readouterr().out
    assert b.geometry.iloc[0].bounds == (750.0, 1800.0, 1250.0, 2200.0)
    assert b.geometry.area.iloc[0] == pytest.approx(0.5 * 0.4 * 1e6)
    assert b["pix_area_km2"].iloc[0] == pytest.approx(0.2) and b["frp_wkm2"].iloc[0] == pytest.approx(40.0)
    assert d.geometry.iloc[0].geom_type == "Point"                       # input untouched
    assert len(footprints.footprint_boxes(d, instruments=None)) == 2


def test_afd_grid_snapping(tmp_path):
    aoi = gpd.GeoSeries([box(-1000100, 1900100, -999000, 1901000)], crs="EPSG:5070")
    g = grids.afd_grid(aoi)
    b = g.geometry.bounds
    assert (b[["minx", "miny"]] % 480 == 0).all().all() and np.allclose(g.geometry.area, 480 ** 2)
    assert list(g.columns) == ["grid_id", "row", "col", "geometry"] and g["grid_id"].is_unique
    assert b["minx"].min() <= -1000100 and b["maxx"].max() >= -999000

    lf = grids.afd_grid(aoi, snap="landfire").geometry.bounds
    assert ((lf[["minx", "miny"]] - 15) % 30 == 0).all().all()
    assert ((lf["minx"] - lf["minx"].min()) % 480 == 0).all()

    import rasterio
    from rasterio.transform import from_origin
    ref = tmp_path / "ref.tif"
    with rasterio.open(ref, "w", driver="GTiff", height=4, width=4, count=1, dtype="uint8",
                       crs="EPSG:26913", transform=from_origin(500007, 4200013, 30, 30)) as dst:
        dst.write(np.zeros((1, 4, 4), "uint8"))
    r = grids.afd_grid(aoi, snap=ref)
    rb = r.geometry.bounds
    assert r.crs.to_epsg() == 26913                                      # the raster's CRS wins
    assert ((rb["minx"] - 500007) % 480 == 0).all() and ((rb["maxy"] - 4200013) % 480 == 0).all()
    with pytest.raises(ValueError, match="whole multiple"):
        grids.afd_grid(aoi, res_m=400, snap=ref)
    with pytest.raises(ValueError, match="5070"):
        grids.afd_grid(aoi, crs="EPSG:26913", snap="landfire")

    wide = gpd.GeoSeries([box(0, 0, 100, 100), box(2000, 2000, 2100, 2100)], crs="EPSG:5070")
    assert len(grids.afd_grid(wide)) == 2 and len(grids.afd_grid(wide, clip=False)) == 25


def test_gridded_frp_reproduces_potr5_allocation(capsys):
    grid = _two_cells()
    assert len(grid) == 2
    # a: 400 m box split evenly across the two cells, day.  b: wholly in cell 0, night, 9 h later.
    # c: zero FRP, dropped.  frp_wkm2 = frp / 0.16 km2 -> a 100, b 30.
    d = _det5070([(480, 240), (240, 240), (240, 240)], [0, 540, 600], frp=[16.0, 4.8, 0.0], dn=["D", "N", "N"])
    out = grids.gridded_frp(footprints.footprint_boxes(d), grid).set_index("col")
    assert "dropped 1 of 3" in capsys.readouterr().out

    c0, c1 = out.loc[0], out.loc[1]
    assert (c0["afd_count"], c1["afd_count"]) == (2, 1)
    assert c0["overlap"] == pytest.approx(1.5) and c1["overlap"] == pytest.approx(0.5)
    assert c0["frp_csum"] == pytest.approx(80.0) and c1["frp_csum"] == pytest.approx(50.0)   # 100*0.5 + 30*1
    assert (c0["frp_max"], c0["frp_min"], c0["frp_mean"], c0["frp_first"]) == pytest.approx((50, 30, 40, 50))
    assert c0["frp_p90"] == pytest.approx(48.0)
    assert c0["dt_max_frp"] == T0.tz_localize(None) or c0["dt_max_frp"] == T0
    assert (c0["day_count"], c0["night_count"], c1["night_count"]) == (1, 1, 0)
    assert c0["frp_max_night"] == pytest.approx(30.0) and c1["frp_max_night"] == 0
    assert c0["frp_csum_day"] == pytest.approx(50.0) and c0["t4_max"] == 330.0
    assert c0["unique_days"] == 2 and c0["last_obs_date"] == pd.Timestamp("2025-07-02")
    local = grids.gridded_frp(footprints.footprint_boxes(d), grid, tz="America/Denver").set_index("col")
    assert local.loc[0, "unique_days"] == 1                              # 23:00 MDT is the same local day
    assert len(grids.gridded_frp(footprints.footprint_boxes(d), grid, drop_zero_frp=False)) == 2
    day_only = grids.gridded_frp(footprints.footprint_boxes(d[d["daynight"] == "D"]), grid)   # no night rows at all
    assert (day_only["night_count"] == 0).all() and (day_only["frp_p90_night"] == 0).all()


def test_first_detection_rules():
    grid = _two_cells()
    # overpass 0: a box that only clips cell 0 (50 m sliver).
    # overpass 1: two identical boxes — 35% of the cell, 70% if areas were summed.
    # overpass 2: two offset boxes whose union covers 58.6% of the cell.
    d = _det5070([(-150, 240), (120, 240), (120, 240), (120, 240), (300, 240)], [0, 100, 100, 200, 200],
                 scan=[0.4, 0.3, 0.3, 0.3, 0.3], track=[0.4, 0.3, 0.3, 0.3, 0.3])
    b = footprints.footprint_boxes(d)

    def first(rule, **kw):
        out = grids.first_detection_grid(b, grid, rule=rule, **kw).set_index("col")
        return out.loc[0]

    assert first("any")["first_detection"] == T0
    assert first("centre")["first_detection"] == T0 + pd.Timedelta(minutes=100)
    f = first("fraction")
    assert f["first_detection"] == T0 + pd.Timedelta(minutes=200) and f["overpass_id"] == 2
    assert f["fraction"] == pytest.approx(450 * 300 / 480 ** 2)
    assert first("fraction", min_fraction=0.3)["first_detection"] == T0 + pd.Timedelta(minutes=100)
    assert len(grids.first_detection_grid(b, grid, min_fraction=0.9)) == 0
    with pytest.raises(ValueError, match="rule"):
        grids.first_detection_grid(b, grid, rule="half")


def test_to_raster_roundtrip(tmp_path):
    import rasterio
    grid = _two_cells().assign(v=[1.5, np.nan], t=[T0 + pd.Timedelta(hours=36), T0])
    with rasterio.open(grids.to_raster(grid, "v", tmp_path / "v.tif", nodata=-9999)) as src:
        assert src.shape == (1, 2) and src.res == (480, 480) and src.crs.to_epsg() == 5070
        assert src.bounds.left == 0 and src.bounds.top == 480
        assert list(src.read(1)[0]) == [1.5, -9999]
    with rasterio.open(grids.to_raster(grid, "t", tmp_path / "t.tif")) as src:
        assert src.tags()["T0"] == "2025-07-01T00:00:00"
        assert src.read(1)[0] == pytest.approx([20 / 24 + 1.5, 20 / 24])


# ── Events ─────────────────────────────────────────────────────────────────────

def test_associate_detections_space_time_and_multi_event(capsys):
    ev = gpd.GeoDataFrame(
        {"fid": ["A", "B"], "ig": ["2025-07-01", "2025-07-01"], "last": ["2025-07-01", None]},
        geometry=[box(0, 0, 1000, 1000), box(500, 0, 2000, 1000)], crs="EPSG:5070",
    )
    #        A only      A and B      B only       nowhere       A only, 2 days late
    d = _det5070([(100, 100), (700, 100), (1500, 100), (5000, 5000), (100, 100)], [0, 0, 0, 0, 2 * 1440])
    d = pd.concat([d.assign(event_id="bbox1"), d.iloc[[1]].assign(event_id="bbox2")])   # fetch-style repeats

    a = events.associate_detections(d, ev, id_col="fid", start_col="ig", end_col="last")
    assert "3 of 5 detections matched" in capsys.readouterr().out
    assert a.groupby("event_id")["detection_id"].apply(list).to_dict() == {
        "A": ["d0", "d1"], "B": ["d1", "d2"]}                            # d4 is 2 days past A's window
    assert a.set_index(["event_id", "detection_id"])["multi_event"].to_dict() == {
        ("A", "d0"): False, ("A", "d1"): True, ("B", "d1"): True, ("B", "d2"): False}

    padded = events.associate_detections(d, ev, id_col="fid", start_col="ig", end_col="last", pad_after_days=2)
    assert "d4" in set(padded.loc[padded["event_id"] == "A", "detection_id"])
    assert len(events.associate_detections(d, ev, id_col="fid")) == 5    # spatial only
    far = events.associate_detections(d, ev, id_col="fid", buffer_m=6000, hull=True)
    assert "d3" in set(far["detection_id"])


def test_fired_events_filters(tmp_path):
    fp = tmp_path / "fired_events.gpkg"
    gpd.GeoDataFrame(
        {"id": [1, 2, 3], "ig_date": ["2020-08-01", "2021-06-10", "2021-09-05"],
         "last_date": ["2020-08-09", "2021-06-12", "2021-09-30"], "ig_year": [2020, 2021, 2021]},
        geometry=[box(-1.0e6, 1.9e6, -0.99e6, 1.91e6), box(-0.5e6, 1.5e6, -0.49e6, 1.51e6),
                  box(0, 1.0e6, 1e4, 1.01e6)], crs="EPSG:5070",
    ).to_file(fp, driver="GPKG")

    g = events.fired_events(fp)
    assert list(g.columns) == ["id", "ig_date", "last_date", "geometry"] and len(g) == 3
    assert g["last_date"].iloc[0] == pd.Timestamp("2020-08-09")
    assert list(events.fired_events(fp, ids=[1, 3])["id"]) == [1, 3]
    assert list(events.fired_events(fp, years=2021)["id"]) == [2, 3]
    assert list(events.fired_events(fp, years=(2020, 2021), ids=[2])["id"]) == [2]
    w, s, e, n = gpd.GeoSeries([box(-1.0e6, 1.9e6, -0.99e6, 1.91e6)], crs="EPSG:5070").to_crs(4326).total_bounds
    assert list(events.fired_events(fp, bbox=(w, s, e, n))["id"]) == [1]


def test_wfigs_query_and_paging(monkeypatch):
    calls = []

    class R:
        def __init__(self, payload):
            self.payload, self.text = payload, str(payload)

        def json(self):
            return self.payload

    def get(url, params=None, timeout=None):
        calls.append(("GET", url, params))
        if params is None:
            return R({"maxRecordCount": 2})
        return R({"objectIdFieldName": "OBJECTID", "objectIds": [1, 2, 3]})

    def post(url, data=None, timeout=None):
        calls.append(("POST", url, data))
        ids = [int(i) for i in data["where"].split("(")[1].rstrip(")").split(",")]
        return R({"features": [{"type": "Feature", "properties": {"poly_IncidentName": f"F{i}"},
                                "geometry": {"type": "Point", "coordinates": [-105.0, 38.0 + i]}} for i in ids]})

    monkeypatch.setattr(events.requests, "get", get)
    monkeypatch.setattr(events.requests, "post", post)
    g = events.fetch_wfigs_perimeters("O'Brien", bbox=BBOX, which="history")
    assert len(g) == 3 and g.crs.to_epsg() == 4326
    assert [c[0] for c in calls] == ["GET", "GET", "POST", "POST"]       # 3 ids, 2 per page
    assert calls[0][1].endswith("WFIGS_Interagency_Perimeters/FeatureServer/0?f=pjson")
    assert "LIKE '%O''Brien%'" in calls[1][2]["where"] and calls[1][2]["inSR"] == 4326
    with pytest.raises(ValueError, match="which"):
        events.fetch_wfigs_perimeters(which="archive")


# ── Plots ──────────────────────────────────────────────────────────────────────

def test_plot_grid_scales():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colors

    grid = _two_cells().assign(v=[0.0, 100.0], t=[T0, T0 + pd.Timedelta(hours=36)])
    assert isinstance(plot.color_norm([1, 10], "linear"), colors.Normalize)
    log = plot.color_norm([0, 2, 50], "log")
    assert isinstance(log, colors.LogNorm) and (log.vmin, log.vmax) == (2, 50)       # zeros below the scale
    assert plot.color_norm([0, 4], "sqrt", vmax=16).vmax == 16
    with pytest.raises(ValueError, match="scale"):
        plot.color_norm([1, 2], "ln")
    with pytest.raises(ValueError, match="No values"):
        plot.color_norm([0, 0], "log")

    fig, ax = plot.plot_grid(grid, "v", scale="log", events=grid)
    assert len(ax.collections[0].get_paths()) == 1                                    # the zero cell is not drawn
    fig2, ax2 = plot.plot_grid(grid, "t")
    assert fig2.axes[-1].get_ylabel() == "days since 2025-07-01"
    assert plot.plot_grid(grid, "v", ax=ax2)[0] is fig2


# ── Treatments ─────────────────────────────────────────────────────────────────

def test_treatment_detections_footprint_and_centre(capsys):
    trt = gpd.GeoDataFrame(
        {"SOURCE_UID": ["burned", "small", "far"], "ACTIVITY": ["Thin", "Rx Fire", "Thin"],
         "YEAR_COMP": [2020, 2022, 2019], "other": 1},
        geometry=[box(0, 0, 400, 400), box(450, 150, 500, 250), box(5000, 5000, 5400, 5400)], crs="EPSG:5070",
    )
    # two 400 m boxes on "burned" (same overpass, different FRP), one a day later next to "small":
    # its box (x 300-700) overlaps "small", its centre (500.5, 200) does not fall in "burned".
    d = _det5070([(200, 200), (200, 200), (500.5, 200)], [0, 1, 1440], frp=[10.0, 30.0, 8.0])
    d.loc[2, "satellite"] = "VIIRS NOAA-20"

    f = treatments.treatment_detections(trt, d, frp_stats=("mean", "max", "p50")).set_index("SOURCE_UID")
    assert "2 of 3 polygons" in capsys.readouterr().out
    assert list(f.columns[:2]) == ["ACTIVITY", "YEAR_COMP"] and "other" not in f       # keep_cols only
    assert f["detected"].to_dict() == {"burned": True, "small": True, "far": False}
    b = f.loc["burned"]
    assert (b["n_detections"], b["n_overpasses"]) == (3, 2) and b["satellites"] == "VIIRS NOAA-20|VIIRS S-NPP"
    assert b["first_detection"] == T0 and b["last_detection"] == T0 + pd.Timedelta(days=1)
    assert b["frp_mean_VIIRS"] == pytest.approx(16.0) and b["frp_max_VIIRS"] == 30.0 and b["frp_p50_VIIRS"] == 10.0
    assert b["cover_fraction"] == pytest.approx(1.0)                                   # boxes cover the unit
    assert f.loc["small", "cover_fraction"] == pytest.approx(1.0) and f.loc["small", "frp_mean_VIIRS"] == 8.0
    assert f.loc["far", "n_detections"] == 0 and pd.isna(f.loc["far", "first_detection"])
    assert f.loc["far", "cover_fraction"] == 0 and f.crs == trt.crs and len(f) == 3

    c = treatments.treatment_detections(trt, d, method="centre").set_index("SOURCE_UID")
    assert c["n_detections"].to_dict() == {"burned": 2, "small": 0, "far": 0} and "cover_fraction" not in c
    assert treatments.treatment_detections(trt, d, method="centre", buffer_m=120)["n_detections"].tolist() == [3, 1, 0]

    day1 = treatments.treatment_detections(trt, d, end="2025-07-01").set_index("SOURCE_UID")
    assert day1["n_detections"].to_dict() == {"burned": 2, "small": 0, "far": 0}
    boxes = footprints.footprint_boxes(d)                                              # boxes in, same answer
    assert treatments.treatment_detections(trt, boxes)["n_detections"].tolist() == [3, 1, 0]
    noid = treatments.treatment_detections(trt.drop(columns="SOURCE_UID"), d, id_col=None)
    assert noid["treatment_id"].tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="method"):
        treatments.treatment_detections(trt, d, method="overlap")


def test_plot_detections_panels_and_colours():
    import matplotlib
    matplotlib.use("Agg")

    d = _det5070([(0, 0), (500, 0), (0, 500), (500, 500)], [0, 100, 1440, 1500], frp=[0.0, 5.0, 50.0, 500.0],
                 sat=["VIIRS S-NPP", "VIIRS S-NPP", "VIIRS NOAA-20", "VIIRS NOAA-21"])
    ev = gpd.GeoDataFrame(geometry=[box(-200, -200, 900, 900)], crs="EPSG:5070")

    fig, axes = plot.plot_detections(d, "frp", by="satellite", scale="log", events=ev)
    assert len(axes) == 3 and len(fig.axes) == 4                                      # 3 panels + one colour bar
    assert len({a.get_xlim() for a in axes}) == 1 and axes[0].get_xlim() == (-244.0, 944.0)
    assert [a.texts[0].get_text().split("  ")[0] for a in axes] == ["VIIRS NOAA-20", "VIIRS NOAA-21", "VIIRS S-NPP"]
    assert axes[2].texts[0].get_text().endswith("(n = 1)")                            # the zero-FRP point is off a log scale

    fig, ax = plot.plot_detections(d, "time", tz="America/Denver")
    assert fig.axes[-1].get_ylabel() == "days since 2025-07-01"
    fig, ax = plot.plot_detections(d, "satellite")
    assert [h.get_label() for h in ax.get_legend().legend_handles] == sorted(d["satellite"].unique())
    fig, ax = plot.plot_detections(footprints.footprint_boxes(d), "frp", ax=ax)       # boxes, existing axes
    assert fig is ax.figure


def test_plot_detection_series_days_and_overpasses():
    import matplotlib
    matplotlib.use("Agg")

    # T0 is 14:00 MDT on 2025-07-01: three detections that day, one the next local day
    d = _det5070([(0, 0)] * 4, [0, 100, 9 * 60, 30 * 60], frp=[1.0, 2.0, 4.0, 8.0],
                 sat=["VIIRS S-NPP", "VIIRS NOAA-20", "VIIRS NOAA-20", "VIIRS S-NPP"])
    fig, ax = plot.plot_detection_series(d, tz="America/Denver", cumulative=True)
    bars = {c.get_label(): [r.get_height() for r in c] for c in ax.containers}
    assert bars == {"VIIRS NOAA-20": [2, 0], "VIIRS S-NPP": [1, 1]}
    assert ax.get_xlabel() == "days since 2025-07-01" and ax.get_ylabel() == "detections"
    assert list(fig.axes[1].lines[0].get_ydata()) == [75.0, 100.0]                     # cumulative share

    utc = plot.plot_detection_series(d, "frp_sum", by=None)[1]                         # UTC days: Jul 1, 2 and 3
    assert [r.get_height() for r in utc.containers[0]] == [3.0, 4.0, 8.0] and utc.get_ylabel() == "FRP (MW)"
    op = plot.plot_detection_series(d, step="overpass")[1]
    assert sum(len(c) for c in op.containers) == 8                                     # 4 overpasses x 2 satellites
    assert plot.plot_detection_series(d, t0="2025-06-30", tz="America/Denver")[1].containers[0][0].get_x() > 0.5
    with pytest.raises(ValueError, match="step"):
        plot.plot_detection_series(d, step="hour")


def test_plot_footprint_workflow_panels_and_zoom():
    import matplotlib
    matplotlib.use("Agg")

    d = _det5070([(240, 240), (480, 240), (700, 300)], [0, 100, 200], frp=[10.0, 20.0, 30.0],
                 sat=["VIIRS S-NPP", "VIIRS NOAA-20", "VIIRS NOAA-20"])
    grid = _two_cells()
    frp = grids.gridded_frp(footprints.footprint_boxes(d), grid)

    fig, axes = plot.plot_footprint_workflow(d, grid_gdf=frp, events=grid)
    assert len(axes) == 3 and [a.texts[0].get_text() for a in axes] == ["(a)", "(b)", "(c)"]
    assert len({(a.get_xlim(), a.get_ylim()) for a in axes}) == 1                      # one shared frame
    assert any(t.get_text().endswith(" km") for t in axes[0].texts)                    # scale bar
    assert len(axes[1].collections[0].get_paths()) == 3                                # every footprint drawn

    fig, axes = plot.plot_footprint_workflow(d, extent=(0, 0, 400, 480))               # no grid: two panels, zoomed
    assert len(axes) == 2 and axes[0].get_xlim() == (0.0, 400.0)
