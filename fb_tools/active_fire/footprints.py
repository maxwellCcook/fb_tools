"""Ground footprints of active-fire detections (ported from potr5-fb ``01c`` ``pixel_area``)."""
import pandas as pd


def footprint_boxes(gdf, crs="EPSG:5070", instruments=("VIIRS",)):
    """
    Scan × track footprint box for each detection.

    The box is centred on the detection, ``scan`` km wide (x) and ``track`` km tall
    (y), axis-aligned in ``crs`` — it is not rotated to the scan direction. Works on
    FIRMS and swath rows alike (both carry ``scan`` / ``track``).

    Parameters
    ----------
    gdf : GeoDataFrame
        Detections in the shared schema.
    crs : any
        Projected CRS the boxes are built in. Use the grid's CRS so the boxes stay
        exact rectangles when gridded.
    instruments : sequence of str or None
        Instruments kept; the rest are dropped with a printed count. None keeps all.

    Returns
    -------
    GeoDataFrame
        Polygon geometry in ``crs``, plus ``pix_area_km2`` (scan × track) and
        ``frp_wkm2`` (``frp / pix_area_km2``). Detections without ``scan`` or
        ``track`` are dropped.
    """
    import shapely

    out = gdf
    if instruments is not None:
        keep = out["instrument"].isin(instruments)
        if (~keep).any():
            print(f"footprints: dropped {(~keep).sum()} detections not in {tuple(instruments)}")
        out = out[keep]
    scan = pd.to_numeric(out["scan"], errors="coerce")
    track = pd.to_numeric(out["track"], errors="coerce")
    ok = scan.notna() & track.notna()
    if (~ok).any():
        print(f"footprints: dropped {(~ok).sum()} detections without scan/track")
    out, scan, track = out[ok].to_crs(crs), scan[ok].to_numpy(float), track[ok].to_numpy(float)

    x, y = out.geometry.x.to_numpy(), out.geometry.y.to_numpy()
    hw, hh = scan * 500.0, track * 500.0            # half width / height, m
    out = out.set_geometry(shapely.box(x - hw, y - hh, x + hw, y + hh), crs=crs)
    out["pix_area_km2"] = scan * track
    out["frp_wkm2"] = pd.to_numeric(out["frp"], errors="coerce") / out["pix_area_km2"]
    return out
