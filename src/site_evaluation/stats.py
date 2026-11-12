"""
Site statistics.

site_table() describes each site from its data alone (no model run): stages, flows, reach length, domain size and how
much of it the reference floods. It is cached in results/site_stats.csv.

attributes() runs every site once and adds what ARC computes: slopes, top widths and depths at the bankfull flow,
floodplain width (HAND), and the land cover of the floodplain. It writes results/site_attributes_<mapper>.csv.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, params, scoring, sites


def _read(path):
    from osgeo import gdal
    gdal.UseExceptions()
    ds = gdal.Open(str(path))
    return ds.GetRasterBand(1).ReadAsArray(), ds


def _cell_area_m2(ds, rows: int) -> tuple[float, float, float]:
    """A cell's area (m²) and the grid's centre (lat, lon), for a geographic or a projected grid."""
    gt = ds.GetGeoTransform()
    lat0 = gt[3] + gt[5] * rows / 2
    lon0 = gt[0] + gt[1] * ds.RasterXSize / 2
    if abs(gt[1]) < 1:      # degrees
        return abs(gt[1] * gt[5]) * 111320.0 ** 2 * np.cos(np.radians(lat0)), lat0, lon0
    return abs(gt[1] * gt[5]), np.nan, np.nan


def site_row(site: str) -> dict:
    import geopandas as gpd
    bm = sites.flow_stats(site)
    flows = sites.stage_flows(site)
    stages = list(flows.columns)
    main = bm.loc[bm["bankfull"].idxmax()]
    top_q = float(flows.loc[int(main.COMID)].max()) if int(main.COMID) in flows.index else np.nan
    row = {"site": site, "name": sites.short_name(site, 60), "year": sites.site_year(site),
           "split": "hold" if site in sites.holdout_sites() else "calib",
           "stages": len(stages), "stage_low": stages[0], "stage_high": stages[-1],
           "groups": len(sites.q_stage(site)), "reaches": len(bm),
           "baseflow_cms": float(main.baseflow), "bankfull_cms": float(main.bankfull), "max_cms": float(main["max"]),
           "top_over_bankfull": top_q / float(main.bankfull) if main.bankfull > 0 else np.nan}
    # reach length of the flowing reaches
    try:
        st = gpd.read_parquet(sites.site_dir(site) / "streams.parquet")
        st = st[st["LINKNO"].astype("int64").isin(set(bm.COMID.astype("int64")))]
        row["stream_km"] = float(st.to_crs(st.estimate_utm_crs()).geometry.length.sum() / 1000)
        row["da_km2"] = float(st["DSContArea"].max() / 1e6) if "DSContArea" in st else np.nan
    except Exception:
        row["stream_km"] = row["da_km2"] = np.nan
    # the scoring domain and how much of it the reference floods at the lowest and highest stage
    bnd_file = sites.boundary_raster(site)
    if bnd_file is not None:
        bnd, ds = _read(bnd_file)
        dom = bnd > 0
        area, row["lat"], row["lon"] = _cell_area_m2(ds, bnd.shape[0])
        row["domain_km2"] = float(dom.sum() * area / 1e6)
        for label, stage in (("low", stages[0]), ("high", stages[-1])):
            r, _ = _read(sites.reference_map(site, stage))
            wet = int(((r > 0) & dom).sum()) if r.shape == dom.shape else None
            row[f"ref_flooded_{label}"] = wet / max(dom.sum(), 1) if wet is not None else np.nan
            if label == "high" and wet is not None and np.isfinite(row["stream_km"]) and row["stream_km"] > 0:
                # the reference's flooded area per metre of flowing stream at the top stage
                row["ref_width_high_m"] = wet * area / (row["stream_km"] * 1000)
    return row


def site_table(refresh: bool = False, log=None) -> pd.DataFrame:
    """Every site's statistics (cached in results/site_stats.csv; refresh rebuilds it)."""
    path = config.results_path("site_stats.csv")
    if path.exists() and not refresh:
        df = pd.read_csv(path)
        if set(df["site"]) == set(sites.all_sites()):
            return df
    rows = []
    for s in sites.all_sites():
        try:
            rows.append(site_row(s))
        except Exception as e:
            rows.append({"site": s, "name": sites.short_name(s, 60), "error": f"{type(e).__name__}: {e}"[:200]})
        if log:
            log(f"  {s}")
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


# --------------------------------------------------------------------------------------------------------------------
# Attributes from a model run
# --------------------------------------------------------------------------------------------------------------------

def _interp_at(q, vals, q0):
    q = np.asarray(q, float)
    o = np.argsort(q)
    return float(np.interp(q0, q[o], np.asarray(vals, float)[o]))


def site_attributes(site: str, run_dir: Path) -> dict:
    """ARC's view of a site's flowing reaches, from one run's folder (VDT, burned DEM, moved streams, land cover)."""
    import geopandas as gpd
    from scipy import ndimage
    bm = sites.flow_stats(site)
    comids = set(bm.COMID.astype("int64"))
    vdt = pd.read_parquet(next((run_dir / "VDT").glob("*VDT_Database_Bathy.parquet")))
    vdt = vdt[vdt.COMID.astype("int64").isin(comids)]
    m = gpd.read_file(run_dir / "STRM" / f"{config.dem()}_matched.gpkg")
    m = m[m.LINKNO.astype("int64").isin(comids)]
    mu = m.to_crs(m.estimate_utm_crs())

    def ends(g):
        line = g if g.geom_type == "LineString" else list(g.geoms)[0]
        last = g if g.geom_type == "LineString" else list(g.geoms)[-1]
        return np.hypot(*(np.array(last.coords[-1]) - np.array(line.coords[0])))

    q_cols = sorted([c for c in vdt.columns if re.fullmatch(r"q_\d+", c)], key=lambda c: int(c[2:]))
    tw, dep = [], []
    for _, r in vdt.iterrows():
        qs = r[q_cols].to_numpy(float)
        if not np.isfinite(qs).any() or r.QBaseflow <= 0:
            continue
        tw.append(_interp_at(qs, r[[c.replace("q_", "t_") for c in q_cols]].to_numpy(float), r.QBaseflow))
        dep.append(_interp_at(qs, r[[c.replace("q_", "wse_") for c in q_cols]].to_numpy(float), r.QBaseflow) - r.Elev)
    out = {"site": site, "slope": float(vdt.Slope.median()), "valley_slope": float(vdt.Valley_Slope.median()),
           "sinuosity_xs": float((vdt.Valley_Slope / vdt.Slope.clip(lower=1e-6)).median()),
           "sinuosity_line": float(mu.geometry.length.sum() / mu.geometry.apply(ends).sum()),
           "tw_bankfull_m": float(np.median(tw)) if tw else np.nan,
           "d_bankfull_m": float(np.median(dep)) if dep else np.nan,
           "strm_order": int(m.strmOrder.max()) if "strmOrder" in m else np.nan}
    out["wd_ratio"] = out["tw_bankfull_m"] / max(out["d_bankfull_m"], 0.05)
    # height above the nearest flowing stream cell, on the burned DEM ARC ran on
    dem, ds = _read(run_dir / "DEM_Updated" / f"{config.dem()}_fixed.tif")
    dem = dem.astype(float)
    strm = np.zeros(dem.shape, bool)
    strm[vdt.Row.astype(int), vdt.Col.astype(int)] = True
    _, (ir, ic) = ndimage.distance_transform_edt(~strm, return_indices=True)
    hand = dem - dem[ir, ic]
    bnd, _ = _read(sites.boundary_raster(site))
    dom = bnd > 0
    area, _, _ = _cell_area_m2(ds, dem.shape[0])
    stream_m = mu.geometry.length.sum()
    for h in (2, 5):
        out[f"hand_width{h}_m"] = float(((hand < h) & dom).sum() * area / stream_m)
    out["confinement"] = out["hand_width5_m"] / max(out["tw_bankfull_m"], 1.0)
    out["relief_p50_m"] = float(np.median(hand[dom]))
    lc, _ = _read(run_dir / "LAND" / f"{config.dem()}_LAND_Raster.tif")
    fp = (hand < 5) & dom
    tot = max(fp.sum(), 1)
    for name, code in {"tree": 10, "grass": 30, "crop": 40, "urban": 50, "water": 80}.items():
        out[f"lc_{name}_fp"] = float((lc[fp] == code).sum() / tot)
    return out


def attributes(mapper: str, p: dict | None = None, workers: int = 8, log=print) -> pd.DataFrame:
    """Run every site once (method "attributes") and compute site_attributes from each worker's folder."""
    from .runner import Runner
    p = params.complete(p or {}, mapper)
    with Runner(mapper, workers) as runner:
        recs = runner.evaluate(p, sites.all_sites(), None, method="attributes", tag="attributes")
    rows = []
    for s, r in recs.items():
        try:
            a = site_attributes(s, Path(r["slot"]) / s)
        except Exception as e:
            log(f"  {s}: FAILED {type(e).__name__}: {e}")
            continue
        a["score"] = scoring.site_score(r)
        rows.append(a)
    df = pd.DataFrame(rows)
    path = config.results_path(f"site_attributes_{mapper}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
