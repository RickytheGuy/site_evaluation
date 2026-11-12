"""
Scoring a modelled flood map against its reference.

Each map is scored over every valid pixel of the site's scoring domain (its boundary raster), on the reference's
grid. Permanent water (land-cover water) counts like any other pixel: it is the river at low flows. The contingency
table gives the metrics in STATS_NAMES. A site's score is its mean MCC over its scored stages (site_score); a stage
that failed counts 0, and so does a failed site.

This is the "exact" scoring of ~/scripts/optimize_fimsites.py, which the October and November studies used.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from . import config, sites

#: "far" is the false alarm ratio, fp / (tp + fp) = 1 - precision (not the false alarm rate fp / (fp + tn))
STATS_NAMES = ["pc", "bias", "recall", "precision", "kappa", "f1", "csi", "far", "mcc"]
#: The metrics stored in the archive for each stage
KEPT_METRICS = ["mcc", "csi", "bias", "recall", "precision"]


# --------------------------------------------------------------------------------------------------------------------
# Rasters: grid- and NoData-aware
# --------------------------------------------------------------------------------------------------------------------

def _open(path) -> "gdal.Dataset":
    from osgeo import gdal
    gdal.UseExceptions()
    if isinstance(path, gdal.Dataset):
        return path
    if Path(str(path)).suffix.lower() not in sites.RASTER_SUFFIXES:
        raise ValueError(f"Not a raster: {path}")
    ds = gdal.Open(str(path))
    if ds is None:
        raise FileNotFoundError(f"GDAL could not open {path}")
    return ds


def _grid(ds) -> tuple:
    return ds.RasterXSize, ds.RasterYSize, tuple(ds.GetGeoTransform()), ds.GetProjectionRef()


def _same_grid(a, b, tol: float = 1e-6) -> bool:
    """Same size, geotransform and projection: only then can two rasters be compared pixel for pixel."""
    from osgeo import osr
    ax, ay, agt, aproj = _grid(a)
    bx, by, bgt, bproj = _grid(b)
    if (ax, ay) != (bx, by):
        return False
    if any(abs(u - v) > tol * max(1.0, abs(v)) for u, v in zip(agt, bgt)):
        return False
    if aproj and bproj:
        return bool(osr.SpatialReference(wkt=aproj).IsSame(osr.SpatialReference(wkt=bproj)))
    return aproj == bproj


def binary_and_valid(ds, threshold: float = 0) -> tuple[np.ndarray, np.ndarray]:
    """"Flooded" (value > threshold) and "a real observation" (not NoData, not NaN), kept apart so padding never
    counts as evidence either way."""
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    nodata = band.GetNoDataValue()
    valid = np.ones(arr.shape, dtype=bool)
    if np.issubdtype(arr.dtype, np.floating):
        valid &= np.isfinite(arr)
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        valid &= arr != nodata
    binary = np.zeros(arr.shape, dtype=np.uint8)
    np.greater(arr, threshold, out=binary, where=valid, casting="unsafe")
    binary[~valid] = 0
    return binary, valid


def _resample_to(binary: np.ndarray, valid: np.ndarray, src, target) -> tuple[np.ndarray, np.ndarray]:
    """A classified raster warped onto the target grid (mode), with its validity mask warped alongside."""
    from osgeo import gdal
    if not src.GetProjectionRef():
        raise ValueError("The raster has no projection, so it cannot be warped onto the reference grid.")
    mem = gdal.GetDriverByName("MEM").Create("", src.RasterXSize, src.RasterYSize, 2, gdal.GDT_Byte)
    mem.SetGeoTransform(src.GetGeoTransform())
    mem.SetProjection(src.GetProjectionRef())
    mem.GetRasterBand(1).WriteArray(binary)
    mem.GetRasterBand(2).WriteArray(valid.astype(np.uint8))
    tx, ty, gt, proj = _grid(target)
    minx, maxy = gt[0], gt[3]
    maxx, miny = gt[0] + tx * gt[1], gt[3] + ty * gt[5]
    if miny > maxy:
        miny, maxy = maxy, miny
    warped = gdal.Warp("", mem, options=gdal.WarpOptions(
        format="MEM", outputType=gdal.GDT_Byte, width=tx, height=ty, outputBounds=(minx, miny, maxx, maxy),
        outputBoundsSRS=proj, dstSRS=proj, resampleAlg="mode"))
    if warped is None:
        raise ValueError("gdal.Warp failed to put the raster on the reference grid")
    out = warped.ReadAsArray()
    return (out[0] > 0).astype(np.uint8), out[1] > 0


def load_on_reference_grid(floodmap, reference, boundary=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The modelled map, the reference and the scoring domain, all on the reference's grid, as 0/1 arrays. The domain
    is the boundary where both maps have data. boundary may be a raster path or a 0/1 array on the reference grid."""
    ref_ds = _open(reference)
    ref_bin, ref_valid = binary_and_valid(ref_ds)
    shape = ref_bin.shape
    if isinstance(floodmap, np.ndarray):
        if floodmap.shape != shape:
            raise ValueError(f"Map shape {floodmap.shape} is not the reference grid's {shape}")
        flood_bin, flood_valid = (floodmap > 0).astype(np.uint8), np.ones(shape, bool)
    else:
        flood_ds = _open(floodmap)
        flood_bin, flood_valid = binary_and_valid(flood_ds)
        if not _same_grid(flood_ds, ref_ds):
            flood_bin, flood_valid = _resample_to(flood_bin, flood_valid, flood_ds, ref_ds)
    domain = ref_valid & flood_valid
    if boundary is not None:
        if isinstance(boundary, np.ndarray):
            if boundary.shape != shape:
                raise ValueError(f"Boundary shape {boundary.shape} is not the reference grid's {shape}")
            domain &= boundary > 0
        else:
            bnd_ds = _open(boundary)
            bnd_bin, bnd_valid = binary_and_valid(bnd_ds)
            if not _same_grid(bnd_ds, ref_ds):
                bnd_bin, bnd_valid = _resample_to(bnd_bin, bnd_valid, bnd_ds, ref_ds)
            domain &= (bnd_bin > 0) & bnd_valid
    return flood_bin, ref_bin, domain.astype(np.uint8)


# --------------------------------------------------------------------------------------------------------------------
# Metrics. Every count is float64: MCC's denominator overflows int64 at full-raster scale.
# --------------------------------------------------------------------------------------------------------------------

def metrics(tp: float, tn: float, fp: float, fn: float) -> dict[str, float]:
    """Every metric of STATS_NAMES from one contingency table."""
    tp, tn, fp, fn = (float(v) for v in (tp, tn, fp, fn))
    total = max(tp + tn + fp + fn, 1e-8)
    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    p_chance = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total * total)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "pc": (tp + tn) / total,
        "bias": (tp + fp) / max(tp + fn, 1e-8),
        "recall": recall,
        "precision": precision,
        "kappa": ((tp + tn) / total - p_chance) / max(1 - p_chance, 1e-8),
        "f1": 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall),
        "csi": tp / max(tp + fp + fn, 1e-8),
        "far": fp / max(tp + fp, 1e-8),
        "mcc": 0.0 if denom == 0 else (tp * tn - fp * fn) / denom,
    }


def contingency(flood: np.ndarray, ref: np.ndarray, domain: np.ndarray) -> tuple[int, int, int, int]:
    """(tp, tn, fp, fn) over the domain."""
    keep = domain > 0
    f, r = flood[keep] > 0, ref[keep] > 0
    tp = int(np.count_nonzero(f & r))
    fp = int(np.count_nonzero(f & ~r))
    fn = int(np.count_nonzero(~f & r))
    tn = int(f.size - tp - fp - fn)
    return tp, tn, fp, fn


def score_map(floodmap, reference, boundary=None) -> dict[str, float]:
    """Every metric of one map against its reference; NaN when the domain has no flooded or no dry reference pixel."""
    flood, ref, domain = load_on_reference_grid(floodmap, reference, boundary)
    tp, tn, fp, fn = contingency(flood, ref, domain)
    if tp + fn == 0 or tn + fp == 0:
        return {k: float("nan") for k in STATS_NAMES}
    return metrics(tp, tn, fp, fn)


# --------------------------------------------------------------------------------------------------------------------
# A site's maps
# --------------------------------------------------------------------------------------------------------------------

def floodmap_files(output_dir: Path) -> dict[float, Path]:
    """The flood maps of one site's run folder (<output_dir>/FloodMap/*stage=<key>.tif), by stage value."""
    out = {}
    for f in Path(output_dir, "FloodMap").glob("*stage=*.tif"):
        if "WSE" in f.name:
            continue
        out[round(sites.stage_to_float(f.stem.split("stage=")[1]), 6)] = f
    return out


def score_site(site: str, output_dir: Path, stages: list[str]) -> list[dict]:
    """
    Each requested stage's metrics for one site's run, lowest stage first: {"stage", "status", **metrics}. A stage
    without a map is "missing"; one that failed to score is "error: ...". Maps of other stages left in the folder are
    ignored (stale maps were once scored by mistake).
    """
    maps = floodmap_files(output_dir)
    bnd = sites.boundary_raster(site)
    boundary = (_open(bnd).ReadAsArray() > 0) if bnd is not None else None
    nan = {k: float("nan") for k in STATS_NAMES}
    rows = []
    for stage in sorted(stages, key=sites.stage_to_float):
        fmap = maps.get(round(sites.stage_to_float(stage), 6))
        if fmap is None:
            rows.append({"stage": stage, "status": "missing", **nan})
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = score_map(fmap, sites.reference_map(site, stage), boundary)
            status = "degenerate" if any(np.isnan(v) for v in m.values()) else "ok"
        except Exception as exc:
            m, status = nan, f"error: {type(exc).__name__}: {exc}"[:200]
        rows.append({"stage": stage, "status": status, **m})
    return rows


def site_score(rec: dict, stages: list[str] | None = None) -> float:
    """A site's mean MCC over its scored stages (or the given ones): a missing or NaN stage counts 0, a site with no
    stages 0."""
    if not rec.get("stages"):
        return 0.0
    vals = [v if v is not None else 0.0 for s, v in zip(rec["stages"], rec["mcc"]) if stages is None or s in stages]
    return float(np.mean(vals)) if vals else 0.0


def objective(recs: dict[str, dict]) -> dict:
    """
    The study objective of one parameter set's records: "value", the mean over the calibration sites of each site's
    score; "holdout", the same over the hold-out sites; "all", over every site given. Only sites present in recs
    count. Also each site's score and the failed sites.
    """
    scores = {s: site_score(r) for s, r in recs.items()}
    calib = [scores[s] for s in sites.calib_sites() if s in scores]
    hold = [scores[s] for s in sites.holdout_sites() if s in scores]
    return {
        "value": float(np.mean(calib)) if calib else float("nan"),
        "holdout": float(np.mean(hold)) if hold else float("nan"),
        "all": float(np.mean(list(scores.values()))) if scores else float("nan"),
        "n_sites": len(scores),
        "site_scores": {s: round(v, 5) for s, v in scores.items()},
        "failed": [s for s, r in recs.items() if not r.get("ok")],
    }
