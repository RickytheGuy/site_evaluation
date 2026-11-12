"""
The benchmark sites: which there are, the calibration / hold-out split, each site's stages, flows and reference maps.

A site folder (config.SITES_ROOT/<site>) holds:
    q_stage.json                    {group: {"stages": {key: cfs}, "rivids": [...], "bankfull": cfs, "baseflow": cfs}}
    flow_files/stage=<key>.csv      COMID, cms: one per stage
    flow_files/bm.csv               COMID, baseflow, bankfull, max (cms)
    reference_warped/<dem>/*.tif    the USGS maps on the model's grid, one per stage, matched by number
    boundary_rasters/<dem>_boundary.tif, DEMs/<dem>.tif, streams.parquet
Stage keys are numbers written with "_" for "." ("13_2" is 13.2).
"""
from __future__ import annotations

import glob
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import config

RASTER_SUFFIXES = {".tif", ".tiff", ".vrt", ".img"}


# --------------------------------------------------------------------------------------------------------------------
# Which sites
# --------------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _all_sites(root: str, dem: str) -> tuple[str, ...]:
    return tuple(sorted(Path(d).parent.parent.name for d in glob.glob(os.path.join(root, "*", "DEMs", f"{dem}.tif"))))


def all_sites() -> list[str]:
    """Every site with a DEM, sorted by name."""
    return list(_all_sites(str(config.SITES_ROOT), config.dem()))


def holdout_sites() -> list[str]:
    """config.HOLDOUT_COUNT sites drawn with config.HOLDOUT_SEED, the same draw as fim3's."""
    sites = all_sites()
    draw = np.random.default_rng(config.HOLDOUT_SEED).permutation(sites)[:config.HOLDOUT_COUNT]
    return sorted(str(s) for s in draw)


def calib_sites() -> list[str]:
    hold = set(holdout_sites())
    return [s for s in all_sites() if s not in hold]


def select_sites(patterns: list[str] | None = None) -> list[str]:
    """The sites whose names contain any of the patterns. "all" (or nothing) means every site; "calib" and "holdout"
    mean the two halves of the split."""
    if not patterns or patterns == ["all"]:
        return all_sites()
    chosen = set()
    for pat in patterns:
        if pat in ("calib", "calibration"):
            chosen.update(calib_sites())
        elif pat in ("hold", "holdout"):
            chosen.update(holdout_sites())
        else:
            chosen.update(s for s in all_sites() if pat.lower() in s.lower())
    if not chosen:
        raise SystemExit(f"No site matches {patterns}")
    return [s for s in all_sites() if s in chosen]


def site_dir(site: str) -> Path:
    return config.SITES_ROOT / site


def short_name(site: str, width: int = 34) -> str:
    """'Big_Blue_River_at_Shelbyville_(2016)' -> 'Big Blue River at Shelbyville', cut to width."""
    name = site.split("_(")[0].replace("_", " ")
    return name if len(name) <= width else name[:width - 1] + "…"


def site_year(site: str) -> int | None:
    m = re.search(r"\((\d{4})\)", site)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------------------------------------------------

def stage_to_float(stage: str | float | int) -> float:
    """"13_2" -> 13.2, "03" -> 3.0, "11.0" -> 11.0. Sort stage keys with this, never float()."""
    if isinstance(stage, (int, float)):
        return float(stage)
    return float(str(stage).strip().replace("_", "."))


def _is_numeric_stage(stage: str) -> bool:
    try:
        stage_to_float(stage)
    except ValueError:
        return False
    return True


def stage_of(flow_file: str | Path) -> str:
    """The stage key of a flow file: '.../stage=13_2.csv' -> '13_2'."""
    return Path(flow_file).stem.split("stage=")[1]


def flow_files(site: str) -> list[str]:
    """Every stage flow file of a site, lowest stage first, after checking them against q_stage.json."""
    files = glob.glob(str(site_dir(site) / "flow_files" / "stage=*.csv"))
    check_site_stages(site_dir(site), files)
    return sorted(files, key=lambda f: stage_to_float(stage_of(f)))


def select_flow_files(files: list[str], site: str, k: int) -> list[str]:
    """k stages evenly spaced by rank from the lowest to the highest, or all of them when a site has fewer. files
    must be in stage order."""
    if not files:
        raise ValueError(f"{site} has no flow files.")
    k = min(k, len(files))
    if k == 1:
        return [files[0]]
    ranks = np.unique(np.round(np.linspace(0, len(files) - 1, k)).astype(int))
    return [files[i] for i in ranks]


def site_flow_files(site: str, k: int | None = None) -> list[str]:
    """The flow files modelled at a site: all of them (k None), or k evenly spaced by rank."""
    files = flow_files(site)
    return files if k is None else select_flow_files(files, site, k)


def site_stages(site: str, k: int | None = None) -> list[str]:
    return [stage_of(f) for f in site_flow_files(site, k)]


def check_site_stages(folder: Path, files: list[str]) -> None:
    """
    Fail loudly, naming the site, on the two ways a hand-edited site folder has gone wrong:
    - Stage keys that are not numbers. The Deerfield sites used the reference raster's stem ('chmontma_01'), and
      scoring died deep in a worker with no site named.
    - Stage files that have no q_stage.json entry. Charlemont carried 18 stage files holding another site's COMIDs,
      and ARC mapped the same default flow for every one of them.
    files must be the site's complete stage list.
    """
    import pandas as pd
    labels = [stage_of(f) for f in files]
    bad = [lab for lab in labels if not _is_numeric_stage(lab)]
    if bad:
        raise ValueError(f"{folder.name}: {len(bad)} stage key(s) are not numeric, e.g. {bad[0]!r}. q_stage.json keys "
                         "must be numbers ('01'), not raster stems ('chmontma_01').")
    q_path = folder / "q_stage.json"
    if not q_path.exists():
        return
    groups = json.loads(q_path.read_text(encoding="utf-8"))
    orphans = sorted(set(labels) - {k for g in groups.values() for k in g.get("stages", {})})
    if not orphans:
        return

    def comids(path: Path) -> set:
        col = pd.read_csv(path)["COMID"]
        return set(pd.to_numeric(col, errors="coerce").dropna().astype("int64"))

    try:
        own = comids(folder / "flow_files" / config.FLOW_STATS_FILE)
        foreign = [o for o in orphans if not comids(folder / "flow_files" / f"stage={o}.csv") & own]
    except (OSError, KeyError):
        foreign = []
    shown = ", ".join(orphans[:8]) + (" ..." if len(orphans) > 8 else "")
    raise ValueError(
        f"{folder.name}: {len(orphans)} stage file(s) have no q_stage.json entry ({shown})."
        + (f" {len(foreign)} of them hold no COMID of this site, so they came from another site." if foreign else "")
        + " Delete them, or add them to q_stage.json.")


# --------------------------------------------------------------------------------------------------------------------
# Flows
# --------------------------------------------------------------------------------------------------------------------

def q_stage(site: str) -> dict:
    return json.loads((site_dir(site) / "q_stage.json").read_text(encoding="utf-8"))


def flow_stats(site: str):
    """bm.csv as a DataFrame: COMID, baseflow, bankfull, max (cms)."""
    import pandas as pd
    return pd.read_csv(site_dir(site) / "flow_files" / config.FLOW_STATS_FILE)


def stage_flows(site: str):
    """Every stage's flow at every reach (cms): a DataFrame indexed by COMID with one column per stage key."""
    import pandas as pd
    cols = {}
    for f in flow_files(site):
        d = pd.read_csv(f)
        cols[stage_of(f)] = d.set_index("COMID").iloc[:, 0]
    return pd.DataFrame(cols)


# --------------------------------------------------------------------------------------------------------------------
# Reference maps
# --------------------------------------------------------------------------------------------------------------------

_NUM_GROUP = re.compile(r"\d+(?:[._]\d+)?")
_NUM_PLAIN = re.compile(r"\d+")


def _stem(path: str | os.PathLike) -> str:
    """A file name with every raster extension removed ('PawtuxetRI_9.tif.tif' -> 'PawtuxetRI_9')."""
    name = Path(path).name
    while True:
        root, ext = os.path.splitext(name)
        if ext.lower() not in RASTER_SUFFIXES or not root:
            return name
        name = root


def _stage_candidates(stem: str) -> tuple[float | None, set[float]]:
    """The number a reference file name's stage is written as (its last number: 'glinil14_9' -> 14.9), and every
    number the name could be read as ('elev_693_8' -> {693.8, 693, 8})."""
    groups = _NUM_GROUP.findall(stem)
    if not groups:
        return None, set()
    primary = stage_to_float(groups[-1])
    every = {stage_to_float(g) for g in groups} | {float(p) for p in _NUM_PLAIN.findall(stem)}
    return primary, every


def reference_map(site: str, stage: str, dem: str | None = None) -> str:
    """
    The reference map of a stage, matched by the stage's numeric value: zero padding ('boirivid_03'), decimals
    ('glinil13_2'), constant suffixes ('elev_693_8'), doubled extensions ('PawtuxetRI_9.tif.tif') and embedded text
    all resolve. Substring matching once scored Pawtuxet's stage 9 against its stage 19 map. An ambiguous stage
    raises.
    """
    dem = dem or config.dem()
    folder = site_dir(site) / "reference_warped" / dem
    files = sorted(f for f in glob.glob(str(folder / "*")) if Path(f).suffix.lower() in RASTER_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No reference maps in {folder}")
    wanted = stage_to_float(stage)
    primary_hits, loose_hits = [], []
    for f in files:
        primary, every = _stage_candidates(_stem(f))
        if primary is not None and np.isclose(primary, wanted):
            primary_hits.append(f)
        elif any(np.isclose(v, wanted) for v in every):
            loose_hits.append(f)
    for hits, how in ((primary_hits, "stage number"), (loose_hits, "embedded number")):
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise ValueError(f"Stage {stage!r} of {site} matches {len(hits)} reference maps by {how}: "
                             f"{[Path(h).name for h in hits]}")
    raise FileNotFoundError(f"No reference map for {site} at stage {stage!r} ({dem}); "
                            f"there are {[Path(f).name for f in files]}")


def boundary_raster(site: str, dem: str | None = None) -> Path | None:
    """The scoring domain on the model's grid, if the site has one."""
    f = site_dir(site) / "boundary_rasters" / f"{dem or config.dem()}_boundary.tif"
    return f if f.exists() else None


def dem_file(site: str, dem: str | None = None) -> Path:
    return site_dir(site) / "DEMs" / f"{dem or config.dem()}.tif"
