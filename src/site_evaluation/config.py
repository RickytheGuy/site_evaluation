"""
Where the data lives. Every model setting is in the settings file (settings.yaml, see settings.py).

Each path can be overridden with an environment variable (SITE_EVAL_SITES, SITE_EVAL_CACHE, SITE_EVAL_WORK,
SITE_EVAL_RESULTS). The results folder can also be set per command with ``site-eval --results DIR``. Code reads these
module attributes when it runs, so setting them at runtime works.
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_path(var: str, default: str) -> Path:
    return Path(os.environ.get(var, default)).expanduser()


#: The benchmark sites: <site>/{DEMs, flow_files, reference_warped, boundary_rasters, q_stage.json, streams.parquet}
SITES_ROOT = _env_path("SITE_EVAL_SITES", "~/fim_sites")
#: The cached preprocessing, one folder per mapper and DEM (``site-eval prepare`` builds it)
CACHE_ROOT = _env_path("SITE_EVAL_CACHE", "~/fim_sites_nencarta_v3")
#: Each worker's own copy of the cache
WORK_ROOT = _env_path("SITE_EVAL_WORK", "~/fim_sites_nencarta_v3_work")
#: The evaluation archives, optimization studies, kept maps, site tables and figures
RESULTS_ROOT = _env_path("SITE_EVAL_RESULTS", "~/site_evaluation/results")

MAPPERS = {"fldpln": "Curve2Flood-FLDPLNpy", "kernel": "Curve2Flood-Kernel Weighted"}

#: The hold-out sites: HOLDOUT_COUNT drawn with this seed from every site, sorted by name
HOLDOUT_COUNT = 10
HOLDOUT_SEED = 12345

#: Each site's flow statistics, <site>/flow_files/FLOW_STATS_FILE: COMID, baseflow, bankfull and max (cms), for the
#: reaches its stage files carry. It replaced the merged reanalysis file (Ricky, 2026-11-10).
FLOW_STATS_FILE = "bm.csv"


def dem() -> str:
    """The DEM in use (the settings file's dem.name)."""
    from . import settings
    return settings.current().dem


def _folder(mapper: str) -> str:
    # FABDEM keeps the folder names it had before the settings file
    return mapper if dem() == "fabdem" else f"{mapper}_{dem()}"


def cache_dir(mapper: str) -> Path:
    return CACHE_ROOT / _folder(mapper)


def work_dir(mapper: str) -> Path:
    return WORK_ROOT / _folder(mapper)


def bathyflow_field(arc: dict) -> str:
    """The bm.csv column ARC cuts the channel to: bankfull when it keeps the channel below its banks
    (Bathy_Use_Banks), baseflow when the water surface is the DEM's (Ricky, 2026-11-10)."""
    return "bankfull" if arc.get("Bathy_Use_Banks") else "baseflow"


def results_path(*parts: str) -> Path:
    """A path under the current results folder (not created)."""
    return RESULTS_ROOT.joinpath(*parts)
