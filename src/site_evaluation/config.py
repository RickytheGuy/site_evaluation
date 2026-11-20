"""
Where the data lives. Every model setting is in the settings file (settings.yaml, see settings.py).

The sites, cache and working copies can be moved with an environment variable (SITE_EVAL_SITES, SITE_EVAL_CACHE,
SITE_EVAL_WORK). The results folder is the settings file's (its top-level `results`); ``site-eval --results DIR``
replaces it for one command. Code reads these when it runs, so setting them at runtime works.
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
#: numba's compiled functions, one folder per version of the model code (pipeline.numba_cache_dir)
NUMBA_ROOT = Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache").expanduser() / "site_evaluation" / "numba"
#: A results folder that replaces the settings file's (site-eval --results); None: the settings file's
RESULTS_OVERRIDE: Path | None = None

#: Worker processes for running and scoring sites (run, optimize, sensitivity): one per core the process may use
#: (Ricky, 2026-11-20: use every core). Measured 2026-11-19 on this machine (i9-13900: 8 performance cores with 2
#: threads each and 16 efficiency cores, 32 threads, two memory channels), every worker kept busy: 1.27 USGS site runs
#: a second on 8 workers, 1.50 on 12, 1.61 on 16, 1.66 on 20, 1.63 on 24 and 1.65 on 30. The runs are bound by memory
#: bandwidth, so past ~16 workers a run slows down about as much as workers are added: every core is busy, but the
#: work isn't done much sooner. A fresh worker holds 0.6 GB between runs and 2.2 GB running Harrisburg (USGS).
WORKERS = (os.process_cpu_count() if hasattr(os, "process_cpu_count") else os.cpu_count()) or 1
#: Site runs a worker does before a fresh process takes its place, a guard against workers that grow. In the USGS
#: screening of 2026-11-20 the median run peaked at 1.6 GB in the first 20 minutes and at 3.2 GB four hours later
#: (4.9 GB on Harrisburg), mostly because the memory runs freed stayed in the threads' malloc arenas, now two
#: (pipeline.worker_environment). With two, a site's peak still rose 7% over 8 parameter sets (16 workers, no
#: replacing). A fresh worker costs about 2.7 s, 0.3% of 50 runs with every core busy.
WORKER_RUNS = 50

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


def results_root() -> Path:
    """The evaluation archives, optimization studies, screening studies, kept maps, site tables and figures: the
    settings file's results folder, unless --results replaces it."""
    from . import settings
    return RESULTS_OVERRIDE or settings.current().results


def results_path(*parts: str) -> Path:
    """A path under the current results folder (not created)."""
    return results_root().joinpath(*parts)
