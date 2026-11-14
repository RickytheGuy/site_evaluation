"""
nencarta for one site: its configuration, the preprocessing every run shares, and one run.

The preprocessing (streams moved and burned, land cover, water mask, flow files, FLDPLN stream info) is built once
per mapper and DEM into config.cache_dir(mapper) by ``site-eval prepare``: nencarta cuts the moved stream network at
lakes only for FLDPLN. It depends on the settings file's nencarta section and DEM, whose fingerprint the cache keeps;
rebuild it after changing them, nencarta, or a site. A run then only redoes ARC, the bathymetry, the FLDPLN library
and the maps.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import warnings
from pathlib import Path

from . import config, params, settings, sites

#: A run's products. They are deleted before every run so nencarta rebuilds them (it skips any step whose output
#: exists). The water mask (Bathymetry/*water_mask*) is preprocessing and stays.
PER_RUN = ["VDT/*", "FLDPLN/*fldpln_library*", "Bathymetry/*ARC_Bathy*", "Bathymetry/*FS_Bathy*", "FloodMap/*",
           "ARC_InputFiles/*", "FLOW/*2yr_flow_initial*", "n_trial*.txt"]
#: Left out when a worker copies the cache
SYNC_EXCLUDES = ["VDT/*", "FLDPLN/*fldpln_library*", "Bathymetry/*ARC_Bathy*", "Bathymetry/*FS_Bathy*",
                 "FloodMap/*", "ARC_InputFiles/*"]
CACHE_SETTINGS = ".settings.json"


def worker_environment() -> None:
    """One thread per worker for numba, BLAS, polars and arrow, and a fixed string hash: nencarta's make_land_cover
    warps the ESA WorldCover tiles in the order of a Python set. Set before the worker processes start."""
    for var in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "NUMEXPR_NUM_THREADS", "GDAL_NUM_THREADS",
                "ARROW_IO_THREADS", "OMP_THREAD_LIMIT"):
        os.environ[var] = "1"
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ["SITE_EVAL_SETTINGS"] = str(settings.current().path)


def watershed_dict(site: str, p: dict, mapper: str, output_dir: Path, n_file: str | None,
                   stages: int | None) -> dict:
    """nencarta's configuration for one site and one parameter set. stages: model all the site's stages (None) or k
    of them evenly spaced by rank."""
    neo, arc, c2f, lib = params.model_args(p, mapper, site)
    return {
        **neo,
        "name": site, "dem": sites.dem_file(site), "flowline": sites.site_dir(site) / "streams.parquet",
        "output_dir": output_dir, "reanalysis_file": sites.site_dir(site) / "flow_files" / config.FLOW_STATS_FILE,
        "user_flow_files": sites.site_flow_files(site, stages), "mannings_text_file": n_file,
        "floodmap_mode": "user", "specified_bathyflow_field": config.bathyflow_field(arc),
        "mapper": config.MAPPERS[mapper], "arc_args": arc, "curve2flood_args": c2f, "fldpln_library_args": lib,
    }


def _quiet_nencarta() -> None:
    try:
        from nencarta import set_log_level
        set_log_level("ERROR")
    except Exception:
        pass


def clear_run_products(site_folder: Path) -> None:
    for pattern in PER_RUN:
        for f in site_folder.glob(pattern):
            if f.is_file():
                f.unlink(missing_ok=True)


# --------------------------------------------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------------------------------------------

def prepare_site(task: tuple[str, str, str]) -> str | None:
    """Build one site's preprocessing from scratch into the cache. Returns the error, or None."""
    site, mapper, settings_path = task
    settings.use(settings_path)
    from nencarta import NencartaConfig, Workspace
    from nencarta.tasks import prepare_inputs_for_dem
    _quiet_nencarta()
    out = config.cache_dir(mapper)
    site_out = out / site
    if site_out.exists():
        shutil.rmtree(site_out)
    try:
        cfg = {**watershed_dict(site, params.default_set(mapper), mapper, out, None, None), "overwrite": True}
        ws = Workspace(NencartaConfig(cfg), sites.dem_file(site))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prepare_inputs_for_dem(ws)
    except Exception as e:
        return f"{type(e).__name__}: {e}"[:600]
    finally:
        clear_run_products(site_out)
    (site_out / CACHE_SETTINGS).write_text(json.dumps({"fingerprint": settings.current().cache_fingerprint(mapper),
                                                       "dem": settings.current().dem, "time": time.time()}))
    return None


def prepare(mapper: str, site_list: list[str], workers: int = 30, log=print) -> list[str]:
    """Rebuild the cache for these sites; the workers copy them again on next use (the cache's .token changes).
    Returns the failed sites."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    worker_environment()
    S = settings.current()
    out = config.cache_dir(mapper)
    out.mkdir(parents=True, exist_ok=True)
    (out / ".token").unlink(missing_ok=True)
    log(f"preparing {len(site_list)} sites for {mapper} on {S.dem} into {out}")
    from .progress import Progress, worker_init
    t0 = time.time()
    failed = []
    pool = ProcessPoolExecutor(min(workers, len(site_list)), mp_context=mp.get_context("forkserver"),
                               initializer=worker_init)
    try:
        with Progress(len(site_list), f"prepare {mapper}", "site", log) as bar:
            tasks = [(s, mapper, str(S.path)) for s in site_list]
            for site, err in zip(site_list, pool.map(prepare_site, tasks, chunksize=1)):
                if err:
                    bar.write(f"  {site:<70} FAILED: {err}")
                    failed.append(site)
                bar.update()
    except BaseException:
        pool.terminate_workers()
        pool.shutdown(wait=True, cancel_futures=True)
        log("Stopped: the cache is incomplete, so runs refuse it until prepare finishes.")
        raise
    pool.shutdown()
    (out / ".token").write_text(f"{time.time():.0f}")
    log(f"prepared {len(site_list) - len(failed)}/{len(site_list)} sites in {(time.time() - t0) / 60:.1f} min"
        + (f"; failed: {failed}" if failed else ""))
    return failed


def cache_token(mapper: str) -> str | None:
    f = config.cache_dir(mapper) / ".token"
    return f.read_text().strip() if f.exists() else None


def check_cache(mapper: str, site_list: list[str]) -> None:
    """Stop unless every site's cache exists and was built with the current nencarta settings and DEM (each site's
    <site>/.settings.json, written by prepare)."""
    S = settings.current()
    where = config.cache_dir(mapper)
    if cache_token(mapper) is None:
        raise SystemExit(f"No preprocessing for {mapper} on {S.dem} in {where}: run  site-eval prepare --mapper {mapper}")
    want = S.cache_fingerprint(mapper)
    stale = []
    for site in site_list:
        f = where / site / CACHE_SETTINGS
        if not f.exists() or json.loads(f.read_text()).get("fingerprint") != want:
            stale.append(site)
    if stale:
        shown = ", ".join(stale[:5]) + (f" and {len(stale) - 5} more" if len(stale) > 5 else "")
        raise SystemExit(f"{len(stale)} site(s) in {where} have no preprocessing built with the nencarta settings of "
                         f"{S.path} ({shown}): run  site-eval prepare --mapper {mapper}"
                         + ("" if len(stale) == len(site_list) else " --sites ..."))


# --------------------------------------------------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------------------------------------------------

def run_in_folder(site: str, p: dict, mapper: str, root: Path, stages: int | None) -> list[str]:
    """Run nencarta for one site in root/<site> (a copy of the cache). Returns the stage keys it modelled."""
    from nencarta import NencartaConfig, Workspace, run_pipeline
    folder = root / site
    clear_run_products(folder)
    n_file = folder / f"n_trial_{os.getpid()}.txt"
    params.write_n(p, n_file, mapper)
    ws = Workspace(NencartaConfig(watershed_dict(site, p, mapper, root, str(n_file), stages)), sites.dem_file(site))
    expected = [sites.stage_of(f) for f in ws.configs.user_flow_files]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_pipeline([ws], None)
    return expected
