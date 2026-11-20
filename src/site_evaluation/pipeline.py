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
    """One thread per worker for numba, BLAS, polars and arrow, a fixed string hash (nencarta's make_land_cover
    warps the ESA WorldCover tiles in the order of a Python set), two malloc arenas, and numba's cache folder for
    this code (numba_cache_dir). Set before the worker processes start.

    The arenas: glibc gives each thread that allocates an arena of its own, and a worker's native threads (polars,
    GDAL and others) made about 70. The memory a run frees is scattered over them and isn't handed back, so a worker
    grew with every run, the more the more its runs' settings varied: 0.6 GB to 1.6 GB in 80 runs of mixed sites
    and settings (2026-11-20), and to 3.2 GB in four hours of the USGS screening. With two arenas the same 80 runs
    stayed at 0.6-0.7 GB, and took no longer."""
    for var in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "NUMEXPR_NUM_THREADS", "GDAL_NUM_THREADS",
                "ARROW_IO_THREADS", "OMP_THREAD_LIMIT"):
        os.environ[var] = "1"
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    os.environ["NUMBA_CACHE_DIR"] = str(numba_cache_dir())


#: The packages whose numba functions the workers compile
MODEL_PACKAGES = ("arc", "curve2flood", "nencarta")
#: Unused numba cache folders older than this many days are removed (the newest few are always kept)
NUMBA_KEEP_DAYS, NUMBA_KEEP_NEWEST = 14, 3


def numba_cache_dir() -> Path:
    """
    numba's cache folder for the model code as it is now: config.NUMBA_ROOT/<key>, the key a hash of the numba and
    Python versions and of the path, size and modification time of every .py file of MODEL_PACKAGES.

    numba recompiles a cached function only when its own file changes, not when a jitted function it calls from
    another file does. After an edit, a cached function can so run the old build of its callee against the new code:
    on 2026-11-19 Harrisburg (USGS) segfaulted in ARC's bank smoothing because bathymetry/banks.py's cache still held
    the hydraulics.py it was compiled with. A folder per version means any edit starts a cache of its own, which each
    worker fills the first time it runs a site (about a minute). Old folders are pruned (NUMBA_KEEP_DAYS).
    """
    import hashlib
    import importlib.metadata
    import importlib.util
    import platform
    h = hashlib.sha1(f"{importlib.metadata.version('numba')} {platform.python_version()}".encode())
    for name in MODEL_PACKAGES:
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.submodule_search_locations:
            h.update(f"{name}: missing".encode())
            continue
        for root in spec.submodule_search_locations:
            for folder, dirs, files in os.walk(root):
                dirs[:] = sorted(d for d in dirs if d != "__pycache__" and not d.startswith("."))
                for f in sorted(files):
                    if f.endswith(".py"):
                        st = os.stat(os.path.join(folder, f))
                        h.update(f"{os.path.relpath(os.path.join(folder, f), root)} {st.st_size} "
                                 f"{st.st_mtime_ns}\n".encode())
    path = config.NUMBA_ROOT / h.hexdigest()[:16]
    path.mkdir(parents=True, exist_ok=True)
    os.utime(path)                                   # marks it in use, for the pruning
    _prune_numba_caches(keep=path)
    return path


def _prune_numba_caches(keep: Path) -> None:
    """Remove the numba cache folders of old code: those unused for NUMBA_KEEP_DAYS, beyond the newest few."""
    try:
        folders = sorted((p for p in config.NUMBA_ROOT.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime,
                         reverse=True)
    except OSError:
        return
    cutoff = time.time() - NUMBA_KEEP_DAYS * 86400
    for p in folders[NUMBA_KEEP_NEWEST:]:
        if p != keep and p.parent == config.NUMBA_ROOT and p.stat().st_mtime < cutoff:
            shutil.rmtree(p, ignore_errors=True)


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
