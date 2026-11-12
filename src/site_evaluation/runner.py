"""
Running and scoring parameter sets on many sites at once, and the archive of every evaluation.

Each worker process owns a copy of the cached preprocessing (config.work_dir(mapper)/wNN), held with a lock for
the worker's lifetime, so several studies can run at once without sharing a copy. A worker copies a site from the
cache (rsync) on first use and again whenever the cache is rebuilt.

Every (site, parameter set) evaluation is appended to the archive, results/<mapper>_evals.jsonl: the parameters, each
stage's metrics, the run time, the code versions and the fixed-settings fingerprint. Any table or figure can be
recomputed from it without running again.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from . import config, params, pipeline, scoring, settings, sites

_SLOT: Path | None = None
_SLOT_LOCK = None


def _claim_slot(mapper: str) -> Path:
    """The first working copy no live process holds: an exclusive flock on wNN.lock, kept for the worker's
    lifetime (the OS drops it when the process dies)."""
    import fcntl
    global _SLOT_LOCK
    root = config.work_dir(mapper)
    root.mkdir(parents=True, exist_ok=True)
    for k in range(1000):
        f = open(root / f"w{k:02d}.lock", "a")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.close()
            continue
        _SLOT_LOCK = f
        return root / f"w{k:02d}"
    raise RuntimeError("no free working copy")


def _init_worker(mapper: str, settings_path: str) -> None:
    global _SLOT
    import sys
    import warnings
    sys.stdout = open(os.devnull, "w")       # nencarta's progress display; errors come back in the records
    settings.use(settings_path)
    _SLOT = _claim_slot(mapper)
    _SLOT.mkdir(parents=True, exist_ok=True)
    warnings.simplefilter("ignore")
    pipeline._quiet_nencarta()


def _sync_site(site: str, mapper: str) -> Path:
    """The worker's copy of a site, refreshed from the cache when missing or when the cache was rebuilt."""
    dst = _SLOT / site
    token = pipeline.cache_token(mapper) or "none"
    marker = dst / ".synced"
    if not (marker.exists() and marker.read_text() == token):
        dst.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", "--delete", *[f"--exclude={e}" for e in pipeline.SYNC_EXCLUDES],
                        f"{config.cache_dir(mapper) / site}/", f"{dst}/"], check=True)
        marker.write_text(token)
    return dst


def run_site(task: tuple) -> dict:
    """Worker: run one site with one complete parameter set, then score it. Never raises: a failed site comes back
    with ok False and its error. keep: a folder to copy the site's flood maps into."""
    site, p, mapper, stages, keep = task
    t0 = time.perf_counter()
    rec = {"site": site, "ok": False}
    try:
        _sync_site(site, mapper)
        expected = pipeline.run_in_folder(site, p, mapper, _SLOT, stages)
        rows = scoring.score_site(site, _SLOT / site, expected)
        rec["stages"] = [r["stage"] for r in rows]
        for m in scoring.KEPT_METRICS:
            rec[m] = [round(float(r[m]), 5) if np.isfinite(r[m]) else None for r in rows]
        rec["ok"] = all(r["status"] == "ok" for r in rows)
        if not rec["ok"]:
            rec["error"] = "stages: " + ", ".join(f"{r['stage']}={r['status']}" for r in rows if r["status"] != "ok")
        if keep:
            dst = Path(keep) / site / "FloodMap"
            dst.mkdir(parents=True, exist_ok=True)
            for f in (_SLOT / site / "FloodMap").glob("*.tif"):
                shutil.copy2(f, dst / f.name)
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:400]
    rec["seconds"] = round(time.perf_counter() - t0, 2)
    rec["slot"] = str(_SLOT)
    return rec


# --------------------------------------------------------------------------------------------------------------------
# The archive
# --------------------------------------------------------------------------------------------------------------------

def archive_path(mapper: str) -> Path:
    return config.results_path(f"{mapper}_evals.jsonl")


def read_archive(mapper: str) -> list[dict]:
    path = archive_path(mapper)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def code_versions() -> dict[str, str]:
    """The commit of nencarta, ARC and curve2flood as installed, with '+' when the working tree has changes."""
    import importlib.util
    out = {}
    for name in ("nencarta", "arc", "curve2flood"):
        try:
            spec = importlib.util.find_spec(name)
            folder = Path(spec.origin).parent
            head = subprocess.run(["git", "-C", str(folder), "rev-parse", "--short", "HEAD"], capture_output=True,
                                  text=True, check=True).stdout.strip()
            dirty = subprocess.run(["git", "-C", str(folder), "status", "--porcelain", "-uno"], capture_output=True,
                                   text=True).stdout.strip()
            out[name] = head + ("+" if dirty else "")
        except Exception:
            out[name] = "?"
    return out


class Archive:
    """Append-only JSONL of every evaluation."""

    def __init__(self, mapper: str):
        self.path = archive_path(mapper)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.code = code_versions()
        self.setup = settings.current().fingerprint(mapper)
        settings.current().snapshot(mapper)

    def add(self, method: str, tag: str, p: dict, rec: dict, stages_mode) -> None:
        row = {"method": method, "tag": tag, "time": time.time(), "stages_mode": stages_mode, "params": p,
               "code": self.code, "setup": self.setup, **{k: v for k, v in rec.items() if k != "slot"}}
        with self.lock, open(self.path, "a") as f:
            f.write(json.dumps(row, default=float) + "\n")


# --------------------------------------------------------------------------------------------------------------------
# The pool
# --------------------------------------------------------------------------------------------------------------------

class Runner:
    """A pool of workers, each with its own working copy. Evaluations of any (site, parameter set) can be submitted
    from several threads at once. Use it as a context manager, under ``if __name__ == "__main__":`` in a script
    (the workers start with forkserver)."""

    def __init__(self, mapper: str, workers: int | None = None):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        if mapper not in config.MAPPERS:
            raise ValueError(f"mapper must be one of {list(config.MAPPERS)}")
        pipeline.worker_environment()
        self.mapper = mapper
        self.workers = workers or max((os.cpu_count() or 4) - 2, 1)
        self.pool = ProcessPoolExecutor(self.workers, mp_context=mp.get_context("forkserver"),
                                        initializer=_init_worker,
                                        initargs=(mapper, str(settings.current().path)))
        self.archive = Archive(mapper)
        self._checked: set[str] = set()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def check(self, site_list: list[str]) -> None:
        """Stop unless these sites' preprocessing was built with the current settings (checked once per site)."""
        todo = [s for s in site_list if s not in self._checked]
        if todo:
            pipeline.check_cache(self.mapper, todo)
            self._checked.update(todo)

    def submit(self, site: str, p: dict, stages: int | None = None, keep: Path | None = None):
        """A future of one site's record. p is completed with the settings file's defaults."""
        self.check([site])
        return self.pool.submit(run_site, (site, params.complete(p, self.mapper), self.mapper, stages,
                                           str(keep) if keep else None))

    def evaluate(self, p: dict, site_list: list[str] | None = None, stages: int | None = None, method: str = "run",
                 tag: str = "", keep: Path | None = None, progress=None, archive: bool = True) -> dict[str, dict]:
        """Every site's record for one parameter set, archived under (method, tag) unless archive is False.
        progress(site, rec) is called as each site finishes."""
        site_list = site_list or sites.all_sites()
        self.check(site_list)
        p = params.complete(p, self.mapper)
        from concurrent.futures import as_completed
        futures = {self.submit(s, p, stages, keep): s for s in site_list}
        recs = {}
        for fut in as_completed(futures):
            s = futures[fut]
            recs[s] = fut.result()
            if archive:
                self.archive.add(method, tag, p, recs[s], stages)
            if progress:
                progress(s, recs[s])
        return {s: recs[s] for s in site_list}

    def objective(self, p: dict, method: str, tag: str, site_list: list[str] | None = None) -> dict:
        """Every site (or the given ones) at every stage for one parameter set: scoring.objective of the records.
        Raises if every site failed."""
        site_list = site_list or sites.all_sites()
        recs = self.evaluate(p, site_list, None, method=method, tag=tag)
        out = scoring.objective(recs)
        if len(out["failed"]) == len(site_list):
            errors = sorted({recs[s].get("error", "") for s in site_list})[:3]
            raise RuntimeError(f"every site failed: {errors}")
        return out

    def close(self):
        self.pool.shutdown(cancel_futures=True)
