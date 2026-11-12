"""
Calibration: one general parameter set for every site, or each site's own.

Global searches score a parameter set by the study objective: the mean over the calibration sites of each site's mean
MCC over all its stages. The hold-out sites are scored on every evaluation too, but never optimized on.

    optuna      optuna's TPE (multivariate), CMA-ES, QMC or random sampler; a study resumes under the same name.
                Trials go to results/optuna_<mapper>.db.
    sobol       a scrambled Sobol screen of the whole space
    cma         pycma CMA-ES on the continuous parameters, switches held where the start point has them
    coord       compass search: every continuous parameter +-step at once, the best gain taken, the step halved
                when nothing gains
    profile     1-D sweeps of each parameter through its range around one set
    switches    each switch flipped alone around one set
                The last five go to results/explore_<mapper>.jsonl.

Per-site tuning (persite) searches the settings file's persite section, written as deviations from a general set G,
for each site on its own (results/persite_<mapper>_<run>.jsonl).

What is searched, over what, and each default all come from the settings file (settings.yaml).

Every evaluation also goes to the archive.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import config, params, scoring, settings, sites
from .runner import Runner

_lock = threading.Lock()


def _stamp() -> str:
    return time.strftime("%H:%M:%S")


# --------------------------------------------------------------------------------------------------------------------
# optuna
# --------------------------------------------------------------------------------------------------------------------

def optuna_storage(mapper: str) -> str:
    config.RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{config.results_path(f'optuna_{mapper}.db')}"


def _suggest(trial, mapper: str, fixed: dict | None) -> dict:
    p = {}
    for name, spec in params.space(mapper).items():
        if fixed and name in fixed:
            p[name] = fixed[name]
            continue
        if spec.when and not params.switch_on(p, spec.when, mapper):
            continue
        if spec.kind == "bool":
            p[name] = trial.suggest_categorical(name, [False, True])
        elif spec.kind == "choice":
            p[name] = trial.suggest_categorical(name, list(spec.choices))
        elif spec.kind == "int":
            p[name] = trial.suggest_int(name, int(spec.low), int(spec.high), log=spec.log)
        else:
            p[name] = trial.suggest_float(name, float(spec.low), float(spec.high), log=spec.log)
    return p


def run_optuna(mapper: str, name: str, sampler: str = "tpe", trials: int = 1000, jobs: int = 3, workers: int = 30,
               startup: int = 50, seed: int = 0, fixed: dict | None = None, enqueue: list[dict] | None = None,
               log=print) -> dict:
    """A global optuna study (resumed if it exists). Returns the best trial's parameters."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    samplers = {
        "tpe": lambda: optuna.samplers.TPESampler(multivariate=True, group=True, constant_liar=True,
                                                  n_startup_trials=startup, seed=seed),
        "tpe_indep": lambda: optuna.samplers.TPESampler(constant_liar=True, n_startup_trials=startup, seed=seed),
        "cmaes": lambda: optuna.samplers.CmaEsSampler(n_startup_trials=startup, seed=seed, restart_strategy="ipop",
                                                      warn_independent_sampling=False),
        "qmc": lambda: optuna.samplers.QMCSampler(seed=seed, scramble=True),
        "random": lambda: optuna.samplers.RandomSampler(seed=seed),
    }
    study = optuna.create_study(study_name=name, storage=optuna_storage(mapper), load_if_exists=True,
                                direction="maximize", sampler=samplers[sampler]())
    study.set_user_attr("definition", {"mapper": mapper, "sampler": sampler, "fixed": fixed,
                                       "holdout": sites.holdout_sites(), "sites": sites.all_sites(),
                                       "setup": settings.current().fingerprint(mapper),
                                       "settings": str(settings.current().path)})
    if enqueue and not study.trials:
        for q in enqueue:
            study.enqueue_trial(params.searched(q, mapper), skip_if_exists=True)
    t_start = time.time()
    with Runner(mapper, workers) as runner:
        def objective(trial):
            p = _suggest(trial, mapper, fixed)
            t = time.time()
            out = runner.objective(p, "optuna", f"{name}#{trial.number}")
            trial.set_user_attr("holdout", out["holdout"])
            trial.set_user_attr("all", out["all"])
            trial.set_user_attr("site_scores", out["site_scores"])
            trial.set_user_attr("failed", out["failed"])
            trial.set_user_attr("wall_s", round(time.time() - t, 1))
            trial.set_user_attr("params_full", params.complete(p, mapper))
            return out["value"]

        def report(study, trial):
            if trial.value is None:
                return
            best = study.best_trial
            log(f"[{_stamp()}] #{trial.number} {trial.value:.4f} hold {trial.user_attrs['holdout']:.4f} "
                f"({trial.user_attrs['wall_s']:.0f}s) | best #{best.number} {best.value:.4f} hold "
                f"{best.user_attrs['holdout']:.4f} | {(time.time() - t_start) / 3600:.2f} h")

        todo = trials - len([t for t in study.trials if t.state.is_finished()])
        if todo > 0:
            study.optimize(objective, n_trials=todo, n_jobs=jobs, callbacks=[report], catch=(RuntimeError,))
    best = study.best_trial
    log(f"best #{best.number} {best.value:.4f} hold {best.user_attrs['holdout']:.4f}: "
        f"{params.describe(best.user_attrs['params_full'], mapper)}")
    return best.user_attrs["params_full"]


# --------------------------------------------------------------------------------------------------------------------
# Own searches
# --------------------------------------------------------------------------------------------------------------------

def explore_path(mapper: str) -> Path:
    return config.results_path(f"explore_{mapper}.jsonl")


def read_explore(mapper: str) -> list[dict]:
    path = explore_path(mapper)
    if not path.exists():
        return []
    return [json.loads(line) for line in open(path) if line.strip()]


def best_so_far(mapper: str, fixed: dict | None = None) -> dict:
    """The best evaluated set over the own searches and the optuna studies (by calibration value), optionally only
    among those agreeing with fixed; the mapper's default set when there is none."""
    from . import results
    df = results.optimization_frame(mapper)
    if fixed and not df.empty:
        df = df[[all(p.get(k) == v for k, v in fixed.items()) for p in df["params"]]]
    if df.empty:
        return params.complete({}, mapper)
    return df.loc[df["value"].idxmax(), "params"]


class Evaluator:
    """Scores parameter sets on every site, a few sets in flight at once, and logs each to explore_<mapper>.jsonl."""

    def __init__(self, mapper: str, workers: int = 30, jobs: int = 3, log=print):
        self.mapper, self.jobs, self.log = mapper, jobs, log
        self.runner = Runner(mapper, workers)
        self.t0, self.count, self.best = time.time(), 0, -1.0
        explore_path(mapper).parent.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.runner.close()

    def one(self, p: dict, method: str, tag: str, extra: dict | None = None) -> dict:
        p = params.complete(p, self.mapper)
        t = time.time()
        out = self.runner.objective(p, method, tag)
        row = {"method": method, "tag": tag, "time": time.time(), "wall_s": round(time.time() - t, 1),
               "unit": [round(float(x), 6) for x in params.to_unit(p, self.mapper)], "params": p, **out,
               "setup": settings.current().fingerprint(self.mapper), **(extra or {})}
        with _lock:
            with open(explore_path(self.mapper), "a") as f:
                f.write(json.dumps(row, default=float) + "\n")
            self.count += 1
            self.best = max(self.best, out["value"])
            if self.count % 10 == 0 or out["value"] >= self.best:
                self.log(f"[{_stamp()}] {method} {tag}: {out['value']:.4f} hold {out['holdout']:.4f} | n={self.count} "
                         f"best {self.best:.4f} | {(time.time() - self.t0) / 3600:.2f} h")
        return row

    def many(self, ps: list[dict], method: str, tags: list[str], extras: list[dict] | None = None) -> list[dict]:
        extras = extras or [None] * len(ps)
        with ThreadPoolExecutor(self.jobs) as ex:
            return list(ex.map(lambda a: self.one(*a), list(zip(ps, [method] * len(ps), tags, extras))))


def continuous_names(mapper: str, base: dict, fixed: dict) -> list[str]:
    """The parameters CMA-ES and the compass search move: every number (not a switch or a choice) not fixed, whose
    switch, if any, is on in base."""
    names = []
    for name, spec in params.space(mapper).items():
        if spec.kind not in ("float", "int") or name in fixed:
            continue
        if spec.when and not params.switch_on(base, spec.when, mapper):
            continue
        names.append(name)
    if not names:
        raise ValueError("nothing to search")
    return names


def _unit_bounds(mapper: str, names: list[str], bounds: dict) -> tuple[np.ndarray, np.ndarray]:
    """Unit-cube limits per parameter: [0, 1] unless bounds gives {name: [lo, hi]} in the parameter's units."""
    lo, hi = np.zeros(len(names)), np.ones(len(names))
    for j, n in enumerate(names):
        if n in bounds:
            lo[j], hi[j] = (float(params.to_unit({n: v}, mapper, [n])[0]) for v in bounds[n])
    return lo, hi


def _start(mapper: str, x0: dict | None, fixed: dict) -> dict:
    base = x0 if x0 is not None else best_so_far(mapper, fixed or None)
    return {**params.complete(base, mapper), **fixed}


def sobol(ev: Evaluator, n: int = 1024, seed: int = 0) -> None:
    from scipy.stats import qmc
    names = params.space_names(ev.mapper)
    done = {r["tag"] for r in read_explore(ev.mapper) if r["method"] == "sobol"}
    pts = qmc.Sobol(len(names), scramble=True, seed=seed).random(n)
    todo = [(i, u) for i, u in enumerate(pts) if f"sobol{seed}_{i}" not in done]
    ev.many([params.from_unit(u, ev.mapper, names) for _, u in todo], "sobol", [f"sobol{seed}_{i}" for i, _ in todo])


def cma(ev: Evaluator, tag: str, budget: int = 600, popsize: int = 12, sigma: float = 0.2, seed: int = 0,
        x0: dict | None = None, fixed: dict | None = None, bounds: dict | None = None) -> None:
    import cma as pycma
    fixed = fixed or {}
    base = _start(ev.mapper, x0, fixed)
    names = continuous_names(ev.mapper, base, fixed)
    lo, hi = _unit_bounds(ev.mapper, names, bounds or {})
    x = np.clip(params.to_unit(base, ev.mapper, names), lo, hi)
    ev.log(f"CMA-ES over {names} from {np.round(x, 3)}")
    es = pycma.CMAEvolutionStrategy(x, sigma, {"bounds": [list(lo), list(hi)], "popsize": popsize, "seed": seed + 1,
                                               "maxfevals": budget, "verbose": -9})
    gen = 0
    while not es.stop():
        xs = es.ask()
        ps = [{**base, **params.from_unit(np.array(u), ev.mapper, names)} for u in xs]
        rows = ev.many(ps, "cma", [f"{tag}_g{gen}_{i}" for i in range(len(ps))], [{"cma_names": names}] * len(ps))
        es.tell(xs, [-r["value"] for r in rows])
        gen += 1
        ev.log(f"  CMA generation {gen}: best {-es.result.fbest:.4f}, sigma {es.sigma:.3f}")


def coord(ev: Evaluator, tag: str, step: float = 0.1, min_step: float = 0.012, max_iter: int = 40,
          tol: float = 1e-4, x0: dict | None = None, fixed: dict | None = None, bounds: dict | None = None) -> dict:
    fixed = fixed or {}
    base = _start(ev.mapper, x0, fixed)
    names = continuous_names(ev.mapper, base, fixed)
    lo, hi = _unit_bounds(ev.mapper, names, bounds or {})

    def at(u):
        return {**base, **params.from_unit(u, ev.mapper, names)}

    x = np.clip(params.to_unit(base, ev.mapper, names), lo, hi)
    cur = ev.one(at(x), "coord", f"{tag}_start")["value"]
    it = 0
    while step >= min_step and it < max_iter:
        cands, tags = [], []
        for j, name in enumerate(names):
            for sgn in (-1, 1):
                y = x.copy()
                y[j] = np.clip(y[j] + sgn * step, lo[j], hi[j])
                if abs(y[j] - x[j]) < 1e-9 or at(y) == at(x):     # at a bound, or an integer that didn't move
                    continue
                cands.append(y)
                tags.append(f"{tag}_i{it}_{name}{'+' if sgn > 0 else '-'}")
        rows = ev.many([at(y) for y in cands], "coord", tags, [{"step": step}] * len(cands))
        k = int(np.argmax([r["value"] for r in rows])) if rows else -1
        if rows and rows[k]["value"] > cur + tol:
            x, cur = cands[k], rows[k]["value"]
            ev.log(f"  compass {it}: step {step:.3f}, took {tags[k]} -> {cur:.4f}")
        else:
            step /= 2
            ev.log(f"  compass {it}: no gain, step -> {step:.3f}")
        it += 1
    ev.log(f"compass search ends at {cur:.4f}: {params.describe(at(x), ev.mapper)}")
    return at(x)


def profile(ev: Evaluator, tag: str, points: int = 13, x0: dict | None = None, only: list[str] | None = None) -> None:
    base = _start(ev.mapper, x0, {})
    ps, tags, extras = [], [], []
    for name, spec in params.space(ev.mapper).items():
        if only and name not in only:
            continue
        cond = spec.when
        if spec.kind == "bool":
            vals = [False, True]
        elif spec.kind == "choice":
            vals = list(spec.choices)
        else:
            vals = sorted({params.from_unit([u], ev.mapper, [name])[name] for u in np.linspace(0, 1, points)})
        for v in vals:
            q = {**base, name: v}
            if cond:
                q[cond] = True
            ps.append(q)
            tags.append(f"{tag}_{name}={v}")
            extras.append({"profile_param": name, "profile_value": v})
    ev.many(ps, "profile", tags, extras)


def switches(ev: Evaluator, tag: str, bases: list[dict] | None = None) -> None:
    bases = bases or [best_so_far(ev.mapper)]
    ps, tags = [], []
    for b, base in enumerate(bases):
        base = params.complete(base, ev.mapper)
        ps.append(base)
        tags.append(f"{tag}_b{b}_base")
        for name, spec in params.space(ev.mapper).items():
            if spec.kind == "bool":
                ps.append({**base, name: not base[name]})
                tags.append(f"{tag}_b{b}_{name}={not base[name]}")
    ev.many(ps, "switches", tags)


# --------------------------------------------------------------------------------------------------------------------
# Per-site tuning
# --------------------------------------------------------------------------------------------------------------------

def reduced_space(mapper: str) -> dict[str, dict]:
    """The settings file's persite section for a mapper: {name: {"scales": [...] or None, "spec": Spec}}."""
    return settings.current().persite(mapper)


def apply_reduced(base: dict, r: dict, reduced: dict) -> dict:
    """G with the reduced parameters r applied: a multiplier scales its parameters, any other sets its own."""
    p = dict(base)
    for name, v in r.items():
        scales = reduced[name]["scales"]
        if scales:
            for k in scales:
                p[k] = base[k] * v
        else:
            p[name] = v
    return p


def _reduced_from_unit(u, names, reduced):
    r = {}
    for x, n in zip(u, names):
        sp = reduced[n]["spec"]
        x = float(np.clip(x, 0, 1))
        lo, hi = sp.low, sp.high
        v = np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo))) if sp.log else lo + x * (hi - lo)
        r[n] = int(round(v)) if sp.kind == "int" else float(v)
    return r


def _reduced_to_unit(r, names, reduced):
    out = []
    for n in names:
        sp = reduced[n]["spec"]
        lo, hi, v = sp.low, sp.high, r[n]
        out.append((np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)) if sp.log else (v - lo) / (hi - lo))
    return np.clip(np.array(out), 0, 1)


def reduced_identity(base: dict, names: list[str], reduced: dict) -> dict:
    """The reduced parameters that give G itself."""
    return {n: (1.0 if reduced[n]["scales"] else base[n]) for n in names}


def persite_path(mapper: str, run: str) -> Path:
    return config.results_path(f"persite_{mapper}_{run}.jsonl")


class SiteSearch:
    def __init__(self, runner: Runner, site: str, base: dict, names: list[str], run: str, reduced: dict):
        self.runner, self.site, self.base, self.names, self.run = runner, site, base, names, run
        self.reduced = reduced
        self.out = persite_path(runner.mapper, run)
        self.cache = {}

    def score(self, r: dict, search: str) -> float:
        key = (search, json.dumps(r, sort_keys=True))
        if key in self.cache:
            return self.cache[key]
        p = params.complete(apply_reduced(self.base, r, self.reduced), self.runner.mapper)
        rec = self.runner.submit(self.site, p, None).result()
        self.runner.archive.add(f"persite_{search}", f"{self.run}|{self.site}", p, rec, None)
        v = scoring.site_score(rec)
        bias = [b for b in (rec.get("bias") or []) if b is not None]
        row = {"site": self.site, "search": search, "r": r, "value": v, "ok": rec["ok"], "time": time.time(),
               "bias_gmean": float(np.exp(np.mean(np.log(np.clip(bias, 1e-3, None))))) if bias else None}
        with _lock, open(self.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        self.cache[key] = v
        return v

    def tpe(self, budget: int, startup: int, seed: int) -> float:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(
            multivariate=True, n_startup_trials=startup, seed=seed))
        st.enqueue_trial(reduced_identity(self.base, self.names, self.reduced))
        for _ in range(budget):
            t = st.ask()
            r = {}
            for n in self.names:
                sp = self.reduced[n]["spec"]
                r[n] = (t.suggest_int(n, int(sp.low), int(sp.high), log=sp.log) if sp.kind == "int"
                        else t.suggest_float(n, float(sp.low), float(sp.high), log=sp.log))
            st.tell(t, self.score(r, "tpe"))
        return st.best_value

    def own(self, n_sobol: int, seed: int, step: float = 0.125, min_step: float = 0.015) -> float:
        """A Sobol design, then a compass search from its best."""
        from scipy.stats import qmc
        pts = qmc.Sobol(len(self.names), scramble=True, seed=seed).random(n_sobol)
        cands = [reduced_identity(self.base, self.names, self.reduced)] + \
                [_reduced_from_unit(u, self.names, self.reduced) for u in pts]
        vals = [self.score(r, "own") for r in cands]
        k = int(np.argmax(vals))
        x, cur = _reduced_to_unit(cands[k], self.names, self.reduced), vals[k]
        while step >= min_step:
            best = (cur, None)
            for j in range(len(self.names)):
                for s in (-1, 1):
                    y = x.copy()
                    y[j] = np.clip(y[j] + s * step, 0, 1)
                    if abs(y[j] - x[j]) < 1e-9:
                        continue
                    v = self.score(_reduced_from_unit(y, self.names, self.reduced), "own")
                    if v > best[0] + 1e-5:
                        best = (v, y)
            if best[1] is None:
                step /= 2
            else:
                cur, x = best
        return cur


def run_persite(mapper: str, base: dict, run: str, site_list: list[str], names: list[str] | None = None,
                searches: tuple[str, ...] = ("tpe", "own"), budget: int = 150, startup: int = 24, n_sobol: int = 64,
                workers: int = 30, seed: int = 0, log=print) -> None:
    base = params.complete(base, mapper)
    reduced = reduced_space(mapper)
    names = names or list(reduced)
    bad = [n for n in names if n not in reduced]
    if bad:
        raise SystemExit(f"not searched per site for {mapper} in the settings file's persite section: {bad}")
    persite_path(mapper, run).parent.mkdir(parents=True, exist_ok=True)
    meta = config.results_path(f"persite_{mapper}_{run}.base.json")
    meta.write_text(json.dumps({"base": base, "names": names}, indent=1, default=float))
    t0 = time.time()
    with Runner(mapper, workers) as runner:
        def job(item):
            site, search = item
            ss = SiteSearch(runner, site, base, names, run, reduced)
            v = ss.tpe(budget, startup, seed) if search == "tpe" else ss.own(n_sobol, seed)
            log(f"[{_stamp()}] {search:<4} {site:<66} best {v:.4f} ({len(ss.cache)} evaluations) | "
                f"{(time.time() - t0) / 3600:.2f} h")

        jobs = [(s, m) for s in site_list for m in searches]
        # each search has one evaluation in flight at a time, so keep enough searches going to fill the pool
        with ThreadPoolExecutor(min(len(jobs), 3 * workers)) as ex:
            list(ex.map(job, jobs))
