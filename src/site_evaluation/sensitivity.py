"""
Which parameters to calibrate (``site-eval sensitivity``): a global screening of every candidate parameter, then a
test of whether the influential ones can be calibrated together.

Candidates (Settings.candidates): every argument in the settings file with a range or choices, and every switch
(true/false with an optimize field), that the mapper uses, except nencarta's (they shape the cached preprocessing) and
those marked ``screen: false`` (fixed beforehand, such as Standard_Step on physical grounds). Each is screened over its
range (a log range on the log scale).

1. Screening: the elementary effects method (Morris 1991) with μ* (Campolongo et al. 2007).
   r trajectories cross a grid of 4 levels per parameter. Each step of a trajectory moves one parameter by
   Δ = 2/3 of its range (a switch is flipped, a choice changes). A step's effect is the change in the objective, the
   mean MCC of the study's sites (by default the calibration sites), signed for a rise of the parameter. Per
   parameter: μ* = the mean |effect| (its influence), μ = the mean effect (its direction), σ = the effects' standard
   deviation (non-linearity and interactions), and a bootstrap 95% interval of μ* over the trajectories.
   The mean can hide a parameter that moves many sites, some up and some down. So each step's change at every site
   is kept too: "site μ*" is a site's mean |change| (averaged over the sites), with its own bootstrap interval, and
   "moved" is the number of sites a step changes by the threshold or more (up/down). Against a threshold (0.01 MCC by
   default: a smaller change is no change):
       influential         μ*'s interval lies above the threshold
       undecided           μ*'s interval straddles it: more trajectories (rounds, below)
       trades sites        μ*'s interval lies below it, site μ*'s above: the mean doesn't move, but the sites do,
                           both ways. Its value picks which sites do well, so it is set on physical grounds or tied
                           to what the sites it helps share, not fixed as if it changed nothing
       undecided (sites)   μ*'s interval lies below it, site μ*'s straddles it: more trajectories
       negligible          both intervals lie below it: fix the parameter at its default
   Fewer than MIN_EFFECTS effects leave a parameter undecided. A step taken while the setting a parameter depends on
   (its `when`) doesn't hold has no effect and is left out. The site effects (each site's μ and μ*) are saved to
   <study>_site_effects.csv.

   Rounds (more_trajectories). After the r full trajectories, the undecided parameters (either kind, or never
   active) get r more trajectories that move only them, and are classified again, until none is left or each has
   max_trajectories (60 by default). A round's trajectory starts at a random grid point of every parameter, like a
   full one, and the parameters it doesn't move stay there, so its effects are drawn as a full trajectory's are and
   pool with them; it costs len(undecided)+1 runs instead of k+1. Looking at the intervals after each round (up to
   three times by default) makes a wrong call somewhat likelier than 5% for a parameter whose μ* sits at the
   threshold, where either call changes little.

2. Identifiability: the collinearity index of Brun et al. (2001), at a reference set (the settings file's defaults).
   Each influential or undecided number is moved ±step of its range. The change in every site's MCC at every stage,
   per unit of the range, is its sensitivity vector s_j; δ_j = RMS(s_j) is its importance. Parameters whose vectors
   are parallel, or whose vector is a combination of others', compensate each other: every n higher with every slope
   steeper gives the same maps. For a set K of unit-length vectors S̃_K, γ_K = 1 / sqrt(λ_min(S̃_Kᵀ S̃_K)). γ = 1 means
   independent effects; above γ_crit the data can't fix all of the set's values. Brun's critical range, 10-15, is
   for smooth outputs; MCC changes in steps as cells flip, which hides collinearity, so the default is 5 (below).
   The report gives
       - γ of the whole set and every combination with γ above γ_crit (an eigenvector of S̃ᵀS̃): moving those
         parameters together, in those proportions and signs, leaves every site's MCC nearly unchanged;
       - the identifiable subset: parameters added from the most important down (--prefer puts named ones first),
         each kept only if the subset's γ stays below γ_crit. A parameter left out is explained by those kept (R² and
         the main ones), so it is better fixed, or tied to them.
   A parameter whose move changes no site-stage MCC by 0.01 is too weak there to judge and is left out of γ. This
   test is local: it holds near the reference set.

   Positive control. slope_adjustment_factor multiplies every conveyance, so it gives exactly the maps of dividing
   every n by it: a structural duplicate. Every collinearity test also moves it
   and each n class, and reports how collinear the method finds it with the n values on these sites. On six test
   sites that was γ ≈ 6 (R² 0.97), not infinity: a set whose γ is near the control's is as confounded as an exact
   duplicate on this output, hence the default γ_crit of 5.

Switches and choices are screened but not tested for collinearity: an influential one is settled by comparing its
settings (site-eval optimize switches).

References
    Morris, M. D. (1991). Factorial sampling plans for preliminary computational experiments. Technometrics 33(2),
        161-174.
    Campolongo, F., Cariboni, J., Saltelli, A. (2007). An effective screening design for sensitivity analysis of
        large models. Environmental Modelling & Software 22(10), 1509-1518.
    Brun, R., Reichert, P., Künsch, H. R. (2001). Practical identifiability analysis of large environmental
        simulation models. Water Resources Research 37(4), 1015-1030.
    Herman, J. D., Kollat, J. B., Reed, P. M., Wagener, T. (2013). Technical note: Method of Morris effectively
        reduces the computational demands of global sensitivity analysis for distributed watershed models. Hydrology
        and Earth System Sciences 17, 2893-2903.
    Pianosi, F. et al. (2016). Sensitivity analysis of environmental models: a systematic review with practical
        workflow. Environmental Modelling & Software 79, 214-232.

Every run is kept in results/sensitivity/<mapper>_<name>.jsonl, with each site's MCC at every stage (not in the
archive, which would grow by a row per site per run). Running the same study again resumes it, adds trajectories,
or only prints the report; the tables also go to CSV files beside it.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, params, progress, scoring, settings

LEVELS = 4
DELTA = LEVELS / (2 * (LEVELS - 1))          # 2/3: a Morris step moves a parameter by this share of its range
MIN_EFFECTS = 5                              # fewer effects than this leave a parameter undecided
_HEADER_FIELDS = ("mapper", "dem", "names", "specs", "sites", "seed", "setup")


def study_path(mapper: str, name: str, suffix: str = ".jsonl") -> Path:
    return config.results_path("sensitivity", f"{mapper}_{name}{suffix}")


def _hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:8]


def _norm(obj):
    return json.loads(json.dumps(obj, default=str))


def _stamp() -> str:
    return time.strftime("%H:%M:%S")


def setup_fingerprint(mapper: str, specs: dict) -> str:
    """What every run of a study shares: the value of each argument that isn't screened, the DEM and its cell size."""
    S = settings.current()
    fixed = {k: S.default(k, mapper) for k, e in S.entries.items()
             if not e.is_from and k not in specs and S.applies(e, mapper)}
    return _hash({"fixed": fixed, "dem": S.dem, "cell_size": S.cell_size, "nominal": S.nominal_cell_size,
                  "flow_stats": config.FLOW_STATS_FILE})


def specs_of(header: dict) -> dict[str, settings.Spec]:
    return {k: settings.Spec(**v) for k, v in header["specs"].items()}


# --------------------------------------------------------------------------------------------------------------------
# Designs
# --------------------------------------------------------------------------------------------------------------------

def morris_design(names: list[str], r: int, seed: int, rnd: dict | None = None) -> list[dict]:
    """r Morris trajectories of len(names)+1 points on the unit cube. Trajectory t depends only on (seed, t), so a
    study can be extended with more trajectories.

    With rnd (a round, from Study.rounds), its trajectories instead: each starts at a random grid point of every
    parameter, as above, but steps only the round's parameters, so it takes len(rnd["names"])+1 points. The others
    stay at their random levels, so each effect is taken at a random point of the whole space, as in a full
    trajectory, and the two pool."""
    k = len(names)
    out = []
    if rnd is not None:
        n_round, moved = rnd["round"], np.array([names.index(n) for n in rnd["names"]])
        for t in range(rnd["trajectories"]):
            rng = np.random.default_rng([seed, 1000 + n_round, t])
            x = rng.integers(0, LEVELS, k) / (LEVELS - 1)
            pre = f"x{n_round}.{t}"
            out.append({"kind": "morris", "key": f"{pre}.0", "round": n_round, "traj": t, "step": 0,
                        "unit": x.round(6).tolist(), "moved": None, "sign": 0})
            for i, j in enumerate(rng.permutation(moved), 1):
                x = x.copy()
                sign = 1 if x[j] + DELTA <= 1 + 1e-9 else -1
                x[j] = x[j] + sign * DELTA
                out.append({"kind": "morris", "key": f"{pre}.{i}", "round": n_round, "traj": t, "step": i,
                            "unit": x.round(6).tolist(), "moved": names[int(j)], "sign": sign})
        return out
    for t in range(r):
        rng = np.random.default_rng([seed, t])
        x = rng.integers(0, LEVELS, k) / (LEVELS - 1)
        out.append({"kind": "morris", "key": f"m{t}.0", "traj": t, "step": 0, "unit": x.round(6).tolist(),
                    "moved": None, "sign": 0})
        for i, j in enumerate(rng.permutation(k), 1):
            x = x.copy()
            sign = 1 if x[j] + DELTA <= 1 + 1e-9 else -1
            x[j] = x[j] + sign * DELTA
            out.append({"kind": "morris", "key": f"m{t}.{i}", "traj": t, "step": i, "unit": x.round(6).tolist(),
                        "moved": names[int(j)], "sign": sign})
    return out


def local_design(ref: dict, names: list[str], specs: dict, mapper: str, step: float) -> tuple[list[dict], dict]:
    """The reference set, and each number moved up and down by step of its range (one side only at a bound; at
    least one whole number for an integer, one step for a number with a step). Also the names left out, with the
    reason."""
    rh = _hash({"ref": ref, "step": step})
    pts = [{"kind": "local", "key": f"L{rh}.base", "ref": rh, "moved": None, "params": ref}]
    skipped = {}
    for n in names:
        s = specs[n]
        if s.kind not in ("float", "int"):
            skipped[n] = "a switch or choice"
            continue
        if s.when and not params.active(ref, s.when, mapper):
            skipped[n] = f"unused at the reference (needs {settings.describe_when(s.when)})"
            continue
        u0 = float(params.to_unit(ref, mapper, [n], specs)[0])
        for sign in (1, -1):
            u = float(np.clip(u0 + sign * step, 0.0, 1.0))
            if abs(u - u0) < 1e-9:
                continue
            v = params.from_unit([u], mapper, [n], specs)[n]
            grain = s.step or (1 if s.kind == "int" else None)
            if grain:
                if v == ref[n]:
                    v = type(v)(np.clip(ref[n] + sign * grain, s.low, s.high))
                if v == ref[n]:
                    continue
            pts.append({"kind": "local", "key": f"L{rh}.{n}{'+' if sign > 0 else '-'}", "ref": rh, "moved": n,
                        "sign": sign, "unit": float(params.to_unit({n: v}, mapper, [n], specs)[0]), "u0": u0,
                        "params": {**ref, n: v}})
    return pts, skipped


# --------------------------------------------------------------------------------------------------------------------
# A study on disk
# --------------------------------------------------------------------------------------------------------------------

class Study:
    """A study's runs: results/sensitivity/<mapper>_<name>.jsonl, a header line, then one line per parameter set."""

    def __init__(self, mapper: str, name: str, header: dict | None = None):
        self.mapper, self.name = mapper, name
        self.path = study_path(mapper, name)
        self.rows: dict[str, dict] = {}
        self.header = None
        if self.path.exists():
            lines = progress.read_jsonl(self.path, repair=header is not None)   # repair only to append
            self.header = lines[0]
            self.rows = {r["key"]: r for r in lines[1:]}
        if header is not None:
            if self.header is None:
                self.header = _norm(header)
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "w") as f:
                    f.write(json.dumps(self.header) + "\n")
            else:
                diff = [k for k in _HEADER_FIELDS if _norm(header.get(k)) != self.header.get(k)]
                if diff:
                    raise SystemExit(f"study {name!r} ({self.path}) was run with other {', '.join(diff)}: rerun "
                                     f"it as it was, or use another --name")
        if self.header is None:
            raise SystemExit(f"no study {self.path}")
        self.names: list[str] = self.header["names"]
        self.specs = specs_of(self.header)
        self.sites: list[str] = self.header["sites"]

    def run(self, points: list[dict], workers: int, jobs: int, log=print, desc: str = "runs") -> None:
        """Run the points not run yet, jobs parameter sets at a time. Each finished one is saved at once, so a stopped
        study continues where it stopped. The first Ctrl-C lets the running sets finish; a second stops at once."""
        todo = [pt for pt in points if pt["key"] not in self.rows]
        if not todo:
            return
        from .runner import Runner
        lock = threading.Lock()
        log(f"[{_stamp()}] {len(todo)} runs of {len(self.sites)} sites ({len(points) - len(todo)} done before)")
        with Runner(self.mapper, workers) as runner, progress.Progress(len(todo), desc, "run", log) as bar:
            def one(pt):
                recs = runner.evaluate(pt["params"], self.sites, None, archive=False)
                if not any(r.get("ok") for r in recs.values()):
                    errors = sorted({r.get("error", "") for r in recs.values()})[:3]
                    raise RuntimeError(f"every site failed at {pt['key']}: {errors}")
                scores = {s: scoring.site_score(r) for s, r in recs.items()}
                row = {**pt, "value": float(np.mean(list(scores.values()))),
                       "sites": {s: round(v, 6) for s, v in scores.items()},
                       "mcc": {s: r.get("mcc") or [] for s, r in recs.items()},
                       "stages": {s: r.get("stages") or [] for s, r in recs.items()},
                       "failed": [s for s, r in recs.items() if not r.get("ok")], "time": time.time()}
                with lock:
                    with open(self.path, "a") as f:
                        f.write(json.dumps(row, default=float) + "\n")
                    self.rows[pt["key"]] = row
                bar.update()

            try:
                progress.thread_map(one, todo, jobs, bar.write)
            except KeyboardInterrupt:
                done = sum(pt["key"] in self.rows for pt in points)
                bar.write(f"Stopped: {done} of {len(points)} runs are saved in {self.path}. The same command "
                          "continues from there.")
                raise

    def failures(self) -> int:
        return sum(len(r.get("failed", [])) for r in self.rows.values())

    def rounds(self) -> list[dict]:
        """The rounds of more trajectories planned so far, in order: round, names, trajectories."""
        return sorted((r for r in self.rows.values() if r["kind"] == "round"), key=lambda r: r["round"])

    def add_round(self, names: list[str], trajectories: int) -> dict:
        """Plan a round (saved before it runs, so a stopped round continues with the same parameters)."""
        n = len(self.rounds()) + 1
        row = {"kind": "round", "key": f"round{n}", "round": n, "names": names, "trajectories": trajectories,
               "time": time.time()}
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        self.rows[row["key"]] = row
        return row


# --------------------------------------------------------------------------------------------------------------------
# 1. Screening
# --------------------------------------------------------------------------------------------------------------------

def _trajectories(study: Study) -> list[tuple[str, list[str]]]:
    """Each trajectory's label and keys: the full ones (m<t>.<i>), then each round's (x<round>.<t>.<i>)."""
    k = len(study.names)
    full = sorted({r["traj"] for r in study.rows.values() if r["kind"] == "morris" and not r.get("round")})
    out = [(f"m{t}", [f"m{t}.{i}" for i in range(k + 1)]) for t in full]
    for rd in study.rounds():
        out += [(f"x{rd['round']}.{t}", [f"x{rd['round']}.{t}.{i}" for i in range(len(rd["names"]) + 1)])
                for t in range(rd["trajectories"])]
    return out


def morris_effects(study: Study) -> pd.DataFrame:
    """One row per elementary effect of the complete trajectories (full ones and rounds'): traj, name, effect (the
    objective's change for a rise of Δ), and each site's change in score."""
    out = []
    for label, keys in _trajectories(study):
        steps = [study.rows.get(key) for key in keys]
        if any(s is None for s in steps):
            continue
        for prev, cur in zip(steps, steps[1:]):
            n = cur["moved"]
            spec = study.specs[n]
            if spec.when and not params.active(prev["params"], spec.when, study.mapper):
                continue
            sign = cur["sign"]
            out.append({"traj": label, "name": n, "effect": sign * (cur["value"] - prev["value"]),
                        **{s: sign * (cur["sites"][s] - prev["sites"][s]) for s in study.sites}})
    return pd.DataFrame(out)


def complete_trajectories(study: Study) -> int:
    """The number of complete full trajectories (those that move every parameter)."""
    k = len(study.names)
    trajs = {r["traj"] for r in study.rows.values() if r["kind"] == "morris" and not r.get("round")}
    return sum(all(f"m{t}.{i}" in study.rows for i in range(k + 1)) for t in trajs)


def trajectory_counts(study: Study) -> dict[str, int]:
    """Per parameter: the complete trajectories that move it, full ones and rounds' together."""
    counts = dict.fromkeys(study.names, complete_trajectories(study))
    for rd in study.rounds():
        done = sum(all(f"x{rd['round']}.{t}.{i}" in study.rows for i in range(len(rd["names"]) + 1))
                   for t in range(rd["trajectories"]))
        for n in rd["names"]:
            counts[n] += done
    return counts


NEEDS_MORE = ("undecided", "undecided (sites)", "never active")


def more_trajectories(study: Study, specs: dict, mapper: str, r: int, max_trajectories: int, threshold: float,
                      workers: int, jobs: int, log=print) -> None:
    """Rounds of more trajectories for the parameters that need them: run the rounds planned before (a stopped one
    continues), then, while some parameter is undecided (on the mean or the sites) or never active, and moved by
    fewer than max_trajectories trajectories, plan another round of r trajectories (fewer near the cap) that moves
    only those parameters, run it, and classify again. A parameter's class only changes when it is moved again, so a
    decided one stays decided."""
    while True:
        for rd in study.rounds():
            pts = morris_design(study.names, 0, study.header["seed"], rd)
            for pt in pts:
                pt["params"] = params.from_unit(pt["unit"], mapper, study.names, specs)
            study.run(pts, workers, jobs, log, f"round {rd['round']}")
        counts = trajectory_counts(study)
        df = screening_table(study, threshold)
        need = set(df.loc[df["class"].isin(NEEDS_MORE), "parameter"])
        need = [n for n in study.names if n in need and counts[n] < max_trajectories]
        if not need:
            return
        size = min(r, max_trajectories - max(counts[n] for n in need))
        rd = study.add_round(need, size)
        log(f"[{_stamp()}] round {rd['round']}: {size} more trajectories ({size * (len(need) + 1)} runs) for "
            f"{len(need)} undecided: {', '.join(need)}")


def classify(n: int, mean_ci: tuple[float, float], site_ci: tuple[float, float], threshold: float) -> str:
    """A candidate's class from the 95% intervals of μ* (the sites' mean MCC) and of site μ* (a site's MCC). The mean
    decides first; only a parameter whose effect on the mean is negligible is judged on the sites, so that one which
    moves sites both ways, cancelling in the mean, isn't fixed as if it changed nothing."""
    if n < MIN_EFFECTS:
        return "undecided"
    if mean_ci[0] >= threshold:
        return "influential"
    if mean_ci[1] >= threshold:
        return "undecided"
    if site_ci[0] >= threshold:
        return "trades sites"
    return "negligible" if site_ci[1] < threshold else "undecided (sites)"


def screening_table(study: Study, threshold: float = 0.01, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per candidate: μ*, its bootstrap 95% interval, μ, σ; site μ* (a site's mean |change|) and its interval; the
    number of sites a step moves by the threshold or more (up, down); the site it moves most; the number of effects;
    and its class (classify). Sorted by μ*."""
    eff = morris_effects(study)
    rng = np.random.default_rng(seed)
    rows = []
    for n in study.names:
        spec = study.specs[n]
        e = eff[eff["name"] == n] if len(eff) else eff
        base = {"parameter": n, "kind": spec.kind + (f" (when {settings.describe_when(spec.when)})" if spec.when else ""), "n": len(e)}
        if len(e) == 0:
            rows.append({**base, "class": "never active"})
            continue
        d = e["effect"].to_numpy(float)
        a = np.abs(d)
        D = e[study.sites].to_numpy(float)           # each step's signed change at each site
        m = np.nanmean(np.abs(D), axis=1)            # each step's mean |site change|
        idx = rng.integers(0, len(a), (n_boot, len(a)))
        lo, hi = np.percentile(a[idx].mean(axis=1), [2.5, 97.5])
        slo, shi = np.percentile(m[idx].mean(axis=1), [2.5, 97.5])
        per_site = e[study.sites].abs().mean(axis=0)
        rows.append({**base, "mu_star": a.mean(), "ci_low": lo, "ci_high": hi, "mu": d.mean(),
                     "sigma": d.std(ddof=1) if len(d) > 1 else np.nan, "site_mu_star": float(per_site.mean()),
                     "site_ci_low": slo, "site_ci_high": shi,
                     "sites_moved": float((np.abs(D) >= threshold).sum(axis=1).mean()),
                     "sites_up": float((D >= threshold).sum(axis=1).mean()),
                     "sites_down": float((D <= -threshold).sum(axis=1).mean()),
                     "top_site": per_site.idxmax(), "top_site_mu_star": float(per_site.max()),
                     "class": classify(len(a), (lo, hi), (slo, shi), threshold)})
    df = pd.DataFrame(rows)
    return df.sort_values("mu_star", ascending=False, na_position="last").reset_index(drop=True)


def site_effects(study: Study) -> pd.DataFrame:
    """Per candidate and site: the number of effects, the mean signed change in the site's MCC for a rise of the
    parameter (μ) and the mean |change| (μ*). Which sites a parameter helps and hurts."""
    eff = morris_effects(study)
    if not len(eff):
        return pd.DataFrame(columns=["parameter", "site", "n", "mu", "mu_star"])
    g = eff.groupby("name", sort=False)[study.sites]
    mu = g.mean().stack().rename("mu")
    mu_star = eff[study.sites].abs().groupby(eff["name"], sort=False).mean().stack().rename("mu_star")
    n = g.count().stack().rename("n")
    out = pd.concat([n, mu, mu_star], axis=1).reset_index()
    out.columns = ["parameter", "site", "n", "mu", "mu_star"]
    return out


# --------------------------------------------------------------------------------------------------------------------
# 2. Identifiability
# --------------------------------------------------------------------------------------------------------------------

def _outputs(study: Study, base: dict) -> list[tuple[str, str]]:
    return [(s, st) for s in study.sites for st in base["stages"].get(s, [])]


def _vector(row: dict, keys: list[tuple[str, str]]) -> np.ndarray:
    look = {}
    for s, stages in row["stages"].items():
        for st, v in zip(stages, row["mcc"][s]):
            look[(s, st)] = 0.0 if v is None else float(v)
    return np.array([look.get(k, 0.0) for k in keys])


def _gamma(G: np.ndarray, idx: list[int]) -> float:
    lam = np.linalg.eigvalsh(G[np.ix_(idx, idx)])[0]
    return float(1.0 / np.sqrt(max(lam, 1e-12)))


def identifiability(study: Study, ref_hash: str, names: list[str], gamma_crit: float = 5.0,
                    prefer: list[str] | None = None) -> dict:
    """Brun's analysis from a study's local runs at one reference set."""
    base = study.rows.get(f"L{ref_hash}.base")
    if base is None:
        raise ValueError("the reference set hasn't been run")
    keys = _outputs(study, base)
    y0, v0 = _vector(base, keys), base["value"]
    cols, used, table = [], [], []
    for n in names:
        up, dn = study.rows.get(f"L{ref_hash}.{n}+"), study.rows.get(f"L{ref_hash}.{n}-")
        if up is None and dn is None:
            continue
        u0 = (up or dn)["u0"]
        yu, vu, uu = (_vector(up, keys), up["value"], up["unit"]) if up else (y0, v0, u0)
        yd, vd, ud = (_vector(dn, keys), dn["value"], dn["unit"]) if dn else (y0, v0, u0)
        s = (yu - yd) / (uu - ud)
        cols.append(s)
        used.append(n)
        table.append({"parameter": n, "delta": float(np.sqrt(np.mean(s ** 2))), "d_objective": (vu - vd) / (uu - ud),
                      "outputs_moved": int(np.sum(np.abs(yu - yd) >= 0.01))})
    out = {"outputs": len(keys), "table": pd.DataFrame(table), "combos": [], "selected": [], "left_out": {},
           "no_effect": [], "gamma_all": np.nan, "cos": None}
    if not used:
        return out
    S = np.column_stack(cols)
    norms = np.linalg.norm(S, axis=0)
    moved = [r["outputs_moved"] for r in table]
    live = [i for i in range(len(used)) if norms[i] > 1e-12 and moved[i] > 0]
    out["no_effect"] = [used[i] for i in range(len(used)) if i not in live]
    lnames = [used[i] for i in live]
    if not lnames:
        return out
    St = S[:, live] / norms[live]
    G = St.T @ St
    out["cos"] = pd.DataFrame(G, index=lnames, columns=lnames)
    out["gamma_all"] = _gamma(G, list(range(len(lnames))))
    lam, V = np.linalg.eigh(G)
    for i, l in enumerate(lam):
        g = 1.0 / np.sqrt(max(l, 1e-12))
        if g < gamma_crit:
            continue
        v = V[:, i] * (np.sign(V[np.argmax(np.abs(V[:, i])), i]) or 1.0)
        out["combos"].append((float(g), [(lnames[j], float(v[j])) for j in np.argsort(-np.abs(v)) if abs(v[j]) >= 0.2]))
    delta = dict(zip(out["table"]["parameter"], out["table"]["delta"]))
    first = [p for p in (prefer or []) if p in lnames]
    order = first + sorted([p for p in lnames if p not in first], key=lambda p: -delta[p])
    sel: list[int] = []
    for n in order:
        j = lnames.index(n)
        g = _gamma(G, sel + [j])
        if g < gamma_crit:
            sel.append(j)
            continue
        A = St[:, sel]
        beta, *_ = np.linalg.lstsq(A, St[:, j], rcond=None)
        r2 = float(1.0 - np.sum((St[:, j] - A @ beta) ** 2))
        by = sorted(zip([lnames[i] for i in sel], beta), key=lambda t: -abs(t[1]))[:3]
        out["left_out"][n] = {"gamma": g, "r2": r2, "by": [(b, float(c)) for b, c in by]}
    out["selected"] = [lnames[i] for i in sel]
    out["gamma_selected"] = _gamma(G, sel) if sel else np.nan
    t = out["table"]
    vif = np.diag(np.linalg.pinv(G))           # 1 / (1 - R²) of each vector regressed on all the others
    t["r2_with_rest"] = [float(1.0 - 1.0 / max(vif[lnames.index(p)], 1.0)) if p in lnames else np.nan
                         for p in t["parameter"]]
    t["decision"] = ["too weak here" if p in out["no_effect"] else "calibrate" if p in out["selected"]
                     else "left out" for p in t["parameter"]]
    out["table"] = t.sort_values("delta", ascending=False).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------------------------------------------------

def _f(v, digits: int) -> str:
    return "—" if v is None or v != v else f"{v:.{digits}f}"


def print_screening(df: pd.DataFrame, study: Study, threshold: float, digits: int) -> None:
    r = complete_trajectories(study)
    k = len(study.names)
    print(f"\nScreening: {study.mapper} on {study.header['dem']}, {len(study.sites)} sites, {k} parameters, {r} Morris "
          f"trajectories ({r * (k + 1)} runs)")
    for rd in study.rounds():
        print(f"  round {rd['round']}: {rd['trajectories']} more trajectories of the {len(rd['names'])} that needed "
              f"them ({rd['trajectories'] * (len(rd['names']) + 1)} runs)")
    print(f"effect = change in the sites' mean MCC when a parameter rises by 2/3 of its range (a switch: turned on); "
          f"threshold {threshold}")
    print(f"site μ* = a site's mean |change in MCC|; moved = sites a step changes by {threshold} or more (up/down), "
          f"of {len(study.sites)}")
    w = max(len(n) for n in df["parameter"])
    print(f"{'parameter':<{w}}  {'μ*':>6}  {'95% interval':>14}  {'μ':>6}  {'σ':>6}  {'site μ*':>7}  "
          f"{'site interval':>14}  {'moved (up/dn)':>13}  {'n':>3}  {'class':<17}  most moved site")
    for _, x in df.iterrows():
        if x["class"] == "never active":
            print(f"{x['parameter']:<{w}}  {'':>6}  {'':>14}  {'':>6}  {'':>6}  {'':>7}  {'':>14}  {'':>13}  {0:>3}  "
                  f"never active ({x['kind']})")
            continue
        ci = f"[{_f(x['ci_low'], digits)}, {_f(x['ci_high'], digits)}]"
        sci = f"[{_f(x['site_ci_low'], digits)}, {_f(x['site_ci_high'], digits)}]"
        c = lambda v: "<1" if 0 < v < 0.5 else f"{v:.0f}"      # a mean count per step
        moved = f"{c(x['sites_moved'])} ({c(x['sites_up'])}/{c(x['sites_down'])})"
        top = f"{x['top_site'][:34]} ({_f(x['top_site_mu_star'], digits)})"
        print(f"{x['parameter']:<{w}}  {_f(x['mu_star'], digits):>6}  {ci:>14}  {_f(x['mu'], digits):>6}  "
              f"{_f(x['sigma'], digits):>6}  {_f(x['site_mu_star'], digits):>7}  {sci:>14}  {moved:>13}  "
              f"{x['n']:>3}  {x['class']:<17}  {top}")


def print_identifiability(res: dict, ref_desc: str, step: float, gamma_crit: float, skipped: dict,
                          digits: int) -> None:
    t = res["table"]
    print(f"\nIdentifiability at {ref_desc}: each number ±{step:g} of its range; {res['outputs']} outputs "
          f"(site × stage MCC)")
    if len(t) == 0:
        print("  nothing to test")
        return
    w = max(len(n) for n in t["parameter"])
    print(f"{'parameter':<{w}}  {'δ':>6}  {'∂obj':>6}  {'moved':>5}  {'R² rest':>7}  decision")
    for _, x in t.iterrows():
        print(f"{x['parameter']:<{w}}  {_f(x['delta'], digits):>6}  {_f(x['d_objective'], digits):>6}  "
              f"{x['outputs_moved']:>5}  {_f(x.get('r2_with_rest'), 2):>7}  {x['decision']}")
    print("δ: RMS change of the site × stage MCCs per unit of range; ∂obj: the objective's change per unit of range; "
          "moved: site-stages whose MCC the move changes by 0.01 or more (none: too weak here to judge); R² rest: "
          "the share of its effect the others can reproduce")
    for n, why in skipped.items():
        print(f"  {n}: not tested ({why})")
    print(f"\ncollinearity index of the {len(t) - len(res['no_effect'])} judged: γ = {_f(res['gamma_all'], 1)} "
          f"(identifiable below {gamma_crit:g})")
    if res["combos"]:
        print("combinations that compensate (moving them together this way leaves every site's MCC nearly unchanged):")
        for g, loads in sorted(res["combos"], key=lambda c: -c[0]):
            print(f"  γ {g:5.1f}:  " + "  ".join(f"{v:+.2f} {n}" for n, v in loads))
    if res["selected"]:
        print(f"identifiable subset (γ = {_f(res['gamma_selected'], 1)}): {', '.join(res['selected'])}")
    for n, info in res["left_out"].items():
        by = ", ".join(f"{b} ({c:+.2f})" for b, c in info["by"])
        print(f"  left out {n}: γ {info['gamma']:.1f} with the subset; R² {info['r2']:.2f} from {by}")


CONTROL = "slope_adjustment_factor"


def control_names(mapper: str) -> list[str]:
    """The positive control's parameters: the slope factor and every n class (empty if the file lacks the factor)."""
    S = settings.current()
    if CONTROL not in S.entries or "range" not in S.entries[CONTROL].raw:
        return []
    return [CONTROL] + [k for k, e in S.entries.items() if e.section == "mannings_n" and "range" in e.raw]


def control_check(study: Study, ref_hash: str, names: list[str]) -> dict | None:
    """How collinear the slope factor's sensitivity vector is with the n values' (R², and γ = 1/sqrt(1 - R²))."""
    res = identifiability(study, ref_hash, names, gamma_crit=1e9)
    t = res["table"]
    row = t[t["parameter"] == CONTROL] if len(t) else t
    if len(row) == 0 or row["decision"].iloc[0] == "too weak here":
        return None
    r2 = float(row["r2_with_rest"].iloc[0])
    return {"r2": r2, "gamma": float(1.0 / np.sqrt(max(1.0 - r2, 1e-12))),
            "n_live": [n for n in t["parameter"] if n != CONTROL and n not in res["no_effect"]]}


def print_control(ctrl: dict | None, gamma_crit: float) -> None:
    if ctrl is None:
        print("positive control: the slope factor's move changes no site-stage MCC by 0.01 here; not informative")
        return
    print(f"positive control: {CONTROL}, an exact duplicate of dividing every n by it, has R² {ctrl['r2']:.2f} with "
          f"the n values here (γ {ctrl['gamma']:.1f}).")
    print(f"  A set with γ near {ctrl['gamma']:.1f} is as confounded as an exact duplicate on this output"
          + ("." if ctrl["gamma"] > gamma_crit else f"; γ_crit {gamma_crit:g} is above that, so lower --gamma."))


def print_summary(df: pd.DataFrame, res: dict | None, threshold: float, counts: dict[str, int], r: int) -> None:
    """The suggestions. counts: the trajectories that moved each parameter; r: the trajectories of a round."""
    def names(cls, numbers=None):
        sel = df[df["class"] == cls]
        return [n for n, k in zip(sel["parameter"], sel["kind"])
                if numbers is None or (k.split()[0] in ("float", "int")) == numbers]

    infl, und = names("influential", True), names("undecided")
    print("\nSuggested")
    if res is not None and len(res["table"]):
        judged = set(res["table"]["parameter"])
        calib = [n for n in res["selected"] if n in infl]
        print(f"  calibrate:          {', '.join(calib) or '—'}")
        tie = [n for n in res["left_out"] if n in infl or n in und]
        if tie:
            print(f"  fix or tie:         {', '.join(tie)} (compensated by the ones kept)")
        weak = [n for n in res["no_effect"] if n in infl]
        if weak:
            print(f"  influential, but too weak at the reference to judge: {', '.join(weak)}")
        untested = [n for n in infl if n not in judged]
        if untested:
            print(f"  influential, not tested (their switch is off at the reference): {', '.join(untested)}")
    else:
        print(f"  influential numbers: {', '.join(infl) or '—'}")
    if names("influential", False):
        print(f"  switches / choices: {', '.join(names('influential', False))} (influential: compare their settings)")
    trade, und_sites = names("trades sites"), names("undecided (sites)")
    if trade:
        print(f"  trades sites:       {', '.join(trade)} (no change in the mean MCC, but sites move both ways: its value "
              f"picks which sites do well; set it on physical grounds, or see which sites it moves in "
              f"_site_effects.csv)")
    if und or und_sites:
        most = max(counts.get(n, 0) for n in und + und_sites)
        more = f"after {most} trajectories; another round: --max-trajectories {most + r}"
    if und:
        print(f"  undecided:          {', '.join(und)} ({more})")
    if und_sites:
        print(f"  undecided (sites):  {', '.join(und_sites)} (no change in the mean MCC; unclear whether they move "
              f"sites; {more})")
    neg = names("negligible")
    print(f"  fix at default:     {len(neg)} negligible (μ* and site μ* intervals below {threshold}): "
          f"{', '.join(neg) or '—'}")
    if names("never active"):
        print(f"  never active:       {', '.join(names('never active'))}")


def run(mapper: str, name: str | None = None, trajectories: int = 20, seed: int = 0, threshold: float = 0.01,
        stage: str = "both", only: list[str] | None = None, identify: list[str] | None = None,
        reference: str | None = None, step: float = 0.1, gamma_crit: float = 5.0, prefer: list[str] | None = None,
        site_list: list[str] | None = None, workers: int = 30, jobs: int = 4, dry_run: bool = False,
        digits: int = 2, max_trajectories: int = 60, log=None) -> None:
    from . import sites
    log = log or (lambda *x: print(*x, flush=True))
    S = settings.current()
    name = name or S.dem
    site_list = site_list or sites.calib_sites()
    specs, left_out = S.candidates(mapper, only)
    if not specs:
        raise SystemExit("no candidates to screen")
    names = list(specs)
    header = {"kind": "header", "mapper": mapper, "dem": S.dem, "names": names,
              "specs": {k: asdict(v) for k, v in specs.items()}, "sites": site_list, "seed": seed,
              "setup": setup_fingerprint(mapper, specs), "levels": LEVELS, "settings": str(S.path),
              "created": time.time()}
    if dry_run:
        print(f"{len(names)} candidates for {mapper} on {S.dem} ({len(site_list)} sites); {trajectories} trajectories "
              f"= {trajectories * (len(names) + 1)} runs, then about {2 * len(names) + 1} or fewer for the "
              f"collinearity test")
        if max_trajectories > trajectories:
            print(f"then, while some are undecided, rounds of {trajectories} more trajectories that move only those, "
                  f"until each has {max_trajectories}: a round costs {trajectories} × (undecided + 1) runs, at most "
                  f"{(max_trajectories - trajectories) * (len(names) + 1)} in all if every one stayed undecided")
        for k, s in specs.items():
            print(f"  {k:<30} {s.kind:<6} {settings.describe_spec(s)}")
        for k, why in left_out.items():
            if not why.endswith(" only"):
                print(f"  not screened: {k} ({why})")
        return
    study = Study(mapper, name, header)
    log(f"study {study.path}")
    if stage in ("both", "screen"):
        design = morris_design(names, trajectories, seed)
        for pt in design:
            pt["params"] = params.from_unit(pt["unit"], mapper, names, specs)
        study.run(design, workers, jobs, log, "screening")
        more_trajectories(study, specs, mapper, trajectories, max_trajectories, threshold, workers, jobs, log)
    screened = complete_trajectories(study) > 0
    if not screened and not (stage == "identify" and identify):
        raise SystemExit("no complete trajectory yet: run the screening first (or name --identify parameters)")
    df = screening_table(study, threshold) if screened else pd.DataFrame(columns=["parameter", "kind", "class"])
    if screened:
        print_screening(df, study, threshold, digits)
        df.to_csv(study_path(mapper, name, "_screening.csv"), index=False)
        site_effects(study).to_csv(study_path(mapper, name, "_site_effects.csv"), index=False)
    res = None
    if stage in ("both", "identify"):
        if identify:
            bad = [n for n in identify if n not in specs]
            if bad:
                raise SystemExit(f"--identify: not candidates: {bad}")
            test = list(identify)
        else:
            test = list(df.loc[df["class"].isin(["influential", "undecided"]), "parameter"])
        control = control_names(mapper)
        all_specs = {**{k: S.spec(S.entries[k], mapper) for k in control}, **specs}
        ref = {**{k: S.default(k, mapper) for k in [*names, *control]}, **params.load(reference, mapper)}
        pts, skipped = local_design(ref, list(dict.fromkeys([*test, *control])), all_specs, mapper, step)
        skipped = {n: why for n, why in skipped.items() if n in test and (why != "a switch or choice" or identify)}
        study.run(pts, workers, jobs, log, "collinearity")
        tested = [n for n in test if any(pt["moved"] == n for pt in pts)]
        res = identifiability(study, pts[0]["ref"], tested, gamma_crit, prefer)
        print_identifiability(res, reference or "the settings file's defaults", step, gamma_crit, skipped, digits)
        if control:
            ctrl = control_check(study, pts[0]["ref"], [c for c in control if any(pt["moved"] == c for pt in pts)])
            print_control(ctrl, gamma_crit)
        if len(res["table"]):
            res["table"].to_csv(study_path(mapper, name, "_identifiability.csv"), index=False)
        if res["cos"] is not None:
            res["cos"].to_csv(study_path(mapper, name, "_collinearity.csv"))
    print_summary(df, res, threshold, trajectory_counts(study), trajectories)
    if study.failures():
        print(f"\nnote: {study.failures()} site runs failed and count as MCC 0 (each row's `failed` in {study.path})")
    print(f"\ntables: {study_path(mapper, name, '_*.csv')}")
