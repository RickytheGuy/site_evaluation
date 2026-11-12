"""
Which parameters to calibrate (``site-eval sensitivity``): a global screening of every candidate parameter, then a
test of whether the influential ones can be calibrated together.

Candidates (Settings.candidates): every argument in the settings file with an optimize field, true or false, that the
mapper uses, except nencarta's (they shape the cached preprocessing) and those marked ``screen: false`` (fixed
beforehand, such as Standard_Step on physical grounds). Each is screened over its range (a log range on the log scale).

1. Screening: the elementary effects method (Morris 1991) with μ* (Campolongo et al. 2007).
   r trajectories cross a grid of 4 levels per parameter. Each step of a trajectory moves one parameter by
   Δ = 2/3 of its range (a switch is flipped, a choice changes). A step's effect is the change in the objective, the
   mean MCC of the study's sites (by default the calibration sites), signed for a rise of the parameter. Per
   parameter: μ* = the mean |effect| (its influence), μ = the mean effect (its direction), σ = the effects' standard
   deviation (non-linearity and interactions), and a bootstrap 95% interval of μ* over the trajectories. Against a
   threshold (0.01 MCC by default: a smaller change is no change):
       influential   the interval lies above the threshold
       negligible    the interval lies below it: fix the parameter at its default
       undecided     the interval straddles it: run more trajectories
   A step taken while the switch a parameter depends on is off has no effect and is left out. "site μ*" is the mean
   of each site's own μ*: far above μ*, the parameter moves each site but in different directions, so a single
   general value gains little from it (a per-site candidate).

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
   every n by it: a structural duplicate, fixed beforehand (screen: false). Every collinearity test also moves it
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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, params, scoring, settings

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

def morris_design(names: list[str], r: int, seed: int) -> list[dict]:
    """r Morris trajectories of len(names)+1 points on the unit cube. Trajectory t depends only on (seed, t), so a
    study can be extended with more trajectories."""
    k = len(names)
    out = []
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
    least one whole number for an integer). Also the names left out, with the reason."""
    rh = _hash({"ref": ref, "step": step})
    pts = [{"kind": "local", "key": f"L{rh}.base", "ref": rh, "moved": None, "params": ref}]
    skipped = {}
    for n in names:
        s = specs[n]
        if s.kind not in ("float", "int"):
            skipped[n] = "a switch or choice"
            continue
        if s.when and not params.switch_on(ref, s.when, mapper):
            skipped[n] = f"{s.when} is off at the reference"
            continue
        u0 = float(params.to_unit(ref, mapper, [n], specs)[0])
        for sign in (1, -1):
            u = float(np.clip(u0 + sign * step, 0.0, 1.0))
            if abs(u - u0) < 1e-9:
                continue
            v = params.from_unit([u], mapper, [n], specs)[n]
            if s.kind == "int":
                if v == ref[n]:
                    v = int(np.clip(ref[n] + sign, s.low, s.high))
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
            with open(self.path) as f:
                lines = [json.loads(line) for line in f if line.strip()]
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

    def run(self, points: list[dict], workers: int, jobs: int, log=print) -> None:
        """Run the points not run yet, jobs parameter sets at a time."""
        todo = [pt for pt in points if pt["key"] not in self.rows]
        if not todo:
            return
        from .runner import Runner
        lock = threading.Lock()
        t0, count = time.time(), [0]
        every = max(1, len(todo) // 20)
        log(f"[{_stamp()}] {len(todo)} runs of {len(self.sites)} sites ({len(points) - len(todo)} done before)")
        with Runner(self.mapper, workers) as runner:
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
                    count[0] += 1
                    if count[0] % every == 0 or count[0] == len(todo):
                        el = time.time() - t0
                        left = el / count[0] * (len(todo) - count[0])
                        log(f"[{_stamp()}] {count[0]}/{len(todo)} runs, {el / 60:.0f} min, ~{left / 60:.0f} min left")

            with ThreadPoolExecutor(jobs) as ex:
                list(ex.map(one, todo))

    def failures(self) -> int:
        return sum(len(r.get("failed", [])) for r in self.rows.values())


# --------------------------------------------------------------------------------------------------------------------
# 1. Screening
# --------------------------------------------------------------------------------------------------------------------

def morris_effects(study: Study) -> pd.DataFrame:
    """One row per elementary effect of the complete trajectories: traj, name, effect (the objective's change for a
    rise of Δ), and each site's change in score."""
    k = len(study.names)
    trajs = sorted({r["traj"] for r in study.rows.values() if r["kind"] == "morris"})
    out = []
    for t in trajs:
        steps = [study.rows.get(f"m{t}.{i}") for i in range(k + 1)]
        if any(s is None for s in steps):
            continue
        for prev, cur in zip(steps, steps[1:]):
            n = cur["moved"]
            spec = study.specs[n]
            if spec.when and not params.switch_on(prev["params"], spec.when, study.mapper):
                continue
            sign = cur["sign"]
            out.append({"traj": t, "name": n, "effect": sign * (cur["value"] - prev["value"]),
                        **{s: sign * (cur["sites"][s] - prev["sites"][s]) for s in study.sites}})
    return pd.DataFrame(out)


def complete_trajectories(study: Study) -> int:
    k = len(study.names)
    trajs = {r["traj"] for r in study.rows.values() if r["kind"] == "morris"}
    return sum(all(f"m{t}.{i}" in study.rows for i in range(k + 1)) for t in trajs)


def screening_table(study: Study, threshold: float = 0.01, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per candidate: μ*, its bootstrap 95% interval, μ, σ, site μ*, the site it moves most, the number of effects,
    and its class. Sorted by μ*."""
    eff = morris_effects(study)
    rng = np.random.default_rng(seed)
    rows = []
    for n in study.names:
        spec = study.specs[n]
        e = eff[eff["name"] == n] if len(eff) else eff
        base = {"parameter": n, "kind": spec.kind + (f" (when {spec.when})" if spec.when else ""), "n": len(e)}
        if len(e) == 0:
            rows.append({**base, "class": "never active"})
            continue
        d = e["effect"].to_numpy(float)
        a = np.abs(d)
        boot = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        per_site = e[study.sites].abs().mean(axis=0)
        cls = ("undecided" if len(a) < MIN_EFFECTS else "influential" if lo >= threshold
               else "negligible" if hi < threshold else "undecided")
        rows.append({**base, "mu_star": a.mean(), "ci_low": lo, "ci_high": hi, "mu": d.mean(),
                     "sigma": d.std(ddof=1) if len(d) > 1 else np.nan, "site_mu_star": float(per_site.mean()),
                     "top_site": per_site.idxmax(), "top_site_mu_star": float(per_site.max()), "class": cls})
    df = pd.DataFrame(rows)
    return df.sort_values("mu_star", ascending=False, na_position="last").reset_index(drop=True)


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
    print(f"effect = change in the sites' mean MCC when a parameter rises by 2/3 of its range (a switch: turned on); "
          f"threshold {threshold}")
    w = max(len(n) for n in df["parameter"])
    print(f"{'parameter':<{w}}  {'μ*':>6}  {'95% interval':>14}  {'μ':>6}  {'σ':>6}  {'site μ*':>7}  {'n':>3}  class"
          f"         most moved site")
    for _, x in df.iterrows():
        if x["class"] == "never active":
            print(f"{x['parameter']:<{w}}  {'':>6}  {'':>14}  {'':>6}  {'':>6}  {'':>7}  {0:>3}  never active "
                  f"({x['kind']})")
            continue
        ci = f"[{_f(x['ci_low'], digits)}, {_f(x['ci_high'], digits)}]"
        top = f"{x['top_site'][:34]} ({_f(x['top_site_mu_star'], digits)})"
        print(f"{x['parameter']:<{w}}  {_f(x['mu_star'], digits):>6}  {ci:>14}  {_f(x['mu'], digits):>6}  "
              f"{_f(x['sigma'], digits):>6}  {_f(x['site_mu_star'], digits):>7}  {x['n']:>3}  {x['class']:<12}  {top}")


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


def print_summary(df: pd.DataFrame, res: dict | None, threshold: float, trajectories: int) -> None:
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
    if und:
        print(f"  undecided:          {', '.join(und)} (more trajectories: --trajectories {2 * trajectories})")
    neg = names("negligible")
    print(f"  fix at default:     {len(neg)} negligible (μ* interval below {threshold}): {', '.join(neg) or '—'}")
    if names("never active"):
        print(f"  never active:       {', '.join(names('never active'))}")


def run(mapper: str, name: str | None = None, trajectories: int = 20, seed: int = 0, threshold: float = 0.01,
        stage: str = "both", only: list[str] | None = None, plausible: bool = False, identify: list[str] | None = None,
        reference: str | None = None, step: float = 0.1, gamma_crit: float = 5.0, prefer: list[str] | None = None,
        site_list: list[str] | None = None, workers: int = 30, jobs: int = 4, dry_run: bool = False,
        digits: int = 2, log=None) -> None:
    from . import sites
    log = log or (lambda *x: print(*x, flush=True))
    S = settings.current()
    name = name or S.dem + ("_plausible" if plausible else "")
    site_list = site_list or sites.calib_sites()
    specs, left_out = S.candidates(mapper, plausible, only)
    if not specs:
        raise SystemExit("no candidates to screen")
    names = list(specs)
    header = {"kind": "header", "mapper": mapper, "dem": S.dem, "names": names,
              "specs": {k: asdict(v) for k, v in specs.items()}, "sites": site_list, "seed": seed,
              "setup": setup_fingerprint(mapper, specs), "levels": LEVELS, "settings": str(S.path),
              "plausible": plausible, "created": time.time()}
    if dry_run:
        print(f"{len(names)} candidates for {mapper} on {S.dem} ({len(site_list)} sites); {trajectories} trajectories "
              f"= {trajectories * (len(names) + 1)} runs, then about {2 * len(names) + 1} or fewer for the "
              f"collinearity test")
        for k, s in specs.items():
            rng = (f"{s.low:g} – {s.high:g}{' log' if s.log else ''}" if s.kind in ("float", "int")
                   else "on/off" if s.kind == "bool" else "/".join(map(str, s.choices)))
            print(f"  {k:<30} {s.kind:<6} {rng}{f'  (when {s.when})' if s.when else ''}")
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
        study.run(design, workers, jobs, log)
    screened = complete_trajectories(study) > 0
    if not screened and not (stage == "identify" and identify):
        raise SystemExit("no complete trajectory yet: run the screening first (or name --identify parameters)")
    df = screening_table(study, threshold) if screened else pd.DataFrame(columns=["parameter", "kind", "class"])
    if screened:
        print_screening(df, study, threshold, digits)
        df.to_csv(study_path(mapper, name, "_screening.csv"), index=False)
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
        study.run(pts, workers, jobs, log)
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
    print_summary(df, res, threshold, complete_trajectories(study))
    if study.failures():
        print(f"\nnote: {study.failures()} site runs failed and count as MCC 0 (each row's `failed` in {study.path})")
    print(f"\ntables: {study_path(mapper, name, '_*.csv')}")
