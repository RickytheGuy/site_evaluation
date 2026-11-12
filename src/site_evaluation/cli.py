"""
site-eval: run, score and calibrate flood maps on the FIM benchmark sites, and look at the results.

    site-eval prepare  --mapper fldpln                       build the cached preprocessing (after a code/site change)
    site-eval run      --mapper fldpln [--params SET] [--sites ...] [--keep]     run and score a parameter set
    site-eval optimize optuna --mapper fldpln --name NAME     calibrate (also sobol, cma, coord, profile, switches,
                                                              persite)
    site-eval show     runs | run TAG | sites | site NAME | studies | top | importance | plausible | persite RUN |
                       params [SET] | space | settings
    site-eval plot     run TAG | compare TAG TAG | stages TAG | convergence | landscape | persite RUN | sites |
                       site NAME | sensitivity [NAME]
    site-eval sensitivity --mapper fldpln                     which parameters matter (Morris screening) and which
                                                              can be calibrated together (collinearity index)
    site-eval attributes --mapper fldpln                      ARC's reach attributes of every site (runs each once)

Every model argument, its default and whether it is searched (and over what) are in the settings file,
settings.yaml (--settings for another). Parameter sets (--params) override its values: a JSON file, a JSON string
or a preset (G_plaus_fldpln, G_plaus_kernel, reference), naming arguments as the file does. Sites (--sites) are name fragments, or calib / holdout / all.
Scores print at 2 decimals (--digits to change): a change under 0.01 MCC is no change.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import config


def _fmt(v, digits: int) -> str:
    return "—" if v is None or v != v else f"{v:.{digits}f}"


def _print_table(df, digits: int, cols: list[str] | None = None) -> None:
    import pandas as pd
    if df is None or len(df) == 0:
        print("(nothing)")
        return
    df = df[cols] if cols else df
    with pd.option_context("display.max_rows", 500, "display.max_columns", 50, "display.width", 200,
                           "display.max_colwidth", 70):
        print(df.to_string(index=False, float_format=lambda v: f"{v:.{digits}f}"))


def _load_params(spec, mapper):
    from . import params
    return params.load(spec, mapper)


# --------------------------------------------------------------------------------------------------------------------
# prepare / run / attributes
# --------------------------------------------------------------------------------------------------------------------

def cmd_prepare(a):
    from . import pipeline, sites
    pipeline.prepare(a.mapper, sites.select_sites(a.sites), a.workers)


def cmd_run(a):
    from . import params, scoring, sites
    from .runner import Runner
    site_list = sites.select_sites(a.sites)
    p = params.complete(_load_params(a.params, a.mapper), a.mapper)
    stages = None if a.stages == "all" else int(a.stages)
    tag = a.tag or time.strftime("run_%Y%m%d_%H%M%S")
    keep = config.results_path("maps", a.mapper, tag) if a.keep else None
    print(f"{a.mapper}, {len(site_list)} sites, stages {a.stages}, tag {tag}: {params.describe(p, a.mapper)}",
          flush=True)
    t0 = time.time()

    def progress(site, rec):
        if not a.quiet:
            err = "" if rec["ok"] else "  " + rec.get("error", "")[:150]
            print(f"  {site:<70} {_fmt(scoring.site_score(rec), a.digits)}  {rec['seconds']:6.1f}s{err}", flush=True)

    with Runner(a.mapper, min(a.workers, len(site_list))) as runner:
        recs = runner.evaluate(p, site_list, stages, method="run", tag=tag, keep=keep, progress=progress)
    out = scoring.objective(recs)
    print(f"calib {_fmt(out['value'], a.digits)}  hold-out {_fmt(out['holdout'], a.digits)}  "
          f"all {_fmt(out['all'], a.digits)}  ({out['n_sites']} sites, {len(out['failed'])} failed, "
          f"{time.time() - t0:.0f} s)")
    if out["failed"]:
        print("failed:", ", ".join(out["failed"]))
    if keep:
        print(f"maps kept in {keep}")


def cmd_attributes(a):
    from . import stats
    df = stats.attributes(a.mapper, _load_params(a.params, a.mapper), a.workers)
    _print_table(df.describe().T.reset_index(), a.digits)
    print(f"wrote {config.results_path(f'site_attributes_{a.mapper}.csv')}")


# --------------------------------------------------------------------------------------------------------------------
# optimize
# --------------------------------------------------------------------------------------------------------------------

def _json_arg(v):
    if v is None:
        return None
    path = Path(v).expanduser()
    return json.loads(path.read_text()) if path.exists() else json.loads(v)


def cmd_optimize(a):
    from . import optimize, params, sites
    fixed = _json_arg(getattr(a, "fixed", None))
    bounds = _json_arg(getattr(a, "bounds", None))
    x0 = params.complete(_load_params(a.x0, a.mapper), a.mapper) if getattr(a, "x0", None) else None
    if a.method == "optuna":
        enqueue = [params.complete(_load_params(s, a.mapper), a.mapper) for s in (a.enqueue or [])]
        optimize.run_optuna(a.mapper, a.name, a.sampler, a.trials, a.jobs, a.workers, a.startup, a.seed, fixed,
                            enqueue)
        return
    if a.method == "persite":
        base = params.complete(_load_params(a.base, a.mapper), a.mapper)
        optimize.run_persite(a.mapper, base, a.run, sites.select_sites(a.sites), a.reduced, tuple(a.search),
                             a.budget, a.startup, a.sobol, a.workers, a.seed)
        return
    with optimize.Evaluator(a.mapper, a.workers, a.jobs) as ev:
        if a.method == "sobol":
            optimize.sobol(ev, a.n, a.seed)
        elif a.method == "cma":
            optimize.cma(ev, a.tag, a.budget, a.popsize, a.sigma, a.seed, x0, fixed, bounds)
        elif a.method == "coord":
            optimize.coord(ev, a.tag, a.step, a.min_step, a.max_iter, a.tol, x0, fixed, bounds)
        elif a.method == "profile":
            optimize.profile(ev, a.tag, a.points, x0, a.only)
        elif a.method == "switches":
            bases = [params.complete(_load_params(b, a.mapper), a.mapper) for b in a.bases] if a.bases else None
            optimize.switches(ev, a.tag, bases)


# --------------------------------------------------------------------------------------------------------------------
# show
# --------------------------------------------------------------------------------------------------------------------

def cmd_show(a):
    from . import params, results, sites, stats
    w = a.what
    if w == "runs":
        df = results.runs_table(a.mapper, a.method)
        if not a.method and not df.empty:
            df = df[~df["method"].isin(["optuna"]) & ~df["method"].str.startswith("persite")]
        _print_table(df.tail(a.last), a.digits, ["time", "method", "tag", "sites", "failed", "calib", "holdout",
                                                 "all", "setup"] if not df.empty else None)
    elif w == "run":
        recs = results.run_records(a.mapper, a.target)
        t = results.run_sites_table(recs).sort_values(a.sort, ascending=a.sort == "site")
        _print_table(t, a.digits, ["site", "split", "score", "stages", "bias", "recall", "precision", "seconds",
                                   "error"])
        from . import scoring
        o = scoring.objective(recs)
        print(f"calib {_fmt(o['value'], a.digits)}  hold-out {_fmt(o['holdout'], a.digits)}  "
              f"all {_fmt(o['all'], a.digits)}  ({o['n_sites']} sites, {len(o['failed'])} failed)")
        first = next(iter(recs.values()))
        print("parameters:", params.describe(params.convert_legacy(first["params"]), a.mapper))
        if first.get("code"):
            print("code:", json.dumps(first["code"]), " setup:", first.get("setup"))
    elif w == "sites":
        df = stats.site_table(a.refresh, log=print if a.refresh else None)
        cols = [c for c in ["name", "year", "split", "stages", "reaches", "da_km2", "stream_km", "domain_km2",
                            "bankfull_cms", "top_over_bankfull", "ref_flooded_low", "ref_flooded_high"] if c in df]
        _print_table(df, a.digits, cols)
        print(f"{len(df)} sites ({(df['split'] == 'hold').sum()} hold-out); table in "
              f"{config.results_path('site_stats.csv')}")
    elif w == "site":
        site = sites.select_sites([a.target])
        if len(site) != 1:
            raise SystemExit(f"{a.target!r} matches {len(site)} sites: {site[:5]}")
        site = site[0]
        df = stats.site_table()
        row = df[df["site"] == site].iloc[0]
        print(site)
        for k, v in row.items():
            if k not in ("site",):
                print(f"  {k:<20} {_fmt(v, a.digits) if isinstance(v, float) else v}")
        print("  q_stage groups:")
        for g, d in sites.q_stage(site).items():
            print(f"    {g}: {len(d['rivids'])} reaches, bankfull {d.get('bankfull')} cfs, baseflow "
                  f"{d.get('baseflow')} cfs, stages {', '.join(list(d['stages'])[:20])}")
        runs = results.archive_frame(a.mapper)
        if not runs.empty:
            mine = runs[runs["site"] == site].tail(a.last)
            print(f"  latest {a.mapper} evaluations:")
            _print_table(mine.assign(time=lambda d: d["time"].map(lambda t: time.strftime("%Y-%m-%d %H:%M",
                                                                                         time.localtime(t)))),
                         a.digits, ["time", "method", "tag", "score", "n_stages", "ok"])
    elif w == "studies":
        df = results.optimization_frame(a.mapper)
        _print_table(results.convergence(df) if not df.empty else None, a.digits)
    elif w == "top":
        df = results.optimization_frame(a.mapper)
        if a.plausible:
            df = df[df["plausible"]]
        d = df.sort_values("value", ascending=False).head(a.k)
        for _, r in d.iterrows():
            print(f"{r['method']:<28} calib {_fmt(r['value'], a.digits)} hold {_fmt(r['holdout'], a.digits)} "
                  f"{'plausible' if r['plausible'] else 'implausible n'} | {params.describe(r['params'], a.mapper)}")
    elif w == "importance":
        imp, r2 = results.importance(results.optimization_frame(a.mapper), a.mapper)
        print(f"surrogate (extra trees), in-sample R² {r2:.2f}; permutation importance:")
        for n, v in imp.items():
            print(f"  {n:<18} {v:.4f}")
    elif w == "plausible":
        df = results.optimization_frame(a.mapper)
        b, gap = results.best_plausible(df)
        p = params.searched(b["params"], a.mapper)
        print(f"best plausible: calib {_fmt(b['value'], a.digits)} hold {_fmt(b['holdout'], a.digits)} "
              f"({b['method']} {b.get('tag', '')}); best overall {_fmt(df['value'].max(), a.digits)}, "
              f"gap {gap:.4f}")
        print(json.dumps(p))
        if a.save:
            Path(a.save).write_text(json.dumps(p, indent=1))
            print(f"wrote {a.save}")
    elif w == "persite":
        if not a.target:
            print("per-site runs:", ", ".join(results.persite_runs(a.mapper)) or "none")
            return
        s = results.persite_summary(a.mapper, a.target)
        _print_table(s, a.digits)
        print(f"median gain {s['gain'].median():.{a.digits}f}; {(s['gain'] >= 0.01).sum()} of {len(s)} sites gain "
              f"≥ 0.01")
    elif w == "params":
        from . import settings
        p = params.complete(_load_params(a.target, a.mapper), a.mapper)
        site = sites.select_sites([a.site])[0] if a.site else sites.all_sites()[0]
        neo, arc, c2f, lib = params.model_args(p, a.mapper, site)
        S = settings.current()
        print(json.dumps({"parameters": p, "plausible n": params.is_plausible(p),
                          f"arc_args (at {site}, cell {S.cell_size_m(site):.1f} m)": arc, "curve2flood_args": c2f,
                          "fldpln_library_args": lib, "nencarta": neo}, indent=1, default=str))
    elif w in ("space", "settings"):
        import pandas as pd
        from . import settings
        S = settings.current()
        rows = pd.DataFrame(S.table(a.mapper))
        if w == "space":
            rows = rows[rows["searched"] != ""]
        elif a.section:
            rows = rows[rows["section"].isin(a.section)]
        _print_table(rows, a.digits)
        print(f"{S.path}: DEM {S.dem}, cell size {S.cell_size}"
              + (f" ({S.nominal_cell_size[S.dem]} m)" if S.cell_size == "nominal" else "")
              + f"; {len(S.space(a.mapper))} parameters searched for {a.mapper}; fixed-settings fingerprint "
              + S.fingerprint(a.mapper))


def cmd_sensitivity(a):
    from . import sensitivity, sites
    sensitivity.run(a.mapper, a.name, a.trajectories, a.seed, a.threshold, a.stage, a.candidates, a.plausible,
                    a.identify, a.reference, a.step, a.gamma, a.prefer,
                    sites.select_sites(a.sites) if a.sites else None, a.workers, a.jobs, a.dry_run, a.digits)


# --------------------------------------------------------------------------------------------------------------------
# plot
# --------------------------------------------------------------------------------------------------------------------

def cmd_plot(a):
    from . import plots, results, sites, stats
    w = a.what
    if w == "run":
        recs = results.run_records(a.mapper, a.target)
        path = plots.run_sites(results.run_sites_table(recs), f"{a.mapper} run {a.target}", a.out)
    elif w == "compare":
        if not a.other:
            raise SystemExit("plot compare TAG_A TAG_B")
        ta = results.run_sites_table(results.run_records(a.mapper, a.target))
        tb = results.run_sites_table(results.run_records(a.other_mapper or a.mapper, a.other))
        path = plots.run_compare(ta, tb, a.target, a.other, a.out)
    elif w == "stages":
        recs = results.run_records(a.mapper, a.target)
        path = plots.run_stages(results.run_stages_table(recs), f"{a.mapper} run {a.target}: MCC by stage", a.out)
    elif w == "convergence":
        path = plots.convergence(results.optimization_frame(a.mapper), a.mapper, a.out)
    elif w == "landscape":
        path = plots.landscape(results.optimization_frame(a.mapper), a.mapper, a.out)
    elif w == "persite":
        path = plots.persite_gains(results.persite_summary(a.mapper, a.target), f"{a.mapper} per-site {a.target}",
                                   a.out)
    elif w == "sites":
        scores = results.run_sites_table(results.run_records(a.mapper, a.target)) if a.target else None
        path = plots.site_stats(stats.site_table(), scores, a.out)
    elif w == "site":
        match = sites.select_sites([a.target])
        if len(match) != 1:
            raise SystemExit(f"{a.target!r} matches {len(match)} sites: {match[:5]}")
        site = match[0]
        rec = maps = None
        if a.tag:
            rec = results.run_records(a.mapper, a.tag).get(site)
            maps = config.results_path("maps", a.mapper, a.tag)
            if not (maps / site).exists():
                print(f"no kept maps for this site in {maps} (run with --keep); showing the reference only")
                maps = None
        path = plots.site_maps(site, a.stages, maps, rec, a.out)
    elif w == "sensitivity":
        from . import sensitivity, settings
        name = a.target or settings.current().dem
        study = sensitivity.Study(a.mapper, name)
        path = plots.sensitivity_screening(sensitivity.screening_table(study, a.threshold), study, a.threshold,
                                           a.out)
        coll = sensitivity.study_path(a.mapper, name, "_collinearity.csv")
        if coll.exists():
            import pandas as pd
            print(f"wrote {path}")
            path = plots.sensitivity_collinearity(
                pd.read_csv(coll, index_col=0), f"{a.mapper} on {study.header['dem']}: similarity of the parameters' "
                "effects", str(Path(a.out).with_name(Path(a.out).stem + "_collinearity.png")) if a.out else
                config.results_path("fig", f"sensitivity_{a.mapper}_{name}_collinearity.png"))
    print(f"wrote {path}")


# --------------------------------------------------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="site-eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=None, help=f"results folder (default {config.RESULTS_ROOT})")
    ap.add_argument("--settings", default=None, help="settings file (default: $SITE_EVAL_SETTINGS or the package's "
                                                     "settings.yaml)")
    sub = ap.add_subparsers(dest="command", required=True)
    mappers = list(config.MAPPERS)

    def common(p, mapper_required=True, digits=True):
        if mapper_required:
            p.add_argument("--mapper", "-m", choices=mappers, required=True)
        else:
            p.add_argument("--mapper", "-m", choices=mappers, default="fldpln")
        if digits:
            p.add_argument("--digits", type=int, default=2, help="decimals for scores (default 2)")

    p = sub.add_parser("prepare", help="build the cached preprocessing for a mapper")
    common(p, digits=False)
    p.add_argument("--sites", nargs="*")
    p.add_argument("--workers", type=int, default=30)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("run", help="run and score one parameter set")
    common(p)
    p.add_argument("--params", help="preset, JSON file or JSON string (default: the mapper's G_plaus set)")
    p.add_argument("--sites", nargs="*")
    p.add_argument("--stages", default="all", help="'all' (default) or k stages per site, evenly spread")
    p.add_argument("--workers", type=int, default=30)
    p.add_argument("--tag", help="name of the run (default run_<date>_<time>)")
    p.add_argument("--keep", action="store_true", help="keep the flood maps (results/maps/<mapper>/<tag>)")
    p.add_argument("--quiet", action="store_true", help="only the summary")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("attributes", help="ARC's reach attributes of every site (runs each once)")
    common(p)
    p.add_argument("--params")
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=cmd_attributes)

    p = sub.add_parser("optimize", help="calibrate: optuna, sobol, cma, coord, profile, switches, persite")
    osub = p.add_subparsers(dest="method", required=True)

    def opt(name, help_):
        q = osub.add_parser(name, help=help_)
        common(q, digits=False)
        q.add_argument("--workers", type=int, default=30)
        q.add_argument("--seed", type=int, default=0)
        return q

    q = opt("optuna", "a global optuna study (resumes by name)")
    q.add_argument("--name", required=True)
    q.add_argument("--sampler", choices=["tpe", "tpe_indep", "cmaes", "qmc", "random"], default="tpe")
    q.add_argument("--trials", type=int, default=1000)
    q.add_argument("--jobs", type=int, default=3, help="parameter sets in flight")
    q.add_argument("--startup", type=int, default=50)
    q.add_argument("--fixed", help="JSON dict (or file) of parameters held fixed")
    q.add_argument("--enqueue", nargs="*", help="parameter sets to try first")
    q = opt("sobol", "a scrambled Sobol screen of the whole space")
    q.add_argument("--n", type=int, default=1024)
    q.add_argument("--jobs", type=int, default=3)
    for name, help_ in (("cma", "CMA-ES on the continuous parameters"), ("coord", "compass search")):
        q = opt(name, help_)
        q.add_argument("--tag", default=name)
        q.add_argument("--jobs", type=int, default=3)
        q.add_argument("--x0", help="start set (default: the best evaluated so far)")
        q.add_argument("--fixed")
        q.add_argument("--bounds", help='JSON {name: [lo, hi]} limits inside the space')
        if name == "cma":
            q.add_argument("--budget", type=int, default=600)
            q.add_argument("--popsize", type=int, default=12)
            q.add_argument("--sigma", type=float, default=0.2)
        else:
            q.add_argument("--step", type=float, default=0.1)
            q.add_argument("--min-step", type=float, default=0.012)
            q.add_argument("--max-iter", type=int, default=40)
            q.add_argument("--tol", type=float, default=1e-4)
    q = opt("profile", "1-D sweeps of each parameter")
    q.add_argument("--tag", default="profile")
    q.add_argument("--jobs", type=int, default=3)
    q.add_argument("--x0")
    q.add_argument("--points", type=int, default=13)
    q.add_argument("--only", nargs="*")
    q = opt("switches", "each switch flipped alone")
    q.add_argument("--tag", default="switches")
    q.add_argument("--jobs", type=int, default=3)
    q.add_argument("--bases", nargs="*")
    q = opt("persite", "per-site tuning of a reduced set around a general set")
    q.add_argument("--run", required=True)
    q.add_argument("--base", help="the general set (default: the mapper's G_plaus)")
    q.add_argument("--reduced", nargs="*", help="fp_mult ch_mult mfd tw_mult trap gsd")
    q.add_argument("--search", nargs="*", default=["tpe", "own"])
    q.add_argument("--budget", type=int, default=150)
    q.add_argument("--startup", type=int, default=24)
    q.add_argument("--sobol", type=int, default=64)
    q.add_argument("--sites", nargs="*")
    p.set_defaults(func=cmd_optimize)

    p = sub.add_parser("sensitivity", help="which parameters matter (Morris screening) and which can be calibrated "
                                           "together (collinearity index); resumes by name",
                       description="Screens every candidate parameter of the settings file (an optimize field, not "
                                   "screen: false) with Morris elementary effects on the mean MCC of the calibration "
                                   "sites, then tests the influential numbers for collinearity (Brun et al. 2001) at "
                                   "the defaults. See the sensitivity module's documentation for the method.")
    common(p)
    p.add_argument("--name", help="study name (default: the DEM's name); the same name resumes or extends it")
    p.add_argument("--trajectories", "-r", type=int, default=20, help="Morris trajectories (default 20)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.01, help="MCC change that counts (default 0.01)")
    p.add_argument("--stage", choices=["both", "screen", "identify"], default="both")
    p.add_argument("--candidates", nargs="*", help="screen only these (they may include screen: false ones)")
    p.add_argument("--plausible", action="store_true", help="screen n over its plausible band, not its range")
    p.add_argument("--identify", nargs="*", help="the numbers to test for collinearity (default: the influential and "
                                                 "undecided ones)")
    p.add_argument("--reference", help="the set the collinearity is tested at (default: the settings file's defaults)")
    p.add_argument("--step", type=float, default=0.1, help="collinearity: the move, as a share of each range (0.1)")
    p.add_argument("--gamma", type=float, default=5.0, help="collinearity index above which a set isn't "
                                                            "identifiable (default 5; see the positive control)")
    p.add_argument("--prefer", nargs="*", help="keep these first in the identifiable subset (e.g. n over slope)")
    p.add_argument("--sites", nargs="*", help="default: the calibration sites")
    p.add_argument("--workers", type=int, default=30)
    p.add_argument("--jobs", type=int, default=4, help="parameter sets in flight")
    p.add_argument("--dry-run", action="store_true", help="list the candidates and the number of runs")
    p.set_defaults(func=cmd_sensitivity)

    p = sub.add_parser("show", help="tables: runs, run TAG, sites, site NAME, studies, top, importance, plausible, "
                                    "persite [RUN], params [SET], space, settings")
    p.add_argument("what", choices=["runs", "run", "sites", "site", "studies", "top", "importance", "plausible",
                                    "persite", "params", "space", "settings"])
    p.add_argument("target", nargs="?", help="the run tag, site, per-site run or parameter set")
    common(p, mapper_required=False)
    p.add_argument("--method", nargs="*", help="runs: only these methods (default: all but optuna and persite)")
    p.add_argument("--last", type=int, default=30, help="runs / site: how many to list")
    p.add_argument("--sort", default="score", choices=["score", "site", "bias", "seconds"])
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--plausible", action="store_true", help="top: only sets with plausible n")
    p.add_argument("--refresh", action="store_true", help="sites: recompute the site table")
    p.add_argument("--save", help="plausible: write the set to this JSON file")
    p.add_argument("--site", help="params: the site whose cell size converts the metre stand-ins")
    p.add_argument("--section", nargs="*", help="settings: only these sections (arc, curve2flood, ...)")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("plot", help="figures: run TAG, compare TAG TAG, stages TAG, convergence, landscape, "
                                    "persite RUN, sites [TAG], site NAME [--tag TAG], sensitivity [NAME]")
    p.add_argument("what", choices=["run", "compare", "stages", "convergence", "landscape", "persite", "sites",
                                    "site", "sensitivity"])
    p.add_argument("target", nargs="?")
    p.add_argument("other", nargs="?", help="compare: the second run's tag")
    common(p, mapper_required=False, digits=False)
    p.add_argument("--other-mapper", choices=mappers, help="compare: the second run's mapper, if different")
    p.add_argument("--tag", help="site: a run whose kept maps to show")
    p.add_argument("--stages", nargs="*", help="site: which stages (default up to 6, evenly spread)")
    p.add_argument("--threshold", type=float, default=0.01, help="sensitivity: MCC change that counts")
    p.add_argument("--out", help="output PNG (default results/fig/<name>.png)")
    p.set_defaults(func=cmd_plot)
    return ap


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    if a.results:
        config.RESULTS_ROOT = Path(a.results).expanduser().resolve()
    from . import settings
    try:
        settings.use(a.settings)
    except settings.SettingsError as exc:
        raise SystemExit(f"settings: {exc}") from None
    if a.command == "show" and a.what in ("run", "site") and a.target is None:
        raise SystemExit(f"show {a.what} needs a run tag or site name")
    if a.command == "plot" and a.what in ("run", "compare", "stages", "persite", "site") and a.target is None:
        raise SystemExit(f"plot {a.what} needs a run tag, per-site run or site name")
    a.func(a)


if __name__ == "__main__":
    main()
