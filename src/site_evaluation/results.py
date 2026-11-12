"""
Reading results back: runs from the archive, optimization studies, per-site tuning, and summaries of each.
Everything returns pandas DataFrames, so it works the same from the command line and from a notebook.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import config, params, scoring, sites
from .runner import read_archive


# --------------------------------------------------------------------------------------------------------------------
# Runs (the archive)
# --------------------------------------------------------------------------------------------------------------------

def archive_frame(mapper: str) -> pd.DataFrame:
    """One row per (site, parameter set) evaluation, with its score."""
    rows = read_archive(mapper)
    if not rows:
        return pd.DataFrame(columns=["method", "tag", "time", "site", "ok", "score"])
    df = pd.DataFrame(rows)
    df["score"] = [scoring.site_score(r) for r in rows]
    df["n_stages"] = [len(r.get("stages") or []) for r in rows]
    for col in ("setup", "error"):
        if col not in df:
            df[col] = None
    return df


def runs_table(mapper: str, methods: list[str] | None = None, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per run (method, tag): when it ran, how many sites, the calibration / hold-out / all-site means over
    the sites it has, and how many failed."""
    df = archive_frame(mapper) if df is None else df
    if df.empty:
        return pd.DataFrame()
    if methods:
        df = df[df["method"].isin(methods)]
    hold = set(sites.holdout_sites())
    out = []
    for (method, tag), g in df.groupby(["method", "tag"], sort=False):
        g = g.drop_duplicates("site", keep="last")
        is_hold = g["site"].isin(hold)
        out.append({"method": method, "tag": tag,
                    "time": time.strftime("%Y-%m-%d %H:%M", time.localtime(g["time"].max())),
                    "sites": len(g), "failed": int((~g["ok"]).sum()),
                    "calib": g.loc[~is_hold, "score"].mean(), "holdout": g.loc[is_hold, "score"].mean(),
                    "all": g["score"].mean(), "setup": g["setup"].iloc[-1],
                    "code": json.dumps(g["code"].iloc[-1]) if "code" in g and isinstance(g["code"].iloc[-1], dict)
                    else ""})
    return pd.DataFrame(out).sort_values("time").reset_index(drop=True)


def run_records(mapper: str, tag: str, method: str | None = None) -> dict[str, dict]:
    """The latest record of each site in one run."""
    recs = {}
    for r in read_archive(mapper):
        if r.get("tag") == tag and (method is None or r.get("method") == method):
            recs[r["site"]] = r
    if not recs:
        raise SystemExit(f"No run tagged {tag!r} in {config.results_path(f'{mapper}_evals.jsonl')}")
    order = {s: i for i, s in enumerate(sites.all_sites())}
    return dict(sorted(recs.items(), key=lambda kv: order.get(kv[0], len(order))))


def run_sites_table(recs: dict[str, dict]) -> pd.DataFrame:
    """Per site of one run: score, stages, median bias, recall, precision, time and error."""
    hold = set(sites.holdout_sites())
    rows = []
    for s, r in recs.items():
        def med(key):
            vals = [v for v in (r.get(key) or []) if v is not None]
            return float(np.median(vals)) if vals else np.nan
        rows.append({"site": s, "split": "hold" if s in hold else "calib", "score": scoring.site_score(r),
                     "stages": len(r.get("stages") or []), "bias": med("bias"), "recall": med("recall"),
                     "precision": med("precision"), "seconds": r.get("seconds"), "ok": r.get("ok"),
                     "error": r.get("error", "")})
    return pd.DataFrame(rows)


def run_stages_table(recs: dict[str, dict]) -> pd.DataFrame:
    """One row per (site, stage) of one run, with the stage's rank scaled to 0 (lowest) - 1 (highest)."""
    rows = []
    for s, r in recs.items():
        st = r.get("stages") or []
        for i, stage in enumerate(st):
            row = {"site": s, "stage": stage, "rank": i / max(len(st) - 1, 1)}
            for m in scoring.KEPT_METRICS:
                vals = r.get(m)
                row[m] = vals[i] if vals and vals[i] is not None else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------------------------
# Optimization studies
# --------------------------------------------------------------------------------------------------------------------

def _site_scores(v) -> dict | None:
    return v if isinstance(v, dict) else None      # fim3 stored a list aligned to its own site list


def optuna_frame(mapper: str, study: str | None = None) -> pd.DataFrame:
    import optuna
    path = config.results_path(f"optuna_{mapper}.db")
    if not path.exists():
        return pd.DataFrame()
    storage = f"sqlite:///{path}"
    rows = []
    for s in optuna.get_all_study_summaries(storage):
        if study and s.study_name != study:
            continue
        st = optuna.load_study(study_name=s.study_name, storage=storage)
        for t in st.get_trials(deepcopy=False):
            if t.value is None or "params_full" not in t.user_attrs:
                continue
            ua = t.user_attrs
            rows.append({"method": f"optuna:{s.study_name}", "order": t.number,
                         "time": t.datetime_complete.timestamp() if t.datetime_complete else np.nan,
                         "value": t.value, "holdout": ua.get("holdout"), "all": ua.get("all"),
                         "params": params.convert_legacy(ua["params_full"]), "site_scores": _site_scores(ua.get("site_scores")),
                         "tag": f"{s.study_name}#{t.number}"})
    return pd.DataFrame(rows)


def explore_frame(mapper: str) -> pd.DataFrame:
    from .optimize import read_explore
    rows = read_explore(mapper)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("time")
    df["order"] = df.groupby("method").cumcount()
    df["method"] = "own:" + df["method"]
    df["site_scores"] = [_site_scores(v) for v in df["site_scores"]]
    df["params"] = [params.convert_legacy(p) for p in df["params"]]
    return df[["method", "order", "time", "value", "holdout", "all", "params", "site_scores", "tag"]]


def optimization_frame(mapper: str) -> pd.DataFrame:
    """Every global evaluation: optuna trials and own searches, with value (calibration), holdout and all."""
    frames = [f for f in (optuna_frame(mapper), explore_frame(mapper)) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["method", "order", "time", "value", "holdout", "all", "params", "site_scores",
                                     "tag"])
    df = pd.concat(frames, ignore_index=True)
    df["plausible"] = [params.is_plausible(p) for p in df["params"]]
    return df


def convergence(df: pd.DataFrame) -> pd.DataFrame:
    """Per method: evaluations, the best value and where it came, its hold-out, and the best after n evaluations."""
    out = []
    for m, g in df.groupby("method"):
        g = g.sort_values("order")
        best = g["value"].cummax().to_numpy()
        i = int(g["value"].to_numpy().argmax())
        row = {"method": m, "n": len(g), "best": best[-1], "at": i, "holdout": g["holdout"].iloc[i]}
        for n in (50, 100, 200, 500, 1000, 2000):
            row[f"best@{n}"] = best[n - 1] if n <= len(g) else np.nan
        out.append(row)
    return pd.DataFrame(out).sort_values("best", ascending=False)


def unit_matrix(df: pd.DataFrame, mapper: str) -> tuple[np.ndarray, list[str]]:
    """Each evaluation's parameters on the unit cube. A conditional parameter whose switch is off is put mid-range:
    it carries no information."""
    names = params.space_names(mapper)
    X = np.array([params.to_unit(params.complete(p, mapper), mapper, names) for p in df["params"]])
    for j, n in enumerate(names):
        spec = params.space(mapper)[n]
        if spec.when:
            off = np.array([not params.switch_on(p, spec.when, mapper) for p in df["params"]])
            X[off, j] = 0.5
    return X, names


def importance(df: pd.DataFrame, mapper: str) -> tuple[pd.Series, float]:
    """Permutation importance of each parameter in an extra-trees surrogate of the calibration value, and the
    surrogate's in-sample R²."""
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.inspection import permutation_importance
    X, names = unit_matrix(df, mapper)
    y = df["value"].to_numpy()
    model = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=3, random_state=0, n_jobs=4).fit(X, y)
    pi = permutation_importance(model, X, y, n_repeats=5, random_state=0, n_jobs=4)
    return pd.Series(pi.importances_mean, index=names).sort_values(ascending=False), model.score(X, y)


def best_plausible(df: pd.DataFrame) -> tuple[pd.Series, float]:
    """The best evaluation whose n lie in the plausible bands, and its gap to the best overall."""
    ok = df[df["plausible"]]
    if ok.empty:
        raise SystemExit("No evaluated set lies inside the plausible n bands")
    b = ok.loc[ok["value"].idxmax()]
    return b, float(df["value"].max() - b["value"])


# --------------------------------------------------------------------------------------------------------------------
# Per-site tuning
# --------------------------------------------------------------------------------------------------------------------

def persite_runs(mapper: str) -> list[str]:
    prefix = f"persite_{mapper}_"
    return sorted(p.name[len(prefix):-len(".jsonl")] for p in config.RESULTS_ROOT.glob(f"{prefix}*.jsonl"))


def persite_frame(mapper: str, run: str) -> pd.DataFrame:
    from .optimize import persite_path
    path = persite_path(mapper, run)
    if not path.exists():
        raise SystemExit(f"No per-site run {run!r} ({path})")
    return pd.DataFrame([json.loads(line) for line in open(path) if line.strip()])


def persite_summary(mapper: str, run: str) -> pd.DataFrame:
    """Per site: the general set's score (the first, identity evaluation), the best score, the gain, and the best
    reduced parameters."""
    df = persite_frame(mapper, run)
    out = []
    for site, g in df.groupby("site"):
        g = g.sort_values("time")
        b = g.loc[g["value"].idxmax()]
        out.append({"site": site, "general": g["value"].iloc[0], "best": b["value"],
                    "gain": b["value"] - g["value"].iloc[0], "evaluations": len(g), **b["r"]})
    return pd.DataFrame(out).sort_values("gain", ascending=False)
