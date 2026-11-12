"""
Figures, written as PNG files (results/fig/ by default). Each function takes data from results.py or stats.py and
returns the path it wrote.

Colours: the dataviz reference palette. Categorical slots are used in fixed order (blue, orange, aqua, yellow, ...);
the first three are safe for any pair under colour-vision deficiency, so the three-class maps use only those.
Magnitude uses one blue ramp, and signed differences use blue against red around a grey midpoint.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config, params, scoring, sites

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")       # better (blue) / no change / worse (red)
SPLIT_COLOR = {"calib": SERIES[0], "hold": SERIES[1]}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
        "lines.linewidth": 2, "figure.dpi": 110,
    })
    return plt


def _out(name: str, out: str | Path | None) -> Path:
    path = Path(out) if out else config.results_path("fig", f"{name}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save(fig, path: Path) -> Path:
    fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def _legend_split(ax, loc="lower right"):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SPLIT_COLOR["calib"], label="calibration"),
                       Patch(color=SPLIT_COLOR["hold"], label="hold-out")], loc=loc)


# --------------------------------------------------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------------------------------------------------

def run_sites(table: pd.DataFrame, title: str, out=None) -> Path:
    """One run: each site's score, best first, coloured by split, with the calibration and hold-out means."""
    plt = _plt()
    t = table.sort_values("score")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.16 * len(t) + 1)))
    y = np.arange(len(t))
    ax.barh(y, t["score"], height=0.72, color=[SPLIT_COLOR[s] for s in t["split"]])
    ax.set_yticks(y, [sites.short_name(s) for s in t["site"]], fontsize=7)
    ax.set_xlim(min(0, t["score"].min() - 0.02), 1)
    ax.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = []
    for split, label, ls in (("calib", "calibration", "-"), ("hold", "hold-out", "--")):
        m = t.loc[t["split"] == split, "score"].mean()
        handles.append(Patch(color=SPLIT_COLOR[split], label=label))
        if np.isfinite(m):
            ax.axvline(m, color=INK2, lw=1, ls=ls)
            handles.append(Line2D([], [], color=INK2, lw=1, ls=ls, label=f"{label} mean {m:.2f}"))
    ax.set_xlabel("site score (mean MCC over its stages)")
    ax.set_title(title)
    ax.legend(handles=handles, loc="lower right")
    return _save(fig, _out("run_sites", out))


def run_compare(a: pd.DataFrame, b: pd.DataFrame, label_a: str, label_b: str, out=None) -> Path:
    """Two runs: each site's score difference (b - a), largest gain first. Changes under 0.01 are grey."""
    plt = _plt()
    d = a.set_index("site")[["score", "split"]].join(b.set_index("site")[["score"]], rsuffix="_b", how="inner")
    d["diff"] = d["score_b"] - d["score"]
    d = d.sort_values("diff")
    colors = [DIVERGING[0] if v >= 0.01 else DIVERGING[2] if v <= -0.01 else AXIS for v in d["diff"]]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.16 * len(d) + 1)))
    y = np.arange(len(d))
    ax.barh(y, d["diff"], height=0.72, color=colors)
    ax.set_yticks(y, [sites.short_name(s) + (" (hold)" if sp == "hold" else "") for s, sp in zip(d.index, d["split"])],
                  fontsize=7)
    ax.axvline(0, color=AXIS, lw=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"site score change: {label_b} − {label_a}")
    up, down = int((d["diff"] >= 0.01).sum()), int((d["diff"] <= -0.01).sum())
    ax.set_title(f"{label_b} vs {label_a}: mean {d['diff'].mean():+.3f}; {up} sites up ≥ 0.01, {down} down")
    return _save(fig, _out("run_compare", out))


def run_stages(stages: pd.DataFrame, title: str, out=None) -> Path:
    """One run: MCC and bias against each stage's rank within its site (0 lowest, 1 highest); thin lines are sites,
    the thick line the median in five rank bins."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0, 1, 6)
    for ax, metric, ylab in ((axes[0], "mcc", "MCC"), (axes[1], "bias", "bias (modelled / reference wet area)")):
        for _, g in stages.groupby("site"):
            ax.plot(g["rank"], g[metric], color=AXIS, lw=0.6)
        mid = (bins[:-1] + bins[1:]) / 2
        idx = np.clip(np.digitize(stages["rank"], bins) - 1, 0, 4)
        med = [stages.loc[idx == k, metric].median() for k in range(5)]
        ax.plot(mid, med, color=SERIES[0], lw=2.5, marker="o", ms=6, label="median")
        ax.set_xlabel("stage rank within site (0 = lowest map)")
        ax.set_ylabel(ylab)
        if metric == "bias":
            ax.set_yscale("log")
            ax.axhline(1, color=INK2, lw=1, ls="--")
        else:
            ax.set_ylim(0, 1)
    axes[0].set_title(title)
    axes[1].set_title("bias by stage")
    return _save(fig, _out("run_stages", out))


# --------------------------------------------------------------------------------------------------------------------
# Optimization
# --------------------------------------------------------------------------------------------------------------------

def convergence(df: pd.DataFrame, mapper: str, out=None) -> Path:
    """Best calibration value so far against evaluations, one line per study or search (the 8 best)."""
    plt = _plt()
    best = df.groupby("method")["value"].max().sort_values(ascending=False)
    methods = list(best.index[:8])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, m in enumerate(methods):
        g = df[df["method"] == m].sort_values("order")
        y = g["value"].cummax().to_numpy()
        ax.plot(np.arange(1, len(y) + 1), y, color=SERIES[i], label=f"{m}  ({y[-1]:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("evaluations")
    ax.set_ylabel("best calibration value so far")
    ax.set_title(f"{mapper}: convergence")
    ax.legend(loc="lower right", fontsize=8)
    if len(best) > 8:
        ax.text(1, -0.14, f"{len(best) - 8} more not shown", transform=ax.transAxes, ha="right", color=MUTED)
    return _save(fig, _out(f"convergence_{mapper}", out))


def landscape(df: pd.DataFrame, mapper: str, out=None) -> Path:
    """Every global evaluation's calibration value against each parameter, one panel per parameter. The best set
    is marked; plausible sets (n inside the bands) are blue, the rest grey."""
    plt = _plt()
    from . import settings
    sp = params.space(mapper)
    bands = settings.current().plausible()
    names = list(sp)
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.5 * nrow), sharey=True)
    P = pd.DataFrame([params.complete(p, mapper) for p in df["params"]])
    lo = np.nanpercentile(df["value"], 5)
    b = int(np.argmax(df["value"].to_numpy()))
    plaus = df["plausible"].to_numpy()
    for ax, n in zip(axes.flat, names):
        spec = sp[n]
        if spec.kind == "choice":
            x = np.array([spec.choices.index(v) if v in spec.choices else np.nan for v in P[n]], float)
            x = x + np.random.default_rng(0).uniform(-0.15, 0.15, len(x))
            ax.set_xticks(range(len(spec.choices)), [str(c) for c in spec.choices], fontsize=7, rotation=25,
                          ha="right")
        else:
            x = P[n].astype(float).to_numpy()
        if spec.kind == "bool":
            x = x + np.random.default_rng(0).uniform(-0.15, 0.15, len(x))
            ax.set_xticks([0, 1], ["off", "on"])
        elif spec.kind in ("float", "int") and spec.log:
            from matplotlib.ticker import NullFormatter
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter())
        if spec.kind in ("float", "int"):           # the whole searched range, however few the points
            pad = (spec.high / spec.low) ** 0.03 if spec.log else 0.03 * (spec.high - spec.low)
            ax.set_xlim(*((spec.low / pad, spec.high * pad) if spec.log else (spec.low - pad, spec.high + pad)))
        ax.scatter(x[~plaus], df["value"][~plaus], s=8, color=AXIS, lw=0)
        ax.scatter(x[plaus], df["value"][plaus], s=8, color=SERIES[0], lw=0, alpha=0.6)
        ax.scatter([x[b]], [df["value"].iloc[b]], s=60, color=SERIES[1], edgecolor=SURFACE, lw=2, zorder=5)
        if n in bands:
            ax.axvspan(*bands[n], color=SERIES[0], alpha=0.06, lw=0)
        ax.set_title(n, fontsize=9)
        ax.set_ylim(lo, df["value"].max() + 0.005)
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("calibration value")
    fig.suptitle(f"{mapper}: {len(df)} evaluations; blue = plausible n, orange = best; shaded = plausible band",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return _save(fig, _out(f"landscape_{mapper}", out))


def persite_gains(summary: pd.DataFrame, title: str, out=None) -> Path:
    """Per-site tuning: each site's general score and its best tuned score."""
    plt = _plt()
    t = summary.sort_values("best")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.16 * len(t) + 1)))
    y = np.arange(len(t))
    ax.hlines(y, t["general"], t["best"], color=AXIS, lw=2)
    ax.scatter(t["general"], y, s=30, color=SERIES[0], label="general set", zorder=3)
    ax.scatter(t["best"], y, s=30, color=SERIES[1], label="tuned for the site", zorder=3)
    ax.set_yticks(y, [sites.short_name(s) for s in t["site"]], fontsize=7)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("site score")
    ax.set_title(f"{title}: median gain {t['gain'].median():.3f}")
    ax.legend(loc="lower right")
    return _save(fig, _out("persite", out))


# --------------------------------------------------------------------------------------------------------------------
# Sites
# --------------------------------------------------------------------------------------------------------------------

STAT_PANELS = [("stages", "stages (maps)", False), ("da_km2", "drainage area (km²)", True),
               ("stream_km", "flowing reach length (km)", True), ("domain_km2", "scoring domain (km²)", True),
               ("bankfull_cms", "bankfull flow (m³/s)", True), ("top_over_bankfull", "top stage flow / bankfull", True),
               ("ref_flooded_low", "reference wet share, lowest map", False),
               ("ref_flooded_high", "reference wet share, highest map", False)]


def site_stats(table: pd.DataFrame, scores: pd.DataFrame | None = None, out=None) -> Path:
    """The sites' statistics: a histogram of each, or, given a run's site scores, the score against each."""
    plt = _plt()
    t = table.copy()
    if scores is not None:
        t = t.merge(scores[["site", "score"]], on="site")
    panels = [p for p in STAT_PANELS if p[0] in t]
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6))
    for ax, (col, label, log) in zip(axes.flat, panels):
        x = pd.to_numeric(t[col], errors="coerce")
        ok = np.isfinite(x) & ((x > 0) if log else True)
        if scores is None:
            vals = np.log10(x[ok]) if log else x[ok]
            ax.hist(vals, bins=15, color=SERIES[0], edgecolor=SURFACE, lw=1)
            ax.set_xlabel(("log10 " if log else "") + label)
            ax.set_ylabel("sites")
        else:
            colors = [SPLIT_COLOR["hold" if s == "hold" else "calib"] for s in t.loc[ok, "split"]]
            ax.scatter(x[ok], t.loc[ok, "score"], s=22, color=colors, edgecolor=SURFACE, lw=1)
            if log:
                from matplotlib.ticker import NullFormatter
                ax.set_xscale("log")
                ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel(label)
            ax.set_ylabel("site score")
            from scipy.stats import spearmanr
            rho = spearmanr(x[ok], t.loc[ok, "score"]).statistic
            ax.set_title(f"Spearman ρ {rho:+.2f}", fontsize=8, fontweight="normal", color=INK2)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    head = f"{len(t)} sites" + (" — score against site statistics" if scores is not None else "")
    fig.suptitle(head, x=0.01, ha="left", fontsize=10, fontweight="bold")
    if scores is not None:
        _legend_split(axes.flat[0], loc="best")
    fig.tight_layout()
    return _save(fig, _out("site_stats" + ("_scores" if scores is not None else ""), out))


def _on_grid(path, ref_ds, resample="bilinear") -> np.ndarray:
    """A raster warped onto the reference's grid."""
    from osgeo import gdal
    gt, w, h = ref_ds.GetGeoTransform(), ref_ds.RasterXSize, ref_ds.RasterYSize
    bounds = (gt[0], gt[3] + h * gt[5], gt[0] + w * gt[1], gt[3])
    ds = gdal.Warp("", str(path), options=gdal.WarpOptions(format="MEM", width=w, height=h, outputBounds=bounds,
                                                           dstSRS=ref_ds.GetProjectionRef(), resampleAlg=resample))
    return ds.GetRasterBand(1).ReadAsArray().astype(float)


def _hillshade(z: np.ndarray, dx: float, dy: float) -> np.ndarray:
    z = np.where(np.isfinite(z) & (z > -1000), z, np.nan)
    z = np.where(np.isnan(z), np.nanmedian(z), z)
    gy, gx = np.gradient(z, dy, dx)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(315), np.radians(45)
    return np.clip(np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect), 0, 1)


def site_maps(site: str, stages: list[str] | None = None, maps_dir: Path | None = None, rec: dict | None = None,
              out=None) -> Path:
    """
    A site's maps over a hillshade of its DEM, one panel per stage (up to 6, evenly spread). Without a run, each
    panel shows the reference. With a kept run's maps (maps_dir/<site>), each shows where both are wet (blue), where
    only the model is (orange: false alarm) and where only the reference is (aqua: miss), with the stage's scores.
    """
    plt = _plt()
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from osgeo import gdal
    gdal.UseExceptions()
    all_st = sites.site_stages(site)
    if stages is None:
        idx = np.unique(np.round(np.linspace(0, len(all_st) - 1, min(6, len(all_st)))).astype(int))
        stages = [all_st[i] for i in idx]
    model_maps = scoring.floodmap_files(maps_dir / site) if maps_dir else {}
    bnd = sites.boundary_raster(site)
    ncol = min(3, len(stages))
    nrow = int(np.ceil(len(stages) / ncol))
    ref0 = gdal.Open(sites.reference_map(site, stages[0]))
    gt = ref0.GetGeoTransform()
    lat = gt[3] + gt[5] * ref0.RasterYSize / 2
    geographic = abs(gt[1]) < 1
    dx = abs(gt[1]) * (111320 * np.cos(np.radians(lat)) if geographic else 1)
    dy = abs(gt[5]) * (111320 if geographic else 1)
    shade = _hillshade(_on_grid(sites.dem_file(site), ref0), dx, dy)
    domain = (_on_grid(bnd, ref0, "near") > 0) if bnd else np.ones_like(shade, bool)
    # crop to the domain with a 10% margin
    rows, cols = np.nonzero(domain)
    pad_r, pad_c = max(5, int(0.1 * np.ptp(rows))), max(5, int(0.1 * np.ptp(cols)))
    win = (slice(max(rows.min() - pad_r, 0), rows.max() + pad_r + 1),
           slice(max(cols.min() - pad_c, 0), cols.max() + pad_c + 1))
    shade, domain = shade[win], domain[win]
    aspect = dy / dx
    cmap = ListedColormap([SERIES[0], SERIES[1], SERIES[2]])
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.6 * nrow * min(aspect * shade.shape[0] /
                                                                               shade.shape[1], 1.6)), squeeze=False)
    rec_mcc = dict(zip(rec["stages"], rec["mcc"])) if rec else {}
    rec_bias = dict(zip(rec["stages"], rec["bias"])) if rec else {}
    for ax, stage in zip(axes.flat, stages):
        ax.imshow(shade, cmap="gray", vmin=-0.3, vmax=1.3, aspect=aspect, interpolation="nearest")
        ref_ds = gdal.Open(sites.reference_map(site, stage))
        ref = (ref_ds.GetRasterBand(1).ReadAsArray() > 0)[win]
        fmap = model_maps.get(round(sites.stage_to_float(stage), 6))
        if fmap is not None:
            flood, refb, dom = scoring.load_on_reference_grid(fmap, sites.reference_map(site, stage), bnd)
            cls = np.full(flood.shape, np.nan)
            cls[(flood > 0) & (refb > 0)] = 0
            cls[(flood > 0) & (refb == 0)] = 1
            cls[(flood == 0) & (refb > 0)] = 2
            cls[dom == 0] = np.nan
            cls = cls[win]
            ax.imshow(cls, cmap=cmap, vmin=0, vmax=2, aspect=aspect, interpolation="nearest", alpha=0.85)
            m = scoring.metrics(*scoring.contingency(flood, refb, dom))
            ax.set_title(f"stage {stage}: MCC {m['mcc']:.2f}, bias {m['bias']:.2f}", fontsize=9)
        else:
            ax.imshow(np.where(ref & domain, 0.0, np.nan), cmap=ListedColormap([SERIES[0]]), aspect=aspect,
                      interpolation="nearest", alpha=0.75)
            extra = f" (MCC {rec_mcc[stage]:.2f}, bias {rec_bias[stage]:.2f})" \
                if rec_mcc.get(stage) is not None else ""
            ax.set_title(f"stage {stage}: reference{extra}", fontsize=9)
        ax.contour(domain.astype(float), levels=[0.5], colors=[INK2], linewidths=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    for ax in list(axes.flat)[len(stages):]:
        ax.set_visible(False)
    if model_maps:
        handles = [Patch(color=SERIES[0], label="both wet"), Patch(color=SERIES[1], label="model only (false alarm)"),
                   Patch(color=SERIES[2], label="reference only (miss)")]
    else:
        handles = [Patch(color=SERIES[0], label="reference wet")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(sites.short_name(site, 80) + f" ({sites.site_year(site)}); outline = scoring domain", x=0.01,
                 ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    safe = site.replace("(", "").replace(")", "")
    return _save(fig, _out(f"site_{safe}", out))


# --------------------------------------------------------------------------------------------------------------------
# Sensitivity
# --------------------------------------------------------------------------------------------------------------------

CLASS_COLOR = {"influential": SERIES[0], "undecided": SERIES[1], "negligible": AXIS}


def sensitivity_screening(table: pd.DataFrame, study, threshold: float = 0.01, out=None) -> Path:
    """Morris screening: each parameter's μ* with its bootstrap 95% interval, coloured by class, and site μ* (the mean
    of each site's own μ*) as a hollow ring; the dashed line is the threshold."""
    from matplotlib.lines import Line2D
    from . import sensitivity
    plt = _plt()
    t = table[table["class"] != "never active"].iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.2 * len(t) + 1.2)))
    y = np.arange(len(t))
    colors = [CLASS_COLOR[c] for c in t["class"]]
    ax.hlines(y, t["ci_low"], t["ci_high"], color=colors, lw=2)
    ax.scatter(t["mu_star"], y, s=36, color=colors, zorder=3, edgecolor=SURFACE, linewidth=1)
    ax.scatter(t["site_mu_star"], y, s=30, facecolor="none", edgecolor=INK2, linewidth=1, zorder=2)
    ax.axvline(threshold, color=INK2, lw=1, ls="--")
    ax.set_yticks(y, [f"{p}" + (" ◆" if k.split()[0] not in ("float", "int") else "") for p, k in
                      zip(t["parameter"], t["kind"])], fontsize=7)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(left=0)
    ax.set_xlabel("μ*: mean |change in the sites' mean MCC| when the parameter rises by 2/3 of its range")
    r = sensitivity.complete_trajectories(study)
    ax.set_title(f"{study.mapper} on {study.header['dem']}: Morris screening, {r} trajectories, "
                 f"{len(study.sites)} sites")
    handles = [Line2D([], [], color=CLASS_COLOR[c], marker="o", lw=2, label=c) for c in CLASS_COLOR
               if c in set(t["class"])]
    handles += [Line2D([], [], color=INK2, marker="o", markerfacecolor="none", lw=0, label="site μ*"),
                Line2D([], [], color=INK2, lw=1, ls="--", label=f"threshold {threshold:g}"),
                Line2D([], [], color="none", marker="", label="◆ switch or choice")]
    ax.legend(handles=handles, loc="lower right")
    return _save(fig, _out(f"sensitivity_{study.mapper}_{study.name}", out))


def sensitivity_collinearity(cos: pd.DataFrame, title: str, out=None) -> Path:
    """Cosines between the parameters' local sensitivity vectors: ±1 means one parameter's effect on every site and
    stage can be undone by the other's."""
    from matplotlib.colors import LinearSegmentedColormap
    plt = _plt()
    k = len(cos)
    cmap = LinearSegmentedColormap.from_list("div", [DIVERGING[2], DIVERGING[1], DIVERGING[0]])
    fig, ax = plt.subplots(figsize=(max(4.5, 0.5 * k + 2.5), max(4, 0.5 * k + 1.5)))
    im = ax.imshow(cos.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(k), cos.columns, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(k), cos.index, fontsize=7)
    ax.grid(False)
    for i in range(k):
        for j in range(k):
            v = cos.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color=SURFACE if abs(v) > 0.6 else INK)
    fig.colorbar(im, ax=ax, fraction=0.04, label="cosine of the sensitivity vectors")
    ax.set_title(title)
    name = Path(out).stem if out else "sensitivity_collinearity"
    return _save(fig, _out(name, out))
