# site_evaluation

Run, score and calibrate nencarta (ARC + curve2flood) flood maps on the USGS FIM benchmark sites, and look at the
results. This package replaces the `~/scripts/fim3*.py` scripts and the scoring code they imported from
`optimize_fimsites.py`. On the same code it gives the same scores at every site and stage.

## Install

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate flood
pip install --no-deps -e ~/site_evaluation
```

nencarta, arc and curve2flood must already be installed from their repos (editable), and GDAL from conda.
`--no-deps` leaves the conda environment as it is.

## The settings file

[`settings.yaml`](settings.yaml) lists every argument a run sets in nencarta, ARC, curve2flood and the FLDPLN
library, one line each, with its default and whether the optimizers search it. Edit the file; no Python changes
are needed. Its header documents the format. In short:

```yaml
Low_Spot_Range: {value: 0, optimize: false, range: [0, 5]}           # fixed at 0; flip optimize to search 0-5
Bathy_Trap_H:   {value: {fldpln: 0.0, kernel: 0.44}, optimize: true, range: [0.0, 0.5]}   # default per mapper
spill_decay:    {value: {usgs: 0.003, other: 0.01}, mapper: fldpln}   # default per DEM; FLDPLN only
Stream_Slope_Method: {value: "local_average_corrected", optimize: false, screen: false, choices: [local_average, ...]}
Valley_Dir_Widths:   {value: 16.0, optimize: false, range: [4, 40], when: Valley_Sections}   # only while that's on
```

`screen: false` keeps an argument out of `site-eval sensitivity`: it is fixed beforehand, for example on physical
grounds.

| Section | Holds |
|---|---|
| `dem` | Which DEM (`fabdem`, `usgs`, `copdem`, ... each site's `DEMs/<name>.tif`), and the cell size the metre stand-ins convert with: the DEM's nominal size (30 m, lidar 10 m) or each site's measured one |
| `mannings_n` | n of each ESA WorldCover class, with the plausible bands |
| `derived` | Stand-ins searched in place of ARC arguments: rotation (`rotate`, `rot_half_width`, `rot_steps`), depth-varying n (`dvn`, `dvn_shallow`, `dvn_L`), the depth power law (`power_depth`), and `Gen_Slope_Dist_m` / `Gen_Dir_Dist_m` in metres, which become ARC's cells for the DEM in use |
| `nencarta` | Every nencarta option site-eval doesn't set itself. These shape the cached preprocessing, so they can't be optimized |
| `arc`, `curve2flood`, `fldpln_library` | Every argument of each, apart from the files and fields nencarta sets (listed in the comments). ARC lines set by a stand-in say `from: <stand-in>` |
| `persite` | What per-site tuning searches: multipliers of groups of n, and other parameters |

- **When you change something under `nencarta` or `dem`**, rebuild the preprocessing with `site-eval prepare
  --mapper M`. Runs refuse sites whose cache was built with other nencarta settings.
- **Each DEM gets its own cache:** `~/fim_sites_nencarta_v3/<mapper>` for FABDEM, `<mapper>_<dem>` otherwise.
- **Mistakes stop the run before anything starts,** with a message that names the line. Examples: a misspelt field,
  optimize without a range, a `when` naming no setting, or a stand-in searched while its ARC argument is set
  directly.
- **Another settings file:** `site-eval --settings my_study.yaml ...` uses it for any command (or set
  `SITE_EVAL_SETTINGS`). Each run's settings are copied to `results/settings/<mapper>_<fingerprint>.yaml`, and the
  fingerprint is stored with every evaluation.
- **See what a file gives:** `site-eval show settings [--section arc]` lists every argument, `show space` the
  searched ones, and `show params [SET] --site NAME` the exact arguments nencarta, ARC and curve2flood get at a site.

## What it does

### 1. Run and score

```bash
site-eval prepare --mapper fldpln            # once, and after changing nencarta, a site, or the nencarta/dem settings
site-eval run --mapper fldpln                # the settings file's defaults, all sites, all stages
site-eval run --mapper kernel --params '{"n_water": 0.05}' --sites Wabash Salamonie --keep --tag my_test
site-eval run --mapper fldpln --params '{"Standard_Step": false}'      # any argument, searched or not
```

- `--params` overrides the settings file's values. It takes a JSON string or file that names arguments as the file
  does, or a preset (`G_plaus_fldpln`, `G_plaus_kernel`: the file's defaults; `reference`: fim3's October 22 set).
- Sets in the older names (fim3, `results_v3`) are converted on reading:
  - `min_flood_depth`, `TW_limit` and `subcell` become `FLDPLN_Min_Flood_Depth`, `TopWidthPlausibleLimit` and
    `Subcell_Channels`;
  - `Gen_*_Dist` in FABDEM cells becomes `Gen_*_Dist_m` at 30 m a cell;
  - `arc_overrides` / `c2f_overrides` become ordinary entries.
- `--sites` takes name fragments, or `calib`, `holdout` or `all`.
- `--stages` is `all` (the default) or k stages per site, evenly spread.
- `--keep` keeps the flood maps for `plot site`.

A site's score is its mean MCC over its stages. A run reports the mean over the calibration sites (the objective),
over the 10 hold-out sites, and over all sites. Every evaluation is appended to `results/<mapper>_evals.jsonl`
(parameters, each stage's MCC, CSI, bias, recall and precision, the code commits and a fingerprint of the fixed
settings), so tables and figures never need a rerun.

### 2. Optimize

```bash
site-eval optimize optuna   --mapper fldpln --name tpe_1 --trials 1000 --jobs 3      # resumes by name
site-eval optimize sobol    --mapper fldpln --n 1024
site-eval optimize cma      --mapper fldpln --budget 600 --fixed '{"dvn": false}'
site-eval optimize coord    --mapper fldpln --x0 G_plaus_fldpln
site-eval optimize profile  --mapper fldpln --x0 G_plaus_fldpln --only n_water min_flood_depth
site-eval optimize switches --mapper fldpln --bases G_plaus_fldpln
site-eval optimize persite  --mapper fldpln --run r1 --base G_plaus_fldpln     # each site on its own
```

What they search, over what, and the defaults all come from the settings file (`site-eval show space`). The global
searches optimize the calibration mean and score the hold-out sites on every evaluation. They save their results as follows:

| Search | Saved to |
|---|---|
| optuna | `results/optuna_<mapper>.db` |
| sobol, cma, coord, profile, switches | `results/explore_<mapper>.jsonl` |
| persite | `results/persite_<mapper>_<run>.jsonl` |

### 3. Choose what to calibrate

```bash
site-eval sensitivity --mapper fldpln --dry-run      # the candidates and the number of runs
site-eval sensitivity --mapper fldpln                # screening, then the collinearity test (1-2 h on 30 workers)
site-eval sensitivity --mapper fldpln -r 40          # the same study with more trajectories (reuses the first 20)
site-eval sensitivity --mapper fldpln --stage identify --prefer n_tree n_water   # rerun only the second step
site-eval plot sensitivity --mapper fldpln
```

The command answers two questions for one DEM and the calibration sites. It never looks at the hold-out sites.

**1. Does the parameter matter?** This is a screening by elementary effects (Morris 1991, with μ* from Campolongo et
al. 2007).
- The candidates are every argument in the settings file with an `optimize` field, true or false. Arguments marked
  `screen: false` are fixed beforehand and left out:
  - Standard_Step, Stream_Slope_Method, Valley_Sections and Bathy_Use_Banks (Ricky's physical choices);
  - slope_adjustment_factor, a structural duplicate: it multiplies every conveyance, so it gives exactly the maps of
    dividing every n by it (max |dMCC| 0.0 at six sites).
- With 20 trajectories, the 50 FLDPLN candidates take 1020 runs.
- Each step of a trajectory moves one parameter by 2/3 of its range (a switch is flipped). It records the change in
  the calibration sites' mean MCC.
- For each parameter:
  - μ\* is the mean |change|;
  - μ is the mean signed change;
  - σ is the spread;
  - the bootstrap 95% interval of μ\* gives the class: **influential** (the interval lies above 0.01 MCC),
    **negligible** (below it: fix the parameter at its default) or **undecided** (it straddles 0.01: run more
    trajectories).
- *site μ\** is the mean of each site's own μ\*. When it is far above μ\*, the parameter moves each site, but in
  different directions.

**2. Can the influential ones be calibrated together?** This uses the collinearity index of Brun et al. (2001).
- At the defaults, each influential number is moved ±10% of its range. The change in every site's MCC at every stage
  is its sensitivity vector.
- Parameters whose vectors line up compensate each other. For example, every n higher and every slope steeper give
  the same maps.
- γ = 1/√λ_min of the unit vectors' Gram matrix. γ = 1 means the effects are independent. Above γ_crit, the data
  can't fix all of a set's values.
- **The threshold is 5, not Brun's 10–15, because of a positive control.** Every test also moves the slope factor and
  each n class, and reports how collinear the method finds that exact duplicate on these sites. On six test sites it
  scored γ 5.5 (R² 0.97), not infinity: MCC changes in steps as cells flip, which hides collinearity. A set near the
  control's γ is as confounded as an exact duplicate. The same test caught `coefficient_width` and
  `exponent_width`, the width power law's two parameters (R² 0.98).
- A parameter whose ±step changes no site-stage MCC by 0.01 is "too weak here" and isn't judged.
- The report gives:
  - each combination with γ ≥ 10, with signs (which parameters compensate, and how);
  - the identifiable subset, built from the most important parameter down (`--prefer` puts physically preferred
    ones first);
  - for each parameter, the share of its effect the others can reproduce (R² rest), and for each one left out,
    which kept ones explain it.
- This step is local: it holds near the reference set.

Influential switches and choices are settled by comparing their settings (`optimize switches`). Runs are kept in
`results/sensitivity/<mapper>_<name>.jsonl`, not in the archive; tables are written to `_screening.csv`,
`_identifiability.csv` and `_collinearity.csv` beside it. The module's docstring
(`src/site_evaluation/sensitivity.py`) gives the method and its references in full:

- Morris (1991), *Technometrics* 33:161–174.
- Campolongo, Cariboni & Saltelli (2007), *Environmental Modelling & Software* 22:1509–1518.
- Brun, Reichert & Künsch (2001), *Water Resources Research* 37:1015–1030.
- Herman et al. (2013), *HESS* 17:2893–2903: Morris ranks parameters like Sobol' at a fraction of the cost.
- Pianosi et al. (2016), *Environmental Modelling & Software* 79:214–232: the screening → identifiability → calibration workflow.

### 4. Display

```bash
site-eval show runs                      # every run: calib / hold-out / all, failures, settings fingerprint
site-eval show run my_test               # per site: score, bias, recall, precision, errors
site-eval show sites                     # site statistics (stages, drainage area, flows, domain, reference extent)
site-eval show site Salamonie            # one site: statistics, q_stage groups, its latest evaluations
site-eval show studies                   # optimization progress per study / search
site-eval show top --plausible           # best sets, optionally only those with plausible n
site-eval show importance                # parameter importance (surrogate model)
site-eval show plausible --save best.json
site-eval show persite r1
site-eval show params G_plaus_kernel     # a set and the nencarta arguments it becomes

site-eval plot run my_test               # site scores
site-eval plot compare tag_a tag_b       # per-site change between two runs (--other-mapper for a mapper comparison)
site-eval plot stages my_test            # MCC and bias against stage rank
site-eval plot sites [my_test]           # site statistics, or a run's scores against them
site-eval plot site Wabash_River_at_Lafayette --tag my_test   # maps: hits, false alarms, misses per stage
site-eval plot convergence | landscape   # optimization progress; value against every parameter
site-eval plot persite r1
site-eval plot sensitivity               # μ* with intervals; cosines of the sensitivity vectors
```

Figures go to `results/fig/` (or `--out`). Scores print at 2 decimals, because a change under 0.01 MCC counts as no
change; use `--digits 4` for more.

`site-eval --results DIR ...` points any command at another results folder. For example,
`--results ~/scripts/results_v3` reads the October studies.

## Layout

| Module | Contents |
|---|---|
| `settings.yaml` (package root) | Every model argument, its default and its search range |
| `settings.py` | Reads and checks the settings file; the search space; a parameter set → nencarta's options and ARC / curve2flood / FLDPLN arguments; the stand-ins; fingerprints |
| `config.py` | Paths (each overridable with `SITE_EVAL_SITES`, `SITE_EVAL_CACHE`, `SITE_EVAL_WORK`, `SITE_EVAL_RESULTS`) and the bankfull/baseflow rule |
| `sites.py` | Site list, calibration / hold-out split (seed 12345), stages, flows, reference maps |
| `params.py` | Parameter sets: presets, the conversion of older sets, the unit cube for the samplers, Manning's table |
| `scoring.py` | Map-vs-reference metrics on the reference grid inside the boundary; site score and objective |
| `pipeline.py` | nencarta configuration, the cached preprocessing, one run |
| `runner.py` | Worker pool (one locked copy of the cache per worker), the archive |
| `optimize.py` | optuna, Sobol, CMA-ES, compass, profiles, switches, per-site tuning |
| `sensitivity.py` | Morris screening and the collinearity test (`site-eval sensitivity`) |
| `results.py` | Everything read back as DataFrames |
| `stats.py` | Site statistics; ARC's reach attributes (`site-eval attributes`) |
| `plots.py` | Figures |
| `cli.py` | `site-eval` |

From Python:

```python
from site_evaluation import Runner, params, sites, results
if __name__ == "__main__":                     # the workers start with forkserver
    with Runner("fldpln", workers=30) as r:
        recs = r.evaluate({"n_water": 0.05}, sites.select_sites(["Wabash"]), tag="nb")   # settings.yaml for the rest
```

The `site-eval` command works from any folder. A Python session or script started in `~` does not: there the
`~/curve2flood` folder shadows the installed curve2flood that nencarta imports.

## From the old scripts

| Old | New |
|---|---|
| `fim3_prepare.py --mapper m` | `site-eval prepare --mapper m` |
| `fim3_eval.py --mapper m --params f --stages all` | `site-eval run --mapper m --params f` |
| `fim3_optuna.py` | `site-eval optimize optuna` |
| `fim3_explore.py sobol / cma / coord / profile / switches` | `site-eval optimize sobol / cma / coord / profile / switches` |
| `fim3_persite.py` | `site-eval optimize persite` |
| `fim3_analyze.py convergence / top / importance` | `site-eval show studies / top / importance`, `plot convergence` |
| `fim3_pick_plausible.py` | `site-eval show plausible --save FILE` |
| `fim3_attributes.py` | `site-eval attributes` |
| `fim3_persite_analyze.py` (gains) | `site-eval show persite RUN`, `plot persite RUN` |
| `fim3_fixedflips.py` | `site-eval run --params '{"Standard_Step": false}'` |

Not carried over:
- the six-stage versions of the objective (October comparisons only);
- the bootstrap ("protocol") scoring, which the studies never used;
- the per-site correlation and rule analyses (`fim3_persite_analyze.py`, `fim3_rules.py`).

The old scripts and `results_v3` are unchanged.
