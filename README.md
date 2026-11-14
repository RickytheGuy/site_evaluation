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

## Workflow

A calibration covers one mapper (`fldpln` or `kernel`) on one DEM, in six steps. Repeat them for the other mapper.
The examples use FLDPLN.

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate flood
export SITE_EVAL_RESULTS=~/site_evaluation/results/fabdem_v1   # a new results folder for this calibration
```

Give each calibration its own results folder. The commands in step 6 pool every study in the folder they read.

### 1. Set up the settings

Edit [`settings.yaml`](settings.yaml). [The settings file](#the-settings-file) below explains its format.
- `dem.name`: the DEM.
- Each argument's `value` (its default), and its `range` (its physically plausible values) or `choices`. Step 3's
  collinearity test is done at the defaults, so make them your best current set.
- `screen: false` on anything you fix beforehand on physical grounds, with a comment saying why.

```bash
site-eval show settings --mapper fldpln          # every argument as the file gives it
site-eval prepare --mapper fldpln                # build the cache (again after any change under nencarta or dem)
site-eval run --mapper fldpln --tag defaults     # the score to beat
```

### 2. Run the sensitivity study

```bash
site-eval sensitivity --mapper fldpln --dry-run  # the candidates and how many runs they take
site-eval sensitivity --mapper fldpln            # about 3 h for FLDPLN on 30 workers (10 s a run), then rounds
```

It screens every candidate on the calibration sites, then tests the influential ones for collinearity. A candidate is
anything with a range or choices, and every switch, unless it is marked `screen: false`.

The screening then runs more trajectories on its own for the parameters it couldn't decide. Each round moves only
those, so it costs 20 × (undecided + 1) runs: with 25 undecided, 520 runs (about 1.5 h). Rounds stop when none is
undecided, or each has 60 trajectories (`--max-trajectories`, 0 for none).

A bar shows how many runs are done and the time left. Each run is saved as it finishes, so you can stop the study
and continue later:
- **Ctrl-C** lets the running parameter sets finish (a few seconds) and saves them. A second Ctrl-C stops at once.
- **The same command** picks up where it stopped.

[Long runs](#long-runs-progress-and-stopping) covers the other commands and runs in the background.

### 3. Read the sensitivity results

```bash
site-eval sensitivity --mapper fldpln            # prints the report again (nothing is rerun)
site-eval plot sensitivity --mapper fldpln       # each parameter's μ* and interval; how alike their effects are
```

The report has two tables and ends with a **Suggested** list.
- **Screening**, one row per parameter. μ\* is the mean change in the sites' mean MCC when the parameter moves
  across 2/3 of its range. The mean can hide a parameter that raises some sites and lowers others, so the table also
  gives *site μ\** (a site's mean change in MCC) with its own interval, and *moved*: how many sites a step changes by
  0.01 or more (up/down). The two 95% intervals give the class:
  - **influential**: μ\*'s interval lies above 0.01 MCC. The parameter matters.
  - **undecided**: μ\*'s interval straddles 0.01. The study has already run rounds of more trajectories for it, up
    to `--max-trajectories`. To run another round: `site-eval sensitivity --mapper fldpln --max-trajectories 80`.
  - **trades sites**: μ\*'s interval lies below 0.01 but site μ\*'s lies above it. The mean doesn't move, but sites
    do, both ways: its value picks which sites do well. `fldpln_fabdem_site_effects.csv` gives each site's change.
  - **undecided (sites)**: μ\*'s interval lies below 0.01 and site μ\*'s straddles it. Also gets rounds.
  - **negligible**: both intervals lie below 0.01. Fix the parameter.
- **Identifiability**, for the influential numbers.
  - The *identifiable subset* can be calibrated together (γ under 5).
  - The kept parameters can reproduce the effect of each one left out (*R² rest* near 1). Calibrating both would
    trade one against the other.
  - The report names the combinations that compensate each other.
- If the slope factor and the n values compete, keep the n values. This reruns only the identifiability part:
  `site-eval sensitivity --mapper fldpln --stage identify --prefer n_tree n_water`.

The tables are also saved as `sensitivity/fldpln_fabdem_*.csv` in the results folder.
[Choose what to calibrate](#choose-what-to-calibrate) explains the method and gives its references.

### 4. Update the settings: what to optimize, what to fix

| Sensitivity result | In `settings.yaml` |
|---|---|
| In the identifiable subset | `optimize: true` |
| An influential switch or choice | `optimize: true` (optuna searches on/off and choices), or fix it on physical grounds |
| Left out of the subset (collinear) | `optimize: false`: it stays at its `value` |
| Negligible | `optimize: false` |
| Trades sites | `optimize: false` at a value set on physical grounds (it picks which sites do well), or tie it to what the sites it helps share |
| Still undecided after more trajectories | Your call |

Give the reason in each line's comment, for example `# negligible: μ* 0.00 (screen of 2026-11-20)`. Then check
what will be searched:

```bash
site-eval show space --mapper fldpln             # every searched argument and its range
```

### 5. Optimize

```bash
site-eval optimize optuna --mapper fldpln --name tpe_1 --trials 1000
```

optuna's TPE sampler searches every `optimize: true` argument over its range.
- It maximizes the calibration sites' mean MCC.
- It scores the hold-out sites on every trial, but never optimizes on them.
- Trials go to `optuna_fldpln.db` in the results folder.
- The same `--name` resumes the study.

[Optimize](#optimize) lists the other searches.

### 6. Read the optimization results

```bash
site-eval show studies --mapper fldpln           # progress: the best after 50, 100, ... trials
site-eval plot convergence --mapper fldpln
site-eval show top --mapper fldpln --in-range    # the best sets, with calibration and hold-out scores
site-eval show importance --mapper fldpln        # which parameters the score depends on
site-eval plot landscape --mapper fldpln         # the score against each parameter
site-eval show best --mapper fldpln --save best_fldpln.json
```

Then run the best set and compare it with the defaults, site by site:

```bash
site-eval run --mapper fldpln --params best_fldpln.json --tag best --keep
site-eval show run best --mapper fldpln          # per site: score, bias, recall, precision
site-eval plot compare defaults best --mapper fldpln
site-eval plot site Wabash_River_at_Lafayette --mapper fldpln --tag best   # maps: hits, false alarms, misses
```

- Judge a set by its hold-out mean as well as its calibration mean. A gain under 0.01 MCC is no gain.
- If the convergence plot is still climbing, run more trials: the same `--name` with a larger `--trials`.
- If the landscape is flat for a parameter, you can fix it too.
- To adopt the result, copy its values into `settings.yaml` as the new defaults (`value`).

## The settings file

[`settings.yaml`](settings.yaml) lists every argument a run sets in nencarta, ARC, curve2flood and the FLDPLN
library, one line each, with its default and whether the optimizers search it. Edit the file; no Python changes
are needed. Its header documents the format. In short:

```yaml
Bathy_Bankless_Max_Baseflow: {value: 20.0, optimize: false, range: [0, 100]}   # fixed at 20; flip optimize to search
Bathy_Trap_H:   {value: {fldpln: 0.0, kernel: 0.44}, optimize: true, range: [0.0, 0.5]}   # default per mapper
spill_decay:    {value: {usgs: 0.003, other: 0.01}, mapper: fldpln}   # default per DEM; FLDPLN only
Stream_Slope_Method: {value: "local_average_corrected", optimize: false, choices: [local_average, ...]}
Standard_Step:       {value: true, optimize: false, screen: false}   # fixed, and not screened either
Valley_Dir_Widths:   {value: 16.0, optimize: false, range: [4, 40], when: Valley_Sections}   # only while that's on
fldmn:          {value: 0.27, optimize: true, range: [0.01, 1], log: true, when: {solver: iterative}}   # that solver only
Gen_Dir_Dist_m: {value: 960.0, optimize: true, range: [10, 1200], log: true, step: cell}   # whole DEM cells only
```

`when` says what an argument depends on, as the tool's code does: a switch (`when: Valley_Sections`), or the values
of any setting, nencarta's included (`when: {solver: iterative}`, `when: {Stream_Slope_Method: [reach_average,
local_average_corrected]}`), or the DEM (`when: {solver: iterative, dem.name: usgs}`: lidar only). While it doesn't
hold, the argument isn't passed to its tool, the optimizers don't search it, and the screening doesn't count its
effects (they would be zero). One whose `when` can never hold is left out of the search and the screening: on
FABDEM, one that needs `dem.name: usgs`.

`step` limits a search to whole multiples of the step. `step: cell` uses the DEM's cell size from
`dem.nominal_cell_size_m`, so the steps follow the DEM. The range shrinks to the multiples inside it: on FABDEM,
[10, 1200] m becomes 30–1200 m (1–40 cells); on a 10 m DEM it stays 10–1200 m (1–120 cells). On a log scale the
steps are spread log-uniformly. Every sampler snaps to the steps. optuna can't step on a log scale, so it searches the
whole number of steps instead (`<name>_steps` in its database, on a log scale). The results still hold metres.

`site-eval sensitivity` screens every argument outside nencarta's section that has a range or choices, and every
switch (true/false with an optimize field). `screen: false` keeps one out: it is fixed beforehand, for example on
physical grounds. A default outside its range, or not among its choices, stops every command with the line's name.

| Section | Holds |
|---|---|
| `dem` | Which DEM (`fabdem`, `usgs`, `copdem`, ... each site's `DEMs/<name>.tif`), and the cell size the metre stand-ins convert with: the DEM's nominal size (30 m, lidar 10 m) or each site's measured one |
| `mannings_n` | n of each ESA WorldCover class; each range is its physically plausible range |
| `derived` | Stand-ins searched in place of ARC arguments: rotation (`rotate`, `rot_half_width`, `rot_steps`), depth-varying n (`dvn`, `dvn_shallow`, `dvn_L`), the depth power law (`power_depth`), and `Gen_Slope_Dist_m` / `Gen_Dir_Dist_m` in metres each way, which become ARC's cells for the DEM in use (searched in whole cells) |
| `nencarta` | Every nencarta option site-eval doesn't set itself. These shape the cached preprocessing, so they can't be optimized |
| `arc`, `curve2flood`, `fldpln_library` | Every argument of each, apart from the files and fields nencarta sets (listed in the comments). ARC lines set by a stand-in say `from: <stand-in>` |
| `persite` | What per-site tuning searches: multipliers of groups of n, and other parameters |

- **When you change something under `nencarta` or `dem`**, rebuild the preprocessing with `site-eval prepare
  --mapper M`. Runs refuse sites whose cache was built with other nencarta settings.
- **Each DEM gets its own cache:** `~/fim_sites_nencarta_v3/<mapper>` for FABDEM, `<mapper>_<dem>` otherwise.
- **Mistakes stop the run before anything starts,** with a message that names the line. Examples: a misspelt field,
  optimize without a range, a `when` naming no setting or a value it can't take, or a stand-in searched while its ARC argument is set
  directly.
- **Another settings file:** `site-eval --settings my_study.yaml ...` uses it for any command (or set
  `SITE_EVAL_SETTINGS`). Each run's settings are copied to `results/settings/<mapper>_<fingerprint>.yaml`, and the
  fingerprint is stored with every evaluation.
- **See what a file gives:** `site-eval show settings [--section arc]` lists every argument, `show space` the
  searched ones, and `show params [SET] --site NAME` the exact arguments nencarta, ARC and curve2flood get at a site.

## Commands in detail

### Long runs: progress and stopping

`prepare`, `run`, `attributes`, `sensitivity` and every `optimize` search show their progress.
- **On a terminal:** a bar with the count, the time taken and the time left.
- **Output going to a file:** a plain line instead, at every 5% and at least every 5 minutes.
- The optimizers also print a line at each new best.

Ctrl-C stops a command cleanly:

| Command | The first Ctrl-C | What is kept | To continue |
|---|---|---|---|
| `sensitivity` | Finishes the running parameter sets | Every finished run | The same command |
| `optimize optuna` | Finishes the running trials | Every finished trial | The same command (same `--name`) |
| `optimize sobol / cma / coord / profile / switches` | Finishes the running sets | Every evaluation, in `explore_<mapper>.jsonl` | sobol: the same command skips the points it has; the others start again |
| `optimize persite`, `run`, `attributes` | Stops at once | Every site run that finished, in the archive | Run it again |
| `prepare` | Stops at once | Nothing usable: runs refuse the cache until a prepare finishes | Run it again |

- **A second Ctrl-C** stops at once, abandoning what is running.
- **A run in the background** (`nohup site-eval ... > run.log 2>&1 &`): `kill -INT <pid>` acts like Ctrl-C. A plain
  `kill <pid>` stops it at once; its worker processes exit within seconds.

### Run and score

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

### Optimize

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

### Choose what to calibrate

```bash
site-eval sensitivity --mapper fldpln --dry-run      # the candidates and the number of runs
site-eval sensitivity --mapper fldpln                # screening, then the collinearity test (FLDPLN: ~3 h on 30 workers)
site-eval sensitivity --mapper fldpln --max-trajectories 80   # another round for those still undecided
site-eval sensitivity --mapper fldpln --stage identify --prefer n_tree n_water   # rerun only the second step
site-eval plot sensitivity --mapper fldpln
```

The command answers two questions for one DEM and the calibration sites. It never looks at the hold-out sites.

**1. Does the parameter matter?** This is a screening by elementary effects (Morris 1991, with μ* from Campolongo et
al. 2007).
- The candidates are every argument in the settings file with a range or choices, and every switch, unless it is
  marked `screen: false` (fixed beforehand: Standard_Step and Low_Spot_Range as of 2026-11-12).
- With 20 trajectories, the 49 FLDPLN candidates take 1000 runs (kernel: 44 candidates, 900 runs), before rounds.
- Each step of a trajectory moves one parameter by 2/3 of its range (a switch is flipped). It records the change in
  the calibration sites' mean MCC, and in each site's MCC.
- For each parameter:
  - μ\* is the mean |change in the mean MCC|;
  - μ is the mean signed change;
  - σ is the spread;
  - *site μ\** is a site's mean |change in MCC|, averaged over the sites. When it is far above μ\*, the parameter
    moves sites in different directions, which cancel in the mean;
  - *moved* is the number of sites a step changes by 0.01 or more (up/down).
- The bootstrap 95% intervals (over the trajectories) of μ\* and site μ\* give the class. The mean decides first:
  **influential** (μ\*'s interval above 0.01 MCC) or **undecided** (it straddles 0.01). If the mean doesn't move,
  the sites decide: **trades sites** (site μ\*'s interval above 0.01), **undecided (sites)** (it straddles 0.01) or
  **negligible** (below it: fix the parameter at its default). A step taken while a parameter's `when` doesn't hold
  is left out.
- **Rounds.** The undecided parameters (either kind, or never active) then get 20 more trajectories that move only
  them, and are classified again. This repeats until none is left or each has 60 (`--max-trajectories`). A round's
  trajectory starts at a random point of every parameter, like a full one, and the parameters it doesn't move stay
  there. So its effects are drawn the same way and pool with the first 20, at a fraction of the runs. A parameter
  is only moved again while it's undecided, so a decided one stays decided. Looking again after each round makes a
  wrong call a little likelier than 5% for a parameter right at 0.01, where either call changes little.
- Only influential and undecided numbers go on to the collinearity test. A parameter that trades sites gains the
  mean MCC little whatever its value, so it isn't calibrated with the others. Name it with `--identify` to test it
  anyway.

**2. Can the influential ones be calibrated together?** This uses the collinearity index of Brun et al. (2001).
- At the defaults, each influential number is moved ±10% of its range (at least one whole step for a whole number or
  a number with a `step`). The change in every site's MCC at every stage is its sensitivity vector.
- Parameters whose vectors line up compensate each other. For example, every n higher and every slope steeper give
  the same maps.
- γ = 1/√λ_min of the unit vectors' Gram matrix. γ = 1 means the effects are independent. Above γ_crit, the data
  can't fix all of a set's values.
- **The threshold is 5, not Brun's 10–15, because of a positive control.** `slope_adjustment_factor` multiplies every
  conveyance, so it gives exactly the maps of dividing every n by it (max |dMCC| 0.0 at six sites). Every test also
  moves it and each n class, and reports how collinear the method finds that exact duplicate on these sites. On six test sites it
  scored γ 5.5 (R² 0.97), not infinity: MCC changes in steps as cells flip, which hides collinearity. A set near the
  control's γ is as confounded as an exact duplicate. The same test caught `coefficient_width` and
  `exponent_width`, the width power law's two parameters (R² 0.98).
- A parameter whose ±step changes no site-stage MCC by 0.01 is "too weak here" and isn't judged.
- The slope factor is screened too, so if it is influential it meets the n values in this test, and the more important
  of the two sides is kept. `--prefer n_tree n_water` keeps the n values.
- The report gives:
  - each combination with γ above γ_crit, with signs (which parameters compensate, and how);
  - the identifiable subset, built from the most important parameter down (`--prefer` puts physically preferred
    ones first);
  - for each parameter, the share of its effect the others can reproduce (R² rest), and for each one left out,
    which kept ones explain it.
- This step is local: it holds near the reference set.

An influential switch or choice is either searched with the numbers (optuna handles on/off and choices) or fixed on
physical grounds. `optimize switches` compares each searched switch flipped alone around a set. Runs are kept in
`results/sensitivity/<mapper>_<name>.jsonl`, not in the archive; tables are written to `_screening.csv`,
`_site_effects.csv`, `_identifiability.csv` and `_collinearity.csv` beside it. The module's docstring
(`src/site_evaluation/sensitivity.py`) gives the method and its references in full:

- Morris (1991), *Technometrics* 33:161–174.
- Campolongo, Cariboni & Saltelli (2007), *Environmental Modelling & Software* 22:1509–1518.
- Brun, Reichert & Künsch (2001), *Water Resources Research* 37:1015–1030.
- Herman et al. (2013), *HESS* 17:2893–2903: Morris ranks parameters like Sobol' at a fraction of the cost.
- Pianosi et al. (2016), *Environmental Modelling & Software* 79:214–232: the screening → identifiability → calibration workflow.

### Display

```bash
site-eval show runs                      # every run: calib / hold-out / all, failures, settings fingerprint
site-eval show run my_test               # per site: score, bias, recall, precision, errors
site-eval show sites                     # site statistics (stages, drainage area, flows, domain, reference extent)
site-eval show site Salamonie            # one site: statistics, q_stage groups, its latest evaluations
site-eval show studies                   # optimization progress per study / search
site-eval show top --in-range           # best sets, optionally only those inside the settings file's ranges
site-eval show importance                # parameter importance (surrogate model)
site-eval show best --save best.json     # the best set inside the ranges
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
| `fim3_pick_plausible.py` | `site-eval show best --save FILE` |
| `fim3_attributes.py` | `site-eval attributes` |
| `fim3_persite_analyze.py` (gains) | `site-eval show persite RUN`, `plot persite RUN` |
| `fim3_fixedflips.py` | `site-eval run --params '{"Standard_Step": false}'` |

Not carried over:
- the six-stage versions of the objective (October comparisons only);
- the bootstrap ("protocol") scoring, which the studies never used;
- the per-site correlation and rule analyses (`fim3_persite_analyze.py`, `fim3_rules.py`).

The old scripts and `results_v3` are unchanged.
