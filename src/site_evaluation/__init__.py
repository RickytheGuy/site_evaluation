"""
site_evaluation: run, score and calibrate nencarta (ARC + curve2flood) flood maps on the USGS FIM benchmark sites.

    from site_evaluation import Runner, sites, params, results
    if __name__ == "__main__":
        with Runner("fldpln", workers=30) as r:
            recs = r.evaluate({"n_water": 0.05}, sites.select_sites(["Wabash"]), tag="test")

Every model argument, its default and its search range are in the settings file (settings.yaml; settings.use(path)
for another). The command line (``site-eval --help``) covers the same ground: prepare, run, optimize, show and plot.
"""
from . import config, params, scoring, settings, sites
from .runner import Runner

__all__ = ["config", "params", "scoring", "settings", "sites", "Runner"]
__version__ = "0.1.0"
