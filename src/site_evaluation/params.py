"""
Parameter sets: flat dicts naming arguments as the settings file does (settings.yaml), the values a run uses in place
of the file's defaults. Anything a set leaves out takes the file's value.

On the command line a set is a JSON file, a JSON string, or a preset name: G_plaus_fldpln and G_plaus_kernel (the
general sets of 2026-10-29, which are also the file's defaults) or reference (fim3's rounded set of 2026-10-22).

Sets from before the settings file (fim3, results_v3, the presets) are converted on reading: min_flood_depth,
TW_limit and subcell become FLDPLN_Min_Flood_Depth, TopWidthPlausibleLimit and Subcell_Channels; Gen_Slope_Dist and
Gen_Dir_Dist, which were cells of FABDEM, become Gen_Slope_Dist_m and Gen_Dir_Dist_m at 30 m a cell; arc_overrides
and c2f_overrides become entries of the set.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import settings

PRESETS = Path(__file__).resolve().parent / "presets"

#: fim3's rounded set of 2026-10-22 (FLDPLN 0.5938 on that day's code), in fim3's names
REFERENCE = {
    "n_tree": 0.15, "n_grass": 0.08, "n_crop": 0.25, "n_urban": 0.014, "n_water": 0.11,
    "rotate": True, "rot_half_width": 10.0, "rot_steps": 1,
    "Gen_Slope_Dist": 2, "Gen_Dir_Dist": 1, "Bathy_Trap_H": 0.48,
    "dvn": False, "power_depth": False, "subcell": False,
    "min_flood_depth": 0.47, "fldmn": 0.06,
    "TW_MultFact": 1.0, "TW_limit": 4168.0,
    "dvn_shallow": 2.0, "dvn_L": 0.2,
}
LEGACY_NAMES = {"min_flood_depth": "FLDPLN_Min_Flood_Depth", "TW_limit": "TopWidthPlausibleLimit",
                "subcell": "Subcell_Channels"}
LEGACY_CELL_M = 30.0        # the sets before the settings file were all on FABDEM


def convert_legacy(p: dict) -> dict:
    """A set in the names before the settings file, in today's (a set already in today's names is unchanged)."""
    q = dict(p)
    for old, new in LEGACY_NAMES.items():
        if old in q:
            q.setdefault(new, q.pop(old))
            q.pop(old, None)
    S = settings.current()
    for cells in ("Gen_Slope_Dist", "Gen_Dir_Dist"):
        e = S.entries.get(cells)
        if cells in q and f"{cells}_m" not in q and e is not None and e.is_from:
            q[f"{cells}_m"] = float(q.pop(cells)) * LEGACY_CELL_M
    for overrides in ("arc_overrides", "c2f_overrides"):
        for k, v in (q.pop(overrides, None) or {}).items():
            q[k] = v
    return q


def space(mapper: str) -> dict[str, settings.Spec]:
    return settings.current().space(mapper)


def space_names(mapper: str) -> list[str]:
    return list(space(mapper))


def default_set(mapper: str) -> dict:
    """The settings file's default for every searched parameter."""
    return settings.current().defaults(mapper)


def complete(p: dict, mapper: str) -> dict:
    """p with the file's default for every searched parameter it leaves out."""
    return {**default_set(mapper), **convert_legacy(p)}


def searched(p: dict, mapper: str) -> dict:
    """The entries of p that are searched for this mapper."""
    sp = space(mapper)
    return {k: v for k, v in p.items() if k in sp}


def load(spec: str | dict | None, mapper: str) -> dict:
    """A parameter set from a preset name, a JSON file or a JSON string (None or "": the file's defaults)."""
    if isinstance(spec, dict):
        return convert_legacy(spec)
    if not spec:
        return default_set(mapper)
    if spec == "reference":
        return convert_legacy(REFERENCE)
    preset = PRESETS / f"{spec}.json"
    if preset.exists():
        return convert_legacy(json.loads(preset.read_text()))
    path = Path(spec).expanduser()
    if path.exists():
        return convert_legacy(json.loads(path.read_text()))
    try:
        return convert_legacy(json.loads(spec))
    except json.JSONDecodeError:
        raise SystemExit(f"--params {spec!r} is not a preset ({', '.join(presets())}), a file or JSON") from None


def presets() -> list[str]:
    return sorted(f.stem for f in PRESETS.glob("*.json")) + ["reference"]


def switch_on(p: dict, name: str, mapper: str) -> bool:
    """Whether a switch is on in a parameter set, or in the settings file when the set leaves it out."""
    return bool(p[name] if name in p else settings.current().default(name, mapper))


def is_plausible(p: dict) -> bool:
    """Whether every value with a plausible band in the settings file lies inside it."""
    return all(lo - 1e-9 <= p[k] <= hi + 1e-9 for k, (lo, hi) in settings.current().plausible().items() if k in p)


# --------------------------------------------------------------------------------------------------------------------
# The unit cube, for the samplers
# --------------------------------------------------------------------------------------------------------------------

def from_unit(u, mapper: str, names: list[str] | None = None, sp: dict | None = None) -> dict:
    """A parameter set from a point of the unit cube, one coordinate per name (in order). A switch is on above 0.5,
    a choice takes the bin its coordinate falls in; conditional parameters are kept even when their switch is off
    (they are then unused). sp: the specs to use (default: the mapper's search space)."""
    sp = sp or space(mapper)
    names = names or list(sp)
    p = {}
    for x, name in zip(u, names):
        s = sp[name]
        x = float(np.clip(x, 0.0, 1.0))
        if s.kind == "bool":
            p[name] = bool(x >= 0.5)
        elif s.kind == "choice":
            p[name] = s.choices[min(int(x * len(s.choices)), len(s.choices) - 1)]
        else:
            lo, hi = s.low, s.high
            v = np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo))) if s.log else lo + x * (hi - lo)
            p[name] = int(np.clip(round(v), lo, hi)) if s.kind == "int" else float(v)
    return p


def to_unit(p: dict, mapper: str, names: list[str] | None = None, sp: dict | None = None) -> np.ndarray:
    sp = sp or space(mapper)
    names = names or list(sp)
    out = []
    for name in names:
        s = sp[name]
        v = p.get(name)
        if s.kind == "bool":
            out.append(0.75 if v else 0.25)
        elif s.kind == "choice":
            i = s.choices.index(v) if v in s.choices else 0
            out.append((i + 0.5) / len(s.choices))
        elif v is None:
            out.append(0.5)
        else:
            lo, hi = s.low, s.high
            out.append((np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)) if s.log else (v - lo) / (hi - lo))
    return np.clip(np.array(out, float), 0, 1)


# --------------------------------------------------------------------------------------------------------------------
# A parameter set -> nencarta
# --------------------------------------------------------------------------------------------------------------------

def model_args(p: dict, mapper: str, site: str | None = None) -> tuple[dict, dict, dict, dict]:
    """nencarta's options, arc_args, curve2flood_args and fldpln_library_args for a parameter set (site: for the
    measured cell size)."""
    return settings.current().model_args(convert_legacy(p), mapper, site)


def write_n(p: dict, file: str | Path, mapper: str) -> None:
    """The Manning's n table nencarta reads (tab-separated, by ESA WorldCover class)."""
    rows = settings.current().mannings_rows(convert_legacy(p), mapper)
    with open(file, "w") as f:
        f.write("LC_ID\tDescription\tManning_n\n")
        f.writelines(f"{i}\t{d}\t{v:.5f}\n" for i, d, v in rows)


def describe(p: dict, mapper: str) -> str:
    """A one-line view of a set: its searched parameters (leaving out those whose switch is off), then any other
    entries it sets."""
    sp = space(mapper)
    out = []
    for k, v in p.items():
        s = sp.get(k)
        if s is not None and s.when and not switch_on(p, s.when, mapper):
            continue
        out.append(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}")
    return " ".join(out)
