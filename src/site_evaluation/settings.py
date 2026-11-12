"""
The settings file (settings.yaml): every argument a run sets in nencarta, ARC, curve2flood and the FLDPLN library,
its default, and whether and over what the optimizers search it. Its header documents the format.

    S = settings.current()                 # the file in use (SITE_EVAL_SETTINGS, --settings, or the package's)
    S.space("fldpln")                      # the search space: {name: Spec}
    S.defaults("fldpln")                   # the default value of every searched parameter
    S.model_args(p, "fldpln", site)        # nencarta's options, arc_args, curve2flood_args, fldpln_library_args

A parameter set names arguments as the file does (an argument whose name appears in two sections is
<section>.<name>). Stand-ins in `derived` set the ARC arguments marked `from`.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from . import config

#: The settings file used when neither SITE_EVAL_SETTINGS nor --settings names another
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "settings.yaml"

SECTIONS = ("mannings_n", "derived", "nencarta", "arc", "curve2flood", "fldpln_library")
#: Where each tool-argument section goes in nencarta's configuration
TOOL_SECTIONS = {"nencarta": None, "arc": "arc_args", "curve2flood": "curve2flood_args",
                 "fldpln_library": "fldpln_library_args"}
ENTRY_FIELDS = {"value", "optimize", "range", "log", "type", "choices", "when", "mapper", "plausible", "from",
                "class", "label", "screen"}
PERSITE_FIELDS = {"scales", "range", "log", "type", "optimize", "mapper"}

#: The stand-ins and the ARC arguments each sets (see derived_args)
DERIVED_GROUPS = {
    "rotate": ("Degree_Manip", "Degree_Interval"),
    "dvn": ("Depth_Varying_N", "k_decay", "shallow_factor", "deep_factor"),
    "power_depth": ("coefficient_depth", "exponent_depth"),
    "Gen_Slope_Dist_m": ("Gen_Slope_Dist",),
    "Gen_Dir_Dist_m": ("Gen_Dir_Dist",),
}
DERIVED_NAMES = ("rotate", "rot_half_width", "rot_steps", "dvn", "dvn_shallow", "dvn_L", "power_depth",
                 "depth_law_coefficient", "depth_law_exponent", "Gen_Slope_Dist_m", "Gen_Dir_Dist_m")
ALL_MAPPERS = (*config.MAPPERS, "multipoint")


class SettingsError(ValueError):
    pass


@dataclass
class Spec:
    """How one parameter is searched."""
    kind: str                       # float, int, bool or choice
    low: float | None = None
    high: float | None = None
    log: bool = False
    choices: tuple = ()
    when: str | None = None         # the switch it depends on

    def as_tuple(self) -> tuple:
        """The old fim3 form: (kind, low, high, log[, when]) or ("bool",)."""
        if self.kind == "bool":
            return ("bool",)
        base = (self.kind, self.low, self.high, self.log)
        return base + ((self.when,) if self.when else ())


@dataclass
class Entry:
    section: str
    name: str                       # as written in the file
    key: str                        # its name in parameter sets
    raw: dict = field(repr=False)

    @property
    def is_from(self) -> bool:
        return "from" in self.raw


@dataclass
class Settings:
    path: Path
    text: str
    dem: str
    cell_size: str
    nominal_cell_size: dict
    entries: dict[str, Entry]       # by key, in file order
    persite_raw: dict

    # ----------------------------------------------------------------------------------------------------------------
    # Values
    # ----------------------------------------------------------------------------------------------------------------

    def resolve(self, value, mapper: str):
        """A value that may be keyed by mapper or DEM ({fldpln: .., kernel: ..}, {usgs: .., other: ..})."""
        while isinstance(value, dict) and value and set(value) <= self._selectors():
            if mapper in value:
                value = value[mapper]
            elif self.dem in value:
                value = value[self.dem]
            elif "other" in value:
                value = value["other"]
            else:
                raise SettingsError(f"no value for mapper {mapper} / DEM {self.dem} in {value}")
        return value

    def _selectors(self) -> set:
        return set(ALL_MAPPERS) | set(self.nominal_cell_size) | {"other"}

    def applies(self, e: Entry, mapper: str) -> bool:
        m = e.raw.get("mapper")
        return m is None or m == mapper

    def default(self, key: str, mapper: str):
        return self.resolve(self.entries[key].raw.get("value"), mapper)

    def optimized(self, e: Entry, mapper: str) -> bool:
        return bool(self.resolve(e.raw.get("optimize", False), mapper)) and self.applies(e, mapper)

    def spec(self, e: Entry, mapper: str) -> Spec:
        raw = e.raw
        when = raw.get("when")
        value = self.resolve(raw.get("value"), mapper)
        if "choices" in raw:
            choices = tuple(raw["choices"])
            if set(choices) == {True, False}:
                return Spec("bool", when=when)
            return Spec("choice", choices=choices, when=when)
        if isinstance(value, bool):
            return Spec("bool", when=when)
        lo, hi = raw["range"]
        is_int = raw.get("type") == "int" or (raw.get("type") is None and isinstance(lo, int)
                                               and isinstance(hi, int) and (value is None or isinstance(value, int)))
        return Spec("int" if is_int else "float", lo, hi, bool(raw.get("log", False)), when=when)

    def space(self, mapper: str) -> dict[str, Spec]:
        """The searched parameters for a mapper, in file order. One that depends on a switch held off (not searched,
        default off) is left out: it would never be used."""
        out = {}
        for k, e in self.entries.items():
            if not self.optimized(e, mapper):
                continue
            when = e.raw.get("when")
            if when and when not in out and not self.default(when, mapper):
                continue
            out[k] = self.spec(e, mapper)
        return out

    def candidates(self, mapper: str, plausible: bool = False, only: list[str] | None = None
                   ) -> tuple[dict[str, Spec], dict[str, str]]:
        """The parameters `site-eval sensitivity` screens, and why each other argument with an optimize field isn't
        screened. A candidate is any argument with an optimize field (true or false) that the mapper uses, outside
        nencarta's section (it shapes the cache), not marked `screen: false`, with a range, choices or a switch's
        true/false. One that depends on a switch which is neither a candidate nor on by default is left out (never
        used). only: these names instead (they may include `screen: false` ones). plausible: search n over its
        plausible band rather than its range."""
        if only:
            bad = [k for k in only if k not in self.entries]
            if bad:
                raise SettingsError(f"not in the settings file: {bad}")
        out, left_out = {}, {}
        for k, e in self.entries.items():
            r = e.raw
            if e.is_from or e.section == "nencarta" or "optimize" not in r:
                if only and k in only:
                    raise SettingsError(f"{k} can't be screened (nencarta's, set by a stand-in, or no optimize field)")
                continue
            if only and k not in only:
                continue
            if not self.applies(e, mapper):
                left_out[k] = f"{r.get('mapper')} only"
                continue
            if r.get("screen") is False and not only:
                left_out[k] = "screen: false"
                continue
            if "range" not in r and "choices" not in r and not isinstance(self.default(k, mapper), bool):
                left_out[k] = "no range"
                continue
            when = r.get("when")
            if when and when not in out and not self.default(when, mapper):
                left_out[k] = f"{when} is off"
                continue
            s = self.spec(e, mapper)
            if plausible and "plausible" in r and s.kind == "float":
                s = Spec(s.kind, *map(float, r["plausible"]), s.log, when=s.when)
            out[k] = s
        return out, left_out

    def defaults(self, mapper: str) -> dict:
        """The default value of every searched parameter."""
        return {k: self.default(k, mapper) for k in self.space(mapper)}

    def plausible(self) -> dict[str, tuple]:
        return {k: tuple(e.raw["plausible"]) for k, e in self.entries.items() if "plausible" in e.raw}

    # ----------------------------------------------------------------------------------------------------------------
    # Cell size and stand-ins
    # ----------------------------------------------------------------------------------------------------------------

    def cell_size_m(self, site: str | None) -> float:
        if self.cell_size == "nominal":
            return float(self.nominal_cell_size[self.dem])
        if site is None:
            raise SettingsError("dem.cell_size is measured, so converting metres to cells needs a site")
        return measured_cell_size(site, self.dem)

    def derived_args(self, v: dict, cell: float) -> dict:
        """The ARC arguments the stand-ins set, from the stand-ins' values v."""
        out = {}
        if v["rotate"] and v["rot_half_width"] > 0:
            # ARC turns the cross section by 0, ±interval, ... up to ±Degree_Manip/2
            out["Degree_Manip"] = 2.0 * v["rot_half_width"] * (1.0 + 1e-9)
            out["Degree_Interval"] = v["rot_half_width"] / int(v["rot_steps"])
        else:
            out["Degree_Manip"], out["Degree_Interval"] = 0.0, 1.0
        if v["dvn"]:
            s, scale = float(v["dvn_shallow"]), float(v["dvn_L"])
            out.update(Depth_Varying_N=True, shallow_factor=s, deep_factor=1.0, k_decay=max((s - 1.0) / scale, 1e-9))
        else:
            out["Depth_Varying_N"] = False
        if v["power_depth"]:
            out["coefficient_depth"] = float(v["depth_law_coefficient"])
            out["exponent_depth"] = float(v["depth_law_exponent"])
        out["Gen_Slope_Dist"] = metres_to_cells(v["Gen_Slope_Dist_m"], cell)
        out["Gen_Dir_Dist"] = metres_to_cells(v["Gen_Dir_Dist_m"], cell)
        return out

    # ----------------------------------------------------------------------------------------------------------------
    # A parameter set -> the tools' arguments
    # ----------------------------------------------------------------------------------------------------------------

    def values(self, p: dict, mapper: str) -> dict:
        """Every entry's value for a mapper: the parameter set's, else the default."""
        unknown = [k for k in p if k not in self.entries]
        if unknown:
            raise SettingsError(f"not in the settings file: {unknown}")
        return {k: (p[k] if k in p else self.default(k, mapper)) for k, e in self.entries.items() if not e.is_from}

    def model_args(self, p: dict, mapper: str, site: str | None) -> tuple[dict, dict, dict, dict]:
        """nencarta's options, arc_args, curve2flood_args and fldpln_library_args for one parameter set. An argument
        is left out when its value is null, when its mapper is another, or while the switch it depends on is off."""
        v = self.values(p, mapper)
        out = {s: {} for s in TOOL_SECTIONS}
        for k, e in self.entries.items():
            if e.section not in TOOL_SECTIONS or e.is_from or not self.applies(e, mapper):
                continue
            when = e.raw.get("when")
            if when and not v[when]:
                continue
            if v[k] is None:
                continue
            out[e.section][e.name] = v[k]
        derived = self.derived_args(v, self.cell_size_m(site))
        for k, e in self.entries.items():
            if e.section == "arc" and e.is_from:
                if k in p:                              # a parameter set may set a stand-in's argument directly
                    out["arc"][e.name] = p[k]
                elif e.name in derived:
                    out["arc"][e.name] = derived[e.name]
        neo = out["nencarta"]
        if neo.get("lakes"):
            neo["lakes"] = str(Path(neo["lakes"]).expanduser())
        return neo, out["arc"], out["curve2flood"], (out["fldpln_library"] if mapper == "fldpln" else {})

    def mannings_rows(self, p: dict, mapper: str) -> list[tuple[int, str, float]]:
        rows = []
        for k, e in self.entries.items():
            if e.section == "mannings_n":
                rows.append((int(e.raw["class"]), str(e.raw["label"]), float(p.get(k, self.default(k, mapper)))))
        return sorted(rows)

    # ----------------------------------------------------------------------------------------------------------------
    # Fingerprints
    # ----------------------------------------------------------------------------------------------------------------

    def fingerprint(self, mapper: str) -> str:
        """A short hash of everything fixed for a mapper: every value not searched, the DEM and the cell sizes.
        Stored with each evaluation."""
        space = self.space(mapper)
        fixed = {k: self.default(k, mapper) for k, e in self.entries.items()
                 if not e.is_from and k not in space and self.applies(e, mapper)}
        blob = {"fixed": fixed, "dem": self.dem, "cell_size": self.cell_size, "nominal": self.nominal_cell_size,
                "flow_stats": config.FLOW_STATS_FILE}
        return hashlib.sha1(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()[:8]

    def cache_fingerprint(self, mapper: str) -> str:
        """A short hash of what the cached preprocessing depends on: the nencarta section and the DEM."""
        neo = {e.name: self.default(k, mapper) for k, e in self.entries.items() if e.section == "nencarta"}
        blob = {"nencarta": neo, "dem": self.dem, "mapper": mapper}
        return hashlib.sha1(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()[:8]

    def snapshot(self, mapper: str) -> Path:
        """A copy of the settings file under results/settings/, named by its fingerprint."""
        path = config.results_path("settings", f"{mapper}_{self.fingerprint(mapper)}.yaml")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {self.path}\n" + self.text)
        return path

    # ----------------------------------------------------------------------------------------------------------------
    # Per-site tuning
    # ----------------------------------------------------------------------------------------------------------------

    def persite(self, mapper: str) -> dict[str, dict]:
        """The reduced parameters per-site tuning searches: {name: {"scales": [...] or None, "spec": Spec}}."""
        out = {}
        for name, raw in self.persite_raw.items():
            if raw.get("mapper") not in (None, mapper) or not self.resolve(raw.get("optimize", False), mapper):
                continue
            if "scales" in raw:
                lo, hi = raw["range"]
                out[name] = {"scales": list(raw["scales"]), "spec": Spec("float", lo, hi, bool(raw.get("log")))}
            else:
                spec = self.spec(self.entries[name], mapper)
                if "range" in raw:
                    lo, hi = raw["range"]
                    spec = Spec(spec.kind, lo, hi, bool(raw.get("log", spec.log)))
                out[name] = {"scales": None, "spec": spec}
        return out

    def table(self, mapper: str) -> list[dict]:
        """Every entry as a row, for display."""
        space = self.space(mapper)
        rows = []
        for k, e in self.entries.items():
            r = e.raw
            if e.is_from:
                value, search = f"from {r['from']}", ""
            else:
                value = self.resolve(r.get("value"), mapper)
                if k in space:
                    s = space[k]
                    search = (f"{s.low} – {s.high}{' log' if s.log else ''}" if s.kind in ("float", "int")
                              else "on/off" if s.kind == "bool" else "/".join(map(str, s.choices)))
                    search += f" (when {s.when})" if s.when else ""
                else:
                    search = ""
            rows.append({"section": e.section, "name": k, "value": value, "searched": search,
                         "used": "" if self.applies(e, mapper) else f"{r.get('mapper')} only"})
        return rows


# --------------------------------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------------------------------

def metres_to_cells(metres: float, cell: float) -> int:
    """A length along the stream in whole cells, at least 1 (halves round up)."""
    return max(1, int(math.floor(float(metres) / cell + 0.5)))


_MEASURED: dict = {}


def measured_cell_size(site: str, dem: str) -> float:
    """A site's DEM cell, in metres: the geometric mean of its width and height (degrees converted at the DEM's
    middle latitude)."""
    key = (site, dem)
    if key not in _MEASURED:
        from osgeo import gdal, osr
        gdal.UseExceptions()
        ds = gdal.Open(str(config.SITES_ROOT / site / "DEMs" / f"{dem}.tif"))
        gt = ds.GetGeoTransform()
        srs = osr.SpatialReference(wkt=ds.GetProjectionRef())
        dx, dy = abs(gt[1]), abs(gt[5])
        if srs.IsGeographic():
            lat = math.radians(gt[3] + gt[5] * ds.RasterYSize / 2)
            dx *= 111320.0 * math.cos(lat)
            dy *= 110574.0
        _MEASURED[key] = math.sqrt(dx * dy)
    return _MEASURED[key]


def _check_entry(where: str, raw, allowed: set) -> dict:
    if not isinstance(raw, dict):
        raise SettingsError(f"{where}: expected {{value: ...}}, got {raw!r}")
    extra = set(raw) - allowed
    if extra:
        raise SettingsError(f"{where}: unknown field(s) {sorted(extra)} (allowed: {sorted(allowed)})")
    return raw


def load(path: str | os.PathLike | None = None) -> Settings:
    """Read and check a settings file."""
    import yaml
    path = Path(path or os.environ.get("SITE_EVAL_SETTINGS") or DEFAULT_PATH).expanduser().resolve()
    text = path.read_text()
    data = yaml.safe_load(text) or {}
    unknown = set(data) - {*SECTIONS, "dem", "persite"}
    if unknown:
        raise SettingsError(f"{path}: unknown section(s) {sorted(unknown)}")
    dem = data.get("dem") or {}
    nominal = dict(dem.get("nominal_cell_size_m") or {})
    name = dem.get("name", "fabdem")
    if dem.get("cell_size", "nominal") not in ("nominal", "measured"):
        raise SettingsError("dem.cell_size must be nominal or measured")
    if dem.get("cell_size", "nominal") == "nominal" and name not in nominal:
        raise SettingsError(f"dem.nominal_cell_size_m has no cell size for {name}")

    # names: <section>.<name> when the same name (ignoring case) appears in two sections
    counts: dict[str, int] = {}
    for sec in SECTIONS:
        for n in (data.get(sec) or {}):
            counts[n.lower()] = counts.get(n.lower(), 0) + 1
    entries = {}
    for sec in SECTIONS:
        for n, raw in (data.get(sec) or {}).items():
            raw = _check_entry(f"{sec}.{n}", raw, ENTRY_FIELDS)
            key = f"{sec}.{n}" if counts[n.lower()] > 1 else n
            entries[key] = Entry(sec, n, key, raw)
    s = Settings(path, text, name, dem.get("cell_size", "nominal"), nominal, entries, data.get("persite") or {})
    _validate(s)
    return s


def _validate(s: Settings) -> None:
    missing = [n for n in DERIVED_NAMES if n not in s.entries]
    if missing:
        raise SettingsError(f"derived is missing {missing}")
    for k, e in s.entries.items():
        r = e.raw
        where = f"{e.section}.{e.name}"
        if e.is_from:
            if e.section != "arc" or r["from"] not in DERIVED_GROUPS or e.name not in DERIVED_GROUPS[r["from"]]:
                raise SettingsError(f"{where}: `from` must name the stand-in that sets it "
                                    f"({[g for g, a in DERIVED_GROUPS.items() if e.name in a]})")
            if set(r) - {"from"}:
                raise SettingsError(f"{where}: a `from` line has no other fields; replace `from` to set it directly")
            continue
        if "value" not in r:
            raise SettingsError(f"{where}: no value")
        if r.get("when") and (r["when"] not in s.entries or s.entries[r["when"]].is_from):
            raise SettingsError(f"{where}: when names {r['when']!r}, which isn't a setting with a value in the file")
        if "screen" in r and not isinstance(r["screen"], bool):
            raise SettingsError(f"{where}: screen must be true or false")
        if r.get("mapper") not in (None, *ALL_MAPPERS):
            raise SettingsError(f"{where}: mapper must be one of {ALL_MAPPERS}")
        for mapper in config.MAPPERS:
            try:
                v = s.resolve(r.get("value"), mapper)
                opt = s.resolve(r.get("optimize", False), mapper)
            except SettingsError as exc:
                raise SettingsError(f"{where}: {exc}") from None
            if not opt or not s.applies(e, mapper):
                continue
            if e.section == "nencarta":
                raise SettingsError(f"{where}: nencarta's options shape the cached preprocessing and can't be "
                                    "optimized")
            if "range" not in r and "choices" not in r and not isinstance(v, bool):
                raise SettingsError(f"{where}: optimize needs a range (numbers) or choices")
            if "range" in r:
                lo, hi = r["range"]
                if not lo < hi:
                    raise SettingsError(f"{where}: range must be [low, high]")
                if r.get("log") and lo <= 0:
                    raise SettingsError(f"{where}: a log range must be above 0")
    for g, args in DERIVED_GROUPS.items():
        native = [a for a in args if a in s.entries and not s.entries[a].is_from]
        for mapper in config.MAPPERS:
            if native and s.optimized(s.entries[g], mapper):
                raise SettingsError(f"derived.{g} is searched, but arc sets {native} directly: set {g}'s optimize to "
                                    "false, or put `from` back")
    for name, raw in s.persite_raw.items():
        _check_entry(f"persite.{name}", raw, PERSITE_FIELDS)
        if "scales" in raw:
            bad = [x for x in raw["scales"] if x not in s.entries]
            if bad or "range" not in raw:
                raise SettingsError(f"persite.{name}: scales must name parameters ({bad} aren't) and needs a range")
        elif name not in s.entries:
            raise SettingsError(f"persite.{name}: not a parameter of the file (or give `scales`)")


_CURRENT: Settings | None = None


def use(path: str | os.PathLike | None) -> Settings:
    """Make a settings file the current one (also for worker processes started afterwards)."""
    global _CURRENT
    _CURRENT = load(path)
    os.environ["SITE_EVAL_SETTINGS"] = str(_CURRENT.path)
    return _CURRENT


def current() -> Settings:
    global _CURRENT
    if _CURRENT is None:
        _CURRENT = load()
    return _CURRENT
