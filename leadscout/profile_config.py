"""Typed views of the profile sections that carry policy: fit, compliance, research,
notification.

`profile.py` loads a YAML file and hands out raw sections. This module turns a section
into a frozen dataclass, with the code defaults applied ONLY for keys the section does
not mention (and for a section the profile omits entirely). A key it does not know is a
`ProfileError`: a typo in a profile must not silently score leads with the default.

The defaults reproduce the numbers the pipeline shipped with. Neutral wording lives in
`config/profiles/default.yaml`, vendor-specific wording in `config/profiles/infra-vendor.yaml`;
the code carries neither.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .profile import OrgProfile, ProfileError, active_profile

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

CONFIDENCE_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

_DEFAULT_SIZE_BANDS = {
    "1-10": (5, 1, 10), "11-50": (30, 11, 50), "51-200": (120, 51, 200),
    "201-1000": (500, 201, 1000), "1000+": (3000, 1000, 10_000_000),
}
_DEFAULT_SCALE_TIERS = (
    (10, 3, ""), (50, 10, ""), (200, 18, ""), (1000, 25, ""), (10000, 30, ""), (None, 24, ""),
)
_DEFAULT_CLOUD_STATE_POINTS = {"CONFIRMED": 40, "LIKELY": 30, "POSSIBLE": 18, "EDGE_ONLY": 5, "UNKNOWN": 0}
_DEFAULT_WORKLOAD = {
    "gpu/ai/ml workload": ("gpu", "ai workload", "ml workload", "machine learning workload", "gpu cluster"),
    "large-scale data": ("petabyte", "terabyte", "large-scale data", "big data"),
    "multi-region/multi-cloud": ("multi-region", "multi-cloud", "multiregion", "multicloud"),
    "kubernetes/iac at scale": ("kubernetes", "terraform", "k8s"),
    "sre/platform hiring signal": ("sre", "site reliability", "platform engineer", "cloud engineer"),
}
_DEFAULT_SPEND_USD = {"infra-heavy": 60, "neutral": 15, "low-infra": 3}
_DEFAULT_TIER_EDGES = ((1_000, "LOW"), (100_000, "MEDIUM"), (1_000_000, "HIGH"))


def _check(section: dict, allowed: set[str], where: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ProfileError(f"{where}: unknown keys {unknown}; expected a subset of {sorted(allowed)}")


def _tuple_of_str(value: Any, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
        raise ProfileError(f"{where} must be a list of strings")
    return tuple(value)


def _int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProfileError(f"{where} must be an integer, got {value!r}")
    return value


def _mapping(value: Any, where: str) -> dict:
    if not isinstance(value, dict):
        raise ProfileError(f"{where} must be a mapping")
    return value


# --- fit -------------------------------------------------------------------------

@dataclass(frozen=True)
class ComplexityRule:
    label: str          # may contain {industry}
    points: int
    field: str          # "text" (site text + summary) or "industry"
    any_of: tuple[str, ...] = ()
    all_of: tuple[str, ...] = ()


_DEFAULT_COMPLEXITY_RULES = (
    ComplexityRule("multiple products/markets", 5, "text",
                   any_of=("multiple products", "product suite", "product portfolio",
                           "multiple markets", "international markets")),
    ComplexityRule("regulated industry '{industry}'", 5, "industry",
                   any_of=("fintech", "financial", "health", "healthcare", "medical", "bank", "insurance")),
    ComplexityRule("international offices", 5, "text",
                   any_of=("international offices", "offices in", "global offices",
                           "worldwide offices", "offices across")),
    ComplexityRule("on-prem + cloud (hybrid infra)", 5, "text", all_of=("on-prem", "cloud")),
)


@dataclass(frozen=True)
class FitConfig:
    size_bands: dict[str, tuple[int, int, int]]        # band -> (employees used, low, high)
    unknown_headcount_points: int
    unknown_headcount_note: str
    scale_tiers: tuple[tuple[int | None, int, str], ...]  # (below, points, note); last has below=None
    cloud_state_points: dict[str, int]
    cloud_points_cap: int
    workload_categories: dict[str, tuple[str, ...]]
    workload_points_per_category: int
    workload_cap: int
    complexity_rules: tuple[ComplexityRule, ...]
    complexity_cap: int
    high_spend_industries: tuple[str, ...]
    low_spend_industries: tuple[str, ...]
    spend_usd_per_employee: dict[str, int]
    spend_tier_edges: tuple[tuple[int, str], ...]
    spend_tier_above: str
    max_score: int
    threshold: int
    sales_ready_min_confidence: str
    requires_cloud_signal: bool

    def band_employees(self) -> dict[str, int]:
        return {band: v[0] for band, v in self.size_bands.items()}

    def band_range(self) -> dict[str, tuple[int, int]]:
        return {band: (v[1], v[2]) for band, v in self.size_bands.items()}


_FIT_KEYS = {"size_bands", "scale", "cloud", "workload_intensity", "complexity", "spend",
             "max_score", "threshold", "sales_ready_min_confidence", "confidence"}


def fit_config(profile: OrgProfile | None = None) -> FitConfig:
    profile = profile or active_profile()
    sec = profile.section("fit")
    where = f"profile {profile.name!r}: fit"
    _check(sec, _FIT_KEYS, where)

    bands = dict(_DEFAULT_SIZE_BANDS)
    if "size_bands" in sec:
        bands = {}
        for band, v in _mapping(sec["size_bands"], f"{where}.size_bands").items():
            v = _mapping(v, f"{where}.size_bands.{band}")
            _check(v, {"employees", "min", "max"}, f"{where}.size_bands.{band}")
            bands[str(band)] = (_int(v.get("employees"), f"{where}.size_bands.{band}.employees"),
                                _int(v.get("min"), f"{where}.size_bands.{band}.min"),
                                _int(v.get("max"), f"{where}.size_bands.{band}.max"))

    scale = _mapping(sec.get("scale") or {}, f"{where}.scale")
    _check(scale, {"unknown", "tiers"}, f"{where}.scale")
    unk = _mapping(scale.get("unknown") or {}, f"{where}.scale.unknown")
    _check(unk, {"points", "note"}, f"{where}.scale.unknown")
    tiers = _DEFAULT_SCALE_TIERS
    if "tiers" in scale:
        raw = scale["tiers"]
        if not isinstance(raw, list) or not raw:
            raise ProfileError(f"{where}.scale.tiers must be a non-empty list")
        built = []
        for i, t in enumerate(raw):
            t = _mapping(t, f"{where}.scale.tiers[{i}]")
            _check(t, {"below", "points", "note"}, f"{where}.scale.tiers[{i}]")
            below = t.get("below")
            if below is not None:
                below = _int(below, f"{where}.scale.tiers[{i}].below")
            built.append((below, _int(t.get("points"), f"{where}.scale.tiers[{i}].points"),
                          str(t.get("note") or "")))
        if built[-1][0] is not None or any(b[0] is None for b in built[:-1]):
            raise ProfileError(f"{where}.scale.tiers: only the LAST tier may omit `below`, and it must")
        tiers = tuple(built)

    cloud = _mapping(sec.get("cloud") or {}, f"{where}.cloud")
    _check(cloud, {"state_points", "points_cap"}, f"{where}.cloud")
    state_points = dict(_DEFAULT_CLOUD_STATE_POINTS)
    if "state_points" in cloud:
        state_points = {str(k): _int(v, f"{where}.cloud.state_points.{k}")
                        for k, v in _mapping(cloud["state_points"], f"{where}.cloud.state_points").items()}

    wl = _mapping(sec.get("workload_intensity") or {}, f"{where}.workload_intensity")
    _check(wl, {"points_per_category", "cap", "categories"}, f"{where}.workload_intensity")
    categories = dict(_DEFAULT_WORKLOAD)
    if "categories" in wl:
        categories = {str(k): _tuple_of_str(v, f"{where}.workload_intensity.categories.{k}")
                      for k, v in _mapping(wl["categories"], f"{where}.workload_intensity.categories").items()}

    cx = _mapping(sec.get("complexity") or {}, f"{where}.complexity")
    _check(cx, {"points_cap", "rules"}, f"{where}.complexity")
    rules = _DEFAULT_COMPLEXITY_RULES
    if "rules" in cx:
        built_rules = []
        for i, r in enumerate(cx["rules"] or []):
            r = _mapping(r, f"{where}.complexity.rules[{i}]")
            _check(r, {"label", "points", "field", "any", "all"}, f"{where}.complexity.rules[{i}]")
            fld = r.get("field", "text")
            if fld not in ("text", "industry"):
                raise ProfileError(f"{where}.complexity.rules[{i}].field must be 'text' or 'industry'")
            built_rules.append(ComplexityRule(
                str(r.get("label") or ""), _int(r.get("points"), f"{where}.complexity.rules[{i}].points"), fld,
                _tuple_of_str(r.get("any"), f"{where}.complexity.rules[{i}].any"),
                _tuple_of_str(r.get("all"), f"{where}.complexity.rules[{i}].all")))
        rules = tuple(built_rules)

    sp = _mapping(sec.get("spend") or {}, f"{where}.spend")
    _check(sp, {"high_industries", "low_industries", "usd_per_employee", "tier_edges", "tier_above"},
           f"{where}.spend")
    usd = dict(_DEFAULT_SPEND_USD)
    if "usd_per_employee" in sp:
        usd = {str(k): _int(v, f"{where}.spend.usd_per_employee.{k}")
               for k, v in _mapping(sp["usd_per_employee"], f"{where}.spend.usd_per_employee").items()}
        missing = {"infra-heavy", "neutral", "low-infra"} - set(usd)
        if missing:
            raise ProfileError(f"{where}.spend.usd_per_employee is missing {sorted(missing)}")
    edges = _DEFAULT_TIER_EDGES
    if "tier_edges" in sp:
        edges = tuple((_int(e[0], f"{where}.spend.tier_edges"), str(e[1])) for e in sp["tier_edges"])

    conf = _mapping(sec.get("confidence") or {}, f"{where}.confidence")
    _check(conf, {"requires_cloud_signal"}, f"{where}.confidence")
    min_conf = str(sec.get("sales_ready_min_confidence", "MEDIUM")).upper()
    if min_conf not in CONFIDENCE_ORDER:
        raise ProfileError(f"{where}.sales_ready_min_confidence must be one of {sorted(CONFIDENCE_ORDER)}")

    return FitConfig(
        size_bands=bands,
        unknown_headcount_points=_int(unk.get("points", 3), f"{where}.scale.unknown.points"),
        unknown_headcount_note=str(unk.get("note", "")),
        scale_tiers=tiers,
        cloud_state_points=state_points,
        cloud_points_cap=_int(cloud.get("points_cap", 55), f"{where}.cloud.points_cap"),
        workload_categories=categories,
        workload_points_per_category=_int(wl.get("points_per_category", 5),
                                          f"{where}.workload_intensity.points_per_category"),
        workload_cap=_int(wl.get("cap", 15), f"{where}.workload_intensity.cap"),
        complexity_rules=rules,
        complexity_cap=_int(cx.get("points_cap", 15), f"{where}.complexity.points_cap"),
        high_spend_industries=_tuple_of_str(sp.get("high_industries", _DEFAULT_HIGH_SPEND),
                                            f"{where}.spend.high_industries"),
        low_spend_industries=_tuple_of_str(sp.get("low_industries", _DEFAULT_LOW_SPEND),
                                           f"{where}.spend.low_industries"),
        spend_usd_per_employee=usd,
        spend_tier_edges=edges,
        spend_tier_above=str(sp.get("tier_above", "VERY_HIGH")),
        max_score=_int(sec.get("max_score", 100), f"{where}.max_score"),
        threshold=_int(sec.get("threshold", 60), f"{where}.threshold"),
        sales_ready_min_confidence=min_conf,
        requires_cloud_signal=bool(conf.get("requires_cloud_signal", True)),
    )


_DEFAULT_HIGH_SPEND = (
    "software", "saas", "fintech", "e-commerce", "ecommerce", "marketplace", "media",
    "streaming", "gaming", "adtech", "data", "analytics", "ai", "machine learning",
    "telecom", "internet", "cloud", "platform", "logistics tech", "mobility", "ride-hailing",
    "ride hailing", "on-demand", "delivery", "app", "transportation network")
_DEFAULT_LOW_SPEND = (
    "restaurant", "bakery", "retail store", "construction", "law firm", "dental",
    "hair", "salon", "real estate agency", "farm", "consulting (individual)")


def size_bands(profile: OrgProfile | None = None) -> tuple[str, ...]:
    """The allowed company_size_band values, in profile order. The one definition."""
    return tuple(fit_config(profile).size_bands)


# --- compliance ------------------------------------------------------------------

@dataclass(frozen=True)
class ComplianceConfig:
    competitors: tuple[str, ...]
    competitors_path: Path | None          # None when the list is inline or empty
    jurisdictions: tuple[str, ...]
    markers: tuple[str, ...]
    jurisdictions_path: Path | None
    competitor_offering: str               # "" -> the "competitor in substance" prompt line is omitted
    generic_name_tokens: frozenset[str]
    generic_name_example: str              # shown in the prompt: the word "<x>" alone is not a match
    generic_words_hint: str                # shown in the prompt: "Generic words (<hint>) never justify..."


_COMPLIANCE_KEYS = {"competitors", "competitors_file", "jurisdictions_file", "competitor_offering",
                    "generic_name_tokens", "generic_name_example", "generic_words_hint"}
_DEFAULT_GENERIC_TOKENS = ("solutions", "systems", "services", "group", "tech")


def _resolve_config_path(value: str, profile: OrgProfile, where: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = CONFIG_DIR / p
    if not p.is_file():
        raise ProfileError(f"profile {profile.name!r}: {where} -> no file at {p}")
    return p


def _yaml_list(path: Path, key: str) -> tuple[str, ...]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return tuple(data.get(key, []))


def compliance_config(profile: OrgProfile | None = None) -> ComplianceConfig:
    profile = profile or active_profile()
    sec = profile.section("compliance")
    where = f"profile {profile.name!r}: compliance"
    _check(sec, _COMPLIANCE_KEYS, where)
    if "competitors" in sec and "competitors_file" in sec:
        raise ProfileError(f"{where}: give `competitors` or `competitors_file`, not both")

    comp_path: Path | None = None
    if "competitors_file" in sec:
        comp_path = _resolve_config_path(str(sec["competitors_file"]), profile, "competitors_file")
        competitors = _yaml_list(comp_path, "competitors")
    elif "competitors" in sec:
        competitors = _tuple_of_str(sec["competitors"], f"{where}.competitors")
    else:
        comp_path = CONFIG_DIR / "do_not_engage.yaml"
        competitors = _yaml_list(comp_path, "competitors")

    jur_path = _resolve_config_path(str(sec.get("jurisdictions_file", "restricted_jurisdictions.yaml")),
                                    profile, "jurisdictions_file")
    tokens = _tuple_of_str(sec["generic_name_tokens"], f"{where}.generic_name_tokens") \
        if "generic_name_tokens" in sec else _DEFAULT_GENERIC_TOKENS
    return ComplianceConfig(
        competitors=competitors, competitors_path=comp_path,
        jurisdictions=_yaml_list(jur_path, "jurisdictions"), markers=_yaml_list(jur_path, "markers"),
        jurisdictions_path=jur_path,
        competitor_offering=str(sec.get("competitor_offering", "")).strip(),
        generic_name_tokens=frozenset(t.lower() for t in tokens),
        generic_name_example=str(sec.get("generic_name_example", "")).strip(),
        generic_words_hint=str(sec.get("generic_words_hint", "inc, co, systems, solutions")).strip(),
    )


# --- research / notification -----------------------------------------------------

@dataclass(frozen=True)
class ResearchConfig:
    tech_signal_example: str
    tech_signal_guidance: str


def research_config(profile: OrgProfile | None = None) -> ResearchConfig:
    profile = profile or active_profile()
    sec = profile.section("research")
    _check(sec, {"tech_signal_example", "tech_signal_guidance"}, f"profile {profile.name!r}: research")
    return ResearchConfig(
        tech_signal_example=str(sec.get("tech_signal_example", "Postgres")).strip(),
        tech_signal_guidance=str(sec.get(
            "tech_signal_guidance",
            "technologies, platforms and vendors the company says it uses, engineering job postings, "
            "scale or volume statements - name the actual signal")).strip(),
    )


@dataclass(frozen=True)
class NotificationConfig:
    # str.format placeholders: {company} {score} {confidence} {flag}
    subject_template: str


def notification_config(profile: OrgProfile | None = None) -> NotificationConfig:
    profile = profile or active_profile()
    sec = profile.section("notification")
    _check(sec, {"subject_template"}, f"profile {profile.name!r}: notification")
    return NotificationConfig(
        subject_template=str(sec.get(
            "subject_template", "[Lead] {company} - fit {score}/100 ({confidence}) - compliance {flag}")))
