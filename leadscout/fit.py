"""Fit score, 0-100. Deterministic, so a rep can argue with it.
"""

from __future__ import annotations

import re

from .cloud import _CLOUD_BEARING_FAMILIES, _EDGE_ONLY_FAMILY
from .models import FitResult, Lead, Research
from .profile import OrgProfile, active_profile
from .profile_config import CONFIDENCE_ORDER, FitConfig, fit_config

# --- Design notes -----------------------------------------------------------------
# Three components, each capped, summing to at most 100 - no confidence cap on the
# score itself (that was v1's approach; see the design notes for why it was dropped):
#
# | component            | range | source                                                                  |
# |-----------------------|-------|--------------------------------------------------------------------------|
# | scale_points          | 0-30  | employee count (LLM-extracted, or the self-reported band)                |
# | cloud_signal_points   | 0-55  | leadscout.cloud's CloudUsageAssessment state + workload-intensity bonus  |
# | complexity_points     | 0-15  | multi-product/market, regulated industry, offices, hybrid infra          |
#
# Every number, keyword list and rationale string below comes from the active organisation
# profile (`config/profiles/*.yaml`, section `fit`; schema in `profile_config.py`). The
# ranges in the table above are those of `config/profiles/infra-vendor.yaml`, the profile the
# original built-in ICP lives in - read it for why cloud evidence outweighs headcount there.
# Cloud spend is scored
# from *evidence* (a deterministic assessment over every provider's Evidence, see
# leadscout/cloud.py) rather than assumed from headcount the way the old size-capped
# formula did. `cloud_signal_points` used to keyword-match `research.tech_signals`/
# `website_text` directly; that path is gone - the LLM's own guess at
# "Kubernetes"/"API"/"cloud" wording no longer scores fit, the evidence-backed
# `CloudUsageAssessment` state does.
#
# A separate `fit_confidence` (HIGH/MEDIUM/LOW) says how much to trust the score, since
# a high score built on thin evidence is exactly what wastes a sales call. `sales_ready`
# (computed in pipeline.py) additionally requires `fit_confidence != LOW`.

def employees_for_scoring(research, lead, profile: OrgProfile | None = None) -> tuple[int | None, str]:
    """(employees, why). The applicant's own size band is a statement about their own company: when an
    inferred figure falls OUTSIDE that band, the band wins and the conflict is stated - an LLM reading
    "5 people in the Berlin office" must not turn a 201-1000 lead into a 5-person one (REVIEW-A §4)."""
    inferred, band = research.estimated_employees, getattr(lead, "company_size_band", "")
    cfg = fit_config(profile)
    lo_hi = cfg.band_range().get(band)
    if inferred and lo_hi and not (lo_hi[0] <= inferred <= lo_hi[1]):
        return cfg.band_employees()[band], (f"inferred {inferred} contradicts the self-reported band {band}; "
                                          f"using the band")
    if inferred:
        return inferred, ""
    if lo_hi:
        return cfg.band_employees()[band], f"self-reported band {band}"
    return None, ""

def _scale_points(employees: int | None, profile: OrgProfile | None = None) -> tuple[int, str]:
    cfg = fit_config(profile)
    if employees is None:
        note = f" ({cfg.unknown_headcount_note})" if cfg.unknown_headcount_note else ""
        return cfg.unknown_headcount_points, (
            f"headcount unknown -> {cfg.unknown_headcount_points}{note}")
    for below, points, note in cfg.scale_tiers:
        if below is None or employees < below:
            suffix = f" ({note})" if note else ""
            return points, f"~{employees} employees -> {points}{suffix}"
    raise AssertionError("unreachable: the last scale tier has no upper bound")


def _extract_signal_and_evidence(raw: str) -> tuple[str, str | None]:
    """"Kubernetes (ev-001)" -> ("Kubernetes", "ev-001"); no citation -> (raw, None)."""
    m = re.match(r"^(.*?)\s*\((ev-\d+)\)\s*$", raw.strip())
    if m:
        return m.group(1).strip(), m.group(2)
    return raw.strip(), None


def _evidence_by_id(research: Research, ev_id: str | None):
    """The actual Evidence item a tech_signal's "(ev-NNN)" citation points at, or
    None when there's no id, or none of `research.evidence` has it (Fix #7: "the
    evidence id cited for a point must be the item that observed that signal" -
    without this lookup, fit.py trusted the LLM's own citation string blindly)."""
    if not ev_id:
        return None
    return next((ev for ev in research.evidence if ev.id == ev_id), None)


def _cloud_signal_points(research: Research, profile: OrgProfile | None = None) -> tuple[int, str, int]:
    """Returns (points capped at 55, human-readable reasoning, count of distinct
    EVIDENCE-BACKED signal categories matched - used by fit_confidence).

    F-13: the two counts deliberately diverge. A workload-intensity category always
    earns its +5 (inference is allowed to score), but it only
    raises the confidence counter when the evidence id it cites belongs to a
    CLOUD-BEARING family. A model-written "Kubernetes (ev-001)" citing the company's
    own generic landing page (`corporate_identity`) is the model's claim about a page
    that observed no such thing - it may move the score, it must not move how much we
    say the score is worth.

    Base points come entirely from `research.cloud_usage.state` - the deterministic,
    evidence-backed assessment in leadscout/cloud.py, never a keyword match against
    tech_signals/website_text (that was the earlier design; see module
    docstring). The only remaining keyword pass is the +5-per-category, capped-at-15
    "workload intensity" bonus on top of that base."""
    cfg = fit_config(profile)
    state = research.cloud_usage.state
    base = cfg.cloud_state_points.get(state, 0)
    citations = [f"cloud usage assessment: {state} -> {base} ({research.cloud_usage.boundary})"]
    signal_categories = 1 if base > 0 else 0

    bonus = 0
    for category, keywords in cfg.workload_categories.items():
        for raw_signal in research.tech_signals:
            name, ev_id = _extract_signal_and_evidence(raw_signal)
            if not any(kw in name.lower() for kw in keywords):
                continue
            # REVIEW-A §8: a signal whose citation resolves to no Evidence item earns nothing -
            # the id must identify the observation that actually saw the signal.
            cited = _evidence_by_id(research, ev_id)
            if cited is None:
                continue
            if category == "kubernetes/iac at scale":
                # Fix #7: a Dockerfile / .github/workflows root signal alone
                # (github.py's own MEDIUM strength - STRONG is reserved for a root
                # .tf file that actually declares a cloud-provider block) is not
                # itself Kubernetes/IaC-at-scale evidence - never award this
                # category's point off that citation. An id we can't resolve back
                # to a real Evidence item is left as-is (nothing to disprove).
                if cited.family == "engineering_footprint" and cited.strength != "STRONG":
                    continue
            bonus += cfg.workload_points_per_category
            # F-13: confidence credit only for a citation into a cloud-bearing family -
            # and never into `edge_delivery`, which `_fit_confidence`'s own family count
            # already excludes. A CDN record observed that this domain is fronted by
            # Cloudflare; it did not observe a Kubernetes cluster, a GPU or a petabyte.
            # Counting it here let the exclusion hold on one path and not the other, and
            # a CDN-only lead reached HIGH and sales_ready (measured).
            if cited.family in _CLOUD_BEARING_FAMILIES and cited.family != _EDGE_ONLY_FAMILY:
                signal_categories += 1
            citations.append(f"workload intensity: {category} +{cfg.workload_points_per_category} "
                             f"({ev_id or 'no evidence id cited'})")
            break
    bonus = min(bonus, cfg.workload_cap)

    points = min(base + bonus, cfg.cloud_points_cap)

    # Fix #7: a PRIVATE_CLOUD/MANAGED_HOSTING/UNKNOWN-with-holder infrastructure
    # footprint (a network holder or a cPanel/managed-hosting trace) never earns a
    # public-cloud spend point above - cloud.py already excludes it from the state
    # computation (its Evidence.provider is None - see infra.py's main invariant),
    # so this is purely a reasoning note explaining the zero, not a score change.
    if base == 0:
        for fp in research.infrastructure:
            if fp.category in ("PRIVATE_CLOUD", "MANAGED_HOSTING", "UNKNOWN"):
                who = fp.provider or fp.network_holder or "operator not established"
                citations.append(
                    f"public-cloud bill not evidenced: {fp.category} footprint ({who}) observed "
                    "but is not a public-cloud spend signal")

    why = "; ".join(citations)
    return points, why, signal_categories


def _complexity_points(research: Research, profile: OrgProfile | None = None) -> tuple[int, str]:
    cfg = fit_config(profile)
    text = ((research.website_text or "")[:3000] + " " + (research.summary or "")).lower()
    ind = (research.industry or "").lower()
    reasons = []
    points = 0
    for rule in cfg.complexity_rules:
        haystack = text if rule.field == "text" else ind
        if rule.any_of and not any(k in haystack for k in rule.any_of):
            continue
        if rule.all_of and not all(k in haystack for k in rule.all_of):
            continue
        if not rule.any_of and not rule.all_of:
            continue
        points += rule.points
        reasons.append(f"{rule.label.format(industry=research.industry)} +{rule.points}")
    points = min(points, cfg.complexity_cap)
    why = "; ".join(reasons) if reasons else "no complexity signals found -> 0"
    return points, why


def _fit_confidence(research: Research, strong_medium_count: int, profile: OrgProfile | None = None) -> str:
    """HIGH = all three of {HQ known, employees known, >=2 strong/medium cloud
    signals}; MEDIUM = exactly two of the three; LOW = otherwise.

    F-13: the cloud-signal leg is NECESSARY, not interchangeable with the other two.
    HQ and headcount describe a company; they say nothing about whether it has a cloud
    bill worth a call, and both can come from the model alone. Without at least two
    evidence-backed cloud signals the score is an inference about an unestablished
    subject, so it is LOW however well-known the company is. This is the one rule
    change of the F-13 fix; the score itself is untouched."""
    hq_known = bool(research.headquarters_country) and research.headquarters_country.strip().lower() != "unknown"
    employees_known = research.estimated_employees is not None
    # ">=2 strong/medium cloud signals": independent cloud-bearing evidence families at MEDIUM or
    # better (e.g. a first-party statement + a network footprint), or workload-signal categories.
    #
    # Read from `qualifying_families` - the pool cloud.py's classification ACCEPTED - and not
    # from `families_present`, which is deliberately a literal picture of everything collected,
    # including what the classification threw out. Measured: adding one cPanel network footprint
    # (MEDIUM, no provider resolved - excluded from every tier, worth zero points) moved the
    # confidence LOW -> HIGH while the cloud state stayed UNKNOWN. Evidence that was not allowed
    # to support the claim must not raise confidence in it either.
    cloud_families = sum(
        1 for fam, info in (research.cloud_usage.qualifying_families or {}).items()
        if fam in _CLOUD_BEARING_FAMILIES and fam != _EDGE_ONLY_FAMILY
        and info.get("max_strength") in ("MEDIUM", "STRONG", "DIRECT")
    )
    signals_ok = max(strong_medium_count, cloud_families) >= 2
    # Profile switch: `confidence.requires_cloud_signal: false` is for a vendor whose ICP is
    # not defined by cloud use - the signal leg then counts like the other two.
    if not signals_ok and fit_config(profile).requires_cloud_signal:
        return "LOW"
    hits = sum([hq_known, employees_known, signals_ok])
    if hits == 3:
        return "HIGH"
    return "MEDIUM" if hits == 2 else "LOW"


# --- Spend prior (relative, illustrative) - separate from the fit score above ----
# A relative prioritisation signal for the tracker, NOT an estimated cloud invoice
# and NOT part of the fit score maths above. Rough USD/employee/month multipliers
# (profile `fit.spend.usd_per_employee`) for a prototype, not measured data - internal
# only, used to rank leads into LOW/MEDIUM/HIGH/VERY_HIGH, never shown as a dollar figure.
# Tier edges (`fit.spend.tier_edges`) are in the same internal "monthly USD-equivalent"
# units (employees x USD/employee/mo); the relative tier keeps the ordering (bigger, more
# infra-heavy company -> higher tier) without implying a precision the inputs don't have.


def _industry_intensity(industry: str, cfg: FitConfig) -> tuple[str, int]:
    ind = (industry or "").lower()
    usd = cfg.spend_usd_per_employee
    if any(k in ind for k in cfg.high_spend_industries):
        return "infra-heavy", usd["infra-heavy"]
    if any(k in ind for k in cfg.low_spend_industries):
        return "low-infra", usd["low-infra"]
    return "neutral", usd["neutral"]


def _tier_for_monthly_usd(usd: float, profile: OrgProfile | None = None) -> str:
    cfg = fit_config(profile)
    for edge, tier in cfg.spend_tier_edges:
        if usd < edge:
            return tier
    return cfg.spend_tier_above


def estimate_cloud_spend(research: Research, employees_fallback: int | None = None,
                         profile: OrgProfile | None = None) -> tuple[str, str]:
    """Spend prior: a relative tier (LOW/MEDIUM/HIGH/VERY_HIGH), not an estimated
    cloud invoice - a prioritisation signal from employees x industry intensity
    alone, before any evidence is looked at (see module docstring and the design
    notes above the constants). Never part of the fit score."""
    label, per_employee = _industry_intensity(research.industry, fit_config(profile))
    employees = research.estimated_employees or employees_fallback
    if employees is None:
        return "LOW", (f"no employee estimate available, so this is a floor, not a size: "
                        f"{label} industry assumption is ${per_employee}/employee/mo")
    monthly = employees * per_employee
    tier = _tier_for_monthly_usd(monthly, profile)
    return tier, (f"~{employees} employees x ${per_employee}/employee/mo ({label} industry "
                  f"'{research.industry or 'unknown'}') internal estimate -> {tier}")


def fit_threshold(profile: OrgProfile | None = None) -> int:
    """The sales_ready bar. The FIT_THRESHOLD env var, when set, wins over the profile."""
    from .config import settings
    if getattr(settings, "fit_threshold_from_env", True):
        return settings.fit_threshold
    return fit_config(profile).threshold


def confidence_qualifies(confidence: str, profile: OrgProfile | None = None) -> bool:
    """True when `confidence` reaches the profile's `sales_ready_min_confidence`."""
    return (CONFIDENCE_ORDER.get(confidence, -1)
            >= CONFIDENCE_ORDER[fit_config(profile).sales_ready_min_confidence])


def score_fit(lead: Lead, research: Research, profile: OrgProfile | None = None) -> FitResult:
    profile = profile or active_profile()
    cfg = fit_config(profile)
    employees, employees_why = employees_for_scoring(research, lead, profile)

    scale, scale_why = _scale_points(employees, profile)
    if employees_why:
        scale_why = f"{scale_why} ({employees_why})"
    cloud, cloud_why, strong_medium_count = _cloud_signal_points(research, profile)
    complexity, complexity_why = _complexity_points(research, profile)
    score = min(scale + cloud + complexity, cfg.max_score)
    confidence = _fit_confidence(research, strong_medium_count, profile)
    spend_band, spend_band_why = estimate_cloud_spend(research, employees, profile)

    reasoning = (f"scale: {scale_why}; cloud signals: {cloud_why}; "
                 f"complexity: {complexity_why}; confidence: {confidence}")
    return FitResult(score=score, confidence=confidence, scale_points=scale,
                     cloud_signal_points=cloud, complexity_points=complexity,
                     reasoning=reasoning, cloud_spend_band=spend_band, cloud_spend_reasoning=spend_band_why)
