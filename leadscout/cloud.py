"""Deterministic cloud-usage classification over the evidence every provider in
`leadscout/providers/` collected. Providers only produce
`ProviderResult`/`Evidence`; this module is the ONLY place that turns those into a
`CloudUsageAssessment` state - `fit.py` then turns that assessment into points.
See the design notes below for the five states, the exclusion rules, and dedup.
"""
from __future__ import annotations

from .models import CloudUsageAssessment, Evidence

# --- Design notes ---------------------------------------------------------------
# Five states, from strongest to weakest evidence:
# - CONFIRMED: at least one CURRENT, DIRECT `first_party_statement` or
#   `vendor_case_study` - the company (or a named cloud vendor about the company)
#   said so directly, recently, and it isn't a SaaS-dependency admission ("we
#   integrate with AWS Cost Explorer" is not "we run on AWS").
# - LIKELY: at least two independent non-edge families, with at least one item
#   among them reaching STRONG (or better).
# - POSSIBLE: one STRONG non-edge family, or at least two MEDIUM non-edge families.
# - EDGE_ONLY: the only family present is `edge_delivery` (a CDN alone says
#   nothing about the actual hosting workload behind it).
# - UNKNOWN: none of the above - this is the floor, never `NO_CLOUD`. Absence of
#   found evidence is not a claim that no cloud is used (see
#   models.CloudUsageAssessment docstring and CLAUDE.md's "unknown must not
#   collapse into valid").
#
# Historical evidence alone never reaches CONFIRMED (freshness must be
# "current"), so it caps at whatever LIKELY/POSSIBLE tier the same evidence
# would otherwise support - no separate downgrade step is needed, this falls
# directly out of the CONFIRMED gate.
#
# `scope == "saas_dependency"` evidence (see providers/trust_pages.py) is
# excluded from every tier's evidence pool entirely: it exists specifically to
# record "this company depends on a SaaS product that happens to run on a cloud
# provider" - the opposite of a hosting claim - so it must never count toward
# any state, however strong its own `strength` value looks.
#
# Only CLOUD-BEARING families can ever count toward a state at all:
# `first_party_statement`, `vendor_case_study`, `ats_hiring`,
# `engineering_footprint`, `network_footprint`, `edge_delivery`.
# `corporate_identity`, `encyclopedic`, and `policy` NEVER contribute to cloud
# state - they exist for `fit_confidence` and the Wikidata-industry bonus (see
# fit.py), not for this classification (bug fixed after Masterplast/Artizan
# reached LIKELY/POSSIBLE off `corporate_identity` + `encyclopedic` alone, with
# no cloud-bearing family present at all - the original hypothesis
# invariant: "no cloud-bearing family present -> UNKNOWN"). Within a
# cloud-bearing family, an item counts only if it names a specific provider
# (AWS, Azure, GCP, or OCI) or its family is `ats_hiring`/`engineering_footprint`
# - both of those are only ever created by their providers (see
# providers/ats.py, providers/github.py) when an infra term/role or an IaC
# provider block was already found, so their mere existence already carries an
# infra signal even without a resolved `provider` value.
#
# Evidence is deduped by (family, provider) before any counting: five ATS job
# postings all naming AWS are "AWS hiring evidence exists", not five
# independent facts.

_STRENGTH_ORDER = {"WEAK": 0, "MEDIUM": 1, "STRONG": 2, "DIRECT": 3}
_HOSTING_FAMILIES = ("first_party_statement", "vendor_case_study")
_EDGE_ONLY_FAMILY = "edge_delivery"
# Cloud-bearing channels whose "skipped" means "no credential, could never run" (capability gap).
_CAPABILITY_SKIP_CHANNELS = frozenset({"vendor"})
# Which providers can produce a cloud-bearing family AT ALL - the channels whose absence
# can make THIS assessment incomplete. Measured by reading every provider module's
# `family=` sites: ats -> ats_hiring, github -> engineering_footprint, footprint ->
# network_footprint, trust_pages -> first_party_statement, vendor -> vendor_case_study.
# gleif, hunter, headcount and wikidata only ever emit corporate_identity/encyclopedic,
# so their availability says nothing about cloud usage and must not weaken the cloud
# reason. tests/test_cloud.py asserts both halves of that split, so adding a provider
# that emits a cloud-bearing family without listing it here fails the suite.
_CLOUD_BEARING_PROVIDERS = frozenset({"ats", "github", "footprint", "trust_pages", "vendor"})
_CLOUD_BEARING_FAMILIES = frozenset({
    "first_party_statement", "vendor_case_study", "ats_hiring",
    "engineering_footprint", "network_footprint", "edge_delivery",
})
# ats_hiring/engineering_footprint items are only ever produced once an infra
# signal (role/tech-keyword co-occurrence, or an IaC provider block) was already
# found - see the module docstring - so they count even without a resolved provider.
# edge_delivery is also included here: its provider is typically a CDN vendor
# (Cloudflare/Fastly/Akamai/...), never one of the four named clouds, but it is
# the sole evidence EDGE_ONLY is built on - excluding it entirely would make
# EDGE_ONLY unreachable.
_INFRA_SIGNAL_FAMILIES_WITHOUT_PROVIDER = frozenset({"ats_hiring", "engineering_footprint", "edge_delivery"})


def _counts_toward_cloud_state(ev: Evidence) -> bool:
    """v0.2.2: no more hardcoded {"AWS","Azure","GCP","OCI"} set - `Evidence.provider`
    is an open string now (the infrastructure model "Main invariant"; a network/service
    fingerprint can name a provider that closed four-way enum never anticipated,
    e.g. a range match naming "Hetzner" or a RIPEstat holder). Any non-empty
    provider on a cloud-bearing family counts, exactly as a named AWS/Azure/GCP/OCI
    provider used to (PART A item 6) - the ats/engineering/edge "inherent infra
    signal without a provider" rule is unchanged."""
    if ev.family not in _CLOUD_BEARING_FAMILIES:
        return False
    if ev.provider:
        return True
    return ev.family in _INFRA_SIGNAL_FAMILIES_WITHOUT_PROVIDER


def _qualifying(evidence: list[Evidence]) -> list[Evidence]:
    """Evidence that may count toward any cloud-usage state at all: a SaaS
    dependency is explicitly not a hosting signal, however strong its own
    `strength` field is; and only cloud-bearing families with a named provider (or
    ats/engineering-footprint's inherent infra signal) qualify at all - see the
    module docstring."""
    return [ev for ev in evidence if ev.scope != "saas_dependency" and _counts_toward_cloud_state(ev)]


def _dedupe_by_family_provider(evidence: list[Evidence]) -> list[Evidence]:
    """One representative Evidence per (family, provider) pair - the strongest
    (by strength, then "current" over "historical") of that group, so five ATS
    postings naming AWS count once, not five times."""
    best: dict[tuple[str, str | None], Evidence] = {}
    for ev in evidence:
        key = (ev.family, ev.provider)
        current = best.get(key)
        if current is None:
            best[key] = ev
            continue
        ev_rank = (_STRENGTH_ORDER.get(ev.strength, -1), ev.freshness == "current")
        cur_rank = (_STRENGTH_ORDER.get(current.strength, -1), current.freshness == "current")
        if ev_rank > cur_rank:
            best[key] = ev
    return list(best.values())


def _families_present(evidence: list[Evidence]) -> dict:
    """family -> {"count": raw evidence count, "max_strength": highest strength
    seen} - computed over ALL evidence (including saas_dependency-scoped items and
    duplicates), so this stays a faithful, literal picture of what was collected,
    independent of the dedup/exclusion the state classification below applies."""
    out: dict[str, dict] = {}
    for ev in evidence:
        entry = out.setdefault(ev.family, {"count": 0, "max_strength": ev.strength})
        entry["count"] += 1
        if _STRENGTH_ORDER.get(ev.strength, -1) > _STRENGTH_ORDER.get(entry["max_strength"], -1):
            entry["max_strength"] = ev.strength
    return out


def _classify(deduped: list[Evidence]) -> tuple[str, list[str]]:
    """Returns (state, reasoning lines) for one already-deduped, already-filtered
    evidence pool (either the whole assessment or one provider's slice)."""
    reasoning: list[str] = []

    confirmed = [
        ev for ev in deduped
        if ev.family in _HOSTING_FAMILIES and ev.strength == "DIRECT" and ev.freshness == "current"
    ]
    if confirmed:
        ev = confirmed[0]
        reasoning.append(
            f"{ev.source_type} evidence ({ev.id}) -> {ev.family} -> DIRECT, current -> CONFIRMED")
        return "CONFIRMED", reasoning

    non_edge = [ev for ev in deduped if ev.family != _EDGE_ONLY_FAMILY]
    non_edge_families = sorted({ev.family for ev in non_edge})
    strong_or_better = [ev for ev in non_edge if _STRENGTH_ORDER.get(ev.strength, -1) >= _STRENGTH_ORDER["STRONG"]]
    medium_or_better = [ev for ev in non_edge if _STRENGTH_ORDER.get(ev.strength, -1) >= _STRENGTH_ORDER["MEDIUM"]]
    medium_families = sorted({ev.family for ev in medium_or_better})

    if len(non_edge_families) >= 2 and strong_or_better:
        ev = strong_or_better[0]
        reasoning.append(
            f"{len(non_edge_families)} independent non-edge families {non_edge_families} "
            f"including {ev.family} at {ev.strength} -> LIKELY")
        return "LIKELY", reasoning

    if strong_or_better:
        ev = strong_or_better[0]
        reasoning.append(f"{ev.source_type} evidence ({ev.id}) -> {ev.family} -> {ev.strength} -> POSSIBLE")
        return "POSSIBLE", reasoning
    if len(medium_families) >= 2:
        reasoning.append(f"{len(medium_families)} independent MEDIUM+ non-edge families {medium_families} -> POSSIBLE")
        return "POSSIBLE", reasoning

    # Fallback, not an exact-match check: edge_delivery evidence exists and nothing
    # else (however much other weak/junk non-edge evidence also exists) reached
    # POSSIBLE or better. An exact "families_seen == {edge_delivery}" equality check
    # would let unrelated weak evidence silently downgrade EDGE_ONLY to UNKNOWN when
    # more evidence is added - a real monotonicity bug this fixed (see
    # test_properties.py's test_adding_more_evidence_never_lowers_cloud_state...).
    if any(ev.family == _EDGE_ONLY_FAMILY for ev in deduped):
        reasoning.append("edge_delivery evidence found, nothing else reaches POSSIBLE -> EDGE_ONLY")
        return "EDGE_ONLY", reasoning

    reasoning.append("no evidence meets the CONFIRMED/LIKELY/POSSIBLE/EDGE_ONLY thresholds -> UNKNOWN")
    return "UNKNOWN", reasoning


counts_toward_cloud_state = _counts_toward_cloud_state


def _boundary_for(state: str, deduped: list[Evidence]) -> str:
    families_seen = {ev.family for ev in deduped}
    if state == "CONFIRMED":
        return ("a current first-party or vendor statement names the provider; this does not establish "
                "that all of the company's workloads run there, nor the size of the bill")
    if state in ("LIKELY", "POSSIBLE"):
        return ("no current first-party source (own trust/status page or a named cloud vendor case study) "
                "names a specific provider as this company's own hosting")
    if state == "EDGE_ONLY":
        return "no evidence beyond CDN/edge delivery - the actual hosting workload behind it is unestablished"
    if not deduped:
        return "no cloud-usage evidence of any kind was found or fetched successfully"
    return f"evidence exists ({sorted(families_seen)}) but none of it reaches the POSSIBLE threshold"


def assess_cloud_usage(
    evidence: list[Evidence],
    provider_results: list | None = None,
    domain_resolved: bool = True,
) -> CloudUsageAssessment:
    """Deterministic classification over every Evidence item any
    provider collected for one lead. Never raises, never returns NO_CLOUD.

    v0.2.2 additions (PART A item 6, scope cut): `provider_results`
    (each lead's `ProviderResult` list) fills `missing_channels` from any
    `degraded` provider; `domain_resolved` (the caller's best signal that the
    submitted website's domain actually resolves - `research.py` passes
    `not r.website_text.startswith("[unreachable: ConnectError]")`, a heuristic,
    not a literal NXDOMAIN check - NOT MEASURED precisely) drives
    `unknown_reason=NO_RESOLVABLE_DOMAIN`. A missing channel means the assessment
    might be incomplete, so it is never allowed to produce the confident-sounding
    `NO_PUBLIC_CLOUD_EVIDENCE` reason - `INSUFFICIENT_EVIDENCE` instead.

    Only `status == "degraded"` counts as missing, deliberately NOT "skipped": in
    the existing provider code, "skipped" is overloaded to mean two different
    things - a genuine capability gap (vendor.py/hunter.py: no API key at all, the
    channel can never run) and a provider that ran completely and simply found
    nothing (ats.py/trust_pages.py: "no ATS board link found" is a normal, complete
    result, not a missing channel). Treating every "skipped" as missing would make
    almost every real lead read as INSUFFICIENT_EVIDENCE instead of the more
    accurate NO_PUBLIC_CLOUD_EVIDENCE (measured against the E0 corpus leads - see
    the PART A report). Distinguishing the two "skipped" meanings would need a
    provider-level API change outside this part's scope - NOT MEASURED further."""
    qualifying = _qualifying(evidence)
    deduped = _dedupe_by_family_provider(qualifying)

    state, reasoning = _classify(deduped)
    boundary = _boundary_for(state, deduped)

    # "degraded" is always missing. "skipped" is overloaded: for ats/trust_pages/github it means the
    # provider ran and found nothing (a complete result), but for `vendor` (no BRAVE_API_KEY) it means a
    # cloud-bearing channel could never run - and then "no public-cloud evidence" would overstate.
    #
    # Availability is scoped to the DECISION, not to the run. This used to count any
    # degraded provider, so a rate-limited GLEIF - an identity lookup that cannot produce
    # a cloud signal at all - turned an otherwise clean NO_PUBLIC_CLOUD_EVIDENCE into
    # INSUFFICIENT_EVIDENCE (measured; with the GLEIF non-200 fix widening what counts as
    # degraded, ordinary 4xx noise would have mislabelled a large share of leads). A
    # decision may only see the outages of the channels that feed IT.
    missing_channels = sorted({
        pr.provider_name for pr in (provider_results or [])
        if pr.provider_name in _CLOUD_BEARING_PROVIDERS
        and (getattr(pr, "status", "ok") == "degraded"
             or (getattr(pr, "status", "ok") == "skipped" and pr.provider_name in _CAPABILITY_SKIP_CHANNELS))
    })
    unknown_reason = None
    if state == "UNKNOWN":
        if not domain_resolved:
            unknown_reason = "NO_RESOLVABLE_DOMAIN"
        elif missing_channels:
            unknown_reason = "INSUFFICIENT_EVIDENCE"
        else:
            unknown_reason = "NO_PUBLIC_CLOUD_EVIDENCE"

    # Cloud providers only (not CDN/edge vendors) - the non-edge-family slice of
    # the deduped, qualifying pool, same evidence the LIKELY/POSSIBLE rules above
    # already used.
    cloud_providers = sorted({
        ev.provider for ev in deduped if ev.provider and ev.family != _EDGE_ONLY_FAMILY
    })

    providers_present = sorted({ev.provider for ev in qualifying if ev.provider}, key=str)
    provider_states: list[dict] = []
    for provider in providers_present:
        provider_evidence = [ev for ev in qualifying if ev.provider == provider]
        provider_deduped = _dedupe_by_family_provider(provider_evidence)
        provider_state, _ = _classify(provider_deduped)
        # Aggregate fields (item 6): computed entirely from fields every provider
        # already sets on its Evidence (`observed_at`, `family`) - no new data
        # collected. `observation_count` is the raw (pre-dedupe) count - "how many
        # times was this provider seen at all"; `independent_family_count` is
        # post-dedupe distinct families - the same "independence" cloud.py's own
        # LIKELY rule requires (>=2 independent non-edge families), so this number
        # answers "how much of the way to LIKELY does this provider alone get".
        observed_ats = sorted(ev.observed_at for ev in provider_evidence if ev.observed_at)
        provider_states.append({
            "provider": provider,
            "state": provider_state,
            "evidence_ids": sorted(ev.id for ev in provider_evidence if ev.id),
            "first_seen_at": observed_ats[0] if observed_ats else None,
            "last_seen_at": observed_ats[-1] if observed_ats else None,
            "observation_count": len(provider_evidence),
            "independent_family_count": len({ev.family for ev in provider_deduped}),
        })

    return CloudUsageAssessment(
        state=state,
        providers=provider_states,
        families_present=_families_present(evidence),
        # What the classification ACCEPTED, over the same deduped, qualifying pool the
        # tier rules above used. fit.py's confidence reads this rather than the literal
        # picture, so an item excluded here cannot raise confidence in a cloud claim it
        # was not allowed to support.
        qualifying_families=_families_present(deduped),
        reasoning=reasoning,
        boundary=boundary,
        cloud_providers=cloud_providers,
        unknown_reason=unknown_reason,
        missing_channels=missing_channels,
    )
