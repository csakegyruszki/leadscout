"""Data shapes that flow through the pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, get_args

from pydantic import BaseModel, Field, field_validator


@dataclass
class Lead:
    name: str
    email: str
    company: str
    website: str
    # Extended fields (optional). Justification in README: company-size band is the
    # single strongest fit signal and costs the applicant one click.
    job_title: str = ""
    company_size_band: str = ""  # e.g. "1-10", "11-50", "51-200", "201-1000", "1000+"


Strength = Literal["DIRECT", "STRONG", "MEDIUM", "WEAK"]
Family = Literal[
    "first_party_statement", "vendor_case_study", "ats_hiring", "engineering_footprint",
    "network_footprint", "edge_delivery", "corporate_identity", "encyclopedic", "policy",
]
# v0.2.2: CloudProvider is an OPEN string, not a closed enum any more (the infrastructure model
# "Main invariant" - there is no provider list; a network/service fingerprint can name a
# provider the four-way AWS/Azure/GCP/OCI enum never anticipated, e.g. "Hetzner",
# "OpenStack"). Validated at construction (non-empty, stripped), not enum-membership.
CloudProvider = str
Scope = Literal["workload", "storage", "edge", "saas_dependency", "unknown"]
Freshness = Literal["current", "historical", "unknown"]

_STRENGTH_VALUES: set[str] = set(get_args(Strength))
_FAMILY_VALUES: set[str] = set(get_args(Family))
_SCOPE_VALUES: set[str] = set(get_args(Scope))
_FRESHNESS_VALUES: set[str] = set(get_args(Freshness))

# --- v0.2.2 infrastructure-model enums (the infrastructure model), SCOPE-CUT to PART A's
# reduced surface (scope cut): only the range matcher's own
# `network_role` and the lightweight `InfrastructureFootprint.category` remain as
# closed enums here; the fuller asset-relationship/control-model/hosting-model axis
# set from the original PART A draft is deferred to a later part.
NetworkRole = Literal["EDGE", "PUBLIC_CLOUD_COMPUTE", "PUBLIC_CLOUD_SERVICE"]
FootprintCategory = Literal["PUBLIC_CLOUD", "PRIVATE_CLOUD", "MANAGED_HOSTING", "EDGE", "UNKNOWN"]

_NETWORK_ROLE_VALUES: set[str] = set(get_args(NetworkRole))
_FOOTPRINT_CATEGORY_VALUES: set[str] = set(get_args(FootprintCategory))


@dataclass
class Evidence:
    """One piece of chain-of-custody-tracked evidence, mirrored into the provtrail
    ledger by provenance.ProvenanceRun.record(). Never carries the lead's contact
    name or email - only company-level facts.

    `family` and `strength` are what `cloud.py`'s assessment reads; `source_type`
    identifies which fetch mechanism produced it (kept distinct from `family`, which
    is the *evidentiary category* used for cloud-usage classification regardless of
    mechanism - e.g. both a trust-center sentence and a status-page component name
    can be `first_party_statement`).

    The five enum-shaped fields below are `Literal[...]` types, validated in
    `__post_init__` - an out-of-enum value raises immediately at construction
    instead of silently drifting into `cloud.py`'s classification (see
    REVIEW-6bA-verified.md #3). `family` has no default: every piece of evidence,
    including plain website/wikipedia/wikidata/opensanctions/policy text that isn't
    itself a cloud-usage signal, must be classified into one of the nine buckets -
    there is no `"unknown"` escape hatch any more. Interpretation call (not fully
    pinned down by the family enum, which has no general/company-facts
    bucket): plain website/website_subpage/opensanctions text is filed under
    `corporate_identity` (general facts about the corporate entity, same bucket as
    GLEIF/RDAP), and wikipedia/wikidata under `encyclopedic` - see research.py,
    provenance.py and sanctions.py call sites.
    """
    id: str  # "ev-001", "ev-002", ...
    # "website"|"website_subpage"|"wikipedia"|"wikidata"|"gleif"|"ats"|"footprint"|
    # "opensanctions"|"policy"
    source_type: str
    url: str
    observed_at: str  # ISO UTC
    content_sha256: str
    strength: Strength
    snippet: str  # <=200 chars
    snapshot_path: str  # relative to repo root
    family: Family
    provider: CloudProvider | None = None
    scope: Scope = "unknown"
    freshness: Freshness = "unknown"
    source_published_at: str | None = None  # nullable, ISO date if known
    origin: str = ""  # e.g. "provider:wikidata" - which code path produced this item
    # Fix #3 (v0.2 PART B): for "website"/"website_subpage" items, the trafilatura
    # main-text extraction of the page - kept SEPARATE from `content_sha256`/
    # `snapshot_path`, which (as of this fix) hash and store the page's RAW response
    # bytes, not this derived text. Before this fix, `content_sha256` silently hashed
    # the extracted text instead of what was actually fetched, so a snapshot never
    # proved the byte-for-byte page content it claimed to. Empty for every other
    # source_type (nothing here redefines what THEIR content_sha256 hashes).
    derived_text: str = ""

    def __post_init__(self) -> None:
        if self.strength not in _STRENGTH_VALUES:
            raise ValueError(f"Evidence.strength must be one of {sorted(_STRENGTH_VALUES)}, got {self.strength!r}")
        if self.family not in _FAMILY_VALUES:
            raise ValueError(f"Evidence.family must be one of {sorted(_FAMILY_VALUES)}, got {self.family!r}")
        if self.provider is not None:
            if not isinstance(self.provider, str) or self.provider != self.provider.strip() or not self.provider:
                raise ValueError(
                    f"Evidence.provider must be a non-empty, stripped string or None, got {self.provider!r}")
        if self.scope not in _SCOPE_VALUES:
            raise ValueError(f"Evidence.scope must be one of {sorted(_SCOPE_VALUES)}, got {self.scope!r}")
        if self.freshness not in _FRESHNESS_VALUES:
            raise ValueError(f"Evidence.freshness must be one of {sorted(_FRESHNESS_VALUES)}, got {self.freshness!r}")


@dataclass
class ProviderResult:
    """What one evidence provider (leadscout/providers/*.py) returns. A failed
    provider is `degraded` and yields NO evidence - never negative evidence (absence
    of a signal is not evidence the signal doesn't exist elsewhere)."""
    provider_name: str
    status: str = "ok"  # "ok"|"degraded"|"skipped"
    evidence: list[Evidence] = field(default_factory=list)
    note: str = ""
    calls: int = 0
    latency_ms: int = 0


@dataclass
class RangeMatch:
    """One IP -> published-cloud-range hit (the infrastructure model "Detection order and
    the range matcher"). Never constructed from a bare network-holder guess - only
    from an authoritative provider feed (AWS/GCP/OCI ip-range JSON, Azure Service
    Tags) or an equivalent CNAME-suffix/InternetDB corroboration in `infra.py`.
    `network_role` is EDGE for CDN/edge services (AWS CLOUDFRONT/GLOBALACCELERATOR/
    ROUTE53, Azure AzureFrontDoor.*), PUBLIC_CLOUD_COMPUTE/PUBLIC_CLOUD_SERVICE
    otherwise. A feed's generic superset entry (AWS "AMAZON") must never override a
    more specific service entry already matched - see `ranges.py`."""
    provider: str
    service: str
    region: str | None
    source_feed: str  # e.g. "aws-ip-ranges", "azure-service-tags"
    source_version: str | None  # feed's own createDate/changeNumber, when it has one
    network_role: str  # NetworkRole

    def __post_init__(self) -> None:
        if self.network_role not in _NETWORK_ROLE_VALUES:
            raise ValueError(f"RangeMatch.network_role must be one of {sorted(_NETWORK_ROLE_VALUES)}, "
                             f"got {self.network_role!r}")


@dataclass
class InfrastructureFootprint:
    """Lightweight rule-based consolidated view (scope cut: PART A ships
    only this reduced shape, not the fuller NetworkObservation/asset-relationship/
    control-model/hosting-model axis set the infrastructure model originally sketched -
    that fuller model is deferred to a later part). Built by `leadscout/infra.py`
    from footprint.py's own per-host observations (range match, RIPEstat holder,
    cPanel trace). Never asserts more than the "Main invariant" allows: a network/
    service fingerprint establishes what is observable, never who operates/pays
    for it - `provider` is ONLY ever a range-matched hyperscaler (or a first-party
    statement) - a bare RIPEstat network holder is never "the provider" (network
    holder != operator, the infrastructure model main invariant). For a non-range IP,
    `provider` stays None and `network_holder` carries the raw RIPEstat holder
    string instead - see `infra.py`, which is also the only place that renders
    this into reasoning text, always as "address registered to X; operator not
    established", never "hosted by"/"runs on"."""
    scope: str = "unknown"  # free-text, e.g. "website", "mail", "edge", "unknown"
    category: str = "UNKNOWN"  # FootprintCategory
    provider: str | None = None  # ONLY a range-matched hyperscaler or first-party statement
    network_holder: str | None = None  # raw RIPEstat/RDAP holder string - registration, not operation
    technologies: list[str] = field(default_factory=list)
    confidence: str = "LOW"  # "HIGH"|"MEDIUM"|"LOW"
    evidence_ids: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.category not in _FOOTPRINT_CATEGORY_VALUES:
            raise ValueError(f"InfrastructureFootprint.category must be one of "
                             f"{sorted(_FOOTPRINT_CATEGORY_VALUES)}, got {self.category!r}")


@dataclass
class CloudUsageAssessment:
    """Output of `leadscout/cloud.py`'s deterministic classification over the
    evidence families collected by every provider. Never a `NO_CLOUD` value - the
    absence of found evidence is `UNKNOWN`, not a claim that no cloud is used."""
    state: str = "UNKNOWN"  # "CONFIRMED"|"LIKELY"|"POSSIBLE"|"EDGE_ONLY"|"UNKNOWN"
    providers: list[dict] = field(default_factory=list)  # [{"provider","state","evidence_ids"}] - kept for
    # backward compatibility (notify.py/tracker.py) until PART B
    families_present: dict = field(default_factory=dict)  # family -> {"count": int, "max_strength": str}
    # The same shape, but over the pool that actually COUNTED toward `state`: saas
    # dependencies, families that do not qualify, and duplicates already removed.
    # `families_present` above stays a literal picture of everything collected (the
    # tracker's Technical sheet shows it); this is what the classification accepted.
    # fit.py's confidence reads THIS one - reading the literal picture let a cPanel
    # network footprint that cloud.py had excluded still count as a cloud signal.
    qualifying_families: dict = field(default_factory=dict)
    reasoning: list[str] = field(default_factory=list)  # observation -> family -> strength -> inference, one per line
    boundary: str = ""  # what is NOT established
    # v0.2.2 additions (scope cut - only these three, not the fuller
    # deployment_models/assessment_coverage set the original PART A draft had):
    cloud_providers: list[str] = field(default_factory=list)  # cloud providers only, not technologies/deps
    unknown_reason: str | None = None  # "NO_RESOLVABLE_DOMAIN"|"NO_PUBLIC_CLOUD_EVIDENCE"|"INSUFFICIENT_EVIDENCE"|None
    missing_channels: list[str] = field(default_factory=list)  # provider names that were degraded/skipped


@dataclass
class Research:
    website_text: str = ""
    website_ok: bool = False
    wikipedia_summary: str = ""
    wikipedia_url: str = ""
    sources: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    provider_results: list[ProviderResult] = field(default_factory=list)
    cloud_usage: CloudUsageAssessment = field(default_factory=CloudUsageAssessment)
    infrastructure: list[InfrastructureFootprint] = field(default_factory=list)
    # LLM-extracted facts
    summary: str = ""
    industry: str = ""
    headquarters_country: str = ""
    # Which source resolved `headquarters_country`: "llm" (the LLM's own extraction,
    # the normal case), or a structured-source fallback used when the LLM came back
    # empty/unknown - "gleif" (accepted GLEIF record's legal-address country),
    # "wikidata" (a website-matched Wikidata entity's P17 country), or "website_tld"
    # (the submitted website's own country-code TLD, the last and weakest rung of
    # the waterfall - a generic TLD like .com/.io gives no country signal at all
    # and never sets this). See research.py's HQ-fallback step
    # and tracker.py's Technical sheet.
    hq_source: str = ""  # "llm"|"gleif"|"wikidata"|"website_tld"
    # Set to "low" only when hq_source == "website_tld" - a ccTLD is a much weaker
    # HQ signal than an LLM extraction or an accepted GLEIF/Wikidata record (a
    # company can easily register a ccTLD domain outside its actual HQ country).
    # compliance.py's prescreen treats a "low"-confidence HQ as established for the
    # "clear requires HQ" safety net, but does NOT treat it as an independent basis
    # for the HQ-based sanctioned-jurisdiction check - the pre-existing website-TLD
    # sanctions rule already covers exactly that case without double-counting the
    # same weak signal at a higher (score 100) confidence than it deserves.
    hq_confidence: str = ""  # ""|"low"
    registered_jurisdiction: str = ""  # from GLEIF legal-entity address; distinct from operational HQ
    estimated_employees: int | None = None
    employees_source: str = ""  # where estimated_employees came from (website/LLM, Diffbot KG, web search)
    employees_hint: str = ""  # unverified web-search headcount, shown to the rep, never scored
    tech_signals: list[str] = field(default_factory=list)
    confidence: str = "low"
    evidence_ids: list[str] = field(default_factory=list)  # from ResearchFacts.evidence_ids
    uncertainties: list[str] = field(default_factory=list)  # from ResearchFacts.uncertainties
    # Wikidata's structured P452 (industry) values, when a website-matched entity was
    # found - kept separate from the free-text `industry` (LLM-extracted) above so
    # fit.py can treat it as its own evidence-bound signal (Part 4 addendum,
    # 2026-09-19), not a keyword match against prose. Paired with the id of the
    # wikidata Evidence item that carries it, for citation.
    wikidata_industries: list[str] = field(default_factory=list)
    wikidata_evidence_id: str | None = None
    # Hunter.io email-verifier/domain-search classification (providers/hunter.py) -
    # sales-rep context ONLY, e.g. "Hunter: accept_all, score 62, MX ok". Never
    # consulted by fit.py/cloud.py/compliance.py: "unverified" covers both a
    # skipped run (no HUNTER_API_KEY) and a degraded one (HTTP error/quota), so an
    # absent Hunter result reads the same as "we don't know", never as "valid".
    contact_quality: str = "unverified"  # "valid"|"risky"|"invalid"|"unverified"
    contact_quality_note: str = ""
    # The organisation the SUBMITTED DOMAIN belongs to, anchored on the domain instead
    # of the typed company name (providers/hunter.py). Identity only: it answers "who
    # applied", never "how big" or "where" - those fields of the same lookup measured
    # unreliable. compliance.py DOES consult this one (unlike contact_quality above)
    # to tell a local operating entity from the group a bare company name resolves to;
    # fit.py and cloud.py still consult neither.
    domain_entity_name: str = ""


@dataclass
class ComplianceResult:
    flagged: bool
    status: str  # "clear" | "review" | "blocked"
    matches: list[dict] = field(default_factory=list)  # {"kind","term","score","reason",...}
    reasoning: str = ""
    # OpenSanctions API result, kept separate from the LLM verdict above so the
    # tracker/email can show what the real screening API said independent of it.
    sanctions_status: str = ""  # "blocked_evidence"|"review_evidence"|"info"|"none"|"skipped"
    sanctions_hits: list[dict] = field(default_factory=list)


@dataclass
class FitResult:
    score: int  # fit_score, 0-100, sum of the three components below - no confidence cap
    confidence: str = "LOW"  # "HIGH"|"MEDIUM"|"LOW" - see fit._fit_confidence
    scale_points: int = 0  # 0-30, from employee count
    cloud_signal_points: int = 0  # 0-55, from the CloudUsageAssessment state + workload-intensity bonus
    complexity_points: int = 0  # 0-15, product/market/regulatory/deployment complexity
    reasoning: str = ""
    # "LOW"|"MEDIUM"|"HIGH"|"VERY_HIGH" - a relative prioritisation signal
    # (headcount x industry), not an estimated cloud invoice; see fit.estimate_cloud_spend.
    cloud_spend_band: str = ""
    cloud_spend_reasoning: str = ""


@dataclass
class LeadOutcome:
    lead: Lead
    research: Research
    compliance: ComplianceResult
    fit: FitResult
    sales_ready: bool
    # Per-lead LLM telemetry, drained from llm.telemetry by pipeline.process_lead.
    llm_calls: int = 0
    llm_cost_usd: float = 0.0
    llm_latency_ms: int = 0
    llm_models: str = ""  # distinct models actually used, e.g. "deepseek/deepseek-v4-flash-0731:free"
    llm_hops: int = 0  # total fallback-chain hops across all calls for this lead
    llm_profile: str = ""  # "demo" | "batch" - which LEADSCOUT_PROFILE chain was used
    # Provenance ledger summary for this lead's run (see provenance.ProvenanceRun).
    run_id: str = ""
    evidence_count: int = 0
    provenance_status: str = ""  # "ok" | "degraded"
    provenance_head: str = ""  # "<seq>:<hash prefix>" or "none"
    # F-15: which run produced this outcome, and which runtime source that run started
    # with - stamped from the immutable RunContext created at run start (see
    # leadscout/runtime_identity.py), so the proof manifest is no longer the only
    # witness to "this artefact came from this code":
    # {"id": "prf-...", "source_fingerprint": "...", "started_at": "..."}.
    proof_run: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# --- LLM output schemas (pydantic) -------------------------------------------------
# Validated once per call; llm.ask_model retries once with the validation error fed
# back to the model before giving up. Keeps a wrong-shaped JSON reply from silently
# corrupting the tracker instead of failing loudly.

class ResearchFacts(BaseModel):
    summary: str = ""
    industry: str = ""
    headquarters_country: str = "unknown"
    estimated_employees: int | None = None
    tech_signals: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "low"
    # Which Evidence.id(s) (see models.Evidence) support these facts, and what's
    # still unclear - both requested directly in the research prompt.
    evidence_ids: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)

    @field_validator("estimated_employees", mode="before")
    @classmethod
    def _coerce_employees(cls, v):
        """Models sometimes answer "500" or 500.0; a non-positive count is not a count."""
        if v is None or v == "":
            return None
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return int(v) if v > 0 else None


class ComplianceVerdict(BaseModel):
    status: Literal["clear", "review", "blocked"] = "review"
    flagged: bool = True
    matches: list[dict] = Field(default_factory=list)
    reasoning: str = ""
