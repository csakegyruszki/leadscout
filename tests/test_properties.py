"""Hypothesis property tests, offline. Each one caught (or now guards against) a real
edge case rather than restating what the example-based tests already cover:

- the mutated-competitor-name property is the flip side of the weak-name fix below -
  a real identity must survive suffix/punctuation/case/order noise;
- the generic-word-only property is what caught the "cloud" (score 100) bug fixed in
  compliance.py's `_is_weak_name` guard;
- the weak-query-never-blocks property is the sanctions.py equivalent, already
  motivated by the measured "CT Inc"/"Cloud Solutions Ltd" false positives;
- the fit-score-bounds and monotonic-band properties guard the arithmetic invariants
  a maintainer would expect to just always hold, for any input, not only the examples
  the unit tests happened to pick.
"""
from __future__ import annotations

import re

from hypothesis import given, settings
from hypothesis import strategies as st

from leadscout.cloud import assess_cloud_usage
from leadscout.compliance import _COMPETITORS, prescreen
from leadscout.fit import _tier_for_monthly_usd, score_fit
from leadscout.models import Evidence, Lead, ProviderResult, Research
from leadscout.sanctions import classify

_SUFFIXES = ("", "Inc", "Inc.", "Ltd", "Ltd.", "GmbH", "Kft", "LLC", "Co", "Co.")
_SEPARATORS = ("", "-", " ", ".")
_PADDING = ("", " ", "  ")
_CASE_MODES = ("lower", "upper", "title", "asis")
_GENERIC_WORDS = ("cloud", "solutions", "systems", "services", "group", "tech")


def _tokens_for(competitor_name: str) -> list[str]:
    """Space-separated words as written, e.g. "SpendWise Cloud" -> ["SpendWise",
    "Cloud"] - NOT split further on camelCase. Splitting "SpendWise" into "Spend" +
    "Wise" for permutation would let a mutation interleave another word between
    them ("Spend Cloud Wise"), which garbles the name at the character level rather
    than reordering it - not a realistic formatting variant, and not something
    prescreen()'s token-level fuzzy matching is meant to survive.
    """
    stripped = re.sub(r"\b(Inc|Incorporated|Ltd|Limited|LLC|Co|Corp)\.?\s*$", "", competitor_name).strip()
    return stripped.split()


@st.composite
def _mutated_competitor_name(draw):
    comp = draw(st.sampled_from(_COMPETITORS))
    tokens = draw(st.permutations(_tokens_for(comp["name"])))
    sep = draw(st.sampled_from(_SEPARATORS))
    case_mode = draw(st.sampled_from(_CASE_MODES))
    suffix = draw(st.sampled_from(_SUFFIXES))
    pad = draw(st.sampled_from(_PADDING))

    joined = sep.join(tokens)
    if case_mode == "lower":
        joined = joined.lower()
    elif case_mode == "upper":
        joined = joined.upper()
    elif case_mode == "title":
        joined = joined.title()
    mutated = f"{pad}{joined}{(' ' + suffix) if suffix else ''}{pad}"
    return comp, mutated


@settings(max_examples=200, deadline=None)
@given(_mutated_competitor_name())
def test_prescreen_never_loses_a_mutated_competitor_identity(pair):
    comp, mutated_name = pair
    hits = prescreen(Lead("t", "t@x.com", mutated_name, "https://example.com"), Research())
    strong_hit = any(h["term"] == comp["name"] and h["score"] >= 90 for h in hits)
    abbrev_hit = any(h["term"] == comp["name"] and h["origin"] == "prescreen:abbreviation" for h in hits)
    assert strong_hit or abbrev_hit, f"lost identity for mutated {mutated_name!r} (from {comp['name']!r})"


_generic_names = st.lists(st.sampled_from(_GENERIC_WORDS), min_size=1, max_size=4).map(" ".join)


@settings(max_examples=200, deadline=None)
@given(_generic_names)
def test_prescreen_never_confidently_matches_a_purely_generic_name(name):
    hits = prescreen(Lead("t", "t@x.com", name, "https://example.com"), Research())
    assert all(h["score"] < 90 for h in hits if h["kind"] == "competitor")


# --- classify(): a weak query name must never reach blocked_evidence --------------

_topics = st.lists(
    st.sampled_from(["sanction", "export.control", "debarment", "corp.public", "reg.action", "crime.fraud"]),
    max_size=4,
)
_weak_query_names = st.sampled_from(["CT", "co", "cloud", "solutions", "x", "ab", "tech solutions", "cloud group"])


@st.composite
def _sanctions_hit(draw):
    return {
        "caption": draw(st.text(min_size=1, max_size=40)),
        "score": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        "match": draw(st.booleans()),
        "schema": draw(st.sampled_from(["Company", "Organization", "Person"])),
        "datasets": [],
        "properties": {"topics": draw(_topics)},
    }


@settings(max_examples=200, deadline=None)
@given(_weak_query_names, st.lists(_sanctions_hit(), max_size=5))
def test_classify_weak_query_never_reaches_blocked_evidence(query_name, hits):
    assert classify(hits, query_name) != "blocked_evidence"


# --- fit.score_fit: bounds and confidence enum always hold ------------------------

_employees = st.one_of(st.none(), st.integers(min_value=0, max_value=10_000_000))
_industries = st.sampled_from(["SaaS", "bakery", "fintech", "manufacturing", "", "construction", "AI"])
_signals = st.lists(st.sampled_from(
    ["aws (ev-001)", "Kubernetes (ev-002)", "GPU workload (ev-003)", "petabyte scale (ev-004)",
     "multi-cloud (ev-005)", "terraform", "cloud", "software", "API", "real-time", "unrelated widget"]),
    max_size=6)


@settings(max_examples=200, deadline=None)
@given(_employees, _industries, _signals)
def test_score_fit_score_in_bounds_and_confidence_is_a_valid_enum(employees, industry, signals):
    research = Research(estimated_employees=employees, industry=industry, tech_signals=signals)
    f = score_fit(Lead("t", "t@x.com", "X", "https://x.com"), research)
    assert 0 <= f.score <= 100
    assert f.confidence in {"HIGH", "MEDIUM", "LOW"}


_usd = st.floats(min_value=0, max_value=5_000_000, allow_nan=False, allow_infinity=False)


@settings(max_examples=200, deadline=None)
@given(_usd, _usd)
def test_tier_for_monthly_usd_is_monotonic(a, b):
    lo, hi = sorted((a, b))
    order = ("LOW", "MEDIUM", "HIGH", "VERY_HIGH")
    assert order.index(_tier_for_monthly_usd(lo)) <= order.index(_tier_for_monthly_usd(hi))


# --- cloud.assess_cloud_usage / fit.score_fit invariants -----------
# `cloud_signal_points` reads CloudUsageAssessment.state, not a keyword match against
# tech_signals/website_text any more - these properties hold over randomly generated
# Evidence, not just the fixed examples in test_cloud.py.

_STATE_ORDER = ("UNKNOWN", "EDGE_ONLY", "POSSIBLE", "LIKELY", "CONFIRMED")
_FAMILY_VALUES = (
    "first_party_statement", "vendor_case_study", "ats_hiring", "engineering_footprint",
    "network_footprint", "edge_delivery", "corporate_identity", "encyclopedic", "policy",
)
_EVIDENCE_STRENGTHS = ("DIRECT", "STRONG", "MEDIUM", "WEAK")
_EVIDENCE_PROVIDERS = (None, "AWS", "Azure", "GCP", "OCI", "Cloudflare", "other")
_EVIDENCE_FRESHNESS = ("current", "historical", "unknown")
_EVIDENCE_SCOPES = ("workload", "storage", "edge", "saas_dependency", "unknown")

_evidence_tuples = st.lists(
    st.tuples(
        st.sampled_from(_FAMILY_VALUES), st.sampled_from(_EVIDENCE_STRENGTHS),
        st.sampled_from(_EVIDENCE_PROVIDERS), st.sampled_from(_EVIDENCE_FRESHNESS),
        st.sampled_from(_EVIDENCE_SCOPES),
    ),
    max_size=8,
)


def _build_evidence(tuples) -> list[Evidence]:
    return [
        Evidence(
            id=f"ev-{i:03d}", source_type="test", url="https://example.com",
            observed_at="2026-09-19T00:00:00+00:00", content_sha256="a" * 64,
            strength=strength, snippet="s", snapshot_path="out/x.txt",
            family=family, provider=provider, freshness=freshness, scope=scope,
        )
        for i, (family, strength, provider, freshness, scope) in enumerate(tuples)
    ]


def _score_for(evidence: list[Evidence]) -> tuple[int, str]:
    """(fit score, cloud state) for a Research built from this evidence alone."""
    research = Research(evidence=evidence, cloud_usage=assess_cloud_usage(evidence))
    f = score_fit(Lead("t", "t@x.com", "X", "https://x.com"), research)
    return f.score, research.cloud_usage.state


@settings(max_examples=200, deadline=None)
@given(_evidence_tuples)
def test_duplicating_a_family_never_raises_cloud_state_or_fit_score(tuples):
    evidence = _build_evidence(tuples)
    score_before, state_before = _score_for(evidence)
    duplicated = evidence + ([evidence[0]] if evidence else [])
    score_after, state_after = _score_for(duplicated)
    assert _STATE_ORDER.index(state_after) == _STATE_ORDER.index(state_before)
    assert score_after == score_before


def test_removing_all_cloud_evidence_gives_unknown_and_zero_cloud_points():
    research = Research(evidence=[], cloud_usage=assess_cloud_usage([]))
    f = score_fit(Lead("t", "t@x.com", "X", "https://x.com"), research)
    assert research.cloud_usage.state == "UNKNOWN"
    assert f.cloud_signal_points == 0


def test_edge_only_evidence_gives_edge_only_state():
    ev = _build_evidence([("edge_delivery", "WEAK", "Cloudflare", "current", "edge")])
    assessment = assess_cloud_usage(ev)
    assert assessment.state == "EDGE_ONLY"


def test_one_current_direct_first_party_statement_gives_confirmed():
    ev = _build_evidence([("first_party_statement", "DIRECT", "AWS", "current", "workload")])
    assessment = assess_cloud_usage(ev)
    assert assessment.state == "CONFIRMED"


@settings(max_examples=200, deadline=None)
@given(_evidence_tuples)
def test_historical_only_direct_never_reaches_confirmed(tuples):
    """Forcing every DIRECT item to "historical" must never leave CONFIRMED
    reachable - it caps at whatever LIKELY/POSSIBLE tier the same evidence
    otherwise supports."""
    forced = [(family, strength, provider, "historical" if strength == "DIRECT" else freshness, scope)
              for family, strength, provider, freshness, scope in tuples]
    evidence = _build_evidence(forced)
    assessment = assess_cloud_usage(evidence)
    assert assessment.state != "CONFIRMED"


@settings(max_examples=200, deadline=None)
@given(_evidence_tuples, _evidence_tuples)
def test_adding_more_evidence_never_lowers_cloud_state_or_fit_score(base_tuples, extra_tuples):
    """Covers both "adding a degraded ProviderResult never lowers state" (a
    degraded provider contributes zero evidence, i.e. adding nothing) and
    "ATS/crt.sh absence never lowers state" (the same fact from the other
    direction: having strictly more evidence can never score strictly lower)."""
    base = _build_evidence(base_tuples)
    extended = base + _build_evidence(extra_tuples)
    # extra_tuples' generated ids collide with base's (both start at ev-000) - make
    # them unique so dedup-by-(family, provider) is the only thing merging entries.
    extended = base + [
        Evidence(**{**vars(ev), "id": f"extra-{i:03d}"}) for i, ev in enumerate(extended[len(base):])
    ]
    score_before, state_before = _score_for(base)
    score_after, state_after = _score_for(extended)
    assert _STATE_ORDER.index(state_after) >= _STATE_ORDER.index(state_before)
    assert score_after >= score_before


@settings(max_examples=200, deadline=None)
@given(_evidence_tuples)
def test_adding_a_degraded_provider_result_never_lowers_score(tuples):
    evidence = _build_evidence(tuples)
    score_before, _ = _score_for(evidence)
    research = Research(evidence=evidence, cloud_usage=assess_cloud_usage(evidence),
                        provider_results=[ProviderResult(provider_name="ats", status="degraded")])
    f = score_fit(Lead("t", "t@x.com", "X", "https://x.com"), research)
    assert f.score == score_before


@settings(max_examples=200, deadline=None)
@given(_evidence_tuples)
def test_cloud_signal_points_always_in_bounds(tuples):
    evidence = _build_evidence(tuples)
    research = Research(evidence=evidence, cloud_usage=assess_cloud_usage(evidence))
    f = score_fit(Lead("t", "t@x.com", "X", "https://x.com"), research)
    assert 0 <= f.cloud_signal_points <= 55
    assert 0 <= f.score <= 100


# --- infra.py: PART A (v0.2.2, scope cut) invariants -----------------

from leadscout import infra  # noqa: E402
from leadscout.models import ProviderResult as _PR  # noqa: E402

_HOLDER_TEXT = st.text(min_size=1, max_size=40).filter(lambda s: s.strip())


@settings(max_examples=100, deadline=None)
@given(st.sampled_from(_EVIDENCE_STRENGTHS))
def test_a_passive_network_footprint_observation_never_reaches_direct_confidence(strength):
    """A passive network/service fingerprint (footprint.py's own range/CNAME/RIPE
    match) is never treated with first-party-statement-level certainty: infra.py's
    confidence scale is only HIGH/MEDIUM/LOW - there is no DIRECT-equivalent tier a
    passive observation could ever reach, whatever strength footprint.py gave it."""
    ev = _build_evidence([("network_footprint", strength, "AWS", "current", "workload")])[0]
    out = infra.build_footprints(_PR(provider_name="footprint", evidence=[ev]))
    assert len(out) == 1
    assert out[0].confidence in ("HIGH", "MEDIUM", "LOW")


@settings(max_examples=100, deadline=None)
@given(st.lists(st.tuples(st.sampled_from(_EVIDENCE_STRENGTHS), st.sampled_from(_EVIDENCE_PROVIDERS)), max_size=6))
def test_edge_delivery_only_evidence_never_contributes_a_cloud_provider(rows):
    """EDGE (CDN-only delivery) never counts as a workload/cloud-provider claim, even
    when it is the ONLY evidence present: `edge_delivery` items must never appear in
    `CloudUsageAssessment.cloud_providers` (cloud.py excludes `_EDGE_ONLY_FAMILY`
    from that list), and infra.py never files them under `category="PUBLIC_CLOUD"`."""
    evidence = [
        Evidence(id=f"ev-{i:03d}", source_type="test", url="https://example.com",
                observed_at="2026-09-19T00:00:00+00:00", content_sha256="a" * 64,
                strength=strength, snippet="s", snapshot_path="out/x.txt",
                family="edge_delivery", provider=provider, freshness="current", scope="edge")
        for i, (strength, provider) in enumerate(rows)
    ]
    assessment = assess_cloud_usage(evidence)
    assert assessment.cloud_providers == []
    footprints = infra.build_footprints(_PR(provider_name="footprint", evidence=evidence))
    assert all(fp.category != "PUBLIC_CLOUD" for fp in footprints)


@settings(max_examples=100, deadline=None)
@given(st.lists(_HOLDER_TEXT, min_size=1, max_size=5, unique=True))
def test_a_bare_network_holder_never_yields_public_cloud(holders):
    """A RIPEstat network holder with no authoritative range match (Hetzner/
    Rackforest-style) never yields PUBLIC_CLOUD, and never becomes `provider` -
    only `network_holder` (the infrastructure model main invariant: network holder !=
    operator; semantic-constraint fix)."""
    result = _PR(provider_name="footprint", evidence=[])
    result._holder_records = [{"ip": f"203.0.113.{i}", "holder": h} for i, h in enumerate(holders)]
    out = infra.build_footprints(result)
    assert all(fp.category != "PUBLIC_CLOUD" for fp in out)
    assert all(fp.provider is None for fp in out)
    assert all(fp.network_holder for fp in out)
