"""leadscout.cloud.assess_cloud_usage: deterministic classification over a list of
Evidence. Pure/offline - no fixtures, no network.
"""
from leadscout.cloud import assess_cloud_usage
from leadscout.models import Evidence, ProviderResult

_BASE = dict(url="https://example.com", observed_at="2026-09-19T00:00:00+00:00",
             content_sha256="a" * 64, snippet="test", snapshot_path="out/x.txt")


def _ev(id_, family, strength, provider=None, scope="unknown", freshness="current", source_type="trust_pages"):
    return Evidence(id=id_, source_type=source_type, family=family, strength=strength,
                     provider=provider, scope=scope, freshness=freshness, **_BASE)


def test_no_evidence_is_unknown():
    result = assess_cloud_usage([])
    assert result.state == "UNKNOWN"
    assert result.boundary


def test_one_current_direct_first_party_statement_is_confirmed():
    ev = _ev("ev-001", "first_party_statement", "DIRECT", provider="AWS", scope="workload")
    result = assess_cloud_usage([ev])
    assert result.state == "CONFIRMED"


def test_one_current_direct_vendor_case_study_is_confirmed():
    ev = _ev("ev-001", "vendor_case_study", "DIRECT", provider="AWS", scope="workload", source_type="vendor")
    result = assess_cloud_usage([ev])
    assert result.state == "CONFIRMED"


def test_historical_direct_first_party_statement_is_not_confirmed():
    ev = _ev("ev-001", "first_party_statement", "DIRECT", provider="AWS", scope="workload", freshness="historical")
    result = assess_cloud_usage([ev])
    assert result.state != "CONFIRMED"


def test_saas_dependency_scope_direct_is_never_confirmed():
    """A SaaS-integration admission ("we integrate with AWS Cost Explorer") is not
    a hosting claim, however strong its own strength value is."""
    ev = _ev("ev-001", "first_party_statement", "DIRECT", provider="AWS", scope="saas_dependency")
    result = assess_cloud_usage([ev])
    assert result.state != "CONFIRMED"


def test_two_non_edge_families_one_strong_is_likely():
    evs = [
        _ev("ev-001", "engineering_footprint", "STRONG", provider="AWS", source_type="github"),
        _ev("ev-002", "network_footprint", "MEDIUM", provider="AWS", source_type="footprint"),
    ]
    result = assess_cloud_usage(evs)
    assert result.state == "LIKELY"


def test_single_strong_non_edge_family_is_possible():
    ev = _ev("ev-001", "engineering_footprint", "STRONG", provider="AWS", source_type="github")
    result = assess_cloud_usage([ev])
    assert result.state == "POSSIBLE"


def test_two_medium_non_edge_families_is_possible():
    evs = [
        _ev("ev-001", "engineering_footprint", "MEDIUM", provider="AWS", source_type="github"),
        _ev("ev-002", "ats_hiring", "MEDIUM", provider="AWS", source_type="ats"),
    ]
    result = assess_cloud_usage(evs)
    assert result.state == "POSSIBLE"


def test_one_medium_family_alone_is_unknown():
    ev = _ev("ev-001", "ats_hiring", "MEDIUM", provider="AWS", source_type="ats")
    result = assess_cloud_usage([ev])
    assert result.state == "UNKNOWN"


def test_edge_delivery_only_is_edge_only():
    ev = _ev("ev-001", "edge_delivery", "WEAK", provider="Cloudflare", scope="edge", source_type="footprint")
    result = assess_cloud_usage([ev])
    assert result.state == "EDGE_ONLY"


def test_edge_delivery_plus_a_real_family_is_not_edge_only():
    evs = [
        _ev("ev-001", "edge_delivery", "WEAK", provider="Cloudflare", scope="edge", source_type="footprint"),
        _ev("ev-002", "engineering_footprint", "STRONG", provider="AWS", source_type="github"),
    ]
    result = assess_cloud_usage(evs)
    assert result.state != "EDGE_ONLY"


def test_duplicate_family_provider_evidence_deduped_not_inflated():
    """Five ATS postings all naming AWS must count as one family instance, not five -
    still only POSSIBLE (one STRONG non-edge family), never LIKELY on its own."""
    evs = [_ev(f"ev-{i:03d}", "ats_hiring", "STRONG", provider="AWS", source_type="ats") for i in range(5)]
    result = assess_cloud_usage(evs)
    assert result.state == "POSSIBLE"


def test_families_present_reports_raw_counts_and_max_strength():
    evs = [
        _ev("ev-001", "ats_hiring", "MEDIUM", provider="AWS", source_type="ats"),
        _ev("ev-002", "ats_hiring", "STRONG", provider="AWS", source_type="ats"),
    ]
    result = assess_cloud_usage(evs)
    assert result.families_present["ats_hiring"] == {"count": 2, "max_strength": "STRONG"}


def test_provider_sub_states_computed_per_provider():
    evs = [
        _ev("ev-001", "first_party_statement", "DIRECT", provider="AWS", scope="workload"),
        _ev("ev-002", "ats_hiring", "MEDIUM", provider="Azure", source_type="ats"),
    ]
    result = assess_cloud_usage(evs)
    by_provider = {p["provider"]: p["state"] for p in result.providers}
    assert by_provider["AWS"] == "CONFIRMED"
    assert by_provider["Azure"] == "UNKNOWN"


def test_provider_aggregate_fields_computed_from_evidence_observed_at_and_family():
    """Item 6: first_seen_at/last_seen_at/observation_count/independent_family_count,
    computed from existing Evidence fields alone - no new data collected."""
    evs = [
        _ev("ev-001", "first_party_statement", "DIRECT", provider="AWS", scope="workload"),
        Evidence(id="ev-002", source_type="ats", family="ats_hiring", strength="MEDIUM",
                 provider="AWS", scope="unknown", freshness="current",
                 url="https://example.com", observed_at="2026-09-01T00:00:00+00:00",
                 content_sha256="b" * 64, snippet="test", snapshot_path="out/y.txt"),
        _ev("ev-003", "ats_hiring", "MEDIUM", provider="Azure", source_type="ats"),
    ]
    result = assess_cloud_usage(evs)
    by_provider = {p["provider"]: p for p in result.providers}
    aws = by_provider["AWS"]
    assert aws["observation_count"] == 2
    assert aws["independent_family_count"] == 2  # first_party_statement + ats_hiring
    assert aws["first_seen_at"] == "2026-09-01T00:00:00+00:00"
    assert aws["last_seen_at"] == "2026-09-19T00:00:00+00:00"
    azure = by_provider["Azure"]
    assert azure["observation_count"] == 1
    assert azure["independent_family_count"] == 1


def test_boundary_is_always_a_non_empty_string():
    for evs in ([], [_ev("ev-001", "ats_hiring", "MEDIUM", provider="AWS", source_type="ats")]):
        assert assess_cloud_usage(evs).boundary


def test_identity_and_encyclopedic_evidence_alone_is_unknown():
    """Regression for Masterplast (LIKELY) and Artizan (POSSIBLE) in commit
    020c40e: `corporate_identity`/`encyclopedic` evidence must never contribute to
    cloud state on its own, however many independent items or however strong -
    only the cloud-bearing families (first_party_statement, vendor_case_study,
    ats_hiring, engineering_footprint, network_footprint, edge_delivery) can."""
    evs = [
        _ev("ev-001", "corporate_identity", "STRONG", provider=None, source_type="website"),
        _ev("ev-002", "encyclopedic", "STRONG", provider=None, source_type="wikipedia"),
        _ev("ev-003", "policy", "STRONG", provider=None, source_type="opensanctions"),
    ]
    result = assess_cloud_usage(evs)
    assert result.state == "UNKNOWN"


def test_no_cloud_bearing_family_is_always_unknown_even_with_many_items():
    evs = [
        _ev(f"ev-{i:03d}", "corporate_identity", "STRONG", provider=None, source_type="website")
        for i in range(10)
    ] + [_ev("ev-010", "encyclopedic", "MEDIUM", provider=None, source_type="wikipedia")]
    result = assess_cloud_usage(evs)
    assert result.state == "UNKNOWN"


# --- v0.2.2 additions (PART A item 6, scope cut): cloud_providers /
# unknown_reason / missing_channels ------------------------------------------------

def test_open_provider_string_counts_toward_cloud_state_not_just_the_big_four():
    """CloudProvider is an open string now (v0.2.2) - `assess_cloud_usage` must no
    longer gate counting on a hardcoded {"AWS","Azure","GCP","OCI"} set (PART A
    item 6): any non-empty provider on a cloud-bearing family counts."""
    evs = [_ev("ev-001", "network_footprint", "STRONG", provider="Hetzner")]
    result = assess_cloud_usage(evs)
    assert result.state == "POSSIBLE"
    assert result.cloud_providers == ["Hetzner"]


def test_unknown_reason_is_none_when_state_is_not_unknown():
    evs = [_ev("ev-001", "first_party_statement", "DIRECT", provider="AWS")]
    result = assess_cloud_usage(evs)
    assert result.state == "CONFIRMED"
    assert result.unknown_reason is None


def test_unknown_reason_no_public_cloud_evidence_when_domain_resolved_and_no_missing_channels():
    result = assess_cloud_usage([], provider_results=[], domain_resolved=True)
    assert result.state == "UNKNOWN"
    assert result.unknown_reason == "NO_PUBLIC_CLOUD_EVIDENCE"
    assert result.missing_channels == []


def test_unknown_reason_no_resolvable_domain_when_domain_not_resolved():
    result = assess_cloud_usage([], provider_results=[], domain_resolved=False)
    assert result.state == "UNKNOWN"
    assert result.unknown_reason == "NO_RESOLVABLE_DOMAIN"


def test_missing_channel_never_yields_no_public_cloud_evidence():
    """A degraded provider means the assessment might be incomplete - never
    allowed to sound confidently negative (NO_PUBLIC_CLOUD_EVIDENCE); it must say
    INSUFFICIENT_EVIDENCE instead, and name the missing channel."""
    provider_results = [ProviderResult(provider_name="footprint", status="degraded", note="crt.sh timeout")]
    result = assess_cloud_usage([], provider_results=provider_results, domain_resolved=True)
    assert result.state == "UNKNOWN"
    assert result.unknown_reason == "INSUFFICIENT_EVIDENCE"
    assert result.missing_channels == ["footprint"]


def test_skipped_provider_does_not_count_as_a_missing_channel():
    """"skipped" is overloaded in the existing provider code: vendor.py/hunter.py
    use it for a genuine capability gap (no API key), but ats.py/trust_pages.py use
    the SAME status for "ran fine, found nothing" - so only "degraded" (an
    unambiguous failure) counts toward missing_channels; see assess_cloud_usage's
    docstring."""
    provider_results = [
        ProviderResult(provider_name="hunter", status="skipped", note="no HUNTER_API_KEY"),
        ProviderResult(provider_name="ats", status="skipped", note="no ATS board link found"),
    ]
    result = assess_cloud_usage([], provider_results=provider_results, domain_resolved=True)
    assert result.missing_channels == []
    assert result.unknown_reason == "NO_PUBLIC_CLOUD_EVIDENCE"


def test_edge_only_provider_excluded_from_cloud_providers():
    evs = [_ev("ev-001", "edge_delivery", "WEAK", provider="Cloudflare", scope="edge")]
    result = assess_cloud_usage(evs)
    assert result.state == "EDGE_ONLY"
    assert result.cloud_providers == []


def test_a_cloud_channel_that_could_never_run_is_a_missing_channel():
    """vendor.py "skipped" means no BRAVE_API_KEY (capability gap), so "no public-cloud evidence" would
    overstate; ats/trust_pages "skipped" means the provider ran and found nothing."""
    from leadscout.models import ProviderResult
    no_key = assess_cloud_usage([], provider_results=[ProviderResult(provider_name="vendor", status="skipped")],
                                domain_resolved=True)
    assert no_key.unknown_reason == "INSUFFICIENT_EVIDENCE" and "vendor" in no_key.missing_channels
    ran = assess_cloud_usage([], provider_results=[ProviderResult(provider_name="ats", status="skipped")],
                             domain_resolved=True)
    assert ran.unknown_reason == "NO_PUBLIC_CLOUD_EVIDENCE" and ran.missing_channels == []


# --- Availability is scoped to the decision (2026-09-20 audit) ---------------------

def _unknown_assessment(provider_results):
    """No cloud evidence at all, so the state is UNKNOWN and only the REASON varies."""
    from leadscout.cloud import assess_cloud_usage
    return assess_cloud_usage([], provider_results=provider_results, domain_resolved=True)


def test_an_identity_provider_outage_does_not_weaken_the_cloud_reason():
    """A rate-limited GLEIF is an identity lookup that cannot produce a cloud signal at
    all. Counting it as a missing cloud channel turned a clean NO_PUBLIC_CLOUD_EVIDENCE
    into INSUFFICIENT_EVIDENCE - a decision reading another decision's outage."""
    from leadscout.models import ProviderResult
    for name in ("gleif", "hunter", "headcount", "wikidata"):
        a = _unknown_assessment([ProviderResult(provider_name=name, status="degraded")])
        assert a.state == "UNKNOWN"
        assert a.missing_channels == [], f"{name} counted as a cloud channel"
        assert a.unknown_reason == "NO_PUBLIC_CLOUD_EVIDENCE", f"{name}: {a.unknown_reason}"


def test_a_cloud_bearing_provider_outage_still_weakens_the_cloud_reason():
    """The other half: these channels DO feed the assessment, so their absence has to
    stop it claiming the confident reason."""
    from leadscout.models import ProviderResult
    for name in ("ats", "github", "footprint", "trust_pages", "vendor"):
        a = _unknown_assessment([ProviderResult(provider_name=name, status="degraded")])
        assert a.state == "UNKNOWN"
        assert a.missing_channels == [name]
        assert a.unknown_reason == "INSUFFICIENT_EVIDENCE", f"{name}: {a.unknown_reason}"


def test_a_mixed_run_reports_only_the_cloud_bearing_outage():
    from leadscout.models import ProviderResult
    a = _unknown_assessment([
        ProviderResult(provider_name="gleif", status="degraded"),
        ProviderResult(provider_name="footprint", status="degraded"),
        ProviderResult(provider_name="hunter", status="degraded"),
    ])
    assert a.missing_channels == ["footprint"]
    assert a.unknown_reason == "INSUFFICIENT_EVIDENCE"
