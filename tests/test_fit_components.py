"""Fit score v2: scale/cloud-signal/complexity components, each independently."""
from leadscout.fit import _cloud_signal_points, _complexity_points, _fit_confidence, _scale_points
from leadscout.models import CloudUsageAssessment, Research


def test_scale_points_table():
    assert _scale_points(None)[0] == 3
    assert _scale_points(5)[0] == 3
    assert _scale_points(30)[0] == 10
    assert _scale_points(150)[0] == 18
    assert _scale_points(500)[0] == 25
    assert _scale_points(5000)[0] == 30
    assert _scale_points(50000)[0] == 24


def test_cloud_signal_points_base_comes_from_cloud_usage_state():
    """the base 0-40 points come entirely from CloudUsageAssessment.
    state, never a keyword match against tech_signals/website_text."""
    r = Research(cloud_usage=CloudUsageAssessment(state="CONFIRMED"))
    points, why, count = _cloud_signal_points(r)
    assert points == 40
    assert count == 1
    assert "CONFIRMED" in why


def _ev(eid: str, family: str = "first_party_statement", strength: str = "STRONG"):
    """Minimal Evidence item so a tech_signal's "(ev-NNN)" citation resolves (REVIEW-A §8:
    a signal citing no existing evidence earns no workload point)."""
    from leadscout.models import Evidence
    return Evidence(id=eid, source_type="website", url="https://x.com", observed_at="2026-09-19T00:00:00Z",
                    content_sha256="x", strength=strength, snippet="s", snapshot_path="", family=family)


def test_cloud_signal_points_workload_intensity_bonus_cites_its_evidence_id():
    r = Research(cloud_usage=CloudUsageAssessment(state="POSSIBLE"), tech_signals=["Kubernetes (ev-002)"],
                 evidence=[_ev("ev-002")])
    points, why, count = _cloud_signal_points(r)
    assert points == 18 + 5
    assert count == 2
    assert "ev-002" in why


def test_cloud_signal_points_caps_at_55():
    r = Research(
        cloud_usage=CloudUsageAssessment(state="CONFIRMED"),  # 40
        tech_signals=["GPU workload (ev-001)", "petabyte scale (ev-002)", "multi-cloud (ev-003)",
                     "terraform (ev-004)", "sre (ev-005)"],  # 5 categories x 5 = 25, capped at 15
        evidence=[_ev(f"ev-00{i}") for i in range(1, 6)],
    )
    points, _, _ = _cloud_signal_points(r)
    assert points == 55


def test_cloud_signal_points_zero_when_nothing_found():
    points, why, count = _cloud_signal_points(Research())
    assert points == 0
    assert count == 0
    assert "UNKNOWN" in why


def test_wikidata_structured_industry_never_earns_a_workload_point():
    """Fix #7 (v0.2 PART B): Wikidata's P452 industry may stay a spend-prior
    signal (fit.estimate_cloud_spend reads `research.industry`, never
    `wikidata_industries`), but it must never earn a workload-intensity point
    here - it observed an industry classification, not a Kubernetes/GPU/scale/
    hiring signal, so citing it for one of those was never legitimate."""
    r = Research(cloud_usage=CloudUsageAssessment(state="POSSIBLE"),
                 wikidata_industries=["software company"], wikidata_evidence_id="ev-003")
    points, why, count = _cloud_signal_points(r)
    assert points == 18
    assert count == 1
    assert "structured industry" not in why


def test_dockerfile_only_github_evidence_earns_no_kubernetes_iac_point():
    """Fix #7: github.py only reaches STRONG when a root .tf file actually
    declares a cloud-provider block; a Dockerfile/.github/workflows-only repo
    stays MEDIUM - that alone must never earn the "kubernetes/iac at scale"
    workload-intensity point."""
    from leadscout.models import Evidence
    ev = Evidence(id="ev-001", source_type="github", url="https://github.com/acme/api",
                 observed_at="2026-09-19T00:00:00Z", content_sha256="x", strength="MEDIUM",
                 snippet="acme/api: root signals: ['Dockerfile', '.github/workflows']",
                 snapshot_path="", family="engineering_footprint")
    r = Research(cloud_usage=CloudUsageAssessment(state="POSSIBLE"),
                 tech_signals=["Kubernetes (ev-001)"], evidence=[ev])
    points, why, count = _cloud_signal_points(r)
    assert points == 18
    assert count == 1
    assert "kubernetes/iac at scale" not in why


def test_github_strong_terraform_evidence_still_earns_the_kubernetes_iac_point():
    """The positive case for the same guard: a root .tf file that DOES declare a
    cloud-provider block reaches STRONG in github.py, and still earns the point."""
    from leadscout.models import Evidence
    ev = Evidence(id="ev-001", source_type="github", url="https://github.com/acme/api",
                 observed_at="2026-09-19T00:00:00Z", content_sha256="x", strength="STRONG",
                 snippet="acme/api: main.tf declares a cloud provider block",
                 snapshot_path="", family="engineering_footprint")
    r = Research(cloud_usage=CloudUsageAssessment(state="POSSIBLE"),
                 tech_signals=["Kubernetes (ev-001)"], evidence=[ev])
    points, _, count = _cloud_signal_points(r)
    assert points == 18 + 5
    assert count == 2


def test_complexity_points_regulated_industry():
    r = Research(industry="fintech")
    points, why = _complexity_points(r)
    assert points == 5
    assert "regulated" in why


def test_complexity_points_hybrid_infra():
    r = Research(website_text="we run on-prem and in the cloud")
    points, _ = _complexity_points(r)
    assert points == 5


def test_complexity_points_caps_at_15():
    r = Research(
        industry="fintech",
        website_text="we offer multiple products across international markets, "
                     "with international offices worldwide, running on-prem and cloud",
    )
    points, _ = _complexity_points(r)
    assert points == 15


def test_hq_and_headcount_without_any_cloud_signal_is_low_not_medium():
    """F-13: the cloud-signal leg is necessary, not one of three interchangeable hits.
    Knowing where a company is and how big it is says nothing about whether it has a
    cloud bill worth a call - and both facts can come from the model alone."""
    r = Research(headquarters_country="US", estimated_employees=500)
    assert _fit_confidence(r, strong_medium_count=0) == "LOW"


def test_fit_confidence_medium_with_two_of_three():
    """MEDIUM is still exactly two of the three - here the cloud signals and the
    headcount, with the HQ unknown."""
    r = Research(estimated_employees=500)
    assert _fit_confidence(r, strong_medium_count=2) == "MEDIUM"


def test_fit_confidence_high_needs_all_three():
    r = Research(headquarters_country="US", estimated_employees=500)
    assert _fit_confidence(r, strong_medium_count=2) == "HIGH"


def test_fit_confidence_unknown_hq_string_counts_as_unknown():
    r = Research(headquarters_country="unknown", estimated_employees=500)
    assert _fit_confidence(r, strong_medium_count=2) == "MEDIUM"


def test_confidence_counts_independent_cloud_families_not_only_workload_bonuses():
    """A first-party statement + a network footprint are two independent MEDIUM+ cloud families,
    so with HQ known the fit confidence is at least MEDIUM even with no workload bonus."""
    r = Research(headquarters_country="United States",
                 cloud_usage=CloudUsageAssessment(state="CONFIRMED", qualifying_families={
                     "first_party_statement": {"count": 1, "max_strength": "DIRECT"},
                     "network_footprint": {"count": 1, "max_strength": "STRONG"}}))
    assert _fit_confidence(r, 1) == "MEDIUM"


def test_confidence_does_not_count_edge_delivery_as_a_cloud_family():
    r = Research(headquarters_country="United States",
                 cloud_usage=CloudUsageAssessment(state="EDGE_ONLY", qualifying_families={
                     "edge_delivery": {"count": 3, "max_strength": "STRONG"},
                     "network_footprint": {"count": 1, "max_strength": "MEDIUM"}}))
    assert _fit_confidence(r, 1) == "LOW"


def test_confidence_ignores_evidence_the_classification_excluded():
    """`families_present` is a literal picture of everything collected, including what
    cloud.py threw out; the confidence must read the pool that actually counted. A
    cPanel network footprint (MEDIUM, no provider resolved) is excluded from every tier
    and worth zero points, so it cannot be one of the two cloud signals either."""
    r = Research(headquarters_country="United States", estimated_employees=500,
                 cloud_usage=CloudUsageAssessment(
                     state="UNKNOWN", unknown_reason="NO_PUBLIC_CLOUD_EVIDENCE",
                     families_present={
                         "ats_hiring": {"count": 1, "max_strength": "MEDIUM"},
                         "network_footprint": {"count": 1, "max_strength": "MEDIUM"}},
                     qualifying_families={"ats_hiring": {"count": 1, "max_strength": "MEDIUM"}}))
    assert _fit_confidence(r, 0) == "LOW"


def test_the_assessment_reports_both_the_literal_and_the_qualifying_pool():
    """End of the same rule, at the source: cloud.py must publish what it ACCEPTED
    separately from what was collected, or fit.py has nothing honest to read."""
    from leadscout.cloud import assess_cloud_usage
    from leadscout.models import Evidence

    def ev(eid, family, strength, provider=None):
        return Evidence(id=eid, source_type="footprint", url="u", observed_at="2026-09-20T00:00:00Z",
                        content_sha256="x", strength=strength, snippet="s", snapshot_path="",
                        family=family, provider=provider)

    assessment = assess_cloud_usage([ev("ev-001", "ats_hiring", "MEDIUM"),
                                    ev("ev-002", "network_footprint", "MEDIUM")])
    assert set(assessment.families_present) == {"ats_hiring", "network_footprint"}
    assert set(assessment.qualifying_families) == {"ats_hiring"}


def test_self_reported_band_wins_over_a_contradicting_inferred_headcount():
    """An inferred figure outside the applicant's own band (e.g. one office's staff) must not shrink the lead."""
    from leadscout.fit import employees_for_scoring
    from leadscout.models import Lead
    lead = Lead("a", "a@b.c", "X", "https://x.com", company_size_band="201-1000")
    employees, why = employees_for_scoring(Research(estimated_employees=5), lead)
    assert employees == 500 and "contradicts" in why
    inside, why = employees_for_scoring(Research(estimated_employees=800), lead)
    assert inside == 800 and why == ""


def test_a_workload_signal_citing_a_generic_page_scores_but_earns_no_confidence():
    """F-13, half one: the citation resolves, so the +5 stands (inference may score),
    but `corporate_identity` is not a cloud-bearing family - the landing page observed
    no Kubernetes cluster - so it must not raise the confidence counter."""
    r = Research(cloud_usage=CloudUsageAssessment(state="UNKNOWN"),
                 tech_signals=["Kubernetes (ev-001)"], evidence=[_ev("ev-001", family="corporate_identity")])
    points, why, count = _cloud_signal_points(r)
    assert points == 5
    assert count == 0
    assert "kubernetes/iac at scale" in why


def test_a_cdn_citation_scores_but_never_earns_cloud_confidence():
    """`edge_delivery` is excluded from `_fit_confidence`'s family count because a CDN
    says nothing about the hosting behind it. The workload-intensity counter must apply
    the same exclusion, or the rule holds on one path and not the other."""
    r = Research(cloud_usage=CloudUsageAssessment(state="EDGE_ONLY"),
                 tech_signals=["Kubernetes (ev-001)", "petabyte-scale data (ev-001)"],
                 evidence=[_ev("ev-001", family="edge_delivery")])
    points, why, count = _cloud_signal_points(r)
    assert points == 5 + 10, "the bonus itself is unchanged"
    assert count == 1, "only the EDGE_ONLY base counts, and one signal is not two"
    assert "kubernetes/iac at scale" in why


def test_a_cdn_only_lead_is_not_sales_ready():
    """End to end over the fit layer: the only captured observation is a Cloudflare edge
    record, so however the score lands, the confidence must be LOW."""
    from leadscout.fit import score_fit
    from leadscout.models import Lead
    r = Research(industry="fintech", headquarters_country="United States", hq_source="llm",
                 estimated_employees=500,
                 summary="multiple products across international markets, offices in Berlin",
                 tech_signals=["Kubernetes (ev-001)", "multi-region (ev-001)",
                              "petabyte-scale data (ev-001)"],
                 evidence=[_ev("ev-001", family="edge_delivery")],
                 cloud_usage=CloudUsageAssessment(
                     state="EDGE_ONLY",
                     families_present={"edge_delivery": {"count": 1, "max_strength": "STRONG"}}))
    fit = score_fit(Lead("a", "a@b.c", "EdgeCo", "https://x.example"), r)
    assert fit.confidence == "LOW"


def test_an_invented_company_with_no_cloud_evidence_is_never_sales_ready():
    """F-13, end to end over the fit layer: an LLM-invented perfect lead - 5000
    employees, fintech, AWS, Kubernetes, multi-region, petabyte data - whose only
    captured evidence is its own generic landing page. The score may stay where the
    formula puts it; confidence must be LOW, which is what denies `sales_ready`
    (pipeline.py: clear AND score >= threshold AND confidence != LOW)."""
    from leadscout.fit import score_fit
    from leadscout.models import Lead
    r = Research(
        summary=("Multiple products across international markets, offices in Berlin and Singapore, "
                 "built on AWS with Kubernetes and petabyte-scale pipelines."),
        industry="fintech", headquarters_country="United States", hq_source="llm",
        estimated_employees=5000,
        tech_signals=["Kubernetes (ev-001)", "multi-region AWS (ev-001)",
                     "petabyte-scale data (ev-001)", "GPU workloads (ev-001)"],
        evidence=[_ev("ev-001", family="corporate_identity")],
        cloud_usage=CloudUsageAssessment(state="UNKNOWN", unknown_reason="NO_PUBLIC_CLOUD_EVIDENCE",
                                         families_present={"corporate_identity": {"count": 1,
                                                                                  "max_strength": "STRONG"}}),
    )
    fit = score_fit(Lead("a", "a@b.c", "FintechPerfect Inc", "https://x.example"), r)
    assert fit.score == 60, "the fit formula is deliberately untouched by this fix"
    assert fit.confidence == "LOW"


def test_a_workload_signal_citing_no_evidence_earns_nothing():
    r = Research(cloud_usage=CloudUsageAssessment(state="POSSIBLE"), tech_signals=["GPU cluster (ev-404)"])
    points, why, count = _cloud_signal_points(r)
    assert points == 18 and count == 1 and "gpu" not in why.lower()
