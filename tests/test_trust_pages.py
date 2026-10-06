"""trust_pages provider: first-party hosting statements vs. SaaS-dependency wording,
and status-page component detection - offline via synthetic HTML/text fixtures
and httpx.MockTransport for the two status-page GETs.
"""
import httpx

from leadscout.providers import trust_pages


def _patch_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(trust_pages.httpx, "Client", fake_client)


def _all_404(request):
    return httpx.Response(404, text="not found")


def test_subprocessor_table_row_is_medium_never_direct(monkeypatch):
    """Fix #5: CLOUD_SUBPROCESSOR is its own predicate now, MEDIUM strength - a
    subprocessor-table row alone must never reach DIRECT/CONFIRMED."""
    _patch_client(monkeypatch, _all_404)
    page = "Subprocessors\nAmazon Web Services — our cloud subprocessor for hosting infrastructure."
    result = trust_pages.run("acme.example", [{"url": "https://acme.example/legal/subprocessors",
                                               "text": page, "locator": "table[0]"}])
    assert result.status == "ok"
    assert len(result.evidence) == 1
    ev = result.evidence[0]
    assert ev.family == "first_party_statement"
    assert ev.strength == "MEDIUM"
    assert ev.provider == "AWS"
    assert ev.url == "https://acme.example/legal/subprocessors#table[0]"


def test_cdn_only_wording_is_edge_delivery_not_first_party_statement(monkeypatch):
    """A statement whose own wording resolves to CDN/edge scope (e.g. "content
    delivery") is an edge observation, not a first-party hosting claim -
    cloud.py's `first_party_statement` family counts toward LIKELY/POSSIBLE
    regardless of scope, so this must be `edge_delivery` (excluded from that) or
    a CDN-only mention would look like real workload hosting evidence."""
    _patch_client(monkeypatch, _all_404)
    page = "Azure Front Door is a subprocessor providing content delivery and edge routing for our platform."
    result = trust_pages.run("acme.example", [page])
    assert len(result.evidence) == 1
    ev = result.evidence[0]
    assert ev.family == "edge_delivery"
    assert ev.scope == "edge"
    assert ev.strength == "MEDIUM"


def test_structured_page_dict_without_locator_carries_bare_url(monkeypatch):
    _patch_client(monkeypatch, _all_404)
    page = "Our infrastructure is hosted on Amazon Web Services (AWS)."
    result = trust_pages.run("acme.example", [{"url": "https://acme.example/trust", "text": page,
                                               "locator": "main_text"}])
    assert result.evidence[0].url == "https://acme.example/trust"


def test_saas_dependency_wording_is_not_a_hosting_claim(monkeypatch):
    """`Evidence.family` has no "saas_dependency" member (only `scope` does, see
    models.Evidence) - a SaaS-integration sentence is still the company's own
    `first_party_statement`, just downgraded to WEAK strength and scope
    "saas_dependency" so cloud.py never counts it as a real hosting claim."""
    _patch_client(monkeypatch, _all_404)
    page = "We integrate with AWS Cost Explorer to show you your cloud spend."
    result = trust_pages.run("acme.example", [page])
    assert len(result.evidence) == 1
    ev = result.evidence[0]
    assert ev.family == "first_party_statement"
    assert ev.scope == "saas_dependency"
    assert ev.strength == "WEAK"


def test_status_page_component_names_provider_is_strong(monkeypatch):
    def handler(request):
        if "status.acme.example" in str(request.url):
            return httpx.Response(
                200, text="<html>Acme status: all systems operational. Component: AWS us-east-1</html>",
            )
        return httpx.Response(404)

    _patch_client(monkeypatch, handler)
    result = trust_pages.run("acme.example", [])
    assert result.status == "ok"
    assert len(result.evidence) == 1
    ev = result.evidence[0]
    assert ev.family == "first_party_statement"
    assert ev.strength == "STRONG"


def test_generic_status_aggregator_without_company_identity_is_ignored(monkeypatch):
    """A third-party status aggregator (or a parked/wildcard host) can return 200 and
    happen to mention a provider word - it must not pass without any page-identity
    signal tying it to THIS company (REVIEW-6bB-verified.md #2)."""

    def handler(request):
        if "status.acme.example" in str(request.url):
            return httpx.Response(200, text="<html>Welcome to StatusAggregator. AWS us-east-1 is up.</html>")
        return httpx.Response(404)

    _patch_client(monkeypatch, handler)
    result = trust_pages.run("acme.example", [])
    assert result.status == "skipped"
    assert result.evidence == []


def test_status_page_with_company_name_but_no_status_wording_is_ignored(monkeypatch):
    """The domain label alone (e.g. picked up by a parked page that echoes the
    hostname) is not enough either - the page must also read like a status page."""

    def handler(request):
        if "status.acme.example" in str(request.url):
            return httpx.Response(200, text="<html>acme.example - this domain is parked. AWS.</html>")
        return httpx.Response(404)

    _patch_client(monkeypatch, handler)
    result = trust_pages.run("acme.example", [])
    assert result.status == "skipped"
    assert result.evidence == []


def test_both_status_urls_404_and_no_page_statement_is_skipped(monkeypatch):
    _patch_client(monkeypatch, _all_404)
    result = trust_pages.run("acme.example", ["Just a generic homepage with no cloud mentions."])
    assert result.status == "skipped"
    assert result.evidence == []


def test_html_fragment_with_no_sentence_produces_nothing(monkeypatch):
    """Regression for the PostHog false positive in commit 020c40e: a raw HTML
    fragment fed straight from the landing page (not extracted text) produced a
    bogus DIRECT OCI statement from "Review proposed improvements and pull
    requests...</p><span data-state..." - detection must run on extracted text,
    and this fragment names no provider at all once tags are stripped."""
    _patch_client(monkeypatch, _all_404)
    page = "<p>Review proposed improvements and pull requests</p><span data-state=\"oci-widget\">x</span>"
    result = trust_pages.run("acme.example", [page])
    assert result.status == "skipped"
    assert result.evidence == []


def test_social_media_never_matches_oci(monkeypatch):
    _patch_client(monkeypatch, _all_404)
    page = "We are active on social media and post updates every week."
    result = trust_pages.run("acme.example", [page])
    assert result.status == "skipped"
    assert result.evidence == []


def test_hosted_on_aws_sentence_is_direct_workload(monkeypatch):
    _patch_client(monkeypatch, _all_404)
    page = "Our infrastructure is hosted on Amazon Web Services (AWS)."
    result = trust_pages.run("acme.example", [page])
    assert result.status == "ok"
    assert len(result.evidence) == 1
    ev = result.evidence[0]
    assert ev.family == "first_party_statement"
    assert ev.strength == "DIRECT"
    assert ev.provider == "AWS"
    assert ev.scope == "workload"


def test_cost_explorer_integration_is_not_direct(monkeypatch):
    _patch_client(monkeypatch, _all_404)
    page = "We integrate with AWS Cost Explorer to show you your cloud spend."
    result = trust_pages.run("acme.example", [page])
    assert len(result.evidence) == 1
    ev = result.evidence[0]
    assert ev.strength != "DIRECT"
    assert ev.scope == "saas_dependency"


def test_network_error_degrades_without_crash(monkeypatch):
    def boom(*args, **kwargs):
        raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr(trust_pages.httpx, "Client", boom)
    result = trust_pages.run("acme.example", [])
    assert result.status == "degraded"


def test_host_verb_is_a_first_party_hosting_statement_but_feature_heading_is_not():
    from leadscout.providers.trust_pages import _find_statements
    hosted = _find_statements("<p>Customers can choose whether to host data on our AWS servers in the EU.</p>")
    assert [(k, p) for k, p, _, _ in hosted] == [("HOSTED_ON", "AWS")]
    assert _find_statements("<dt>AWS cloud security</dt>") == []


def test_infrastructure_noun_in_security_tooling_sentence_is_not_hosting():
    from leadscout.providers.trust_pages import _find_statements
    tooling = _find_statements("<p>Threat alerts from our infrastructure are triaged with AWS GuardDuty.</p>")
    assert not [s for s in tooling if s[0] == "HOSTED_ON"]
    asserted = _find_statements("<p>Our infrastructure runs on Google Cloud in Frankfurt.</p>")
    assert [(k, p) for k, p, _, _ in asserted] == [("HOSTED_ON", "GCP")]


def test_a_hosting_verb_is_not_a_hosting_claim():
    """REVIEW-B adversarial pass: five wordings that must NOT become a first-party hosting claim."""
    from leadscout.providers.trust_pages import _find_statements
    for text in ("<p>We host webinars with AWS startups every Thursday.</p>",
                 "<p>Do you host customer workloads on AWS?</p>",
                 "<p>As an AWS reseller, we offer hosting plans to customers.</p>",
                 "<p>Competitor: AWS hosting. Our product: self-managed.</p>",
                 "<p>Bitrise on AWS runs in your own AWS account.</p>"):
        assert not [s for s in _find_statements(text) if s[0] == "HOSTED_ON"], text


def test_a_real_first_party_statement_still_passes_with_company_or_first_person_subject():
    from leadscout.providers.trust_pages import _find_statements
    by_name = _find_statements("<p>Zapier is hosted on Amazon Web Services (AWS) in the United States.</p>",
                               company_label="zapier")
    assert [(k, p) for k, p, _, _ in by_name] == [("HOSTED_ON", "AWS")]
    first_person = _find_statements("<p>Our production databases are hosted on Google Cloud.</p>")
    assert [(k, p) for k, p, _, _ in first_person] == [("HOSTED_ON", "GCP")]
    # ...but the same sentence without a subject naming this company is not its claim
    assert not _find_statements("<p>Zapier is hosted on Amazon Web Services (AWS).</p>")
