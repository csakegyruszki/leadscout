"""footprint provider: CNAME/IP/RIPE classification, within-family aggregation, and
graceful degradation - offline, by monkeypatching the network-touching helpers
(crtsh_subdomains/classify_host/_domain_age_evidence) rather than every httpx call,
since the interesting logic here is the classification + aggregation, not the raw
transport (already exercised the MockTransport way in test_ats.py/test_github.py).
Measured shapes (raw/dns-rdap-ripe__*.json): zapier.com -> AMAZON-02 (AWS);
snapp.ir -> ArvanCloud (edge, Iran); masterplast.hu -> Rackforest (no cloud match).
"""
import httpx

from leadscout.providers import footprint


def test_classify_cname_cloudfront_is_aws_edge():
    assert footprint._classify_cname("d123.cloudfront.net") == ("AWS", "CloudFront", "edge")


def test_classify_cname_s3_is_aws_storage():
    assert footprint._classify_cname("bucket.s3.amazonaws.com") == ("AWS", "S3", "storage")


def test_classify_cname_execute_api_is_aws_workload():
    assert footprint._classify_cname("abc123.execute-api.us-east-1.amazonaws.com") == ("AWS", "API Gateway", "workload")


def test_classify_cname_generic_amazonaws_is_aws_workload():
    assert footprint._classify_cname("something.us-east-1.amazonaws.com") == ("AWS", "AWS", "workload")


def test_classify_cname_arvancloud_unmapped_returns_none():
    assert footprint._classify_cname("edge.arvancloud.ir") is None


def test_aggregate_edge_only_never_claims_named_provider():
    """snapp.ir-style: an edge CDN with no non-edge scope must produce edge_delivery
    WEAK, and never a network_footprint claim for AWS/Azure/GCP/OCI."""
    obs = [("cdn.snapp.ir", None, "CDN", "edge")]
    out = footprint._aggregate(obs)
    assert len(out) == 1
    assert out[0].family == "edge_delivery"
    assert out[0].strength == "WEAK"
    assert out[0].provider not in ("AWS", "Azure", "GCP", "OCI")


def test_aggregate_single_aws_host_workload_is_medium():
    obs = [("api.example.com", "AWS", "AWS", "workload")]
    out = footprint._aggregate(obs)
    assert len(out) == 1
    assert out[0].family == "network_footprint"
    assert out[0].strength == "MEDIUM"
    assert out[0].provider == "AWS"


def test_aggregate_three_aws_hosts_cloudfront_s3_execute_api_is_strong():
    """The measured synthetic case: three first-party hosts on CloudFront+S3+
    execute-api -> ONE AWS evidence, STRONG, distinct services = 3, scope non-edge."""
    obs = [
        ("cdn.example.com", "AWS", "CloudFront", "edge"),
        ("assets.example.com", "AWS", "S3", "storage"),
        ("api.example.com", "AWS", "API Gateway", "workload"),
    ]
    out = footprint._aggregate(obs)
    assert len(out) == 1
    ev = out[0]
    assert ev.family == "network_footprint"
    assert ev.strength == "STRONG"
    assert ev.provider == "AWS"
    assert "3" in ev.snippet  # distinct hosts
    assert "storage" in ev.snippet or "workload" in ev.snippet  # non-edge scope present


def test_aggregate_duplicate_host_does_not_create_second_evidence():
    obs = [
        ("api.example.com", "AWS", "API Gateway", "workload"),
        ("api.example.com", "AWS", "API Gateway", "workload"),
    ]
    out = footprint._aggregate(obs)
    assert len(out) == 1


def test_aggregate_evidence_has_a_url():
    """REVIEW-6bB-verified.md #4: network_footprint Evidence must not have url=""."""
    obs = [("api.example.com", "AWS", "API Gateway", "workload")]
    out = footprint._aggregate(obs)
    assert out[0].url == "https://api.example.com"


def test_aggregate_mixed_scope_precedence_is_workload_over_storage_over_edge():
    """REVIEW-6bB-verified.md #4: deterministic precedence regardless of iteration
    order - workload beats storage beats edge; an edge-only host never becomes the
    aggregate's scope when a non-edge host is also present."""
    obs = [
        ("cdn.example.com", "AWS", "CloudFront", "edge"),
        ("assets.example.com", "AWS", "S3", "storage"),
    ]
    out = footprint._aggregate(obs)
    assert out[0].scope == "storage"

    obs2 = [
        ("assets.example.com", "AWS", "S3", "storage"),
        ("api.example.com", "AWS", "API Gateway", "workload"),
    ]
    out2 = footprint._aggregate(obs2)
    assert out2[0].scope == "workload"


def test_run_zapier_style_one_aws_medium_or_strong_evidence(monkeypatch):
    monkeypatch.setattr(footprint, "crtsh_subdomains", lambda domain, client: ["api.zapier.com"])
    monkeypatch.setattr(footprint, "classify_host", lambda host, client, budget=None: ("AWS", "AWS", "workload"))
    monkeypatch.setattr(footprint, "_domain_age_evidence", lambda domain, client: None)
    result = footprint.run("zapier.com")
    assert result.status == "ok"
    assert len(result.evidence) == 1
    assert result.evidence[0].provider == "AWS"
    assert result.evidence[0].family == "network_footprint"


def test_run_snapp_style_edge_only_never_names_a_cloud_provider(monkeypatch):
    monkeypatch.setattr(footprint, "crtsh_subdomains", lambda domain, client: ["cdn.snapp.ir"])
    monkeypatch.setattr(footprint, "classify_host", lambda host, client, budget=None: (None, "CDN", "edge"))
    monkeypatch.setattr(footprint, "_domain_age_evidence", lambda domain, client: None)
    result = footprint.run("snapp.ir")
    assert result.status == "ok"
    assert len(result.evidence) == 1
    assert result.evidence[0].family == "edge_delivery"
    assert result.evidence[0].provider not in ("AWS", "Azure", "GCP", "OCI")


def test_run_masterplast_style_no_cloud_match_status_ok_no_evidence(monkeypatch):
    monkeypatch.setattr(footprint, "crtsh_subdomains", lambda domain, client: ["shop.masterplast.hu"])
    monkeypatch.setattr(footprint, "classify_host", lambda host, client, budget=None: None)
    monkeypatch.setattr(footprint, "_domain_age_evidence", lambda domain, client: None)
    result = footprint.run("masterplast.hu")
    assert result.status == "ok"
    assert result.evidence == []


def test_run_crtsh_timeout_degrades_without_exception(monkeypatch):
    def boom(domain, client):
        import httpx
        raise httpx.TimeoutException("crt.sh slow")

    monkeypatch.setattr(footprint, "crtsh_subdomains", boom)
    result = footprint.run("slow-example.com")
    assert result.status == "degraded"
    assert result.evidence == []


def test_run_domain_age_evidence_included_when_present(monkeypatch):
    from leadscout.models import Evidence

    monkeypatch.setattr(footprint, "crtsh_subdomains", lambda domain, client: [])
    age_ev = Evidence(id="", source_type="footprint", url="https://rdap.org/domain/zapier.com",
                       observed_at="2026-09-19T00:00:00Z", content_sha256="", strength="WEAK",
                       snippet="domain registered 2011-10-30, age 5000 days", snapshot_path="",
                       family="corporate_identity")
    monkeypatch.setattr(footprint, "_domain_age_evidence", lambda domain, client: age_ev)
    result = footprint.run("zapier.com")
    assert result.status == "ok"
    assert len(result.evidence) == 1
    assert result.evidence[0].family == "corporate_identity"


def test_domain_age_evidence_follows_a_redirecting_rdap_response():
    """Regression: rdap.org/domain/<domain> 302-redirects to the actual RIR's RDAP
    server. A client without follow_redirects=True saw only the 302 and produced no
    evidence - caught by the live zapier.com smoke run, not by any mocked test
    before this one (every other test here mocks _domain_age_evidence itself)."""
    def handler(request):
        if str(request.url) == "https://rdap.org/domain/zapier.com":
            return httpx.Response(302, headers={"location": "https://rdap.verisign.com/com/v1/domain/zapier.com"})
        return httpx.Response(200, json={"events": [{"eventAction": "registration",
                                                       "eventDate": "2011-10-30T20:11:40Z"}]})

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
        ev = footprint._domain_age_evidence("zapier.com", client)
    assert ev is not None
    assert ev.family == "corporate_identity"
    assert "2011-10-30" in ev.snippet


def test_max_http_calls_ceiling_is_enforced(monkeypatch):
    """REVIEW-6bB-verified.md #5: a large subdomain list must stop making classify
    calls once the ceiling is hit, rather than an unbounded fan-out."""
    hosts = [f"host{i}.example.com" for i in range(20)]
    monkeypatch.setattr(footprint, "crtsh_subdomains", lambda domain, client: hosts)
    calls_made = []

    def counting_classify(host, client, budget=None):
        calls_made.append(host)
        if budget is not None:
            budget.spend(2)  # a real classify_host makes ~2 DoH/RIPEstat calls per host
        return ("AWS", "AWS", "workload")

    monkeypatch.setattr(footprint, "classify_host", counting_classify)
    monkeypatch.setattr(footprint, "_domain_age_evidence", lambda domain, client: None)
    result = footprint.run("example.com", max_calls=5)
    assert result.status == "ok"
    # 1 call is already spent on crtsh_subdomains itself, so at most 4 hosts can be
    # classified before the ceiling (5) is reached.
    assert len(calls_made) <= 4
    assert result.calls <= 5
    assert "MAX_HTTP_CALLS" in result.note


def test_ripestat_lookup_is_memoised_across_shared_ips():
    """Two different hostnames resolving to the same IP must hit RIPEstat once, not
    twice, within a single provider run (REVIEW-6bB-verified.md #5)."""
    ripe_calls = []

    def handler(request):
        ripe_calls.append(str(request.url))
        return httpx.Response(200, json={"data": {"asns": [{"holder": "AMAZON-02"}]}})

    budget = footprint._Budget(40)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        first = footprint._ripe_holder_provider("203.0.113.9", client, budget)
        second = footprint._ripe_holder_provider("203.0.113.9", client, budget)
    # PART A item 4: _ripe_holder_provider now returns the raw RIPEstat holder
    # string as a third element (infra.py's open-string network holder for hosts
    # with no authoritative range match) - the memoisation behaviour under test
    # (one RIPEstat call for two lookups of the same IP) is unchanged.
    assert first == second == ("AWS", False, "AMAZON-02")
    assert len(ripe_calls) == 1


def test_crtsh_subdomains_queried_with_registrable_domain_not_a_www_host(monkeypatch, tmp_path):
    """Item 2 (the actual observed bug): out/cache/crtsh/www.masterplast.hu.json held
    a single stale entry because crtsh_subdomains was queried with the "www." host
    research.py handed it. research.py now strips "www." before calling this - this
    test locks in that crtsh_subdomains itself queries crt.sh with whatever domain
    string it is GIVEN (the registrable-domain responsibility lives at the call
    site, leadscout/util.py's registrable_domain), by asserting the actual crt.sh
    query string never contains "www."."""
    monkeypatch.setattr(footprint, "CRTSH_CACHE", tmp_path / "crtsh")
    seen_queries = []

    def handler(request):
        seen_queries.append(str(request.url))
        return httpx.Response(200, json=[{"name_value": "api.masterplast.hu"}])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        subs = footprint.crtsh_subdomains("masterplast.hu", client)
    assert subs == ["api.masterplast.hu"]
    assert all("www." not in q for q in seen_queries)
    assert "masterplast.hu" in seen_queries[0]


def test_crtsh_subdomains_never_caches_an_empty_result(monkeypatch, tmp_path):
    """An empty crt.sh response (rate-limit glitch, wrong host, or a real zero-certs
    domain) must NOT be written to the cache - caching it would turn one bad query
    into a permanent false negative (the observed www.masterplast.hu.json symptom)."""
    cache_dir = tmp_path / "crtsh"
    monkeypatch.setattr(footprint, "CRTSH_CACHE", cache_dir)

    def handler(request):
        return httpx.Response(200, text="[]")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        subs = footprint.crtsh_subdomains("example.com", client)
    assert subs == []
    assert not (cache_dir / "example.com.json").exists()


def test_crtsh_subdomains_caches_a_real_result_and_reuses_it(monkeypatch, tmp_path):
    cache_dir = tmp_path / "crtsh"
    monkeypatch.setattr(footprint, "CRTSH_CACHE", cache_dir)
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=[{"name_value": "api.example.com"}])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        first = footprint.crtsh_subdomains("example.com", client)
        second = footprint.crtsh_subdomains("example.com", client)
    assert first == second == ["api.example.com"]
    assert len(calls) == 1  # second call served from cache
    assert (cache_dir / "example.com.json").exists()
