"""GitHub provider: org-link gating, IaC root-signal classification, STRONG-only-for-
provider-block-in-a-non-sample-repo, and graceful degradation - offline via
httpx.MockTransport. Fixtures are reconstructed from the real Zapier org/repo shapes
measured 2026-09-19 (raw/github__zapier.json); the raw capture itself is body-truncated
past ~2.7KB so it isn't valid JSON to replay directly - these fixtures keep the same
field names and the same repo/signal facts (`kubechecks` has Dockerfile + workflows,
zapier-platform-example-app-custom-auth is a sample-named repo) from PROVIDERS.md's
`extracted` summary of that run.
"""
import json
from pathlib import Path

import httpx

from leadscout.providers import github

FIXTURES = Path(__file__).parent / "fixtures" / "github"


def _load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _patch_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(github.httpx, "Client", fake_client)


def test_find_org_from_homepage_link():
    assert github.find_org("footer", '<a href="https://github.com/zapier">GitHub</a>') == "zapier"


def test_find_org_ignores_non_org_github_paths():
    assert github.find_org('<a href="https://github.com/pricing">Pricing</a>') is None


def test_find_org_from_hrefs_when_visible_text_has_no_url():
    """Item 1: a "GitHub" footer link's visible text carries no URL at all on a
    crawled subpage (trafilatura keeps "GitHub", not the href) - the raw href list
    must still resolve the org."""
    assert github.find_org("GitHub", "Follow us", hrefs=["https://github.com/zapier"]) == "zapier"


def test_find_org_hrefs_skip_non_org_paths_too():
    assert github.find_org(hrefs=["https://github.com/pricing", "https://github.com/zapier"]) == "zapier"


def test_find_org_prefers_non_lowercase_spelling_of_the_same_org():
    """A properly-cased mention of the SAME org (e.g. a careers page's "star us
    on GitHub" CTA) wins over an earlier all-lowercase mention (e.g. an about
    page's generic footer link) - GitHub's API is case-insensitive either way,
    but the properly-cased form is more often the org's real display name."""
    assert github.find_org(hrefs=[
        "https://posthog.com/about", "https://github.com/posthog",
        "https://posthog.com/careers", "https://github.com/PostHog",
    ]) == "PostHog"


def test_find_org_keeps_first_distinct_org_even_with_later_casing_variants():
    """The case-preference rule only applies within the SAME org - a genuinely
    different org link later in the list must never override the first one."""
    assert github.find_org(hrefs=["https://github.com/acme", "https://github.com/OtherOrg"]) == "acme"


def test_no_link_skips_any_api_call(monkeypatch):
    def handler(request):
        raise AssertionError("must not call the GitHub API without a detected org link")

    _patch_client(monkeypatch, handler)
    result = github.run("Some Co", ["plain landing page, no github link"])
    assert result.status == "ok"
    assert result.evidence == []
    assert result.calls == 0


def test_zapier_org_kubechecks_medium_and_example_repo_not_strong(monkeypatch):
    """The measured shape: an org that IS linked, a repo (kubechecks) with
    Dockerfile + CI workflows (MEDIUM, no .tf) and a sample-named repo that must
    never reach STRONG even if it had a .tf (it doesn't, in this fixture)."""
    org = _load("org_zapier.json")
    repos = _load("repos_zapier.json")
    contents = {
        "kubechecks": _load("contents_kubechecks_root.json"),
        "zapier-mcp": [],
        "connectors": [],
        "infra-terraform": _load("contents_infra_terraform_root.json"),
        "zapier-platform-example-app-custom-auth": _load("contents_example_repo_root.json"),
    }
    workflows = {"kubechecks": _load("contents_kubechecks_workflows.json")}
    tf_body = (FIXTURES / "main_tf_provider_aws.txt").read_text(encoding="utf-8")

    def handler(request):
        url = str(request.url)
        if url.endswith("/orgs/zapier"):
            return httpx.Response(200, json=org)
        if "orgs/zapier/repos" in url:
            return httpx.Response(200, json=repos)
        if url.endswith(".github/workflows"):
            repo = url.split("/repos/zapier/")[1].split("/contents")[0]
            return httpx.Response(200, json=workflows.get(repo, []))
        if "raw.githubusercontent.com" in url:
            return httpx.Response(200, text=tf_body)
        if "/contents/" in url:
            repo = url.split("/repos/zapier/")[1].split("/contents")[0]
            return httpx.Response(200, json=contents.get(repo, []))
        raise AssertionError(f"unexpected call to {url}")

    _patch_client(monkeypatch, handler)
    result = github.run("Zapier", ["landing", '<a href="https://github.com/zapier">GitHub</a>'])
    assert result.status == "ok"
    by_repo = {e.url.rsplit("/", 1)[1]: e for e in result.evidence}

    assert by_repo["kubechecks"].strength == "MEDIUM"
    assert by_repo["kubechecks"].family == "engineering_footprint"
    assert "zapier-platform-example-app-custom-auth" not in by_repo or \
        by_repo["zapier-platform-example-app-custom-auth"].strength != "STRONG"
    # repos with no root signals (zapier-mcp, connectors) yield no evidence
    assert "zapier-mcp" not in by_repo
    assert "connectors" not in by_repo


def test_non_sample_repo_with_provider_block_is_strong(monkeypatch):
    org = _load("org_zapier.json")
    repos = [{"name": "infra-terraform", "language": "HCL"}]
    contents_root = _load("contents_infra_terraform_root.json")
    tf_body = (FIXTURES / "main_tf_provider_aws.txt").read_text(encoding="utf-8")

    def handler(request):
        url = str(request.url)
        if url.endswith("/orgs/zapier"):
            return httpx.Response(200, json=org)
        if "orgs/zapier/repos" in url:
            return httpx.Response(200, json=repos)
        if "raw.githubusercontent.com" in url:
            return httpx.Response(200, text=tf_body)
        if "/contents/" in url:
            return httpx.Response(200, json=contents_root)
        raise AssertionError(f"unexpected call to {url}")

    _patch_client(monkeypatch, handler)
    result = github.run("Zapier", ['<a href="https://github.com/zapier">GitHub</a>'])
    assert len(result.evidence) == 1
    assert result.evidence[0].strength == "STRONG"


def test_sample_repo_with_tf_never_strong(monkeypatch):
    org = _load("org_zapier.json")
    repos = [{"name": "acme-example", "language": "HCL"}]
    contents_root = [
        {"name": "main.tf", "path": "main.tf", "type": "file",
         "download_url": "https://raw.githubusercontent.com/zapier/acme-example/main/main.tf"},
    ]
    tf_body = (FIXTURES / "main_tf_provider_aws.txt").read_text(encoding="utf-8")

    def handler(request):
        url = str(request.url)
        if url.endswith("/orgs/zapier"):
            return httpx.Response(200, json=org)
        if "orgs/zapier/repos" in url:
            return httpx.Response(200, json=repos)
        if "raw.githubusercontent.com" in url:
            return httpx.Response(200, text=tf_body)
        if "/contents/" in url:
            return httpx.Response(200, json=contents_root)
        raise AssertionError(f"unexpected call to {url}")

    _patch_client(monkeypatch, handler)
    result = github.run("Zapier", ['<a href="https://github.com/zapier">GitHub</a>'])
    assert len(result.evidence) == 1
    assert result.evidence[0].strength == "MEDIUM"


def test_org_not_linked_makes_zero_calls():
    result = github.run("Artizan Bakery", ["no github link on this homepage"])
    assert result.calls == 0
    assert result.status == "ok"


def test_rate_limit_degrades_without_exception(monkeypatch):
    def handler(request):
        return httpx.Response(403, text="rate limited")

    _patch_client(monkeypatch, handler)
    result = github.run("Zapier", ['<a href="https://github.com/zapier">GitHub</a>'])
    assert result.status == "degraded"
    assert result.evidence == []


def test_network_error_degrades_with_no_evidence(monkeypatch):
    def boom(*args, **kwargs):
        raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr(github.httpx, "Client", boom)
    result = github.run("Zapier", ['<a href="https://github.com/zapier">GitHub</a>'])
    assert result.status == "degraded"
    assert result.evidence == []


def test_malformed_json_response_degrades_with_no_evidence(monkeypatch):
    """A 200 with an invalid JSON body must not crash research_lead - the provider
    boundary catches Exception broadly, not only httpx.HTTPError
    (REVIEW-6bA-verified.md #2)."""
    org = _load("org_zapier.json")

    def handler(request):
        url = str(request.url)
        if url.endswith("/orgs/zapier"):
            return httpx.Response(200, json=org)
        if "orgs/zapier/repos" in url:
            return httpx.Response(200, headers={"content-type": "application/json"}, text="not json{{{")
        raise AssertionError(f"unexpected call to {url}")

    _patch_client(monkeypatch, handler)
    result = github.run("Zapier", ['<a href="https://github.com/zapier">GitHub</a>'])
    assert result.status == "degraded"
    assert result.evidence == []


def test_repo_scan_rate_limit_discards_evidence_already_collected(monkeypatch):
    """One repo already yielded evidence before a later repo 403s mid-scan: the
    ProviderResult must still discard ALL evidence on degraded, per the invariant
    "degraded => evidence == []" (REVIEW-6bA-verified.md #6) - a partial success is
    not a partial result."""
    org = _load("org_zapier.json")
    repos = [{"name": "kubechecks", "language": "Go"}, {"name": "later-repo", "language": "Go"}]
    kubechecks_root = _load("contents_kubechecks_root.json")
    kubechecks_workflows = _load("contents_kubechecks_workflows.json")

    def handler(request):
        url = str(request.url)
        if url.endswith("/orgs/zapier"):
            return httpx.Response(200, json=org)
        if "orgs/zapier/repos" in url:
            return httpx.Response(200, json=repos)
        if url.endswith("kubechecks/contents/.github/workflows"):
            return httpx.Response(200, json=kubechecks_workflows)
        if url.endswith("kubechecks/contents/"):
            return httpx.Response(200, json=kubechecks_root)
        if url.endswith("later-repo/contents/"):
            return httpx.Response(403, text="rate limited")
        raise AssertionError(f"unexpected call to {url}")

    _patch_client(monkeypatch, handler)
    result = github.run("Zapier", ['<a href="https://github.com/zapier">GitHub</a>'])
    assert result.status == "degraded"
    assert result.evidence == []
