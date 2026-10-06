"""GitHub engineering-footprint detection: only when the site itself links a
github.com/<org> - never guessed. Unauthenticated GitHub calls are capped at 15
(measured 2026-09-19: Zapier org, 298 repos, 12 repos checked - 19 calls with an
authenticated token; we keep a hard budget so an unauthenticated run degrades
gracefully instead of hitting the 60/hour anonymous rate limit).
"""

from __future__ import annotations

import difflib
import hashlib
import re
from datetime import UTC, datetime
from urllib.parse import urlsplit

import httpx

from ..config import settings
from ..models import Evidence, ProviderResult
from ._base import UA, fallback_evidence_id, snapshot_path

# --- Design notes -----------------------------------------------------------------
# IaC in a repo root is `engineering_footprint` evidence: MEDIUM for a Dockerfile,
# terraform/helm/k8s directory, or a `.github/workflows` CI setup; STRONG only for a
# real cloud-provider block inside a root `.tf` file (`provider "aws"|"azurerm"|
# "google"` or a matching *_cluster/*_bucket resource), and never STRONG in a repo
# whose name looks like a sample/tutorial (example|sample|demo|template|starter) -
# those exist to be copied, not to describe the org's own infrastructure.

API = "https://api.github.com"

MAX_CALLS = 15
MAX_REPOS_CHECKED = 8

_ORG_LINK = re.compile(r"github\.com/([A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))(?:[/\"'\s<]|$)")
# Paths GitHub itself uses that are never an organisation account.
_NON_ORG_SEGMENTS = {
    "login", "join", "about", "sponsors", "marketplace", "topics", "collections",
    "trending", "features", "pricing", "security", "enterprise", "settings",
    "notifications", "issues", "pulls", "orgs", "apps", "site",
}
_SAMPLE_NAME = re.compile(r"(example|sample|demo|template|starter)", re.I)
_IAC_DIRS = {"terraform", "helm", "k8s", "kubernetes"}
_TF_PROVIDER = re.compile(r'provider\s+"(aws|azurerm|google)"', re.I)
_TF_RESOURCE = re.compile(
    r"\b(aws_eks_cluster|aws_s3_bucket|azurerm_kubernetes_cluster|google_container_cluster)\b")


def find_org(*texts: str, hrefs: list[str] | tuple[str, ...] | None = None) -> str | None:
    """First github.com/<org> link, checking `hrefs` first, `texts` as a fallback.

    A "GitHub" nav/footer link's visible text rarely contains the URL itself, so
    trafilatura-extracted `texts` (used for crawled subpages) misses it; `hrefs`
    (see providers/website.py's `extract_hrefs`/`crawl_subpages`) is the actual
    <a href> targets. `texts` stays as a fallback for raw HTML passed directly
    (the landing page currently is) or an org URL appearing as plain text.

    When more than one href names the SAME org (case-insensitively) with
    different casing, a non-all-lowercase spelling wins over an all-lowercase
    one - GitHub's own org slug is case-insensitive for the API call this feeds
    into either way, but a casual footer/nav link is more often written in
    lowercase than the org's own "star us on GitHub" CTA, which typically
    preserves its real display name (measured 2026-09-19: a real site linked
    both "github.com/acme" from an /about footer and the properly-cased
    "github.com/Acme" from its /careers and handbook pages)."""
    first: str | None = None
    for href in hrefs or ():
        m = _ORG_LINK.search(href)
        if not m or m.group(1).lower() in _NON_ORG_SEGMENTS:
            continue
        candidate = m.group(1)
        if first is None:
            first = candidate
        elif candidate.lower() == first.lower() and first.islower() and not candidate.islower():
            first = candidate
    if first is not None:
        return first
    joined = "\n".join(t for t in texts if t)
    for m in _ORG_LINK.finditer(joined):
        candidate = m.group(1)
        if candidate.lower() not in _NON_ORG_SEGMENTS:
            return candidate
    return None


def _headers() -> dict:
    h = {"User-Agent": UA, "Accept": "application/vnd.github+json"}
    if settings.github_token:
        h["Authorization"] = f"Bearer {settings.github_token}"
    return h


def _get(client: httpx.Client, url: str, calls: list[int]) -> httpx.Response:
    calls[0] += 1
    return client.get(url)


def _is_sample_repo(name: str) -> bool:
    return bool(_SAMPLE_NAME.search(name))


def _org_verified(org_json: dict, company: str) -> bool:
    """Whether the GitHub org itself (not just a link to it) can be tied back to
    the company, for infra.py's PRIVATE_CLOUD rule 5 ("an unverified organisation
    caps confidence at MEDIUM"). "Verified" here means the org's own profile says
    so: its `blog`/website field's host resembles the company name, or its
    display `name` is a close match to the company name - never guessed from the
    mere fact that the site links to *a* github.com/<org> (that's already required
    just to call find_org() at all, see the module docstring)."""
    name = str(org_json.get("name") or "").strip().lower()
    company_l = company.strip().lower()
    if name and difflib.SequenceMatcher(None, name, company_l).ratio() >= 0.6:
        return True
    blog = str(org_json.get("blog") or "").strip()
    if blog:
        host = urlsplit(blog if "://" in blog else f"//{blog}").netloc.lower()
        host = host.removeprefix("www.")
        tokens = [t for t in re.split(r"[^a-z0-9]+", company_l) if len(t) >= 3]
        if host and any(t in host for t in tokens):
            return True
    return False


def _attach(result: ProviderResult, *, org: str | None, repos: list[dict] | None,
           org_verified: bool) -> ProviderResult:
    """Stashes repo names+descriptions (and org-verification) on the returned
    `ProviderResult` without changing its declared shape (models.ProviderResult has
    no generic detail field) - the same pattern footprint.py uses for
    `Evidence._detail`. They are recorded here so that the PRIVATE_CLOUD conjunction
    infra.py describes (rule 5, "an unverified organisation ...") can read them via
    `getattr(result, "_repos", [])` rather than re-fetch the org, and so that the
    evidence-producing HTTP calls above stay the only ones made.

    NOT YET CONSUMED, as of this writing: infra.py's rule 5 is explicitly deferred
    (see its module docstring and the TODO at infra.py:125), and nothing outside this
    module reads `_repos` or `_org_verified`. So `org_verified` is computed and stored
    but does not currently gate anything - an unverified organisation's
    `engineering_footprint` evidence reaches cloud.py exactly like a verified one's.
    An earlier version of this docstring asserted the consumer already existed; it did
    not, and describing a deferred consumer as a live one is the kind of claim this
    project is supposed to avoid making about itself."""
    result._repos = repos or []  # type: ignore[attr-defined]
    result._org = org  # type: ignore[attr-defined]
    result._org_verified = org_verified  # type: ignore[attr-defined]
    return result


def _root_signals(entries: list[dict]) -> tuple[set[str], bool]:
    """Returns (signals, has_github_dir) - the caller fetches .github/workflows
    itself (one more call) only when has_github_dir is True."""
    names = {e.get("name", "") for e in entries if isinstance(e, dict)}
    signals = set()
    if "Dockerfile" in names:
        signals.add("Dockerfile")
    signals |= {n for n in names if n.lower() in _IAC_DIRS}
    for e in entries:
        if isinstance(e, dict) and e.get("name", "").endswith(".tf"):
            signals.add(e["name"])
    return signals, ".github" in names


def run(company: str, texts: list[str], hrefs: list[str] | None = None, run=None) -> ProviderResult:
    org = find_org(*texts, hrefs=hrefs)
    if not org:
        return _attach(ProviderResult(provider_name="github", status="ok", note="no github.com org link found"),
                       org=None, repos=[], org_verified=False)

    # A GITHUB_TOKEN raises the unauthenticated 60/hour rate limit to 5000/hour
    # (measured 2026-09-19), so the same-run call budget can afford to look at more
    # repos/files without risking a mid-scan 403.
    max_calls = 20 if settings.github_token else MAX_CALLS

    evidence: list[Evidence] = []
    calls = [0]
    try:
        with httpx.Client(timeout=settings.http_timeout, headers=_headers()) as client:
            r = _get(client, f"{API}/orgs/{org}", calls)
            if r.status_code in (403, 429):
                return _attach(ProviderResult(provider_name="github", status="degraded",
                                       note=f"org lookup rate-limited (HTTP {r.status_code})", calls=calls[0]),
                               org=org, repos=[], org_verified=False)
            # 404 is the one non-200 that MEANS something: GitHub's `GET /orgs/{org}`
            # answers 404 when the organisation does not exist, so that is a completed
            # measurement with an empty result. Everything else - 401 on bad credentials,
            # 400, 5xx - is a measurement that did not happen, and reporting it as
            # `ok`/"org not found" made a failed lookup indistinguishable from a real
            # negative (measured: 401, 400, 500 and 503 all returned "org someco not
            # found"). The rate-limit codes are handled above and stay `degraded`.
            if r.status_code == 404:
                return _attach(ProviderResult(provider_name="github", status="ok",
                                       note=f"org {org} not found", calls=calls[0]),
                               org=org, repos=[], org_verified=False)
            if r.status_code != 200:
                return _attach(ProviderResult(provider_name="github", status="degraded",
                                       note=f"org lookup failed (HTTP {r.status_code})", calls=calls[0]),
                               org=org, repos=[], org_verified=False)
            # A 200 carrying something other than usable JSON is not a completed
            # measurement either - it used to fall through as `{}`, i.e. an org that
            # exists but reveals nothing, which is the same "absence became evidence"
            # shape as the non-200 branch above (found by the provider-contract matrix,
            # not by a provider-specific test).
            try:
                org_json = r.json() if "json" in r.headers.get("content-type", "") else None
            except ValueError:
                org_json = None
            if not isinstance(org_json, dict):
                return _attach(ProviderResult(provider_name="github", status="degraded",
                                       note="org lookup returned an unusable body", calls=calls[0]),
                               org=org, repos=[], org_verified=False)
            org_verified = _org_verified(org_json, company)

            rr = _get(client, f"{API}/orgs/{org}/repos?per_page=10&sort=updated", calls)
            repos = rr.json() if rr.status_code == 200 and "json" in rr.headers.get("content-type", "") else []
            if not isinstance(repos, list):
                repos = []
            repo_summaries = [{"name": rp.get("name", ""), "description": rp.get("description") or ""}
                              for rp in repos if isinstance(rp, dict) and rp.get("name")]

            for repo in repos:
                if calls[0] >= max_calls:
                    break
                name = repo.get("name", "")
                if not name:
                    continue
                cr = _get(client, f"{API}/repos/{org}/{name}/contents/", calls)
                if cr.status_code == 403:
                    # A failed provider yields NO evidence - never partial evidence
                    # collected before the failure (REVIEW-6bA-verified.md #6).
                    return _attach(ProviderResult(provider_name="github", status="degraded",
                                           note="rate-limited during repo scan", calls=calls[0]),
                                   org=org, repos=repo_summaries, org_verified=org_verified)
                if cr.status_code != 200:
                    continue
                # Captured before parsing (REVIEW-6bA-verified.md #1): this repo's
                # Evidence.content_sha256 traces back to the exact directory-listing
                # bytes GitHub returned, not a locally re-serialised summary dict.
                cr_raw = cr.content
                cr_body = cr.json()
                entries = cr_body if isinstance(cr_body, list) else []
                signals, has_github_dir = _root_signals(entries)
                if has_github_dir and calls[0] < max_calls:
                    wr = _get(client, f"{API}/repos/{org}/{name}/contents/.github/workflows", calls)
                    if wr.status_code == 200 and isinstance(wr.json(), list) and wr.json():
                        signals.add(".github/workflows")
                if not signals:
                    continue

                tf_files = [s for s in signals if s.endswith(".tf")]
                strength = "MEDIUM"
                detail = f"root signals: {sorted(signals)}"
                if tf_files and not _is_sample_repo(name) and calls[0] < max_calls:
                    tf_entry = next(e for e in entries if e.get("name") == tf_files[0])
                    dl = tf_entry.get("download_url")
                    if dl:
                        calls[0] += 1
                        tf_resp = client.get(dl)
                        if tf_resp.status_code == 200 and (
                            _TF_PROVIDER.search(tf_resp.text) or _TF_RESOURCE.search(tf_resp.text)
                        ):
                            strength = "STRONG"
                            detail = f"{tf_files[0]} declares a cloud provider block"

                eid = fallback_evidence_id("github", run, len(evidence) + 1)
                ev = Evidence(
                    id=eid, source_type="github", url=f"https://github.com/{org}/{name}",
                    observed_at=datetime.now(UTC).isoformat(),
                    content_sha256=hashlib.sha256(cr_raw).hexdigest(), strength=strength,
                    snippet=f"{org}/{name}: {detail}"[:200],
                    snapshot_path=snapshot_path(run, eid),
                    family="engineering_footprint", freshness="current", origin="provider:github",
                )
                evidence.append(ev)
                if run is not None:
                    run.record(ev, cr_raw, stage="github")
                if len(evidence) >= MAX_REPOS_CHECKED:
                    break
    except Exception as e:  # noqa: BLE001 - a malformed/unreachable GitHub response must degrade, never crash
        return _attach(
            ProviderResult(provider_name="github", status="degraded",
                          note=f"{type(e).__name__}: {e}", calls=calls[0]),
            org=org, repos=locals().get("repo_summaries", []),
            org_verified=locals().get("org_verified", False),
        )

    note = f"org {org}, {calls[0]} calls, {len(evidence)} IaC evidence item(s)"
    return _attach(ProviderResult(provider_name="github", status="ok", evidence=evidence, note=note, calls=calls[0]),
                   org=org, repos=repo_summaries, org_verified=org_verified)
