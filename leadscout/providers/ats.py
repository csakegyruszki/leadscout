"""ATS (applicant tracking system) job board detection: Greenhouse, Lever, Ashby.

Never guesses a slug (measured 2026-09-19: a guessed slug 404s for all five sample
companies - see the design notes, "Providers"): this provider only calls a board's
public API when a link to that specific board was actually found on the fetched
pages.
"""
from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime

import httpx

from ..config import settings
from ..models import Evidence, ProviderResult
from ._base import UA, fallback_evidence_id, snapshot_path

_LINK_PATTERNS = {
    "greenhouse": re.compile(r"(?:boards|job-boards)\.greenhouse\.io/([a-zA-Z0-9_-]+)", re.I),
    "lever": re.compile(r"jobs\.lever\.co/([a-zA-Z0-9_-]+)", re.I),
    "ashby": re.compile(r"jobs\.ashbyhq\.com/([a-zA-Z0-9_-]+)", re.I),
}

# A named cloud provider must be present for this to be evidence at all
# (REVIEW-6bA-verified.md #7): Kubernetes/Terraform/SRE text alone, with no provider
# token, used to emit evidence via _INFRA_TERMS - fixed by requiring a provider match
# first and only using role/workload words to decide STRONG vs MEDIUM.
_PROVIDER_TOKENS = {
    "aws": "AWS", "amazon web services": "AWS",
    "azure": "Azure",
    "gcp": "GCP", "google cloud": "GCP",
    "oracle cloud": "OCI", "oci": "OCI",
    "eks": "AWS", "aks": "Azure", "gke": "GCP",
}
_WORKLOAD_WORDS = ("engineer", "sre", "site reliability", "platform", "devops", "infrastructure",
                   "finops", "kubernetes", "terraform", "gpu", "data platform", "kafka",
                   "snowflake", "databricks")

# The real, public board URL for a job - not a placeholder - so Evidence.url points
# somewhere a reader can actually verify the posting.
_BOARD_URL = {
    "greenhouse": "https://boards.greenhouse.io/{slug}",
    "lever": "https://jobs.lever.co/{slug}",
    "ashby": "https://jobs.ashbyhq.com/{slug}",
}


def find_board_links(*texts: str, hrefs: list[str] | tuple[str, ...] | None = None) -> dict[str, str]:
    """Board name -> slug, checking `hrefs` first, `texts` as a fallback.

    `hrefs` (actual <a href> targets, any host - see providers/website.py's
    `extract_hrefs`/`crawl_subpages`) is where a board link actually lives on most
    pages: the visible anchor text usually just says "Careers" or "Jobs", so
    trafilatura's extracted TEXT (used for `texts` on crawled subpages) never
    contains the URL at all. `texts` stays as a fallback for callers that pass raw
    HTML directly (the landing page currently does) or a board URL that happens to
    appear as plain text.
    """
    found: dict[str, str] = {}
    for board, pattern in _LINK_PATTERNS.items():
        for href in hrefs or ():
            m = pattern.search(href)
            if m:
                found[board] = m.group(1)
                break
    joined = "\n".join(t for t in texts if t)
    for board, pattern in _LINK_PATTERNS.items():
        if board in found:
            continue
        m = pattern.search(joined)
        if m:
            found[board] = m.group(1)
    return found


def _fetch_jobs(board: str, slug: str, client: httpx.Client) -> tuple[int, list[dict], bytes]:
    """Returns (http_status, jobs[{title,text}], raw_bytes) for one board's API.

    `raw_bytes` (`r.content`) is captured immediately on receipt, before any JSON
    parsing - every derived Evidence's `content_sha256` then traces back to the
    exact bytes that were on the wire, not a python re-serialisation of an
    already-parsed dict (REVIEW-6bA-verified.md #1)."""
    if board == "greenhouse":
        r = client.get(f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs", params={"content": "true"})
        raw_bytes = r.content
        body = r.json() if "json" in r.headers.get("content-type", "") else {}
        jobs = body.get("jobs", []) if r.status_code == 200 else []
        return r.status_code, [{"title": j.get("title", ""), "text": j.get("content", "")} for j in jobs], raw_bytes
    if board == "lever":
        r = client.get(f"https://api.lever.co/v0/postings/{slug}", params={"mode": "json"})
        raw_bytes = r.content
        body = r.json() if "json" in r.headers.get("content-type", "") else {}
        jobs = body if r.status_code == 200 and isinstance(body, list) else []
        lever_jobs = [
            {"title": j.get("text", ""), "text": str(j.get("descriptionPlain", ""))} for j in jobs
        ]
        return r.status_code, lever_jobs, raw_bytes
    if board == "ashby":
        r = client.get(f"https://api.ashbyhq.com/posting-api/job-board/{slug}")
        raw_bytes = r.content
        body = r.json() if "json" in r.headers.get("content-type", "") else {}
        jobs_raw = body.get("jobs", []) if r.status_code == 200 else []
        ashby_jobs = [{"title": j.get("title", ""), "text": j.get("descriptionPlain", "")} for j in jobs_raw]
        return r.status_code, ashby_jobs, raw_bytes
    return 404, [], b""


def _job_signal(text: str) -> tuple[str, str] | None:
    """(provider, strength), or None if no named cloud provider token is present -
    a job merely mentioning Kubernetes/Terraform/SRE is not evidence by itself
    (REVIEW-6bA-verified.md #7). STRONG only when a provider token AND a
    role/workload word co-occur; otherwise MEDIUM."""
    low = text.lower()
    provider = next((p for token, p in _PROVIDER_TOKENS.items() if token in low), None)
    if provider is None:
        return None
    strength = "STRONG" if any(w in low for w in _WORKLOAD_WORDS) else "MEDIUM"
    return provider, strength


def run(company: str, texts: list[str], hrefs: list[str] | None = None, run=None) -> ProviderResult:
    """`texts` are the fetched pages (landing + crawled subpages) to search for a
    board link; `hrefs` are the actual <a href> targets collected across those same
    pages (see providers/website.py). Only a found link triggers an API call - see
    module docstring."""
    links = find_board_links(*texts, hrefs=hrefs)
    if not links:
        return ProviderResult(provider_name="ats", status="skipped", note="no ATS board link found")

    evidence: list[Evidence] = []
    calls = 0
    try:
        with httpx.Client(timeout=settings.http_timeout, headers={"User-Agent": UA}) as client:
            for board, slug in links.items():
                status, jobs, raw_bytes = _fetch_jobs(board, slug, client)
                calls += 1
                if status != 200:
                    continue
                # The raw response bytes are the actual snapshot content for every
                # job derived from it - not a re-dump of the already-parsed dict
                # (REVIEW-6bA-verified.md #1).
                content_sha256 = hashlib.sha256(raw_bytes).hexdigest()
                for job in jobs:
                    signal = _job_signal(job["text"])
                    if signal is None:
                        continue
                    provider, strength = signal
                    eid = fallback_evidence_id("ats", run, len(evidence) + 1)
                    ev = Evidence(
                        id=eid, source_type="ats", url=_BOARD_URL[board].format(slug=slug),
                        observed_at=datetime.now(UTC).isoformat(),
                        content_sha256=content_sha256, strength=strength,
                        snippet=f"{job['title']}: {job['text'][:150]}"[:200],
                        snapshot_path=snapshot_path(run, eid),
                        family="ats_hiring", provider=provider, scope="workload",
                        freshness="current", origin="provider:ats",
                    )
                    evidence.append(ev)
                    if run is not None:
                        run.record(ev, raw_bytes, stage="ats")
    except Exception as e:  # noqa: BLE001 - a malformed/unreachable board must degrade, never crash the pipeline
        return ProviderResult(provider_name="ats", status="degraded", note=f"{type(e).__name__}: {e}", calls=calls)

    note = f"boards checked: {list(links)}; {len(evidence)} evidence job(s) naming a provider"
    return ProviderResult(provider_name="ats", status="ok", evidence=evidence, note=note, calls=calls)
