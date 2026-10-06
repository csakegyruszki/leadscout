"""Vendor case-study discovery via Brave web search - discovery only, never trusted
on its own; a search hit alone is never evidence (see design notes below for what
promotes it to `vendor_case_study` DIRECT). No `BRAVE_API_KEY` -> skipped, never
degraded (optional discovery step, not a required one).
"""
from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from html import unescape
from urllib.parse import urlparse

import httpx

from ..compliance import normalise_name
from ..config import settings
from ..models import Evidence, ProviderResult
from ..util import UnsafeTargetError, safe_get
from ._base import UA, fallback_evidence_id, snapshot_path

# --- Design notes -----------------------------------------------------------------
# A search hit is NOT evidence: only a page actually fetched, on an allowlisted
# vendor-owned host, that WHOLE-WORD names the company (never a substring match -
# "Snapp" must not match "Snappet"), whose title or first H1 is about that company
# (not a customer wall that merely lists the name among many logos), is promoted to
# `vendor_case_study` DIRECT. A company name with <= 1 distinctive token ("Snapp",
# "Zapier") also requires the lead's own registrable domain to appear on the page,
# since a single short token is cheap to false-match on an unrelated customer page.
# This guards against a reseller/aggregator blog that merely talks *about*
# AWS/Azure/GCP/OCI, a customer-wall page whose real subject is a different
# company, and a search engine's own query-term echo. At most 4 queries per lead,
# one per major cloud vendor.

API = "https://api.search.brave.com/res/v1/web/search"

_ALLOWLIST = {"aws.amazon.com", "cloud.google.com", "oracle.com", "customers.microsoft.com", "microsoft.com"}
_PROVIDER_BY_HOST = {
    "aws.amazon.com": "AWS", "cloud.google.com": "GCP",
    "oracle.com": "OCI", "customers.microsoft.com": "Azure", "microsoft.com": "Azure",
}

_HISTORICAL_YEARS = 3

_DATE_PATTERNS = (
    re.compile(r'<time[^>]*datetime="([^"]+)"', re.I),
    re.compile(r'property="article:published_time"[^>]*content="([^"]+)"', re.I),
    re.compile(r'content="([^"]+)"[^>]*property="article:published_time"', re.I),
    re.compile(r"Published\s*[:\-]?\s*([A-Za-z]+ \d{1,2},? \d{4})"),
)


def _query_templates(company: str) -> list[str]:
    return [
        f'"{company}" site:aws.amazon.com/solutions/case-studies',
        f'"{company}" site:cloud.google.com/customers',
        f'"{company}" site:oracle.com/customers',
        f'site:microsoft.com "{company}" Azure customer',
    ]


def _scrub(text: str) -> str:
    """Strip the Brave key from anything that could reach a note/log (httpx can echo headers/urls)."""
    key = settings.brave_api_key
    return text.replace(key, "<redacted>") if key else text


def _brave_search(query: str, client: httpx.Client) -> list[dict]:
    r = client.get(API, params={"q": query}, headers={"X-Subscription-Token": settings.brave_api_key})
    if r.status_code != 200:
        return []
    return r.json().get("web", {}).get("results", [])


def _parse_published(html: str) -> str | None:
    for pattern in _DATE_PATTERNS:
        m = pattern.search(html)
        if m:
            return m.group(1)
    return None


def _freshness_for(published: str | None) -> str:
    if not published:
        return "unknown"
    date_str = published.strip()
    dt = None
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")  # ISO date or datetime prefix
    except ValueError:
        for fmt in ("%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y"):
            try:
                dt = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue
    if dt is None:
        return "unknown"
    age_years = (datetime.now(UTC).replace(tzinfo=None) - dt).days / 365.25
    return "historical" if age_years > _HISTORICAL_YEARS else "current"


_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.I | re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
_H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.I | re.S)


def _visible_text(html: str) -> str:
    """Strip script/style blocks and tags - a company-name match on raw HTML can
    hit an unrelated attribute, comment, or script literal."""
    stripped = _SCRIPT_STYLE_RE.sub(" ", html)
    return unescape(_TAG_RE.sub(" ", stripped))


def _title_and_h1(html: str) -> tuple[str, str]:
    title_m = _TITLE_RE.search(html)
    h1_m = _H1_RE.search(html)
    title = unescape(_TAG_RE.sub(" ", title_m.group(1))) if title_m else ""
    h1 = unescape(_TAG_RE.sub(" ", h1_m.group(1))) if h1_m else ""
    return title, h1


def _whole_name_regex(company_norm: str) -> re.Pattern:
    tokens = company_norm.split()
    return re.compile(r"\b" + r"\s+".join(re.escape(t) for t in tokens) + r"\b")


def _company_confirmed(company_norm: str, html: str, domain: str) -> bool:
    """Whole-word match of the full company name in the visible text AND in the
    page's title or first H1 (rejects a customer-wall page whose real subject is a
    different company but that happens to list this one among many logos). A
    single-distinctive-token name additionally requires the lead's own registrable
    domain to appear on the page."""
    name_re = _whole_name_regex(company_norm)
    visible_norm = normalise_name(_visible_text(html))
    if not name_re.search(visible_norm):
        return False
    title, h1 = _title_and_h1(html)
    if not (name_re.search(normalise_name(title)) or name_re.search(normalise_name(h1))):
        return False
    if len(company_norm.split()) <= 1 and domain and domain not in html.lower():
        return False
    return True


def _finalise(ev: Evidence, raw: bytes, run) -> Evidence:
    eid = fallback_evidence_id("vendor", run, id(ev))
    ev.id = eid
    ev.content_sha256 = hashlib.sha256(raw).hexdigest()
    ev.snapshot_path = snapshot_path(run, eid, "html")
    if run is not None:
        run.record(ev, raw, stage="vendor")
    return ev


def run(company: str, website_url: str = "", run=None) -> ProviderResult:
    if not settings.brave_api_key:
        return ProviderResult(provider_name="vendor", status="skipped", note="no BRAVE_API_KEY")

    company_norm = normalise_name(company)
    domain = urlparse(website_url).netloc.lower() if website_url else ""
    if domain.startswith("www."):
        domain = domain[4:]
    evidence: list[Evidence] = []
    seen_urls: set[str] = set()
    calls = 0
    try:
        with httpx.Client(headers={"User-Agent": UA}, timeout=settings.http_timeout) as client:
            for query in _query_templates(company):
                results = _brave_search(query, client)
                calls += 1
                for result in results:
                    url = result.get("url", "")
                    host = urlparse(url).netloc.lower()
                    if host not in _ALLOWLIST or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    try:
                        page = safe_get(client, url)
                    except (httpx.HTTPError, UnsafeTargetError):
                        continue
                    calls += 1
                    # Re-validate on the FINAL host after redirects, not the search-result URL -
                    # a redirect can leave the allowlist entirely (e.g. a vanity/short link).
                    final_host = urlparse(str(page.url)).netloc.lower()
                    if final_host not in _ALLOWLIST:
                        continue
                    content_type = page.headers.get("content-type", "").lower()
                    if content_type and "html" not in content_type:
                        continue
                    if page.status_code != 200 or len(page.text) <= 500:
                        continue
                    if not _company_confirmed(company_norm, page.text, domain):
                        continue
                    published = _parse_published(page.text)
                    ev = Evidence(
                        id="", source_type="vendor", url=url, observed_at=datetime.now(UTC).isoformat(),
                        content_sha256="", strength="DIRECT", snippet=result.get("title", "")[:200],
                        snapshot_path="", family="vendor_case_study", provider=_PROVIDER_BY_HOST[host],
                        scope="workload", freshness=_freshness_for(published),
                        source_published_at=published, origin="provider:vendor",
                    )
                    evidence.append(_finalise(ev, page.text.encode("utf-8"), run))
                    break  # one accepted case study per query is enough
    except Exception as e:  # noqa: BLE001 - a malformed/unreachable response must degrade, never crash
        return ProviderResult(provider_name="vendor", status="degraded",
                               note=f"{type(e).__name__}: {e}", calls=calls)

    note = f"{calls} call(s), {len(evidence)} vendor_case_study evidence item(s)"
    return ProviderResult(provider_name="vendor", status="ok", evidence=evidence, note=note, calls=calls)
