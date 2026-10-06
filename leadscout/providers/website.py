"""Bounded same-origin crawl: after the landing page, follow up to 4 links whose
path looks like an "about the company" page, via the same httpx + trafilatura
extraction as the landing page, with hard time/page-count budgets. No headless
browser by default; `needs_js_render`/`maybe_render_with_crawl4ai` (bottom of this
module) detect a likely JS-rendered shell and can optionally escalate to Crawl4AI.
"""
from __future__ import annotations

import logging
import os
import re
import time
from urllib.parse import urljoin, urlparse

import httpx
import tldextract
import trafilatura

from ..util import USER_AGENT, UnsafeTargetError, assert_public_url, safe_get

logger = logging.getLogger("leadscout")

UA = USER_AGENT

# Priority order: the first pattern a link's path matches decides its priority;
# links are fetched in that order, first MAX_PAGES only. Trust/subprocessor/status
# pages (added for providers/trust_pages.py) rank alongside the engineering ones -
# a trust-center or subprocessor page naming AWS/Azure/GCP/OCI is at least as
# strong a cloud signal as a careers page.
def _segment(word: str) -> str:
    """`word` matched only as a whole path segment (or hyphen-suffixed variant,
    e.g. "/about-us"), never as a prefix of a longer word - `r"/team"` unanchored
    used to also match "/blog/teamwork/" or ".../integrations/teamwork-desk" (a
    real bug the old landing-page-only `select_subpages` never had enough URL
    variety to expose; Fix #6's much larger sitemap/frontier candidate pool did,
    measured 2026-09-19 against Zapier's sitemap: 180+ unrelated app-marketplace/
    blog-tag URLs matched "/team" alone before this fix)."""
    return rf"/{word}(?![a-zA-Z])"


_PATH_PATTERNS = [re.compile(_segment(p), re.I) for p in (
    "about", "company", "team", "careers", "jobs",
    "engineering", "technology", "platform", "security",
    "infrastructure",
    "trust(?:-center)?", "sub-?processors?", "privacy", "dpa",
    "status", "architecture", "legal", "compliance",
)] + [re.compile(r"/blog/engineering(?![a-zA-Z])", re.I)]
# Fix #6's own keyword list (v0.2 PART B): security, trust, subprocessor,
# sub-processor, privacy, legal, dpa, compliance, infrastructure, status - a
# NARROWER set than `_PATH_PATTERNS` above, used only for candidates discovered
# beyond the landing page itself (sitemap URLs, and links found on a page the
# crawl actually fetched). The landing page's own links keep the broader
# `_PATH_PATTERNS` scoring (unchanged since before this fix) - a "company"/"team"/
# "careers" landing link is a deliberate, bounded choice (at most a few dozen
# candidates); the SAME broad list applied to a whole sitemap or a fetched page's
# full href list is not (measured 2026-09-19: Zapier's blog alone has hundreds of
# "/blog/team-..."/"/blog/company-..." post slugs that are not trust/security
# signals - matching Fix #6's own keyword list here, not the wider landing set,
# keeps discovery scoped to what it's actually for).
_DISCOVERY_PATTERNS = [re.compile(_segment(p), re.I) for p in (
    "security", "trust(?:-center)?", "sub-?processors?", "privacy",
    "legal", "dpa", "compliance", "infrastructure", "status",
)]
# A discovery-keyword match under one of these top-level sections is never a
# first-party trust statement, however the keyword happened to match - a blog
# post ("/blog/security-behaviors"), a user-submitted forum question
# ("/questions/security"), or a job posting whose role title happens to contain
# the word ("/careers/security-engineer") are all content ABOUT the topic, not
# the company's own authoritative statement.
_NON_AUTHORITATIVE_SECTIONS = frozenset({
    "blog", "questions", "careers", "jobs", "community", "forum", "tag", "tags", "news", "press",
})
_HIRING_OR_ENG_PATTERNS = re.compile(r"/(careers|jobs|engineering)", re.I)
_CLOUD_TECH_KEYWORDS = ("aws", "amazon web services", "azure", "gcp", "google cloud",
                        "kubernetes", "terraform", "docker", "cloud", "sre",
                        "site reliability", "platform engineer")

MAX_PAGES = 5
MAX_CHARS_PER_PAGE = 6000
PER_PAGE_TIMEOUT_S = 10.0
TOTAL_BUDGET_S = 25.0
# Fix #6: sitemap discovery is capped at this many FETCHED sitemap files total
# (robots.txt Sitemap: entries + /sitemap.xml + /sitemap_index.xml, plus at most
# one level of a sitemapindex's own child sitemaps) - discovery only, never counted
# against MAX_PAGES/TOTAL_BUDGET_S (it finds candidate URLs, it doesn't fetch them
# as pages).
MAX_SITEMAP_FILES = 6
_LOC_RE = re.compile(r"<loc>\s*(.*?)\s*</loc>", re.I | re.S)
# Non-page alternate/asset links a discovery source can surface but that are
# never worth fetching as a crawl candidate (e.g. a Next.js page's own
# `rel="alternate" type="text/markdown"` self-link).
_ASSET_RE = re.compile(
    r"\.(css|js|mjs|png|jpe?g|gif|svg|webp|ico|woff2?|ttf|pdf|zip|mp4|webm|md)(\?|$)", re.I)


def _same_origin(base: str, candidate: str) -> bool:
    b, c = urlparse(base), urlparse(candidate)
    return b.scheme == c.scheme and b.netloc == c.netloc


# A company's corporate, careers and security content often lives on a SIBLING host
# rather than under the consumer site's path tree. Lidl (2026-09-20): the lead
# submitted `lidl.hu`, whose own landing page nominates `vallalat.lidl.hu` - the
# corporate site, which states the Hungarian headcount in the first person - and
# `jobs.lidl.hu`. A same-origin-only crawl reaches neither, so the run reported "no
# company size evident from the given sources" about a company that publishes it.
#
# Two gates, both required. IDENTITY: the sibling sits under the same registrable
# domain as the submitted site AND was nominated by a first-party page we actually
# fetched - a host merely observed in certificate-transparency or DNS data is NOT a
# licence to crawl it. PURPOSE: its label has to look like corporate content, or the
# budget goes to the recipe and customer-service subdomains the same landing page
# also links (`konyha.`, `ugyfelszolgalat.` in that measured example).
#
# The purpose list is keywords, with the coverage limit that implies: it carries the
# common non-English forms this tool's leads actually use, but a corporate subdomain
# named in a language not listed here is missed. That is a bounded miss - the crawl
# then behaves exactly as it did before - not a wrong answer.
_SIBLING_PURPOSE_LABELS = frozenset({
    "about", "company", "corporate", "corp", "group", "investor", "investors", "ir",
    "press", "news", "media", "career", "careers", "job", "jobs", "work", "hiring",
    "security", "trust", "legal", "compliance", "privacy", "engineering", "eng",
    "tech", "technology", "developer", "developers", "status", "cloud",
    # Common non-English equivalents of "company"/"careers", by measured need:
    "vallalat", "vallalati", "ceg", "unternehmen", "firma", "empresa", "entreprise",
    "azienda", "bedrijf", "foretag", "yritys", "spolecnost", "karriere", "kariera",
    "karrier", "praca", "empleo", "lavoro", "emploi", "banen",
})
# Never worth a page slot even under the same registrable domain: infrastructure and
# transactional hosts that carry no company description.
_SIBLING_DENY_LABELS = frozenset({
    "cdn", "static", "assets", "img", "images", "files", "download", "downloads",
    "api", "mail", "smtp", "mx", "webmail", "ns", "ns1", "ns2", "vpn", "login",
    "auth", "sso", "account", "accounts", "id", "shop", "store", "checkout", "cart",
    "pay", "payment", "track", "tracking", "analytics", "ads", "www",
})
MAX_SIBLING_HOSTS = 2
# Registrable-domain comparison needs the public suffix. util.registrable_domain only
# strips a leading "www." (its docstring says so), so it reads `vallalat.lidl.hu` and
# `lidl.hu` as different domains. The first version of this used a hand-kept table of
# two-level suffixes; that is the thing the Public Suffix List exists to replace,
# because where the registrable boundary falls is registry policy, not an algorithm.
# `suffix_list_urls=()` pins it to the snapshot bundled with the library: no network
# call on first use, and the same answer in the offline tests as in a live run.
#
# `include_psl_private_domains=True` is what makes this an identity gate rather than a
# platform gate. The library's default is False, which only honours the PSL's ICANN
# section, so every tenant of a multi-tenant host collapses to one registrable domain:
# measured, `companya.github.io` and `careers.github.io` both read as `github.io`, and
# `_first_party_sibling_roots` then accepted an unrelated tenant's site as first-party
# to the lead - the exact cross-entity attribution the gate exists to stop. The PSL's
# private section is where those multi-tenant boundaries are published
# (github.io, herokuapp.com, vercel.app, s3.amazonaws.com, ...), so it has to be read
# too. ICANN-suffix answers are unchanged: `vallalat.lidl.hu` -> `lidl.hu` either way.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)


def _registrable(host: str) -> str:
    parts = _EXTRACT(host.lower().strip("."))
    return ".".join(x for x in (parts.domain, parts.suffix) if x)


def _first_party_sibling_roots(html: str, base_url: str) -> list[str]:
    """Root URLs of purpose-relevant sibling hosts THIS page links to, same
    registrable domain only, at most `MAX_SIBLING_HOSTS`, in the order the page
    nominates them. They compete for the same MAX_PAGES/TOTAL_BUDGET_S as every other
    candidate: this widens where the crawl may look, never how much it may fetch."""
    base_host = urlparse(base_url).netloc.lower()
    base_reg = _registrable(base_host)
    roots: dict[str, str] = {}
    for link in _all_absolute_hrefs(html, base_url):
        parsed = urlparse(link)
        host = parsed.netloc.lower()
        if not host or host == base_host or host in roots:
            continue
        if _registrable(host) != base_reg:
            continue  # identity gate: same registrable domain as the submitted site
        label = host.split(".", 1)[0]
        if label in _SIBLING_DENY_LABELS:
            continue
        if not any(part in _SIBLING_PURPOSE_LABELS for part in re.split(r"[-_]", label)):
            continue  # purpose gate
        roots[host] = f"{parsed.scheme or 'https'}://{host}/"
        if len(roots) >= MAX_SIBLING_HOSTS:
            break
    return list(roots.values())


def _priority(path: str, patterns: list[re.Pattern] = _PATH_PATTERNS) -> int | None:
    for i, pattern in enumerate(patterns):
        if pattern.search(path):
            return i
    return None


def _all_absolute_hrefs(html: str, base_url: str) -> list[str]:
    """Every absolute href on the page, same-origin or not (deduped, order kept).

    Off-domain links matter here: an ATS board (jobs.ashbyhq.com/<slug>) or a
    GitHub org (github.com/<org>) is never same-origin with the company's own
    site, so a same-origin filter is the wrong tool for finding them - this is the
    raw link list providers/ats.py and providers/github.py scan.
    """
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, re.I)
    out, seen = [], set()
    for href in hrefs:
        if href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        absolute = urljoin(base_url, href)
        if absolute not in seen:
            seen.add(absolute)
            out.append(absolute)
    return out


def extract_hrefs(html: str, base_url: str) -> list[str]:
    """Public wrapper: all absolute hrefs on one fetched page. research.py calls
    this for the landing page; `crawl_subpages` below captures it per subpage too -
    a board/org link that only appears in a subpage's <a href> (not in its
    trafilatura-extracted text, and not on the landing page) would otherwise be
    invisible to ats.py/github.py (they used to scan extracted TEXT only)."""
    return _all_absolute_hrefs(html, base_url)


def _extract_same_origin_links(html: str, base_url: str) -> list[str]:
    return [link for link in _all_absolute_hrefs(html, base_url) if _same_origin(base_url, link)]


def _scored_candidates(links: list[str], patterns: list[re.Pattern] = _PATH_PATTERNS) -> list[tuple[int, str]]:
    """(_priority, link) for every link that matches one of `patterns` (default
    `_PATH_PATTERNS` - the landing page's own broader list; `crawl_subpages` passes
    the narrower `_DISCOVERY_PATTERNS` for sitemap/mid-crawl candidates), deduped
    by normalised path (first occurrence wins), sorted best-first."""
    scored: list[tuple[int, str]] = []
    seen_paths: set[str] = set()
    for link in links:
        path = urlparse(link).path or "/"
        norm = path.rstrip("/").lower()
        if norm in seen_paths:
            continue
        pr = _priority(path, patterns)
        if pr is None:
            continue
        seen_paths.add(norm)
        scored.append((pr, link))
    scored.sort(key=lambda t: t[0])
    return scored


def select_subpages(html: str, base_url: str) -> list[str]:
    """Up to MAX_PAGES same-origin links from the landing page alone, deduped by
    path, in priority order. Fix #6 (v0.2 PART B): the previous "reserve one slot
    for a trust-family link" hack is gone - a trust/security/subprocessor/privacy/
    legal/dpa/compliance/infrastructure/status link that doesn't rank into the
    landing page's own top MAX_PAGES no longer needs a forced slot here, because
    `crawl_subpages` below now ALSO discovers such links from fetched keyword
    pages' own hrefs and from the site's sitemap - a genuine scored frontier
    instead of a single static list with one hardcoded exception.

    Plus, since 2026-09-20, the purpose-relevant sibling hosts this page nominates
    under the same registrable domain (see `_first_party_sibling_roots`), ranked after
    the same-origin path matches: a site whose own tree already answers "who is this
    company" keeps its existing behaviour, and only the leftover slots go wider."""
    same_origin = [link for _, link in _scored_candidates(_extract_same_origin_links(html, base_url))]
    siblings = [s for s in _first_party_sibling_roots(html, base_url) if s not in same_origin]
    return (same_origin + siblings)[:MAX_PAGES]


def _robots_disallows(base_url: str, path: str, client: httpx.Client) -> bool:
    """Simple `Disallow` check for the `*` user-agent group - best-effort, not a
    full robots.txt parser (no wildcards/Allow precedence)."""
    try:
        parsed = urlparse(base_url)
        r = safe_get(client, f"{parsed.scheme}://{parsed.netloc}/robots.txt", timeout=5)
        if r.status_code != 200:
            return False
        disallowed, applies = [], False
        for line in r.text.splitlines():
            line = line.strip()
            if line.lower().startswith("user-agent:"):
                applies = line.split(":", 1)[1].strip() == "*"
            elif applies and line.lower().startswith("disallow:"):
                rule = line.split(":", 1)[1].strip()
                if rule:
                    disallowed.append(rule)
        return any(path.startswith(rule) for rule in disallowed)
    except (httpx.HTTPError, OSError, UnsafeTargetError):
        return False


_SITEMAP_LINE_RE = re.compile(r"(?im)^\s*sitemap:\s*(\S+)")


def _discover_sitemap_urls(base_url: str, client: httpx.Client) -> list[str]:
    """Fix #6: sitemap-listed URLs as extra crawl candidates - robots.txt's own
    `Sitemap:` line(s) first, then the two conventional guesses (`/sitemap.xml`,
    `/sitemap_index.xml`) appended after (never ahead of a site's own declared
    list). If robots.txt itself redirects to a DIFFERENT origin (e.g. www -> apex),
    the same robots.txt+guesses are ALSO tried at that origin - a submitted URL is
    often the www form while the site's own canonical origin is the apex (or vice
    versa), and only probing the one origin the lead happened to submit would miss
    a site's real sitemap declaration. At most MAX_SITEMAP_FILES distinct sitemap
    files are ever fetched (a BFS queue, so a sitemapindex's own listed children
    still compete for the same budget, however deep - in practice rarely more than
    one level before the cap is hit). A `<loc>` ending in ".xml" is queued as
    another sitemap file to fetch; anything else is a page-URL candidate.
    Discovery only: this never touches the page-fetch budget in `crawl_subpages`.
    Never raises - any failure just means fewer candidates, same contract as the
    rest of this module."""
    parsed = urlparse(base_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    origins = [origin]
    seeds: list[str] = []
    for o in origins:
        try:
            r = safe_get(client, f"{o}/robots.txt", timeout=5)
            if r.status_code == 200:
                seeds += _SITEMAP_LINE_RE.findall(r.text)
            # httpx.URL.host is a str; .netloc is bytes - using the latter here
            # embedded a "b'...'" repr into the origin string (measured while
            # writing this fix).
            redirected_origin = f"{r.url.scheme}://{r.url.host}"
            if redirected_origin != o and redirected_origin not in origins:
                origins.append(redirected_origin)
        except (httpx.HTTPError, OSError, UnsafeTargetError):
            pass
        seeds += [f"{o}/sitemap.xml", f"{o}/sitemap_index.xml"]

    seen: set[str] = set()
    queue = list(dict.fromkeys(seeds))
    page_urls: list[str] = []
    while queue and len(seen) < MAX_SITEMAP_FILES:
        sm = queue.pop(0)
        if sm in seen:
            continue
        seen.add(sm)
        try:
            r = safe_get(client, sm, timeout=5)
        except (httpx.HTTPError, OSError, UnsafeTargetError):
            continue
        if r.status_code != 200:
            continue
        for loc in _LOC_RE.findall(r.text):
            (queue if loc.endswith(".xml") else page_urls).append(loc)
    return page_urls


def _strength_for(path: str, text: str) -> str:
    """Strong only for a careers/jobs/engineering page that actually names cloud
    tech; a generic about/team page (or a hiring page with no tech content) is
    medium - page type alone isn't strong evidence of infra intensity."""
    is_hiring_or_eng = bool(_HIRING_OR_ENG_PATTERNS.search(path))
    names_cloud_tech = any(kw in text.lower() for kw in _CLOUD_TECH_KEYWORDS)
    return "STRONG" if (is_hiring_or_eng and names_cloud_tech) else "MEDIUM"


def crawl_subpages(landing_url: str, landing_html: str) -> list[dict]:
    """Fetch up to MAX_PAGES same-origin pages, within a shared TOTAL_BUDGET_S time
    budget (not reset per page), each respecting a simple robots.txt Disallow check.

    Fix #6 (v0.2 PART B): a genuine scored FRONTIER, not a single static list
    computed once from the landing page alone. The candidate pool starts as the
    landing page's own same-origin links plus the site's sitemap-listed URLs
    (`_discover_sitemap_urls` - discovery only, never counted against the page/time
    budget above, scored with the narrower `_DISCOVERY_PATTERNS`); every time a
    fetched page's own hrefs surface a NEW same-origin, `_DISCOVERY_PATTERNS`-
    matching link, it is added to the frontier too and can still be picked up
    before the budget runs out. This replaces the old "reserve one slot for a
    trust-family link" hack: a trust/security/subprocessor/privacy/legal/dpa/
    compliance page now competes for a budget slot like any other candidate, from
    three sources instead of one.

    Never raises: any per-page or whole-crawl failure just means fewer results.
    Returns a list of {"url", "text", "strength", "hrefs", "html"} dicts, HTML
    pages only. `hrefs` is every absolute link found ON that page (any host, not
    just same-origin) - ats.py/github.py need it because a careers page's board/
    org link often lives only in a footer/nav <a href>, stripped out of the
    trafilatura-extracted `text`. `html` is the raw fetched HTML (Fix #3/#4):
    research.py uses it as the page's Evidence snapshot content and as input to
    leadscout/extract.py's structured (FAQ/details/table) extraction.
    """
    results: list[dict] = []
    deadline = time.monotonic() + TOTAL_BUDGET_S
    try:
        with httpx.Client(timeout=PER_PAGE_TIMEOUT_S,
                          headers={"User-Agent": UA}) as client:
            # Landing hrefs keep the broader `_PATH_PATTERNS` scoring (unchanged
            # since before this fix, and fetched first below) - "about"/"careers"/
            # "team"/"company" landing links are a bounded, known-good pool
            # already exercised before Fix #6. Sitemap/mid-crawl-discovered
            # candidates use the narrower `_DISCOVERY_PATTERNS` (Fix #6's own
            # keyword list) AND are limited to at most ONE candidate per matched
            # keyword (`_merge_narrow` below, ties broken by the shortest path -
            # a site's own canonical trust/security page is reliably its
            # shortest-path one, e.g. "/security" beats "/blog/tags/security" and
            # "/handbook/engineering/security"): a large site's sitemap can list
            # dozens of pages that happen to mention "security"/"privacy" in a
            # blog slug or a docs/Q&A path, and trying every one of them would
            # burn the whole page budget on one topic instead of surveying the
            # site (measured 2026-09-19 against PostHog's real sitemap: 4 distinct
            # "security"-matching URLs, only one of which is the actual trust page).
            frontier = _scored_candidates(_extract_same_origin_links(landing_html, landing_url))
            # Purpose-relevant sibling hosts the landing page itself nominates, ranked
            # after every same-origin path match: a site that already answers "who is
            # this company" in its own tree keeps its previous behaviour exactly, and
            # only leftover budget goes wider (see `_first_party_sibling_roots`).
            frontier += [(len(_PATH_PATTERNS) + i, url)
                         for i, url in enumerate(_first_party_sibling_roots(landing_html, landing_url))]
            narrow_best: dict[int, tuple[tuple[int, int], str]] = {}  # priority -> (rank, url)
            narrow_closed: set[int] = set()  # priorities already attempted - never revisited

            def _narrow_rank(path: str) -> tuple[int, int]:
                """Shortest path wins within a keyword: a site's canonical policy/trust page is
                usually its shortest-path one (e.g. a top-level "/privacy"). No depth preference -
                nothing here may be tuned to where one site happens to keep its pages."""
                return (0, len(path))

            def _merge_narrow(urls: list[str]) -> None:
                for pr, url in _scored_candidates(urls, _DISCOVERY_PATTERNS):
                    if pr in narrow_closed:
                        continue
                    path = urlparse(url).path or "/"
                    first_segment = path.strip("/").split("/", 1)[0].lower()
                    if first_segment in _NON_AUTHORITATIVE_SECTIONS:
                        continue  # a blog post/forum question/job listing is never a trust statement
                    rank = _narrow_rank(path)
                    cur = narrow_best.get(pr)
                    if cur is None or rank < cur[0]:
                        narrow_best[pr] = (rank, url)

            try:
                sitemap_links = [
                    u for u in _discover_sitemap_urls(landing_url, client) if _same_origin(landing_url, u)
                ]
                _merge_narrow(sitemap_links)
            except (httpx.HTTPError, OSError, UnsafeTargetError):
                pass

            fetched_paths: set[str] = set()
            attempts = 0  # actual page GETs made - the real budget, not just successes

            while attempts < MAX_PAGES and time.monotonic() < deadline:
                if frontier:
                    frontier.sort(key=lambda t: t[0])
                    _pr, url = frontier.pop(0)
                else:
                    open_narrow = {pr: v for pr, v in narrow_best.items() if pr not in narrow_closed}
                    if not open_narrow:
                        break
                    best_pr = min(open_narrow)
                    url = open_narrow[best_pr][1]
                    narrow_closed.add(best_pr)  # one attempt per keyword, regardless of outcome
                parsed_candidate = urlparse(url)
                path = parsed_candidate.path or "/"
                # Keyed by (host, path), not path alone: sibling hosts are all rooted at
                # "/", so a path-only key silently dropped every sibling after the first.
                norm = (parsed_candidate.netloc.lower(), path.rstrip("/").lower())
                if norm in fetched_paths:
                    continue
                fetched_paths.add(norm)
                # A broad-frontier (landing) candidate that ALSO matches a
                # discovery keyword closes that slot too, so sitemap discovery
                # doesn't separately spend a budget slot on the same topic.
                narrow_hit = _priority(path, _DISCOVERY_PATTERNS)
                if narrow_hit is not None:
                    narrow_closed.add(narrow_hit)
                # Against the CANDIDATE's own origin: a sibling host publishes its own
                # robots.txt, and asking the landing host about it would both miss its
                # rules and apply rules that do not govern it.
                if _robots_disallows(url, path, client):
                    continue
                attempts += 1
                try:
                    r = safe_get(client, url)
                    # A MISSING content-type header is not evidence of a non-HTML
                    # response (measured 2026-09-19 against the frozen replay
                    # corpus: several real, genuinely-HTML captured pages carry no
                    # content-type header at all) - only an EXPLICIT, clearly
                    # non-HTML value rejects the page; trafilatura's own
                    # extraction is the real fallback safety net for anything
                    # that slips through (a non-HTML body just yields no text).
                    content_type = r.headers.get("content-type", "").lower()
                    if r.status_code >= 400 or (content_type and "html" not in content_type):
                        continue
                    text = (trafilatura.extract(r.text, include_comments=False, include_tables=False) or "").strip()
                    rendered = maybe_render_with_crawl4ai(url, r.text, text)
                    if rendered:
                        text = rendered
                    hrefs = _all_absolute_hrefs(r.text, url)
                    if text:
                        results.append({"url": url, "text": text[:MAX_CHARS_PER_PAGE],
                                        "strength": _strength_for(path, text), "hrefs": hrefs, "html": r.text})
                    # Frontier growth: mirrors capture_universe.py's own d1->d2
                    # rule exactly - hrefs are only harvested for the frontier from
                    # a page whose OWN path already matches a discovery keyword
                    # (an "about"/"company" landing link is not itself a keyword
                    # page, so its hrefs don't get a second, wider look here; a
                    # fetched "/security-compliance" page's own links do). A
                    # `.md`/asset-style href (e.g. Next.js's own `rel="alternate"
                    # type="text/markdown"` self-link) is excluded - never a page
                    # to actually crawl.
                    if narrow_hit is not None:
                        new_links = [h for h in hrefs if _same_origin(landing_url, h) and not _ASSET_RE.search(h)]
                        _merge_narrow(new_links)
                except (httpx.HTTPError, OSError, UnsafeTargetError):
                    continue
    except (httpx.HTTPError, OSError, UnsafeTargetError):
        return results
    return results


# --- Crawl4AI escalation trigger --------------------------------------------------
# A static httpx+trafilatura fetch sees nothing behind a client-side-rendered shell
# (React/Angular/Next.js apps that ship an almost-empty index.html). Four independent
# signals of that: the extracted text is just short; the HTML is mostly script/markup
# with very little of it surviving extraction; a known JS-framework mount point is
# present alongside little text; or most "lines" of the extracted text are stray
# short fragments (nav labels, button text) rather than prose. Any one firing is
# enough - they're independent evidence of the same shell problem, not a vote.
SHORT_TEXT_CHARS = 300
MIN_TEXT_HTML_RATIO = 0.02
SHORT_LINE_MAX_WORDS = 3
SHORT_LINE_FRACTION = 0.70
_JS_SHELL_MARKERS = ('<div id="root"></div>', 'id="__next"', "ng-app")


def needs_js_render(html: str, text: str) -> tuple[bool, str]:
    """Returns (should_render, trigger) - trigger is "" when should_render is False.
    Only the first matching trigger is reported; see module comment above for why
    checking them all isn't needed (any one is sufficient, not cumulative)."""
    if any(marker in html for marker in _JS_SHELL_MARKERS) and len(text) < SHORT_TEXT_CHARS:
        return True, "js_shell_marker"
    if len(text) < SHORT_TEXT_CHARS:
        return True, "short_text"
    if html and len(text) / len(html) < MIN_TEXT_HTML_RATIO:
        return True, "low_text_html_ratio"
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines:
        short = sum(1 for ln in lines if len(ln.split()) <= SHORT_LINE_MAX_WORDS)
        if short / len(lines) >= SHORT_LINE_FRACTION:
            return True, "mostly_short_lines"
    return False, ""


def _crawl4ai_fetch(url: str) -> str | None:
    """Actual Crawl4AI headless-browser call, isolated in its own function so tests
    can monkeypatch it directly instead of driving a real browser (see
    REVIEW-6bB-verified.md #3's offline fake-renderer test). Returns the rendered
    page's extracted text, or None on any failure - never raises."""
    try:
        import crawl4ai  # noqa: F401 - already checked importable by the caller
        from crawl4ai import WebCrawler
    except ImportError:
        return None
    try:
        # The browser resolves and connects by itself, so it gets none of safe_get's pinning:
        # at least refuse a URL that is not public BEFORE it navigates. Redirects the browser
        # follows afterwards are not re-validated (keep LEADSCOUT_JS_RENDER off for mail-derived URLs).
        assert_public_url(url)
    except UnsafeTargetError:
        logger.warning("js render refused: %s is not a public URL", url)
        return None
    try:
        crawler = WebCrawler()
        crawler.warmup()
        result = crawler.run(url=url)
        rendered = (getattr(result, "extracted_content", "") or getattr(result, "markdown", "") or "").strip()
        return rendered or None
    except Exception:  # noqa: BLE001 - a broken/slow render must fall back, never crash the pipeline
        return None


def maybe_render_with_crawl4ai(url: str, html: str, text: str) -> str | None:
    """If a trigger fires, only actually escalates when BOTH `LEADSCOUT_JS_RENDER=1`
    and the optional `crawl4ai` package (the `[js]` extra) is importable - otherwise
    logs the suggestion and returns None so the caller keeps the static-fetch text.
    When both gates pass, calls `_crawl4ai_fetch` and, if it returns rendered text,
    that text REPLACES the shell text the static fetch produced (the caller is
    expected to use the return value in place of its own `text`, not merge it).
    """
    should_render, trigger = needs_js_render(html, text)
    if not should_render:
        return None
    if os.getenv("LEADSCOUT_JS_RENDER") != "1":
        logger.info("js render suggested by %s, disabled", trigger)
        return None
    try:
        import crawl4ai  # noqa: F401
    except ImportError:
        logger.info("js render suggested by %s, disabled (crawl4ai not installed)", trigger)
        return None
    rendered = _crawl4ai_fetch(url)
    if not rendered:
        logger.info("js render triggered by %s for %s, but crawl4ai returned nothing usable", trigger, url)
        return None
    logger.info("js render triggered by %s for %s: %d chars replace the %d-char shell text",
                trigger, url, len(rendered), len(text))
    return rendered
