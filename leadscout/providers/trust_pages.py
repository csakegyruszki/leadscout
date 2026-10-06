"""First-party cloud-provider statements: trust/security/subprocessor pages (already
crawled by website.py's extended link priority - see that module) plus one GET each
to a guessed status-page URL.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from html import unescape

import httpx

from ..config import settings
from ..models import Evidence, ProviderResult
from ..util import UnsafeTargetError, safe_get
from ._base import UA, fallback_evidence_id, snapshot_path

# --- Design notes -----------------------------------------------------------------
# A page merely *naming* AWS/Azure/GCP/OCI is not evidence by itself. Fix #5 (v0.2
# PART B) reduces sentence-level detection to exactly three predicates, in this
# priority order:
#   - INTEGRATES_WITH: integration/app-catalog wording ("we integrate with AWS Cost
#     Explorer") - a SaaS dependency, not the company's own hosting. Still recorded
#     (for audit trail) as a WEAK, scope="saas_dependency" first_party_statement,
#     which cloud.py's `_qualifying` already excludes from every state entirely -
#     "excluded" in that sense, never a hosting claim.
#   - CLOUD_SUBPROCESSOR: a subprocessor-table row naming a cloud provider - real
#     signal (the company disclosed it as ITS OWN subprocessor), but a table row is
#     weaker than an explicit sentence, so MEDIUM, never DIRECT/CONFIRMED alone
#     (cloud.py's CONFIRMED gate requires DIRECT).
#   - HOSTED_ON: an explicit current first-party hosting sentence (hosting/hosted/
#     infrastructure/data-center wording, NOT "subprocessor") - DIRECT.
# A status-page component naming a provider (or a region/Kubernetes/object-storage
# term) is a stronger signal still (the company is telling its own users which
# infra failed) and goes straight to STRONG, unaffected by the above.
#
# Every sentence-level Evidence now carries the actual page URL it was found on
# (v0.1.2 recorded url="" - a trust statement with nothing to point back to).
# `pages` is `[{"url", "text", "locator"}, ...]` (leadscout/extract.py's structured
# chunks - main text, FAQPage JSON-LD Q/A, <details>, <table> - flattened across the
# landing page and every crawled subpage); a non-"main_text" locator is appended to
# the Evidence url as a URL fragment so a reader can tell "the sentence itself" from
# "row 2 of the subprocessor table" apart, e.g. "https://x/legal#table[0]".

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.I | re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def _visible_text(text: str) -> str:
    """Detection must run on extracted text only, never raw HTML - a raw markup
    fragment ("...</p><span data-state...") is not a sentence and must not be
    treated as one; tag/attribute soup can also hide an accidental substring hit."""
    stripped = _SCRIPT_STYLE_RE.sub(" ", text)
    return unescape(_TAG_RE.sub(" ", stripped))


# Case rules: AWS/GCP/OCI are UPPERCASE-ONLY, matched on the original (non-lowered)
# text - "social media" must never match OCI, and a lowercase "aws"/"gcp" is too weak
# a signal on its own. The multi-word/full names are case-insensitive.
_PROVIDER_PATTERNS = (
    (re.compile(r"\bAWS\b"), "AWS"),
    (re.compile(r"\bAmazon Web Services\b", re.I), "AWS"),
    (re.compile(r"\bAzure\b", re.I), "Azure"),
    (re.compile(r"\bGCP\b"), "GCP"),
    (re.compile(r"\bGoogle Cloud\b", re.I), "GCP"),
    (re.compile(r"\bOracle Cloud\b", re.I), "OCI"),
    (re.compile(r"\bOCI\b"), "OCI"),
)
# "host"/"hosts" too: "we host your data on <provider>" is the plainest first-party hosting sentence.
# The bare noun "infrastructure" is NOT a hosting verb: security-tooling sentences ("alerts from the
# infrastructure through <provider> GuardDuty") mention it without stating where anything is hosted.
# Only the asserting form "infrastructure is/runs/lives ..." counts (v0.2.1, found by the blind holdout).
_HOSTED_ON_RE = re.compile(
    r"\b(?:host|hosts|hosting|hosted|data cent(?:er|re))\b|\binfrastructure\s+(?:is|runs|lives)\b", re.I)
_SUBPROCESSOR_RE = re.compile(r"\bsub-?processors?\b", re.I)
_SAAS_DEPENDENCY_WORDS = ("integrate", "integration", "connector", "app marketplace", "cost explorer")
# A hosting VERB is not a hosting CLAIM (REVIEW-B). A sentence only asserts that this company hosts
# something when it speaks in the first person AND is not one of these other senses of the same words.
_FIRST_PERSON_RE = re.compile(r"\b(we|we're|our|ours|us)\b", re.I)
_EVENT_OBJECT_RE = re.compile(r"\b(webinar|webinars|event|events|meetup|meetups|conference|conferences|"
                              r"podcast|workshop|workshops|hackathon|session|sessions|talk|talks|"
                              r"office hours|ama)\b", re.I)
# Reseller / customer-side deployment: the infrastructure belongs to someone else.
_CUSTOMER_SIDE_RE = re.compile(r"\b(reseller|resell|partner program|on your behalf|your own|your account|"
                               r"customer'?s own|for our customers|for customers|managed for you|"
                               r"in your (?:aws|azure|gcp|google cloud|oracle) account)\b", re.I)
_COMPARISON_RE = re.compile(r"\b(vs\.?|versus|compared to|comparison|competitor|alternative to)\b", re.I)
_STORAGE_WORDS = ("storage", "object storage", "s3", "blob")
_CDN_WORDS = ("cdn", "content delivery", "edge")
_STATUS_COMPONENT_PATTERNS = _PROVIDER_PATTERNS + (
    (re.compile(r"\bus-east\b", re.I), None), (re.compile(r"\bus-west\b", re.I), None),
    (re.compile(r"\beu-west\b", re.I), None), (re.compile(r"\beu-central\b", re.I), None),
    (re.compile(r"\bkubernetes\b", re.I), None), (re.compile(r"\bobject storage\b", re.I), None),
)
# A generic third-party status aggregator (or a parked/wildcard host) can return 200
# and still happen to mention a provider word - require the page to actually be
# ABOUT this company (its domain label somewhere in the page) and to read like a
# real status page, not just any page (REVIEW-6bB-verified.md #2).
_STATUS_IDENTITY_WORDS = ("status", "uptime", "incident", "operational")
# A component named after a DEPENDENCY ("AWS Billing Integration", "Azure Marketplace sync") says the
# company integrates with that provider, not that it runs on it (REVIEW-B).
_STATUS_DEPENDENCY_WORDS = ("integration", "integrations", "billing", "marketplace", "partner", "connector",
                            "sync", "webhook", "export", "import")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+\s*(?:[-*•]\s*)?")


def _asserts_own_hosting(sentence: str, company_label: str = "") -> bool:
    """True only for an affirmative first-person statement about this company's own hosting.

    Rejects (each measured as a false positive in the REVIEW-B adversarial pass):
    a question ("Do you host customer workloads on AWS?"), the event sense of "host"
    ("We host webinars with AWS startups"), reseller/customer-side wording ("As an AWS reseller,
    we offer hosting plans to customers", "runs in your own AWS account"), and comparison copy
    ("Competitor | AWS hosting | Our product | self-managed").
    """
    if "?" in sentence:
        return False
    subject_ok = bool(_FIRST_PERSON_RE.search(sentence)) or (
        bool(company_label) and re.search(r"\b" + re.escape(company_label) + r"\b", sentence, re.I) is not None)
    if not subject_ok:
        return False
    return not (_EVENT_OBJECT_RE.search(sentence) or _CUSTOMER_SIDE_RE.search(sentence)
                or _COMPARISON_RE.search(sentence))


def _scope_for(sentence: str) -> str:
    low = sentence.lower()
    if any(w in low for w in _STORAGE_WORDS):
        return "storage"
    if any(w in low for w in _CDN_WORDS):
        return "edge"
    return "workload"


def _find_statements(text: str, company_label: str = "") -> list[tuple[str, str, str, str]]:
    """Returns (kind, provider, scope, sentence) for each sentence naming a
    provider, `kind` one of "INTEGRATES_WITH" | "CLOUD_SUBPROCESSOR" | "HOSTED_ON"
    (Fix #5's three predicates - checked in that priority order: integration
    wording always wins over subprocessor/hosting wording in the same sentence, and
    subprocessor wording wins over generic hosting wording). A sentence naming a
    provider with none of the three present yields nothing - naming alone was never
    evidence."""
    out = []
    for sentence in _SENTENCE_SPLIT.split(_visible_text(text)):
        provider = next((p for pat, p in _PROVIDER_PATTERNS if pat.search(sentence)), None)
        if not provider:
            continue
        low = sentence.lower()
        if any(w in low for w in _SAAS_DEPENDENCY_WORDS):
            out.append(("INTEGRATES_WITH", provider, "saas_dependency", sentence.strip()))
        elif _SUBPROCESSOR_RE.search(sentence):
            out.append(("CLOUD_SUBPROCESSOR", provider, _scope_for(sentence), sentence.strip()))
        elif _HOSTED_ON_RE.search(sentence) and _asserts_own_hosting(sentence, company_label):
            out.append(("HOSTED_ON", provider, _scope_for(sentence), sentence.strip()))
    return out


_STRENGTH_BY_KIND = {"HOSTED_ON": "DIRECT", "CLOUD_SUBPROCESSOR": "MEDIUM", "INTEGRATES_WITH": "WEAK"}


def _guess_status_urls(domain: str) -> list[str]:
    label = domain.split(".")[0]
    return [f"https://status.{domain}", f"https://{label}.statuspage.io"]


def _dependency_component(visible: str, pat: re.Pattern) -> bool:
    """True when the provider word only appears inside a dependency-style component name."""
    for m in pat.finditer(visible):
        line = visible[max(0, m.start() - 60):m.end() + 60].lower()
        if not any(w in line for w in _STATUS_DEPENDENCY_WORDS):
            return False
    return True


def _status_page_evidence(domain: str, client: httpx.Client) -> Evidence | None:
    label = domain.split(".")[0].lower()
    for url in _guess_status_urls(domain):
        try:
            r = safe_get(client, url, timeout=10)
        except (httpx.HTTPError, UnsafeTargetError):
            continue
        if r.status_code != 200:
            continue
        visible = _visible_text(r.text)
        low = visible.lower()
        if label not in low or not any(w in low for w in _STATUS_IDENTITY_WORDS):
            continue
        hit_pat, hit_provider = next(
            ((pat, p) for pat, p in _STATUS_COMPONENT_PATTERNS
             if pat.search(visible) and not _dependency_component(visible, pat)), (None, None)
        )
        if hit_pat:
            provider = hit_provider or "other"
            hit_text = hit_pat.search(visible).group(0)
            return Evidence(
                id="", source_type="trust_pages", url=url, observed_at=datetime.now(UTC).isoformat(),
                content_sha256="", strength="STRONG",
                snippet=f"status page component names '{hit_text}'"[:200], snapshot_path="",
                family="first_party_statement", provider=provider, scope="workload",
                freshness="current", origin="provider:trust_pages",
            )
    return None


def _finalise(ev: Evidence, raw: bytes, run) -> Evidence:
    eid = fallback_evidence_id("trust", run, id(ev))
    ev.id = eid
    ev.content_sha256 = hashlib.sha256(raw).hexdigest()
    ev.snapshot_path = snapshot_path(run, eid, "txt")
    if run is not None:
        run.record(ev, raw, stage="trust_pages")
    return ev


def run(domain: str, pages: list[dict], run=None) -> ProviderResult:
    """`pages` is `[{"url", "text", "locator"}, ...]` - leadscout/extract.py's
    structured chunks (trafilatura main text + FAQPage JSON-LD Q/A + <details>/
    <table> text) flattened across the already-fetched landing page and every
    crawled subpage (see website.py's extended crawl priority - trust/security/
    subprocessor/status/privacy/dpa/architecture paths score highest there). The
    status-page GET is the only network call this provider makes itself.

    Backward-compat: a bare list of strings (no locator/url) is still accepted -
    each string is treated as one page with no url/locator, same as before Fix #5.
    """
    # The company's own name is an acceptable grammatical subject for its hosting statement
    # ("Zapier is hosted on AWS"), alongside first person ("we/our") - see `_asserts_own_hosting`.
    company_label = domain.split(".")[0].lower()
    normalised: list[dict] = [
        p if isinstance(p, dict) else {"url": "", "text": p, "locator": "main_text"} for p in pages
    ]
    evidence: list[Evidence] = []
    for page in normalised:
        text, url, locator = page.get("text", ""), page.get("url", ""), page.get("locator", "main_text")
        ev_url = url if (not locator or locator == "main_text") else f"{url}#{locator}"
        for kind, provider, scope, sentence in _find_statements(text, company_label=company_label):
            strength = _STRENGTH_BY_KIND[kind]
            # A statement whose own wording resolves to CDN/edge scope (e.g. a
            # subprocessor-table row naming a CDN vendor's "edge-network, content
            # delivery" service) is an edge/CDN observation, not a first-party
            # hosting claim - cloud.py's own edge_delivery family, never
            # first_party_statement (which counts toward LIKELY/POSSIBLE
            # regardless of scope="edge"; only the family exclusion actually
            # keeps a CDN-only signal from being read as workload hosting).
            family = "edge_delivery" if scope == "edge" else "first_party_statement"
            ev = Evidence(
                id="", source_type="trust_pages", url=ev_url, observed_at=datetime.now(UTC).isoformat(),
                content_sha256="", strength=strength, snippet=sentence[:200], snapshot_path="",
                family=family, provider=provider, scope=scope,
                freshness="current", origin="provider:trust_pages",
            )
            evidence.append(_finalise(ev, sentence.encode("utf-8"), run))

    calls = 0
    try:
        with httpx.Client(headers={"User-Agent": UA}, timeout=settings.http_timeout) as client:
            calls = len(_guess_status_urls(domain))
            status_ev = _status_page_evidence(domain, client)
    except Exception as e:  # noqa: BLE001 - a malformed/unreachable response must degrade, never crash
        # A failed provider yields NO evidence - never the sentences already found
        # in page_texts before the status-page GET failed (REVIEW-6bA-verified.md #6).
        return ProviderResult(provider_name="trust_pages", status="degraded",
                               note=f"{type(e).__name__}: {e}", calls=calls)
    if status_ev:
        evidence.append(_finalise(status_ev, status_ev.snippet.encode("utf-8"), run))

    if not evidence:
        return ProviderResult(provider_name="trust_pages", status="skipped",
                               note="no trust-page statement or status-page component found", calls=calls)
    return ProviderResult(provider_name="trust_pages", status="ok", evidence=evidence,
                           note=f"{len(evidence)} first_party_statement item(s)", calls=calls)
